# =============================================================
#  Transkription_Notes_Pipeline - tests/test_db.py
#  Unit tests for db.py: schema creation and the v1 -> v3 migration,
#  transcript upsert semantics, FTS5 search plus its LIKE fallback,
#  slide and run_log inserts, source-path repointing after a move,
#  the known-speakers roster, and the DB file management helpers.
#
#  Runs against a REAL sqlite database under pytest's tmp_path - db.py
#  takes an explicit db_path/conn everywhere and imports config only in
#  its __main__ block, so nothing here touches the project's own
#  transkription_notes_pipeline.db. No whisperx/torch/PIL involved.
#
#  Run: C:\Python\Python311\python.exe -m pytest tests/test_db.py -v
# =============================================================

import sqlite3

import pytest

import db


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def make_db(tmp_path, name="test.db"):
    """Validated, empty database under tmp_path. Returns (path_str, conn).
    Every db.py writer commits immediately and pytest discards tmp_path, so
    closing is usually unnecessary - except before rename_db, which on
    Windows cannot move a file while a handle is open."""
    path = str(tmp_path / name)
    assert db.validate_schema(path) is True
    return path, db.get_connection(path)


def make_slide_record(video_path="C:\\v\\talk.mp4", timestamp_sec=12.5, **over):
    """Full slide record. insert_slide uses named parameters, so every key
    must be present - see TestInsertSlide.test_partial_record_raises."""
    record = {
        "video_path": video_path,
        "timestamp_sec": timestamp_sec,
        "snapshot_path": "snapshots/frame_000001.jpg",
        "hash_value": "abcdef",
        "title": "Retention index calibration",
        "bullets": '["Use C7-C30 alkanes", "Check the column"]',
        "slide_type": "content",
        "transcript_seg": "We calibrate with an alkane ladder.",
        "speaker": "SPEAKER_01",
    }
    record.update(over)
    return record


def table_names(conn):
    return {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}


def trigger_names(conn):
    return {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger'").fetchall()}


# -----------------------------------------------------------------
# _fts_query
# -----------------------------------------------------------------

class TestFtsQuery:
    def test_wraps_whole_query_as_one_literal_phrase(self):
        assert db._fts_query("retention index") == '"retention index"'

    def test_doubles_embedded_double_quotes(self):
        assert db._fts_query('say "hello"') == '"say ""hello"""'

    def test_special_characters_are_not_treated_as_fts5_syntax(self):
        """A bare GC-MS or OR would be parsed as FTS5 operators/columns and
        raise; quoting the whole thing keeps it a literal phrase."""
        assert db._fts_query("GC-MS OR LC") == '"GC-MS OR LC"'


# -----------------------------------------------------------------
# validate_schema
# -----------------------------------------------------------------

class TestValidateSchema:
    def test_creates_every_base_table(self, tmp_path):
        _, conn = make_db(tmp_path)
        assert {"transcripts", "slides", "run_log", "meta",
                "known_speakers"} <= table_names(conn)

    def test_creates_both_fts_tables(self, tmp_path):
        _, conn = make_db(tmp_path)
        assert {"notes_fts", "slides_fts"} <= table_names(conn)

    def test_creates_all_six_sync_triggers(self, tmp_path):
        _, conn = make_db(tmp_path)
        assert {"transcripts_ai", "transcripts_ad", "transcripts_au",
                "slides_ai", "slides_ad", "slides_au"} <= trigger_names(conn)

    def test_records_schema_version(self, tmp_path):
        _, conn = make_db(tmp_path)
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        assert row[0] == str(db.SCHEMA_VERSION)

    def test_is_idempotent(self, tmp_path):
        path, conn = make_db(tmp_path)
        db.upsert_transcript(conn, "C:\\a\\one.mp4", "large-v3", "de", 5, 60.0)
        conn.close()
        assert db.validate_schema(path) is True  # second run must not wipe data
        conn2 = db.get_connection(path)
        assert len(db.list_transcripts(conn2)) == 1

    def test_migrates_v1_database_by_adding_notes_text(self, tmp_path):
        """A schema-v1 transcripts table has no notes_text column. Rather
        than asserting on the return value (validate_schema swallows every
        exception and returns False), check the column really appears."""
        path = str(tmp_path / "old.db")
        old = sqlite3.connect(path)
        old.executescript("""
            CREATE TABLE transcripts (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path       TEXT NOT NULL UNIQUE,
                file_name       TEXT NOT NULL,
                whisper_model   TEXT,
                language        TEXT,
                segment_count   INTEGER,
                duration_sec    REAL,
                notes_generated INTEGER DEFAULT 0,
                notes_backend   TEXT,
                prompt_template TEXT,
                processed_at    TEXT
            );
        """)
        old.execute("INSERT INTO transcripts (file_path, file_name) VALUES ('C:\\\\x.mp4', 'x.mp4')")
        old.commit()
        old.close()

        assert db.validate_schema(path) is True
        conn = db.get_connection(path)
        assert db._column_exists(conn, "transcripts", "notes_text")
        assert len(db.list_transcripts(conn)) == 1  # pre-existing row survived


# -----------------------------------------------------------------
# upsert_transcript / list / get_completed
# -----------------------------------------------------------------

class TestUpsertTranscript:
    def test_insert_then_read_back(self, tmp_path):
        _, conn = make_db(tmp_path)
        db.upsert_transcript(conn, "C:\\rec\\call.mp4", "large-v3", "de", 42, 601.44,
                             notes_generated=True, notes_backend="ollama",
                             prompt_template="meeting", notes_text="## Summary\nAll good.")
        rows = db.list_transcripts(conn)
        assert len(rows) == 1
        assert rows[0]["file_name"] == "call.mp4"      # derived from file_path
        assert rows[0]["duration_sec"] == 601.4        # rounded to 1 decimal
        assert rows[0]["notes_generated"] == 1         # bool stored as int

    def test_same_path_updates_instead_of_duplicating(self, tmp_path):
        _, conn = make_db(tmp_path)
        db.upsert_transcript(conn, "C:\\rec\\call.mp4", "medium", "en", 1, 10.0)
        db.upsert_transcript(conn, "C:\\rec\\call.mp4", "large-v3", "de", 99, 20.0)
        rows = db.list_transcripts(conn)
        assert len(rows) == 1
        assert rows[0]["whisper_model"] == "large-v3"
        assert rows[0]["segment_count"] == 99

    def test_blank_notes_text_does_not_erase_existing_notes(self, tmp_path):
        """The upsert's CASE WHEN guard: a whisper-only re-run passes
        notes_text="" and must not wipe notes generated earlier."""
        _, conn = make_db(tmp_path)
        db.upsert_transcript(conn, "C:\\rec\\call.mp4", "large-v3", "de", 5, 10.0,
                             notes_text="## Summary\nkeep me")
        db.upsert_transcript(conn, "C:\\rec\\call.mp4", "large-v3", "de", 6, 11.0,
                             notes_text="")
        assert db.list_transcripts(conn)[0]["notes_text"] == "## Summary\nkeep me"

    def test_get_completed_transcript_requires_notes_generated(self, tmp_path):
        _, conn = make_db(tmp_path)
        db.upsert_transcript(conn, "C:\\rec\\a.mp4", "large-v3", "de", 5, 10.0,
                             notes_generated=False)
        assert db.get_completed_transcript(conn, "C:\\rec\\a.mp4") is None
        db.upsert_transcript(conn, "C:\\rec\\a.mp4", "large-v3", "de", 5, 10.0,
                             notes_generated=True)
        assert db.get_completed_transcript(conn, "C:\\rec\\a.mp4") is not None

    def test_get_completed_transcript_unknown_path(self, tmp_path):
        _, conn = make_db(tmp_path)
        assert db.get_completed_transcript(conn, "C:\\nope.mp4") is None


# -----------------------------------------------------------------
# search_notes / search_slides  (FTS5 path and LIKE fallback)
# -----------------------------------------------------------------

class TestSearchNotes:
    def _seed(self, conn):
        db.upsert_transcript(conn, "C:\\rec\\gc.mp4", "large-v3", "en", 3, 30.0,
                             notes_generated=True,
                             notes_text="We discussed the retention index ladder at length.")
        db.upsert_transcript(conn, "C:\\rec\\other.mp4", "large-v3", "en", 3, 30.0,
                             notes_generated=True,
                             notes_text="Unrelated budget conversation.")

    def test_blank_query_returns_nothing(self, tmp_path):
        _, conn = make_db(tmp_path)
        self._seed(conn)
        assert db.search_notes(conn, "") == []
        assert db.search_notes(conn, "   ") == []
        assert db.search_notes(conn, None) == []

    def test_fts5_finds_phrase_and_builds_snippet(self, tmp_path):
        _, conn = make_db(tmp_path)
        self._seed(conn)
        hits = db.search_notes(conn, "retention index")
        assert [h["file_name"] for h in hits] == ["gc.mp4"]
        assert "retention" in hits[0]["snippet"].lower()

    def test_fts5_tolerates_punctuation_in_the_query(self, tmp_path):
        """Unquoted, "GC-MS." would be FTS5 syntax and raise; _fts_query
        makes it a literal phrase instead."""
        _, conn = make_db(tmp_path)
        db.upsert_transcript(conn, "C:\\rec\\ms.mp4", "large-v3", "en", 1, 5.0,
                             notes_generated=True, notes_text="Bought a GC-MS. It works.")
        hits = db.search_notes(conn, "GC-MS.")
        assert [h["file_name"] for h in hits] == ["ms.mp4"]

    def test_no_match_returns_empty_list(self, tmp_path):
        _, conn = make_db(tmp_path)
        self._seed(conn)
        assert db.search_notes(conn, "dioxin furan congener") == []

    def test_like_fallback_when_fts5_unavailable(self, tmp_path, monkeypatch):
        """Forced via monkeypatch so pytest restores the module global:
        FTS5_AVAILABLE is read by three functions and a bare assignment
        would leak into every later test in the session."""
        _, conn = make_db(tmp_path)
        self._seed(conn)
        monkeypatch.setattr(db, "FTS5_AVAILABLE", False)
        hits = db.search_notes(conn, "retention index")
        assert [h["file_name"] for h in hits] == ["gc.mp4"]
        # The fallback snippet is a plain 200-char prefix, not a marked-up one.
        assert hits[0]["snippet"].startswith("We discussed the retention index")

    def test_like_fallback_also_matches_on_file_name(self, tmp_path, monkeypatch):
        _, conn = make_db(tmp_path)
        self._seed(conn)
        monkeypatch.setattr(db, "FTS5_AVAILABLE", False)
        assert [h["file_name"] for h in db.search_notes(conn, "other")] == ["other.mp4"]

    def test_limit_is_respected(self, tmp_path):
        _, conn = make_db(tmp_path)
        for i in range(5):
            db.upsert_transcript(conn, f"C:\\rec\\f{i}.mp4", "large-v3", "en", 1, 5.0,
                                 notes_generated=True, notes_text="shared keyword here")
        assert len(db.search_notes(conn, "shared keyword", limit=2)) == 2


class TestSearchSlides:
    def _seed(self, conn):
        db.insert_slide(conn, make_slide_record(timestamp_sec=1.0))
        db.insert_slide(conn, make_slide_record(
            timestamp_sec=2.0, title="Budget overview",
            bullets='["Q3 numbers"]', transcript_seg="Now the budget."))

    def test_blank_query_returns_nothing(self, tmp_path):
        _, conn = make_db(tmp_path)
        self._seed(conn)
        assert db.search_slides(conn, "") == []

    def test_fts5_finds_slide_by_title(self, tmp_path):
        _, conn = make_db(tmp_path)
        self._seed(conn)
        hits = db.search_slides(conn, "Budget overview")
        assert [h["timestamp_sec"] for h in hits] == [2.0]

    def test_fts5_finds_slide_by_aligned_transcript(self, tmp_path):
        _, conn = make_db(tmp_path)
        self._seed(conn)
        hits = db.search_slides(conn, "alkane ladder")
        assert [h["timestamp_sec"] for h in hits] == [1.0]

    def test_like_fallback_when_fts5_unavailable(self, tmp_path, monkeypatch):
        _, conn = make_db(tmp_path)
        self._seed(conn)
        monkeypatch.setattr(db, "FTS5_AVAILABLE", False)
        hits = db.search_slides(conn, "Budget")
        assert [h["timestamp_sec"] for h in hits] == [2.0]


class TestSearchAll:
    def test_returns_both_result_sets(self, tmp_path):
        path, conn = make_db(tmp_path)
        db.upsert_transcript(conn, "C:\\rec\\gc.mp4", "large-v3", "en", 1, 5.0,
                             notes_generated=True, notes_text="calibration standard")
        db.insert_slide(conn, make_slide_record(title="calibration standard"))
        conn.close()
        out = db.search_all(path, "calibration standard")
        assert set(out) == {"notes", "slides"}
        assert len(out["notes"]) == 1
        assert len(out["slides"]) == 1


# -----------------------------------------------------------------
# Slides
# -----------------------------------------------------------------

class TestInsertSlide:
    def test_returns_rowid_and_stores_the_row(self, tmp_path):
        _, conn = make_db(tmp_path)
        slide_id = db.insert_slide(conn, make_slide_record())
        assert slide_id == 1
        rows = db.get_slides_for_video(conn, "C:\\v\\talk.mp4")
        assert len(rows) == 1
        assert rows[0]["title"] == "Retention index calibration"

    def test_partial_record_raises_rather_than_inserting(self, tmp_path):
        """The INSERT uses named parameters, so a dict missing keys fails
        inside sqlite with no friendly message - worth pinning so a caller
        building records by hand finds out here."""
        _, conn = make_db(tmp_path)
        with pytest.raises(sqlite3.ProgrammingError):
            db.insert_slide(conn, {"video_path": "C:\\v\\talk.mp4", "timestamp_sec": 1.0})
        assert db.get_slides_for_video(conn, "C:\\v\\talk.mp4") == []

    def test_slides_are_returned_in_timestamp_order(self, tmp_path):
        _, conn = make_db(tmp_path)
        for ts in (30.0, 10.0, 20.0):
            db.insert_slide(conn, make_slide_record(timestamp_sec=ts))
        got = [r["timestamp_sec"] for r in db.get_slides_for_video(conn, "C:\\v\\talk.mp4")]
        assert got == [10.0, 20.0, 30.0]

    def test_slides_are_scoped_per_video(self, tmp_path):
        _, conn = make_db(tmp_path)
        db.insert_slide(conn, make_slide_record(video_path="C:\\v\\a.mp4"))
        db.insert_slide(conn, make_slide_record(video_path="C:\\v\\b.mp4"))
        assert len(db.get_slides_for_video(conn, "C:\\v\\a.mp4")) == 1


class TestUpdateSlideAnnotation:
    def test_updates_matched_slide_and_reports_true(self, tmp_path):
        _, conn = make_db(tmp_path)
        db.insert_slide(conn, make_slide_record(timestamp_sec=12.5, title="", bullets="[]"))
        ok = db.update_slide_annotation(conn, "C:\\v\\talk.mp4", 12.5,
                                        "Filled in", '["a", "b"]', "content")
        assert ok is True
        row = db.get_slides_for_video(conn, "C:\\v\\talk.mp4")[0]
        assert row["title"] == "Filled in"
        assert row["bullets"] == '["a", "b"]'

    def test_unmatched_timestamp_reports_false(self, tmp_path):
        _, conn = make_db(tmp_path)
        db.insert_slide(conn, make_slide_record(timestamp_sec=12.5))
        assert db.update_slide_annotation(
            conn, "C:\\v\\talk.mp4", 99.0, "x", "[]", "content") is False

    def test_retry_does_not_create_a_duplicate_row(self, tmp_path):
        _, conn = make_db(tmp_path)
        db.insert_slide(conn, make_slide_record(timestamp_sec=5.0))
        db.update_slide_annotation(conn, "C:\\v\\talk.mp4", 5.0, "one", "[]", "content")
        db.update_slide_annotation(conn, "C:\\v\\talk.mp4", 5.0, "two", "[]", "content")
        assert len(db.get_slides_for_video(conn, "C:\\v\\talk.mp4")) == 1


# -----------------------------------------------------------------
# update_source_path (after move_processed_files relocates a source)
# -----------------------------------------------------------------

class TestUpdateSourcePath:
    def test_repoints_both_transcripts_and_slides(self, tmp_path):
        _, conn = make_db(tmp_path)
        old, new = "C:\\rec\\talk.mp4", "C:\\rec\\_processed\\talk.mp4"
        db.upsert_transcript(conn, old, "large-v3", "de", 5, 60.0, notes_generated=True)
        db.insert_slide(conn, make_slide_record(video_path=old))

        db.update_source_path(conn, old, new)

        row = db.list_transcripts(conn)[0]
        assert row["file_path"] == new
        assert row["file_name"] == "talk.mp4"
        assert len(db.get_slides_for_video(conn, new)) == 1
        assert db.get_slides_for_video(conn, old) == []

    def test_zero_matching_rows_is_not_an_error(self, tmp_path):
        """Audio-only runs have no slides rows and --no-whisper runs have no
        transcripts row, so a no-op update must stay silent."""
        _, conn = make_db(tmp_path)
        db.update_source_path(conn, "C:\\nothing.mp4", "C:\\still\\nothing.mp4")
        assert db.list_transcripts(conn) == []

    def test_leaves_other_recordings_untouched(self, tmp_path):
        _, conn = make_db(tmp_path)
        db.upsert_transcript(conn, "C:\\rec\\a.mp4", "large-v3", "de", 1, 1.0)
        db.upsert_transcript(conn, "C:\\rec\\b.mp4", "large-v3", "de", 1, 1.0)
        db.update_source_path(conn, "C:\\rec\\a.mp4", "C:\\done\\a.mp4")
        paths = {r["file_path"] for r in db.list_transcripts(conn)}
        assert paths == {"C:\\done\\a.mp4", "C:\\rec\\b.mp4"}


# -----------------------------------------------------------------
# Known speakers roster
# -----------------------------------------------------------------

class TestKnownSpeakers:
    def test_insert_then_look_up_by_name_and_id(self, tmp_path):
        _, conn = make_db(tmp_path)
        speaker_id = db.insert_known_speaker(conn, "Jane Doe", b"\x01\x02\x03\x04", 4)
        assert speaker_id == 1
        by_name = db.get_known_speaker_by_name(conn, "Jane Doe")
        by_id = db.get_known_speaker(conn, speaker_id)
        assert by_name["embedding"] == b"\x01\x02\x03\x04"
        assert by_name["embedding_dim"] == 4
        assert by_name["sample_count"] == 1
        assert by_id["name"] == "Jane Doe"

    def test_duplicate_name_raises_integrity_error(self, tmp_path):
        _, conn = make_db(tmp_path)
        db.insert_known_speaker(conn, "Jane Doe", b"\x01", 1)
        with pytest.raises(sqlite3.IntegrityError):
            db.insert_known_speaker(conn, "Jane Doe", b"\x02", 1)

    def test_unknown_lookups_return_none(self, tmp_path):
        _, conn = make_db(tmp_path)
        assert db.get_known_speaker_by_name(conn, "Nobody") is None
        assert db.get_known_speaker(conn, 999) is None

    def test_list_is_sorted_case_insensitively(self, tmp_path):
        _, conn = make_db(tmp_path)
        for name in ("zoe", "Adam", "bea"):
            db.insert_known_speaker(conn, name, b"\x01", 1)
        assert [s["name"] for s in db.list_known_speakers(conn)] == ["Adam", "bea", "zoe"]

    def test_update_embedding_replaces_vector_and_sample_count(self, tmp_path):
        _, conn = make_db(tmp_path)
        speaker_id = db.insert_known_speaker(conn, "Jane", b"\x01\x01", 2)
        db.update_known_speaker_embedding(conn, speaker_id, b"\x09\x09", 4)
        row = db.get_known_speaker(conn, speaker_id)
        assert row["embedding"] == b"\x09\x09"
        assert row["sample_count"] == 4

    def test_rename(self, tmp_path):
        _, conn = make_db(tmp_path)
        speaker_id = db.insert_known_speaker(conn, "Jane", b"\x01", 1)
        db.rename_known_speaker(conn, speaker_id, "Jane Doe")
        assert db.get_known_speaker(conn, speaker_id)["name"] == "Jane Doe"
        assert db.get_known_speaker_by_name(conn, "Jane") is None

    def test_rename_onto_an_existing_name_raises(self, tmp_path):
        _, conn = make_db(tmp_path)
        db.insert_known_speaker(conn, "Jane", b"\x01", 1)
        other = db.insert_known_speaker(conn, "John", b"\x02", 1)
        with pytest.raises(sqlite3.IntegrityError):
            db.rename_known_speaker(conn, other, "Jane")

    def test_delete_removes_only_that_speaker(self, tmp_path):
        _, conn = make_db(tmp_path)
        keep = db.insert_known_speaker(conn, "Keep", b"\x01", 1)
        drop = db.insert_known_speaker(conn, "Drop", b"\x02", 1)
        db.delete_known_speaker(conn, drop)
        assert [s["speaker_id"] for s in db.list_known_speakers(conn)] == [keep]


# -----------------------------------------------------------------
# run_log
# -----------------------------------------------------------------

class TestInsertRunLog:
    def test_returns_rowid_and_stores_status(self, tmp_path):
        _, conn = make_db(tmp_path)
        run_id = db.insert_run_log(conn, "C:\\rec\\a.mp4", "video", 12, "success", "all good")
        assert run_id == 1
        row = conn.execute("SELECT * FROM run_log WHERE run_id = ?", (run_id,)).fetchone()
        assert row["status"] == "success"
        assert row["slides_found"] == 12
        assert row["started_at"]  # server-side default is populated


# -----------------------------------------------------------------
# DB file management
# -----------------------------------------------------------------

class TestCreateNewDb:
    def test_creates_a_validated_empty_database(self, tmp_path):
        path = str(tmp_path / "fresh.db")
        assert db.create_new_db(path) == path
        conn = db.get_connection(path)
        assert "transcripts" in table_names(conn)
        assert db.list_transcripts(conn) == []

    def test_refuses_to_overwrite_an_existing_file(self, tmp_path):
        path = tmp_path / "taken.db"
        path.write_text("not really a database", encoding="utf-8")
        with pytest.raises(FileExistsError):
            db.create_new_db(str(path))


class TestRenameDb:
    def test_moves_the_file(self, tmp_path):
        """The connection must be closed first: rename_db is documented as
        safe only between runs, and on Windows os.rename raises
        PermissionError (WinError 32) while any handle is still open."""
        old, conn = make_db(tmp_path, "old.db")
        conn.close()
        new = str(tmp_path / "new.db")
        assert db.rename_db(old, new) == new
        assert not (tmp_path / "old.db").exists()
        assert (tmp_path / "new.db").exists()

    def test_renamed_database_is_still_readable(self, tmp_path):
        old, conn = make_db(tmp_path, "old.db")
        db.upsert_transcript(conn, "C:\\rec\\a.mp4", "large-v3", "de", 7, 70.0)
        conn.close()
        new = db.rename_db(old, str(tmp_path / "new.db"))
        assert len(db.list_transcripts(db.get_connection(new))) == 1

    def test_missing_source_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            db.rename_db(str(tmp_path / "nope.db"), str(tmp_path / "new.db"))

    def test_existing_target_raises(self, tmp_path):
        old, conn = make_db(tmp_path, "old.db")
        conn.close()
        taken = tmp_path / "taken.db"
        taken.write_text("x", encoding="utf-8")
        with pytest.raises(FileExistsError):
            db.rename_db(old, str(taken))


class TestGetDbStats:
    def test_missing_file_reports_exists_false(self, tmp_path):
        stats = db.get_db_stats(str(tmp_path / "absent.db"))
        assert stats == {"exists": False, "path": str(tmp_path / "absent.db")}

    def test_counts_rows_across_tables(self, tmp_path):
        path, conn = make_db(tmp_path)
        db.upsert_transcript(conn, "C:\\rec\\a.mp4", "large-v3", "de", 1, 1.0,
                             notes_generated=True, notes_text="notes here")
        db.upsert_transcript(conn, "C:\\rec\\b.mp4", "large-v3", "de", 1, 1.0,
                             notes_generated=False)
        db.insert_slide(conn, make_slide_record())
        db.insert_run_log(conn, "C:\\rec\\a.mp4", "video", 1, "success")
        conn.close()

        stats = db.get_db_stats(path)
        assert stats["exists"] is True
        assert stats["transcripts"] == 2
        assert stats["notes_generated"] == 1
        assert stats["slides"] == 1
        assert stats["runs"] == 1
        assert stats["size_bytes"] > 0


# -----------------------------------------------------------------
# CLI printers
# -----------------------------------------------------------------

class TestPrintSummary:
    def test_missing_database_prints_a_notice_and_returns(self, tmp_path, capsys):
        db.print_summary(str(tmp_path / "absent.db"))
        assert capsys.readouterr().out  # says something rather than raising

    def test_lists_a_processed_file(self, tmp_path, capsys):
        path, conn = make_db(tmp_path)
        db.upsert_transcript(conn, "C:\\rec\\call.mp4", "large-v3", "de", 5, 60.0)
        conn.close()
        db.print_summary(path)
        assert "call.mp4" in capsys.readouterr().out
