# =============================================================
#  Transkription_Notes_Pipeline - tests/test_run_pipeline.py
#  Unit tests for run_pipeline.py's pure/dispatch logic (task #70):
#  apply_slide_edits (omit/merge/title_overrides) and run_batch_rows
#  (skip-if-done, per-row override merging, stop_check). Heavy stages
#  (transcription, VLM, db I/O) are monkeypatched out, so this needs
#  no whisperx/torch/pyannote/ffmpeg - runs in the bare sandbox.
#
#  Run: pytest tests/test_run_pipeline.py -v
# =============================================================

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_pipeline
import db


# -----------------------------------------------------------------
# apply_slide_edits
# -----------------------------------------------------------------

def make_slide(idx, title=None, bullets=None, ts=0.0, transcript=""):
    return {
        "title": title if title is not None else f"Slide {idx}",
        "bullets": bullets or [],
        "timestamp_sec": ts,
        "frame_index": idx,
        "transcript_seg": transcript,
        "snapshot_path": f"snap_{idx}.png",
    }


class TestApplySlideEdits:
    def test_no_edits_returns_equivalent_list(self):
        slides = [make_slide(0), make_slide(1), make_slide(2)]
        out = run_pipeline.apply_slide_edits(slides, {})
        assert [s["title"] for s in out] == ["Slide 0", "Slide 1", "Slide 2"]

    def test_does_not_mutate_inputs(self):
        slides = [make_slide(0), make_slide(1)]
        edits = {"omit_indices": [0], "title_overrides": {"1": "Changed"}}
        run_pipeline.apply_slide_edits(slides, edits)
        assert slides[0]["title"] == "Slide 0"
        assert slides[1]["title"] == "Slide 1"

    def test_omit_drops_slide(self):
        slides = [make_slide(0), make_slide(1), make_slide(2)]
        out = run_pipeline.apply_slide_edits(slides, {"omit_indices": [1]})
        assert [s["title"] for s in out] == ["Slide 0", "Slide 2"]

    def test_merge_next_keeps_last_title_and_min_timestamp(self):
        slides = [
            make_slide(0, title="Building...", ts=10.0, bullets=["a"]),
            make_slide(1, title="Complete", ts=12.0, bullets=["a", "b"]),
        ]
        out = run_pipeline.apply_slide_edits(slides, {"merge_next_indices": [0]})
        assert len(out) == 1
        assert out[0]["title"] == "Complete"
        assert out[0]["timestamp_sec"] == 10.0
        assert out[0]["bullets"] == ["a", "b"]

    def test_chained_merge_combines_three(self):
        slides = [make_slide(0, ts=1.0), make_slide(1, ts=2.0), make_slide(2, ts=3.0)]
        out = run_pipeline.apply_slide_edits(slides, {"merge_next_indices": [0, 1]})
        assert len(out) == 1
        assert out[0]["timestamp_sec"] == 1.0
        assert out[0]["title"] == "Slide 2"

    def test_title_override_applied_before_merge(self):
        """A title override on the slide that gets merged away should not
        surface; the override on the slide whose title survives the merge
        (the last one in the group) should."""
        slides = [make_slide(0, title="A"), make_slide(1, title="B")]
        edits = {"merge_next_indices": [0], "title_overrides": {"0": "A-edited", "1": "B-edited"}}
        out = run_pipeline.apply_slide_edits(slides, edits)
        assert out[0]["title"] == "B-edited"

    def test_title_override_alone(self):
        slides = [make_slide(0, title="Original")]
        out = run_pipeline.apply_slide_edits(slides, {"title_overrides": {"0": "New Title"}})
        assert out[0]["title"] == "New Title"

    def test_omit_and_merge_combined(self):
        slides = [make_slide(0, ts=1.0), make_slide(1, ts=2.0), make_slide(2, ts=3.0), make_slide(3, ts=4.0)]
        # omit slide 1, merge slide 0 forward (should skip over the omitted
        # slide 1 and merge with slide 2).
        out = run_pipeline.apply_slide_edits(
            slides, {"omit_indices": [1], "merge_next_indices": [0]})
        assert len(out) == 2
        assert out[0]["timestamp_sec"] == 1.0
        assert out[0]["title"] == "Slide 2"
        assert out[1]["title"] == "Slide 3"

    def test_empty_slide_list(self):
        assert run_pipeline.apply_slide_edits([], {"omit_indices": [0]}) == []


# -----------------------------------------------------------------
# run_batch_rows
# -----------------------------------------------------------------

class FakeConn:
    def close(self):
        pass


class TestRunBatchRows:
    def _patch_db(self, monkeypatch, completed_files=()):
        """completed_files: set of file paths db should report as already
        processed (skip-if-done), everything else counts as not done."""
        monkeypatch.setattr(db, "validate_schema", lambda path: True)
        monkeypatch.setattr(db, "get_connection", lambda path: FakeConn())

        def fake_get_completed(conn, file_path):
            if file_path in completed_files:
                return {"processed_at": "2026-01-01T00:00:00"}
            return None

        monkeypatch.setattr(db, "get_completed_transcript", fake_get_completed)

    def test_calls_process_file_for_each_row(self, monkeypatch, tmp_path):
        self._patch_db(monkeypatch)
        calls = []

        def fake_process_file(file, overrides, stop_check=None):
            calls.append((file, overrides))
            return True

        monkeypatch.setattr(run_pipeline, "process_file", fake_process_file)
        monkeypatch.setattr(run_pipeline, "write_log", lambda *a, **kw: None)

        rows = [
            {"file": str(tmp_path / "a.mp4"), "language": "en"},
            {"file": str(tmp_path / "b.mp4")},
        ]
        run_pipeline.run_batch_rows(rows, {"db_path": str(tmp_path / "x.db")})
        assert len(calls) == 2
        assert calls[0][1]["whisper_language"] == "en"
        assert "whisper_language" not in calls[1][1] or calls[1][1].get("whisper_language") in (None, "")

    def test_skips_blank_file_rows(self, monkeypatch, tmp_path):
        self._patch_db(monkeypatch)
        calls = []
        monkeypatch.setattr(run_pipeline, "process_file",
                            lambda file, overrides, stop_check=None: calls.append(file) or True)
        monkeypatch.setattr(run_pipeline, "write_log", lambda *a, **kw: None)

        rows = [{"file": ""}, {"file": "   "}, {"file": str(tmp_path / "a.mp4")}]
        run_pipeline.run_batch_rows(rows, {"db_path": str(tmp_path / "x.db")})
        assert calls == [str(tmp_path / "a.mp4")]

    def test_skips_already_completed_unless_forced(self, monkeypatch, tmp_path):
        done_file = str(tmp_path / "done.mp4")
        self._patch_db(monkeypatch, completed_files={done_file})
        calls = []
        monkeypatch.setattr(run_pipeline, "process_file",
                            lambda file, overrides, stop_check=None: calls.append(file) or True)
        monkeypatch.setattr(run_pipeline, "write_log", lambda *a, **kw: None)

        rows = [{"file": done_file}, {"file": str(tmp_path / "new.mp4")}]
        run_pipeline.run_batch_rows(rows, {"db_path": str(tmp_path / "x.db")})
        assert calls == [str(tmp_path / "new.mp4")]

    def test_force_retranscribe_reprocesses_completed(self, monkeypatch, tmp_path):
        done_file = str(tmp_path / "done.mp4")
        self._patch_db(monkeypatch, completed_files={done_file})
        calls = []
        monkeypatch.setattr(run_pipeline, "process_file",
                            lambda file, overrides, stop_check=None: calls.append(file) or True)
        monkeypatch.setattr(run_pipeline, "write_log", lambda *a, **kw: None)

        rows = [{"file": done_file}]
        run_pipeline.run_batch_rows(
            rows, {"db_path": str(tmp_path / "x.db"), "force_retranscribe": True})
        assert calls == [done_file]

    def test_stop_check_halts_before_remaining_rows(self, monkeypatch, tmp_path):
        self._patch_db(monkeypatch)
        calls = []
        monkeypatch.setattr(run_pipeline, "process_file",
                            lambda file, overrides, stop_check=None: calls.append(file) or True)
        monkeypatch.setattr(run_pipeline, "write_log", lambda *a, **kw: None)

        rows = [{"file": str(tmp_path / f"{i}.mp4")} for i in range(4)]
        # Stop before the 3rd row is processed.
        stop_after = {"count": 0}

        def stop_check():
            stop_after["count"] += 1
            return stop_after["count"] > 2

        run_pipeline.run_batch_rows(rows, {"db_path": str(tmp_path / "x.db")}, stop_check=stop_check)
        assert len(calls) == 2

    def test_row_qa_times_passed_through(self, monkeypatch, tmp_path):
        self._patch_db(monkeypatch)
        captured = {}
        monkeypatch.setattr(
            run_pipeline, "process_file",
            lambda file, overrides, stop_check=None: captured.update(overrides) or True)
        monkeypatch.setattr(run_pipeline, "write_log", lambda *a, **kw: None)

        rows = [{"file": str(tmp_path / "a.mp4"), "qa_start_time_sec": 975.0, "qa_end_time_sec": None}]
        run_pipeline.run_batch_rows(rows, {"db_path": str(tmp_path / "x.db")})
        assert captured["qa_start_time_sec"] == 975.0
        assert "qa_end_time_sec" not in captured or captured.get("qa_end_time_sec") is None

    def test_exception_in_process_file_does_not_abort_batch(self, monkeypatch, tmp_path):
        self._patch_db(monkeypatch)
        calls = []

        def fake_process_file(file, overrides, stop_check=None):
            calls.append(file)
            if "bad" in file:
                raise RuntimeError("boom")
            return True

        monkeypatch.setattr(run_pipeline, "process_file", fake_process_file)
        monkeypatch.setattr(run_pipeline, "write_log", lambda *a, **kw: None)

        rows = [{"file": str(tmp_path / "bad.mp4")}, {"file": str(tmp_path / "good.mp4")}]
        run_pipeline.run_batch_rows(rows, {"db_path": str(tmp_path / "x.db")})
        assert len(calls) == 2


# -----------------------------------------------------------------
# run_batch_rows: progress_callback (task #74)
# -----------------------------------------------------------------

class TestRunBatchRowsProgressCallback:
    def _patch_db(self, monkeypatch, completed_files=()):
        monkeypatch.setattr(db, "validate_schema", lambda path: True)
        monkeypatch.setattr(db, "get_connection", lambda path: FakeConn())

        def fake_get_completed(conn, file_path):
            if file_path in completed_files:
                return {"processed_at": "2026-01-01T00:00:00"}
            return None

        monkeypatch.setattr(db, "get_completed_transcript", fake_get_completed)

    def test_running_then_done_on_success(self, monkeypatch, tmp_path):
        self._patch_db(monkeypatch)
        monkeypatch.setattr(run_pipeline, "process_file",
                            lambda file, overrides, stop_check=None: True)
        monkeypatch.setattr(run_pipeline, "write_log", lambda *a, **kw: None)

        events = []
        rows = [{"file": str(tmp_path / "a.mp4")}]
        run_pipeline.run_batch_rows(rows, {"db_path": str(tmp_path / "x.db")},
                                    progress_callback=lambda i, status: events.append((i, status)))
        assert events == [(1, "running"), (1, "done")]

    def test_failed_status_when_process_file_returns_false(self, monkeypatch, tmp_path):
        self._patch_db(monkeypatch)
        monkeypatch.setattr(run_pipeline, "process_file",
                            lambda file, overrides, stop_check=None: False)
        monkeypatch.setattr(run_pipeline, "write_log", lambda *a, **kw: None)

        events = []
        rows = [{"file": str(tmp_path / "a.mp4")}]
        run_pipeline.run_batch_rows(rows, {"db_path": str(tmp_path / "x.db")},
                                    progress_callback=lambda i, status: events.append((i, status)))
        assert events == [(1, "running"), (1, "failed")]

    def test_skipped_status_for_already_done_row(self, monkeypatch, tmp_path):
        done_file = str(tmp_path / "done.mp4")
        self._patch_db(monkeypatch, completed_files={done_file})
        monkeypatch.setattr(run_pipeline, "process_file",
                            lambda file, overrides, stop_check=None: True)
        monkeypatch.setattr(run_pipeline, "write_log", lambda *a, **kw: None)

        events = []
        rows = [{"file": done_file}]
        run_pipeline.run_batch_rows(rows, {"db_path": str(tmp_path / "x.db")},
                                    progress_callback=lambda i, status: events.append((i, status)))
        assert events == [(1, "skipped")]

    def test_error_status_on_exception(self, monkeypatch, tmp_path):
        self._patch_db(monkeypatch)

        def fake_process_file(file, overrides, stop_check=None):
            raise RuntimeError("boom")

        monkeypatch.setattr(run_pipeline, "process_file", fake_process_file)
        monkeypatch.setattr(run_pipeline, "write_log", lambda *a, **kw: None)

        events = []
        rows = [{"file": str(tmp_path / "a.mp4")}]
        run_pipeline.run_batch_rows(rows, {"db_path": str(tmp_path / "x.db")},
                                    progress_callback=lambda i, status: events.append((i, status)))
        assert events == [(1, "running"), (1, "error")]

    def test_index_matches_row_position_skipping_blank_files(self, monkeypatch, tmp_path):
        """A blank-file row is skipped before ever reaching the loop body,
        so it must not consume an index - the visible index always matches
        enumerate(rows, 1) counting from the row's actual list position."""
        self._patch_db(monkeypatch)
        monkeypatch.setattr(run_pipeline, "process_file",
                            lambda file, overrides, stop_check=None: True)
        monkeypatch.setattr(run_pipeline, "write_log", lambda *a, **kw: None)

        events = []
        rows = [{"file": ""}, {"file": str(tmp_path / "a.mp4")}]
        run_pipeline.run_batch_rows(rows, {"db_path": str(tmp_path / "x.db")},
                                    progress_callback=lambda i, status: events.append((i, status)))
        assert events == [(2, "running"), (2, "done")]

    def test_none_callback_does_not_raise(self, monkeypatch, tmp_path):
        self._patch_db(monkeypatch)
        monkeypatch.setattr(run_pipeline, "process_file",
                            lambda file, overrides, stop_check=None: True)
        monkeypatch.setattr(run_pipeline, "write_log", lambda *a, **kw: None)

        rows = [{"file": str(tmp_path / "a.mp4")}]
        # No progress_callback given at all - must behave exactly like the
        # pre-task-#74 signature, no exception.
        run_pipeline.run_batch_rows(rows, {"db_path": str(tmp_path / "x.db")})

    def test_callback_exception_is_swallowed(self, monkeypatch, tmp_path):
        """A GUI display glitch inside the callback must never take down
        the batch run itself."""
        self._patch_db(monkeypatch)
        monkeypatch.setattr(run_pipeline, "process_file",
                            lambda file, overrides, stop_check=None: True)
        monkeypatch.setattr(run_pipeline, "write_log", lambda *a, **kw: None)

        def bad_callback(i, status):
            raise ValueError("boom from GUI")

        rows = [{"file": str(tmp_path / "a.mp4")}]
        # Must not raise, and process_file must still have been called.
        run_pipeline.run_batch_rows(rows, {"db_path": str(tmp_path / "x.db")},
                                    progress_callback=bad_callback)


# -----------------------------------------------------------------
# get_speaker_labels / rename_speakers (task #76)
# -----------------------------------------------------------------

def make_segments(*speakers_and_text):
    """speakers_and_text: list of (speaker, text) tuples -> segment dicts
    with incrementing start/end, matching transcriber.transcribe's output
    shape closely enough for rename_speakers/get_speaker_labels."""
    segs = []
    for i, (speaker, text) in enumerate(speakers_and_text):
        segs.append({
            "start": float(i), "end": float(i) + 0.9, "text": text,
            "speaker": speaker, "words": [],
        })
    return segs


class TestGetSpeakerLabels:
    def test_missing_file_returns_empty(self, tmp_path):
        assert run_pipeline.get_speaker_labels(str(tmp_path / "nope_segments.json")) == []

    def test_corrupt_json_returns_empty(self, tmp_path):
        p = tmp_path / "a_segments.json"
        p.write_text("not valid json{{{", encoding="utf-8")
        assert run_pipeline.get_speaker_labels(str(p)) == []

    def test_returns_sorted_distinct_labels(self, tmp_path):
        p = tmp_path / "a_segments.json"
        segs = make_segments(("SPEAKER_01", "hi"), ("SPEAKER_00", "hello"), ("SPEAKER_01", "again"))
        p.write_text(json.dumps(segs), encoding="utf-8")
        assert run_pipeline.get_speaker_labels(str(p)) == ["SPEAKER_00", "SPEAKER_01"]

    def test_no_speaker_field_returns_empty(self, tmp_path):
        p = tmp_path / "a_segments.json"
        segs = [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": "", "words": []}]
        p.write_text(json.dumps(segs), encoding="utf-8")
        assert run_pipeline.get_speaker_labels(str(p)) == []


class TestRenameSpeakers:
    def _write_segments(self, tmp_path, stem="a"):
        p = tmp_path / f"{stem}_segments.json"
        segs = make_segments(("SPEAKER_00", "hello there"), ("SPEAKER_01", "hi back"),
                             ("SPEAKER_00", "how are you"))
        p.write_text(json.dumps(segs), encoding="utf-8")
        return p

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_pipeline.rename_speakers(str(tmp_path / "nope_segments.json"), {"SPEAKER_00": "Alice"})

    def test_renames_matching_labels_and_rewrites_json(self, tmp_path, monkeypatch):
        p = self._write_segments(tmp_path)
        monkeypatch.setattr(run_pipeline.transcriber, "write_transcript_files",
                            lambda *a, **kw: {"speakers": str(tmp_path / "a_transcript_speakers.txt"),
                                              "text": str(tmp_path / "a_text.txt"),
                                              "srt": str(tmp_path / "a_transcript.srt")})
        result = run_pipeline.rename_speakers(str(p), {"SPEAKER_00": "Alice"})
        assert result["renamed_segments"] == 2
        assert result["labels_found"] == ["SPEAKER_00", "SPEAKER_01"]
        saved = json.loads(p.read_text(encoding="utf-8"))
        assert [s["speaker"] for s in saved] == ["Alice", "SPEAKER_01", "Alice"]

    def test_unmapped_labels_untouched(self, tmp_path, monkeypatch):
        p = self._write_segments(tmp_path)
        monkeypatch.setattr(run_pipeline.transcriber, "write_transcript_files",
                            lambda *a, **kw: {"speakers": "x", "text": "y", "srt": "z"})
        run_pipeline.rename_speakers(str(p), {"SPEAKER_00": "Alice"})
        saved = json.loads(p.read_text(encoding="utf-8"))
        assert "SPEAKER_01" in [s["speaker"] for s in saved]

    def test_blank_new_name_is_noop(self, tmp_path, monkeypatch):
        p = self._write_segments(tmp_path)
        monkeypatch.setattr(run_pipeline.transcriber, "write_transcript_files",
                            lambda *a, **kw: {"speakers": "x", "text": "y", "srt": "z"})
        result = run_pipeline.rename_speakers(str(p), {"SPEAKER_00": "", "SPEAKER_01": None})
        assert result["renamed_segments"] == 0

    def test_same_name_is_noop(self, tmp_path, monkeypatch):
        p = self._write_segments(tmp_path)
        monkeypatch.setattr(run_pipeline.transcriber, "write_transcript_files",
                            lambda *a, **kw: {"speakers": "x", "text": "y", "srt": "z"})
        result = run_pipeline.rename_speakers(str(p), {"SPEAKER_00": "SPEAKER_00"})
        assert result["renamed_segments"] == 0

    def test_regenerate_notes_calls_run_notes_only(self, tmp_path, monkeypatch):
        p = self._write_segments(tmp_path)
        speakers_path = str(tmp_path / "a_transcript_speakers.txt")
        monkeypatch.setattr(run_pipeline.transcriber, "write_transcript_files",
                            lambda *a, **kw: {"speakers": speakers_path, "text": "y", "srt": "z"})
        calls = []
        monkeypatch.setattr(run_pipeline, "run_notes_only",
                            lambda transcript_file, overrides: calls.append(transcript_file))
        result = run_pipeline.rename_speakers(str(p), {"SPEAKER_00": "Alice"}, regenerate_notes=True)
        assert calls == [speakers_path]
        assert result["notes_regenerated"] is True

    def test_regenerate_notes_skipped_when_nothing_renamed(self, tmp_path, monkeypatch):
        p = self._write_segments(tmp_path)
        monkeypatch.setattr(run_pipeline.transcriber, "write_transcript_files",
                            lambda *a, **kw: {"speakers": "x", "text": "y", "srt": "z"})
        calls = []
        monkeypatch.setattr(run_pipeline, "run_notes_only",
                            lambda transcript_file, overrides: calls.append(transcript_file))
        result = run_pipeline.rename_speakers(str(p), {}, regenerate_notes=True)
        assert calls == []
        assert result["notes_regenerated"] is False

    def test_regenerate_notes_failure_is_logged_not_raised(self, tmp_path, monkeypatch):
        p = self._write_segments(tmp_path)
        monkeypatch.setattr(run_pipeline.transcriber, "write_transcript_files",
                            lambda *a, **kw: {"speakers": "x", "text": "y", "srt": "z"})

        def fake_notes_only(transcript_file, overrides):
            raise RuntimeError("LLM unreachable")

        monkeypatch.setattr(run_pipeline, "run_notes_only", fake_notes_only)
        result = run_pipeline.rename_speakers(str(p), {"SPEAKER_00": "Alice"}, regenerate_notes=True)
        assert result["renamed_segments"] == 2
        assert result["notes_regenerated"] is False


# -----------------------------------------------------------------
# commit_speaker_identities (task #80)
#
# Uses a real temp SQLite db (db.validate_schema/get_connection) rather
# than mocking db.py: the interesting behavior here IS the DB roundtrip
# (insert vs. running-average update), so a mock would just restate the
# implementation. No pyannote/torch needed - embeddings are plain numpy
# arrays constructed by hand.
# -----------------------------------------------------------------

class TestCommitSpeakerIdentities:
    def _make_db(self, tmp_path):
        path = str(tmp_path / "test.db")
        db.validate_schema(path)
        return path

    def test_new_name_enrolls_new_speaker(self, tmp_path):
        import numpy as np
        db_path = self._make_db(tmp_path)
        suggestions = {"SPEAKER_00": {"suggested_name": None, "score": 0.0,
                                       "embedding": np.array([1.0, 0.0, 0.0])}}
        result = run_pipeline.commit_speaker_identities(
            db_path, suggestions, {"SPEAKER_00": "Anna"})
        assert result["SPEAKER_00"]["action"] == "enrolled"
        conn = db.get_connection(db_path)
        row = db.get_known_speaker_by_name(conn, "Anna")
        conn.close()
        assert row is not None
        assert row["sample_count"] == 1

    def test_existing_name_updates_running_average(self, tmp_path):
        import numpy as np
        import speaker_id
        db_path = self._make_db(tmp_path)
        conn = db.get_connection(db_path)
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        sid = db.insert_known_speaker(conn, "Anna", speaker_id.serialize_embedding(vec), 3)
        conn.close()

        suggestions = {"SPEAKER_00": {"suggested_name": "Anna", "score": 0.95,
                                       "embedding": np.array([1.0, 0.0, 0.0])}}
        result = run_pipeline.commit_speaker_identities(
            db_path, suggestions, {"SPEAKER_00": "Anna"})
        assert result["SPEAKER_00"] == {"speaker_id": sid, "action": "matched"}

        conn = db.get_connection(db_path)
        row = db.get_known_speaker(conn, sid)
        conn.close()
        assert row["sample_count"] == 2

    def test_blank_name_choice_is_skipped(self, tmp_path):
        import numpy as np
        db_path = self._make_db(tmp_path)
        suggestions = {"SPEAKER_00": {"suggested_name": None, "score": 0.0,
                                       "embedding": np.array([1.0, 0.0, 0.0])}}
        result = run_pipeline.commit_speaker_identities(
            db_path, suggestions, {"SPEAKER_00": "   "})
        assert result == {}
        conn = db.get_connection(db_path)
        assert db.list_known_speakers(conn) == []
        conn.close()

    def test_label_missing_from_suggestions_is_skipped(self, tmp_path):
        db_path = self._make_db(tmp_path)
        result = run_pipeline.commit_speaker_identities(
            db_path, {}, {"SPEAKER_00": "Anna"})
        assert result == {}

    def test_multiple_labels_mixed_enroll_and_match(self, tmp_path):
        import numpy as np
        import speaker_id
        db_path = self._make_db(tmp_path)
        conn = db.get_connection(db_path)
        vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        sid = db.insert_known_speaker(conn, "Bert", speaker_id.serialize_embedding(vec), 3)
        conn.close()

        suggestions = {
            "SPEAKER_00": {"suggested_name": None, "score": 0.0,
                           "embedding": np.array([1.0, 0.0, 0.0])},
            "SPEAKER_01": {"suggested_name": "Bert", "score": 0.99,
                           "embedding": np.array([0.0, 1.0, 0.0])},
        }
        result = run_pipeline.commit_speaker_identities(
            db_path, suggestions, {"SPEAKER_00": "Carla", "SPEAKER_01": "Bert"})
        assert result["SPEAKER_00"]["action"] == "enrolled"
        assert result["SPEAKER_01"] == {"speaker_id": sid, "action": "matched"}
