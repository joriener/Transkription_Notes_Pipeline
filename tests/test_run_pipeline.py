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

import base64
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_pipeline
import db
import anthropic


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

    def test_row_srt_import_path_calls_import_before_process_file(self, monkeypatch, tmp_path):
        """Task: batch YouTube import - a row with srt_import_path set
        should import the .srt (skipping transcription for that file)
        BEFORE process_file runs, not instead of it."""
        self._patch_db(monkeypatch)
        calls = []
        monkeypatch.setattr(
            run_pipeline, "import_srt_transcript",
            lambda file, srt, overrides: calls.append(("import", file, srt)) or True)
        monkeypatch.setattr(
            run_pipeline, "process_file",
            lambda file, overrides, stop_check=None: calls.append(("process", file)) or True)
        monkeypatch.setattr(run_pipeline, "write_log", lambda *a, **kw: None)

        rows = [{"file": str(tmp_path / "a.mp4"), "srt_import_path": str(tmp_path / "a.srt")}]
        run_pipeline.run_batch_rows(rows, {"db_path": str(tmp_path / "x.db")})

        assert calls == [
            ("import", str(tmp_path / "a.mp4"), str(tmp_path / "a.srt")),
            ("process", str(tmp_path / "a.mp4")),
        ]

    def test_row_without_srt_import_path_skips_import_entirely(self, monkeypatch, tmp_path):
        self._patch_db(monkeypatch)
        import_calls = []
        monkeypatch.setattr(
            run_pipeline, "import_srt_transcript",
            lambda file, srt, overrides: import_calls.append(file) or True)
        monkeypatch.setattr(run_pipeline, "process_file",
                            lambda file, overrides, stop_check=None: True)
        monkeypatch.setattr(run_pipeline, "write_log", lambda *a, **kw: None)

        rows = [{"file": str(tmp_path / "a.mp4")}]
        run_pipeline.run_batch_rows(rows, {"db_path": str(tmp_path / "x.db")})
        assert import_calls == []

    def test_failed_srt_import_skips_file_and_counts_as_error(self, monkeypatch, tmp_path):
        """A failed import must NOT fall through to a real transcription -
        that would silently defeat the point of specifying the .srt."""
        self._patch_db(monkeypatch)
        process_calls = []
        monkeypatch.setattr(run_pipeline, "import_srt_transcript",
                            lambda file, srt, overrides: False)
        monkeypatch.setattr(
            run_pipeline, "process_file",
            lambda file, overrides, stop_check=None: process_calls.append(file) or True)
        monkeypatch.setattr(run_pipeline, "write_log", lambda *a, **kw: None)

        rows = [{"file": str(tmp_path / "a.mp4"), "srt_import_path": str(tmp_path / "a.srt")}]
        run_pipeline.run_batch_rows(rows, {"db_path": str(tmp_path / "x.db")})
        assert process_calls == []


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


# -----------------------------------------------------------------
# derive_heading_from_filename / resolve_output_prefix date-derived
# naming (Word export heading/filename feature)
# -----------------------------------------------------------------

class TestDeriveHeadingFromFilename:
    def test_date_and_time_with_generic_prefix_stripped(self):
        assert run_pipeline.derive_heading_from_filename(
            "Video_2020-04-07_154005.mp4") == "2020-04-07_154005"

    def test_compact_date_with_topic(self):
        assert run_pipeline.derive_heading_from_filename(
            "20260704_Sales_Call.mp4") == "2026-07-04_Sales Call"

    def test_zoom_style_prefix_with_dash_time(self):
        assert run_pipeline.derive_heading_from_filename(
            "GMT20260704-143022_Weekly_Sync.mp4") == "2026-07-04_143022_Weekly Sync"

    def test_date_only_no_time_no_topic(self):
        assert run_pipeline.derive_heading_from_filename("2026-07-04.wav") == "2026-07-04"

    def test_no_date_returns_none(self):
        assert run_pipeline.derive_heading_from_filename("random_no_date_filename.mp4") is None

    def test_extension_stripped_before_matching(self):
        # A ".mp4" extension must never itself be mistaken for part of
        # the date/topic.
        result = run_pipeline.derive_heading_from_filename("20260704_Topic.mp4")
        assert ".mp4" not in result

    def test_only_generic_word_after_date_yields_no_topic(self):
        assert run_pipeline.derive_heading_from_filename("Meeting_2026-07-04.mp4") == "2026-07-04"


class TestExtractDatetimeFromFilename:
    """extract_datetime_from_filename (task #94): same pattern as
    derive_heading_from_filename, but returns a real datetime for
    correlate_calendar_recordings.py to compare against calendar event
    times, instead of a formatted heading string."""

    def test_date_and_time_extracted(self):
        result = run_pipeline.extract_datetime_from_filename("Video_2020-04-07_154005.mp4")
        assert result == datetime(2020, 4, 7, 15, 40, 5)

    def test_compact_date_and_time_extracted(self):
        result = run_pipeline.extract_datetime_from_filename("20260704_143022_Weekly_Sync.mp4")
        assert result == datetime(2026, 7, 4, 14, 30, 22)

    def test_date_only_no_time_returns_none(self):
        # No HHMMSS component - not precise enough to compare against
        # calendar event times, caller should fall back to file mtime.
        assert run_pipeline.extract_datetime_from_filename("2026-07-04.wav") is None

    def test_no_date_returns_none(self):
        assert run_pipeline.extract_datetime_from_filename("random_filename.mp4") is None

    def test_out_of_range_values_return_none_not_raise(self):
        # Matches the regex shape but is not a real date/time (month 13).
        assert run_pipeline.extract_datetime_from_filename("20261304_250000.mp4") is None


class TestResolveOutputPrefixDateHeading:
    def test_auto_derives_stem_from_filename_date(self):
        cfg = {"use_filename_date_heading": True}
        result = run_pipeline.resolve_output_prefix("/tmp/Video_2020-04-07_154005.mp4", cfg)
        assert result == str(Path("/tmp") / "2020-04-07_154005")

    def test_disabled_keeps_raw_stem(self):
        cfg = {"use_filename_date_heading": False}
        result = run_pipeline.resolve_output_prefix("/tmp/Video_2020-04-07_154005.mp4", cfg)
        assert result == str(Path("/tmp") / "Video_2020-04-07_154005")

    def test_basename_override_always_wins(self):
        cfg = {"output_basename_override": "MyCustomName", "use_filename_date_heading": True}
        result = run_pipeline.resolve_output_prefix("/tmp/Video_2020-04-07_154005.mp4", cfg)
        assert result == str(Path("/tmp") / "MyCustomName")

    def test_no_date_in_filename_falls_back_to_raw_stem(self):
        cfg = {"use_filename_date_heading": True}
        result = run_pipeline.resolve_output_prefix("/tmp/random_no_date.mp4", cfg)
        assert result == str(Path("/tmp") / "random_no_date")

    def test_output_dir_override_combined_with_date_stem(self, tmp_path):
        out_dir = tmp_path / "out"
        cfg = {"use_filename_date_heading": True, "output_dir_override": str(out_dir)}
        result = run_pipeline.resolve_output_prefix("/tmp/Video_2020-04-07_154005.mp4", cfg)
        assert result == str(out_dir / "2020-04-07_154005")

    def test_default_behavior_uses_date_heading_when_key_absent(self):
        # use_filename_date_heading defaults to True when the key is
        # missing from cfg entirely (matches config.py's default).
        cfg = {}
        result = run_pipeline.resolve_output_prefix("/tmp/Video_2020-04-07_154005.mp4", cfg)
        assert result == str(Path("/tmp") / "2020-04-07_154005")


# -----------------------------------------------------------------
# _sanitize_filename_component / build_date_subject_filename /
# resolve_output_prefix's use_date_subject_filename option
# -----------------------------------------------------------------

class TestSanitizeFilenameComponent:
    def test_strips_windows_illegal_characters(self):
        assert run_pipeline._sanitize_filename_component('Q3: Review <Draft>?') == "Q3 Review Draft"

    def test_collapses_whitespace_and_newlines(self):
        assert run_pipeline._sanitize_filename_component("Weekly\n  Sync   Call") == "Weekly Sync Call"

    def test_strips_trailing_dots_and_spaces(self):
        assert run_pipeline._sanitize_filename_component("Draft v2. ") == "Draft v2"

    def test_truncates_to_max_length(self):
        long_subject = "A" * 200
        out = run_pipeline._sanitize_filename_component(long_subject)
        assert len(out) == run_pipeline._MAX_SUBJECT_LEN

    def test_blank_input_returns_blank(self):
        assert run_pipeline._sanitize_filename_component("") == ""
        assert run_pipeline._sanitize_filename_component("   ") == ""


class TestBuildDateSubjectFilename:
    def test_uses_meeting_date_when_valid_iso(self):
        result = run_pipeline.build_date_subject_filename(
            "2026-07-03", "Q3 Kickoff", "/tmp/random_no_date.mp4")
        assert result == "20260703_Q3 Kickoff"

    def test_falls_back_to_filename_date_when_meeting_date_blank(self):
        result = run_pipeline.build_date_subject_filename(
            "", "Q3 Kickoff", "/tmp/Video_2020-04-07_154005.mp4")
        assert result == "20200407_Q3 Kickoff"

    def test_falls_back_to_filename_date_when_meeting_date_invalid(self):
        result = run_pipeline.build_date_subject_filename(
            "not-a-date", "Q3 Kickoff", "/tmp/Video_2020-04-07_154005.mp4")
        assert result == "20200407_Q3 Kickoff"

    def test_blank_title_returns_none(self):
        assert run_pipeline.build_date_subject_filename(
            "2026-07-03", "", "/tmp/Video_2020-04-07_154005.mp4") is None
        assert run_pipeline.build_date_subject_filename(
            "2026-07-03", "   ", "/tmp/Video_2020-04-07_154005.mp4") is None

    def test_falls_back_to_file_mtime_when_no_other_date(self, tmp_path):
        f = tmp_path / "random_no_date.mp4"
        f.write_bytes(b"x")
        result = run_pipeline.build_date_subject_filename("", "Q3 Kickoff", str(f))
        assert result.endswith("_Q3 Kickoff")
        assert len(result.split("_")[0]) == 8  # YYYYMMDD

    def test_sanitizes_title(self):
        result = run_pipeline.build_date_subject_filename(
            "2026-07-03", "Q3: Review?", "/tmp/random_no_date.mp4")
        assert result == "20260703_Q3 Review"


class TestResolveOutputPrefixDateSubject:
    def test_uses_date_subject_when_enabled_and_title_set(self):
        cfg = {"use_date_subject_filename": True, "meeting_date": "2026-07-03",
               "meeting_title": "Q3 Kickoff"}
        result = run_pipeline.resolve_output_prefix("/tmp/random_no_date.mp4", cfg)
        assert result == str(Path("/tmp") / "20260703_Q3 Kickoff")

    def test_falls_back_to_filename_date_heading_when_title_blank(self):
        cfg = {"use_date_subject_filename": True, "meeting_title": "",
               "use_filename_date_heading": True}
        result = run_pipeline.resolve_output_prefix("/tmp/Video_2020-04-07_154005.mp4", cfg)
        assert result == str(Path("/tmp") / "2020-04-07_154005")

    def test_basename_override_wins_over_date_subject(self):
        cfg = {"use_date_subject_filename": True, "meeting_title": "Q3 Kickoff",
               "output_basename_override": "MyCustomName"}
        result = run_pipeline.resolve_output_prefix("/tmp/random_no_date.mp4", cfg)
        assert result == str(Path("/tmp") / "MyCustomName")

    def test_disabled_by_default(self):
        # use_date_subject_filename defaults to False - a meeting_title
        # being set must not silently change naming unless the option is
        # explicitly turned on.
        cfg = {"meeting_title": "Q3 Kickoff", "meeting_date": "2026-07-03"}
        result = run_pipeline.resolve_output_prefix("/tmp/random_no_date.mp4", cfg)
        assert result == str(Path("/tmp") / "random_no_date")


# -----------------------------------------------------------------
# _dedupe_date_subject_stem: two different recordings on the same day
# with the same meeting title must not silently overwrite each other's
# output files under use_date_subject_filename.
# -----------------------------------------------------------------

class TestDedupeDateSubjectStem:
    def test_no_sidecar_yet_stem_used_as_is(self, tmp_path):
        result = run_pipeline._dedupe_date_subject_stem(
            tmp_path, "20260703_Q3 Kickoff", str(tmp_path / "a.mp4"))
        assert result == "20260703_Q3 Kickoff"

    def test_same_source_reuses_stem_for_cache_hit(self, tmp_path):
        # Simulates a cache-aware re-run of the SAME source file: the
        # sidecar from the earlier run already points at this exact file,
        # so the stem must be reused, not de-duplicated away, or the
        # *_segments.json cache would never be hit again.
        source = tmp_path / "a.mp4"
        source.write_bytes(b"x")
        sidecar = tmp_path / "20260703_Q3 Kickoff_source_media.txt"
        sidecar.write_text(str(source), encoding="utf-8")
        result = run_pipeline._dedupe_date_subject_stem(
            tmp_path, "20260703_Q3 Kickoff", str(source))
        assert result == "20260703_Q3 Kickoff"

    def test_different_source_appends_numeric_suffix(self, tmp_path):
        other_source = tmp_path / "b.mp4"
        other_source.write_bytes(b"x")
        sidecar = tmp_path / "20260703_Q3 Kickoff_source_media.txt"
        sidecar.write_text(str(other_source), encoding="utf-8")

        new_source = tmp_path / "c.mp4"
        new_source.write_bytes(b"x")
        result = run_pipeline._dedupe_date_subject_stem(
            tmp_path, "20260703_Q3 Kickoff", str(new_source))
        assert result == "20260703_Q3 Kickoff_2"

    def test_skips_taken_suffixes_until_free_one_found(self, tmp_path):
        other_source = tmp_path / "b.mp4"
        other_source.write_bytes(b"x")
        (tmp_path / "20260703_Q3 Kickoff_source_media.txt").write_text(
            str(other_source), encoding="utf-8")
        (tmp_path / "20260703_Q3 Kickoff_2_source_media.txt").write_text(
            str(other_source), encoding="utf-8")

        new_source = tmp_path / "c.mp4"
        new_source.write_bytes(b"x")
        result = run_pipeline._dedupe_date_subject_stem(
            tmp_path, "20260703_Q3 Kickoff", str(new_source))
        assert result == "20260703_Q3 Kickoff_3"

    def test_resolve_output_prefix_applies_dedup_end_to_end(self, tmp_path):
        other_source = tmp_path / "b.mp4"
        other_source.write_bytes(b"x")
        (tmp_path / "20260703_Q3 Kickoff_source_media.txt").write_text(
            str(other_source), encoding="utf-8")

        new_source = tmp_path / "c.mp4"
        new_source.write_bytes(b"x")
        cfg = {"use_date_subject_filename": True, "meeting_date": "2026-07-03",
               "meeting_title": "Q3 Kickoff"}
        result = run_pipeline.resolve_output_prefix(str(new_source), cfg)
        assert result == str(tmp_path / "20260703_Q3 Kickoff_2")


# -----------------------------------------------------------------
# db.update_source_path / move_processed_source_file / run_batch_folder's
# processed_subfolder_name exclusion (post-run archiving option)
# -----------------------------------------------------------------

class TestUpdateSourcePath:
    def _make_db(self, tmp_path):
        path = str(tmp_path / "test.db")
        db.validate_schema(path)
        return path

    def test_updates_transcripts_file_path_and_name(self, tmp_path):
        db_path = self._make_db(tmp_path)
        conn = db.get_connection(db_path)
        db.upsert_transcript(conn, "/old/a.mp4", "large-v2", "en", 10, 100.0)
        db.update_source_path(conn, "/old/a.mp4", "/new/_processed/a.mp4")
        row = conn.execute("SELECT file_path, file_name FROM transcripts").fetchone()
        conn.close()
        assert row["file_path"] == "/new/_processed/a.mp4"
        assert row["file_name"] == "a.mp4"

    def test_no_matching_row_is_a_silent_no_op(self, tmp_path):
        db_path = self._make_db(tmp_path)
        conn = db.get_connection(db_path)
        db.update_source_path(conn, "/old/nope.mp4", "/new/nope.mp4")  # must not raise
        conn.close()


class TestMoveProcessedSourceFile:
    def _make_db(self, tmp_path):
        path = str(tmp_path / "test.db")
        db.validate_schema(path)
        return path

    def test_disabled_returns_original_path_unchanged(self, tmp_path):
        f = tmp_path / "a.mp4"
        f.write_bytes(b"x")
        cfg = {"move_processed_files": False}
        result = run_pipeline.move_processed_source_file(str(f), str(tmp_path / "a"), cfg)
        assert result == str(f)
        assert f.exists()

    def test_moves_file_into_processed_subfolder(self, tmp_path):
        f = tmp_path / "a.mp4"
        f.write_bytes(b"x")
        cfg = {"move_processed_files": True, "db_path": self._make_db(tmp_path)}
        result = run_pipeline.move_processed_source_file(str(f), str(tmp_path / "a"), cfg)
        assert result == str(tmp_path / "_processed" / "a.mp4")
        assert not f.exists()
        assert Path(result).exists()

    def test_custom_subfolder_name(self, tmp_path):
        f = tmp_path / "a.mp4"
        f.write_bytes(b"x")
        cfg = {"move_processed_files": True, "processed_subfolder_name": "Archive",
               "db_path": self._make_db(tmp_path)}
        result = run_pipeline.move_processed_source_file(str(f), str(tmp_path / "a"), cfg)
        assert result == str(tmp_path / "Archive" / "a.mp4")

    def test_rewrites_source_media_sidecar(self, tmp_path):
        f = tmp_path / "a.mp4"
        f.write_bytes(b"x")
        output_prefix = str(tmp_path / "a")
        sidecar = Path(output_prefix + "_source_media.txt")
        sidecar.write_text(str(f), encoding="utf-8")
        cfg = {"move_processed_files": True, "db_path": self._make_db(tmp_path)}
        result = run_pipeline.move_processed_source_file(str(f), output_prefix, cfg)
        assert sidecar.read_text(encoding="utf-8") == result

    def test_updates_db_source_path(self, tmp_path):
        f = tmp_path / "a.mp4"
        f.write_bytes(b"x")
        db_path = self._make_db(tmp_path)
        conn = db.get_connection(db_path)
        db.upsert_transcript(conn, str(f), "large-v2", "en", 10, 100.0)
        conn.close()
        cfg = {"move_processed_files": True, "db_path": db_path}
        result = run_pipeline.move_processed_source_file(str(f), str(tmp_path / "a"), cfg)
        conn = db.get_connection(db_path)
        row = conn.execute("SELECT file_path FROM transcripts").fetchone()
        conn.close()
        assert row["file_path"] == result

    def test_does_not_overwrite_existing_file_at_destination(self, tmp_path):
        f = tmp_path / "a.mp4"
        f.write_bytes(b"original")
        dest_dir = tmp_path / "_processed"
        dest_dir.mkdir()
        (dest_dir / "a.mp4").write_bytes(b"already there")
        cfg = {"move_processed_files": True, "db_path": self._make_db(tmp_path)}
        result = run_pipeline.move_processed_source_file(str(f), str(tmp_path / "a"), cfg)
        assert result == str(f)
        assert f.read_bytes() == b"original"
        assert (dest_dir / "a.mp4").read_bytes() == b"already there"


class TestRunBatchFolderExcludesProcessedSubfolder:
    def test_processed_subfolder_excluded_from_scan(self, tmp_path, monkeypatch):
        (tmp_path / "a.mp4").write_bytes(b"x")
        processed = tmp_path / "_processed"
        processed.mkdir()
        (processed / "already_done.mp4").write_bytes(b"x")

        seen = []
        monkeypatch.setattr(
            run_pipeline, "process_file",
            lambda f, overrides, stop_check=None: (seen.append(f), True)[1])

        run_pipeline.run_batch_folder(str(tmp_path), {}, recursive=True)

        # Exact list, not a substring check: tmp_path itself is derived
        # from this test's own name ("..._excluded_from_scan"), which
        # happens to contain the literal text "_processed" - a substring
        # check on the full path would false-positive on that, unrelated
        # to whether the _processed SUBFOLDER was actually excluded.
        assert seen == [str(tmp_path / "a.mp4")]

    def test_custom_subfolder_name_excluded(self, tmp_path, monkeypatch):
        (tmp_path / "a.mp4").write_bytes(b"x")
        archive = tmp_path / "Archive"
        archive.mkdir()
        (archive / "already_done.mp4").write_bytes(b"x")

        seen = []
        monkeypatch.setattr(
            run_pipeline, "process_file",
            lambda f, overrides, stop_check=None: (seen.append(f), True)[1])

        run_pipeline.run_batch_folder(
            str(tmp_path), {"processed_subfolder_name": "Archive"}, recursive=True)

        assert seen == [str(tmp_path / "a.mp4")]

    def test_enhanced_tmp_wav_stray_file_excluded(self, tmp_path, monkeypatch):
        # A leftover "<prefix>_enhanced_tmp.wav" from an interrupted
        # enhance_audio run (see process_file, Step 1) matches "*.wav" and
        # must not be picked up as its own source recording - it points
        # at audio that was already unlink()d, so processing it fails.
        (tmp_path / "a.mp4").write_bytes(b"x")
        (tmp_path / "a_enhanced_tmp.wav").write_bytes(b"x")

        seen = []
        monkeypatch.setattr(
            run_pipeline, "process_file",
            lambda f, overrides, stop_check=None: (seen.append(f), True)[1])

        run_pipeline.run_batch_folder(str(tmp_path), {}, recursive=True)

        assert seen == [str(tmp_path / "a.mp4")]


# -----------------------------------------------------------------
# find_source_media (V1.19 regression fix)
#
# use_filename_date_heading (V1.17, default True) makes the segments.json
# stem a derived date/topic heading instead of the raw source filename
# stem. find_source_media used to do a plain same-stem glob match, which
# broke for nearly every run once that default shipped: it silently
# returned None, which disabled both the Rename Speakers dialog's
# voiceprint suggestions and its Play Sample button (both gated on this
# lookup succeeding). These tests cover the fix: a "_source_media.txt"
# sidecar (authoritative, written going forward), the legacy exact-stem
# match (still correct when date-heading is off), and a reverse-derive
# fallback so files already processed under V1.17/V1.18 resolve without
# reprocessing.
# -----------------------------------------------------------------

class TestFindSourceMedia:
    def test_sidecar_is_authoritative(self, tmp_path):
        src = tmp_path / "Video_2020-04-07_154005.mp4"
        src.write_bytes(b"fake")
        segments_json = tmp_path / "2020-04-07_154005_segments.json"
        segments_json.write_text("[]")
        (tmp_path / "2020-04-07_154005_source_media.txt").write_text(str(src))
        assert run_pipeline.find_source_media(str(segments_json)) == str(src)

    def test_legacy_exact_stem_match_without_sidecar(self, tmp_path):
        src = tmp_path / "meeting_notes.wav"
        src.write_bytes(b"fake")
        segments_json = tmp_path / "meeting_notes_segments.json"
        segments_json.write_text("[]")
        assert run_pipeline.find_source_media(str(segments_json)) == str(src)

    def test_reverse_derive_fallback_for_date_heading_stem(self, tmp_path):
        # Simulates a file processed under V1.17/V1.18 before the sidecar
        # existed: segments.json carries the derived stem, the source
        # file still has its original name, exact-stem match fails.
        src = tmp_path / "Video_2020-04-07_154005.mp4"
        src.write_bytes(b"fake")
        segments_json = tmp_path / "2020-04-07_154005_segments.json"
        segments_json.write_text("[]")
        assert run_pipeline.find_source_media(str(segments_json)) == str(src)

    def test_returns_none_when_source_media_missing(self, tmp_path):
        segments_json = tmp_path / "2020-04-07_154005_segments.json"
        segments_json.write_text("[]")
        assert run_pipeline.find_source_media(str(segments_json)) is None

    def test_sidecar_ignored_if_it_points_to_a_missing_file(self, tmp_path):
        # Sidecar exists but the path it records is stale (media moved/
        # deleted) - must fall through to the other strategies rather
        # than returning a dangling path.
        src = tmp_path / "Video_2020-04-07_154005.mp4"
        src.write_bytes(b"fake")
        segments_json = tmp_path / "2020-04-07_154005_segments.json"
        segments_json.write_text("[]")
        (tmp_path / "2020-04-07_154005_source_media.txt").write_text(
            str(tmp_path / "gone.mp4"))
        assert run_pipeline.find_source_media(str(segments_json)) == str(src)


# -----------------------------------------------------------------
# export_speaker_roster_json / apply_speaker_roster_json (task #92)
#
# "Process several recordings overnight, review every speaker in one
# JSON file the next morning" - no pyannote/torch needed for these
# tests: hf_token="" makes export skip voiceprint suggestion computation
# entirely (same code path as source media being unreachable), so the
# tests cover the file-discovery, JSON structure, and apply/rename/
# roster-commit logic on their own merits.
# -----------------------------------------------------------------

class TestExportSpeakerRosterJson:
    def _make_processed_file(self, tmp_path, stem, speakers, media_name=None):
        media_name = media_name or f"{stem}.wav"
        src = tmp_path / media_name
        src.write_bytes(b"fake")
        segments = [{"start": i * 1.0, "end": i * 1.0 + 0.5, "speaker": spk,
                     "text": "hello", "words": []}
                    for i, spk in enumerate(speakers)]
        segments_json = tmp_path / f"{stem}_segments.json"
        segments_json.write_text(json.dumps(segments), encoding="utf-8")
        (tmp_path / f"{stem}_source_media.txt").write_text(str(src), encoding="utf-8")
        return segments_json, src

    def _make_db(self, tmp_path):
        path = str(tmp_path / "test.db")
        db.validate_schema(path)
        return path

    def test_scans_folder_and_lists_every_speaker(self, tmp_path):
        self._make_processed_file(tmp_path, "call_one", ["SPEAKER_00", "SPEAKER_01"])
        self._make_processed_file(tmp_path, "call_two", ["SPEAKER_00"])
        db_path = self._make_db(tmp_path)
        out_json = tmp_path / "roster.json"

        result = run_pipeline.export_speaker_roster_json(
            str(tmp_path), str(out_json), hf_token="", threshold=0.75, db_path=db_path)

        assert result["files"] == 2
        assert result["speakers"] == 3
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert len(payload["files"]) == 2
        labels_by_file = {Path(f["segments_json"]).name: [s["label"] for s in f["speakers"]]
                          for f in payload["files"]}
        assert labels_by_file["call_one_segments.json"] == ["SPEAKER_00", "SPEAKER_01"]
        assert labels_by_file["call_two_segments.json"] == ["SPEAKER_00"]

    def test_blank_hf_token_yields_no_suggestions_but_still_lists_speakers(self, tmp_path):
        self._make_processed_file(tmp_path, "call_one", ["SPEAKER_00"])
        db_path = self._make_db(tmp_path)
        out_json = tmp_path / "roster.json"

        run_pipeline.export_speaker_roster_json(
            str(tmp_path), str(out_json), hf_token="", threshold=0.75, db_path=db_path)

        payload = json.loads(out_json.read_text(encoding="utf-8"))
        speaker = payload["files"][0]["speakers"][0]
        assert speaker["suggested_name"] is None
        assert speaker["name"] == ""
        assert "embedding_b64" not in speaker

    def test_file_with_no_speaker_labels_is_omitted(self, tmp_path):
        # Diarization was off for this run: segments exist but carry no
        # "speaker" key at all.
        stem = "no_diarization"
        segments = [{"start": 0.0, "end": 1.0, "text": "hi", "words": []}]
        (tmp_path / f"{stem}_segments.json").write_text(json.dumps(segments), encoding="utf-8")
        db_path = self._make_db(tmp_path)
        out_json = tmp_path / "roster.json"

        result = run_pipeline.export_speaker_roster_json(
            str(tmp_path), str(out_json), hf_token="", threshold=0.75, db_path=db_path)

        assert result["files"] == 0

    def test_not_a_folder_raises(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            run_pipeline.export_speaker_roster_json(
                str(tmp_path / "missing"), str(tmp_path / "out.json"),
                hf_token="", threshold=0.75, db_path=str(tmp_path / "x.db"))

    def test_includes_source_media_path(self, tmp_path):
        segments_json, src = self._make_processed_file(tmp_path, "call_one", ["SPEAKER_00"])
        db_path = self._make_db(tmp_path)
        out_json = tmp_path / "roster.json"

        run_pipeline.export_speaker_roster_json(
            str(tmp_path), str(out_json), hf_token="", threshold=0.75, db_path=db_path)

        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert payload["files"][0]["source_media"] == str(src)

    def test_extracts_sample_clip_per_speaker_when_ffmpeg_available(self, tmp_path, monkeypatch):
        # No real ffmpeg/audio in the test sandbox: stub check_ffmpeg and
        # extract_speaker_sample_clip so this covers the wiring (clip path
        # built, mkdir'd, entry/count updated) without needing a real
        # media file or binary.
        import speaker_id
        monkeypatch.setattr(speaker_id, "check_ffmpeg", lambda: True)

        def fake_extract(source_media_path, segments, speaker_label, out_path, max_duration=6.0):
            Path(out_path).write_bytes(b"fake-wav")
            return out_path
        monkeypatch.setattr(speaker_id, "extract_speaker_sample_clip", fake_extract)

        self._make_processed_file(tmp_path, "call_one", ["SPEAKER_00", "SPEAKER_01"])
        db_path = self._make_db(tmp_path)
        out_json = tmp_path / "roster.json"

        result = run_pipeline.export_speaker_roster_json(
            str(tmp_path), str(out_json), hf_token="", threshold=0.75, db_path=db_path)

        assert result["samples"] == 2
        samples_dir = Path(result["samples_dir"])
        assert samples_dir == tmp_path / "roster_samples"
        assert (samples_dir / "call_one__SPEAKER_00.wav").exists()
        assert (samples_dir / "call_one__SPEAKER_01.wav").exists()

        payload = json.loads(out_json.read_text(encoding="utf-8"))
        speakers = payload["files"][0]["speakers"]
        assert all(s["sample_clip"] for s in speakers)

    def test_sample_clip_is_null_when_ffmpeg_unavailable(self, tmp_path, monkeypatch):
        import speaker_id
        monkeypatch.setattr(speaker_id, "check_ffmpeg", lambda: False)

        self._make_processed_file(tmp_path, "call_one", ["SPEAKER_00"])
        db_path = self._make_db(tmp_path)
        out_json = tmp_path / "roster.json"

        result = run_pipeline.export_speaker_roster_json(
            str(tmp_path), str(out_json), hf_token="", threshold=0.75, db_path=db_path)

        assert result["samples"] == 0
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert payload["files"][0]["speakers"][0]["sample_clip"] is None

    def test_one_speakers_failed_clip_does_not_block_others(self, tmp_path, monkeypatch):
        import speaker_id
        monkeypatch.setattr(speaker_id, "check_ffmpeg", lambda: True)

        def flaky_extract(source_media_path, segments, speaker_label, out_path, max_duration=6.0):
            if speaker_label == "SPEAKER_00":
                raise RuntimeError("ffmpeg exploded")
            Path(out_path).write_bytes(b"fake-wav")
            return out_path
        monkeypatch.setattr(speaker_id, "extract_speaker_sample_clip", flaky_extract)

        self._make_processed_file(tmp_path, "call_one", ["SPEAKER_00", "SPEAKER_01"])
        db_path = self._make_db(tmp_path)
        out_json = tmp_path / "roster.json"

        result = run_pipeline.export_speaker_roster_json(
            str(tmp_path), str(out_json), hf_token="", threshold=0.75, db_path=db_path)

        assert result["files"] == 1
        assert result["samples"] == 1
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        speakers = {s["label"]: s["sample_clip"] for s in payload["files"][0]["speakers"]}
        assert speakers["SPEAKER_00"] is None
        assert speakers["SPEAKER_01"] is not None


class TestApplySpeakerRosterJson:
    def _make_db(self, tmp_path):
        path = str(tmp_path / "test.db")
        db.validate_schema(path)
        return path

    def _make_segments_file(self, tmp_path, stem, speakers):
        segments = [{"start": i * 1.0, "end": i * 1.0 + 0.5, "speaker": spk,
                     "text": "hello", "words": []}
                    for i, spk in enumerate(speakers)]
        segments_json = tmp_path / f"{stem}_segments.json"
        segments_json.write_text(json.dumps(segments), encoding="utf-8")
        media = tmp_path / f"{stem}.wav"
        media.write_bytes(b"fake")
        return segments_json

    def test_applies_rename_across_multiple_files(self, tmp_path):
        seg1 = self._make_segments_file(tmp_path, "call_one", ["SPEAKER_00"])
        seg2 = self._make_segments_file(tmp_path, "call_two", ["SPEAKER_00"])
        db_path = self._make_db(tmp_path)
        roster = {
            "db_path": db_path,
            "files": [
                {"segments_json": str(seg1), "speakers": [
                    {"label": "SPEAKER_00", "name": "Joerg Riener"}]},
                {"segments_json": str(seg2), "speakers": [
                    {"label": "SPEAKER_00", "name": "Massimo"}]},
            ],
        }
        roster_json = tmp_path / "roster.json"
        roster_json.write_text(json.dumps(roster), encoding="utf-8")

        result = run_pipeline.apply_speaker_roster_json(str(roster_json))

        assert result["files_updated"] == 2
        assert result["speakers_renamed"] == 2
        assert result["errors"] == []
        renamed_1 = json.loads(seg1.read_text(encoding="utf-8"))
        renamed_2 = json.loads(seg2.read_text(encoding="utf-8"))
        assert renamed_1[0]["speaker"] == "Joerg Riener"
        assert renamed_2[0]["speaker"] == "Massimo"

    def test_blank_name_skips_speaker(self, tmp_path):
        seg1 = self._make_segments_file(tmp_path, "call_one", ["SPEAKER_00"])
        db_path = self._make_db(tmp_path)
        roster = {
            "db_path": db_path,
            "files": [{"segments_json": str(seg1), "speakers": [
                {"label": "SPEAKER_00", "name": ""}]}],
        }
        roster_json = tmp_path / "roster.json"
        roster_json.write_text(json.dumps(roster), encoding="utf-8")

        result = run_pipeline.apply_speaker_roster_json(str(roster_json))

        assert result["files_updated"] == 0
        assert result["speakers_renamed"] == 0
        unchanged = json.loads(seg1.read_text(encoding="utf-8"))
        assert unchanged[0]["speaker"] == "SPEAKER_00"

    def test_name_equal_to_label_skips_speaker(self, tmp_path):
        seg1 = self._make_segments_file(tmp_path, "call_one", ["SPEAKER_00"])
        db_path = self._make_db(tmp_path)
        roster = {
            "db_path": db_path,
            "files": [{"segments_json": str(seg1), "speakers": [
                {"label": "SPEAKER_00", "name": "SPEAKER_00"}]}],
        }
        roster_json = tmp_path / "roster.json"
        roster_json.write_text(json.dumps(roster), encoding="utf-8")

        result = run_pipeline.apply_speaker_roster_json(str(roster_json))

        assert result["files_updated"] == 0

    def test_embedding_commits_to_roster(self, tmp_path):
        import numpy as np
        import speaker_id
        seg1 = self._make_segments_file(tmp_path, "call_one", ["SPEAKER_00"])
        db_path = self._make_db(tmp_path)
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb_b64 = base64.b64encode(speaker_id.serialize_embedding(vec)).decode("ascii")
        roster = {
            "db_path": db_path,
            "files": [{"segments_json": str(seg1), "speakers": [
                {"label": "SPEAKER_00", "name": "Anna",
                 "embedding_b64": emb_b64, "embedding_dim": 3}]}],
        }
        roster_json = tmp_path / "roster.json"
        roster_json.write_text(json.dumps(roster), encoding="utf-8")

        result = run_pipeline.apply_speaker_roster_json(str(roster_json))

        assert result["roster_updates"] == 1
        conn = db.get_connection(db_path)
        row = db.get_known_speaker_by_name(conn, "Anna")
        conn.close()
        assert row is not None

    def test_no_embedding_still_renames_but_no_roster_update(self, tmp_path):
        seg1 = self._make_segments_file(tmp_path, "call_one", ["SPEAKER_00"])
        db_path = self._make_db(tmp_path)
        roster = {
            "db_path": db_path,
            "files": [{"segments_json": str(seg1), "speakers": [
                {"label": "SPEAKER_00", "name": "Anna"}]}],
        }
        roster_json = tmp_path / "roster.json"
        roster_json.write_text(json.dumps(roster), encoding="utf-8")

        result = run_pipeline.apply_speaker_roster_json(str(roster_json))

        assert result["files_updated"] == 1
        assert result["roster_updates"] == 0

    def test_missing_segments_file_logs_error_and_continues(self, tmp_path):
        seg_good = self._make_segments_file(tmp_path, "call_good", ["SPEAKER_00"])
        db_path = self._make_db(tmp_path)
        roster = {
            "db_path": db_path,
            "files": [
                {"segments_json": str(tmp_path / "missing_segments.json"),
                 "speakers": [{"label": "SPEAKER_00", "name": "Ghost"}]},
                {"segments_json": str(seg_good),
                 "speakers": [{"label": "SPEAKER_00", "name": "RealPerson"}]},
            ],
        }
        roster_json = tmp_path / "roster.json"
        roster_json.write_text(json.dumps(roster), encoding="utf-8")

        result = run_pipeline.apply_speaker_roster_json(str(roster_json))

        assert result["files_updated"] == 1
        assert len(result["errors"]) == 1
        assert "missing_segments.json" in result["errors"][0]["file"]


# -----------------------------------------------------------------
# _resolve_ocr_language (Translate tab/CLI, image inputs)
# -----------------------------------------------------------------

class TestResolveOcrLanguage:
    def test_auto_resolves_to_eng_plus_deu(self):
        assert run_pipeline._resolve_ocr_language({"translate_ocr_language": "auto"}) == "eng+deu"

    def test_missing_key_defaults_to_auto_behaviour(self):
        assert run_pipeline._resolve_ocr_language({}) == "eng+deu"

    def test_blank_defaults_to_auto_behaviour(self):
        assert run_pipeline._resolve_ocr_language({"translate_ocr_language": ""}) == "eng+deu"

    def test_known_code_mapped_via_tesseract_lang_map(self):
        assert run_pipeline._resolve_ocr_language({"translate_ocr_language": "zh"}) == "chi_sim"
        assert run_pipeline._resolve_ocr_language({"translate_ocr_language": "de"}) == "deu"

    def test_raw_tesseract_string_passed_through(self):
        # Not a key in LANGUAGE_NAMES/TESSERACT_LANG_MAP - a power user
        # typing a raw Tesseract --lang string directly should reach
        # Tesseract unchanged.
        assert run_pipeline._resolve_ocr_language({"translate_ocr_language": "deu+fra"}) == "deu+fra"


# -----------------------------------------------------------------
# run_notes_batch_via_batch_api: Anthropic Message Batches API path.
# Only anthropic.Anthropic itself is faked (via monkeypatch.setattr, not
# a sys.modules stub) - the real, installed anthropic SDK still supplies
# MessageCreateParamsNonStreaming/Request, which are harmless local
# dataclass-like constructors with no network access.
# -----------------------------------------------------------------

class _FakeTextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeRequestCounts:
    def __init__(self, processing=0, succeeded=0, errored=0):
        self.processing = processing
        self.succeeded = succeeded
        self.errored = errored


class _FakeBatch:
    def __init__(self, id, processing_status):
        self.id = id
        self.processing_status = processing_status
        self.request_counts = _FakeRequestCounts()


class _FakeSucceededResult:
    def __init__(self, custom_id, text):
        self.custom_id = custom_id
        self.result = type("R", (), {
            "type": "succeeded",
            "message": type("M", (), {"content": [_FakeTextBlock(text)]})(),
        })()


class _FakeErroredResult:
    def __init__(self, custom_id):
        self.custom_id = custom_id
        self.result = type("R", (), {"type": "errored"})()


class _FakeBatchesAPI:
    """retrieve() always reports "ended" on first poll - these tests
    exercise the happy/error paths, not the actual polling loop timing."""
    def __init__(self, results):
        self._results = results
        self.created_requests = None
        self.canceled_id = None

    def create(self, requests):
        self.created_requests = requests
        return _FakeBatch("batch_test123", "in_progress")

    def retrieve(self, batch_id):
        return _FakeBatch(batch_id, "ended")

    def results(self, batch_id):
        return self._results

    def cancel(self, batch_id):
        self.canceled_id = batch_id
        return _FakeBatch(batch_id, "canceling")


def _install_fake_anthropic_client(monkeypatch, batches_api):
    fake_messages = type("Messages", (), {"batches": batches_api})()
    fake_client = type("Client", (), {"messages": fake_messages})()
    monkeypatch.setattr(anthropic, "Anthropic", lambda api_key: fake_client)
    return fake_client


class TestRunNotesBatchViaBatchAPI:
    def _make_transcript(self, tmp_path, name="call"):
        tf = tmp_path / f"{name}_transcript_speakers.txt"
        tf.write_text("[00:00:00] [SPEAKER_00] Hello world.", encoding="utf-8")
        return tf

    def _base_cfg_overrides(self, tmp_path, monkeypatch):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "meeting.md").write_text("## Summary\nWrite a summary.\n", encoding="utf-8")
        monkeypatch.setattr(run_pipeline, "PROMPTS_DIR", prompts_dir)

        db_path = tmp_path / "test.db"
        db.validate_schema(str(db_path))

        return {
            "llm_backend": "anthropic",
            "anthropic_api_key": "sk-ant-real-key",
            "claude_model": "claude-sonnet-5",
            "prompt_template": "meeting",
            "db_path": str(db_path),
            "notes_format_txt": True,
            "notes_format_html": False,
            "notes_format_docx": False,
            "notes_format_pdf": False,
        }

    def test_non_anthropic_backend_rejected(self, tmp_path, monkeypatch, caplog):
        self._make_transcript(tmp_path)
        overrides = self._base_cfg_overrides(tmp_path, monkeypatch)
        overrides["llm_backend"] = "ollama"
        batches_api = _FakeBatchesAPI(results=[])
        client = _install_fake_anthropic_client(monkeypatch, batches_api)

        run_pipeline.run_notes_batch_via_batch_api(str(tmp_path), overrides)

        assert batches_api.created_requests is None  # never even submitted

    def test_missing_api_key_rejected(self, tmp_path, monkeypatch):
        self._make_transcript(tmp_path)
        overrides = self._base_cfg_overrides(tmp_path, monkeypatch)
        overrides["anthropic_api_key"] = ""
        batches_api = _FakeBatchesAPI(results=[])
        _install_fake_anthropic_client(monkeypatch, batches_api)

        run_pipeline.run_notes_batch_via_batch_api(str(tmp_path), overrides)

        assert batches_api.created_requests is None

    def test_skips_files_that_already_have_notes(self, tmp_path, monkeypatch):
        tf = self._make_transcript(tmp_path)
        Path(str(tf).replace("_transcript_speakers.txt", "_notes.txt")).write_text(
            "already done", encoding="utf-8")
        overrides = self._base_cfg_overrides(tmp_path, monkeypatch)
        batches_api = _FakeBatchesAPI(results=[])
        _install_fake_anthropic_client(monkeypatch, batches_api)

        run_pipeline.run_notes_batch_via_batch_api(str(tmp_path), overrides)

        assert batches_api.created_requests is None  # nothing pending, never submitted

    def test_successful_batch_writes_notes_file(self, tmp_path, monkeypatch):
        tf = self._make_transcript(tmp_path)
        overrides = self._base_cfg_overrides(tmp_path, monkeypatch)

        # custom_id "item-1" for the first (only) pending file, matching
        # run_notes_batch_via_batch_api's own 1-based numbering scheme.
        batches_api = _FakeBatchesAPI(
            results=[_FakeSucceededResult("item-1", "## Summary\ngenerated notes body")])
        _install_fake_anthropic_client(monkeypatch, batches_api)

        run_pipeline.run_notes_batch_via_batch_api(str(tmp_path), overrides)

        assert batches_api.created_requests is not None
        assert len(batches_api.created_requests) == 1
        assert batches_api.created_requests[0]["custom_id"] == "item-1"

        notes_file = Path(str(tf).replace("_transcript_speakers.txt", "_notes.txt"))
        assert notes_file.exists()
        assert "generated notes body" in notes_file.read_text(encoding="utf-8")

    def test_errored_result_logged_not_written(self, tmp_path, monkeypatch):
        tf = self._make_transcript(tmp_path)
        overrides = self._base_cfg_overrides(tmp_path, monkeypatch)
        batches_api = _FakeBatchesAPI(results=[_FakeErroredResult("item-1")])
        _install_fake_anthropic_client(monkeypatch, batches_api)

        run_pipeline.run_notes_batch_via_batch_api(str(tmp_path), overrides)

        notes_file = Path(str(tf).replace("_transcript_speakers.txt", "_notes.txt"))
        assert not notes_file.exists()

    def test_multiple_files_get_distinct_custom_ids(self, tmp_path, monkeypatch):
        self._make_transcript(tmp_path, name="call_a")
        self._make_transcript(tmp_path, name="call_b")
        overrides = self._base_cfg_overrides(tmp_path, monkeypatch)
        batches_api = _FakeBatchesAPI(results=[
            _FakeSucceededResult("item-1", "## Summary\nnotes A"),
            _FakeSucceededResult("item-2", "## Summary\nnotes B"),
        ])
        _install_fake_anthropic_client(monkeypatch, batches_api)

        run_pipeline.run_notes_batch_via_batch_api(str(tmp_path), overrides)

        assert len(batches_api.created_requests) == 2
        ids = {r["custom_id"] for r in batches_api.created_requests}
        assert ids == {"item-1", "item-2"}


# -----------------------------------------------------------------
# _unique_snapshot_path
#
# Snapshots used to be re-encoded from the extracted JPEG frame into PNG,
# purely so the filename could end in ".png". The frame is now copied
# unchanged, so the final snapshot name has to carry whatever extension the
# staged frame actually has (".jpg" with config's default frame_format).
# -----------------------------------------------------------------

class TestUniqueSnapshotPath:
    def test_defaults_to_png_for_older_callers(self, tmp_path):
        got = run_pipeline._unique_snapshot_path(tmp_path, "talk", 1)
        assert got.name == "talk_slide001.png"

    def test_honours_an_explicit_suffix(self, tmp_path):
        got = run_pipeline._unique_snapshot_path(tmp_path, "talk", 7, suffix=".jpg")
        assert got.name == "talk_slide007.jpg"

    def test_slide_number_is_zero_padded_to_three_digits(self, tmp_path):
        got = run_pipeline._unique_snapshot_path(tmp_path, "talk", 42, suffix=".jpg")
        assert got.name == "talk_slide042.jpg"

    def test_collision_gets_a_numeric_suffix_keeping_the_extension(self, tmp_path):
        (tmp_path / "talk_slide001.jpg").write_bytes(b"first")
        got = run_pipeline._unique_snapshot_path(tmp_path, "talk", 1, suffix=".jpg")
        assert got.name == "talk_slide001_2.jpg"

    def test_repeated_collisions_keep_counting(self, tmp_path):
        (tmp_path / "talk_slide001.jpg").write_bytes(b"a")
        (tmp_path / "talk_slide001_2.jpg").write_bytes(b"b")
        got = run_pipeline._unique_snapshot_path(tmp_path, "talk", 1, suffix=".jpg")
        assert got.name == "talk_slide001_3.jpg"

    def test_blank_suffix_falls_back_to_png(self, tmp_path):
        got = run_pipeline._unique_snapshot_path(tmp_path, "talk", 1, suffix="")
        assert got.name == "talk_slide001.png"

    def test_never_returns_an_existing_path(self, tmp_path):
        (tmp_path / "talk_slide001.jpg").write_bytes(b"a")
        got = run_pipeline._unique_snapshot_path(tmp_path, "talk", 1, suffix=".jpg")
        assert not got.exists()


# -----------------------------------------------------------------
# Batch runs and the transcriber model cache
#
# The runners enable transcriber's model cache so a batch loads the whisper,
# alignment and diarization weights once instead of once per file, then turn
# it off again afterwards. Turning it off matters as much as turning it on:
# releasing without disabling used to leave the cache enabled for the rest
# of the process, so later single-file runs kept their models resident with
# nothing left to release them.
# -----------------------------------------------------------------

class TestBatchModelCache:
    def _patch_db(self, monkeypatch):
        monkeypatch.setattr(db, "validate_schema", lambda path: True)
        monkeypatch.setattr(db, "get_connection", lambda path: FakeConn())
        monkeypatch.setattr(db, "get_completed_transcript", lambda conn, file_path: None)

    def _patch_run(self, monkeypatch, seen):
        """process_file records whether the cache was enabled while it ran."""
        import transcriber

        def fake_process_file(file, overrides, stop_check=None):
            seen.append(transcriber._CACHE_ENABLED)
            return True

        monkeypatch.setattr(run_pipeline, "process_file", fake_process_file)
        monkeypatch.setattr(run_pipeline, "write_log", lambda *a, **kw: None)

    def test_cache_is_enabled_while_the_batch_runs(self, monkeypatch, tmp_path):
        import transcriber
        self._patch_db(monkeypatch)
        seen = []
        self._patch_run(monkeypatch, seen)
        rows = [{"file": str(tmp_path / "a.mp4")}, {"file": str(tmp_path / "b.mp4")}]
        run_pipeline.run_batch_rows(rows, {"db_path": str(tmp_path / "x.db")})
        assert seen == [True, True]

    def test_cache_is_disabled_again_afterwards(self, monkeypatch, tmp_path):
        import transcriber
        self._patch_db(monkeypatch)
        self._patch_run(monkeypatch, [])
        run_pipeline.run_batch_rows([{"file": str(tmp_path / "a.mp4")}],
                                    {"db_path": str(tmp_path / "x.db")})
        assert transcriber._CACHE_ENABLED is False
        assert transcriber._ASR_CACHE == {}

    def test_release_models_alone_does_not_disable_the_cache(self, monkeypatch):
        """Documents the split: release_models() frees memory, and
        enable_model_cache(False) is what ends the caching window. The
        runners deliberately call the latter."""
        import transcriber
        monkeypatch.setattr(transcriber, "_CACHE_ENABLED", True)
        transcriber.release_models()
        assert transcriber._CACHE_ENABLED is True
        transcriber.enable_model_cache(False)
        assert transcriber._CACHE_ENABLED is False
