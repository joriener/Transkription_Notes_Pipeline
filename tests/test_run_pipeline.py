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
