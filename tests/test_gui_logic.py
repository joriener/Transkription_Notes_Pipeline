# =============================================================
#  Transkription_Notes_Pipeline - tests/test_gui_logic.py
#  Unit tests for gui_logic.py's pure-logic helpers (task #70).
#  No tkinter dependency: these run in any plain Python 3.11 env,
#  including the CI sandbox with no torch/whisperx installed.
#
#  Run: pytest tests/test_gui_logic.py -v
# =============================================================

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gui_logic


# -----------------------------------------------------------------
# parse_time_to_seconds
# -----------------------------------------------------------------

class TestParseTimeToSeconds:
    def test_blank_returns_none(self):
        assert gui_logic.parse_time_to_seconds("") is None
        assert gui_logic.parse_time_to_seconds("   ") is None
        assert gui_logic.parse_time_to_seconds(None) is None

    def test_plain_seconds(self):
        assert gui_logic.parse_time_to_seconds("975") == 975.0
        assert gui_logic.parse_time_to_seconds("12.5") == 12.5

    def test_mm_ss(self):
        assert gui_logic.parse_time_to_seconds("16:15") == 16 * 60 + 15
        assert gui_logic.parse_time_to_seconds("00:00") == 0

    def test_hh_mm_ss(self):
        assert gui_logic.parse_time_to_seconds("1:02:03") == 3600 + 2 * 60 + 3

    def test_mm_ss_with_fractional_seconds(self):
        assert gui_logic.parse_time_to_seconds("1:02.5") == 62.5

    def test_too_many_colons_raises(self):
        with pytest.raises(ValueError):
            gui_logic.parse_time_to_seconds("1:02:03:04")

    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            gui_logic.parse_time_to_seconds("not a time")


class TestSafeParseTime:
    def test_valid_passes_through(self):
        assert gui_logic._safe_parse_time("16:15") == 975.0

    def test_invalid_returns_none_instead_of_raising(self):
        assert gui_logic._safe_parse_time("garbage") is None

    def test_blank_returns_none(self):
        assert gui_logic._safe_parse_time("") is None


# -----------------------------------------------------------------
# build_batch_row
# -----------------------------------------------------------------

MEETING_COLS = ("file", "language", "title", "date", "comments")
VIDEO_COLS = ("file", "language", "title", "date", "comments", "qa_start", "qa_end")


class TestBuildBatchRow:
    def test_blank_file_returns_none(self):
        assert gui_logic.build_batch_row(MEETING_COLS, ("", "en", "T", "2026-01-01", "c")) is None
        assert gui_logic.build_batch_row(MEETING_COLS, ("   ", "", "", "", "")) is None

    def test_meeting_row_full(self):
        row = gui_logic.build_batch_row(
            MEETING_COLS, ("a.mp4", "en", "Kickoff", "2026-07-01", "notes"))
        assert row == {
            "file": "a.mp4",
            "language": "en",
            "meeting_title": "Kickoff",
            "meeting_date": "2026-07-01",
            "meeting_comments": "notes",
        }
        assert "qa_start_time_sec" not in row

    def test_meeting_row_blank_optionals_become_none(self):
        row = gui_logic.build_batch_row(MEETING_COLS, ("a.mp4", "", "", "", ""))
        assert row["file"] == "a.mp4"
        assert row["language"] is None
        assert row["meeting_title"] is None
        assert row["meeting_date"] is None
        assert row["meeting_comments"] is None

    def test_video_row_adds_qa_keys(self):
        row = gui_logic.build_batch_row(
            VIDEO_COLS, ("a.mp4", "", "", "", "", "16:15", "20:00"))
        assert row["qa_start_time_sec"] == 975.0
        assert row["qa_end_time_sec"] == 1200.0

    def test_video_row_blank_qa_is_none(self):
        row = gui_logic.build_batch_row(VIDEO_COLS, ("a.mp4", "", "", "", "", "", ""))
        assert row["qa_start_time_sec"] is None
        assert row["qa_end_time_sec"] is None

    def test_video_row_bad_qa_value_falls_back_to_none(self):
        """build_batch_row never raises: an unparseable Q&A cell becomes
        None rather than aborting the whole batch. The GUI validates Q&A
        text explicitly (with a clear warning) before a run starts."""
        row = gui_logic.build_batch_row(VIDEO_COLS, ("a.mp4", "", "", "", "", "not-a-time", ""))
        assert row["qa_start_time_sec"] is None

    def test_short_values_tuple_is_padded(self):
        """Treeview can hand back fewer values than columns (trailing
        blank cells are sometimes omitted); build_batch_row must not
        raise/index-error on that."""
        row = gui_logic.build_batch_row(VIDEO_COLS, ("a.mp4", "en"))
        assert row["file"] == "a.mp4"
        assert row["language"] == "en"
        assert row["qa_start_time_sec"] is None
        assert row["qa_end_time_sec"] is None


# -----------------------------------------------------------------
# parse_list_line
# -----------------------------------------------------------------

class TestParseListLine:
    def test_blank_line_is_none(self):
        assert gui_logic.parse_list_line("") is None
        assert gui_logic.parse_list_line("   \n") is None

    def test_comment_line_is_none(self):
        assert gui_logic.parse_list_line("# a comment") is None

    def test_file_only(self):
        assert gui_logic.parse_list_line("C:\\videos\\a.mp4") == {
            "file": "C:\\videos\\a.mp4", "language": ""}

    def test_file_and_language(self):
        assert gui_logic.parse_list_line("C:\\videos\\a.mp4|en") == {
            "file": "C:\\videos\\a.mp4", "language": "en"}

    def test_blank_file_before_pipe_is_none(self):
        assert gui_logic.parse_list_line("|en") is None

    def test_strips_whitespace(self):
        assert gui_logic.parse_list_line("  a.mp4 | en  \n") == {"file": "a.mp4", "language": "en"}


# -----------------------------------------------------------------
# csv_header_index
# -----------------------------------------------------------------

class TestPersistedKeysFor:
    def test_video_includes_common_and_video_only(self):
        keys = gui_logic.persisted_keys_for("video")
        assert "whisper_model" in keys
        assert "enable_slides" in keys
        assert "hash_threshold" in keys

    def test_meeting_excludes_video_only(self):
        keys = gui_logic.persisted_keys_for("meeting")
        assert "whisper_model" in keys
        assert "enable_slides" not in keys
        assert "hash_threshold" not in keys


class TestBuildStateDict:
    def test_drops_unpersisted_keys(self):
        raw = {"whisper_model": "large-v3", "path_var": "/tmp/a.mp4", "enable_slides": True}
        out = gui_logic.build_state_dict("meeting", raw)
        assert out == {"whisper_model": "large-v3"}

    def test_video_keeps_video_only_keys(self):
        raw = {"whisper_model": "large-v3", "enable_slides": True, "fps": 3}
        out = gui_logic.build_state_dict("video", raw)
        assert out == {"whisper_model": "large-v3", "enable_slides": True, "fps": 3}

    def test_missing_keys_are_skipped_not_defaulted(self):
        out = gui_logic.build_state_dict("meeting", {"whisper_model": "base"})
        assert out == {"whisper_model": "base"}


class TestMergePersistedState:
    def test_saved_overrides_defaults(self):
        defaults = {"whisper_model": "base", "language": "auto"}
        saved = {"whisper_model": "large-v3"}
        merged = gui_logic.merge_persisted_state("meeting", saved, defaults)
        assert merged == {"whisper_model": "large-v3", "language": "auto"}

    def test_unrecognized_saved_key_ignored(self):
        defaults = {"whisper_model": "base"}
        saved = {"whisper_model": "large-v3", "some_future_key": "x"}
        merged = gui_logic.merge_persisted_state("meeting", saved, defaults)
        assert "some_future_key" not in merged

    def test_video_only_key_ignored_for_meeting_kind(self):
        """A gui_state.json that somehow has a video-only key under
        "meeting" (e.g. hand-edited, or a downgrade from a future
        version) must not leak into the meeting tab's settings."""
        defaults = {"whisper_model": "base"}
        saved = {"whisper_model": "large-v3", "enable_slides": True}
        merged = gui_logic.merge_persisted_state("meeting", saved, defaults)
        assert "enable_slides" not in merged

    def test_empty_saved_returns_defaults_unchanged(self):
        defaults = {"whisper_model": "base", "language": "auto"}
        assert gui_logic.merge_persisted_state("meeting", {}, defaults) == defaults
        assert gui_logic.merge_persisted_state("meeting", None, defaults) == defaults


class TestLoadSaveStateFile:
    def test_load_missing_file_returns_empty_dict(self, tmp_path):
        assert gui_logic.load_state_file(tmp_path / "nope.json") == {}

    def test_load_corrupt_json_returns_empty_dict(self, tmp_path):
        p = tmp_path / "gui_state.json"
        p.write_text("not valid json{{{", encoding="utf-8")
        assert gui_logic.load_state_file(p) == {}

    def test_load_non_object_json_returns_empty_dict(self, tmp_path):
        p = tmp_path / "gui_state.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")
        assert gui_logic.load_state_file(p) == {}

    def test_save_then_load_round_trip(self, tmp_path):
        p = tmp_path / "gui_state.json"
        state = {"meeting": {"whisper_model": "large-v3"}, "video": {"fps": 3}}
        assert gui_logic.save_state_file(p, state) is True
        assert gui_logic.load_state_file(p) == state

    def test_save_to_unwritable_path_returns_false_not_raises(self, tmp_path):
        # A directory used as the target file path can't be opened for
        # writing - save_state_file must swallow the OSError.
        bad_path = tmp_path / "a_directory"
        bad_path.mkdir()
        assert gui_logic.save_state_file(bad_path, {"meeting": {}}) is False


class TestCsvHeaderIndex:
    def test_recognized_header(self):
        idx = gui_logic.csv_header_index(
            ["File", "Language", "Title"], {"file", "language", "title", "date", "comments"})
        assert idx == {"file": 0, "language": 1, "title": 2}

    def test_no_recognized_columns_returns_empty(self):
        idx = gui_logic.csv_header_index(
            ["foo", "bar"], {"file", "language", "title", "date", "comments"})
        assert idx == {}

    def test_case_insensitive(self):
        idx = gui_logic.csv_header_index(["FILE", "LANGUAGE"], {"file", "language"})
        assert idx == {"file": 0, "language": 1}

    def test_partial_match_still_returns_present_columns_only(self):
        idx = gui_logic.csv_header_index(
            ["file", "comments"], {"file", "language", "title", "date", "comments"})
        assert idx == {"file": 0, "comments": 1}
