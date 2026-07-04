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
