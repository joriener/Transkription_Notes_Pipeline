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
VIDEO_COLS_WITH_IMAGE = VIDEO_COLS + ("title_image",)
VIDEO_COLS_WITH_SRT = VIDEO_COLS_WITH_IMAGE + ("srt_path",)


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

    def test_meeting_cols_have_no_title_image_key(self):
        """Meeting tab's batch table has no title_image column at all -
        the key must not appear in the output dict, matching qa_start/
        qa_end's same "not in row_override_keys" absence."""
        row = gui_logic.build_batch_row(MEETING_COLS, ("a.mp4", "", "", "", ""))
        assert "title_slide_image_path" not in row

    def test_video_row_title_image_set(self):
        row = gui_logic.build_batch_row(
            VIDEO_COLS_WITH_IMAGE,
            ("a.mp4", "", "", "", "", "", "", "cover.png"))
        assert row["title_slide_image_path"] == "cover.png"

    def test_video_row_title_image_blank_is_none(self):
        """A blank per-row cell means None - run_batch_rows then falls
        back to the shared Video/Webinar tab field for that file."""
        row = gui_logic.build_batch_row(
            VIDEO_COLS_WITH_IMAGE,
            ("a.mp4", "", "", "", "", "", "", ""))
        assert row["title_slide_image_path"] is None

    def test_meeting_cols_have_no_srt_import_key(self):
        """Meeting tab's batch table has no srt_path column at all -
        batch rows there should never carry an srt_import_path key."""
        row = gui_logic.build_batch_row(MEETING_COLS, ("a.mp4", "", "", "", ""))
        assert "srt_import_path" not in row

    def test_video_row_srt_path_set(self):
        row = gui_logic.build_batch_row(
            VIDEO_COLS_WITH_SRT,
            ("a.mp4", "", "", "", "", "", "", "", "C:/vids/a.srt"))
        assert row["srt_import_path"] == "C:/vids/a.srt"

    def test_video_row_srt_path_blank_is_none(self):
        row = gui_logic.build_batch_row(
            VIDEO_COLS_WITH_SRT,
            ("a.mp4", "", "", "", "", "", "", "", ""))
        assert row["srt_import_path"] is None


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


class TestBuildTabStateDict:
    """Tab-owned keys only: recording_type/prompt_template/output_dir,
    plus video-only keys for kind="video". Shared engine settings
    (whisper_model, llm_backend, ...) are deliberately excluded here -
    see TestBuildSharedStateDict."""

    def test_drops_unpersisted_and_shared_keys(self):
        raw = {"output_dir": "/tmp/out", "whisper_model": "large-v3",
               "path_var": "/tmp/a.mp4", "enable_slides": True}
        out = gui_logic.build_tab_state_dict("meeting", raw)
        assert out == {"output_dir": "/tmp/out"}

    def test_video_keeps_video_only_keys(self):
        raw = {"output_dir": "/tmp/out", "whisper_model": "large-v3",
               "enable_slides": True, "fps": 3}
        out = gui_logic.build_tab_state_dict("video", raw)
        assert out == {"output_dir": "/tmp/out", "enable_slides": True, "fps": 3}

    def test_missing_keys_are_skipped_not_defaulted(self):
        out = gui_logic.build_tab_state_dict("meeting", {"output_dir": "/tmp/out"})
        assert out == {"output_dir": "/tmp/out"}


class TestBuildSharedStateDict:
    def test_keeps_only_shared_keys(self):
        raw = {
            "whisper_model": "large-v3", "language": "de", "llm_backend": "ollama",
            "enable_diarization": True, "no_summary": False, "force_retranscribe": False,
            "prompt_template": "webinar", "recording_type": "Webinar Transcript",
            "enable_slides": True, "output_dir": "/tmp/out",
        }
        out = gui_logic.build_shared_state_dict(raw)
        assert out == {
            "whisper_model": "large-v3", "language": "de", "llm_backend": "ollama",
            "enable_diarization": True, "no_summary": False, "force_retranscribe": False,
        }

    def test_missing_keys_are_skipped(self):
        out = gui_logic.build_shared_state_dict({"whisper_model": "base"})
        assert out == {"whisper_model": "base"}

    def test_empty_input_returns_empty(self):
        assert gui_logic.build_shared_state_dict({}) == {}


class TestMergePersistedState:
    def test_saved_tab_overrides_defaults(self):
        defaults = {"output_dir": "", "language": "auto"}
        saved_tab = {"output_dir": "/tmp/out"}
        merged = gui_logic.merge_persisted_state("meeting", saved_tab, {}, defaults)
        assert merged == {"output_dir": "/tmp/out", "language": "auto"}

    def test_saved_shared_overrides_defaults(self):
        defaults = {"whisper_model": "base", "output_dir": ""}
        saved_shared = {"whisper_model": "large-v3"}
        merged = gui_logic.merge_persisted_state("meeting", {}, saved_shared, defaults)
        assert merged == {"whisper_model": "large-v3", "output_dir": ""}

    def test_unrecognized_saved_key_ignored(self):
        defaults = {"whisper_model": "base"}
        saved_shared = {"whisper_model": "large-v3", "some_future_key": "x"}
        merged = gui_logic.merge_persisted_state("meeting", {}, saved_shared, defaults)
        assert "some_future_key" not in merged

    def test_video_only_key_ignored_for_meeting_kind(self):
        """A gui_state.json that somehow has a video-only key under
        "meeting" (e.g. hand-edited, or a downgrade from a future
        version) must not leak into the meeting tab's settings."""
        defaults = {"output_dir": ""}
        saved_tab = {"output_dir": "/tmp/out", "enable_slides": True}
        merged = gui_logic.merge_persisted_state("meeting", saved_tab, {}, defaults)
        assert "enable_slides" not in merged

    def test_shared_key_migrated_from_tab_section_when_shared_empty(self):
        """A pre-centralization gui_state.json has no "shared" section at
        all - PERSISTED_KEYS_SHARED still live under each tab's own
        section back then. The first load after updating must carry that
        value over instead of silently resetting it to CONFIG defaults."""
        defaults = {"whisper_model": "base"}
        saved_tab = {"whisper_model": "large-v3"}
        merged = gui_logic.merge_persisted_state("meeting", saved_tab, {}, defaults)
        assert merged["whisper_model"] == "large-v3"

    def test_shared_key_in_saved_shared_wins_over_tab_section(self):
        """Once state["shared"] actually holds a key, that value wins
        over any (now-stale) copy still sitting in the tab's own
        section - the migration fallback only fires when saved_shared
        has no entry for the key at all."""
        defaults = {"whisper_model": "base"}
        saved_tab = {"whisper_model": "large-v2"}
        saved_shared = {"whisper_model": "large-v3"}
        merged = gui_logic.merge_persisted_state("meeting", saved_tab, saved_shared, defaults)
        assert merged["whisper_model"] == "large-v3"

    def test_empty_saved_returns_defaults_unchanged(self):
        defaults = {"whisper_model": "base", "language": "auto"}
        assert gui_logic.merge_persisted_state("meeting", {}, {}, defaults) == defaults
        assert gui_logic.merge_persisted_state("meeting", None, None, defaults) == defaults


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


class TestSanitizeTemplateName:
    def test_blank_returns_none(self):
        assert gui_logic.sanitize_template_name("") is None
        assert gui_logic.sanitize_template_name("   ") is None
        assert gui_logic.sanitize_template_name(None) is None

    def test_adds_md_extension(self):
        assert gui_logic.sanitize_template_name("standup") == "standup.md"

    def test_keeps_existing_md_extension(self):
        assert gui_logic.sanitize_template_name("standup.md") == "standup.md"

    def test_strips_whitespace(self):
        assert gui_logic.sanitize_template_name("  standup  ") == "standup.md"

    def test_rejects_forward_slash(self):
        assert gui_logic.sanitize_template_name("sub/standup") is None

    def test_rejects_backslash(self):
        assert gui_logic.sanitize_template_name("sub\\standup") is None

    def test_rejects_parent_dir_traversal(self):
        assert gui_logic.sanitize_template_name("../../etc/passwd") is None
        assert gui_logic.sanitize_template_name("..") is None

    def test_rejects_readme_case_insensitive(self):
        assert gui_logic.sanitize_template_name("readme") is None
        assert gui_logic.sanitize_template_name("README.md") is None
        assert gui_logic.sanitize_template_name("ReadMe") is None

    def test_case_preserved_for_non_readme_names(self):
        assert gui_logic.sanitize_template_name("Standup") == "Standup.md"
