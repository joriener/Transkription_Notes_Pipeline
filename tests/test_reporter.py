# =============================================================
#  Transkription_Notes_Pipeline - tests/test_reporter.py
#  Unit tests for reporter.py: the pure text/markdown helpers, the
#  slide exporters (CSV/JSON/HTML/print-HTML/timing summary), the
#  notes writers, and save_pdf_from_html's backend fallback chain.
#
#  No dependency stubbing needed for most of this: reporter.py imports
#  only stdlib at module level and defers docx/numpy/playwright/
#  weasyprint/pdfkit into the function that needs them. The PDF chain is
#  covered by monkeypatching reporter's own private backends, so no
#  browser or GTK runtime is required. Snapshot image files never have to
#  exist, because the reports emit a relative snapshots/<name> img tag.
#
#  Generation timestamps (datetime.now) are deliberately never asserted
#  on - only substrings around them.
#
#  Run: C:\Python\Python311\python.exe -m pytest tests/test_reporter.py -v
# =============================================================

import json

import pytest

import reporter


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def make_slide(timestamp_sec=10.0, title="Calibration", bullets=None,
               transcript_seg="We calibrate with an alkane ladder. Then we run it.",
               snapshot_path="C:\\out\\_snap_tmp\\frame_000012.jpg", **over):
    slide = {
        "timestamp_sec": timestamp_sec,
        "title": title,
        "bullets": ["Use C7-C30 alkanes", "Check the column"] if bullets is None else bullets,
        "slide_type": "content",
        "transcript_seg": transcript_seg,
        "snapshot_path": snapshot_path,
        "speaker": "SPEAKER_01",
    }
    slide.update(over)
    return slide


def make_qa_slide(timestamp_sec=600.0, qa_pairs=None):
    """The synthetic Q&A card appended by run_pipeline._add_qa_slide."""
    return {
        "timestamp_sec": timestamp_sec,
        "title": "Q&A Session",
        "bullets": [],
        "slide_type": "qa_session",
        "transcript_seg": "Any questions?",
        "snapshot_path": "",
        "qa_pairs": qa_pairs if qa_pairs is not None else [
            {"question": "Which column?", "answer": "A DB-5ms."},
        ],
    }


# -----------------------------------------------------------------
# format_ts
# -----------------------------------------------------------------

class TestFormatTs:
    def test_zero(self):
        assert reporter.format_ts(0) == "0:00:00"

    def test_truncates_fractional_seconds(self):
        assert reporter.format_ts(90.9) == "0:01:30"

    def test_hours_are_not_zero_padded(self):
        assert reporter.format_ts(3661) == "1:01:01"


# -----------------------------------------------------------------
# _first_sentence
# -----------------------------------------------------------------

class TestFirstSentence:
    def test_splits_on_full_stop(self):
        assert reporter._first_sentence("One. Two. Three.") == "One."

    def test_splits_on_question_and_exclamation(self):
        assert reporter._first_sentence("Really? Yes.") == "Really?"
        assert reporter._first_sentence("Wow! Indeed.") == "Wow!"

    def test_no_boundary_returns_whole_string(self):
        assert reporter._first_sentence("no terminator here") == "no terminator here"

    def test_blank_and_none(self):
        assert reporter._first_sentence("") == ""
        assert reporter._first_sentence(None) == ""

    def test_decimal_number_is_not_a_boundary(self):
        """The split needs whitespace after the period, so "1.5 mL" stays
        in one piece."""
        assert reporter._first_sentence("Inject 1.5 mL now. Then wait.") == "Inject 1.5 mL now."


# -----------------------------------------------------------------
# _parse_notes_sections
# -----------------------------------------------------------------

class TestParseNotesSections:
    def test_splits_on_heading_and_classifies_items(self):
        md = ('## SUMMARY\n'
              'A plain line.\n'
              '- a bullet\n'
              '"a quoted line"\n'
              '## NEXT STEPS\n'
              '- another bullet\n')
        sections = reporter._parse_notes_sections(md)
        assert [s["title"] for s in sections] == ["SUMMARY", "NEXT STEPS"]
        kinds = [i["type"] for i in sections[0]["items"]]
        assert kinds == ["text", "bullet", "quote"]
        assert sections[0]["items"][1]["text"] == "a bullet"

    def test_content_before_the_first_heading_is_dropped(self):
        sections = reporter._parse_notes_sections("stray preamble\n## ONLY\n- x\n")
        assert [s["title"] for s in sections] == ["ONLY"]

    def test_no_headings_gives_no_sections(self):
        assert reporter._parse_notes_sections("just text\n- and a bullet\n") == []

    def test_qa_prefix_is_normalized(self):
        sections = reporter._parse_notes_sections("## QA\n- Q:no space\n")
        assert sections[0]["items"][0]["text"] == "Q: no space"


# -----------------------------------------------------------------
# _markdown_to_html
# -----------------------------------------------------------------

class TestMarkdownToHtml:
    def test_heading_bullets_and_quote(self):
        out = reporter._markdown_to_html('## SUMMARY\n- one\n- two\n\n"quoted"\n')
        assert "<h2>SUMMARY</h2>" in out
        assert out.count("<li>") == 2
        assert "<blockquote>&quot;quoted&quot;</blockquote>" in out or \
               '<blockquote>"quoted"</blockquote>' in out

    def test_list_is_closed_at_end_of_input(self):
        out = reporter._markdown_to_html("## S\n- only item")
        assert out.count("<ul>") == 1
        assert out.count("</ul>") == 1

    def test_plain_line_becomes_paragraph(self):
        assert "<p>hello</p>" in reporter._markdown_to_html("hello")

    def test_question_answer_prefix_is_bolded(self):
        out = reporter._markdown_to_html("- Q: which column?")
        assert "<strong>Q:</strong>" in out


# -----------------------------------------------------------------
# save_slide_timing_summary  (fully deterministic, no datetime.now)
# -----------------------------------------------------------------

class TestSaveSlideTimingSummary:
    def test_numbers_slides_and_counts_them(self, tmp_path):
        out = tmp_path / "timing.txt"
        reporter.save_slide_timing_summary(
            [make_slide(timestamp_sec=10.0, title="First"),
             make_slide(timestamp_sec=70.0, title="Second")],
            out, video_name="talk.mp4")
        text = out.read_text(encoding="utf-8")
        assert "Slides detected: 2" in text
        assert "Slide 001  0:00:10  First" in text
        assert "Slide 002  0:01:10  Second" in text

    def test_qa_card_is_labelled_and_excluded_from_the_count(self, tmp_path):
        out = tmp_path / "timing.txt"
        reporter.save_slide_timing_summary(
            [make_slide(timestamp_sec=10.0, title="First"), make_qa_slide(timestamp_sec=600.0)],
            out, video_name="talk.mp4")
        text = out.read_text(encoding="utf-8")
        assert "Slides detected: 1" in text     # the Q&A card is not a slide
        assert "Q&A" in text
        assert "Slide 002" not in text          # numbering never reaches the Q&A card

    def test_optional_meeting_info_is_included_only_when_set(self, tmp_path):
        out = tmp_path / "timing.txt"
        reporter.save_slide_timing_summary([make_slide()], out, video_name="talk.mp4")
        assert "Meeting:" not in out.read_text(encoding="utf-8")

        reporter.save_slide_timing_summary(
            [make_slide()], out, video_name="talk.mp4",
            meeting_title="Quarterly review", meeting_date="2026-09-09",
            meeting_comments="Bring the quote.")
        text = out.read_text(encoding="utf-8")
        assert "Meeting: Quarterly review" in text
        assert "Date: 2026-09-09" in text
        assert "Comments: Bring the quote." in text

    def test_recording_speed_is_noted_only_when_not_realtime(self, tmp_path):
        out = tmp_path / "timing.txt"
        reporter.save_slide_timing_summary([make_slide()], out, recording_speed=1.0)
        assert "Recording speed" not in out.read_text(encoding="utf-8")

        reporter.save_slide_timing_summary([make_slide()], out, recording_speed=1.5)
        assert "Recording speed: 1.5x" in out.read_text(encoding="utf-8")

    def test_creates_missing_parent_directory(self, tmp_path):
        out = tmp_path / "nested" / "deeper" / "timing.txt"
        reporter.save_slide_timing_summary([make_slide()], out)
        assert out.is_file()

    def test_empty_slide_list(self, tmp_path):
        out = tmp_path / "timing.txt"
        reporter.save_slide_timing_summary([], out, video_name="talk.mp4")
        assert "Slides detected: 0" in out.read_text(encoding="utf-8")


# -----------------------------------------------------------------
# save_csv
# -----------------------------------------------------------------

class TestSaveCsv:
    def test_header_and_row_values(self, tmp_path):
        out = tmp_path / "slides.csv"
        reporter.save_csv([make_slide(timestamp_sec=125.0, title="Calibration")], out)
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("slide_id,timestamp,timestamp_sec,title")
        assert "0:02:05" in lines[1]        # formatted alongside the raw seconds
        assert "Calibration" in lines[1]

    def test_slide_id_is_one_based(self, tmp_path):
        out = tmp_path / "slides.csv"
        reporter.save_csv([make_slide(), make_slide(timestamp_sec=20.0)], out)
        rows = out.read_text(encoding="utf-8").splitlines()[1:]
        assert rows[0].startswith("1,")
        assert rows[1].startswith("2,")

    def test_bullet_list_is_pipe_joined(self, tmp_path):
        out = tmp_path / "slides.csv"
        reporter.save_csv([make_slide(bullets=["one", "two", "three"])], out)
        assert "one | two | three" in out.read_text(encoding="utf-8")

    def test_non_list_bullets_are_coerced(self, tmp_path):
        """Defensive path: a weaker VLM can return bullets as a bare string
        or nested objects, which must not crash the export."""
        out = tmp_path / "slides.csv"
        reporter.save_csv([make_slide(bullets="already a string")], out)
        assert "already a string" in out.read_text(encoding="utf-8")
        reporter.save_csv([make_slide(bullets=[{"a": 1}, 2])], out)
        assert out.is_file()

    def test_unknown_slide_keys_are_ignored(self, tmp_path):
        out = tmp_path / "slides.csv"
        reporter.save_csv([make_slide(unexpected_key="ignore me")], out)
        assert "ignore me" not in out.read_text(encoding="utf-8")

    def test_creates_missing_parent_directory(self, tmp_path):
        out = tmp_path / "nested" / "slides.csv"
        reporter.save_csv([make_slide()], out)
        assert out.is_file()


# -----------------------------------------------------------------
# save_json
# -----------------------------------------------------------------

class TestSaveJson:
    def test_round_trips_the_slide_list(self, tmp_path):
        out = tmp_path / "slides.json"
        reporter.save_json([make_slide(title="Calibration")], out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data) == 1
        assert data[0]["title"] == "Calibration"
        assert data[0]["bullets"] == ["Use C7-C30 alkanes", "Check the column"]

    def test_umlauts_are_not_escaped(self, tmp_path):
        out = tmp_path / "slides.json"
        reporter.save_json([make_slide(title="Säulenwechsel")], out)
        assert "Säulenwechsel" in out.read_text(encoding="utf-8")

    def test_creates_missing_parent_directory(self, tmp_path):
        out = tmp_path / "nested" / "slides.json"
        reporter.save_json([make_slide()], out)
        assert out.is_file()


# -----------------------------------------------------------------
# save_html
# -----------------------------------------------------------------

class TestSaveHtml:
    def test_writes_a_self_contained_page(self, tmp_path):
        out = tmp_path / "report.html"
        reporter.save_html([make_slide()], out, video_name="talk.mp4")
        html = out.read_text(encoding="utf-8")
        assert "<html" in html.lower()
        assert "talk.mp4" in html

    def test_snapshot_is_referenced_relatively_and_need_not_exist(self, tmp_path):
        out = tmp_path / "report.html"
        reporter.save_html([make_slide(snapshot_path="C:\\anywhere\\frame_000012.jpg")],
                           out, video_name="talk.mp4")
        html = out.read_text(encoding="utf-8")
        assert 'src="snapshots/frame_000012.jpg"' in html
        assert "C:\\anywhere" not in html      # only the basename is emitted

    def test_bullets_are_truncated_to_five(self, tmp_path):
        out = tmp_path / "report.html"
        reporter.save_html([make_slide(bullets=[f"b{i}" for i in range(9)])],
                           out, video_name="talk.mp4")
        html = out.read_text(encoding="utf-8")
        assert "<li>b4</li>" in html
        assert "<li>b5</li>" not in html

    def test_qa_card_does_not_consume_a_slide_number(self, tmp_path):
        out = tmp_path / "report.html"
        reporter.save_html([make_slide(title=""), make_qa_slide(), make_slide(title="")],
                           out, video_name="talk.mp4")
        html = out.read_text(encoding="utf-8")
        assert "Slide 1" in html
        assert "Slide 2" in html      # the slide after the Q&A card, not "Slide 3"
        assert "Slide 3" not in html

    def test_qa_pairs_are_rendered(self, tmp_path):
        out = tmp_path / "report.html"
        reporter.save_html([make_qa_slide(qa_pairs=[
            {"question": "Which column?", "answer": "A DB-5ms."}])],
            out, video_name="talk.mp4")
        html = out.read_text(encoding="utf-8")
        assert "Which column?" in html
        assert "A DB-5ms." in html

    def test_first_sentence_transcript_mode(self, tmp_path):
        out = tmp_path / "report.html"
        reporter.save_html([make_slide()], out, transcript_mode="first_sentence")
        html = out.read_text(encoding="utf-8")
        assert "We calibrate with an alkane ladder." in html
        assert "Then we run it." not in html

    def test_show_flags_suppress_their_sections(self, tmp_path):
        out = tmp_path / "report.html"
        reporter.save_html([make_slide()], out, show_image=False,
                           show_bullets=False, show_transcript=False)
        html = out.read_text(encoding="utf-8")
        assert "snapshots/" not in html
        assert "Use C7-C30 alkanes" not in html

    def test_meeting_info_appears_when_set(self, tmp_path):
        out = tmp_path / "report.html"
        reporter.save_html([make_slide()], out, meeting_title="Quarterly review",
                           meeting_date="2026-09-09", meeting_comments="Bring the quote.")
        html = out.read_text(encoding="utf-8")
        assert "Quarterly review" in html
        assert "2026-09-09" in html
        assert "Bring the quote." in html

    def test_title_slide_cover_is_rendered(self, tmp_path):
        out = tmp_path / "report.html"
        reporter.save_html(
            [make_slide()], out,
            title_slide={"image_path": "C:\\out\\cover.jpg",
                         "title": "Cover title", "subtitle": "Cover subtitle"})
        html = out.read_text(encoding="utf-8")
        assert "Cover title" in html
        assert "Cover subtitle" in html

    def test_empty_slide_list_still_writes_a_page(self, tmp_path):
        out = tmp_path / "report.html"
        reporter.save_html([], out, video_name="talk.mp4")
        assert out.is_file()
        assert "<html" in out.read_text(encoding="utf-8").lower()

    def test_creates_missing_parent_directory(self, tmp_path):
        out = tmp_path / "nested" / "report.html"
        reporter.save_html([make_slide()], out)
        assert out.is_file()


# -----------------------------------------------------------------
# save_html_for_pdf  (one slide per page, input to save_pdf_from_html)
# -----------------------------------------------------------------

class TestSaveHtmlForPdf:
    def test_forces_a_page_break_per_slide(self, tmp_path):
        out = tmp_path / "print.html"
        reporter.save_html_for_pdf([make_slide(), make_slide(timestamp_sec=60.0)],
                                   out, video_name="talk.mp4")
        assert "page-break-after" in out.read_text(encoding="utf-8")

    def test_snapshot_is_referenced_relatively(self, tmp_path):
        out = tmp_path / "print.html"
        reporter.save_html_for_pdf([make_slide()], out)
        assert 'src="snapshots/frame_000012.jpg"' in out.read_text(encoding="utf-8")

    def test_creates_missing_parent_directory(self, tmp_path):
        out = tmp_path / "nested" / "print.html"
        reporter.save_html_for_pdf([make_slide()], out)
        assert out.is_file()


# -----------------------------------------------------------------
# save_pdf_from_html  (backend chain, no real PDF engine needed)
# -----------------------------------------------------------------

class TestSavePdfFromHtml:
    def _html(self, tmp_path):
        p = tmp_path / "in.html"
        p.write_text("<html><body>x</body></html>", encoding="utf-8")
        return p

    def test_first_backend_wins_and_the_others_are_not_tried(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(reporter, "_save_pdf_playwright",
                            lambda h, p: calls.append("playwright") or True)
        monkeypatch.setattr(reporter, "_save_pdf_weasyprint",
                            lambda h, p: calls.append("weasyprint") or True)
        monkeypatch.setattr(reporter, "_save_pdf_pdfkit",
                            lambda h, p: calls.append("pdfkit") or True)
        assert reporter.save_pdf_from_html(self._html(tmp_path), tmp_path / "out.pdf") is True
        assert calls == ["playwright"]

    def test_import_error_falls_through_to_the_next_backend(self, tmp_path, monkeypatch):
        calls = []

        def missing(h, p):
            raise ImportError("not installed")

        monkeypatch.setattr(reporter, "_save_pdf_playwright", missing)
        monkeypatch.setattr(reporter, "_save_pdf_weasyprint",
                            lambda h, p: calls.append("weasyprint") or True)
        assert reporter.save_pdf_from_html(self._html(tmp_path), tmp_path / "out.pdf") is True
        assert calls == ["weasyprint"]

    def test_generic_exception_also_falls_through(self, tmp_path, monkeypatch):
        """A broken GTK/DLL load raises OSError rather than ImportError, so
        the chain must not stop on it either."""
        calls = []

        def broken(h, p):
            raise OSError("cannot load library libgobject-2.0-0.dll")

        monkeypatch.setattr(reporter, "_save_pdf_playwright", broken)
        monkeypatch.setattr(reporter, "_save_pdf_weasyprint", broken)
        monkeypatch.setattr(reporter, "_save_pdf_pdfkit",
                            lambda h, p: calls.append("pdfkit") or True)
        assert reporter.save_pdf_from_html(self._html(tmp_path), tmp_path / "out.pdf") is True
        assert calls == ["pdfkit"]

    def test_all_backends_failing_returns_false(self, tmp_path, monkeypatch):
        def broken(h, p):
            raise RuntimeError("nope")

        for name in ("_save_pdf_playwright", "_save_pdf_weasyprint", "_save_pdf_pdfkit"):
            monkeypatch.setattr(reporter, name, broken)
        assert reporter.save_pdf_from_html(self._html(tmp_path), tmp_path / "out.pdf") is False

    def test_creates_the_pdf_parent_directory(self, tmp_path, monkeypatch):
        seen = {}

        def record(h, p):
            seen["parent_exists"] = p.parent.is_dir()
            return True

        monkeypatch.setattr(reporter, "_save_pdf_playwright", record)
        reporter.save_pdf_from_html(self._html(tmp_path), tmp_path / "nested" / "out.pdf")
        assert seen["parent_exists"] is True


# -----------------------------------------------------------------
# Notes writers
# -----------------------------------------------------------------

class TestSaveNotesTxt:
    def test_header_fields_and_body(self, tmp_path):
        out = tmp_path / "notes.txt"
        reporter.save_notes_txt("## SUMMARY\nAll good.", out, "call.mp4", "ollama",
                                model_name="qwen3:14b", event_date="2026-09-09",
                                comments="Bring the quote.")
        text = out.read_text(encoding="utf-8")
        assert "Notes: call.mp4" in text
        assert "Meeting date: 2026-09-09" in text
        assert "Comments: Bring the quote." in text
        assert "LLM backend: ollama (qwen3:14b)" in text
        assert "All good." in text

    def test_optional_fields_are_omitted_when_blank(self, tmp_path):
        out = tmp_path / "notes.txt"
        reporter.save_notes_txt("body", out, "call.mp4", "ollama")
        text = out.read_text(encoding="utf-8")
        assert "Meeting date:" not in text
        assert "Comments:" not in text
        assert "LLM backend: ollama\n" in text     # no model suffix

    def test_does_not_create_a_missing_parent_directory(self, tmp_path):
        """Deliberate asymmetry with every slide exporter, which does mkdir.
        Pinned so a caller writing notes into a fresh folder knows it must
        create it first."""
        out = tmp_path / "nested" / "notes.txt"
        with pytest.raises(FileNotFoundError):
            reporter.save_notes_txt("body", out, "call.mp4", "ollama")


class TestSaveNotesHtml:
    def test_returns_the_path_and_renders_sections(self, tmp_path):
        out = tmp_path / "notes.html"
        result = reporter.save_notes_html("## SUMMARY\n- one\n- two\n", out, "call.mp4")
        assert result == out
        html = out.read_text(encoding="utf-8")
        assert "<h2>SUMMARY</h2>" in html
        assert html.count("<li>") == 2
        assert "call.mp4" in html

    def test_meeting_info_appears_when_set(self, tmp_path):
        out = tmp_path / "notes.html"
        reporter.save_notes_html("## S\nx", out, "call.mp4", title="MEETING NOTES",
                                 event_date="2026-09-09", comments="Bring the quote.")
        html = out.read_text(encoding="utf-8")
        assert "2026-09-09" in html
        assert "Bring the quote." in html

    def test_does_not_create_a_missing_parent_directory(self, tmp_path):
        out = tmp_path / "nested" / "notes.html"
        with pytest.raises(FileNotFoundError):
            reporter.save_notes_html("## S\nx", out, "call.mp4")
