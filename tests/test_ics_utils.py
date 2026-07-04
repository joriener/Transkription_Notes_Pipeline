# =============================================================
#  Transkription_Notes_Pipeline - tests/test_ics_utils.py
#  Unit tests for ics_utils.py (task #93 extension): the existing .ics
#  VEVENT parser, plus the new labeled-fields .txt/.docx parser and the
#  parse_meeting_info_file dispatcher.
# =============================================================

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ics_utils


# -----------------------------------------------------------------
# parse_ics
# -----------------------------------------------------------------

ICS_TEMPLATE = """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
SUMMARY:{summary}
DTSTART:{dtstart}
{description_line}END:VEVENT
END:VCALENDAR
"""


class TestParseIcs:
    def _write_ics(self, tmp_path, summary="Weekly Sync", dtstart="20260704T090000",
                    description=None):
        desc_line = f"DESCRIPTION:{description}\n" if description else ""
        content = ICS_TEMPLATE.format(summary=summary, dtstart=dtstart, description_line=desc_line)
        p = tmp_path / "invite.ics"
        p.write_text(content, encoding="utf-8")
        return p

    def test_title_and_date(self, tmp_path):
        p = self._write_ics(tmp_path)
        result = ics_utils.parse_ics(str(p))
        assert result["title"] == "Weekly Sync"
        assert result["date"] == "2026-07-04"

    def test_description_becomes_comments(self, tmp_path):
        p = self._write_ics(tmp_path, description="Bring the Q3 numbers.")
        result = ics_utils.parse_ics(str(p))
        assert result["comments"] == "Bring the Q3 numbers."

    def test_no_description_yields_blank_comments(self, tmp_path):
        p = self._write_ics(tmp_path)
        result = ics_utils.parse_ics(str(p))
        assert result["comments"] == ""

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ics_utils.parse_ics(str(tmp_path / "missing.ics"))

    def test_no_vevent_raises(self, tmp_path):
        p = tmp_path / "empty.ics"
        p.write_text("BEGIN:VCALENDAR\nEND:VCALENDAR\n", encoding="utf-8")
        with pytest.raises(ValueError):
            ics_utils.parse_ics(str(p))


# -----------------------------------------------------------------
# parse_labeled_text (shared by .txt and .docx)
# -----------------------------------------------------------------

class TestParseLabeledText:
    def test_all_three_fields(self):
        text = "Title: Quarterly Review\nDate: 2026-07-04\nComments: Bring laptop.\n"
        result = ics_utils.parse_labeled_text(text)
        assert result == {"title": "Quarterly Review", "date": "2026-07-04",
                           "comments": "Bring laptop."}

    def test_comments_span_multiple_lines_until_next_label(self):
        text = (
            "Title: Kickoff\n"
            "Comments: First line of notes.\n"
            "Second line of notes.\n"
            "Third line.\n"
            "Date: 2026-07-04\n"
        )
        result = ics_utils.parse_labeled_text(text)
        assert result["comments"] == "First line of notes.\nSecond line of notes.\nThird line."
        assert result["date"] == "2026-07-04"

    def test_synonyms_recognized(self):
        text = "Subject: Board Meeting\nNotes: Confidential agenda.\n"
        result = ics_utils.parse_labeled_text(text)
        assert result["title"] == "Board Meeting"
        assert result["comments"] == "Confidential agenda."

    def test_case_insensitive_labels(self):
        text = "TITLE: Standup\ncomments: quick sync\n"
        result = ics_utils.parse_labeled_text(text)
        assert result["title"] == "Standup"
        assert result["comments"] == "quick sync"

    def test_ddmmyyyy_date_normalized_to_iso(self):
        text = "Date: 04.07.2026\n"
        result = ics_utils.parse_labeled_text(text)
        assert result["date"] == "2026-07-04"

    def test_unrecognized_date_format_passed_through(self):
        text = "Date: next Tuesday\n"
        result = ics_utils.parse_labeled_text(text)
        assert result["date"] == "next Tuesday"

    def test_lines_before_first_label_are_ignored(self):
        text = "Some preamble text.\nTitle: Real Title\n"
        result = ics_utils.parse_labeled_text(text)
        assert result["title"] == "Real Title"

    def test_no_labels_returns_all_blank(self):
        text = "Just some random notes with no structure.\n"
        result = ics_utils.parse_labeled_text(text)
        assert result == {"title": "", "date": "", "comments": ""}

    def test_blank_lines_between_comment_lines_are_skipped_not_joined_as_empty(self):
        text = "Comments: Line one.\n\nLine two.\n"
        result = ics_utils.parse_labeled_text(text)
        assert result["comments"] == "Line one.\nLine two."


# -----------------------------------------------------------------
# parse_txt
# -----------------------------------------------------------------

class TestParseTxt:
    def test_reads_labeled_txt_file(self, tmp_path):
        p = tmp_path / "meeting.txt"
        p.write_text("Title: Sales Call\nDate: 2026-07-04\nComments: Discuss pricing.\n",
                      encoding="utf-8")
        result = ics_utils.parse_txt(str(p))
        assert result["title"] == "Sales Call"
        assert result["comments"] == "Discuss pricing."

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ics_utils.parse_txt(str(tmp_path / "missing.txt"))

    def test_no_recognized_labels_raises(self, tmp_path):
        p = tmp_path / "plain.txt"
        p.write_text("Just some unrelated text.\n", encoding="utf-8")
        with pytest.raises(ValueError):
            ics_utils.parse_txt(str(p))


# -----------------------------------------------------------------
# parse_docx
# -----------------------------------------------------------------

class TestParseDocx:
    def _write_docx(self, tmp_path, lines):
        from docx import Document
        doc = Document()
        for line in lines:
            doc.add_paragraph(line)
        p = tmp_path / "meeting.docx"
        doc.save(str(p))
        return p

    def test_reads_labeled_docx_paragraphs(self, tmp_path):
        p = self._write_docx(tmp_path, [
            "Title: Product Demo", "Date: 2026-07-04",
            "Comments: Show the new dashboard first.",
        ])
        result = ics_utils.parse_docx(str(p))
        assert result["title"] == "Product Demo"
        assert result["date"] == "2026-07-04"
        assert result["comments"] == "Show the new dashboard first."

    def test_comments_across_multiple_paragraphs(self, tmp_path):
        p = self._write_docx(tmp_path, [
            "Comments: Paragraph one.", "Paragraph two.", "Title: Wrap-up",
        ])
        result = ics_utils.parse_docx(str(p))
        assert result["comments"] == "Paragraph one.\nParagraph two."
        assert result["title"] == "Wrap-up"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ics_utils.parse_docx(str(tmp_path / "missing.docx"))

    def test_no_recognized_labels_raises(self, tmp_path):
        p = self._write_docx(tmp_path, ["Just some unrelated paragraph."])
        with pytest.raises(ValueError):
            ics_utils.parse_docx(str(p))


# -----------------------------------------------------------------
# parse_meeting_info_file dispatcher
# -----------------------------------------------------------------

class TestParseMeetingInfoFile:
    def test_dispatches_ics(self, tmp_path):
        p = tmp_path / "invite.ics"
        p.write_text(
            "BEGIN:VCALENDAR\nBEGIN:VEVENT\nSUMMARY:Call\nDTSTART:20260704T090000\n"
            "END:VEVENT\nEND:VCALENDAR\n", encoding="utf-8")
        result = ics_utils.parse_meeting_info_file(str(p))
        assert result["title"] == "Call"

    def test_dispatches_txt(self, tmp_path):
        p = tmp_path / "notes.txt"
        p.write_text("Title: Text Dispatch\n", encoding="utf-8")
        result = ics_utils.parse_meeting_info_file(str(p))
        assert result["title"] == "Text Dispatch"

    def test_dispatches_docx(self, tmp_path):
        from docx import Document
        doc = Document()
        doc.add_paragraph("Title: Docx Dispatch")
        p = tmp_path / "notes.docx"
        doc.save(str(p))
        result = ics_utils.parse_meeting_info_file(str(p))
        assert result["title"] == "Docx Dispatch"

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "notes.pdf"
        p.write_text("irrelevant", encoding="utf-8")
        with pytest.raises(ValueError):
            ics_utils.parse_meeting_info_file(str(p))
