# =============================================================
#  Transkription_Notes_Pipeline - ics_utils.py
#  Meeting-info extraction for the GUI's "Load meeting info..." button
#  (task #35, extended in task #93) and the --ics CLI flag.
#
#  Three source formats, one common return shape -
#  {"title": str, "date": str, "comments": str} (each "" if not found):
#    .ics  - minimal, dependency-free iCalendar (RFC 5545) reader.
#            Extracts SUMMARY/DTSTART/DESCRIPTION from the first VEVENT
#            block. Not a full RFC 5545 parser: no timezone conversion,
#            no recurrence handling, no VALARM/VTODO.
#    .txt  - labeled-fields format: lines starting with "Title:",
#            "Date:", or "Comments:" (case-insensitive; "Subject"/
#            "Meeting" and "Notes"/"Note"/"Comment" are accepted
#            synonyms for Title/Comments). A label's value continues on
#            following lines until the next recognized label or EOF, so
#            Comments can span multiple lines/paragraphs.
#    .docx - same labeled-fields format, read via python-docx
#            (paragraph text only; tables/headers/footers not read).
#
#  parse_meeting_info_file(path) dispatches on the file extension and is
#  what callers should normally use; parse_ics/parse_txt/parse_docx are
#  also exposed individually for anything that already assumes .ics.
# =============================================================

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)


def _unfold_lines(raw: str) -> list[str]:
    """RFC 5545 line folding: a line starting with a single space or tab
    is a continuation of the previous line. Unfold before parsing."""
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    unfolded: list[str] = []
    for line in lines:
        if line.startswith((" ", "\t")) and unfolded:
            unfolded[-1] += line[1:]
        else:
            unfolded.append(line)
    return unfolded


def _unescape_text(value: str) -> str:
    """Undo RFC 5545 TEXT escaping (SUMMARY/DESCRIPTION values)."""
    value = value.replace("\\n", " ").replace("\\N", " ")
    value = value.replace("\\,", ",").replace("\\;", ";")
    value = value.replace("\\\\", "\\")
    return value.strip()


def _extract_field(vevent_lines: list[str], field: str) -> str:
    """Return the value of the first 'FIELD' or 'FIELD;PARAM=...' line,
    split on the first unescaped colon (params come before it)."""
    prefix_bare = field + ":"
    prefix_params = field + ";"
    for line in vevent_lines:
        if line.startswith(prefix_bare) or line.startswith(prefix_params):
            _, _, value = line.partition(":")
            return value.strip()
    return ""


def _format_date(dtstart_value: str) -> str:
    """DTSTART is 'YYYYMMDD' (all-day) or 'YYYYMMDDTHHMMSS[Z]'. Return
    an ISO date (YYYY-MM-DD); the time-of-day and timezone are not
    preserved, this is a best-effort date-only extraction."""
    digits = re.match(r"(\d{4})(\d{2})(\d{2})", dtstart_value)
    if not digits:
        return ""
    return f"{digits.group(1)}-{digits.group(2)}-{digits.group(3)}"


def parse_ics(path: str) -> dict:
    """
    Parse the first VEVENT in an .ics file. Returns
    {"title": str, "date": str, "comments": str} (ISO date, or "" for
    any field not found).
    Raises FileNotFoundError / ValueError on unreadable or empty input;
    callers should catch and show the message to the user rather than
    guessing at a title/date.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    raw = p.read_text(encoding="utf-8", errors="replace")
    lines = _unfold_lines(raw)

    try:
        start = next(i for i, l in enumerate(lines) if l.strip().upper() == "BEGIN:VEVENT")
        end = next(i for i, l in enumerate(lines) if l.strip().upper() == "END:VEVENT")
    except StopIteration:
        raise ValueError("No VEVENT block found in this .ics file.")

    vevent = [l.strip() for l in lines[start:end]]

    summary = _unescape_text(_extract_field(vevent, "SUMMARY"))
    dtstart = _extract_field(vevent, "DTSTART")
    date = _format_date(dtstart)
    comments = _unescape_text(_extract_field(vevent, "DESCRIPTION"))

    if not summary and not date:
        raise ValueError("Could not find SUMMARY or DTSTART in the VEVENT block.")

    log.info("Parsed .ics: title=%r date=%r (from %s)", summary, date, p.name)
    return {"title": summary, "date": date, "comments": comments}


# =============================================================
# Labeled-fields text format (.txt/.docx), task #93
# =============================================================

_LABEL_SYNONYMS = {
    "title": "title",
    "subject": "title",
    "meeting": "title",
    "date": "date",
    "comments": "comments",
    "comment": "comments",
    "notes": "comments",
    "note": "comments",
}

_LABEL_LINE_RE = re.compile(
    r"^\s*(" + "|".join(re.escape(k) for k in _LABEL_SYNONYMS) + r")\s*:\s*(.*)$",
    re.IGNORECASE,
)

_DDMMYYYY_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$")
_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})")


def _normalize_date(value: str) -> str:
    """Best-effort normalization to ISO YYYY-MM-DD. Accepts an already-
    ISO value (leading YYYY-MM-DD, extra text after it is dropped) or
    German-style DD.MM.YYYY. Anything else is returned unchanged rather
    than dropped - a raw, non-ISO date the user typed is still more
    useful pre-filled than an empty field."""
    value = value.strip()
    m = _ISO_DATE_RE.match(value)
    if m:
        y, mo, d = m.groups()
        return f"{y}-{int(mo):02d}-{int(d):02d}"
    m = _DDMMYYYY_RE.match(value)
    if m:
        d, mo, y = m.groups()
        return f"{y}-{int(mo):02d}-{int(d):02d}"
    return value


def parse_labeled_text(text: str) -> dict:
    """
    Parse the labeled-fields meeting-info format shared by .txt and
    .docx sources: lines starting with "Title:"/"Subject:"/"Meeting:",
    "Date:", or "Comments:"/"Comment:"/"Notes:"/"Note:" (case-
    insensitive). A label's value continues on every following line
    until the next recognized label or the end of the text, so Comments
    can span multiple lines/paragraphs - this is the field most likely
    to hold free-form notes.

    Lines before the first recognized label are ignored (no implicit
    "first line is the title" guessing - task #93 was scoped to the
    explicit labeled format only). Returns {"title": str, "date": str,
    "comments": str}, each "" if that label was never found. Never
    raises: a file with none of the labels just returns all-blank,
    which the caller (parse_meeting_info_file) reports the same way
    parse_ics does when a VEVENT has neither SUMMARY nor DTSTART.
    """
    result = {"title": "", "date": "", "comments": ""}
    current: str | None = None
    buffers: dict[str, list[str]] = {"title": [], "date": [], "comments": []}

    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        m = _LABEL_LINE_RE.match(raw_line)
        if m:
            label = _LABEL_SYNONYMS[m.group(1).lower()]
            current = label
            first_value = m.group(2).strip()
            if first_value:
                buffers[label].append(first_value)
        elif current is not None:
            if raw_line.strip():
                buffers[current].append(raw_line.strip())

    result["title"] = " ".join(buffers["title"]).strip()
    result["date"] = _normalize_date(" ".join(buffers["date"]).strip())
    result["comments"] = "\n".join(buffers["comments"]).strip()
    return result


def parse_txt(path: str) -> dict:
    """Parse meeting info from a plain-text file (see parse_labeled_text
    for the expected format). Raises FileNotFoundError / ValueError,
    same contract as parse_ics."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    text = p.read_text(encoding="utf-8", errors="replace")
    result = parse_labeled_text(text)
    if not any(result.values()):
        raise ValueError(
            "Could not find a Title:, Date:, or Comments: line in this file."
        )
    log.info("Parsed .txt: title=%r date=%r comments=%d char(s) (from %s)",
              result["title"], result["date"], len(result["comments"]), p.name)
    return result


def parse_docx(path: str) -> dict:
    """Parse meeting info from a Word document's paragraph text (see
    parse_labeled_text for the expected format). Tables, headers, and
    footers are not read - only the main body paragraphs, in order.
    Raises FileNotFoundError / ValueError, same contract as parse_ics;
    also raises whatever python-docx raises for a corrupt/non-.docx
    file (e.g. a renamed .doc), since there is no reliable fallback."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    from docx import Document
    document = Document(str(p))
    text = "\n".join(para.text for para in document.paragraphs)
    result = parse_labeled_text(text)
    if not any(result.values()):
        raise ValueError(
            "Could not find a Title:, Date:, or Comments: line in this document."
        )
    log.info("Parsed .docx: title=%r date=%r comments=%d char(s) (from %s)",
              result["title"], result["date"], len(result["comments"]), p.name)
    return result


def parse_meeting_info_file(path: str) -> dict:
    """
    Dispatch to parse_ics/parse_txt/parse_docx by file extension - the
    entry point the GUI's "Load meeting info..." button and the --ics
    CLI flag should use, so both transparently accept any of the three
    supported formats. Raises ValueError for an unsupported extension,
    and otherwise whatever the matching parser raises.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".ics":
        return parse_ics(path)
    if suffix == ".txt":
        return parse_txt(path)
    if suffix == ".docx":
        return parse_docx(path)
    raise ValueError(
        f"Unsupported file type: {suffix or '(none)'}. Expected .ics, .txt, or .docx."
    )
