# =============================================================
#  Transkription_Notes_Pipeline - ics_utils.py
#  Minimal, dependency-free .ics (iCalendar, RFC 5545) reader.
#  Extracts just SUMMARY and DTSTART from the first VEVENT block,
#  enough to pre-fill "Meeting title" / "Meeting date" in the GUI
#  from an Outlook/Google/Teams invite. Not a full RFC 5545 parser:
#  no timezone conversion, no recurrence handling, no VALARM/VTODO.
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
    {"title": str, "date": str} (ISO date, or "" if not found).
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

    if not summary and not date:
        raise ValueError("Could not find SUMMARY or DTSTART in the VEVENT block.")

    log.info("Parsed .ics: title=%r date=%r (from %s)", summary, date, p.name)
    return {"title": summary, "date": date}
