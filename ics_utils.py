# =============================================================
#  Transkription_Notes_Pipeline - ics_utils.py
#  Meeting-info extraction for the GUI's "Load meeting info..." button
#  (task #35, extended in task #93) and the --ics CLI flag. Also home to
#  parse_ics_events (task #94), which reads every VEVENT in a calendar
#  export rather than just the first one, for correlating a whole
#  calendar against a folder of recordings - see
#  correlate_calendar_recordings.py.
#
#  Three source formats, one common return shape -
#  {"title": str, "date": str, "comments": str, "agenda": str,
#  "attendees": list[dict]} (text fields "" and attendees [] if not
#  found; task #95 added agenda/attendees):
#    .ics  - minimal, dependency-free iCalendar (RFC 5545) reader.
#            Extracts SUMMARY/DTSTART/DESCRIPTION/ATTENDEE from the
#            first VEVENT block, plus a best-effort "Agenda:" section
#            inside DESCRIPTION. Not a full RFC 5545 parser: no
#            timezone conversion, no recurrence handling, no
#            VALARM/VTODO.
#    .txt  - labeled-fields format: lines starting with "Title:",
#            "Date:", "Comments:", "Agenda:", or "Invitees:"
#            (case-insensitive; "Subject"/"Meeting", "Notes"/"Note"/
#            "Comment", and "Attendees"/"Attendee"/"Participants"/
#            "Participant" are accepted synonyms). A label's value
#            continues on following lines until the next recognized
#            label or EOF, so Comments/Agenda can span multiple lines,
#            and Invitees can list one name per line or comma/semicolon
#            -separated on one line (optionally 'Name <email>').
#    .docx - same labeled-fields format, read via python-docx
#            (paragraph text only; tables/headers/footers not read).
#
#  attendee_names(attendees) returns just the display names, in order,
#  for use as a speaker-name suggestion pick-list in the GUI's Rename
#  Speakers dialog (task #95): when a recording's meeting info came
#  with a calendar invite or a "Invitees:" line, those real names are
#  offered ahead of a blind guess.
#
#  parse_meeting_info_file(path) dispatches on the file extension and is
#  what callers should normally use; parse_ics/parse_txt/parse_docx are
#  also exposed individually for anything that already assumes .ics.
# =============================================================

import logging
import re
from datetime import datetime, timezone
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


# =============================================================
# Attendees and Agenda (task #95): the "Topic/Agenda/Description/
# Attendees" a calendar invite carries, used both for the correlation
# CSV report (correlate_calendar_recordings.py) and, via attendee_names,
# as a speaker-name suggestion pick-list in the GUI's Rename Speakers
# dialog. Attendees are always {"name": str, "email": str} dicts, in
# the order they appear, from every source (.ics ATTENDEE lines or a
# .txt/.docx "Invitees:"/"Attendees:"/"Participants:" line) - one shape
# regardless of which parser produced it.
# =============================================================

_ATTENDEE_CN_RE = re.compile(r'CN=("?)([^;:]+)\1', re.IGNORECASE)
_MAILTO_RE = re.compile(r"mailto:(.+)$", re.IGNORECASE)
_AGENDA_LABEL_RE = re.compile(r"agenda\s*:?\s*", re.IGNORECASE)


def _parse_ics_attendees(vevent_lines: list[str]) -> list[dict]:
    """Every ATTENDEE line in a VEVENT block, e.g.
    'ATTENDEE;CN=Jane Doe;ROLE=REQ-PARTICIPANT:mailto:jane@example.com'.
    Returns [{"name": str, "email": str}, ...] in the order they appear;
    "name" falls back to the email address when no CN= param is present.
    ORGANIZER is deliberately not included - a real invite's "attendees"
    are distinct from its organizer."""
    attendees: list[dict] = []
    for line in vevent_lines:
        if not (line.startswith("ATTENDEE;") or line.startswith("ATTENDEE:")):
            continue
        params, _, value = line.partition(":")
        mailto_match = _MAILTO_RE.search(value.strip())
        email = mailto_match.group(1).strip() if mailto_match else value.strip()
        cn_match = _ATTENDEE_CN_RE.search(params)
        name = _unescape_text(cn_match.group(2)) if cn_match else email
        if name or email:
            attendees.append({"name": name or email, "email": email})
    return attendees


def _extract_agenda_from_raw(raw_description: str) -> str:
    """Best-effort: if the still-escaped DESCRIPTION value contains an
    'Agenda' label (common in Teams/Zoom/Outlook invite bodies that
    bundle join info, agenda, and notes into one DESCRIPTION field),
    return the text following it up to the next blank line (an
    escaped '\\n\\n' in the raw RFC 5545 value) or the end of the
    field. Returns "" if no such label is found - Agenda is not a
    standard RFC 5545 property, so this never invents content that
    is not literally present under that label. Operates on the raw,
    still-\\n-escaped value (not the already-unescaped comments text),
    since unescaping collapses paragraph breaks to single spaces."""
    if not raw_description:
        return ""
    m = _AGENDA_LABEL_RE.search(raw_description)
    if not m:
        return ""
    rest = raw_description[m.end():]
    para, _, _ = rest.partition("\\n\\n")
    return _unescape_text(para)


def format_attendees(attendees: list[dict]) -> str:
    """Render an attendees list as one semicolon-separated display
    string for CSV/UI use, e.g. 'Jane Doe <jane@example.com>; John
    Smith <john@example.com>'."""
    parts = []
    for a in attendees:
        name = (a.get("name") or "").strip()
        email = (a.get("email") or "").strip()
        if name and email and name != email:
            parts.append(f"{name} <{email}>")
        elif email:
            parts.append(email)
        elif name:
            parts.append(name)
    return "; ".join(parts)


def attendee_names(attendees: list[dict]) -> list[str]:
    """Just the display names from an attendees list, in order, for use
    as a speaker-name suggestion/autocomplete pick-list (task #95) -
    falls back to the email address when no name was present, and never
    returns a blank entry."""
    names = []
    for a in attendees:
        name = (a.get("name") or "").strip() or (a.get("email") or "").strip()
        if name:
            names.append(name)
    return names


def parse_ics(path: str) -> dict:
    """
    Parse the first VEVENT in an .ics file. Returns {"title": str,
    "date": str, "comments": str, "agenda": str, "attendees": list[dict]}
    (ISO date, "" for any text field not found, [] if no ATTENDEE lines).
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
    raw_description = _extract_field(vevent, "DESCRIPTION")
    comments = _unescape_text(raw_description)
    agenda = _extract_agenda_from_raw(raw_description)
    attendees = _parse_ics_attendees(vevent)

    if not summary and not date:
        raise ValueError("Could not find SUMMARY or DTSTART in the VEVENT block.")

    log.info("Parsed .ics: title=%r date=%r attendees=%d (from %s)",
              summary, date, len(attendees), p.name)
    return {"title": summary, "date": date, "comments": comments,
            "agenda": agenda, "attendees": attendees}


# =============================================================
# Whole-calendar reading (task #94): every VEVENT, with full start/end
# datetimes rather than just the first event's date-only DTSTART. Built
# for correlate_calendar_recordings.py, which matches a folder of
# meeting recordings against a full calendar export by time rather than
# loading meeting info for one already-identified file.
# =============================================================

_ICS_DATETIME_RE = re.compile(
    r"^(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})"
    r"(?:T(?P<h>\d{2})(?P<mi>\d{2})(?P<s>\d{2})(?P<z>Z)?)?$"
)


def _parse_ics_datetime(value: str) -> tuple[datetime | None, bool]:
    """
    Parse one RFC 5545 DATE or DATE-TIME value (DTSTART/DTEND's raw
    string; any ";TZID=..." param was already stripped off by
    _extract_field, since that only keeps the part after the last
    unescaped colon). Returns (datetime, is_all_day):
      "YYYYMMDD"          - all-day date, no time -> (datetime at
                            midnight, True).
      "YYYYMMDDTHHMMSS"   - floating/local time -> (that datetime,
                            False). Treated as already being in local
                            time, since the TZID (if any) isn't visible
                            here - this module is deliberately not a
                            full RFC 5545 parser (see module docstring).
      "YYYYMMDDTHHMMSSZ"  - UTC -> converted to naive local time via
                            the OS timezone, (that datetime, False).
    Returns (None, False) if value is blank or doesn't match either
    pattern (e.g. a malformed line) - callers treat that as "no time
    available" rather than raising, so one bad VEVENT can't abort
    reading the rest of the calendar.
    """
    m = _ICS_DATETIME_RE.match(value.strip())
    if not m:
        return None, False
    if m.group("h") is None:
        return datetime(int(m.group("y")), int(m.group("m")), int(m.group("d"))), True
    dt = datetime(int(m.group("y")), int(m.group("m")), int(m.group("d")),
                  int(m.group("h")), int(m.group("mi")), int(m.group("s")))
    if m.group("z"):
        dt = dt.replace(tzinfo=timezone.utc).astimezone().replace(tzinfo=None)
    return dt, False


def parse_ics_events(path: str) -> list[dict]:
    """
    Parse EVERY VEVENT block in an .ics file, unlike parse_ics above
    (which only reads the first one and only its date, not time) - this
    is what correlate_calendar_recordings.py scans a whole calendar
    export with.

    Returns a list of {"title": str, "start": datetime | None,
    "end": datetime | None, "all_day": bool, "comments": str,
    "agenda": str, "attendees": list[dict]}, one per VEVENT, in the
    order they appear in the file. "end" falls back to "start" when
    DTEND is missing or unparseable, so a point-in-time or slightly
    malformed event still gets a usable end time rather than None
    outright; "start"/"end" are both None only if DTSTART itself was
    missing/unparseable. "agenda" is a best-effort extraction of an
    "Agenda:"-labeled section inside DESCRIPTION (see
    _extract_agenda_from_raw), "" if not present. "attendees" is every
    ATTENDEE line as {"name", "email"} dicts (see _parse_ics_attendees),
    [] if none.

    Raises FileNotFoundError if path doesn't exist. Never raises for a
    file with zero VEVENT blocks (or a dangling BEGIN:VEVENT with no
    matching END:VEVENT, which is simply not parsed past that point) -
    just returns an empty (or partial) list; the caller decides whether
    that's worth reporting.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    raw = p.read_text(encoding="utf-8", errors="replace")
    lines = _unfold_lines(raw)

    events: list[dict] = []
    i = 0
    while i < len(lines):
        if lines[i].strip().upper() == "BEGIN:VEVENT":
            try:
                end_idx = next(j for j in range(i, len(lines))
                               if lines[j].strip().upper() == "END:VEVENT")
            except StopIteration:
                break  # unterminated VEVENT at EOF - stop rather than misparse the tail
            vevent = [l.strip() for l in lines[i:end_idx]]
            summary = _unescape_text(_extract_field(vevent, "SUMMARY"))
            raw_description = _extract_field(vevent, "DESCRIPTION")
            comments = _unescape_text(raw_description)
            agenda = _extract_agenda_from_raw(raw_description)
            attendees = _parse_ics_attendees(vevent)
            start, all_day = _parse_ics_datetime(_extract_field(vevent, "DTSTART"))
            end, _ = _parse_ics_datetime(_extract_field(vevent, "DTEND"))
            events.append({
                "title": summary,
                "start": start,
                "end": end if end is not None else start,
                "all_day": all_day,
                "comments": comments,
                "agenda": agenda,
                "attendees": attendees,
            })
            i = end_idx + 1
        else:
            i += 1

    log.info("Parsed %d VEVENT(s) from %s", len(events), p.name)
    return events


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
    "agenda": "agenda",
    "invitees": "attendees",
    "invitee": "attendees",
    "attendees": "attendees",
    "attendee": "attendees",
    "participants": "attendees",
    "participant": "attendees",
}

_LABEL_LINE_RE = re.compile(
    r"^\s*(" + "|".join(re.escape(k) for k in _LABEL_SYNONYMS) + r")\s*:\s*(.*)$",
    re.IGNORECASE,
)

_DDMMYYYY_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$")
_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})")

_ATTENDEE_EMAIL_ANGLE_RE = re.compile(r"^(?P<name>.*?)\s*<(?P<email>[^<>]+)>\s*$")


def _split_attendee_tokens(raw: str) -> list[dict]:
    """Split a free-form 'Invitees:'/'Attendees:'/'Participants:' value
    (as typed by hand into a .txt/.docx meeting-info file, unlike the
    structured ATTENDEE lines a real .ics carries) into individual
    attendee dicts. Splits on comma, semicolon, or newline; a token may
    optionally carry an email in angle brackets, e.g. 'Jane Doe
    <jane@example.com>'. Blank tokens are dropped."""
    attendees: list[dict] = []
    for tok in re.split(r"[,\n;]+", raw):
        tok = tok.strip()
        if not tok:
            continue
        m = _ATTENDEE_EMAIL_ANGLE_RE.match(tok)
        if m:
            name = m.group("name").strip()
            email = m.group("email").strip()
            attendees.append({"name": name or email, "email": email})
        else:
            attendees.append({"name": tok, "email": ""})
    return attendees


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
    "comments": str, "agenda": str, "attendees": list[dict]} (task #95
    added "Agenda:" and "Invitees:"/"Attendees:"/"Participants:"), each
    text field "" and attendees [] if that label was never found. Never
    raises: a file with none of the labels just returns all-blank,
    which the caller (parse_meeting_info_file) reports the same way
    parse_ics does when a VEVENT has neither SUMMARY nor DTSTART.
    """
    result = {"title": "", "date": "", "comments": "", "agenda": "", "attendees": []}
    current: str | None = None
    buffers: dict[str, list[str]] = {
        "title": [], "date": [], "comments": [], "agenda": [], "attendees": [],
    }

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
    result["agenda"] = "\n".join(buffers["agenda"]).strip()
    result["attendees"] = _split_attendee_tokens("\n".join(buffers["attendees"]))
    return result


def parse_txt(path: str) -> dict:
    """Parse meeting info from a plain-text file (see parse_labeled_text
    for the expected format). Raises FileNotFoundError / ValueError,
    same contract as parse_ics.

    Falls back to treating the ENTIRE file as free-form comments when no
    Title:/Date:/Comments: (etc.) label is found anywhere in it (task:
    YouTube description.txt import - a YouTube video description has no
    labels at all, and previously had to be manually prefixed with a
    "Comments:" line before "Load meeting info..." would accept it). A
    properly labeled file is completely unaffected - this only kicks in
    when parse_labeled_text found nothing to work with."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    text = p.read_text(encoding="utf-8", errors="replace")
    result = parse_labeled_text(text)
    if not any(result.values()):
        stripped = text.strip()
        if not stripped:
            raise ValueError(
                "Could not find a Title:, Date:, or Comments: line in this file, "
                "and the file is empty."
            )
        result["comments"] = stripped
        log.info("Parsed .txt: no Title:/Date:/Comments: labels found - using the "
                  "entire file (%d char(s)) as comments (from %s).",
                  len(stripped), p.name)
        return result
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
