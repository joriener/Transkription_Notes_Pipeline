# =============================================================
#  Transkription_Notes_Pipeline - gui_logic.py
#  Pure-logic helpers factored out of gui.py so they can be unit
#  tested without a tkinter/display dependency (see
#  tests/test_gui_logic.py). Nothing in this module may import
#  tkinter, and it must not import gui.py (gui.py imports this,
#  never the other way around).
# =============================================================


def parse_time_to_seconds(text: str) -> float | None:
    """
    Parse a Q&A section time field (task #61) as seconds, "MM:SS", or
    "HH:MM:SS". Blank input returns None (disabled). Raises ValueError
    for anything else unparseable, so callers can show a clear message
    instead of silently ignoring a typo.
    """
    text = (text or "").strip()
    if not text:
        return None
    if ":" in text:
        parts = text.split(":")
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        raise ValueError(f"Cannot parse time '{text}' (use SS, MM:SS, or HH:MM:SS).")
    return float(text)


def _safe_parse_time(text: str) -> float | None:
    """Best-effort wrapper: never raises, used where a stray typo must
    not abort building a whole batch. Callers that need to surface a
    clear error to the user should call parse_time_to_seconds directly."""
    try:
        return parse_time_to_seconds(text)
    except ValueError:
        return None


def build_batch_row(batch_columns, values) -> dict | None:
    """
    Given batch_columns (a tuple of column names - ("file", "language",
    "title", "date", "comments"), with "qa_start"/"qa_end" appended for
    the Video/Webinar tab) and values (the raw string cell values from
    one Treeview row, in the same order), return the row dict
    run_pipeline.run_batch_rows expects, or None if the file cell is
    blank (caller should skip the row).

    Q&A time values that fail to parse fall back to None rather than
    raising: the GUI validates Q&A text explicitly before a run and
    shows the user a clear warning, so this function stays a pure,
    always-succeeding transform.
    """
    values = list(values) + [""] * (len(batch_columns) - len(values))
    row = dict(zip(batch_columns, values))
    file_ = (row.get("file") or "").strip()
    if not file_:
        return None
    out = {
        "file": file_,
        "language": (row.get("language") or "").strip() or None,
        "meeting_title": (row.get("title") or "").strip() or None,
        "meeting_date": (row.get("date") or "").strip() or None,
        "meeting_comments": (row.get("comments") or "").strip() or None,
    }
    if "qa_start" in batch_columns:
        out["qa_start_time_sec"] = _safe_parse_time((row.get("qa_start") or "").strip())
        out["qa_end_time_sec"] = _safe_parse_time((row.get("qa_end") or "").strip())
    return out


def parse_list_line(line: str) -> dict | None:
    """
    Parse one line of a legacy "path|lang" batch list file. Returns
    {"file": ..., "language": ...} or None for a blank/comment line.
    Language is "" (not None) when absent, matching the batch table's
    own blank-cell convention.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split("|", 1)
    file_ = parts[0].strip()
    if not file_:
        return None
    lang = parts[1].strip() if len(parts) > 1 else ""
    return {"file": file_, "language": lang}


def csv_header_index(header, known_cols) -> dict:
    """
    Given a CSV header row (list of strings) and a set of known column
    names, return {col_name: index} for the ones present (case-
    insensitive lookup), or {} if none of known_cols appear in header
    at all (caller should then fall back to the plain file[,language]
    format instead of treating the first row as a header).
    """
    lowered = [h.strip().lower() for h in header]
    if not (known_cols & set(lowered)):
        return {}
    return {name: lowered.index(name) for name in known_cols if name in lowered}
