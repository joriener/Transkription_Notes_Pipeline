# =============================================================
#  Transkription_Notes_Pipeline - gui_logic.py
#  Pure-logic helpers factored out of gui.py so they can be unit
#  tested without a tkinter/display dependency (see
#  tests/test_gui_logic.py). Nothing in this module may import
#  tkinter, and it must not import gui.py (gui.py imports this,
#  never the other way around).
# =============================================================

import json
from pathlib import Path


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


# =============================================================
# Persisted GUI settings (task #71)
#
# gui_state.json (gitignored, lives next to gui.py) remembers the last
# whisper model/language/LLM backend/prompt template/thresholds/output
# folder/stage checkboxes used on each of the Meeting and Video/Webinar
# tabs, so re-opening the GUI doesn't reset them to config.py's defaults
# every time. Deliberately NOT persisted: the file/folder path, batch
# table contents, or meeting title/date/comments/Q&A times - those
# describe one specific run, not a lasting preference.
# =============================================================

STATE_FILENAME = "gui_state.json"

# Keys persisted for every Run tab (Meeting and Video/Webinar).
PERSISTED_KEYS_COMMON = (
    "recording_type", "prompt_template", "whisper_model", "language",
    "llm_backend", "output_dir", "enable_diarization", "no_summary",
    "force_retranscribe",
)

# Additional keys persisted only for the Video/Webinar tab.
PERSISTED_KEYS_VIDEO_ONLY = (
    "enable_slides", "enable_vlm", "hash_threshold", "animation_threshold",
    "fps", "min_slide_duration_sec", "recording_speed",
    "convert_video_to_realtime", "report_html", "report_csv", "report_json",
    "report_srt", "report_pdf", "report_slide_timing", "zip_snapshots",
    "report_show_image", "report_show_bullets", "report_show_transcript",
    "report_transcript_mode",
)


def persisted_keys_for(kind: str) -> tuple:
    """Which settings keys get saved/restored for this tab kind ("meeting"
    or "video")."""
    if kind == "video":
        return PERSISTED_KEYS_COMMON + PERSISTED_KEYS_VIDEO_ONLY
    return PERSISTED_KEYS_COMMON


def build_state_dict(kind: str, values: dict) -> dict:
    """
    Filter a dict of {key: current_value} down to only the keys this tab
    kind persists, dropping anything else (e.g. a key that doesn't apply
    to this kind, or isn't meant to be persisted at all). Used when
    snapshotting a tab's settings to write into gui_state.json.
    """
    keys = persisted_keys_for(kind)
    return {k: values[k] for k in keys if k in values}


def merge_persisted_state(kind: str, saved: dict, defaults: dict) -> dict:
    """
    Return defaults with any matching, persistable key from saved applied
    on top. Keys in saved that this kind doesn't persist (e.g. left over
    from an older gui_state.json, or a video-only key under "meeting")
    are ignored rather than applied.
    """
    keys = set(persisted_keys_for(kind))
    merged = dict(defaults)
    for k, v in (saved or {}).items():
        if k in keys:
            merged[k] = v
    return merged


def load_state_file(path) -> dict:
    """
    Read gui_state.json. Returns {} if the file is missing, empty, or not
    valid JSON, and if its top level isn't an object - persisted settings
    are a convenience, never a reason to fail GUI startup.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(text)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def save_state_file(path, state: dict) -> bool:
    """
    Write gui_state.json (pretty-printed for easy manual inspection/
    editing). Returns False instead of raising on failure (e.g. a
    read-only folder) - losing persisted settings should never block
    closing the app.
    """
    try:
        Path(path).write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        return True
    except OSError:
        return False
