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
    "title", "date", "comments"), with "qa_start"/"qa_end"/"title_image"/
    "srt_path" appended for the Video/Webinar tab) and values (the raw
    string cell values from one Treeview row, in the same order), return
    the row dict run_pipeline.run_batch_rows expects, or None if the file
    cell is blank (caller should skip the row).

    Q&A time values that fail to parse fall back to None rather than
    raising: the GUI validates Q&A text explicitly before a run and
    shows the user a clear warning, so this function stays a pure,
    always-succeeding transform.

    title_image (task: per-webinar title slide, since each recording can
    have its own cover image, unlike a setting shared across a whole
    batch): a blank cell means None, which falls back to whatever the
    Video/Webinar tab's shared "Title slide image" field holds for that
    run - same fallback convention as language/title/date/comments above.
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
    if "title_image" in batch_columns:
        out["title_slide_image_path"] = (row.get("title_image") or "").strip() or None
    if "srt_path" in batch_columns:
        # srt_path (task: batch YouTube import): a per-row .srt to import
        # as this file's transcript instead of transcribing it - see
        # run_pipeline.run_batch_rows' srt_import_path row key. A blank
        # cell means None, i.e. this row gets a real transcription like
        # any other (no shared-field fallback exists for this one, unlike
        # title_image, since there is no single "shared" .srt to fall
        # back to across a whole batch of different recordings).
        out["srt_import_path"] = (row.get("srt_path") or "").strip() or None
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

# Keys unique to each Run tab (Meeting vs Video/Webinar) - never shared,
# because each tab genuinely means something different by them: its own
# recording-type presets, its own prompt template selection, and (by
# design) its own default output folder.
PERSISTED_KEYS_PER_TAB = (
    "recording_type", "prompt_template", "output_dir", "use_date_subject_filename",
    "move_processed_files",
)

# Keys shared between the Meeting and Video/Webinar tabs: transcription/
# LLM engine settings that describe the SAME source material regardless
# of which tab processes it. Persisted ONCE under state["shared"], not
# duplicated per tab. Previously these lived under each tab's own section
# (the old PERSISTED_KEYS_COMMON) and could silently drift apart - e.g.
# Meeting's llm_backend left on "anthropic" while Video's stayed on
# "ollama", with nothing surfacing that the two had diverged. gui.py now
# backs these with ONE tkinter Variable per key, shared by both
# RunTabController instances, so editing either tab updates both live and
# there is exactly one value to persist. This set intentionally matches
# the old COPYABLE_KEYS: that was already the maintainers' own judgment
# of "safe to share between the two tabs" for the former manual "Copy
# settings to other tab" action (task #72, now removed - nothing left to
# copy once these are always in sync).
PERSISTED_KEYS_SHARED = (
    "whisper_model", "language", "output_language", "llm_backend",
    "enable_diarization", "no_summary", "force_retranscribe", "enhance_audio",
)

# Additional keys persisted only for the Video/Webinar tab.
PERSISTED_KEYS_VIDEO_ONLY = (
    "enable_slides", "enable_vlm", "hash_threshold", "animation_threshold",
    "fps", "min_slide_duration_sec", "recording_speed",
    "convert_video_to_realtime", "report_html", "report_csv", "report_json",
    "report_srt", "report_pdf", "report_slide_timing", "zip_snapshots",
    "report_show_image", "report_show_bullets", "report_show_transcript",
    "report_transcript_mode", "qa_include_in_summary",
)


def persisted_keys_for(kind: str) -> tuple:
    """All settings keys this tab kind reads/writes, spanning both its own
    per-tab section and the shared section ("meeting" and "video" no
    longer keep independent copies of the shared keys)."""
    keys = PERSISTED_KEYS_PER_TAB + PERSISTED_KEYS_SHARED
    if kind == "video":
        return keys + PERSISTED_KEYS_VIDEO_ONLY
    return keys


def build_tab_state_dict(kind: str, values: dict) -> dict:
    """
    Filter a dict of {key: current_value} down to only the keys THIS TAB
    OWNS (per-tab keys, plus video-only keys for the video tab) - the
    shared keys are deliberately excluded here, see build_shared_state_dict.
    Used when snapshotting a tab's settings to write into gui_state.json
    under state[kind].
    """
    keys = PERSISTED_KEYS_PER_TAB + (PERSISTED_KEYS_VIDEO_ONLY if kind == "video" else ())
    return {k: values[k] for k in keys if k in values}


def build_shared_state_dict(values: dict) -> dict:
    """
    Filter a dict of {key: current_value} down to only the shared keys
    (PERSISTED_KEYS_SHARED), written once into gui_state.json under
    state["shared"] regardless of which tab triggered the save - since
    both tabs' widgets are backed by the same Variable objects, either
    tab's values dict already holds the single, current, shared value for
    each of these keys.
    """
    return {k: values[k] for k in PERSISTED_KEYS_SHARED if k in values}


def merge_persisted_state(kind: str, saved_tab: dict, saved_shared: dict, defaults: dict) -> dict:
    """
    Return defaults with this tab's own persisted keys and the shared
    persisted keys applied on top. Keys in saved_tab/saved_shared that
    don't belong where they were found (e.g. a video-only key saved under
    "meeting", or a leftover key from an older gui_state.json) are
    ignored rather than applied.

    Migration for pre-centralization files: a gui_state.json written
    before the shared-settings centralization has no top-level "shared"
    section at all - PERSISTED_KEYS_SHARED were saved as part of each
    tab's own section back then (the old PERSISTED_KEYS_COMMON). Without
    this fallback, the first launch after updating would find
    saved_shared empty and silently reset whisper_model/language/
    llm_backend/etc. back to CONFIG defaults, discarding the user's
    last-used values even though nothing else changed. So: only when
    saved_shared has NO entry at all for a shared key do we fall back to
    that key's value in saved_tab (its pre-centralization home). Once
    this state gets saved again (build_shared_state_dict), state["shared"]
    is populated and this fallback stops being exercised for that
    installation. A shared key that legitimately IS present in
    saved_shared always wins over saved_tab, so this never overrides an
    already-migrated value.
    """
    tab_keys = set(PERSISTED_KEYS_PER_TAB) | (set(PERSISTED_KEYS_VIDEO_ONLY) if kind == "video" else set())
    merged = dict(defaults)
    for k, v in (saved_tab or {}).items():
        if k in tab_keys:
            merged[k] = v
        elif k in PERSISTED_KEYS_SHARED and k not in (saved_shared or {}):
            merged[k] = v
    for k, v in (saved_shared or {}).items():
        if k in PERSISTED_KEYS_SHARED:
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


# =============================================================
# Templates tab (task #77)
# =============================================================

def sanitize_template_name(name: str) -> str | None:
    """
    Validate and normalize a user-entered prompt template name for the
    Templates tab's New/Import/Copy actions. Strips whitespace, rejects
    blank input, rejects any path separator or ".." component (so a
    filename can never write outside prompts_dir), rejects "readme" (the
    folder's own doc file, not a template), and ensures a ".md"
    extension. Returns the normalized filename, or None if invalid.
    """
    name = (name or "").strip()
    if not name:
        return None
    if "/" in name or "\\" in name or ".." in name:
        return None
    if not name.lower().endswith(".md"):
        name += ".md"
    if name.lower() == "readme.md":
        return None
    return name
