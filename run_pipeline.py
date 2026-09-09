# =============================================================
#  Transkription_Notes_Pipeline V1.1 - run_pipeline.py
#  Master orchestrator, combining the former
#  Audio_Transkription_Notes_Pipeline (audio-only meeting/webinar
#  notes) and Video_Transkription_Notes_Pipeline (slide-change
#  detection + VLM annotation + webinar summary) into one tool.
#
#  Slide detection/VLM annotation only run for video files with
#  enable_slides=True (default). They are automatically skipped
#  for audio-only extensions (mp3, wav, m4a, ogg, flac, aac, wma).
#
#  V1.1: --output-dir override, --batch-folder auto-discovery,
#  independent notes format toggles (txt/html/pdf/docx), PDF export
#  now tries Playwright/Chromium before weasyprint/pdfkit.
#
#  CLI usage:
#    python run_pipeline.py                                  (GUI file picker)
#    python run_pipeline.py "file.mp4"                        (direct path)
#    python run_pipeline.py "file.mp4" de                     (force language)
#    python run_pipeline.py --prompt-template webinar "f.mp4" (choose prompts/webinar.md)
#    python run_pipeline.py --no-slides "webinar.mp4"         (audio-style run on a video)
#    python run_pipeline.py --no-vlm "webinar.mp4"            (slides, no VLM annotation)
#    python run_pipeline.py --no-whisper "webinar.mp4"        (slides only, no transcript)
#    python run_pipeline.py --no-summary "file.mp4"           (transcript only, no notes)
#    python run_pipeline.py --dry-run "webinar.mp4"           (slide timestamps only)
#    python run_pipeline.py --force-retranscribe "file.mp4"
#    python run_pipeline.py --diarize "file.mp4"
#    python run_pipeline.py --whisper-model base "file.mp4"
#    python run_pipeline.py --threshold 5 "webinar.mp4"
#    python run_pipeline.py --output-dir "D:\Notes\2026-07" "file.mp4"
#    python run_pipeline.py --docx "file.mp4"                 (also write notes .docx)
#    python run_pipeline.py --pdf "file.mp4"                  (also write notes .pdf)
#    python run_pipeline.py --notes-only "file_transcript_speakers.txt"
#    python run_pipeline.py --notes-batch "D:/Videos"
#    python run_pipeline.py --notes-batch-api "D:/Videos"      (Anthropic Batches API: 50% cheaper, slower)
#    python run_pipeline.py --file-list "list.txt"             (resumes automatically, skips already-done files)
#    python run_pipeline.py --batch-folder "D:/Videos"         (every supported file in a folder)
#    python run_pipeline.py --search "retention index"         (full-text search notes + slides)
#    python run_pipeline.py --log-file "file.mp4"              (also write this run's log to logs/)
#    python run_pipeline.py --mode audio_transcript "call.mp3"  (generic audio transcript, no slides)
#    python run_pipeline.py --mode video_transcript "demo.mp4"  (generic video transcript, slides on)
#    python run_pipeline.py --meeting-title "Q3 Roadmap" --meeting-date 2026-07-03 "call.mp4"
#    python run_pipeline.py --ics "invite.ics" "call.mp4"       (meeting title/date from a calendar invite)
#    python run_pipeline.py --comments "Follow-up needed on pricing" "call.mp4"
#    python run_pipeline.py --output-name "2026-07-03_Q3-Kickoff" "call.mp4"  (custom base filename)
#    python run_pipeline.py --date-subject-filename --meeting-title "Q3 Kickoff" "call.mp4"
#                                                               (output files named 20260703_Q3 Kickoff)
#    python run_pipeline.py --reannotate-failed "file_slides/file_slides.json"  (retry failed VLM slides only)
#    python run_pipeline.py --min-slide-duration 3.5 "webinar.mp4"  (animation debounce override)
#    python run_pipeline.py --qa-start 1215 "webinar.mp4"       (Q&A from 20:15 to the end: no new
#                                                                 slides there; Q&A folded into the
#                                                                 same summary unless --qa-exclude-
#                                                                 from-summary is also given)
#    python run_pipeline.py --gui                             (parameter GUI)
#
#  FORMAT list.txt (pipe separator for language is optional):
#    D:\Videos\meeting_en.mp4
#    D:\Videos\webinar_de.mp4|de
#    # Lines starting with # are ignored
#
#  OUTPUT FILES (next to the source file, or under --output-dir if set):
#    *_transcript_speakers.txt   Full transcript, timestamps + speakers
#    *_text.txt                  Plain text grouped by speaker
#    *_transcript.srt            Subtitle file
#    *_segments.json             Cached whisperx segments (re-run without re-transcribing)
#    *_notes.txt / *_notes.html  Notes/summary (unless --no-summary)
#    *_notes.pdf / *_notes.docx  Optional, see --pdf / --docx or the GUI checkboxes
#  Additional files when slide detection ran (video mode):
#    *_slides/<stem>_report.html/.pdf, _slides.csv, _slides.json, snapshots/
# =============================================================

import warnings
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

import argparse
import base64
import json
import logging
import re
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

logging.getLogger("fontTools").setLevel(logging.ERROR)
logging.getLogger("weasyprint").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("whisperx").setLevel(logging.WARNING)

from config import CONFIG, PROMPTS_DIR, is_audio_only, SUPPORTED_EXTENSIONS
import db
import notes
import reporter
import speaker_id
import transcriber

log = logging.getLogger("Transkription_Notes_Pipeline")


def setup_logging(level: str = "INFO", log_to_file: bool = False,
                  log_dir: str | None = None) -> Path | None:
    """Configure root logging. If log_to_file is set, also writes a
    timestamped copy of this run's log under log_dir (used by the GUI Settings tab / CLI banner); otherwise None."""
    handlers = [logging.StreamHandler()]
    log_file_path = None
    if log_to_file:
        log_dir_path = Path(log_dir or (Path(__file__).parent / "logs"))
        log_dir_path.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir_path / f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        handlers.append(logging.FileHandler(log_file_path, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    if log_file_path:
        log.info("Logging to file: %s", log_file_path)
    return log_file_path


# ---------------------------------------------------------------------------
# Config merge helper - never mutate the module-level CONFIG so that GUI
# runs (same process, repeated calls) stay isolated per run.
# ---------------------------------------------------------------------------

def build_run_config(overrides: dict) -> dict:
    cfg = dict(CONFIG)
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


_FILENAME_DATETIME_RE = re.compile(
    r"(?P<y>20\d{2})-?(?P<m>\d{2})-?(?P<d>\d{2})"          # date: YYYY-MM-DD or YYYYMMDD
    r"(?:[_-]?(?P<h>\d{2})(?P<mi>\d{2})(?P<s>\d{2}))?"      # optional time: HHMMSS
)

# Generic recorder/app names that show up as filename prefixes but carry
# no meaning as a "topic" - stripped out of derive_heading_from_filename's
# best-effort topic extraction rather than kept as noise.
_GENERIC_FILENAME_WORDS = {"video", "audio", "recording", "meeting", "aufnahme", "gmt", "rec"}


def derive_heading_from_filename(filename: str) -> str | None:
    """
    Best-effort extraction of a date (and time, if present) out of a
    source recording filename, reformatted as "$Y-$M-$D_$H$N$S" (or just
    "$Y-$M-$D" if no time component is found) - e.g.
    "Video_2020-04-07_154005.mp4" -> "2020-04-07_154005". If anything
    else recognizable remains in the filename after removing the date/
    time and generic recorder-name words (see _GENERIC_FILENAME_WORDS),
    it's appended as a topic, e.g. "20260704_Sales_Call.mp4" ->
    "2026-07-04_Sales Call".

    Returns None if no recognizable date pattern is found anywhere in
    the filename, so callers can fall back to their own default. This is
    always just a fallback: CONFIG["meeting_title"] (and, for the output
    filename, output_basename_override) take precedence whenever set -
    a wrong or missing match here never blocks giving a file a proper
    name, since the user can always set those instead.
    """
    stem = Path(filename).stem
    match = _FILENAME_DATETIME_RE.search(stem)
    if not match:
        return None

    y, m, d = match.group("y"), match.group("m"), match.group("d")
    if match.group("h"):
        date_part = f"{y}-{m}-{d}_{match.group('h')}{match.group('mi')}{match.group('s')}"
    else:
        date_part = f"{y}-{m}-{d}"

    remainder = (stem[:match.start()] + stem[match.end():]).strip("_- ")
    topic_words = [w for w in re.split(r"[_\-\s]+", remainder)
                  if w and w.lower() not in _GENERIC_FILENAME_WORDS]
    topic = " ".join(topic_words).strip()
    return f"{date_part}_{topic}" if topic else date_part


def extract_datetime_from_filename(filename: str) -> datetime | None:
    """
    Same date/time pattern as derive_heading_from_filename above, but
    returns an actual datetime instead of a formatted heading string -
    used by correlate_calendar_recordings.py (task #94) to compare a
    recording's filename-embedded timestamp against calendar event
    times. Returns None if no recognizable date is found, or if a time
    component is missing: a date-only filename (no HHMMSS) isn't precise
    enough to usefully compare against calendar event times, so callers
    should fall back to the file's own last-modified time instead in
    that case rather than getting a misleadingly exact midnight
    timestamp.
    """
    stem = Path(filename).stem
    match = _FILENAME_DATETIME_RE.search(stem)
    if not match or not match.group("h"):
        return None
    try:
        return datetime(
            int(match.group("y")), int(match.group("m")), int(match.group("d")),
            int(match.group("h")), int(match.group("mi")), int(match.group("s")))
    except ValueError:
        return None  # e.g. an out-of-range value that still matched the regex shape


# Characters illegal in a Windows filename, plus control characters -
# stripped from meeting_title before it becomes part of an output
# filename (see build_date_subject_filename). Not just cosmetic: the
# project's own NAS paths (X:\Agilent\...) run on Windows/SMB, where
# writing any of these raises WinError 123 ("filename, directory name,
# or volume label syntax is incorrect") rather than something obvious.
_FILENAME_ILLEGAL_CHARS_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')

# Meeting/webinar titles (calendar subjects especially) can run long;
# cap the Subject portion so "YYYYMMDD_<subject>" plus this project's own
# suffixes (_transcript_speakers.txt, _slides/<name>_report_pdf.html,
# etc.) stays comfortably under Windows' historical 260-char path limit
# even several directories deep on a NAS share.
_MAX_SUBJECT_LEN = 80


def _sanitize_filename_component(text: str) -> str:
    """
    Make an arbitrary string (e.g. a calendar event's Subject) safe to use
    as part of a Windows filename: strips illegal characters, collapses
    whitespace runs (including the newlines some calendar exports embed
    in SUMMARY) to single spaces, trims trailing dots/spaces (Windows
    silently strips these anyway, but leaving them in causes confusing
    mismatches versus what actually gets created on disk), and truncates
    to _MAX_SUBJECT_LEN.
    """
    cleaned = _FILENAME_ILLEGAL_CHARS_RE.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned[:_MAX_SUBJECT_LEN].strip(" .")


def build_date_subject_filename(meeting_date: str, meeting_title: str, source_file: str) -> str | None:
    """
    Build a "YYYYMMDD_Subject" output basename (CONFIG["use_date_subject_filename"]).
    Returns None if meeting_title is blank after sanitizing - there is no
    sensible Subject to build around, and resolve_output_prefix falls
    back to its own default in that case rather than naming the file
    "20260703_" with nothing after the underscore.

    Date resolution order: 1) meeting_date, if it parses as an ISO
    "YYYY-MM-DD" (the format --meeting-date/--ics/the GUI's Meeting info
    fields already normalise to); 2) a date embedded in the source
    filename (extract_datetime_from_filename); 3) the source file's own
    last-modified date - always produces a date, never blocks naming
    just because neither of the above matched.
    """
    subject = _sanitize_filename_component(meeting_title or "")
    if not subject:
        return None

    date_obj = None
    if meeting_date:
        try:
            date_obj = datetime.strptime(meeting_date.strip(), "%Y-%m-%d")
        except ValueError:
            log.warning("meeting_date '%s' is not YYYY-MM-DD - falling back to filename/mtime "
                       "for the YYYYMMDD_Subject filename.", meeting_date)
    if date_obj is None:
        date_obj = extract_datetime_from_filename(Path(source_file).name)
    if date_obj is None:
        try:
            date_obj = datetime.fromtimestamp(Path(source_file).stat().st_mtime)
        except OSError:
            date_obj = datetime.now()

    return f"{date_obj.strftime('%Y%m%d')}_{subject}"


def _dedupe_date_subject_stem(out_dir: Path, stem: str, source_file: str) -> str:
    """
    Guard against two different recordings on the same day with the same
    (or same-after-sanitizing) meeting title computing to the identical
    "YYYYMMDD_Subject" stem and silently overwriting each other's output
    files - build_date_subject_filename() itself has no way to know about
    sibling files, so this runs afterwards, once we know the target
    directory.

    Uses the "<stem>_source_media.txt" sidecar (written near the end of
    Step 1 in process_file, see build_date_subject_filename's docstring)
    as the source of truth for "who does this stem already belong to":
      - no sidecar yet at this stem -> stem is free, use it as-is (first
        run, or a legacy pre-V1.19 output with no sidecar - unchanged
        behavior, we can't tell those apart and assume the best).
      - sidecar exists and points at THIS source_file -> this is a
        cache-aware re-run of the same recording (force_retranscribe=False,
        --no-summary re-run, etc.) - reuse the stem so the *_segments.json
        cache actually gets hit, exactly like before this change.
      - sidecar exists and points at a DIFFERENT file -> real collision,
        append "_2", "_3", ... until a free or same-source stem is found.
    """
    source_resolved = str(Path(source_file).resolve())
    candidate = stem
    n = 2
    while True:
        sidecar = out_dir / f"{candidate}_source_media.txt"
        if not sidecar.exists():
            return candidate
        try:
            recorded = sidecar.read_text(encoding="utf-8").strip()
        except OSError:
            recorded = ""
        if recorded and str(Path(recorded).resolve()) == source_resolved:
            return candidate
        candidate = f"{stem}_{n}"
        n += 1


def import_srt_transcript(video_file: str, srt_file: str, overrides: dict | None = None) -> bool:
    """
    Import an already-existing .srt (e.g. downloaded alongside a YouTube
    video, together with its description.txt) as this video's transcript,
    writing the exact same cache files a real whisperx transcription run
    would (*_segments.json, *_transcript_speakers.txt, *_text.txt,
    *_transcript.srt, *_source_media.txt) plus a transcripts DB row - so
    process_file's existing cache-aware shortcut picks it up and skips
    whisperx entirely on the next run against this file (see Step 1 of
    process_file: "Transcript cache found - loading ..."). This is the
    building block for both the --import-srt CLI flag and the GUI's
    "Import .srt as transcript..." button.

    Refuses to overwrite an existing transcript cache unless
    overrides["force_retranscribe"] is set (mirrors the
    --force-retranscribe convention used for a real re-transcription),
    since the whole point of importing an .srt is normally to AVOID
    redoing a transcription that has already happened - accidentally
    clobbering one with a lower-quality caption file would be a silent
    downgrade rather than a helpful shortcut.

    Returns True on success, False on any failure (missing files, an
    .srt with no parseable cues, or an existing cache without
    force_retranscribe) - every failure is logged, never raised.
    """
    video_path = Path(video_file)
    srt_path = Path(srt_file)
    if not video_path.exists():
        log.error("Import .srt: video file not found: %s", video_file)
        return False
    if not srt_path.exists():
        log.error("Import .srt: .srt file not found: %s", srt_file)
        return False

    cfg = build_run_config(overrides or {})
    db.validate_schema(cfg["db_path"])
    output_prefix = resolve_output_prefix(str(video_path), cfg)
    segments_cache = Path(output_prefix + "_segments.json")
    force = bool(cfg.get("force_retranscribe"))
    if not force and transcriber.transcript_cache_exists(output_prefix) and segments_cache.exists():
        log.error(
            "Import .srt: a transcript cache already exists for %s (%s) - pass "
            "--force-retranscribe (CLI) or enable it in the GUI to overwrite it "
            "with this .srt.", video_path.name, segments_cache.name)
        return False

    segments = transcriber.parse_srt(str(srt_path))
    if not segments:
        log.error("Import .srt: no parseable cues found in %s.", srt_file)
        return False

    if cfg.get("enable_diarization"):
        log.info(
            "Import .srt: diarization is enabled but has no effect here - an "
            "imported .srt carries no speaker labels (no audio pass is run). "
            "Diarize a real whisperx transcription instead if per-speaker "
            "labels are needed.")

    segments = _rescale_segments(segments, cfg.get("recording_speed", 1.0))
    transcriber.write_transcript_files(str(video_path), segments, output_prefix,
                                       recording_speed=cfg.get("recording_speed", 1.0))
    segments_cache.write_text(
        json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    Path(output_prefix + "_source_media.txt").write_text(str(video_path), encoding="utf-8")

    duration = segments[-1].get("end", 0) if segments else 0
    conn = db.get_connection(cfg["db_path"])
    db.upsert_transcript(
        conn, str(video_path), "imported-srt",
        cfg.get("whisper_language") or "auto", len(segments), duration,
    )
    conn.close()

    log.info("Import .srt: %d segment(s) imported from %s -> %s.",
             len(segments), srt_path.name, segments_cache.name)
    return True


def resolve_output_prefix(source_file: str, cfg: dict) -> str:
    """
    Return the "<dir>/<stem>" prefix used to name every output file.
    Honors output_dir_override (--output-dir / GUI field): when set, ALL
    outputs for this run go there instead of next to the source file.
    Honors output_basename_override (--output-name / GUI field): when
    set, ALL output files use this name instead of the source file's stem
    (e.g. "2026-07-02_Q3-Kickoff" instead of "Video_2026-07-02_100348").

    Otherwise, when use_date_subject_filename is on and meeting_title is
    set, the stem is "YYYYMMDD_Subject" (see build_date_subject_filename),
    de-duplicated against sibling files in the target directory (see
    _dedupe_date_subject_stem) in case another recording already used the
    same date/title.

    Otherwise, when use_filename_date_heading is on (default), the stem
    is derived from a date found in the source filename via
    derive_heading_from_filename - the same cleanup shown in the example
    above happens automatically, without needing --output-name. Falls
    back to the raw filename stem if no date is found, or if
    use_filename_date_heading is off.
    """
    override_stem = (cfg.get("output_basename_override") or "").strip()
    date_subject_stem = None
    if not override_stem and cfg.get("use_date_subject_filename", False):
        date_subject_stem = build_date_subject_filename(
            cfg.get("meeting_date", ""), cfg.get("meeting_title", ""), source_file)

    override = (cfg.get("output_dir_override") or "").strip()
    out_dir = Path(override) if override else Path(source_file).parent

    if override_stem:
        stem = override_stem
    elif date_subject_stem:
        stem = _dedupe_date_subject_stem(out_dir, date_subject_stem, source_file)
    elif cfg.get("use_filename_date_heading", True):
        stem = derive_heading_from_filename(Path(source_file).name) or Path(source_file).stem
    else:
        stem = Path(source_file).stem

    if override:
        out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / stem)


def _resolve_qa_autodetect(cfg: dict) -> bool:
    """
    Whether Q&A start auto-detection should run for this run.

    cfg["qa_autodetect_start"] is tri-state: True/False force it either way,
    and None (config's default, and what the GUI checkbox sends when ticked)
    means "the webinar prompt template only". Other recording types rarely
    have a formal Q&A block, and a false boundary in one would quietly cut
    part of the summary.
    """
    setting = cfg.get("qa_autodetect_start")
    if setting is None:
        return cfg.get("prompt_template") == "webinar"
    return bool(setting)


def _apply_qa_autodetect(cfg: dict, segments: list[dict]) -> None:
    """
    Fill in cfg["qa_start_time_sec"] from the transcript when it was not set
    by hand, so the rest of the Q&A pipeline (frame filter, summary split,
    the synthetic qa_session slide, generate_qa_pairs) works without the
    boundary having been timed manually.

    Mutates the run's local cfg only, never CONFIG or gui_state.json, so a
    detected value cannot leak into another batch row or become sticky.

    Deliberately never sets qa_end_time_sec: it reports a start only, so the
    Q&A runs to the end of the recording, which is how webinars actually
    work. That also keeps the excised frame range strictly trailing, well
    away from the mid-recording excision path in the slide detector.
    """
    if not _resolve_qa_autodetect(cfg) or not segments:
        return
    if cfg.get("qa_start_time_sec"):
        log.info("Q&A auto-detect skipped: Q&A start already set manually (%s).",
                 reporter.format_ts(cfg["qa_start_time_sec"]))
        return

    hit = notes.detect_qa_start(segments)
    if hit is None and cfg.get("llm_backend"):
        # Only on a miss, and only over the transcript tail rather than the
        # whole thing. Silently skipped when no backend answers.
        hit = notes.detect_qa_start_llm(
            segments,
            llm_backend=cfg["llm_backend"],
            ollama_base_url=cfg.get("ollama_base_url", ""),
            ollama_notes_model=cfg.get("ollama_notes_model", ""),
            anthropic_api_key=cfg.get("anthropic_api_key", ""),
            claude_model=cfg.get("claude_model", ""),
        )
    if not hit:
        log.info("Q&A auto-detect: no Q&A cue phrase found in the final 40%% of "
                 "the recording - no Q&A section set.")
        return

    detected, cue = hit
    cfg["qa_start_time_sec"] = detected
    snippet = next(
        ((s.get("text", "") or "").strip()[:120] for s in segments
         if (s.get("start", 0.0) or 0.0) == detected), "")
    log.info(
        "Q&A auto-detect: Q&A session appears to start at %s (%.1fs), cue \"%s\" "
        "in: \"%s\". Wrong? Set the Q&A section's \"Starts at\" field in the GUI "
        "(or --qa-start) to override, or untick auto-detect.",
        reporter.format_ts(detected), detected, cue, snippet)


def _unique_snapshot_path(snapshot_dir: Path, base_name: str, slide_idx: int,
                          suffix: str = ".png") -> Path:
    """
    Build "<base_name>_slideNNN<suffix>" under snapshot_dir. If that exact
    filename already exists (e.g. this output name was already used for
    a different source file in the same folder), append "_2", "_3", ...
    until a free filename is found, instead of overwriting.

    suffix defaults to ".png" for callers that predate frame_format support;
    process_file passes the staged frame's own extension, which is ".jpg"
    whenever config's frame_format is left at its default.
    """
    suffix = suffix or ".png"
    candidate = snapshot_dir / f"{base_name}_slide{slide_idx:03d}{suffix}"
    if not candidate.exists():
        return candidate
    n = 2
    while True:
        candidate = snapshot_dir / f"{base_name}_slide{slide_idx:03d}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def zip_snapshot_dir(snapshot_dir: Path, zip_path: Path) -> bool:
    """Zip every file directly inside snapshot_dir into zip_path. Returns
    True on success, False if snapshot_dir has no files to zip."""
    import zipfile
    files = [p for p in snapshot_dir.iterdir() if p.is_file()] if snapshot_dir.exists() else []
    if not files:
        return False
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)
    log.info("Snapshots zipped: %s", zip_path)
    return True


def _rescale_segments(segments: list[dict], speed: float) -> list[dict]:
    """
    Convert every timestamp in transcript segments (and their word-level
    alignments) from the recording's own internal clock to real
    wall-clock time: real_time = video_time / speed. Mutates and returns
    the same list. speed == 1.0 is a no-op.

    Applied once, immediately after a fresh whisperx transcription, before
    the segments are cached or written to any file (SRT, transcript .txt,
    JSON cache), so every downstream consumer sees consistent, already-
    converted times without needing its own conversion logic.
    """
    if speed == 1.0:
        return segments
    for seg in segments:
        seg["start"] = round(seg.get("start", 0.0) / speed, 3)
        seg["end"] = round(seg.get("end", 0.0) / speed, 3)
        for w in seg.get("words", []) or []:
            if "start" in w:
                w["start"] = round(w["start"] / speed, 3)
            if "end" in w:
                w["end"] = round(w["end"] / speed, 3)
    return segments


def _rescale_slide_changes(changes: list, speed: float) -> list:
    """
    Convert every detected slide's timestamp from the recording's own
    internal clock to real wall-clock time (see _rescale_segments).
    Mutates and returns the same list. speed == 1.0 is a no-op.
    """
    if speed == 1.0:
        return changes
    for c in changes:
        c.timestamp_sec = round(c.timestamp_sec / speed, 3)
    return changes


def _build_title_slide(cfg: dict, snapshot_dir: Path, stem: str) -> dict | None:
    """
    Copy the optional title-slide cover image (title_slide_image_path)
    into snapshot_dir and return the {"image_path", "title", "subtitle"}
    dict reporter.save_html/save_html_for_pdf expect, or None if not
    configured/found. Purely cosmetic: never written to CSV/JSON, never
    counted as a detected slide.
    """
    src = (cfg.get("title_slide_image_path") or "").strip()
    if not src:
        return None
    src_path = Path(src)
    if not src_path.exists():
        log.warning("Title slide image not found: %s", src_path)
        return None
    dest = snapshot_dir / f"{stem}_titleslide{src_path.suffix or '.png'}"
    try:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest)
    except Exception as exc:
        log.warning("Could not copy title slide image: %s", exc)
        return None
    return {
        "image_path": str(dest),
        "title": cfg.get("meeting_title") or "Title",
        "subtitle": cfg.get("meeting_date", ""),
    }


# ---------------------------------------------------------------------------
# Single-file processing
# ---------------------------------------------------------------------------

def process_file(file: str, overrides: dict | None = None, stop_check=None) -> bool:
    """
    Run the full pipeline on a single audio or video file.
    overrides: dict of CONFIG keys to override for this run (from CLI or GUI).
    stop_check: optional callable returning True once the user has requested
    a stop (see gui.py's Stop button, backed by a threading.Event). This is
    a COOPERATIVE stop, checked between pipeline stages and between VLM
    slide calls, not a hard kill: a stage already in progress (e.g. one
    whisperx transcription call, one Ollama request) always finishes first,
    so partial output is never corrupted. Returns True on success, False
    on failure or if stopped before any stage completed.
    """
    overrides = overrides or {}
    cfg = build_run_config(overrides)

    def stopped() -> bool:
        return stop_check is not None and stop_check()

    file = str(Path(file))
    if not Path(file).exists():
        log.error("File not found: %s", file)
        return False

    filename = Path(file).name
    output_prefix = resolve_output_prefix(file, cfg)
    audio_only = is_audio_only(file)
    enable_slides = cfg["enable_slides"] and not audio_only
    if cfg["enable_slides"] and audio_only:
        log.info("Audio-only file detected: slide detection disabled automatically.")

    log.info("=" * 60)
    log.info("PROCESSING: %s", filename)
    log.info("Mode: %s | Language: %s | Prompt: %s | Output: %s",
             "video+slides" if enable_slides else "audio", cfg["whisper_language"],
             cfg["prompt_template"], Path(output_prefix).parent)
    log.info("=" * 60)
    log.info("STAGE:start")

    db.validate_schema(cfg["db_path"])

    # -----------------------------------------------------------------
    # Step 1: Transcription (shared by both modes, cache-aware)
    # -----------------------------------------------------------------
    segments: list[dict] = []
    segments_cache = Path(output_prefix + "_segments.json")
    speakers_cache_exists = transcriber.transcript_cache_exists(output_prefix)

    if stopped():
        log.info("Stop requested before transcription - aborting run for %s.", filename)
        return False

    if cfg["enable_whisper"]:
        log.info("STAGE:transcribe")
        if not cfg["force_retranscribe"] and speakers_cache_exists and segments_cache.exists():
            log.info("Transcript cache found - loading %s (use --force-retranscribe to redo).",
                     segments_cache.name)
            segments = json.loads(segments_cache.read_text(encoding="utf-8"))
            if cfg.get("recording_speed", 1.0) != 1.0:
                log.info("Using cached transcript - recording_speed is only applied on a fresh "
                         "transcription. Use --force-retranscribe if you changed recording_speed "
                         "since this cache was created.")
        else:
            # Optional audio-cleanup pass (V1.24, off by default): only
            # ever applies to this transcription call, on a temporary
            # copy next to the segments cache. The original file is
            # never touched, and is still what get_speaker_suggestions/
            # Play Sample read from directly.
            transcribe_from = file
            enhanced_tmp = None
            if cfg.get("enhance_audio", False):
                enhanced_tmp = output_prefix + "_enhanced_tmp.wav"
                if transcriber.enhance_audio(file, enhanced_tmp):
                    transcribe_from = enhanced_tmp
                else:
                    enhanced_tmp = None  # nothing to clean up, fall back silently

            try:
                segments = transcriber.transcribe(
                    file_path=transcribe_from,
                    model_size=cfg["whisper_model"],
                    language=cfg["whisper_language"],
                    device=cfg["whisper_device"],
                    batch_size=cfg["whisper_batch_size"],
                    compute_type=cfg["whisper_compute_type"],
                    hf_token=cfg.get("hf_token"),
                    enable_diarization=cfg["enable_diarization"],
                    use_vocabulary=cfg["whisper_use_vocabulary"],
                    min_speakers=cfg.get("diarization_min_speakers"),
                    max_speakers=cfg.get("diarization_max_speakers"),
                )
            finally:
                if enhanced_tmp:
                    Path(enhanced_tmp).unlink(missing_ok=True)

            if not segments:
                log.error("Transcription returned no segments - aborting.")
                return False

            segments = _rescale_segments(segments, cfg.get("recording_speed", 1.0))
            transcriber.write_transcript_files(file, segments, output_prefix,
                                               recording_speed=cfg.get("recording_speed", 1.0))
            segments_cache.write_text(
                json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            duration = segments[-1].get("end", 0) if segments else 0
            conn = db.get_connection(cfg["db_path"])
            db.upsert_transcript(
                conn, file, cfg["whisper_model"],
                cfg["whisper_language"] or "auto", len(segments), duration,
            )
            conn.close()

        # Record the exact source path next to the segments cache (V1.19),
        # regardless of whether this was a cache hit or a fresh
        # transcription. This is the authoritative lookup find_source_media
        # uses first: output_prefix's stem may be a derived date/topic
        # heading (use_filename_date_heading) or an explicit
        # output_basename_override, neither of which reliably matches the
        # raw source filename, which previously broke source-media lookup
        # for the Rename Speakers dialog's voiceprint suggestions and Play
        # Sample button.
        Path(output_prefix + "_source_media.txt").write_text(file, encoding="utf-8")
    else:
        log.info("Transcription disabled (--no-whisper).")

    # -----------------------------------------------------------------
    # Step 1b: Q&A start auto-detection (webinar recordings)
    #
    # Deliberately OUTSIDE the enable_slides block below: an audio-only
    # webinar run has no slides but still needs qa_start_time_sec for the
    # summary split in Step 3. Placed here because segments are populated by
    # now on both paths (a fresh transcription and a cached one) and already
    # rescaled to real time, which is the clock qa_start_time_sec uses - so
    # the detected value must NOT be rescaled again.
    #
    # Written into this run's local cfg only, never back into CONFIG or
    # gui_state.json, so it cannot leak into another batch row or become
    # sticky for the next run.
    # -----------------------------------------------------------------
    _apply_qa_autodetect(cfg, segments)

    # -----------------------------------------------------------------
    # Step 2: Slide detection + VLM annotation (video mode only)
    # -----------------------------------------------------------------
    slides_annotated: list[dict] = []
    if enable_slides and stopped():
        log.info("Stop requested before slide detection - skipping slides for %s.", filename)
        enable_slides = False
    if enable_slides:
        log.info("STAGE:slides")
        import extractor
        import detector
        import annotator

        slides_dir = Path(output_prefix + "_slides")
        snapshot_dir = slides_dir / "snapshots"
        frames_dir = slides_dir / "_frames_tmp"
        slides_dir.mkdir(parents=True, exist_ok=True)

        try:
            frames = extractor.extract_frames(
                video_path=file, output_dir=frames_dir,
                fps=cfg["fps"], fmt=cfg["frame_format"], quality=cfg["frame_quality"],
            )
        except RuntimeError as exc:
            log.error("Frame extraction failed: %s", exc)
            frames = []

        qa_start = cfg.get("qa_start_time_sec")
        qa_end = cfg.get("qa_end_time_sec")
        # Each surviving frame's position in the ORIGINAL extracted sequence.
        # Handed to the detector below so timestamps stay tied to the real
        # video clock even when the Q&A filter removes frames from the middle
        # (list position alone would shift every later slide earlier by the
        # excised duration).
        frame_indices = list(range(len(frames)))
        if qa_start and frames:
            # Q&A section (task #61): drop frames in the Q&A range before
            # slide-change detection ever sees them, so no new slides can
            # be recorded there. Times are converted from the already-
            # converted real-time clock (matching qa_start_time_sec's
            # convention) back to the raw video clock frame_index/fps is
            # measured in, using the same recording_speed factor applied
            # elsewhere (see _rescale_slide_changes).
            speed = cfg.get("recording_speed", 1.0)
            qa_start_raw = qa_start * speed
            qa_end_raw = qa_end * speed if qa_end else None
            before_count = len(frames)
            kept = [
                (idx, f) for idx, f in enumerate(frames)
                if not (idx / cfg["fps"] >= qa_start_raw
                       and (qa_end_raw is None or idx / cfg["fps"] < qa_end_raw))
            ]
            frame_indices = [idx for idx, _ in kept]
            frames = [f for _, f in kept]
            skipped = before_count - len(frames)
            if skipped:
                log.info("Q&A range configured (from %.0fs%s): skipped %d frame(s), no new "
                         "slides will be detected there.", qa_start,
                         f" to {qa_end:.0f}s" if qa_end else " to end of recording", skipped)

        changes = detector.detect_slide_changes(
            frames=frames, fps=cfg["fps"], threshold=cfg["hash_threshold"],
            algorithm=cfg["hash_algorithm"], min_slide_duration_sec=cfg["min_slide_duration_sec"],
            animation_threshold=cfg.get("animation_threshold", 0),
            frame_indices=frame_indices,
        ) if frames else []
        changes = _rescale_slide_changes(changes, cfg.get("recording_speed", 1.0))

        if not changes:
            log.warning("No slide changes detected - skipping slide report.")
        elif cfg["dry_run"]:
            log.info("--- DRY RUN: slide timestamps ---")
            for c in changes:
                log.info("  t=%.1fs  frame=%d  dist=%d", c.timestamp_sec, c.frame_index, c.hamming_distance)
        else:
            # Snapshots are staged under a temp subfolder using the
            # ffmpeg-derived frame_NNNNNN name (annotator.annotate_batch
            # matches images to slides by that name). Once annotation is
            # done, each snapshot is moved into snapshot_dir under its
            # final, human-readable name and the temp copy is gone - the
            # public snapshots/ folder never contains a frame_*.png file.
            snap_tmp_dir = slides_dir / "_snap_tmp"
            snap_tmp_dir.mkdir(parents=True, exist_ok=True)
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            # Staged under the frame's own filename, extension included, so
            # no format conversion happens: the extracted frame is already a
            # finished image. Decoding each one and re-encoding it as PNG cost
            # 0.2-0.6s per slide and inflated it 5-10x, which then made both
            # the VLM payload and the Chromium PDF render heavier for no gain.
            # ".png" is not required anywhere - annotator.annotate_batch
            # matches by filename and the reports embed whatever path they are
            # given.
            for change in changes:
                dest = snap_tmp_dir / change.frame_path.name
                try:
                    shutil.copy2(change.frame_path, dest)
                except Exception as exc:
                    log.warning("Could not save snapshot %s: %s", dest, exc)

            if cfg["enable_vlm"]:
                log.info("Running VLM annotation with model: %s", cfg["ollama_vlm_model"])
                slides_annotated = annotator.annotate_batch(
                    slides=changes, snapshot_dir=snap_tmp_dir,
                    model=cfg["ollama_vlm_model"],
                    ollama_url=cfg["ollama_base_url"].rstrip("/") + "/api/generate",
                    prompt=annotator.build_prompt(cfg["vlm_prompt"], cfg.get("output_language", "auto")),
                    timeout_sec=cfg["vlm_timeout_sec"],
                    stop_check=stop_check,
                )
            else:
                slides_annotated = [
                    {
                        "frame_index": c.frame_index, "timestamp_sec": c.timestamp_sec,
                        "snapshot_path": str(snap_tmp_dir / c.frame_path.name),
                        "hash_value": c.hash_value, "hamming_distance": c.hamming_distance,
                        "title": "", "bullets": [], "slide_type": "",
                    }
                    for c in changes
                ]

            # Move each snapshot into snapshot_dir under a unique,
            # human-readable name: "<output basename>_slideNNN.png".
            # Base name matches whatever the rest of this run's output
            # files use (output_basename_override if set, else the video
            # stem), so a "copy image" from the HTML report gives a
            # meaningful filename instead of a cryptic frame_000123.png.
            # If that name is already taken (e.g. re-running the same
            # output name against a different source video into the same
            # folder), a numeric suffix is appended rather than silently
            # overwriting the existing file.
            base_name = Path(output_prefix).name
            for idx, slide in enumerate(slides_annotated, start=1):
                staged = Path(slide.get("snapshot_path", ""))
                if not staged.exists():
                    continue
                final_dest = _unique_snapshot_path(snapshot_dir, base_name, idx,
                                                   suffix=staged.suffix)
                try:
                    shutil.move(str(staged), str(final_dest))
                    slide["snapshot_path"] = str(final_dest)
                except Exception as exc:
                    log.warning("Could not finalize snapshot name for slide %d: %s", idx, exc)
            shutil.rmtree(snap_tmp_dir, ignore_errors=True)

            if segments:
                slide_ts = [s["timestamp_sec"] for s in slides_annotated]
                aligned = transcriber.align_transcript_to_slides(segments, slide_ts)
                speaker_map = transcriber.get_speaker_map(segments)
                for slide, text in zip(slides_annotated, aligned):
                    slide["transcript_seg"] = text
                    if speaker_map:
                        slide["speaker"] = _dominant_speaker(speaker_map, slide["timestamp_sec"])

            log.info("STAGE:db")
            conn = db.get_connection(cfg["db_path"])
            for slide in slides_annotated:
                db.insert_slide(conn, {
                    "video_path": file,
                    "timestamp_sec": slide["timestamp_sec"],
                    "snapshot_path": slide.get("snapshot_path", ""),
                    "hash_value": slide.get("hash_value", ""),
                    "title": slide.get("title", ""),
                    "bullets": json.dumps(slide.get("bullets", []), ensure_ascii=False),
                    "slide_type": slide.get("slide_type", ""),
                    "transcript_seg": slide.get("transcript_seg", ""),
                    "speaker": slide.get("speaker", ""),
                })
            db.insert_run_log(conn, file, "video", len(slides_annotated), "success")
            conn.close()

            stem = base_name  # matches the snapshot filenames set above
            m_title = cfg.get("meeting_title", "")
            m_date = cfg.get("meeting_date", "")
            m_comments = cfg.get("meeting_comments", "")
            if cfg["report_json"]:
                reporter.save_json(slides_annotated, slides_dir / f"{stem}_slides.json")
            # Q&A slide (task: separate Q&A slide + parsed pairs in the
            # report): must run AFTER _slides.json is saved above - that
            # file is the pristine raw-detection record Slide Review /
            # rebuild_outputs reads back later and must never contain this
            # synthetic entry (see _add_qa_slide's docstring). Everything
            # below this point (CSV/HTML/PDF/timing summary) uses the
            # now-extended list.
            if segments:
                _add_qa_slide(slides_annotated, segments, cfg)
            if cfg["report_csv"]:
                reporter.save_csv(slides_annotated, slides_dir / f"{stem}_slides.csv")
            show_image = cfg.get("report_show_image", True)
            show_bullets = cfg.get("report_show_bullets", True)
            show_transcript = cfg.get("report_show_transcript", True)
            transcript_mode = cfg.get("report_transcript_mode", "full")
            title_slide = _build_title_slide(cfg, snapshot_dir, stem)
            rec_speed = cfg.get("recording_speed", 1.0)
            if cfg["report_html"]:
                reporter.save_html(slides_annotated, slides_dir / f"{stem}_report.html", filename,
                                   meeting_title=m_title, meeting_date=m_date, meeting_comments=m_comments,
                                   show_image=show_image, show_bullets=show_bullets,
                                   show_transcript=show_transcript, transcript_mode=transcript_mode,
                                   title_slide=title_slide, recording_speed=rec_speed)
            if cfg["report_pdf"]:
                pdf_html = slides_dir / f"{stem}_report_pdf.html"
                reporter.save_html_for_pdf(slides_annotated, pdf_html, filename,
                                           meeting_title=m_title, meeting_date=m_date, meeting_comments=m_comments,
                                           show_image=show_image, show_bullets=show_bullets,
                                           show_transcript=show_transcript, transcript_mode=transcript_mode,
                                           title_slide=title_slide, recording_speed=rec_speed)
                reporter.save_pdf_from_html(pdf_html, slides_dir / f"{stem}_report.pdf")
            if cfg.get("report_slide_timing", True):
                reporter.save_slide_timing_summary(
                    slides_annotated, slides_dir / f"{stem}_slide_timing.txt", filename,
                    meeting_title=m_title, meeting_date=m_date, meeting_comments=m_comments,
                    recording_speed=rec_speed)
            if cfg.get("zip_snapshots", False):
                zip_snapshot_dir(snapshot_dir, slides_dir / f"{stem}_snapshots.zip")

        if frames_dir.exists():
            shutil.rmtree(frames_dir, ignore_errors=True)

    # -----------------------------------------------------------------
    # Step 2b: Real-time (1x) speed video conversion (optional, opt-in)
    # -----------------------------------------------------------------
    # Independent of enable_slides: even a plain audio/talking-head-style
    # video run may want the sped-up recording normalized back to real time.
    if not audio_only and not stopped() and cfg.get("convert_video_to_realtime", False):
        speed = cfg.get("recording_speed", 1.0)
        if speed == 1.0:
            log.info("convert_video_to_realtime is on, but recording_speed is 1.0 - "
                     "nothing to convert.")
        else:
            log.info("STAGE:normalize")
            import extractor
            realtime_path = Path(output_prefix + "_realtime.mp4")
            ok = extractor.convert_to_realtime_speed(file, realtime_path, speed)
            if ok:
                log.info("Real-time-speed video saved: %s (matches the already-converted "
                         ".srt/transcript timestamps, safe to play together).", realtime_path)
            else:
                log.error("Video speed normalization failed - see ffmpeg output above.")

    # -----------------------------------------------------------------
    # Step 3: Notes / summary generation
    # -----------------------------------------------------------------
    if stopped():
        log.info("Stop requested before notes generation - skipping notes for %s.", filename)
    elif cfg["enable_notes"] and not cfg.get("no_summary", False) and segments:
        try:
            prompt_path = notes.resolve_prompt_path(PROMPTS_DIR, cfg["prompt_template"])
        except FileNotFoundError as exc:
            log.error(str(exc))
            prompt_path = None

        if prompt_path:
            log.info("STAGE:notes")

            # Q&A section (task #61, single-prompt behavior task #112): a
            # Q&A block at the end of a webinar is, by default, kept in the
            # transcript and summarised by the SAME prompt/pass as the rest
            # of the recording (its content lands in the prompt's own
            # "## QUESTIONS & ANSWERS" section - see prompts/webinar.md),
            # rather than being run through a second prompt
            # (prompts/qa_summary.md) and appended afterwards. Set
            # cfg["qa_include_in_summary"] to False (GUI: "Include Q&A in
            # the summary" checkbox; CLI: --qa-exclude-from-summary) to drop
            # the Q&A portion from the summary entirely instead. Times use
            # the already-converted real-time clock, same as segments
            # (segments were rescaled by recording_speed back in Step 1).
            qa_start = cfg.get("qa_start_time_sec")
            qa_end = cfg.get("qa_end_time_sec")
            qa_include_in_summary = cfg.get("qa_include_in_summary", True)

            if qa_start and not qa_include_in_summary:
                main_segments = [s for s in segments if s.get("start", 0) < qa_start]
                if len(main_segments) == len(segments):
                    log.warning(
                        "qa_start_time_sec (%.0fs) is set but no transcript segments start "
                        "at/after it - check the value: it must be already-converted real "
                        "time in seconds, matching the transcript/slide report timestamps.",
                        qa_start)
                transcript_text = notes.build_transcript_text_from_segments(main_segments)
            elif enable_slides:
                # qa_start may be None here (nothing configured - plain
                # transcript, no marker) or set with qa_include_in_summary
                # True (folded in, WITH the marker so the model reliably
                # finds it - see build_transcript_text_with_qa_marker).
                transcript_text = notes.build_transcript_text_with_qa_marker(
                    segments, qa_start, qa_end)
            else:
                speakers_file = transcriber.transcript_cache_paths(output_prefix)["speakers"]
                transcript_text = transcriber.read_transcript_text(speakers_file)

            notes_text = notes.generate_notes(
                transcript_text=transcript_text,
                prompt_path=prompt_path,
                llm_backend=cfg["llm_backend"],
                ollama_base_url=cfg["ollama_base_url"],
                ollama_notes_model=cfg["ollama_notes_model"],
                anthropic_api_key=cfg["anthropic_api_key"],
                claude_model=cfg["claude_model"],
                filename=filename,
                single_pass_limit=cfg["single_pass_limit"],
                chunk_size=cfg["chunk_size"],
                meeting_title=cfg.get("meeting_title", ""),
                meeting_date=cfg.get("meeting_date", ""),
                meeting_comments=cfg.get("meeting_comments", ""),
                output_language=cfg.get("output_language", "auto"),
                ollama_num_ctx=cfg.get("ollama_notes_num_ctx", 16_384),
            )

            if notes_text:
                title = (cfg.get("meeting_title")
                        or derive_heading_from_filename(filename)
                        or (prompt_path.stem.upper() + " NOTES"))
                event_date = cfg.get("meeting_date", "")
                comments = cfg.get("meeting_comments", "")
                model_label = (cfg["ollama_notes_model"] if cfg["llm_backend"] == "ollama"
                              else cfg["claude_model"])

                notes_html_path = None
                if cfg.get("notes_format_txt", True):
                    reporter.save_notes_txt(notes_text, output_prefix + "_notes.txt",
                                            filename, cfg["llm_backend"], model_label,
                                            event_date=event_date, comments=comments)
                if cfg.get("notes_format_html", True):
                    notes_html_path = reporter.save_notes_html(
                        notes_text, output_prefix + "_notes.html", filename,
                        title=title, generated_by=cfg["llm_backend"], event_date=event_date,
                        comments=comments,
                    )
                if cfg.get("notes_format_docx", False):
                    reporter.save_notes_docx(
                        notes_text, output_prefix + "_notes.docx", filename,
                        title=title, generated_by=cfg["llm_backend"], event_date=event_date,
                        comments=comments,
                    )
                # Notes PDF: explicit toggle, or automatic when this was a
                # video-with-slides run and slide-report PDFs are enabled
                # (matches the original Video_Transkription_Notes_Pipeline
                # behaviour of always producing a summary PDF alongside HTML).
                want_pdf = cfg.get("notes_format_pdf", False) or (enable_slides and cfg["report_pdf"])
                if want_pdf:
                    if notes_html_path is None:
                        notes_html_path = reporter.save_notes_html(
                            notes_text, output_prefix + "_notes.html", filename,
                            title=title, generated_by=cfg["llm_backend"], event_date=event_date,
                            comments=comments,
                        )
                    reporter.save_pdf_from_html(notes_html_path, Path(output_prefix + "_notes.pdf"))

                conn = db.get_connection(cfg["db_path"])
                db.upsert_transcript(
                    conn, file, cfg["whisper_model"], cfg["whisper_language"] or "auto",
                    len(segments), segments[-1].get("end", 0) if segments else 0,
                    notes_generated=True, notes_backend=cfg["llm_backend"],
                    prompt_template=Path(prompt_path).name, notes_text=notes_text,
                )
                conn.close()
    elif cfg.get("no_summary"):
        log.info("Notes generation disabled (--no-summary).")

    if cfg.get("move_processed_files", False) and not cfg.get("dry_run", False):
        log.info("STAGE:archive")
        move_processed_source_file(file, output_prefix, cfg)

    log.info("STAGE:done")
    log.info("DONE: %s", filename)
    return True


def move_processed_source_file(file: str, output_prefix: str, cfg: dict) -> str:
    """
    Move the original source audio/video file into processed_subfolder_name
    next to it (CONFIG["move_processed_files"], off by default). Returns
    the file's final path: the moved location on success, or the original
    file unchanged if the option is off, the destination already has a
    same-named file (never overwrites - logs a warning and leaves the
    source in place instead), or the move itself fails for any reason
    (e.g. the NAS share drops mid-move) - archiving is a nice-to-have,
    never worth losing track of a just-processed recording over.

    Keeps everything that can reference the source file consistent with
    the new location: rewrites the "<output_prefix>_source_media.txt"
    sidecar (find_source_media's first, authoritative lookup - used by
    the GUI's Play Sample button and the Rename Speakers dialog's
    voiceprint suggestions) and, via db.update_source_path, the
    transcripts/slides DB rows. Callers should use the RETURNED path for
    anything after this point in the same run, not the original file
    argument, in case a move happened.
    """
    if not cfg.get("move_processed_files", False):
        return file

    subfolder = (cfg.get("processed_subfolder_name") or "_processed").strip() or "_processed"
    source = Path(file)
    dest_dir = source.parent / subfolder
    dest_path = dest_dir / source.name

    if dest_path.exists():
        log.warning("Archiving skipped: %s already exists in %s - leaving source file in place.",
                   source.name, dest_dir)
        return file

    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(dest_path))
    except OSError as exc:
        log.warning("Archiving failed (%s) - leaving source file in place: %s", exc, source)
        return file

    log.info("Archived source file to: %s", dest_path)

    sidecar = Path(output_prefix + "_source_media.txt")
    if sidecar.exists():
        sidecar.write_text(str(dest_path), encoding="utf-8")

    try:
        conn = db.get_connection(cfg["db_path"])
        db.update_source_path(conn, str(source), str(dest_path))
        conn.close()
    except Exception as exc:
        log.warning("DB source-path update after archiving failed (non-fatal): %s", exc)

    return str(dest_path)


def _dominant_speaker(speaker_map: dict, timestamp: float) -> str:
    best, best_dist = "", float("inf")
    for spk, times in speaker_map.items():
        for t in times:
            d = abs(t - timestamp)
            if d < best_dist:
                best_dist = d
                best = spk
    return best


def _add_qa_slide(slides: list[dict], segments: list[dict], cfg: dict) -> None:
    """
    Mutates `slides` in place (rewrites the LAST slide's transcript_seg,
    then appends one new entry): if qa_start_time_sec is configured,
    splits the Q&A portion out of the last detected slide - whose
    display window otherwise runs to the end of the recording (frames in
    the Q&A range are dropped before slide-change detection ever sees
    them, so no new slide can be detected there - see process_file) -
    into its own synthetic "Q&A Session" entry, complete with (LLM
    permitting) discrete Q: / A: pairs extracted via
    notes.generate_qa_pairs, instead of leaving the whole live Q&A
    discussion glued onto the last real slide's transcript as one
    continuous, unstructured block.

    Used by process_file (fresh run, called AFTER _slides.json is saved
    - see that call site's comment), reannotate_failed_slides, and
    rebuild_outputs (Slide Review's "Rebuild outputs", loading segments
    from the cached *_segments.json). No-op if qa_start_time_sec isn't
    set, there are no cached segments to re-derive timing from, or there
    are no slides to attach the split to.

    Only the last slide's transcript_seg is touched; every earlier slide
    keeps whatever text it already had (from process_file's alignment or
    a prior apply_slide_edits merge) rather than being recomputed from
    scratch, so an unrelated edit elsewhere in the slide list can't
    change earlier slides' transcripts as a side effect of this.
    """
    qa_start = cfg.get("qa_start_time_sec")
    qa_end = cfg.get("qa_end_time_sec")
    if not qa_start or not segments or not slides:
        return
    if slides[-1].get("slide_type") == "qa_session":
        return  # already split (e.g. called twice on the same in-memory list)

    last_ts = slides[-1].get("timestamp_sec", 0)
    boundaries = [last_ts, qa_start] + ([qa_end] if qa_end else [])
    windows = transcriber.align_transcript_to_slides(segments, boundaries)
    if windows:
        slides[-1]["transcript_seg"] = windows[0]
    qa_text = windows[1] if len(windows) > 1 else ""
    if not qa_text.strip():
        log.warning(
            "qa_start_time_sec (%.0fs) is set but no transcript segments fall in "
            "that window - no Q&A slide added.", qa_start)
        return

    qa_pairs = notes.generate_qa_pairs(
        qa_text, llm_backend=cfg["llm_backend"], ollama_base_url=cfg["ollama_base_url"],
        ollama_notes_model=cfg["ollama_notes_model"], anthropic_api_key=cfg["anthropic_api_key"],
        claude_model=cfg["claude_model"], output_language=cfg.get("output_language", "auto"),
        ollama_num_ctx=cfg.get("ollama_notes_num_ctx", 16_384),
    )
    slides.append({
        "frame_index": -1, "timestamp_sec": qa_start, "snapshot_path": "",
        "hash_value": "", "hamming_distance": 0,
        "title": "Q&A Session", "bullets": [], "slide_type": "qa_session",
        "transcript_seg": qa_text, "qa_pairs": qa_pairs, "speaker": "",
    })
    log.info("Q&A slide added (from %.0fs%s): %d pair(s) extracted.",
             qa_start, f" to {qa_end:.0f}s" if qa_end else " to end of recording", len(qa_pairs))


def reannotate_failed_slides(slides_json_path: str, overrides: dict | None = None,
                             stop_check=None) -> tuple[int, int]:
    """
    Re-run VLM annotation only for slides whose title AND bullets are both
    empty (the signature of a VLM JSON parse failure, see
    annotator._safe_parse_json), using the snapshot images already on disk.
    Does not re-run frame extraction, slide-change detection, or
    transcription. Updates the slide JSON in place, regenerates
    CSV/HTML/PDF/timing reports, and updates the matching rows in the
    database via db.update_slide_annotation (matched on video_path +
    timestamp_sec, so a retry never creates duplicate DB rows).
    Returns (retried_count, still_failed_count).
    """
    overrides = overrides or {}
    cfg = build_run_config(overrides)
    slides_json_path = Path(slides_json_path)
    if not slides_json_path.exists():
        log.error("Slides JSON not found: %s", slides_json_path)
        return (0, 0)

    slides = json.loads(slides_json_path.read_text(encoding="utf-8"))
    failed_idx = [i for i, s in enumerate(slides) if not s.get("title") and not s.get("bullets")]
    if not failed_idx:
        log.info("No failed slides found in %s - nothing to re-annotate.", slides_json_path.name)
        return (0, 0)

    log.info("STAGE:slides")
    log.info("Re-annotating %d failed slide(s) of %d total, model=%s.",
             len(failed_idx), len(slides), cfg["ollama_vlm_model"])

    import annotator
    retried, still_failed = 0, 0
    for count, idx in enumerate(failed_idx, 1):
        if stop_check is not None and stop_check():
            log.info("Stop requested, re-annotation halted at %d/%d.", count, len(failed_idx))
            break
        slide = slides[idx]
        snap = Path(slide.get("snapshot_path", ""))
        if not snap.exists():
            log.warning("Snapshot missing for slide %d (%s) - cannot re-annotate.", idx + 1, snap)
            still_failed += 1
            continue
        annotation = annotator.annotate_slide(
            snap, model=cfg["ollama_vlm_model"],
            ollama_url=cfg["ollama_base_url"].rstrip("/") + "/api/generate",
            prompt=annotator.build_prompt(cfg["vlm_prompt"], cfg.get("output_language", "auto")),
            timeout_sec=cfg["vlm_timeout_sec"],
        )
        if annotation.get("title") or annotation.get("bullets"):
            slide["title"] = annotation.get("title", "")
            slide["bullets"] = annotation.get("bullets", [])
            slide["slide_type"] = annotation.get("slide_type", "")
            retried += 1
        else:
            still_failed += 1

    slides_json_path.write_text(json.dumps(slides, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Updated JSON: %s (retried=%d, still_failed=%d)", slides_json_path.name, retried, still_failed)

    # Derive the original video and output stem from the slides folder
    # naming convention (<stem>_slides/<stem>_slides.json), so CSV/HTML/
    # PDF/timing reports and the DB update use the same paths the original
    # run would have used.
    slides_dir = slides_json_path.parent
    stem = slides_json_path.stem.replace("_slides", "")

    segments_json_path = slides_dir.parent / f"{stem}_segments.json"
    qa_segments = []
    if segments_json_path.exists():
        try:
            qa_segments = json.loads(segments_json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Could not read %s for Q&A slide rebuild: %s", segments_json_path.name, exc)
    _add_qa_slide(slides, qa_segments, cfg)

    video_path = None
    for candidate in slides_dir.parent.glob(f"{stem}.*"):
        if candidate.suffix.lower() in {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}:
            video_path = str(candidate)
            break
    video_name = Path(video_path).name if video_path else stem

    if retried and video_path:
        log.info("STAGE:db")
        db.validate_schema(cfg["db_path"])
        conn = db.get_connection(cfg["db_path"])
        for idx in failed_idx:
            slide = slides[idx]
            if not (slide.get("title") or slide.get("bullets")):
                continue
            db.update_slide_annotation(
                conn, video_path, slide["timestamp_sec"],
                slide.get("title", ""), json.dumps(slide.get("bullets", []), ensure_ascii=False),
                slide.get("slide_type", ""),
            )
        conn.close()
    elif retried:
        log.warning("Could not locate source video next to %s - DB not updated "
                    "(JSON/reports were still refreshed).", slides_dir)

    m_title = cfg.get("meeting_title", "")
    m_date = cfg.get("meeting_date", "")
    m_comments = cfg.get("meeting_comments", "")
    show_image = cfg.get("report_show_image", True)
    show_bullets = cfg.get("report_show_bullets", True)
    show_transcript = cfg.get("report_show_transcript", True)
    transcript_mode = cfg.get("report_transcript_mode", "full")
    title_slide = _build_title_slide(cfg, slides_dir / "snapshots", stem)
    rec_speed = cfg.get("recording_speed", 1.0)

    if cfg["report_csv"]:
        reporter.save_csv(slides, slides_dir / f"{stem}_slides.csv")
    if cfg["report_html"]:
        reporter.save_html(slides, slides_dir / f"{stem}_report.html", video_name,
                           meeting_title=m_title, meeting_date=m_date, meeting_comments=m_comments,
                           show_image=show_image, show_bullets=show_bullets,
                           show_transcript=show_transcript, transcript_mode=transcript_mode,
                           title_slide=title_slide, recording_speed=rec_speed)
    if cfg["report_pdf"]:
        pdf_html = slides_dir / f"{stem}_report_pdf.html"
        reporter.save_html_for_pdf(slides, pdf_html, video_name,
                                   meeting_title=m_title, meeting_date=m_date, meeting_comments=m_comments,
                                   show_image=show_image, show_bullets=show_bullets,
                                   show_transcript=show_transcript, transcript_mode=transcript_mode,
                                   title_slide=title_slide, recording_speed=rec_speed)
        reporter.save_pdf_from_html(pdf_html, slides_dir / f"{stem}_report.pdf")
    if cfg.get("report_slide_timing", True):
        reporter.save_slide_timing_summary(slides, slides_dir / f"{stem}_slide_timing.txt", video_name,
                                           meeting_title=m_title, meeting_date=m_date,
                                           meeting_comments=m_comments, recording_speed=rec_speed)

    log.info("STAGE:done")
    log.info("Re-annotation done: %d fixed, %d still failed.", retried, still_failed)
    return (retried, still_failed)


# ---------------------------------------------------------------------------
# Post-hoc speaker renaming (task #76): rename SPEAKER_00/SPEAKER_01/...
# diarization labels to real names after the fact, using the cached
# *_segments.json as the source of truth. Used by gui.py's
# SpeakerRenameDialog.
# ---------------------------------------------------------------------------

def get_speaker_labels(segments_json_path: str) -> list[str]:
    """
    Return the sorted distinct speaker labels found in an existing
    *_segments.json cache (written by process_file's transcription stage),
    or [] if the file is missing, unreadable, or has no speaker field on
    any segment (e.g. diarization was off for that run).
    """
    path = Path(segments_json_path)
    if not path.exists():
        return []
    try:
        segments = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    return sorted({s.get("speaker") for s in segments if s.get("speaker")})


def rename_speakers(segments_json_path: str, speaker_mapping: dict,
                    overrides: dict | None = None, regenerate_notes: bool = False) -> dict:
    """
    Rename speaker labels in an existing transcript (task #76), using the
    cached *_segments.json (produced by process_file's transcription
    stage) as the source of truth for the word-level segment data. Does
    NOT re-run whisperx, alignment, or diarization.

    speaker_mapping: {old_label: new_label}. Entries with a blank/falsy
    new_label, or where new_label equals old_label, are no-ops. A label
    not present in the transcript is silently ignored.

    Rewrites *_segments.json in place (so the rename survives a later
    cache-aware run that reloads it, e.g. a --no-summary re-run), then
    regenerates *_transcript_speakers.txt, *_text.txt, and
    *_transcript.srt from the renamed segments via
    transcriber.write_transcript_files. If regenerate_notes is True and
    at least one segment was renamed, also regenerates *_notes.* by
    calling run_notes_only on the freshly rewritten
    *_transcript_speakers.txt (notes generation failure there is logged
    as a warning, not raised, since the rename itself already succeeded).

    Returns {"renamed_segments": <count>, "labels_found": [...],
    "speakers_path": <path>, "notes_regenerated": <bool, only present if
    regenerate_notes was requested>}.

    Raises FileNotFoundError if segments_json_path does not exist: a
    rename needs the word-level segment cache, and there is no reliable
    way to recover it by re-parsing the already-formatted .txt output
    (word-level timestamps would be lost).
    """
    path = Path(segments_json_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No segment cache found: {path}. Speaker renaming needs the "
            f"*_segments.json produced during transcription; re-run the "
            f"pipeline once (with --force-retranscribe if it already ran "
            f"without this cache) to create it."
        )
    segments = json.loads(path.read_text(encoding="utf-8"))
    labels_found = sorted({s.get("speaker") for s in segments if s.get("speaker")})
    mapping = {k: v for k, v in (speaker_mapping or {}).items() if v and v != k}

    renamed_count = 0
    for seg in segments:
        spk = seg.get("speaker")
        if spk and spk in mapping:
            seg["speaker"] = mapping[spk]
            renamed_count += 1
    path.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")

    # Derive output_prefix and a display name for the transcript header
    # from the segments.json naming convention (<prefix>_segments.json),
    # matching how reannotate_failed_slides locates its source video.
    output_prefix = str(path.parent / path.name.replace("_segments.json", ""))
    source_label = Path(output_prefix).name
    for candidate in path.parent.glob(f"{source_label}.*"):
        if candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
            source_label = str(candidate)
            break

    cfg = build_run_config(overrides or {})
    paths = transcriber.write_transcript_files(
        source_label, segments, output_prefix,
        recording_speed=cfg.get("recording_speed", 1.0),
    )
    log.info("Speaker rename complete: %d segment(s) updated. Labels: %s",
             renamed_count, labels_found)

    result = {
        "renamed_segments": renamed_count,
        "labels_found": labels_found,
        "speakers_path": paths["speakers"],
    }
    if regenerate_notes:
        if renamed_count:
            try:
                run_notes_only(paths["speakers"], overrides or {})
                result["notes_regenerated"] = True
            except Exception as exc:
                log.warning("Notes regeneration after speaker rename failed: %s", exc)
                result["notes_regenerated"] = False
        else:
            log.info("No segments were renamed - skipping notes regeneration.")
            result["notes_regenerated"] = False
    return result


# ---------------------------------------------------------------------------
# Speaker identification (task #80): cross-meeting voiceprints on top of
# the post-hoc rename utility above. This does NOT run automatically as
# part of process_file - it is opt-in and triggered when the user opens
# the Rename Speakers dialog with enable_speaker_id on (see gui.py
# SpeakerRenameDialog), matching the review-first workflow: compute
# suggestions, let the user confirm/correct each name (with an audio
# preview - see speaker_id.py), then commit. rename_speakers above still
# does the actual rewriting of the local transcript; the two functions
# below only manage the global known_speakers roster.
# ---------------------------------------------------------------------------

def find_source_media(segments_json_path: str) -> str | None:
    """
    Locate the original audio/video file next to a *_segments.json cache.

    Tries, in order:
      1. The "<prefix>_source_media.txt" sidecar written by process_file
         (V1.19) - the authoritative record of the exact source path used
         for that run, immune to any output-naming scheme, including
         output_basename_override, where the stem carries no relationship
         to the source filename at all.
      2. A same-stem match next to the segments.json (the pre-V1.17
         behaviour: <stem>.<supported extension> in the same folder).
         Still correct whenever use_filename_date_heading is off, or for
         runs that predate the sidecar file.
      3. A reverse-derived match: for each supported media file in the
         folder, recompute the date-derived heading
         (derive_heading_from_filename) that resolve_output_prefix would
         have used for it, and compare against the segments.json stem.
         Covers files already processed under V1.17/V1.18's
         use_filename_date_heading default (True), before this sidecar
         existed, where the raw filename no longer matches the output
         prefix - this was a real regression: it silently broke the
         Rename Speakers dialog's voiceprint suggestions and Play Sample
         button, since both are gated on this lookup succeeding.

    Returns None if none of the above find a match (e.g. the source media
    was moved or deleted after transcription) - rename_speakers still
    works without it since it only needs the segment cache, but
    get_speaker_suggestions/the Play Sample button need to re-read the
    actual audio.
    """
    path = Path(segments_json_path)
    stem = path.name.replace("_segments.json", "")

    sidecar = path.parent / f"{stem}_source_media.txt"
    if sidecar.exists():
        recorded = sidecar.read_text(encoding="utf-8").strip()
        if recorded and Path(recorded).exists():
            return recorded

    for candidate in path.parent.glob(f"{stem}.*"):
        if candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
            return str(candidate)

    for candidate in path.parent.iterdir():
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
            derived = derive_heading_from_filename(candidate.name)
            if derived and derived == stem:
                return str(candidate)

    return None


def get_speaker_suggestions(segments_json_path: str, source_media_path: str,
                             hf_token: str, threshold: float, db_path: str,
                             device: str = "cpu") -> dict:
    """
    For an existing *_segments.json cache, compute one voiceprint per
    local speaker label and match it against the global known_speakers
    roster (db.py).

    Returns {label: {"suggested_name": str|None, "score": float,
    "embedding": np.ndarray}}. "embedding" is included so a later call to
    commit_speaker_identities can reuse it without recomputing (which
    would mean reloading pyannote and re-decoding the source audio a
    second time). Returns {} if the segments file is missing, has no
    speaker labels at all (diarization was off for that run), or
    source_media_path can't be found alongside it.

    Raises RuntimeError if pyannote.audio itself can't be loaded
    (propagated from speaker_id.extract_speaker_embeddings, e.g. the
    gated model's terms haven't been accepted on HuggingFace yet) - the
    caller should surface this to the user rather than silently
    returning {}, since it means the whole feature is unusable until
    fixed, not just this one file.
    """
    path = Path(segments_json_path)
    if not path.exists():
        return {}
    segments = json.loads(path.read_text(encoding="utf-8"))
    labels = sorted({s.get("speaker") for s in segments if s.get("speaker")})
    if not labels:
        return {}
    if not Path(source_media_path).exists():
        raise FileNotFoundError(
            f"Source media not found: {source_media_path}. Speaker "
            f"identification needs to re-read the original audio; plain "
            f"renaming (get_speaker_labels/rename_speakers) still works "
            f"without it."
        )

    import whisperx  # already a hard dependency of this whole pipeline
    audio = whisperx.load_audio(str(source_media_path))
    embeddings = speaker_id.extract_speaker_embeddings(
        audio, segments, hf_token, device=device)

    conn = db.get_connection(db_path)
    try:
        known = db.list_known_speakers(conn)
    finally:
        conn.close()

    suggestions = {}
    for label, emb in embeddings.items():
        match = speaker_id.match_speaker(emb, known, threshold)
        suggestions[label] = {
            "suggested_name": match["name"],
            "score": match["score"],
            "embedding": emb,
        }
    return suggestions


def commit_speaker_identities(db_path: str, suggestions: dict, name_choices: dict) -> dict:
    """
    After the user reviews get_speaker_suggestions' output and picks a
    final name per label (name_choices: {label: name_or_blank_to_skip}),
    persist each non-blank choice into the known_speakers roster:
      - name matches an existing known speaker (exact name match) ->
        running-average update of that speaker's embedding, sample_count+1.
      - name is new -> enroll a brand-new known_speakers row.

    Returns {label: {"speaker_id": int, "action": "matched"|"enrolled"}}
    for labels that were committed; blank/skipped labels, and labels not
    present in suggestions (e.g. embedding extraction failed for that
    speaker), are omitted from the result.

    Does NOT touch the transcript/segment files themselves - call
    rename_speakers separately with the same label->name mapping to
    actually rewrite the local transcript, exactly like the existing
    Rename Speakers workflow (task #76).
    """
    conn = db.get_connection(db_path)
    result = {}
    try:
        for label, name in (name_choices or {}).items():
            name = (name or "").strip()
            if not name or label not in suggestions:
                continue
            embedding = suggestions[label]["embedding"]
            existing = db.get_known_speaker_by_name(conn, name)
            if existing:
                new_vec, new_count = speaker_id.update_running_average(
                    speaker_id.deserialize_embedding(
                        existing["embedding"], existing["embedding_dim"]),
                    existing["sample_count"], embedding,
                )
                db.update_known_speaker_embedding(
                    conn, existing["speaker_id"],
                    speaker_id.serialize_embedding(new_vec), new_count,
                )
                result[label] = {"speaker_id": existing["speaker_id"], "action": "matched"}
            else:
                dim = embedding.shape[0]
                new_id = db.insert_known_speaker(
                    conn, name, speaker_id.serialize_embedding(embedding), dim,
                )
                result[label] = {"speaker_id": new_id, "action": "enrolled"}
    finally:
        conn.close()
    return result


# ---------------------------------------------------------------------------
# Batch speaker roster export/apply (task #92): after an overnight batch of
# several recordings, review every detected speaker across ALL files in one
# JSON file instead of opening each file's Rename Speakers dialog one at a
# time. export_speaker_roster_json scans a folder for *_segments.json,
# computes voiceprint suggestions per file (best-effort - a failure on one
# file logs a warning and still includes that file with blank suggestions,
# rather than aborting the whole export), and writes one combined JSON.
# apply_speaker_roster_json reads the same file back (after the user has
# edited the "name" fields) and applies the renames + known_speakers roster
# updates across every file in one pass.
# ---------------------------------------------------------------------------

def export_speaker_roster_json(folder: str, output_json_path: str,
                                hf_token: str, threshold: float, db_path: str,
                                device: str = "cpu", recursive: bool = False) -> dict:
    """
    Scan folder (optionally recursive, same discovery rule as
    run_batch_folder) for every *_segments.json produced by a previous
    run, compute voiceprint suggestions for each local speaker label
    against the known_speakers roster, and write ONE combined JSON file
    with every file's speakers side by side - built for reviewing an
    overnight batch of several recordings in one sitting instead of
    opening each file's Rename Speakers dialog individually.

    Each speaker entry's "name" field is pre-filled with its suggested
    match (if any cleared threshold), so the user only has to correct
    the wrong ones and fill in genuinely new speakers; leaving "name"
    blank (or unchanged and equal to "label") means "skip this one" when
    the file is later applied with apply_speaker_roster_json.

    Alongside the JSON, also extracts one short sample clip (.wav, best
    -effort, via speaker_id.extract_speaker_sample_clip, same helper the
    single-file Rename Speakers dialog's "Play sample" button uses) per
    speaker into a "<json stem>_samples" folder next to output_json_path,
    named "<recording stem>__<label>.wav" - so a batch of 232 speakers
    across 47 files can be listened to (double-click each .wav in
    Explorer) while filling in "name" fields, instead of having to reopen
    every file's dialog just to hear a voice. A speaker's "sample_clip"
    field is that path if extraction succeeded, else null (no source
    media found, ffmpeg missing, or the clip failed) - a missing preview
    must never abort the export.

    A missing source recording, or a pyannote/voiceprint failure, for
    one file is logged and that file is still included with blank
    suggestions - one bad file must never lose the rest of the batch's
    entries. Once pyannote.audio itself turns out to be unusable at all
    (e.g. gated model terms not accepted), the same failure would just
    repeat for every remaining file, so that case stops trying voiceprint
    suggestions for the rest of this export and adds one top-level
    "warning" field to the output instead of one warning per file.

    Returns {"path": output_json_path, "files": <count>, "speakers": <count>,
    "samples": <count>, "samples_dir": <path>}.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")

    pattern_fn = folder_path.rglob if recursive else folder_path.glob
    segment_files = sorted(pattern_fn("*_segments.json"))

    samples_dir = Path(output_json_path).parent / f"{Path(output_json_path).stem}_samples"
    ffmpeg_available = speaker_id.check_ffmpeg()
    if not ffmpeg_available:
        log.warning("ffmpeg not found (bundled ffmpeg\\bin\\ or PATH) - speaker sample "
                    "clips will not be extracted for this export; speakers must be "
                    "named from labels/transcript context alone.")

    pyannote_unavailable: str | None = None
    file_entries = []
    total_speakers = 0
    total_samples = 0

    for segments_json in segment_files:
        try:
            segments = json.loads(segments_json.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("Could not read %s: %s - skipping.", segments_json, exc)
            continue
        labels = sorted({s.get("speaker") for s in segments if s.get("speaker")})
        if not labels:
            continue

        source_media = find_source_media(str(segments_json))
        suggestions: dict = {}
        if pyannote_unavailable is None and source_media and hf_token:
            try:
                suggestions = get_speaker_suggestions(
                    str(segments_json), source_media, hf_token, threshold, db_path, device=device)
            except FileNotFoundError as exc:
                log.warning("%s", exc)
            except RuntimeError as exc:
                pyannote_unavailable = str(exc)
                log.warning("Voiceprint suggestions unavailable for the rest of this "
                            "export: %s", exc)

        recording_stem = segments_json.stem.replace("_segments", "")
        speakers = []
        for label in labels:
            info = suggestions.get(label) or {}
            suggested_name = info.get("suggested_name")
            entry = {
                "label": label,
                "suggested_name": suggested_name,
                "score": round(info["score"], 3) if info.get("score") is not None else None,
                "name": suggested_name or "",
                "sample_clip": None,
            }
            if "embedding" in info:
                entry["embedding_b64"] = base64.b64encode(
                    speaker_id.serialize_embedding(info["embedding"])).decode("ascii")
                entry["embedding_dim"] = int(info["embedding"].shape[0])
            if ffmpeg_available and source_media:
                clip_path = samples_dir / f"{recording_stem}__{label}.wav"
                try:
                    samples_dir.mkdir(parents=True, exist_ok=True)
                    result_path = speaker_id.extract_speaker_sample_clip(
                        source_media, segments, label, str(clip_path),
                        max_duration=CONFIG.get("speaker_sample_seconds", 15.0))
                    if result_path:
                        entry["sample_clip"] = result_path
                        total_samples += 1
                except Exception as exc:
                    log.warning("Could not extract sample clip for %s in %s: %s",
                                label, segments_json, exc)
            speakers.append(entry)
        total_speakers += len(speakers)

        file_entries.append({
            "output_prefix": str(segments_json).replace("_segments.json", ""),
            "segments_json": str(segments_json),
            "source_media": source_media,
            "speakers": speakers,
        })

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "db_path": db_path,
        "note": "Edit each speaker's \"name\" field, then apply with "
                "apply_speaker_roster_json (GUI: Settings > Apply speaker "
                "roster JSON...). Leave \"name\" blank, or equal to \"label\", "
                "to skip a speaker. Where present, \"sample_clip\" is a short "
                ".wav in the samples folder next to this JSON: play it to hear "
                "that voice before naming them.",
        "files": file_entries,
    }
    if pyannote_unavailable:
        payload["warning"] = (
            "Voiceprint suggestions could not be computed for some or all "
            f"files: {pyannote_unavailable}. Speakers are still listed "
            "below with blank suggestions - fill in \"name\" manually."
        )

    Path(output_json_path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Speaker roster exported: %d file(s), %d speaker(s), %d sample clip(s) -> %s "
             "(samples: %s)",
             len(file_entries), total_speakers, total_samples, output_json_path, samples_dir)
    return {"path": output_json_path, "files": len(file_entries), "speakers": total_speakers,
            "samples": total_samples, "samples_dir": str(samples_dir)}


def apply_speaker_roster_json(json_path: str, overrides: dict | None = None,
                               regenerate_notes: bool = False) -> dict:
    """
    Read back a JSON file produced by export_speaker_roster_json (after
    the user has edited each speaker's "name" field) and, for every file
    listed, rename its speaker labels (rename_speakers) and commit any
    named speaker to the known_speakers roster (commit_speaker_identities)
    - the same two steps the Rename Speakers dialog's Apply button runs
    for a single file, done here for every file in the batch in one call.

    A speaker is applied only if its "name" is non-blank and differs
    from "label" (matching rename_speakers/commit_speaker_identities'
    own no-op rule) - this is what makes leaving "name" blank, or
    unedited and equal to "label", mean "skip this speaker".

    Roster commits reuse the embedding captured at export time
    (embedding_b64/embedding_dim), so applying does NOT need
    pyannote.audio, a HuggingFace token, or the source recording to be
    reachable again - only rename_speakers' *_segments.json is touched
    for that part. Entries with no embedding_b64 (suggestions were
    unavailable at export time) still get the transcript rename, just no
    roster commit for that one speaker.

    One file's error (e.g. its *_segments.json was moved or deleted
    since export) is logged and skipped rather than aborting the rest of
    the batch. Returns {"files_updated": <count>, "speakers_renamed":
    <count>, "roster_updates": <count>, "errors": [{"file":..., "error":...}]}.
    """
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    overrides = overrides or {}
    db_path = payload.get("db_path") or CONFIG["db_path"]

    files_updated = 0
    speakers_renamed = 0
    roster_updates = 0
    errors = []

    for entry in payload.get("files", []):
        segments_json = entry.get("segments_json")
        try:
            mapping = {}
            suggestions = {}
            for spk in entry.get("speakers", []):
                label = spk.get("label")
                name = (spk.get("name") or "").strip()
                if not label or not name or name == label:
                    continue
                mapping[label] = name
                if spk.get("embedding_b64"):
                    emb = speaker_id.deserialize_embedding(
                        base64.b64decode(spk["embedding_b64"]), spk["embedding_dim"])
                    suggestions[label] = {"embedding": emb}

            if not mapping:
                continue

            result = rename_speakers(segments_json, mapping, overrides, regenerate_notes)
            speakers_renamed += result["renamed_segments"]
            files_updated += 1

            if suggestions:
                commit_result = commit_speaker_identities(db_path, suggestions, mapping)
                roster_updates += len(commit_result)
        except Exception as exc:
            log.error("Applying speaker roster failed for %s: %s", segments_json, exc)
            errors.append({"file": segments_json, "error": str(exc)})

    log.info("Speaker roster applied: %d file(s) updated, %d segment(s) renamed, "
             "%d roster update(s), %d error(s).",
             files_updated, speakers_renamed, roster_updates, len(errors))
    return {
        "files_updated": files_updated,
        "speakers_renamed": speakers_renamed,
        "roster_updates": roster_updates,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Slide review/reprocess (task #58): omit or merge detected slides after
# the fact, then rebuild only the report outputs, without re-running
# transcription, frame extraction, slide-change detection, or VLM
# annotation. Used by gui.py's SlideReviewDialog.
# ---------------------------------------------------------------------------

def apply_slide_edits(slides: list[dict], edits: dict) -> list[dict]:
    """
    Apply title/omit/merge edits to a list of slide dicts loaded from an
    existing *_slides.json, returning a new, rebuilt list. Does not
    mutate slides or edits.

    edits schema (also the on-disk format of <stem>_slides_edits.json):
      {"omit_indices": [1, 4], "merge_next_indices": [6, 7],
       "title_overrides": {"2": "Corrected Title"}}
    omit_indices: 0-based indices into the ORIGINAL slides list to drop
    entirely from the rebuilt output.
    merge_next_indices: 0-based indices whose slide is merged INTO the
    next non-omitted slide: the group's earliest timestamp/frame_index
    wins (so timing stays anchored to when the content first appeared),
    bullets are concatenated in order (duplicates dropped), transcript
    segments are concatenated, and the LAST slide's title/snapshot/type
    are kept (normally the most complete view of that content). Chaining
    (e.g. both 6 and 7 marked) merges 6, 7, 8 into a single entry.
    title_overrides (task #64): maps a 0-based ORIGINAL slide index
    (string key, for JSON compatibility) to a replacement title. Applied
    BEFORE merge, so a merged group's kept title (see _merge_slide_group)
    reflects the edit.
    """
    omit = set(edits.get("omit_indices", []) or [])
    merge_next = set(edits.get("merge_next_indices", []) or [])
    title_overrides = edits.get("title_overrides") or {}
    n = len(slides)

    if title_overrides:
        slides = [dict(s) for s in slides]
        for key, new_title in title_overrides.items():
            idx = int(key)
            if 0 <= idx < n:
                slides[idx]["title"] = new_title

    result: list[dict] = []
    i = 0
    while i < n:
        if i in omit:
            i += 1
            continue
        group = [i]
        cur = i
        while cur in merge_next:
            nxt = cur + 1
            while nxt in omit and nxt < n:
                nxt += 1
            if nxt >= n:
                break
            group.append(nxt)
            cur = nxt
        result.append(_merge_slide_group([slides[j] for j in group]))
        i = group[-1] + 1
    return result


def _merge_slide_group(group: list[dict]) -> dict:
    if len(group) == 1:
        return dict(group[0])
    merged = dict(group[-1])
    merged["timestamp_sec"] = min(s.get("timestamp_sec", 0) for s in group)
    merged["frame_index"] = min(s.get("frame_index", 0) for s in group)
    bullets: list[str] = []
    for s in group:
        for b in s.get("bullets", []) or []:
            if b not in bullets:
                bullets.append(b)
    merged["bullets"] = bullets
    transcript_parts = [s.get("transcript_seg", "") for s in group if s.get("transcript_seg")]
    merged["transcript_seg"] = " ".join(transcript_parts)
    return merged


def rebuild_outputs(slides_json_path: str, edits: dict, overrides: dict | None = None) -> bool:
    """
    Re-run only the report-generation stage (CSV/HTML/PDF/timing summary)
    from an existing *_slides.json plus a set of omit/merge edits (see
    apply_slide_edits), WITHOUT re-running frame extraction, slide-change
    detection, VLM annotation, or transcription. Fast: typically seconds
    even for a long recording. The original *_slides.json is never
    modified; edits are written to a separate <stem>_slides_edits.json,
    and the rebuilt full slide list is written to
    <stem>_slides_rebuilt.json (report_json only) so the raw detection
    data always stays recoverable. Returns True on success.
    """
    overrides = overrides or {}
    cfg = build_run_config(overrides)
    slides_json_path = Path(slides_json_path)
    if not slides_json_path.exists():
        log.error("Slides JSON not found: %s", slides_json_path)
        return False

    original = json.loads(slides_json_path.read_text(encoding="utf-8"))
    slides = apply_slide_edits(original, edits)
    log.info("Rebuilding outputs: %d slide(s) -> %d after edits.", len(original), len(slides))

    slides_dir = slides_json_path.parent
    stem = slides_json_path.stem.replace("_slides", "")

    segments_json_path = slides_dir.parent / f"{stem}_segments.json"
    segments = []
    if segments_json_path.exists():
        try:
            segments = json.loads(segments_json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Could not read %s for Q&A slide rebuild: %s", segments_json_path.name, exc)
    _add_qa_slide(slides, segments, cfg)

    video_path = None
    for candidate in slides_dir.parent.glob(f"{stem}.*"):
        if candidate.suffix.lower() in {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}:
            video_path = str(candidate)
            break
    video_name = Path(video_path).name if video_path else stem

    m_title = cfg.get("meeting_title", "")
    m_date = cfg.get("meeting_date", "")
    m_comments = cfg.get("meeting_comments", "")
    show_image = cfg.get("report_show_image", True)
    show_bullets = cfg.get("report_show_bullets", True)
    show_transcript = cfg.get("report_show_transcript", True)
    transcript_mode = cfg.get("report_transcript_mode", "full")
    title_slide = _build_title_slide(cfg, slides_dir / "snapshots", stem)
    rec_speed = cfg.get("recording_speed", 1.0)

    if cfg["report_csv"]:
        reporter.save_csv(slides, slides_dir / f"{stem}_slides.csv")
    if cfg["report_json"]:
        reporter.save_json(slides, slides_dir / f"{stem}_slides_rebuilt.json")
    if cfg["report_html"]:
        reporter.save_html(slides, slides_dir / f"{stem}_report.html", video_name,
                           meeting_title=m_title, meeting_date=m_date, meeting_comments=m_comments,
                           show_image=show_image, show_bullets=show_bullets,
                           show_transcript=show_transcript, transcript_mode=transcript_mode,
                           title_slide=title_slide, recording_speed=rec_speed)
    if cfg["report_pdf"]:
        pdf_html = slides_dir / f"{stem}_report_pdf.html"
        reporter.save_html_for_pdf(slides, pdf_html, video_name,
                                   meeting_title=m_title, meeting_date=m_date, meeting_comments=m_comments,
                                   show_image=show_image, show_bullets=show_bullets,
                                   show_transcript=show_transcript, transcript_mode=transcript_mode,
                                   title_slide=title_slide, recording_speed=rec_speed)
        reporter.save_pdf_from_html(pdf_html, slides_dir / f"{stem}_report.pdf")
    if cfg.get("report_slide_timing", True):
        reporter.save_slide_timing_summary(slides, slides_dir / f"{stem}_slide_timing.txt", video_name,
                                           meeting_title=m_title, meeting_date=m_date,
                                           meeting_comments=m_comments, recording_speed=rec_speed)

    edits_path = slides_dir / f"{stem}_slides_edits.json"
    edits_path.write_text(json.dumps(edits, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Rebuild complete. Edits saved: %s", edits_path)
    return True


# ---------------------------------------------------------------------------
# Batch modes
# ---------------------------------------------------------------------------

def write_log(log_file: Path, entry: str) -> None:
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(entry + "\n")


def run_file_list(list_file: str, overrides: dict, stop_check=None) -> None:
    """Batch-process every file listed in list_file (one path per line,
    optional |language suffix). Works for audio and video files alike -
    slide detection still runs per line exactly as configured.

    Resume/retry: mirrors run_notes_batch's skip behaviour. Before
    processing a line, checks the database for a completed run (notes
    already generated) for that exact file path and skips it, unless
    force_retranscribe is set in overrides. This makes an interrupted
    batch safe to simply re-run from the same list.txt / paste box."""
    entries = []
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|", 1)
            entries.append((parts[0].strip(), parts[1].strip() if len(parts) > 1 else None))

    cfg = build_run_config(overrides)
    force = bool(cfg.get("force_retranscribe"))
    db.validate_schema(cfg["db_path"])

    log_file = Path(list_file).parent / f"batch_log_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt"
    log.info("BATCH (file list): %d file(s)", len(entries))
    write_log(log_file, f"Batch: {datetime.now()} | {list_file} | {len(entries)} files")
    ok, errors, skipped = 0, 0, 0
    # Reuse the whisper/alignment/diarization models across every file in
    # this batch instead of reloading them per file (roughly 30-90s each).
    transcriber.enable_model_cache(True)
    for i, (file, lang) in enumerate(entries, 1):
        if stop_check is not None and stop_check():
            log.info("Stop requested - halting batch after %d/%d file(s).", i - 1, len(entries))
            write_log(log_file, f"Stopped by user before [{i}]: {file}")
            break
        if not force:
            conn = db.get_connection(cfg["db_path"])
            done = db.get_completed_transcript(conn, str(Path(file)))
            conn.close()
            if done:
                log.info("[%d/%d] SKIPPED (already processed on %s): %s",
                         i, len(entries), done.get("processed_at", "?"), file)
                write_log(log_file, f"[{i}] SKIPPED (already done {done.get('processed_at','?')}): {file}")
                skipped += 1
                continue

        log.info("[%d/%d] %s", i, len(entries), file)
        t_start = datetime.now()
        run_overrides = dict(overrides)
        if lang:
            run_overrides["whisper_language"] = lang
        try:
            success = process_file(file, run_overrides, stop_check=stop_check)
            dur = (datetime.now() - t_start).seconds
            write_log(log_file, f"[{i}] {'OK' if success else 'NOT FOUND'} ({dur}s): {file}")
            ok += 1 if success else 0
            errors += 0 if success else 1
        except Exception as e:
            dur = (datetime.now() - t_start).seconds
            log.error("ERROR: %s", e)
            write_log(log_file, f"[{i}] ERROR ({dur}s): {file} -- {e}")
            write_log(log_file, traceback.format_exc())
            errors += 1
    # Turn the cache off again AND hand the memory back. Releasing without
    # disabling would leave _CACHE_ENABLED set for the rest of the process,
    # so later single-file runs would keep their models resident with
    # nothing left to release them. The per-file try/except above means the
    # loop always reaches this point.
    transcriber.enable_model_cache(False)
    write_log(log_file, f"Done: OK={ok} Errors={errors} Skipped={skipped}")
    log.info("BATCH DONE: OK=%d  Errors=%d  Skipped=%d  Log: %s", ok, errors, skipped, log_file)


def run_batch_rows(rows: list[dict], overrides: dict, stop_check=None, progress_callback=None) -> None:
    """
    Batch-process a list of per-file row dicts (task #62), built from the
    GUI's editable batch table. Unlike run_file_list's "path|lang" text
    format, each row can carry its own settings on top of the shared
    overrides, since Q&A timing/title/date/comments vary per recording
    and don't fit cleanly into one delimited line.

    Row keys (only "file" is required; all others optional, blank/None
    means "use the shared value from the fields above"): file, language,
    meeting_title, meeting_date, meeting_comments, qa_start_time_sec,
    qa_end_time_sec, title_slide_image_path, srt_import_path.

    srt_import_path (task: batch YouTube import - each video in the batch
    can have its own .srt): if set, import_srt_transcript is called for
    that row BEFORE process_file, so this row's transcription is skipped
    in favor of the given .srt (see import_srt_transcript's docstring for
    the cache files this writes). Unlike the other row keys, a failed
    import does NOT fall through to a real whisperx transcription - that
    would silently defeat the point of specifying the .srt in the first
    place - it is logged and counted as an error for that row instead.

    Resume/retry and logging mirror run_file_list: a file already fully
    processed (notes generated) is skipped unless force_retranscribe is
    set, and a per-run log is written next to the first file's folder.

    progress_callback(index, status) (task #74), if given, is called with
    the row's 1-based position in rows (matching enumerate(rows, 1) below,
    i.e. counting only rows that reach this loop body - a row with a blank
    "file" is skipped before ever incrementing the visible index) and one
    of "running", "done", "failed", "error", "skipped". Used by the GUI to
    drive a live Status column in the batch table; never raises on its
    own account (any exception from the callback itself is swallowed, so
    a GUI display glitch can't take down a batch run).
    """
    def _report(index: int, status: str) -> None:
        if progress_callback is not None:
            try:
                progress_callback(index, status)
            except Exception:
                pass

    cfg = build_run_config(overrides)
    force = bool(cfg.get("force_retranscribe"))
    db.validate_schema(cfg["db_path"])

    first_file = next((r.get("file") for r in rows if r.get("file")), None)
    log_dir = Path(first_file).parent if first_file else Path(".")
    log_file = log_dir / f"batch_log_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt"
    log.info("BATCH (table): %d file(s)", len(rows))
    write_log(log_file, f"Batch (table): {datetime.now()} | {len(rows)} files")

    row_override_keys = ("language", "meeting_title", "meeting_date", "meeting_comments",
                         "qa_start_time_sec", "qa_end_time_sec", "title_slide_image_path")
    # "language" is a row-only convenience name; process_file's config key is
    # whisper_language, matching run_file_list's existing |lang convention.
    row_to_cfg_key = {"language": "whisper_language"}

    ok, errors, skipped = 0, 0, 0
    # Reuse the whisper/alignment/diarization models across every file in
    # this batch instead of reloading them per file (roughly 30-90s each).
    transcriber.enable_model_cache(True)
    for i, row in enumerate(rows, 1):
        file = (row.get("file") or "").strip()
        if not file:
            continue
        if stop_check is not None and stop_check():
            log.info("Stop requested - halting batch after %d/%d file(s).", i - 1, len(rows))
            write_log(log_file, f"Stopped by user before [{i}]: {file}")
            break

        if not force:
            conn = db.get_connection(cfg["db_path"])
            done = db.get_completed_transcript(conn, str(Path(file)))
            conn.close()
            if done:
                log.info("[%d/%d] SKIPPED (already processed on %s): %s",
                         i, len(rows), done.get("processed_at", "?"), file)
                write_log(log_file, f"[{i}] SKIPPED (already done {done.get('processed_at','?')}): {file}")
                skipped += 1
                _report(i, "skipped")
                continue

        log.info("[%d/%d] %s", i, len(rows), file)
        _report(i, "running")
        t_start = datetime.now()
        run_overrides = dict(overrides)
        for key in row_override_keys:
            val = row.get(key)
            if val not in (None, ""):
                run_overrides[row_to_cfg_key.get(key, key)] = val

        srt_import_path = (row.get("srt_import_path") or "").strip()
        if srt_import_path:
            if not import_srt_transcript(file, srt_import_path, run_overrides):
                log.error("[%d/%d] .srt import failed for %s <- %s - skipping "
                          "this file rather than falling back to a real "
                          "transcription.", i, len(rows), file, srt_import_path)
                write_log(log_file, f"[{i}] SRT IMPORT FAILED: {file} <- {srt_import_path}")
                errors += 1
                _report(i, "error")
                continue
        try:
            success = process_file(file, run_overrides, stop_check=stop_check)
            dur = (datetime.now() - t_start).seconds
            write_log(log_file, f"[{i}] {'OK' if success else 'NOT FOUND'} ({dur}s): {file}")
            ok += 1 if success else 0
            errors += 0 if success else 1
            _report(i, "done" if success else "failed")
        except Exception as e:
            dur = (datetime.now() - t_start).seconds
            log.error("ERROR: %s", e)
            write_log(log_file, f"[{i}] ERROR ({dur}s): {file} -- {e}")
            write_log(log_file, traceback.format_exc())
            errors += 1
            _report(i, "error")
    # Turn the cache off again AND hand the memory back. Releasing without
    # disabling would leave _CACHE_ENABLED set for the rest of the process,
    # so later single-file runs would keep their models resident with
    # nothing left to release them. The per-file try/except above means the
    # loop always reaches this point.
    transcriber.enable_model_cache(False)
    write_log(log_file, f"Done: OK={ok} Errors={errors} Skipped={skipped}")
    log.info("BATCH DONE: OK=%d  Errors=%d  Skipped=%d  Log: %s", ok, errors, skipped, log_file)


def run_batch_folder(folder: str, overrides: dict, recursive: bool = False, stop_check=None) -> None:
    """
    Process every supported audio/video file found directly in folder
    (or recursively with recursive=True), without needing a hand-written
    list.txt. Skips a file if its *_transcript_speakers.txt already exists
    and --force-retranscribe was not requested, same cache rule as a
    single run.
    """
    folder = Path(folder)
    if not folder.is_dir():
        log.error("Not a folder: %s", folder)
        return

    # Exclude processed_subfolder_name (see move_processed_source_file):
    # without this, a recursive re-scan of the same parent folder would
    # pick archived files back up and reprocess them - the whole point of
    # move_processed_files is that a file, once done, never comes up
    # again. Filtering here (rather than only relying on files no longer
    # matching a plain, non-recursive scan) makes recursive=True safe too.
    cfg = build_run_config(overrides)
    processed_dirname = (cfg.get("processed_subfolder_name") or "_processed").strip() or "_processed"

    # Exclude "<output_prefix>_enhanced_tmp.wav" stray files: the optional
    # enhance_audio cleanup pass (see process_file, Step 1) writes this
    # temporary file next to the source and unlink()s it afterwards, but a
    # run interrupted mid-transcription (crash, --stop, power loss) can
    # leave it behind. Left in place, a later batch/recursive scan picks
    # it up as if it were its own source recording (it matches *.wav) and
    # fails with "File not found" once whisperx tries to actually read it
    # via its original, by-then-deleted path.
    pattern_fn = folder.rglob if recursive else folder.glob
    files = sorted(
        p for ext in SUPPORTED_EXTENSIONS for p in pattern_fn(f"*{ext}")
        if processed_dirname not in p.relative_to(folder).parts
        and not p.name.endswith("_enhanced_tmp.wav")
    )
    if not files:
        log.info("No supported audio/video files found in: %s", folder)
        return

    log_file = folder / f"batch_folder_log_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt"
    log.info("BATCH (folder): %d file(s) in %s", len(files), folder)
    write_log(log_file, f"Batch folder: {datetime.now()} | {folder} | {len(files)} files")
    ok, errors = 0, 0
    # Reuse the whisper/alignment/diarization models across every file in
    # this batch instead of reloading them per file (roughly 30-90s each).
    transcriber.enable_model_cache(True)
    for i, file in enumerate(files, 1):
        if stop_check is not None and stop_check():
            log.info("Stop requested - halting batch after %d/%d file(s).", i - 1, len(files))
            write_log(log_file, f"Stopped by user before [{i}]: {file.name}")
            break
        log.info("[%d/%d] %s", i, len(files), file.name)
        t_start = datetime.now()
        try:
            success = process_file(str(file), overrides, stop_check=stop_check)
            dur = (datetime.now() - t_start).seconds
            write_log(log_file, f"[{i}] {'OK' if success else 'FAILED'} ({dur}s): {file.name}")
            ok += 1 if success else 0
            errors += 0 if success else 1
        except Exception as e:
            dur = (datetime.now() - t_start).seconds
            log.error("ERROR: %s", e)
            write_log(log_file, f"[{i}] ERROR ({dur}s): {file.name} -- {e}")
            write_log(log_file, traceback.format_exc())
            errors += 1
    # Turn the cache off again AND hand the memory back. Releasing without
    # disabling would leave _CACHE_ENABLED set for the rest of the process,
    # so later single-file runs would keep their models resident with
    # nothing left to release them. The per-file try/except above means the
    # loop always reaches this point.
    transcriber.enable_model_cache(False)
    write_log(log_file, f"Done: OK={ok} Errors={errors}")
    log.info("BATCH FOLDER DONE: OK=%d  Errors=%d  Log: %s", ok, errors, log_file)


def run_notes_batch(folder: str, overrides: dict) -> None:
    folder = Path(folder)
    if not folder.is_dir():
        log.error("Not a folder: %s", folder)
        return
    transcripts = sorted(folder.glob("*_transcript_speakers.txt"))
    if not transcripts:
        log.info("No *_transcript_speakers.txt files found in: %s", folder)
        return
    log_file = folder / f"notes_batch_log_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt"
    log.info("NOTES BATCH: %d transcript(s)", len(transcripts))
    write_log(log_file, f"Notes batch: {datetime.now()} | {folder}")
    ok, errors = 0, 0
    for i, tf in enumerate(transcripts, 1):
        log.info("[%d/%d] %s", i, len(transcripts), tf.name)
        notes_file = Path(str(tf).replace("_transcript_speakers.txt", "_notes.txt"))
        if notes_file.exists():
            log.info("Skipped: _notes.txt already exists.")
            write_log(log_file, f"[{i}] SKIPPED: {tf.name}")
            continue
        t_start = datetime.now()
        try:
            run_notes_only(str(tf), overrides)
            dur = (datetime.now() - t_start).seconds
            write_log(log_file, f"[{i}] OK ({dur}s): {tf.name}")
            ok += 1
        except Exception as e:
            dur = (datetime.now() - t_start).seconds
            log.error("ERROR: %s", e)
            write_log(log_file, f"[{i}] ERROR ({dur}s): {tf.name} -- {e}")
            write_log(log_file, traceback.format_exc())
            errors += 1
    write_log(log_file, f"Done: OK={ok} Errors={errors}")
    log.info("NOTES BATCH DONE: OK=%d  Errors=%d  Log: %s", ok, errors, log_file)


def _finalize_notes_outputs(transcript_file: str, notes_text: str, cfg: dict,
                            prompt_path: Path, output_prefix: str, filename: str,
                            stem: str) -> None:
    """
    Write the notes.txt/html/docx/pdf output files and update the DB row
    for one already-generated notes_text. Extracted out of run_notes_only
    so this exact same post-processing (report writing, DB upsert with
    the same db_key resolution) runs identically whether notes_text came
    from a synchronous call (run_notes_only) or from an Anthropic Message
    Batch result retrieved later (run_notes_batch_via_batch_api) -
    duplicating this would risk the two paths silently drifting apart
    (e.g. one gaining a new output format the other forgets).
    """
    title = (cfg.get("meeting_title")
            or derive_heading_from_filename(filename)
            or (prompt_path.stem.upper() + " NOTES"))
    event_date = cfg.get("meeting_date", "")
    comments = cfg.get("meeting_comments", "")
    model_label = cfg["ollama_notes_model"] if cfg["llm_backend"] == "ollama" else cfg["claude_model"]
    notes_html_path = None
    if cfg.get("notes_format_txt", True):
        reporter.save_notes_txt(notes_text, output_prefix + "_notes.txt", filename,
                                cfg["llm_backend"], model_label, event_date=event_date,
                                comments=comments)
    if cfg.get("notes_format_html", True):
        notes_html_path = reporter.save_notes_html(notes_text, output_prefix + "_notes.html", filename,
                                                    title=title, generated_by=cfg["llm_backend"],
                                                    event_date=event_date, comments=comments)
    if cfg.get("notes_format_docx", False):
        reporter.save_notes_docx(notes_text, output_prefix + "_notes.docx", filename,
                                 title=title, generated_by=cfg["llm_backend"], event_date=event_date,
                                 comments=comments)
    if cfg.get("notes_format_pdf", False):
        if notes_html_path is None:
            notes_html_path = reporter.save_notes_html(notes_text, output_prefix + "_notes.html", filename,
                                                        title=title, generated_by=cfg["llm_backend"],
                                                        event_date=event_date, comments=comments)
        reporter.save_pdf_from_html(notes_html_path, Path(output_prefix + "_notes.pdf"))

    # notes-only mode has no fresh whisperx segment count (transcript
    # already existed). Key the db row on the original media file if it
    # can still be found next to the transcript (same file_path used
    # during transcription, so this updates that row rather than
    # creating a duplicate); otherwise fall back to the transcript path
    # itself so the notes are still searchable via FTS5.
    transcript_text = transcriber.read_transcript_text(transcript_file)
    media_dir = Path(transcript_file).parent
    db_key = transcript_file
    for candidate in media_dir.glob(f"{stem}.*"):
        if candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
            db_key = str(candidate)
            break
    segment_count = transcript_text.count("\n") if transcript_text else 0
    conn = db.get_connection(cfg["db_path"])
    db.upsert_transcript(
        conn, db_key, cfg.get("whisper_model", ""), cfg["whisper_language"] or "auto",
        segment_count, 0, notes_generated=True, notes_backend=cfg["llm_backend"],
        prompt_template=Path(prompt_path).name, notes_text=notes_text,
    )
    conn.close()


def run_notes_only(transcript_file: str, overrides: dict) -> None:
    """Generate notes from an existing *_transcript_speakers.txt (no whisperx).

    Q&A section (task #61/#112): when qa_start_time_sec isn't set, this
    just rereads transcript_file verbatim - it already holds the FULL
    transcript (writing it out happens in Step 1, before any Q&A
    handling). When qa_start_time_sec IS set, transcript_file alone is no
    longer enough - it has no timestamps left, and either the exclude
    case (drop everything from qa_start_time_sec on) or the include case
    (mark where Q&A starts for the model - see
    notes.build_transcript_text_with_qa_marker) needs them - so the
    transcript is rebuilt instead from the matching *_segments.json cache
    (the same file rename_speakers.py's rename uses), the same way
    process_file's Step 3 does it. This is what lets the Q&A boundary be
    set - or corrected - AFTER the original run, typically once Slide
    Review has pinned down where Q&A actually starts, and the summary
    regenerated from here (GUI: "Notes-only (existing transcript)" mode,
    or --notes-only) without re-transcribing.
    """
    cfg = build_run_config(overrides)
    stem = Path(transcript_file).stem.replace("_transcript_speakers", "")
    output_prefix = resolve_output_prefix(str(Path(transcript_file).with_name(stem)), cfg)
    filename = Path(transcript_file).name

    qa_start = cfg.get("qa_start_time_sec")
    qa_end = cfg.get("qa_end_time_sec")
    qa_include_in_summary = cfg.get("qa_include_in_summary", True)
    transcript_text = None
    if qa_start:
        segments_json_path = Path(str(transcript_file).replace("_transcript_speakers.txt", "_segments.json"))
        if segments_json_path.exists():
            segments = json.loads(segments_json_path.read_text(encoding="utf-8"))
            if not qa_include_in_summary:
                main_segments = [s for s in segments if s.get("start", 0) < qa_start]
                if len(main_segments) == len(segments):
                    log.warning(
                        "qa_start_time_sec (%.0fs) is set but no transcript segments start "
                        "at/after it - check the value: it must be already-converted real "
                        "time in seconds, matching the transcript/slide report timestamps.",
                        qa_start)
                transcript_text = notes.build_transcript_text_from_segments(main_segments)
            else:
                transcript_text = notes.build_transcript_text_with_qa_marker(
                    segments, qa_start, qa_end)
        else:
            log.warning(
                "qa_start_time_sec is set but no matching segment cache (%s) was found "
                "next to %s - regenerating notes from the plain transcript file, "
                "unfiltered/unmarked (Q&A stays in the summary either way, but without "
                "the exclude filter or the marker that helps the model find it).",
                segments_json_path.name, filename)
    if transcript_text is None:
        transcript_text = transcriber.read_transcript_text(transcript_file)

    prompt_path = notes.resolve_prompt_path(PROMPTS_DIR, cfg["prompt_template"])
    notes_text = notes.generate_notes(
        transcript_text=transcript_text, prompt_path=prompt_path,
        llm_backend=cfg["llm_backend"], ollama_base_url=cfg["ollama_base_url"],
        ollama_notes_model=cfg["ollama_notes_model"], anthropic_api_key=cfg["anthropic_api_key"],
        claude_model=cfg["claude_model"], filename=filename,
        single_pass_limit=cfg["single_pass_limit"], chunk_size=cfg["chunk_size"],
        meeting_title=cfg.get("meeting_title", ""), meeting_date=cfg.get("meeting_date", ""),
        meeting_comments=cfg.get("meeting_comments", ""),
        output_language=cfg.get("output_language", "auto"),
        ollama_num_ctx=cfg.get("ollama_notes_num_ctx", 16_384),
    )
    if not notes_text:
        raise RuntimeError("Notes generation failed (see log above).")

    _finalize_notes_outputs(transcript_file, notes_text, cfg, prompt_path, output_prefix,
                            filename, stem)


def run_notes_batch_via_batch_api(folder: str, overrides: dict,
                                  poll_interval_sec: int = 30, stop_check=None) -> None:
    """
    Anthropic-only alternative to run_notes_batch: submits every pending
    file's notes-generation request as ONE Anthropic Message Batch instead
    of one live API call per file. Both input and output tokens are
    billed at 50% of standard prices, and the prompt-caching discount on
    the shared system prompt (see notes._build_anthropic_request) stacks
    on top when the same recording_type/prompt_template/output_language
    is used across the batch.

    Deliberately a SEPARATE function from run_notes_batch rather than a
    flag on it: this only makes sense for cfg["llm_backend"] == "anthropic"
    (there is no Ollama batch API), it needs the whole file list up front
    to submit as one batch instead of processing file-by-file, and it can
    take anywhere from a couple of minutes up to the batch's 24-hour
    expiry window to come back - not something you want a synchronous,
    live-progress GUI/CLI run to silently turn into. run_notes_batch (one
    call per file, immediate per-file results) is unchanged and remains
    the default; use this one explicitly (--notes-batch-api) when
    processing enough files that the cost/throughput trade-off is worth
    the wait.

    poll_interval_sec: how often to check the batch's processing_status
    while waiting (Anthropic's own guidance uses 60s; 30s here trades a
    few extra, cheap status-check calls for a shorter perceived wait on
    the smaller batches this pipeline is likely to submit).

    stop_check (optional): checked between polls. If it returns True, the
    batch is canceled server-side (client.messages.batches.cancel) rather
    than just abandoning our wait - a canceled batch still returns partial
    results for anything that finished before cancellation, which are
    then processed exactly like a normal completed batch's results.
    """
    folder = Path(folder)
    if not folder.is_dir():
        log.error("Not a folder: %s", folder)
        return
    transcripts = sorted(folder.glob("*_transcript_speakers.txt"))
    if not transcripts:
        log.info("No *_transcript_speakers.txt files found in: %s", folder)
        return

    cfg = build_run_config(overrides)
    if cfg["llm_backend"] != "anthropic":
        log.error("--notes-batch-api requires llm_backend=anthropic (got '%s'). "
                 "Use --notes-batch for the ollama backend.", cfg["llm_backend"])
        return
    api_key = cfg.get("anthropic_api_key", "")
    if not api_key or api_key.startswith("sk-ant-..."):
        log.error("ANTHROPIC_API_KEY not set in keys.cfg - skipping batch submission.")
        return

    prompt_path = notes.resolve_prompt_path(PROMPTS_DIR, cfg["prompt_template"])
    log_file = folder / f"notes_batch_api_log_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt"
    write_log(log_file, f"Notes batch (Anthropic Batches API): {datetime.now()} | {folder}")

    # Build one batch request per pending file, same skip rule as
    # run_notes_batch (an existing _notes.txt means this file was already
    # done - by either mode; the two are interchangeable from here on).
    pending: dict[str, Path] = {}  # custom_id -> transcript file path
    for i, tf in enumerate(transcripts, 1):
        notes_file = Path(str(tf).replace("_transcript_speakers.txt", "_notes.txt"))
        if notes_file.exists():
            write_log(log_file, f"SKIPPED (already has notes): {tf.name}")
            continue
        pending[f"item-{i}"] = tf

    if not pending:
        log.info("Nothing to submit - every transcript already has notes.")
        return

    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request as BatchRequest

    requests = []
    for custom_id, tf in pending.items():
        transcript_text = transcriber.read_transcript_text(str(tf))
        request_body = notes.build_anthropic_batch_request(
            transcript_text=transcript_text, prompt_path=prompt_path, filename=tf.name,
            claude_model=cfg["claude_model"],
            meeting_title=cfg.get("meeting_title", ""), meeting_date=cfg.get("meeting_date", ""),
            meeting_comments=cfg.get("meeting_comments", ""),
            output_language=cfg.get("output_language", "auto"),
        )
        requests.append(BatchRequest(
            custom_id=custom_id,
            params=MessageCreateParamsNonStreaming(**request_body),
        ))

    client = anthropic.Anthropic(api_key=api_key)
    log.info("NOTES BATCH (Anthropic Batches API): submitting %d request(s).", len(requests))
    batch = client.messages.batches.create(requests=requests)
    write_log(log_file, f"Submitted batch {batch.id} with {len(requests)} request(s).")
    log.info("Batch %s submitted - polling every %ds (Ctrl+C-safe; the batch keeps "
             "running server-side even if this process stops).", batch.id, poll_interval_sec)

    import time
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        if stop_check is not None and stop_check():
            log.info("Stop requested - canceling batch %s.", batch.id)
            client.messages.batches.cancel(batch.id)
            write_log(log_file, f"Canceled by user: {batch.id}")
            while batch.processing_status != "ended":
                time.sleep(poll_interval_sec)
                batch = client.messages.batches.retrieve(batch.id)
            break
        log.info("Batch %s: %s (processing=%d succeeded=%d errored=%d)", batch.id,
                 batch.processing_status, batch.request_counts.processing,
                 batch.request_counts.succeeded, batch.request_counts.errored)
        time.sleep(poll_interval_sec)

    ok, errors = 0, 0
    for result in client.messages.batches.results(batch.id):
        tf = pending.get(result.custom_id)
        if tf is None:
            log.warning("Batch result with unknown custom_id %r - skipping.", result.custom_id)
            continue
        if result.result.type != "succeeded":
            log.error("Batch item failed (%s) for %s.", result.result.type, tf.name)
            write_log(log_file, f"ERROR ({result.result.type}): {tf.name}")
            errors += 1
            continue
        notes_text = notes._extract_text_block(result.result.message.content)
        if not notes_text:
            log.error("Batch item for %s succeeded but had no text block.", tf.name)
            write_log(log_file, f"ERROR (no text block): {tf.name}")
            errors += 1
            continue
        try:
            file_stem = tf.stem.replace("_transcript_speakers", "")
            output_prefix = resolve_output_prefix(str(tf.with_name(file_stem)), cfg)
            _finalize_notes_outputs(str(tf), notes_text, cfg, prompt_path, output_prefix,
                                    tf.name, file_stem)
            write_log(log_file, f"OK: {tf.name}")
            ok += 1
        except Exception as e:
            log.error("Finalizing batch result for %s failed: %s", tf.name, e)
            write_log(log_file, f"ERROR (finalize): {tf.name} -- {e}")
            errors += 1

    write_log(log_file, f"Done: OK={ok} Errors={errors}")
    log.info("NOTES BATCH (Anthropic Batches API) DONE: OK=%d  Errors=%d  Log: %s",
             ok, errors, log_file)


# ---------------------------------------------------------------------------
# Document translation (Translate tab/CLI): txt/docx/pptx/pdf/images.
# Independent of the audio/video pipeline above - no whisper/notes/slides
# involved. translator.py does the LLM calls, doc_translate.py does the
# per-format file I/O; this section only wires config + batch/log
# conventions around them, mirroring run_file_list/run_batch_folder.
# ---------------------------------------------------------------------------

def _make_translate_fn(cfg: dict, target_language: str):
    """Build a translate_fn(text) -> str|None closure binding the
    configured LLM backend/model, for doc_translate.translate_file. Kept
    here (not in translator.py/doc_translate.py) so neither of those
    modules needs to know about config.py/CONFIG at all."""
    import translator

    def _fn(text: str):
        return translator.translate_text(
            text, target_language,
            llm_backend=cfg["llm_backend"],
            ollama_base_url=cfg["ollama_base_url"],
            ollama_translate_model=cfg.get("ollama_translate_model") or cfg["ollama_notes_model"],
            anthropic_api_key=cfg["anthropic_api_key"],
            claude_model=cfg["claude_model"],
            chunk_size=cfg.get("translate_chunk_size", 6_000),
            ollama_num_ctx=cfg.get("ollama_translate_num_ctx", 8_192),
        )
    return _fn


def _resolve_ocr_language(cfg: dict) -> str:
    """
    Map cfg["translate_ocr_language"] (a code from config.LANGUAGE_NAMES,
    "auto", or a raw Tesseract --lang string) to what
    doc_translate.translate_file's ocr_lang expects. This is the OCR
    SOURCE language/script (what Tesseract should expect to see in the
    image) - unrelated to target_language (what the translation is
    written into). Getting this wrong is the #1 cause of "translation
    doesn't make sense": Tesseract silently misreads a non-Latin script
    (e.g. Chinese slides) as English-shaped glyphs, and the LLM then
    faithfully "translates" that garbage into fluent-sounding nonsense
    (observed 2026-07-18 - see doc_translate._ocr_image_text).
    "auto" is not true language identification (Tesseract has no
    reliable way to do that) - it resolves to a fixed "eng+deu" default.
    """
    from config import TESSERACT_LANG_MAP
    code = cfg.get("translate_ocr_language", "auto")
    if not code or code == "auto":
        return "eng+deu"
    return TESSERACT_LANG_MAP.get(code, code)


def translate_single_file(path: str, target_language: str, overrides: dict):
    """Translate one file. Returns the written output Path. Raises on
    failure (unsupported type, missing file, OCR/extraction error) -
    callers decide how to surface that (GUI dialog, CLI exit code)."""
    import doc_translate
    cfg = build_run_config(overrides)
    translate_fn = _make_translate_fn(cfg, target_language)
    return doc_translate.translate_file(
        path, target_language, translate_fn,
        output_dir_override=cfg.get("output_dir_override", ""),
        ocr_lang=_resolve_ocr_language(cfg),
    )


def translate_batch_folder(folder: str, target_language: str, overrides: dict,
                           stop_check=None, progress_callback=None) -> tuple[int, int]:
    """Translate every supported file (doc_translate.TRANSLATABLE_EXTENSIONS)
    found directly in folder (not recursive, matching run_batch_folder's
    default). progress_callback(index, status), if given, is called with
    the file's 1-based position and one of "running"/"done"/"failed"/
    "error" - same convention as run_batch_rows, used by the GUI's batch
    table Status column."""
    import doc_translate
    cfg = build_run_config(overrides)
    translate_fn = _make_translate_fn(cfg, target_language)

    folder_path = Path(folder)
    files = sorted(
        p for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in doc_translate.TRANSLATABLE_EXTENSIONS
    )
    log_file = folder_path / f"translate_log_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt"
    log.info("TRANSLATE BATCH (folder): %d file(s)", len(files))
    write_log(log_file, f"Translate batch: {datetime.now()} | {folder} | {len(files)} files | -> {target_language}")

    ok, errors = 0, 0
    for i, f in enumerate(files, 1):
        if stop_check is not None and stop_check():
            log.info("Stop requested - halting translation batch after %d/%d file(s).", i - 1, len(files))
            write_log(log_file, f"Stopped by user before [{i}]: {f.name}")
            break
        if progress_callback is not None:
            try:
                progress_callback(i, "running")
            except Exception:
                pass
        log.info("[%d/%d] %s", i, len(files), f.name)
        try:
            dest = doc_translate.translate_file(
                str(f), target_language, translate_fn,
                output_dir_override=cfg.get("output_dir_override", ""),
                ocr_lang=_resolve_ocr_language(cfg),
            )
            write_log(log_file, f"[{i}] OK: {f.name} -> {dest.name}")
            ok += 1
            if progress_callback is not None:
                try:
                    progress_callback(i, "done")
                except Exception:
                    pass
        except Exception as e:
            log.error("Translation failed for %s: %s", f.name, e)
            write_log(log_file, f"[{i}] ERROR: {f.name} -- {e}")
            errors += 1
            if progress_callback is not None:
                try:
                    progress_callback(i, "failed")
                except Exception:
                    pass
    write_log(log_file, f"Done: OK={ok} Errors={errors}")
    log.info("TRANSLATE BATCH DONE: OK=%d  Errors=%d  Log: %s", ok, errors, log_file)
    return ok, errors


def translate_file_list(list_file: str, target_language: str, overrides: dict,
                        stop_check=None, progress_callback=None) -> tuple[int, int]:
    """Translate every path listed in list_file (one per line, blank/#
    lines ignored), mirroring run_file_list's plain-text list format."""
    import doc_translate
    cfg = build_run_config(overrides)
    translate_fn = _make_translate_fn(cfg, target_language)

    entries = []
    with open(list_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(line)

    log_file = Path(list_file).parent / f"translate_log_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt"
    log.info("TRANSLATE BATCH (file list): %d file(s)", len(entries))
    write_log(log_file, f"Translate batch: {datetime.now()} | {list_file} | {len(entries)} files | -> {target_language}")

    ok, errors = 0, 0
    for i, path in enumerate(entries, 1):
        if stop_check is not None and stop_check():
            log.info("Stop requested - halting translation batch after %d/%d file(s).", i - 1, len(entries))
            write_log(log_file, f"Stopped by user before [{i}]: {path}")
            break
        if progress_callback is not None:
            try:
                progress_callback(i, "running")
            except Exception:
                pass
        log.info("[%d/%d] %s", i, len(entries), path)
        try:
            dest = doc_translate.translate_file(
                path, target_language, translate_fn,
                output_dir_override=cfg.get("output_dir_override", ""),
                ocr_lang=_resolve_ocr_language(cfg),
            )
            write_log(log_file, f"[{i}] OK: {path} -> {dest.name}")
            ok += 1
            if progress_callback is not None:
                try:
                    progress_callback(i, "done")
                except Exception:
                    pass
        except Exception as e:
            log.error("Translation failed for %s: %s", path, e)
            write_log(log_file, f"[{i}] ERROR: {path} -- {e}")
            errors += 1
            if progress_callback is not None:
                try:
                    progress_callback(i, "failed")
                except Exception:
                    pass
    write_log(log_file, f"Done: OK={ok} Errors={errors}")
    log.info("TRANSLATE BATCH DONE: OK=%d  Errors=%d  Log: %s", ok, errors, log_file)
    return ok, errors


# ---------------------------------------------------------------------------
# GUI file picker fallback (tkinter, no file argument given)
# ---------------------------------------------------------------------------

def select_file_gui() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        patterns = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS))
        file = filedialog.askopenfilename(
            title="Select audio or video file",
            filetypes=[("Audio/Video", patterns), ("All files", "*.*")],
        )
        root.destroy()
        return file
    except Exception:
        print("GUI not available. Provide the file path as a command-line argument.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Transkription_Notes_Pipeline: transcription + AI notes, "
                     "with optional slide detection for recorded webinars."
    )
    p.add_argument("file", nargs="?", help="Path to audio/video file (omit for GUI file picker).")
    p.add_argument("language", nargs="?",
                   help="Legacy positional language override: de, en, auto.")
    p.add_argument("--gui", action="store_true", help="Launch the parameter GUI instead of the CLI.")
    p.add_argument("--prompt-template", type=str,
                   help="Name of a .md file in prompts/ (e.g. meeting, webinar, or a custom template).")
    p.add_argument("--mode", choices=["meeting", "webinar", "audio_transcript", "video_transcript"],
                   help="Recording-type shortcut: sets --prompt-template and, unless --no-slides/"
                        "--no-vlm is also given, sensible slide-detection defaults (audio_transcript/"
                        "meeting = no slides, video_transcript/webinar = slides + VLM).")
    p.add_argument("--meeting-title", type=str, metavar="TITLE",
                   help="Meeting/event title, used as the notes document title.")
    p.add_argument("--meeting-date", type=str, metavar="YYYY-MM-DD",
                   help="Meeting/event date, shown in the notes document header.")
    p.add_argument("--comments", type=str, metavar="TEXT",
                   help="Free-text comments, shown in the notes and slide report headers.")
    p.add_argument("--output-name", type=str, metavar="NAME",
                   help="Base filename for every output file instead of the source file's stem.")
    p.add_argument("--title-image", type=str, metavar="FILE",
                   help="Image file (jpg/png) shown as a cover page before Slide 1 in the "
                        "HTML/PDF slide report.")
    p.add_argument("--recording-speed", type=float, metavar="FACTOR",
                   help="Recording speed factor; every timestamp in every output is converted "
                        "as real_time = video_time / FACTOR. Default: 1.0 (no change).")
    p.add_argument("--normalize-speed", action="store_true",
                   help="When --recording-speed is not 1.0, also produce a re-encoded "
                        "<stem>_realtime.mp4 that plays at actual real-time (1x) speed, "
                        "staying in sync with the already-converted .srt/transcript. "
                        "Requires a full ffmpeg re-encode (slow for long videos).")
    p.add_argument("--ics", type=str, metavar="FILE",
                   help="Read meeting title/date/comments from an .ics calendar invite (first "
                        "VEVENT), or a .txt/.docx file with \"Title:\"/\"Date:\"/\"Comments:\" "
                        "lines (see ics_utils.parse_labeled_text). Explicit --meeting-title/"
                        "--meeting-date/--comments still take precedence.")
    p.add_argument("--whisper-model", type=str, help="tiny|base|small|medium|large-v2|large-v3.")
    p.add_argument("--language", dest="language_flag", type=str, help="de|en|auto (flag form).")
    p.add_argument("--output-language", type=str, metavar="LANG",
                   help="Force the generated notes/summary AND VLM slide title/bullets into "
                        "this language (e.g. de, en, fr), independent of --language/whisper's "
                        "own transcription language. \"auto\" (default): no instruction added, "
                        "notes/slides follow the transcript's/slide's own language. The verbatim "
                        "transcript file is never translated.")
    p.add_argument("--llm-backend", choices=["ollama", "anthropic"])
    p.add_argument("--no-slides", action="store_true", help="Skip slide detection/VLM even for a video file.")
    p.add_argument("--no-vlm", action="store_true", help="Detect slides but skip VLM annotation.")
    p.add_argument("--no-whisper", action="store_true", help="Skip transcription.")
    p.add_argument("--no-summary", action="store_true", help="Skip notes/summary generation.")
    p.add_argument("--force-retranscribe", action="store_true", help="Ignore transcript cache.")
    p.add_argument("--import-srt", type=str, metavar="SRT_FILE",
                   help="Import SRT_FILE as the transcript for 'file' (the positional video/audio "
                        "argument) instead of running whisperx - writes the same cache files a real "
                        "transcription would, so any later run (Notes-only or full Video/Webinar "
                        "with slides) skips re-transcription. Combine with --force-retranscribe to "
                        "overwrite an existing transcript cache.")
    p.add_argument("--date-subject-filename", action="store_true",
                   help="Name output files \"YYYYMMDD_Subject\" from meeting_date/meeting_title "
                        "instead of the source filename (falls back to the usual naming when "
                        "meeting_title is blank). See CONFIG['use_date_subject_filename'].")
    p.add_argument("--move-processed", action="store_true",
                   help="After a file finishes successfully, move the source audio/video file "
                        "into a processed_subfolder_name subfolder next to it (default "
                        "\"_processed\"). Off by default. See CONFIG['move_processed_files'].")
    p.add_argument("--enhance-audio", action="store_true",
                   help="Run a conservative ffmpeg cleanup pass (rumble/hum filter, mild "
                        "denoise, volume normalization) on a temporary copy before "
                        "transcription. Off by default; original file is never modified.")
    p.add_argument("--dry-run", action="store_true", help="Slide timestamps only, no annotation/whisper/reports.")
    p.add_argument("--diarize", action="store_true", help="Enable speaker diarization (requires HF_TOKEN).")
    p.add_argument("--threshold", type=int, help="Slide-change hash threshold override.")
    p.add_argument("--min-slide-duration", type=float, metavar="SEC",
                   help="Minimum seconds a slide must be shown to count as a real change "
                        "(debounce for animations/transitions). Default: 2.0.")
    p.add_argument("--qa-start", type=float, metavar="SEC",
                   help="Start of a Q&A section, in seconds (already-converted real time, "
                        "matching the transcript/slide report). No new slides are detected from "
                        "this point onward (to --qa-end, or the end of the recording). By "
                        "default the matching transcript stays in the main transcript and is "
                        "summarised by the same prompt/pass as the rest of the recording; pass "
                        "--qa-exclude-from-summary to drop it from the summary instead.")
    p.add_argument("--qa-end", type=float, metavar="SEC",
                   help="End of the Q&A section, in seconds. Only used together with --qa-start. "
                        "Default: to the end of the recording.")
    p.add_argument("--qa-exclude-from-summary", action="store_true",
                   help="Only used together with --qa-start. Drop the Q&A section from the "
                        "notes/summary entirely (its 'QUESTIONS & ANSWERS' section falls back to "
                        "'No Q&A session') instead of the default of folding it into the same "
                        "summary prompt. See CONFIG['qa_include_in_summary'].")
    p.add_argument("--qa-autodetect", action="store_true",
                   help="Work out --qa-start automatically from cue phrases in the transcript "
                        "(\"now to the questions and answers\", \"kommen wir zu den Fragen\", "
                        "German and English). On by default for --mode webinar only. An "
                        "explicit --qa-start always wins. Never sets --qa-end. See "
                        "CONFIG['qa_autodetect_start'].")
    p.add_argument("--no-qa-autodetect", action="store_true",
                   help="Never auto-detect the Q&A start, not even for --mode webinar.")
    p.add_argument("--animation-threshold", type=int, metavar="N",
                   help="Distances between this and --threshold are treated as the current slide "
                        "still building (e.g. bullets appearing one at a time), instead of a new "
                        "slide: the slide's snapshot updates to the more complete frame, its start "
                        "timestamp stays the same, and no new slide is added. 0 disables this "
                        "(default). Must be less than --threshold.")
    p.add_argument("--fps", type=int, help="Frame extraction fps override.")
    p.add_argument("--output-dir", type=str, metavar="DIR",
                   help="Write all outputs for this run into DIR instead of next to the source file.")
    p.add_argument("--pdf", action="store_true", help="Also write notes as PDF (independent of slide mode).")
    p.add_argument("--docx", action="store_true", help="Also write notes as a Word .docx document.")
    p.add_argument("--notes-only", metavar="TRANSCRIPT_FILE",
                   help="Generate notes from an existing *_transcript_speakers.txt.")
    p.add_argument("--notes-batch", metavar="FOLDER",
                   help="Generate notes for every *_transcript_speakers.txt in FOLDER.")
    p.add_argument("--notes-batch-api", metavar="FOLDER",
                   help="Like --notes-batch, but submits every pending file as ONE Anthropic "
                        "Message Batch (50%% off standard API prices, stacks with prompt "
                        "caching) instead of one live call per file. Anthropic backend only "
                        "(llm_backend=anthropic); can take minutes to hours to come back - "
                        "see run_notes_batch_via_batch_api's docstring.")
    p.add_argument("--file-list", metavar="LIST_FILE",
                   help="Batch-process every file listed in LIST_FILE (one path per line).")
    p.add_argument("--batch-folder", metavar="FOLDER",
                   help="Batch-process every supported audio/video file found directly in FOLDER.")
    p.add_argument("--search", metavar="QUERY",
                   help="Full-text search notes and slide titles/bullets in the database, then exit.")
    p.add_argument("--reannotate-failed", metavar="SLIDES_JSON_FILE",
                   help="Re-run VLM annotation only for slides with empty title/bullets "
                        "(failed JSON parse) in an existing *_slides.json, using the "
                        "snapshots already on disk. Updates JSON/CSV/HTML/PDF/DB in place.")
    p.add_argument("--log-file", action="store_true",
                   help="Also write this run's log to a timestamped file under log_dir (see config.py).")

    # --- Document translation (Translate tab/CLI): independent of the
    # audio/video pipeline above - no whisper/notes/slides involved.
    p.add_argument("--translate", metavar="FILE",
                   help="Translate a single txt/docx/pptx/pdf/image file into --target-language. "
                        "docx/pptx/txt keep their format; pdf/images are written as a translated "
                        ".docx (see doc_translate.py). Legacy .doc/.ppt are not supported.")
    p.add_argument("--translate-batch-folder", metavar="FOLDER",
                   help="Translate every supported file found directly in FOLDER into "
                        "--target-language.")
    p.add_argument("--translate-file-list", metavar="LIST_FILE",
                   help="Translate every path listed in LIST_FILE (one per line, # comments "
                        "ignored) into --target-language.")
    p.add_argument("--target-language", type=str, metavar="LANG",
                   help="Target language for --translate/--translate-batch-folder/"
                        "--translate-file-list, e.g. de, en, fr (see config.LANGUAGE_NAMES). "
                        "Required for those flags; unrelated to --language/--output-language above.")
    p.add_argument("--ocr-language", type=str, metavar="LANG",
                   help="OCR SOURCE language/script for image inputs to --translate/etc., e.g. "
                        "de, zh (see config.TESSERACT_LANG_MAP), or a raw Tesseract --lang string "
                        "(e.g. chi_sim, deu+fra). NOT the translation target - this is what script "
                        "Tesseract should expect to see in the image. Getting this wrong silently "
                        "garbles non-Latin scripts into fluent-sounding nonsense once translated. "
                        "Default: config.py's translate_ocr_language (\"auto\" = \"eng+deu\").")
    return p.parse_args()


def overrides_from_args(args: argparse.Namespace) -> dict:
    prompt_template = args.prompt_template
    if args.mode and not prompt_template:
        prompt_template = args.mode  # "meeting"/"webinar"/"audio_transcript"/"video_transcript" -> <name>.md

    language = args.language_flag or args.language

    # Recording-type slide/VLM defaults, same presets as the GUI's
    # "Recording type" dropdown. Only applied when --mode is given and
    # not overridden by an explicit --no-slides/--no-vlm flag.
    mode_defaults = {
        "audio_transcript": (False, False),
        "video_transcript": (True, True),
        "meeting":           (False, False),
        "webinar":           (True, True),
    }
    mode_slides, mode_vlm = mode_defaults.get(args.mode, (None, None))

    meeting_title = args.meeting_title
    meeting_date = args.meeting_date
    meeting_comments = args.comments
    if args.ics:
        try:
            import ics_utils
            parsed = ics_utils.parse_meeting_info_file(args.ics)
            meeting_title = meeting_title or parsed.get("title") or None
            meeting_date = meeting_date or parsed.get("date") or None
            meeting_comments = meeting_comments or parsed.get("comments") or None
        except Exception as exc:
            log.warning("Could not read --ics %s: %s", args.ics, exc)

    overrides = {
        "prompt_template":     prompt_template,
        "whisper_model":       args.whisper_model,
        "whisper_language":    language,
        "output_language":     args.output_language,
        "llm_backend":         args.llm_backend,
        "enable_slides":       False if args.no_slides else mode_slides,
        "enable_vlm":          False if args.no_vlm else mode_vlm,
        "enable_whisper":      False if args.no_whisper else None,
        "no_summary":          True if args.no_summary else None,
        "force_retranscribe":  True if args.force_retranscribe else None,
        "use_date_subject_filename": True if args.date_subject_filename else None,
        "move_processed_files": True if args.move_processed else None,
        "enhance_audio":       True if args.enhance_audio else None,
        "dry_run":             True if args.dry_run else None,
        "enable_diarization":  True if args.diarize else None,
        "hash_threshold":      args.threshold,
        "min_slide_duration_sec": args.min_slide_duration,
        "animation_threshold": args.animation_threshold,
        "qa_start_time_sec": args.qa_start,
        "qa_end_time_sec": args.qa_end,
        "qa_include_in_summary": False if args.qa_exclude_from_summary else None,
        # Tri-state: --qa-autodetect forces on, --no-qa-autodetect forces off,
        # neither leaves CONFIG's None (= webinar template only) in place.
        "qa_autodetect_start": (True if args.qa_autodetect
                                else (False if args.no_qa_autodetect else None)),
        "recording_speed":     args.recording_speed,
        "convert_video_to_realtime": True if args.normalize_speed else None,
        "title_slide_image_path": args.title_image,
        "fps":                 args.fps,
        "output_dir_override": args.output_dir,
        "notes_format_pdf":    True if args.pdf else None,
        "notes_format_docx":   True if args.docx else None,
        "meeting_title":       meeting_title,
        "meeting_date":        meeting_date,
        "meeting_comments":    meeting_comments,
        "output_basename_override": args.output_name,
        "translate_ocr_language": args.ocr_language,
    }
    return {k: v for k, v in overrides.items() if v is not None}


def main() -> int:
    args = parse_args()
    setup_logging(
        CONFIG.get("log_level", "INFO"),
        log_to_file=args.log_file or CONFIG.get("log_to_file", False),
        log_dir=CONFIG.get("log_dir"),
    )

    if args.gui:
        import gui
        gui.launch()
        return 0

    if args.search:
        db.print_search_results(CONFIG["db_path"], args.search)
        return 0

    overrides = overrides_from_args(args)

    if args.translate or args.translate_batch_folder or args.translate_file_list:
        target_language = args.target_language or CONFIG.get("translate_target_language", "en")
        if not args.target_language:
            log.info("--target-language not given, using default: %s", target_language)
        if args.translate:
            dest = translate_single_file(args.translate, target_language, overrides)
            log.info("Translated: %s", dest)
            return 0
        if args.translate_batch_folder:
            ok, errors = translate_batch_folder(args.translate_batch_folder, target_language, overrides)
            return 0 if errors == 0 else 1
        ok, errors = translate_file_list(args.translate_file_list, target_language, overrides)
        return 0 if errors == 0 else 1

    if args.reannotate_failed:
        retried, still_failed = reannotate_failed_slides(args.reannotate_failed, overrides)
        return 0 if still_failed == 0 else 1

    if args.notes_batch:
        run_notes_batch(args.notes_batch, overrides)
        return 0

    if args.notes_batch_api:
        run_notes_batch_via_batch_api(args.notes_batch_api, overrides)
        return 0

    if args.notes_only:
        run_notes_only(args.notes_only, overrides)
        return 0

    if args.file_list:
        run_file_list(args.file_list, overrides)
        return 0

    if args.batch_folder:
        run_batch_folder(args.batch_folder, overrides)
        return 0

    file = args.file or select_file_gui()
    if not file:
        log.info("No file selected.")
        return 0

    if args.import_srt:
        ok = import_srt_transcript(file, args.import_srt, overrides)
        return 0 if ok else 1

    ok = process_file(file, overrides)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
