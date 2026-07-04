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
#    python run_pipeline.py --reannotate-failed "file_slides/file_slides.json"  (retry failed VLM slides only)
#    python run_pipeline.py --min-slide-duration 3.5 "webinar.mp4"  (animation debounce override)
#    python run_pipeline.py --qa-start 1215 "webinar.mp4"       (Q&A from 20:15 to the end: no new
#                                                                 slides there, separate Q&A summary)
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
import json
import logging
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


def resolve_output_prefix(source_file: str, cfg: dict) -> str:
    """
    Return the "<dir>/<stem>" prefix used to name every output file.
    Honors output_dir_override (--output-dir / GUI field): when set, ALL
    outputs for this run go there instead of next to the source file.
    Honors output_basename_override (--output-name / GUI field): when
    set, ALL output files use this name instead of the source file's stem
    (e.g. "2026-07-02_Q3-Kickoff" instead of "Video_2026-07-02_100348").
    """
    stem = (cfg.get("output_basename_override") or "").strip() or Path(source_file).stem
    override = (cfg.get("output_dir_override") or "").strip()
    if override:
        out_dir = Path(override)
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir / stem)
    return str(Path(source_file).parent / stem)


def _unique_snapshot_path(snapshot_dir: Path, base_name: str, slide_idx: int) -> Path:
    """
    Build "<base_name>_slideNNN.png" under snapshot_dir. If that exact
    filename already exists (e.g. this output name was already used for
    a different source file in the same folder), append "_2", "_3", ...
    until a free filename is found, instead of overwriting.
    """
    candidate = snapshot_dir / f"{base_name}_slide{slide_idx:03d}.png"
    if not candidate.exists():
        return candidate
    n = 2
    while True:
        candidate = snapshot_dir / f"{base_name}_slide{slide_idx:03d}_{n}.png"
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
            segments = transcriber.transcribe(
                file_path=file,
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
    else:
        log.info("Transcription disabled (--no-whisper).")

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
            frames = [
                f for idx, f in enumerate(frames)
                if not (idx / cfg["fps"] >= qa_start_raw
                       and (qa_end_raw is None or idx / cfg["fps"] < qa_end_raw))
            ]
            skipped = before_count - len(frames)
            if skipped:
                log.info("Q&A range configured (from %.0fs%s): skipped %d frame(s), no new "
                         "slides will be detected there.", qa_start,
                         f" to {qa_end:.0f}s" if qa_end else " to end of recording", skipped)

        changes = detector.detect_slide_changes(
            frames=frames, fps=cfg["fps"], threshold=cfg["hash_threshold"],
            algorithm=cfg["hash_algorithm"], min_slide_duration_sec=cfg["min_slide_duration_sec"],
            animation_threshold=cfg.get("animation_threshold", 0),
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
            for change in changes:
                dest = snap_tmp_dir / (change.frame_path.stem + ".png")
                try:
                    from PIL import Image
                    with Image.open(change.frame_path) as img:
                        img.save(dest, format="PNG")
                except Exception as exc:
                    log.warning("Could not save snapshot %s: %s", dest, exc)

            if cfg["enable_vlm"]:
                log.info("Running VLM annotation with model: %s", cfg["ollama_vlm_model"])
                slides_annotated = annotator.annotate_batch(
                    slides=changes, snapshot_dir=snap_tmp_dir,
                    model=cfg["ollama_vlm_model"],
                    ollama_url=cfg["ollama_base_url"].rstrip("/") + "/api/generate",
                    prompt=cfg["vlm_prompt"], timeout_sec=cfg["vlm_timeout_sec"],
                    stop_check=stop_check,
                )
            else:
                slides_annotated = [
                    {
                        "frame_index": c.frame_index, "timestamp_sec": c.timestamp_sec,
                        "snapshot_path": str(snap_tmp_dir / (c.frame_path.stem + ".png")),
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
                final_dest = _unique_snapshot_path(snapshot_dir, base_name, idx)
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
            if cfg["report_csv"]:
                reporter.save_csv(slides_annotated, slides_dir / f"{stem}_slides.csv")
            if cfg["report_json"]:
                reporter.save_json(slides_annotated, slides_dir / f"{stem}_slides.json")
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

            # Q&A section (task #61): if configured, keep the Q&A portion
            # out of the main transcript fed to the main prompt (so it
            # does not dilute the main summary), and collect it
            # separately for its own focused Q&A summary below. Times use
            # the already-converted real-time clock, same as segments
            # (segments were rescaled by recording_speed back in Step 1).
            qa_start = cfg.get("qa_start_time_sec")
            qa_end = cfg.get("qa_end_time_sec")
            qa_segments: list[dict] = []
            if qa_start:
                main_segments = [s for s in segments if s.get("start", 0) < qa_start]
                qa_segments = [
                    s for s in segments
                    if s.get("start", 0) >= qa_start and (not qa_end or s.get("start", 0) < qa_end)
                ]
                transcript_text = notes.build_transcript_text_from_segments(main_segments)
            elif enable_slides:
                transcript_text = notes.build_transcript_text_from_segments(segments)
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
            )

            if notes_text and qa_segments:
                qa_transcript_text = notes.build_transcript_text_from_segments(qa_segments)
                try:
                    qa_prompt_path = notes.resolve_prompt_path(PROMPTS_DIR, "qa_summary")
                except FileNotFoundError as exc:
                    log.warning("Q&A summary skipped: %s", exc)
                    qa_prompt_path = None
                if qa_prompt_path:
                    qa_summary = notes.generate_notes(
                        transcript_text=qa_transcript_text,
                        prompt_path=qa_prompt_path,
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
                    )
                    if qa_summary:
                        notes_text = notes_text.rstrip() + "\n\n" + qa_summary.strip()
                        log.info("Q&A summary appended (%d transcript segment(s), from %.0fs%s).",
                                 len(qa_segments), qa_start,
                                 f" to {qa_end:.0f}s" if qa_end else " to end of recording")
                    else:
                        log.warning("Q&A summary generation failed - main notes saved without it.")
            elif qa_start and not qa_segments:
                log.warning(
                    "qa_start_time_sec (%.0fs) is set but no transcript segments start at/after "
                    "it - check the value: it must be already-converted real time in seconds, "
                    "matching the transcript/slide report timestamps.", qa_start)

            if notes_text:
                title = cfg.get("meeting_title") or (prompt_path.stem.upper() + " NOTES")
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

    log.info("STAGE:done")
    log.info("DONE: %s", filename)
    return True


def _dominant_speaker(speaker_map: dict, timestamp: float) -> str:
    best, best_dist = "", float("inf")
    for spk, times in speaker_map.items():
        for t in times:
            d = abs(t - timestamp)
            if d < best_dist:
                best_dist = d
                best = spk
    return best


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
            prompt=cfg["vlm_prompt"], timeout_sec=cfg["vlm_timeout_sec"],
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
    qa_end_time_sec.

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
                         "qa_start_time_sec", "qa_end_time_sec")
    # "language" is a row-only convenience name; process_file's config key is
    # whisper_language, matching run_file_list's existing |lang convention.
    row_to_cfg_key = {"language": "whisper_language"}

    ok, errors, skipped = 0, 0, 0
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

    pattern_fn = folder.rglob if recursive else folder.glob
    files = sorted(
        p for ext in SUPPORTED_EXTENSIONS for p in pattern_fn(f"*{ext}")
    )
    if not files:
        log.info("No supported audio/video files found in: %s", folder)
        return

    log_file = folder / f"batch_folder_log_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt"
    log.info("BATCH (folder): %d file(s) in %s", len(files), folder)
    write_log(log_file, f"Batch folder: {datetime.now()} | {folder} | {len(files)} files")
    ok, errors = 0, 0
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


def run_notes_only(transcript_file: str, overrides: dict) -> None:
    """Generate notes from an existing *_transcript_speakers.txt (no whisperx)."""
    cfg = build_run_config(overrides)
    stem = Path(transcript_file).stem.replace("_transcript_speakers", "")
    output_prefix = resolve_output_prefix(str(Path(transcript_file).with_name(stem)), cfg)
    filename = Path(transcript_file).name
    transcript_text = transcriber.read_transcript_text(transcript_file)

    prompt_path = notes.resolve_prompt_path(PROMPTS_DIR, cfg["prompt_template"])
    notes_text = notes.generate_notes(
        transcript_text=transcript_text, prompt_path=prompt_path,
        llm_backend=cfg["llm_backend"], ollama_base_url=cfg["ollama_base_url"],
        ollama_notes_model=cfg["ollama_notes_model"], anthropic_api_key=cfg["anthropic_api_key"],
        claude_model=cfg["claude_model"], filename=filename,
        single_pass_limit=cfg["single_pass_limit"], chunk_size=cfg["chunk_size"],
        meeting_title=cfg.get("meeting_title", ""), meeting_date=cfg.get("meeting_date", ""),
    )
    if not notes_text:
        raise RuntimeError("Notes generation failed (see log above).")

    title = cfg.get("meeting_title") or (prompt_path.stem.upper() + " NOTES")
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
                   help="Read --meeting-title/--meeting-date from the first VEVENT in an .ics file "
                        "(explicit --meeting-title/--meeting-date still take precedence).")
    p.add_argument("--whisper-model", type=str, help="tiny|base|small|medium|large-v2|large-v3.")
    p.add_argument("--language", dest="language_flag", type=str, help="de|en|auto (flag form).")
    p.add_argument("--llm-backend", choices=["ollama", "anthropic"])
    p.add_argument("--no-slides", action="store_true", help="Skip slide detection/VLM even for a video file.")
    p.add_argument("--no-vlm", action="store_true", help="Detect slides but skip VLM annotation.")
    p.add_argument("--no-whisper", action="store_true", help="Skip transcription.")
    p.add_argument("--no-summary", action="store_true", help="Skip notes/summary generation.")
    p.add_argument("--force-retranscribe", action="store_true", help="Ignore transcript cache.")
    p.add_argument("--dry-run", action="store_true", help="Slide timestamps only, no annotation/whisper/reports.")
    p.add_argument("--diarize", action="store_true", help="Enable speaker diarization (requires HF_TOKEN).")
    p.add_argument("--threshold", type=int, help="Slide-change hash threshold override.")
    p.add_argument("--min-slide-duration", type=float, metavar="SEC",
                   help="Minimum seconds a slide must be shown to count as a real change "
                        "(debounce for animations/transitions). Default: 2.0.")
    p.add_argument("--qa-start", type=float, metavar="SEC",
                   help="Start of a Q&A section, in seconds (already-converted real time, "
                        "matching the transcript/slide report). No new slides are detected from "
                        "this point onward (to --qa-end, or the end of the recording). The "
                        "matching transcript gets a separate, focused Q&A summary "
                        "(prompts/qa_summary.md) appended to the notes, instead of being folded "
                        "into the main summary.")
    p.add_argument("--qa-end", type=float, metavar="SEC",
                   help="End of the Q&A section, in seconds. Only used together with --qa-start. "
                        "Default: to the end of the recording.")
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
    if args.ics:
        try:
            import ics_utils
            parsed = ics_utils.parse_ics(args.ics)
            meeting_title = meeting_title or parsed.get("title") or None
            meeting_date = meeting_date or parsed.get("date") or None
        except Exception as exc:
            log.warning("Could not read --ics %s: %s", args.ics, exc)

    overrides = {
        "prompt_template":     prompt_template,
        "whisper_model":       args.whisper_model,
        "whisper_language":    language,
        "llm_backend":         args.llm_backend,
        "enable_slides":       False if args.no_slides else mode_slides,
        "enable_vlm":          False if args.no_vlm else mode_vlm,
        "enable_whisper":      False if args.no_whisper else None,
        "no_summary":          True if args.no_summary else None,
        "force_retranscribe":  True if args.force_retranscribe else None,
        "dry_run":             True if args.dry_run else None,
        "enable_diarization":  True if args.diarize else None,
        "hash_threshold":      args.threshold,
        "min_slide_duration_sec": args.min_slide_duration,
        "animation_threshold": args.animation_threshold,
        "qa_start_time_sec": args.qa_start,
        "qa_end_time_sec": args.qa_end,
        "recording_speed":     args.recording_speed,
        "convert_video_to_realtime": True if args.normalize_speed else None,
        "title_slide_image_path": args.title_image,
        "fps":                 args.fps,
        "output_dir_override": args.output_dir,
        "notes_format_pdf":    True if args.pdf else None,
        "notes_format_docx":   True if args.docx else None,
        "meeting_title":       meeting_title,
        "meeting_date":        meeting_date,
        "meeting_comments":    args.comments,
        "output_basename_override": args.output_name,
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

    if args.reannotate_failed:
        retried, still_failed = reannotate_failed_slides(args.reannotate_failed, overrides)
        return 0 if still_failed == 0 else 1

    if args.notes_batch:
        run_notes_batch(args.notes_batch, overrides)
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

    ok = process_file(file, overrides)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
