# =============================================================
#  Transkription_Notes_Pipeline V1.0 - config.py
#  Central configuration. Combines the former
#  Audio_Transkription_Notes_Pipeline (meeting/webinar notes from
#  audio-only or video files) and Video_Transkription_Notes_Pipeline
#  (slide-change detection + VLM annotation + webinar summary)
#  into a single tool.
#
#  Runtime: Python 3.11  (C:\Python\Python311\python.exe)
#  Secrets are loaded from keys.cfg (see keys.cfg.example).
#  Edit this file to change defaults; most values can also be
#  overridden per run via CLI flags or the GUI (gui.py).
# =============================================================

from pathlib import Path
from keys_loader import load_keys

# Project root: directory this config.py lives in. All default
# paths are relative to it so the project is portable.
_HERE = Path(__file__).parent.resolve()

# Load secrets from keys.cfg (git-ignored). Missing file: warning only.
_KEYS = load_keys()

CONFIG = {

    # =========================================================
    # INPUT
    # =========================================================
    # Path to the file to process. Normally set via CLI argument
    # or the GUI, not edited here.
    "input_path": "",

    # =========================================================
    # PIPELINE STAGES - what runs
    # =========================================================
    # Slide detection + VLM annotation only makes sense for videos
    # that actually contain visual slides (screen recordings,
    # webinars). It is automatically disabled for audio-only
    # files (mp3, wav, m4a, ogg, flac, aac, wma) regardless of
    # this setting. For plain talking-head recordings, set to
    # False to skip straight to transcription + notes.
    "enable_slides":   True,

    # VLM annotation of detected slides. Only relevant when
    # enable_slides is True. Set False for a fast "detect only"
    # run (snapshots + timestamps, no Ollama vision calls).
    "enable_vlm":      True,

    # Speech-to-text transcription. Rarely disabled, but useful
    # to skip when only slide snapshots are needed.
    "enable_whisper":  True,

    # LLM-generated meeting/webinar notes from the transcript.
    # Set False for a transcript-only run (--no-summary).
    "enable_notes":    True,

    # =========================================================
    # RECORDING SPEED  (applies to every timestamp everywhere:
    # slide report, timing summary, transcript .txt, .srt, and DB)
    # =========================================================
    # Some screen recorders capture at a different speed than real
    # time (e.g. frame-skipping to shrink the file). Set this to the
    # recording's speed factor so every timestamp in every output
    # reflects the actual wall-clock time of the event instead of
    # the video's own internal clock:
    #   real_time_sec = video_time_sec / recording_speed
    # 1.0 = no change (default). Example: recording_speed = 1.5 means
    # the video plays 1.5x faster than real life, so a slide detected
    # at 0:10:00 in the video is shown as 0:06:40 everywhere.
    # Applied once, right after a fresh transcription/slide-detection
    # pass; a cached transcript is NOT re-scaled, so use
    # --force-retranscribe if you change this after the first run.
    "recording_speed": 1.0,

    # Optional, opt-in: when recording_speed != 1.0, also produce a
    # re-encoded copy of the video that plays at actual real-time (1x)
    # speed, so it stays in sync with the already-converted .srt/transcript
    # instead of drifting. Written as <stem>_realtime.mp4 next to the other
    # outputs; the original source file is never modified. Off by default
    # because it requires a full ffmpeg re-encode (slow for long videos).
    # Set via --normalize-speed or the GUI checkbox next to Recording speed.
    "convert_video_to_realtime": False,

    # =========================================================
    # TITLE SLIDE  (optional cover image for the slide report)
    # =========================================================
    # Path to an image file (jpg/png) shown as a cover page before
    # Slide 1 in the HTML/PDF slide report, captioned with
    # meeting_title/meeting_date below. Purely cosmetic: not counted
    # in "Slides detected" and never written to the CSV/JSON slide
    # index. Blank (default) = no cover page. Set via --title-image
    # or the GUI "Title slide image" field.
    "title_slide_image_path": "",

    # =========================================================
    # FRAME EXTRACTION  (ffmpeg, video mode only)
    # =========================================================
    "fps":            1,      # frames/sec extracted; raise to 2 for fast slide decks
    "frame_format":   "jpg",  # jpg (smaller/faster) or png (lossless)
    "frame_quality":  90,     # JPEG quality 1-95, ignored for png

    # =========================================================
    # SLIDE-CHANGE DETECTION  (imagehash, video mode only)
    # =========================================================
    "hash_threshold":          8,        # 0=identical .. 64=max; raise to 10-12 for webcam overlay
    "hash_algorithm":          "phash",  # phash | dhash | whash | ahash
    "min_slide_duration_sec":  2.0,      # debounce for rapid transitions/animations; also the noise
                                          # filter window: a change must stay on screen this long to
                                          # count as real, so a brief flash/glitch is discarded.
    "animation_threshold":     4,        # 0=disabled. Must be < hash_threshold. Distances in this
                                          # band are treated as "same slide still building" (e.g.
                                          # bullets appearing one at a time): no new slide is recorded,
                                          # the current slide's snapshot is updated to the more complete
                                          # frame, but its start timestamp stays the same. Default: half
                                          # of hash_threshold.

    # =========================================================
    # Q&A SECTION  (optional, e.g. the Q&A block at the end of a
    # webinar where only the speakers are shown, no slides)
    # =========================================================
    # Start of the Q&A section, in seconds, using the ALREADY-CONVERTED
    # real time also shown everywhere else (transcript, slide report;
    # i.e. after recording_speed is applied). None/0 = disabled
    # (default). From this point onward (to qa_end_time_sec, or the end
    # of the recording if that is not set): no new slides are detected
    # (frames in this range are excluded before slide-change detection
    # runs, so the last slide shown before the range simply stays as the
    # final one), and the matching transcript is summarised separately
    # by prompts/qa_summary.md and appended to the notes, instead of
    # being folded into the main summary. Set via --qa-start or the GUI
    # "Q&A section" fields (accepts seconds or mm:ss / hh:mm:ss there).
    "qa_start_time_sec": None,

    # End of the Q&A section, in seconds (same real-time convention as
    # qa_start_time_sec). None (default) = to the end of the recording.
    # Only used together with qa_start_time_sec. Set via --qa-end or the
    # GUI.
    "qa_end_time_sec": None,

    # =========================================================
    # EXTERNAL TOOLS
    # =========================================================
    "losslesscut_path": "",  # Full path to LosslessCut.exe (free, MIT licence,
                              # https://github.com/mifi/lossless-cut). Used by the GUI's
                              # "Open in LosslessCut" button to launch it with the selected source
                              # file preloaded, for manual cut/mute editing. Leave blank to
                              # auto-detect common install locations, or browse for it once when
                              # prompted (remembered for the session).

    "db_browser_path": "",  # Full path to DB Browser for SQLite (free, MIT licence,
                             # https://sqlitebrowser.org/). Used by the GUI Settings tab's
                             # "Open in DB Browser for SQLite" button (task #75) to inspect/edit
                             # the pipeline database directly, opened against db_path. Leave
                             # blank to auto-detect common install locations, or browse for it
                             # once when prompted (remembered for the session).

    # =========================================================
    # VLM SLIDE ANNOTATION  (Ollama vision model, video mode only)
    # =========================================================
    "ollama_vlm_model": _KEYS.get("OLLAMA_VLM_MODEL", "qwen2.5vl:7b"),
    "vlm_prompt": (
        "Analyse this presentation slide. "
        "Reply with ONLY a valid JSON object, no other text. "
        "Keys: title (string, max 10 words), "
        "bullets (array of strings, max 5 items, max 8 words each), "
        "slide_type (one of: title|content|diagram|table|blank). "
        'Example: {"title":"My Slide","bullets":["Point one"],"slide_type":"content"}'
    ),
    "vlm_timeout_sec": 60,

    # =========================================================
    # SPEECH TRANSCRIPTION  (whisperx)
    # =========================================================
    # tiny | base | small | medium | large-v2 | large-v3
    "whisper_model":    "large-v3",

    # "auto" = auto-detect, or "en" / "de" / ... for a fixed language.
    "whisper_language": "auto",

    # "auto" resolves to cuda if available, else cpu (see transcriber.py).
    "whisper_device":      "auto",
    "whisper_batch_size":  16,          # reduce to 4-8 on CPU / low VRAM
    "whisper_compute_type": "float16",  # float16 (GPU) | int8 (CPU)

    # Bias Whisper towards GC/MS domain vocabulary. See vocabulary.py.
    "whisper_use_vocabulary": True,

    # Ignore the transcript cache and re-run whisperx even if
    # *_transcript_speakers.txt / *_segments.json already exist.
    "force_retranscribe": False,

    # =========================================================
    # SPEAKER DIARIZATION  (pyannote, optional)
    # =========================================================
    "enable_diarization":       True,
    "hf_token":                 _KEYS.get("HF_TOKEN", ""),
    "diarization_min_speakers": None,
    "diarization_max_speakers": None,

    # =========================================================
    # SPEAKER IDENTIFICATION (task #79, pyannote voiceprints, optional)
    #
    # Builds a global, cross-meeting roster of known speakers in
    # known_speakers (see db.py). Requires enable_diarization above (there
    # is nothing to identify without per-segment speaker labels first) and
    # the same hf_token, since it reuses the pyannote.audio install that
    # whisperx already pulls in. speaker_id_threshold is the minimum cosine
    # similarity (0-1) to auto-suggest an existing name instead of treating
    # a voice as new; the Rename Speakers dialog always shows the
    # suggestion for confirmation rather than applying it silently.
    # =========================================================
    "enable_speaker_id":        True,
    "speaker_id_threshold":     0.75,

    # =========================================================
    # NOTES / SUMMARY GENERATION  (LLM)
    # =========================================================
    # Filename (relative to prompts_dir) of the Markdown prompt
    # template to use. Any .md file dropped into prompts/ shows
    # up as a selectable option in the GUI - no code change needed.
    "prompt_template": "meeting.md",

    # "ollama" (local, free) or "anthropic" (Claude API)
    "llm_backend": _KEYS.get("LLM_BACKEND", "ollama"),

    "ollama_base_url":   _KEYS.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    # Default text model for notes/summary. qwen3:14b as of 2026-07-02
    # (same VRAM budget as qwen2.5:14b, better instruction following).
    # Override with OLLAMA_MODEL in keys.cfg if you need a different model.
    "ollama_notes_model": _KEYS.get("OLLAMA_MODEL", "qwen3:14b"),
    "anthropic_api_key":  _KEYS.get("ANTHROPIC_API_KEY", ""),
    # claude-sonnet-5 is the current balanced default. Use claude-opus-4-8
    # for long, topic-dense transcripts where quality matters most, or
    # claude-haiku-4-5 for fast/cheap runs on short meetings.
    "claude_model":       "claude-sonnet-5",

    # Map-reduce chunking for long transcripts (Ollama backend).
    # Transcripts shorter than single_pass_limit: one LLM call.
    # Longer: split into chunk_size pieces, each summarised, then merged.
    "single_pass_limit": 20_000,   # chars
    "chunk_size":        12_000,   # chars per chunk

    # =========================================================
    # OUTPUT FILES
    # =========================================================
    # Default behaviour: everything is written next to the source file
    # (portable, works from any folder or NAS share). Set output_dir_override
    # (or --output-dir / the GUI field) to redirect ALL outputs for a run -
    # transcript, notes, and slide report - into one folder instead, using
    # the same <stem>_... naming inside it.
    "output_dir_override": "",

    # Base filename used for every output file instead of the source
    # file's stem (transcript, notes, *_slides/ folder, snapshots, CSV/
    # JSON/HTML/PDF). Blank (default) keeps the source filename. Set via
    # --output-name or the GUI "Output filename" field.
    "output_basename_override": "",

    # When output_basename_override is blank, derive the shared output
    # prefix (and, separately, the notes heading when meeting_title is
    # blank) from a date/time found in the source filename instead of
    # the raw filename itself - e.g. "Video_2020-04-07_154005.mp4"
    # becomes "2020-04-07_154005" (see
    # run_pipeline.derive_heading_from_filename). Falls back to the raw
    # filename stem if no recognizable date is found. Set to False to
    # keep the original raw-filename-based naming.
    "use_filename_date_heading": True,

    # Legacy internal path, used only as a fallback base for the video-mode
    # temp frame extraction directory when no per-run output dir is set.
    "output_dir":      str(_HERE / "output"),
    "snapshot_subdir": "snapshots",

    # Notes/summary formats - each independently toggleable.
    "notes_format_txt":  True,   # plain text with header
    "notes_format_html": True,   # styled, print-friendly HTML
    "notes_format_pdf":  False,  # PDF via reporter.save_pdf_from_html (Playwright/weasyprint/pdfkit)
    "notes_format_docx": False,  # real Word document via python-docx

    # Slide-report formats (video mode only, each independently toggleable).
    "report_html": True,   # slide-index HTML report with thumbnails
    "report_csv":  True,   # slide-index CSV (Excel-compatible)
    "report_json": True,   # full slide-index JSON
    "report_srt":  True,   # SRT subtitle file from whisperx segments
    "report_pdf":  True,   # PDF versions (Playwright preferred, weasyprint/pdfkit fallback)
    "report_slide_timing": True,  # plain slide-number + timestamp list (txt), no images/bullets

    # Zip the snapshots/ folder into <stem>_snapshots.zip after a video
    # run. Off by default (report_html/report_pdf already embed the
    # images); useful when sharing just the slide images with someone
    # who does not need the full report.
    "zip_snapshots": False,

    # Slide report content selection (report_html/report_pdf only; CSV/
    # JSON always contain the full data regardless of these flags). Lets
    # a review-only report be built without images, or a slide-timing
    # style report without a wall of transcript text.
    "report_show_image":      True,
    "report_show_bullets":    True,
    "report_show_transcript": True,

    # "full" = complete aligned transcript segment per slide.
    # "first_sentence" = only the first sentence, so a slide with a long
    # spoken segment still fits on one PDF page.
    "report_transcript_mode": "full",

    # =========================================================
    # DATABASE
    # =========================================================
    "db_path": str(_HERE / "transkription_notes_pipeline.db"),

    # =========================================================
    # PROMPTS
    # =========================================================
    "prompts_dir": str(_HERE / "prompts"),

    # =========================================================
    # DRY RUN / LOGGING
    # =========================================================
    # Detect slide changes only; skip VLM, whisper, db, reports.
    "dry_run": False,

    # DEBUG | INFO | WARNING | ERROR
    "log_level": "INFO",

    # Also write each run's log to a timestamped file under log_dir,
    # in addition to the console/GUI log panel. Off by default so the
    # project folder does not accumulate files with a default install.
    "log_to_file": False,
    "log_dir": str(_HERE / "logs"),

    # =========================================================
    # MEETING INFO  (optional, GUI "Meeting info" section or
    # --meeting-title / --meeting-date / --ics on the CLI)
    # =========================================================
    # Shown in the notes header/meta and passed to the LLM as context
    # so summaries reference the actual meeting name/date rather than
    # just the source filename. Both blank by default (falls back to
    # the filename and today's date, see notes.py/reporter.py).
    "meeting_title": "",
    "meeting_date": "",

    # Free-text notes added by the user (e.g. context not captured by
    # the recording itself). Shown in the notes header and the slide
    # report header, alongside title/date. Blank by default.
    "meeting_comments": "",

    # =========================================================
    # GUI THEME (colors)
    #
    # gui.py's apply_theme() reads gui_theme below and applies it to every
    # ttk widget (buttons, tabs, treeviews, entries, checkboxes, progress
    # bars, scrollbars) plus the handful of plain tk widgets that ttk
    # styling can't reach (the log panel, template editor, requirements
    # output, the Templates list, and the slide thumbnail-grid canvas).
    #
    # To customize: edit the hex values below (each *_100 is the
    # lightest shade of that color, *_900 the darkest) and restart the
    # GUI. Set gui_theme_enabled to False to fall back to your system's
    # native Tk look with no custom colors at all.
    #
    # gui_ttk_theme must be "clam" (or another fully Tk-drawn theme) for
    # the colors below to actually show up: Windows' native "vista"/
    # "winnative" ttk themes ignore most color options since widgets are
    # drawn by the OS theming engine, not by Tk.
    # =========================================================
    "gui_theme_enabled": True,
    "gui_ttk_theme": "clam",
    "gui_theme": {
        # Primary: Bright Navy Blue - buttons, selected tab, headings,
        # progress bar, focus highlights.
        "primary_100": "#F2F6FF",
        "primary_200": "#BCD2FE",
        "primary_300": "#84B1F9",
        "primary_400": "#4B92EB",
        "primary_500": "#1773CF",
        "primary_600": "#085FA4",
        "primary_700": "#024C7A",
        "primary_800": "#003650",
        "primary_900": "#001C26",
        # Accent: Lime Shot - checked/selected indicators, "on" states.
        "accent_100": "#F2FFF5",
        "accent_200": "#BCFFC8",
        "accent_300": "#85FD93",
        "accent_400": "#4EFA58",
        "accent_500": "#18F218",
        "accent_600": "#13BF08",
        "accent_700": "#128C02",
        "accent_800": "#105900",
        "accent_900": "#092600",
        # Neutral - window/frame backgrounds, body text, borders.
        "neutral_100": "#FAFAFC",
        "neutral_200": "#E8E9EC",
        "neutral_300": "#D7D8DB",
        "neutral_400": "#C6C7CB",
        "neutral_500": "#B5B7BA",
        "neutral_600": "#8E9195",
        "neutral_700": "#696D70",
        "neutral_800": "#45494B",
        "neutral_900": "#222526",
    },
}

# ---------------------------------------------------------------------------
# Derived paths and constants - computed from CONFIG, do not edit directly.
# ---------------------------------------------------------------------------
OUTPUT_DIR   = Path(CONFIG["output_dir"])
SNAPSHOT_DIR = OUTPUT_DIR / CONFIG["snapshot_subdir"]
FRAMES_DIR   = OUTPUT_DIR / "_frames_tmp"
PROMPTS_DIR  = Path(CONFIG["prompts_dir"])

# Extensions that never contain a video stream: slide detection and
# VLM annotation are automatically skipped for these regardless of
# enable_slides.
AUDIO_ONLY_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"}

# Extensions treated as video (candidates for slide detection).
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}

# All supported input extensions (used by the GUI file picker).
SUPPORTED_EXTENSIONS = AUDIO_ONLY_EXTENSIONS | VIDEO_EXTENSIONS


def is_audio_only(path: str) -> bool:
    """True if the file extension indicates no video stream is possible."""
    return Path(path).suffix.lower() in AUDIO_ONLY_EXTENSIONS
