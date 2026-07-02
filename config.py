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
    "min_slide_duration_sec":  2.0,      # debounce for rapid transitions/animations

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
    "enable_diarization":       False,
    "hf_token":                 _KEYS.get("HF_TOKEN", ""),
    "diarization_min_speakers": None,
    "diarization_max_speakers": None,

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
