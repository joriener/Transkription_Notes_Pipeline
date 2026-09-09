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

import shutil
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
    # final one). Set via --qa-start or the GUI "Q&A section" fields
    # (accepts seconds or mm:ss / hh:mm:ss there).
    "qa_start_time_sec": None,

    # End of the Q&A section, in seconds (same real-time convention as
    # qa_start_time_sec). None (default) = to the end of the recording.
    # Only used together with qa_start_time_sec. Set via --qa-end or the
    # GUI.
    "qa_end_time_sec": None,

    # Whether the Q&A section (qa_start_time_sec) is folded into the
    # SAME notes/summary prompt-pass as the rest of the recording
    # (True, default - its content lands in the prompt's own
    # "## QUESTIONS & ANSWERS" section, e.g. in prompts/webinar.md), or
    # dropped from the summary entirely (False - that section falls back
    # to its "No Q&A session" placeholder). Only used together with
    # qa_start_time_sec; there is no separate qa_summary.md prompt/pass
    # any more - one recording, one summary. Set via the GUI "Include
    # Q&A in the summary" checkbox or --qa-exclude-from-summary.
    "qa_include_in_summary": True,

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
    "ollama_vlm_model": _KEYS.get("OLLAMA_VLM_MODEL", "qwen3-vl:4b-instruct"),
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
    # "auto" resolves with the device: float16 on GPU, int8 on CPU. Any
    # explicit faster-whisper type is now honoured as given (it used to be
    # silently discarded by device resolution), so "int8_float16" is
    # reachable on GPU - typically 1.3-2x faster decode at roughly half the
    # VRAM for large-v3.
    "whisper_compute_type": "auto",     # auto | float16 | int8_float16 | int8 | float32

    # Bias Whisper towards GC/MS domain vocabulary. See vocabulary.py.
    "whisper_use_vocabulary": True,

    # Ignore the transcript cache and re-run whisperx even if
    # *_transcript_speakers.txt / *_segments.json already exist.
    "force_retranscribe": False,

    # Optional ffmpeg audio-cleanup pass before transcription (V1.24):
    # highpass filter for rumble/hum, mild denoise for background
    # hiss/fan noise, and loudness normalization for uneven volume
    # between speakers on the same recording. Off by default: audio
    # processing can occasionally change what Whisper transcribes, so
    # this is opt-in rather than silently altering every run. Only
    # affects the transcription pass itself, on a temporary copy - the
    # original file is never modified and is still what Play Sample and
    # voiceprint extraction read from directly. See
    # transcriber.enhance_audio.
    "enhance_audio": False,

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

    # Target length (seconds) of the audio preview clip extracted per
    # speaker for the Rename Speakers dialog's "Play sample"/"Improve
    # audio" buttons and the batch speaker-roster export's sample_clip
    # files (see speaker_id.extract_speaker_sample_clip). That helper
    # picks the speaker's longest segments first and concatenates as
    # many as needed to approach this length, rather than only ever
    # using one single segment - most utterances are shorter than this
    # on their own. Raise for a longer listen before naming a speaker;
    # lower to keep the batch export's samples folder smaller.
    "speaker_sample_seconds":   15.0,

    # =========================================================
    # NOTES / SUMMARY GENERATION  (LLM)
    # =========================================================
    # Filename (relative to prompts_dir) of the Markdown prompt
    # template to use. Any .md file dropped into prompts/ shows
    # up as a selectable option in the GUI - no code change needed.
    "prompt_template": "meeting.md",

    # Output language for the generated notes/summary AND the VLM slide
    # title/bullets, independent of whisper_language (which only affects
    # transcription accuracy/ASR). "auto" (default): no instruction is
    # added, the LLM/VLM responds in whichever language comes naturally
    # (usually matching the transcript/slide). Set to "en", "de", etc. to
    # force notes and slide annotations into that language regardless of
    # the source recording's language, e.g. transcribe an English call but
    # get German notes. The verbatim transcript file itself
    # (*_transcript_speakers.txt/.srt) is never translated and always
    # reflects whisper's own output. See notes.language_instruction and
    # annotator.build_prompt. Set via --output-language or the GUI's
    # "Output language" dropdown next to Language.
    "output_language": "auto",

    # "ollama" (local, free) or "anthropic" (Claude API)
    "llm_backend": _KEYS.get("LLM_BACKEND", "ollama"),

    "ollama_base_url":   _KEYS.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    # Default text model for notes/summary. qwen3:14b as of 2026-07-02
    # (same VRAM budget as qwen2.5:14b, better instruction following).
    # Override with OLLAMA_MODEL in keys.cfg if you need a different model.
    "ollama_notes_model": _KEYS.get("OLLAMA_MODEL", "qwen3:14b"),
    # Explicit context window for the notes/summary Ollama calls (task:
    # investigated 2026-07-18 after notes came back empty/generic - see
    # notes._chat_ollama's docstring). Sized to comfortably hold
    # single_pass_limit/chunk_size worth of transcript text plus the
    # response; NOT left unset, since Ollama's own per-model default
    # varies (some models default too small, silently truncating the
    # transcript before the model ever sees it - symptom: notes claim
    # "no transcript provided" despite a real one being sent). Lower this
    # if VRAM is tight and your transcripts are short; raise it if you
    # also raise single_pass_limit/chunk_size below.
    "ollama_notes_num_ctx": 16_384,
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
    # DOCUMENT TRANSLATION  (Translate tab/CLI: txt/docx/pptx/pdf/images)
    # =========================================================
    # Target language for the Translate tab/CLI. Unlike output_language
    # above, "auto" is not meaningful here - a translation run always
    # needs a real target, so the GUI dropdown/CLI flag default to a
    # concrete code rather than "auto". See translator.translate_text
    # and doc_translate.py.
    "translate_target_language": "en",

    # Ollama model used for document translation. Falls back to
    # ollama_notes_model above if left blank, so most users need not set
    # this separately; override only if you prefer a different model
    # specifically for translation (e.g. a smaller/faster one, or a
    # model marketed for translation quality).
    "ollama_translate_model": _KEYS.get("OLLAMA_TRANSLATE_MODEL", ""),

    # Explicit context window for translation Ollama calls (see
    # notes.py's ollama_notes_num_ctx comment for why this is not left
    # unset). Smaller than ollama_notes_num_ctx since translation units
    # are normally one paragraph/slide-text-frame at a time, not a whole
    # transcript; raise it if you translate very long unstructured .txt
    # files with few paragraph breaks.
    "ollama_translate_num_ctx": 8_192,

    # Character size above which translate_text splits a text unit into
    # multiple chunks (paragraph-aware, see translator._chunk_text).
    # Only engages for large plain .txt/.pdf/OCR text - docx/pptx
    # paragraphs are almost always far shorter than this already.
    "translate_chunk_size": 6_000,

    # OCR source language for image inputs (.jpg/.png/etc.) in the
    # Translate tab/CLI. NOT the translation target - this tells
    # Tesseract what script/language to expect while READING the image,
    # before translation ever happens. Wrong here means garbled/gibberish
    # OCR text (e.g. German umlauts misread under an English-only OCR
    # pass), which then gets faithfully "translated" into more garbage -
    # this looks like a translation bug but is actually an OCR-language
    # problem. "auto" (default) is not true language identification
    # (Tesseract has no reliable way to do that) - it is a fixed,
    # reasonable default of "eng+deu" (English technical terms in
    # German-language slides is the common case here). Set to a code
    # from LANGUAGE_NAMES below (mapped via TESSERACT_LANG_MAP) for a
    # single, more accurate targeted OCR pass, or type a raw Tesseract
    # --lang string directly (e.g. "deu+fra") for anything not in that
    # map. Run `tesseract --list-langs` to see which language packs are
    # actually installed; missing ones fall back to Tesseract's own
    # default and log a warning (see doc_translate._ocr_image_text).
    "translate_ocr_language": "auto",

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

    # Opt-in alternative to use_filename_date_heading above: when True AND
    # meeting_title is set (Meeting info fields, --meeting-title, --ics
    # import, a .txt/.docx "Title:" line, or a correlate_calendar_recordings.py
    # batch CSV), the output basename becomes "YYYYMMDD_Subject" - date
    # first, then the meeting/webinar title, e.g. "20260703_Q3 Kickoff".
    # Takes priority over use_filename_date_heading (but NOT over an
    # explicit output_basename_override, which always wins outright) -
    # see run_pipeline.build_date_subject_filename/resolve_output_prefix.
    # Falls back silently to the use_filename_date_heading behaviour when
    # meeting_title is blank, so turning this on is safe even for a batch
    # where only some files have a matched title.
    #
    # Date resolution order: 1) meeting_date if set (ISO "YYYY-MM-DD",
    # e.g. from an .ics import or calendar correlation) - 2) a date
    # embedded in the source filename - 3) the source file's own
    # last-modified date. Subject is sanitized for filesystem-illegal
    # characters (\\ / : * ? " < > |) and truncated to keep the whole
    # filename a reasonable length.
    "use_date_subject_filename": False,

    # =========================================================
    # POST-RUN ARCHIVING
    # =========================================================
    # Opt-in: after a file finishes successfully (transcript + notes, or
    # transcript-only with --no-summary - never on failure, never on a
    # video --dry-run), move the ORIGINAL source audio/video file into a
    # processed_subfolder_name subfolder next to it. Every already-written
    # output (transcript, notes, slide report, DB rows) stays exactly
    # where it was - only the source recording itself relocates, so a
    # source folder full of raw recordings gradually empties out into a
    # tidy "already done" subfolder instead of growing forever.
    #
    # Safe by construction: run_pipeline.move_processed_source_file
    # rewrites the *_source_media.txt sidecar and the transcripts/slides
    # DB rows to the new path, so Play Sample, "Improve audio", and the
    # Rename Speakers dialog's voiceprint suggestions keep working for
    # already-processed meetings exactly as before. run_batch_folder
    # (--batch-folder / GUI "Process folder", recursive or not) also
    # excludes this subfolder from its own file scan, so an archived file
    # is never picked up and reprocessed by a later run over the same
    # parent folder. Off by default: moving files is a bigger behaviour
    # change than any other option on this page, worth trying deliberately
    # before leaving it on for an unattended batch.
    "move_processed_files": False,

    # Subfolder name (created next to each source file, not a single
    # global folder - so this works the same whether your recordings for
    # different months/projects live in one flat folder or many). Rename
    # to whatever fits your own convention.
    "processed_subfolder_name": "_processed",

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

# Display names for output_language (and whisper_language) codes, used to
# turn a short code into a clear instruction for the notes LLM and the VLM
# (e.g. "de" -> "German"). Keep in sync with gui.py's LANGUAGES list.
# "auto" maps to "" since it means "no language instruction", not a real
# language name.
LANGUAGE_NAMES = {
    "auto": "",
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "ja": "Japanese",
    "zh": "Chinese",
    "nl": "Dutch",
    "uk": "Ukrainian",
    "pt": "Portuguese",
}

# Tesseract OCR uses ISO 639-2 three-letter language codes, NOT the
# two-letter codes above - this maps our short codes to Tesseract's
# --lang values for the Translate tab/CLI's image inputs. See
# translate_ocr_language's comment above and doc_translate.py.
TESSERACT_LANG_MAP = {
    "en": "eng",
    "de": "deu",
    "fr": "fra",
    "es": "spa",
    "it": "ita",
    "ja": "jpn",
    "zh": "chi_sim",
    "nl": "nld",
    "uk": "ukr",
    "pt": "por",
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


# =========================================================
# BUNDLED FFMPEG  (optional, portable install alongside PATH)
#
# Drop a full ffmpeg build (as downloaded from https://ffmpeg.org/download.html,
# e.g. a gyan.dev "full" Windows build: ffmpeg.exe, ffplay.exe, ffprobe.exe,
# a doc/ folder, presets/) into <project root>\ffmpeg\bin\ and every ffmpeg/
# ffplay/ffprobe call in this project (extractor.py, speaker_id.py,
# transcriber.py, gui.py) prefers it over PATH - no system-wide ffmpeg
# install or PATH edit needed. Falls back to PATH ("ffmpeg"/"ffplay"/
# "ffprobe") when the bundled copy is not present, and to None (caller
# decides how to degrade) when neither is found.
# =========================================================
_BUNDLED_FFMPEG_DIR = _HERE / "ffmpeg" / "bin"

# Prepend the bundled ffmpeg\bin dir to THIS PROCESS's PATH, not just to the
# calls in this project that go through get_ffmpeg_path() below. Libraries
# this project depends on (whisperx's wx.load_audio, torchaudio, pyannote's
# VAD) shell out to a bare "ffmpeg"/"ffprobe" internally and rely on PATH
# alone - they never see the bundled-path preference below. If the process's
# own PATH does not contain ffmpeg (e.g. GUI/scheduled-task launch with a
# stale or different environment than an interactive shell), those internal
# calls fail with WinError 2 ("cannot find the specified file") even though
# a fresh terminal's `ffmpeg -version` works fine and this project's own
# ffmpeg-aware calls (extractor.py, speaker_id.py, transcriber.enhance_audio)
# succeed via the bundled copy. Doing this once, at import time, fixes it
# for every subprocess call anywhere in the process.
if _BUNDLED_FFMPEG_DIR.is_dir():
    import os as _os
    _os.environ["PATH"] = str(_BUNDLED_FFMPEG_DIR) + _os.pathsep + _os.environ.get("PATH", "")


def _find_bundled_tool(name: str) -> str | None:
    """name without extension, e.g. "ffmpeg" - checks both "<name>.exe"
    (Windows, the expected case for this project) and the bare name
    (in case a non-Windows build is ever dropped in)."""
    for candidate in (_BUNDLED_FFMPEG_DIR / f"{name}.exe", _BUNDLED_FFMPEG_DIR / name):
        if candidate.is_file():
            return str(candidate)
    return None


def get_ffmpeg_path() -> str | None:
    """Full path to a bundled ffmpeg, else "ffmpeg" if found on PATH,
    else None (not available at all)."""
    return _find_bundled_tool("ffmpeg") or shutil.which("ffmpeg")


def get_ffplay_path() -> str | None:
    """Same as get_ffmpeg_path, for ffplay - used to play audio clips
    directly (e.g. speaker sample previews) without going through the
    OS's default media player/app."""
    return _find_bundled_tool("ffplay") or shutil.which("ffplay")


def get_ffprobe_path() -> str | None:
    """Same as get_ffmpeg_path, for ffprobe."""
    return _find_bundled_tool("ffprobe") or shutil.which("ffprobe")
