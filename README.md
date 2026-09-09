# Transkription_Notes_Pipeline

**Combined transcription + AI notes pipeline for audio and video recordings**,
with optional slide-change detection for recorded webinars/presentations.

Version 1.6, 2026-07-18. Merges two previously separate tools into one codebase:

- `Audio_Transkription_Notes_Pipeline` - WhisperX transcription + speaker
  diarization + LLM meeting/webinar notes (audio or video, no slide analysis)
- `Video_Transkription_Notes_Pipeline` - slide-change detection, VLM slide
  annotation, and an AI-generated webinar summary

Both original projects are left untouched on disk for reference; this is a
new, independent tool.

---

## Quick start

```powershell
# 1. Install dependencies
.\install.ps1

# 2. Copy the secrets template and fill in your values
Copy-Item keys.cfg.example keys.cfg

# 3. Launch the GUI
C:\Python\Python311\python.exe run_pipeline.py --gui
```

Pick a file, choose your parameters, click Run. Everything below is detail
for when the defaults are not enough.

---

## What's new in V1.1 (2026-07-02)

- **Output folder override**: `--output-dir` / a GUI field redirects every
  output for a run into one folder instead of writing next to the source
  file. Leave it blank to keep the default (next to source).
- **Folder auto-discovery batch mode**: `--batch-folder D:\Videos` or the
  GUI's "Folder (auto-discover)" mode processes every supported audio/video
  file found directly in a folder, no hand-written `list.txt` needed.
- **Settings tab in the GUI**: `keys.cfg` is now editable as a structured
  form (masked API key fields, dropdowns) instead of a text file. An "Open
  config.py in editor" button covers everything else.
- **Per-format output selection**: notes txt/html/pdf/docx and slide-report
  html/csv/json/srt/pdf are each independently toggleable, in the GUI or
  via `--pdf` / `--docx` on the CLI.
- **Real Word (.docx) notes export** via `python-docx`, alongside the
  existing HTML.
- **PDF backend changed to Playwright/Chromium first**, falling back to
  weasyprint then pdfkit. See "PDF export" below for why.
- **Model defaults refreshed**: `claude-sonnet-5` (was a stale
  `claude-sonnet-4-20250514` string); `qwen3:14b` is now the default
  Ollama notes model (was `qwen2.5:14b`), set via `OLLAMA_MODEL` in
  `keys.cfg`.

---

## What's new in V1.2 (2026-07-02)

- **Full-text search (FTS5)**: `python run_pipeline.py --search "retention index"`
  or the GUI's new **Search tab** searches generated notes and slide
  titles/bullets across every processed file. Falls back to a plain `LIKE`
  query automatically if the local sqlite3 build lacks FTS5.
- **`--file-list` resume/retry**: batch runs now skip files that already
  have notes generated in the database, matching `--notes-batch`'s existing
  behaviour. An interrupted batch is safe to just re-run; `--force-retranscribe`
  overrides the skip.
- **GUI per-stage progress bar**: the Run tab shows which stage is active
  (transcribing / detecting slides / generating notes) instead of only the
  raw log stream.
- **Batch input via paste box**: File list (batch) mode now has a paste box
  in the GUI, plus "Load .txt/.csv" and "Save box to .txt" buttons, as an
  alternative to hand-editing `list.txt`.
- **Requirements check**: Settings tab has "Check requirements" (native
  Python import check, no PowerShell needed) and "Install missing", which
  pip-installs exactly what's missing.
- **Database file management**: Settings tab can open an existing `.db`,
  create a new one, or rename the current one, with live stats (rows,
  size). `db_path` is a per-session override, same pattern as the output
  folder.
- **Logging options**: log level selector and an optional "save this run's
  log to a file" toggle, both in the GUI Settings tab and as `--log-file`
  / `log_level` / `log_to_file` / `log_dir` in `config.py`.
- **NAS/UNC path review**: see "NAS and network paths" below.

---

## What's new in V1.3 (2026-07-02)

- **Recording-type presets**: a "Recording type" dropdown in the GUI (Audio
  Transcript / Video Transcript / Meeting Transcript / Webinar Transcript)
  sets the prompt template and slide-detection/VLM defaults in one click.
  Same shortcut on the CLI via `--mode audio_transcript|video_transcript|
  meeting|webinar`. Fields stay editable afterward; this only fills in a
  sensible starting point. Two new generic templates, `prompts/
  audio_transcript.md` and `prompts/video_transcript.md`, avoid assuming
  the recording is a meeting or webinar when it is neither.
- **Meeting info (title/date/.ics import)**: a "Meeting info" section in the
  GUI Run tab (Title, Date, "Load from .ics...") or `--meeting-title` /
  `--meeting-date` / `--ics` on the CLI. The title/date are shown in the
  notes header (txt/html/docx) and passed to the LLM as context so the
  summary references the actual meeting name/date instead of just the
  source filename. `.ics` import (`ics_utils.py`, stdlib only, no new
  dependency) reads SUMMARY/DTSTART from the first VEVENT in an Outlook/
  Google/Teams invite file.
- **Friendly slide snapshot filenames**: each detected slide is now also
  saved as `<video_stem>_slide001.png`, `_slide002.png`, etc., alongside
  the original ffmpeg-derived `frame_NNNNNN.png`. The HTML/PDF slide
  report, CSV, JSON, and database all reference the friendly name, so
  copying an image out of the report gives a readable filename instead of
  a cryptic `frame_000218.png` path. Each slide card in the HTML report
  also always shows a "Slide N" number badge.
- **Slide timing summary report**: a new lightweight `<stem>_slide_timing.txt`
  alongside the full HTML/PDF/CSV/JSON slide report, listing only the
  slide number and the timestamp it appeared at (no images, bullets, or
  transcript text) for a quick "when did slide N change" reference.
  Toggle via the GUI's "timing summary (txt)" checkbox or `report_slide_timing`
  in `config.py`.
- **Default Ollama notes model changed to `qwen3:14b`** (was `qwen2.5:14b`):
  same VRAM budget, better instruction following. Override with
  `OLLAMA_MODEL` in `keys.cfg` if you prefer the previous default.

---

## What's new in V1.4 (2026-07-18)

- **Output language, independent of transcription language**: a new
  "Output language" dropdown in the GUI (next to "Language") or
  `--output-language` on the CLI forces the generated notes/summary AND
  the VLM slide title/bullets into a chosen language, regardless of the
  language actually spoken/shown in the recording - e.g. transcribe an
  English call but get German notes. `"auto"` (default) keeps the
  previous behaviour: no instruction added, notes/slides follow the
  transcript's/slide's own language. The verbatim transcript file itself
  (`*_transcript_speakers.txt` / `.srt`) is never translated and always
  reflects whisper's own output. See `notes.language_instruction` and
  `annotator.build_prompt`.

---

## What's new in V1.5 (2026-07-18)

- **Translate tab/CLI**: a new, independent "Translate" tab (and
  `--translate`/`--translate-batch-folder`/`--translate-file-list` on the
  CLI) translates documents into a chosen language, unrelated to the
  audio/video pipeline above - no whisper/notes/slides involved. Same
  Single file / File list (batch) / Folder (auto-discover) modes as the
  Run tabs.
  - **.txt, .docx, .pptx**: translated in place, keeping their own
    format. docx/pptx formatting (bold/italic/font/color), images, and
    table structure survive; PowerPoint **speaker notes** are translated
    too, not just slide text.
  - **.pdf and images** (.jpg/.jpeg/.png/.bmp/.tiff): PDF has no editable
    layout to translate into, so text is extracted (pypdf) or OCR'd
    (pytesseract + Tesseract-OCR, images only) and written into a clean,
    new **translated .docx** instead - the original visual layout is not
    reconstructed.
  - **Not supported**: legacy binary `.doc`/`.ppt` (pre-2007) - save as
    `.docx`/`.pptx` in Office first, then translate that file.
  - Uses the same `llm_backend`/Ollama-or-Anthropic setup as notes
    generation; optionally point translation at a different Ollama model
    via `OLLAMA_TRANSLATE_MODEL` in keys.cfg (blank = reuse `OLLAMA_MODEL`).
  - See `translator.py` (LLM translation core, paragraph-aware chunking)
    and `doc_translate.py` (per-format file I/O).

---

## What's new in V1.6 (2026-07-18)

- **Translate tab/CLI: .html/.htm support**, including this pipeline's
  own generated notes/slide-report HTML. Kept in HTML format (not
  converted to .docx): parsed with BeautifulSoup, only actual visible
  text nodes and a couple of human-readable attributes (`alt`, `title`)
  are translated - `<style>`/`<script>` content, every tag, class, id,
  inline style, and `src`/`href` URL are left completely untouched, so a
  translated slide report still opens and looks like a slide report. See
  `doc_translate.translate_html`.
- **Fixed empty/very slow translations and notes on "thinking" Ollama
  models**: hybrid reasoning models (Qwen3 and similar) can spend the
  entire response budget on the hidden reasoning phase and leave
  `message.content` empty, with the real answer only in
  `message.thinking` - observed as ~90s-per-call VLM/translation calls
  that then failed with "Empty translation result". `notes.py`,
  `translator.py`, and `annotator.py`'s Ollama calls now send
  `"think": false` and fall back to reading `message.thinking` if
  `content` is still empty.
- **Fixed garbled OCR/translation on non-English images** (e.g. Chinese
  slide screenshots): Tesseract was called with no explicit `lang`,
  which silently reads whatever script is in the image through an
  English letter-shape model - it does not fail, it emits confident but
  completely wrong Latin-lookalike text, which a capable LLM then
  "translates" into fluent-sounding but entirely invented nonsense. A
  new **"OCR language" dropdown** in the Translate tab (or
  `--ocr-language` on the CLI; `translate_ocr_language` in config.py,
  default `"auto"` = `"eng+deu"`) sets Tesseract's actual `--lang`,
  independent of the translation target language. Set it to the slide's
  real language, e.g. `"zh"` for Chinese. A requested language pack that
  is not installed falls back to Tesseract's default and logs a warning
  instead of failing the file outright - check what is actually
  installed with `tesseract --list-langs`, and download any missing
  `.traineddata` from https://github.com/tesseract-ocr/tessdata (a
  fresh Windows Tesseract-OCR install typically bundles English only).

---

## PDF export: which backend, and why

PDF export now tries backends in this order and uses the first one that
works, so nothing needs to be configured:

1. **Playwright/Chromium** (recommended, installed by `install.ps1`). No
   native library dependency beyond the downloaded Chromium binary, so it
   does not collide with anything else on your machine.
2. **weasyprint** (fallback). Depends on the GTK3 native library stack on
   Windows.
3. **pdfkit + wkhtmltopdf** (fallback). Needs the wkhtmltopdf binary in PATH.

### Why the switch from weasyprint

If you have Tesseract-OCR installed (used by `DocsSorter` and
`Reisekosten_Scans`), weasyprint can fail with:

```
cannot load library 'C:\Program Files\Tesseract-OCR\libgobject-2.0-0.dll': error 0x7e
```

Tesseract-OCR ships its own `libgobject-2.0-0.dll`, and its install folder
sits earlier in PATH than any GTK3 runtime. weasyprint's library loader
finds Tesseract's incompatible copy first and fails to resolve its
dependencies (`error 0x7e` = `ERROR_MOD_NOT_FOUND`). Playwright's headless
Chromium has no such conflict, hence it is now tried first.

If you want to keep weasyprint working as a fallback too, install the
official GTK3 runtime and make sure its `bin` folder comes before
`C:\Program Files\Tesseract-OCR\` in your PATH.

---

## Usage

### GUI (recommended for interactive use)

```powershell
C:\Python\Python311\python.exe run_pipeline.py --gui
```

**Run tab**: pick a file, folder, or list (or paste paths directly into the
batch box for File list mode); set Whisper model, language, output
language (notes/slide-annotation language, independent of the
transcription language above), prompt template; toggle slide detection /
VLM / diarization; choose which output formats to write; optionally set
an output folder; click Run and watch the log panel and per-stage
progress bar.

**Translate tab**: pick a file, folder, or list (same three modes as the
Run tabs), choose a target language, and translate txt/docx/pptx/html/
pdf/image files - independent of the audio/video pipeline, no whisper/
notes/slides involved. docx/pptx/txt/html keep their format (formatting,
images, tables, PowerPoint speaker notes, and HTML styling all carry
over - this includes the notes/slide-report HTML this pipeline itself
generates, see below); pdf/images are written as a translated .docx.
Legacy `.doc`/`.ppt` are not supported - save as `.docx`/`.pptx` first.

**Search tab**: full-text search over generated notes and slide
titles/bullets; double-click a result to open its folder.

**Settings tab**: edit `keys.cfg` values (LLM backend, Ollama models,
Anthropic key, HuggingFace token) as a form; check/install missing
dependencies; open, create, or rename the database file; set log level and
file logging; or open `config.py` / `keys.cfg` directly in your editor for
anything not exposed as a field. Settings saved here take effect on the
next run, not the current GUI session.

### Command line

```powershell
# Single file (GUI file picker if no path given)
python run_pipeline.py "D:\Videos\meeting.mp4"

# Force language
python run_pipeline.py "D:\Videos\meeting.mp4" de

# Transcribe an English call but get German notes + German slide bullets
# (verbatim transcript file stays in English; "auto" = previous default)
python run_pipeline.py --output-language de "D:\Videos\call_en.mp4"

# Choose a prompt template explicitly (any file in prompts/, without .md)
python run_pipeline.py --prompt-template webinar "D:\Videos\webinar.mp4"

# Redirect all outputs for this run into one folder
python run_pipeline.py --output-dir "D:\Notes\2026-07" "D:\Videos\meeting.mp4"

# Also write notes as Word (.docx) and/or PDF
python run_pipeline.py --docx --pdf "D:\Videos\meeting.mp4"

# Audio-style run on a video: skip slide detection entirely
python run_pipeline.py --no-slides "D:\Videos\talking_head.mp4"

# Video with slides, but skip the VLM annotation step (fast, no Ollama vision call)
python run_pipeline.py --no-vlm "D:\Videos\webinar.mp4"

# Slides only, no transcription
python run_pipeline.py --no-whisper "D:\Videos\webinar.mp4"

# Transcript only, no notes/summary
python run_pipeline.py --no-summary "D:\Videos\meeting.mp4"

# Dry run: slide timestamps only, no annotation/whisper/reports
python run_pipeline.py --dry-run "D:\Videos\webinar.mp4"

# Speaker diarization (requires HF_TOKEN in keys.cfg)
python run_pipeline.py --diarize "D:\Videos\meeting.mp4"

# Re-transcribe, ignoring the cache
python run_pipeline.py --force-retranscribe "D:\Videos\meeting.mp4"

# Notes only, from an existing transcript
python run_pipeline.py --notes-only "D:\Videos\meeting_transcript_speakers.txt"

# Notes for every *_transcript_speakers.txt in a folder
python run_pipeline.py --notes-batch "D:\Videos"

# Batch from a list file (one path per line, optional |language suffix)
python run_pipeline.py --file-list "list.txt"

# Batch: every supported audio/video file found directly in a folder
python run_pipeline.py --batch-folder "D:\Videos"

# Full-text search notes + slides in the database, then exit
python run_pipeline.py --search "retention index"

# Also write this run's log to a timestamped file under log_dir (config.py)
python run_pipeline.py --log-file "D:\Videos\meeting.mp4"

# Recording-type shortcut: sets prompt template + slide/VLM defaults
python run_pipeline.py --mode audio_transcript "D:\Calls\call.mp3"
python run_pipeline.py --mode video_transcript "D:\Videos\demo.mp4"

# Meeting title/date shown in the notes header, passed to the LLM as context
python run_pipeline.py --meeting-title "Q3 Roadmap Review" --meeting-date 2026-07-03 "D:\Videos\call.mp4"

# Same, read from a calendar invite instead (SUMMARY/DTSTART of the first VEVENT)
python run_pipeline.py --ics "D:\Invites\roadmap.ics" "D:\Videos\call.mp4"

# --- Translate tab/CLI: independent of everything above, no whisper/notes/slides ---

# Single file: docx/pptx/txt/html keep their format; pdf/images -> translated .docx
python run_pipeline.py --translate "D:\Docs\proposal.docx" --target-language de

# Translate one of this pipeline's own generated reports in place
python run_pipeline.py --translate "D:\Videos\webinar_slides\webinar_report.html" --target-language de

# Batch: every supported file found directly in a folder
python run_pipeline.py --translate-batch-folder "D:\Docs\to_translate" --target-language en

# Batch from a list file (one path per line, # comments ignored)
python run_pipeline.py --translate-file-list "translate_list.txt" --target-language fr

# Chinese slide screenshots -> German docx: --ocr-language is the SOURCE
# script Tesseract should read, --target-language is the translation output
python run_pipeline.py --translate-batch-folder "D:\Videos\webinar_slides\snapshots" --ocr-language zh --target-language de
```

`--file-list` resumes automatically: if a file already has notes generated
in the database, it is skipped and logged as such. Re-run the same
`list.txt` (or paste box) after an interruption without re-processing
everything; `--force-retranscribe` disables the skip and reprocesses all
listed files.

`list.txt` format:

```
D:\Videos\meeting_en.mp4
D:\Videos\webinar_de.mp4|de
# Lines starting with # are ignored
```

---

## Installation

```powershell
# CPU install (default)
.\install.ps1

# CUDA GPU install (recommended if you have an NVIDIA GPU)
.\install.ps1 -GPU

# Verify installation
.\install.ps1 -Verify
```

`install.ps1` also installs Playwright and downloads its Chromium binary
(~300 MB, one time) for PDF export. Use `-CoreOnly` to skip torch, whisperx,
and Playwright if you only need the slide-detection stage for now.

### Checking what's installed

```powershell
python requirements_check.py            # report, exit code 1 if required packages missing
python requirements_check.py --install  # pip install everything missing
```

Same check backs the GUI: Settings tab, "Requirements" section, "Check
requirements" / "Install missing" buttons. Native Python import checks, no
PowerShell dependency, so it also works if `install.ps1` itself cannot run
for some reason.

---

## Logging

Console/GUI log level defaults to `INFO`. To also persist a run's log to a
file:

```powershell
python run_pipeline.py --log-file "D:\Videos\meeting.mp4"
```

or set `log_to_file: True` / `log_level` / `log_dir` in `config.py` to make
it the default for every run. The GUI Settings tab has the same three
controls (log level dropdown, "save this run's log to a file" checkbox, log
folder field); logs land in `log_dir` (default: `logs/` under the project
folder) as `run_YYYY-MM-DD_HH-MM-SS.log`.

---

## Configuration

### 1. `keys.cfg` - secrets

Edit via the GUI Settings tab, or manually:

```powershell
Copy-Item keys.cfg.example keys.cfg
# Edit keys.cfg: LLM_BACKEND, OLLAMA_MODEL, OLLAMA_VLM_MODEL, HF_TOKEN, ANTHROPIC_API_KEY
```

Model recommendations (2026-07-02, see keys.cfg.example for details):

| Setting | Current default | Alternative |
|---|---|---|
| `OLLAMA_MODEL` (notes) | `qwen3:14b` - better instruction following, same VRAM as qwen2.5:14b | `qwen2.5:14b` (previous default) |
| `OLLAMA_VLM_MODEL` (slides) | `qwen2.5vl:7b` | still solid; `qwen3-vl` leads open-weight OCR if available for your Ollama version |
| Anthropic backend | `claude-sonnet-5` | `claude-opus-4-8` for long/dense transcripts, `claude-haiku-4-5` for short/fast runs |

Check `ollama list` and https://ollama.com/library before switching models;
availability and exact tags can change.

### 2. `vocabulary.py` - domain vocabulary

```powershell
Copy-Item vocabulary.py.example vocabulary.py
# Edit vocabulary.py: add your instrument names, acronyms, terms
python vocabulary.py   # preview the generated prompt
```

### 3. `prompts/*.md` - notes/summary templates

Any `.md` file in `prompts/` is selectable in the GUI dropdown and via
`--prompt-template <name>`. Shipped: `meeting.md`, `webinar.md`.
See `prompts/README.md` for the format and how to add your own.

### 4. `config.py` - pipeline defaults

All settings that are not exposed as a CLI flag or GUI control live here,
each with an explanatory comment. Key ones:

| Key | Default | Description |
|---|---|---|
| `enable_slides` | `True` | Slide detection/VLM stage; auto-off for audio-only files |
| `output_dir_override` | `""` | Redirect all outputs; blank = next to source file |
| `whisper_model` | `large-v3` | ASR model size |
| `whisper_language` | `auto` | `en`, `de`, or `auto` |
| `whisper_use_vocabulary` | `True` | Inject domain vocabulary |
| `prompt_template` | `meeting.md` | Default notes template |
| `output_language` | `auto` | Force notes/summary + VLM slide title/bullets into this language, independent of `whisper_language`. `auto` = no instruction added (previous behaviour). Verbatim transcript file is never translated. |
| `notes_format_txt/html/pdf/docx` | `True/True/False/False` | Which notes formats to write by default |
| `single_pass_limit` / `chunk_size` | `20000` / `12000` | Map-reduce thresholds for long transcripts |
| `hash_threshold` | `8` | Slide-change sensitivity |
| `translate_target_language` | `en` | Default target language for the Translate tab/CLI |
| `ollama_translate_model` | `""` (reuses `ollama_notes_model`) | Set via `OLLAMA_TRANSLATE_MODEL` in keys.cfg to use a different model for translation |
| `ollama_translate_num_ctx` / `translate_chunk_size` | `8192` / `6000` | Context window / chunk size for translation LLM calls |
| `translate_ocr_language` | `auto` (= `eng+deu`) | Tesseract OCR source language for image inputs; NOT the translation target. Set to a code like `zh` for Chinese slides, or a raw Tesseract `--lang` string |

---

## Output files

By default, written next to the source file. Set `--output-dir` / the GUI
field to redirect all of them into one folder instead:

| File | Description |
|---|---|
| `*_transcript_speakers.txt` | Full timestamped transcript with speaker labels |
| `*_text.txt` | Plain text grouped by speaker |
| `*_transcript.srt` | SRT subtitle file |
| `*_segments.json` | Cached whisperx segments (enables instant re-runs) |
| `*_notes.txt` / `*_notes.html` | Structured notes (default formats; toggle via GUI or config.py) |
| `*_notes.pdf` | Notes as PDF (opt-in via `--pdf` / GUI checkbox, or automatic in video mode) |
| `*_notes.docx` | Notes as a real Word document (opt-in via `--docx` / GUI checkbox) |

Additional, only when slide detection ran (video mode), under
`<output>/<stem>_slides/`:

```
<stem>_slides/
  <stem>_report.html      Slide thumbnails with annotations
  <stem>_report.pdf       PDF version
  <stem>_slides.csv       Slide index (Excel-compatible)
  <stem>_slides.json      Full JSON index
  snapshots/              PNG snapshot per detected slide
```

### Translate tab/CLI output

Independent of the above - one output file per input, named
`<stem>_<target_language><ext>`, next to the source file (or under an
output folder override): `.txt`/`.docx`/`.pptx`/`.html`/`.htm` keep the
source's own extension; `.pdf` and images become `.docx`. Not written to the database
- translation runs are not tracked/searchable the way transcription runs
are.

---

## Database

```powershell
python db.py
```

Prints every processed file with model, language, segment count, duration,
and notes-generation status. The database (`transkription_notes_pipeline.db`
by default) holds four tables: `transcripts` (one row per processed file,
including the full generated notes text), `slides` (per-slide index across
all processed videos), `run_log` (audit trail), and `meta` (schema version).

### Full-text search

```powershell
python run_pipeline.py --search "retention index"
```

Searches `transcripts.notes_text` and the slide `title`/`bullets`/aligned
transcript segment via a SQLite FTS5 index, kept in sync automatically on
every insert/update, no manual re-indexing step. The GUI's **Search tab**
does the same with clickable results (double-click opens the containing
folder). If the local Python's sqlite3 build was compiled without FTS5
(rare; python.org's official Windows builds and the sandbox this project
was built in both include it), search falls back to a plain `LIKE` query
automatically and logs a one-time warning.

### Managing the database file

Settings tab, "Database" section: open an existing `.db`, create a new
empty one, rename the current one, or read live stats (row counts, size).
This sets `db_path` as a session override, the same pattern as the output
folder field; to change the permanent default, edit `db_path` in
`config.py`. Programmatically: `db.create_new_db(path)`, `db.rename_db(old,
new)`, `db.get_db_stats(path)`.

---

## Project structure

```
Transkription_Notes_Pipeline/
  run_pipeline.py     CLI entry point / orchestrator (start here)
  gui.py               tkinter parameter GUI (Run + Translate + Search + Settings tabs)
  config.py            Central configuration, all defaults documented
  keys_loader.py       Reads/writes keys.cfg (used by config.py and the Settings tab)
  keys.cfg.example     Secret keys template (copy to keys.cfg)
  db.py                 Unified SQLite schema, FTS5 search, db file management
  requirements_check.py  Native Python dependency check/install (backs GUI + install.ps1 -Verify)
  transcriber.py           whisperx transcribe/align/diarize + transcript file writers
  notes.py                  LLM notes/summary generation (prompt-template driven)
  reporter.py                 All output writers: notes txt/html/docx, slide CSV/JSON/HTML/PDF, PDF conversion
  extractor.py                  ffmpeg frame extraction (video mode)
  detector.py                     pHash slide-change detection (video mode)
  annotator.py                      Ollama VLM slide annotation (video mode)
  translator.py                       LLM translation core (Translate tab/CLI, paragraph-aware chunking)
  doc_translate.py                      Per-format file I/O for translation: txt/docx/pptx/html/pdf/images
  vocabulary.py                       GC/MS domain vocabulary for Whisper (copy from .example)
  prompts/                              Markdown notes/summary templates (meeting.md, webinar.md, ...)
  install.ps1                             One-shot dependency installer
  requirements.txt                          Python dependencies with comments
  test_nas_path.ps1                           Standalone NAS/UNC path smoke test (run locally)
```

---

## Requirements

- Python 3.11 (`C:\Python\Python311\python.exe`)
- ffmpeg in PATH
- Ollama running locally (`ollama serve`) for the local LLM backend and
  VLM slide annotation - or an Anthropic API key for the `anthropic` backend
- HuggingFace token - optional, for speaker diarization
- tkinter - included with the standard python.org Windows installer; needed
  for the GUI only, not for the CLI
- Playwright + Chromium - installed by `install.ps1`, used for PDF export
- python-pptx, pypdf, beautifulsoup4 - for the Translate tab/CLI (docx
  already required above); optional if you never use that tab
- pytesseract + the Tesseract-OCR binary - optional, only for translating
  image inputs (.jpg/.png/etc.) in the Translate tab/CLI; the binary
  itself is not pip-installable, see
  https://github.com/tesseract-ocr/tesseract

---

## NAS and network paths

`\\192.168.0.104\Public\Agilent\` is a private LAN address; it cannot be
reached or tested from outside your network, so the following is a code
review plus a script to run locally, not a live end-to-end test result.

- **Path handling**: `config.py`/`run_pipeline.py` use `pathlib.Path`
  throughout, which handles UNC paths (`\\server\share\...`) correctly on
  Windows, `.parent`, `.stem`, and `/` joins all work the same as for local
  paths. `extractor.py` calls `ffmpeg`/`ffprobe` via `subprocess.run()` with
  an argument list (no `shell=True`), so spaces in NAS paths are not a
  quoting problem.
- **ffmpeg on UNC paths**: most current ffmpeg builds read `-i \\server\...`
  directly without issue. Some older or minimal builds have had trouble
  with UNC input paths; if frame extraction fails only for NAS files, map
  the share to a drive letter first (`net use Z: \\192.168.0.104\Public\Agilent`)
  and use `Z:\...` instead as a reliable workaround.
- **SQLite and the database file specifically**: keep `db_path` on local
  disk. SQLite's WAL journal mode (used by `db.py` for concurrent-safe
  writes) requires a shared-memory `-shm` file that only works between
  processes on the same machine; the SQLite documentation states this
  explicitly and does not recommend running SQLite over a network
  filesystem at all, WAL or not. Source videos and `--output-dir` on the
  NAS are fine; only `transkription_notes_pipeline.db` should stay local
  (the default already does this).
- **Local test script**: `test_nas_path.ps1` (project root) checks share
  reachability, write access, Python `pathlib` handling, and optionally
  `ffprobe` on a real file, run it on your machine against the actual share.

Sources: [SQLite Write-Ahead Logging](https://sqlite.org/wal.html),
[SQLite Over a Network, Caveats and Considerations](https://sqlite.org/useovernet.html)

---

## Further improvement ideas

Not implemented, worth considering if they become a real bottleneck:

- **Duplicate detection**: hash source files (not just transcripts) to
  warn before re-processing the same recording under a different filename.
- **Config profiles**: named presets in `config.py` (e.g. "fast draft" vs.
  "high quality") selectable from the GUI instead of hand-adjusting every
  field per run.
- **Cancel-run button**: the GUI currently has no way to stop a run once
  started short of closing the window; the worker thread is daemonized so
  the process exits cleanly, but there is no graceful mid-run abort.
- **Retry-on-transient-failure** for Ollama/Anthropic API calls (timeouts,
  connection resets), currently a single attempt per chunk.
- **CSV export of search results** from the Search tab, for pasting into a
  report.
- **Editable vocabulary in the GUI** instead of only `vocabulary.py`, so
  domain terms can be added without opening a text editor.
- **Translate tab known limitations**: docx headers/footers/textboxes and
  nested tables (a table inside another table's cell) are not translated
  (only body paragraphs, top-level tables, and PowerPoint speaker notes
  are); a docx/pptx paragraph that mixes formatting mid-sentence (e.g.
  "Please **confirm** by Friday") keeps only its first run's formatting
  after translation, since the whole paragraph is translated as one unit
  (see `doc_translate.py`'s module docstring for why); an HTML text node
  spanning several fields glued together with `&nbsp;`/other inline
  separators (e.g. reporter.py's "title &nbsp;|&nbsp; date" meta line)
  is sent to the LLM as a single chunk, so the exact separator/spacing is
  not guaranteed to survive translation byte-for-byte; Translate tab
  settings (target language, output folder) are not yet persisted to
  `gui_state.json` the way the Run tabs' settings are.

---

## License

MIT
