# =============================================================
#  Transkription_Notes_Pipeline - gui.py
#  tkinter parameter GUI, four tabs:
#    Meeting       - audio/meeting-only transcription workflow: file/batch
#                    selection, whisper/notes params, meeting info, batch
#                    table, output formats, per-stage progress, live log.
#                    No slide detection/VLM/Q&A - see Video / Webinar tab.
#    Video/Webinar - everything the Meeting tab has, plus slide detection/
#                    VLM annotation, the Q&A section, slide-report
#                    formats, and Review/Edit Slides + Re-annotate failed
#                    slides.
#    Translate     - translate txt/docx/pptx/pdf/image files into a
#                    chosen language (single file / file list (batch) /
#                    folder auto-discover), independent of the audio/
#                    video pipeline above. See translator.py/
#                    doc_translate.py.
#    Search        - full-text search (FTS5) over generated notes and
#                    slide titles/bullets
#    Settings      - keys.cfg as a structured form (masked API keys),
#                    requirements check/install, database file
#                    management, logging options, "Open config.py"
#
#  Launch:
#    python run_pipeline.py --gui
#    python gui.py
# =============================================================

import csv
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk

from PIL import Image, ImageTk

from config import CONFIG, PROMPTS_DIR, SUPPORTED_EXTENSIONS, get_ffplay_path
import correlate_calendar_recordings
import db
import doc_translate
import gui_logic
import ics_utils
import keys_loader
import notes as notes_mod
import run_pipeline
import speaker_id

log = logging.getLogger(__name__)

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
LANGUAGES = ["auto", "en", "de", "fr", "es", "it", "ja", "zh", "nl", "uk", "pt"]
LLM_BACKENDS = ["ollama", "anthropic"]
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]
MODES = ["Single file", "File list (batch)", "Folder (auto-discover)",
        "Notes-only (existing transcript)", "Notes-batch (folder)"]

# Translate tab: only the three generic modes apply (no notes-only/
# notes-batch equivalent - there is no separate "transcript" stage to
# redo). Kept as its own list/constant rather than slicing MODES so the
# two tabs' mode sets can diverge freely later without cross-talk.
TRANSLATE_MODES = ["Single file", "File list (batch)", "Folder (auto-discover)"]

# Real target languages only - "auto" (valid for whisper_language/
# output_language, meaning "no instruction") is not meaningful as a
# translation target.
TRANSLATE_LANGUAGES = [l for l in LANGUAGES if l != "auto"]

# Persisted GUI settings (task #71): last-used model/language/backend/
# prompt template/thresholds/output folder/stage checkboxes per tab, so
# reopening the GUI doesn't reset to config.py's defaults every time.
# Gitignored; see gui_logic.py's "Persisted GUI settings" section for the
# exact key list and what is deliberately excluded (path, batch table,
# meeting info, Q&A times).
STATE_PATH = Path(__file__).parent / gui_logic.STATE_FILENAME

# Recording-type presets, split per tab (Meeting vs Video/Webinar): picking
# one sets the prompt template. Slide detection/VLM are no longer part of
# the preset since each tab already fixes that (Meeting tab never runs
# slides; Video/Webinar tab's own Stages checkboxes control it).
MEETING_RECORDING_TYPES = {
    "Audio Transcript":   {"prompt_template": "audio_transcript"},
    "Meeting Transcript": {"prompt_template": "meeting"},
}
VIDEO_RECORDING_TYPES = {
    "Video Transcript":   {"prompt_template": "video_transcript"},
    "Webinar Transcript": {"prompt_template": "webinar"},
}

# Log-line markers emitted by run_pipeline.process_file() at each stage
# boundary (see run_pipeline.py "STAGE:..." log.info calls). Matched as
# a substring against the formatted log line to drive the progress bar
# without any deeper pipeline refactor.
STAGE_PROGRESS = [
    ("STAGE:start",      5,   "Starting"),
    ("STAGE:transcribe", 25,  "Transcribing"),
    ("STAGE:slides",     50,  "Detecting / annotating slides"),
    ("STAGE:db",         60,  "Adding to database"),
    ("STAGE:normalize",  70,  "Normalizing video to real-time speed"),
    ("STAGE:notes",      85,  "Generating notes"),
    ("STAGE:done",       100, "Finishing"),
]


def _parse_time_to_seconds(text: str) -> float | None:
    """Thin re-export of gui_logic.parse_time_to_seconds, kept under this
    name since it's referenced throughout this file; the actual parsing
    logic lives in gui_logic.py so it can be unit-tested without tkinter."""
    return gui_logic.parse_time_to_seconds(text)


# =============================================================
# GUI color theme
#
# apply_theme() below reads CONFIG["gui_theme"] (see config.py's "GUI
# THEME" section) and applies it via ttk.Style to every ttk widget in
# the app. _THEME is populated once at startup and also used directly
# by the handful of plain-tk widgets ttk styling can't reach (Canvas,
# Listbox, ScrolledText, and the Toplevel dialog backgrounds) via
# _theme_color(). _DEFAULT_THEME is the fallback if config.py's
# gui_theme is missing or missing a key (e.g. an older config.py from
# before this existed).
# =============================================================

_THEME: dict = {}

_DEFAULT_THEME = {
    "primary_100": "#F2F6FF", "primary_200": "#BCD2FE", "primary_300": "#84B1F9",
    "primary_400": "#4B92EB", "primary_500": "#1773CF", "primary_600": "#085FA4",
    "primary_700": "#024C7A", "primary_800": "#003650", "primary_900": "#001C26",
    "accent_100": "#F2FFF5", "accent_200": "#BCFFC8", "accent_300": "#85FD93",
    "accent_400": "#4EFA58", "accent_500": "#18F218", "accent_600": "#13BF08",
    "accent_700": "#128C02", "accent_800": "#105900", "accent_900": "#092600",
    "neutral_100": "#FAFAFC", "neutral_200": "#E8E9EC", "neutral_300": "#D7D8DB",
    "neutral_400": "#C6C7CB", "neutral_500": "#B5B7BA", "neutral_600": "#8E9195",
    "neutral_700": "#696D70", "neutral_800": "#45494B", "neutral_900": "#222526",
}


def _theme_color(key: str) -> str:
    """Look up one theme color token (e.g. "primary_500"), falling back
    to the built-in default palette if it's missing from _THEME."""
    return _THEME.get(key) or _DEFAULT_THEME[key]


def _style_toplevel(win: tk.Toplevel):
    """Set a Toplevel's own background to match the theme - ttk.Style
    only reaches ttk widgets placed inside it, not the Toplevel window
    itself. Called at the top of each dialog's __init__."""
    try:
        win.configure(bg=_theme_color("neutral_100"))
    except tk.TclError:
        pass


def apply_theme(root: tk.Tk):
    """
    Apply CONFIG["gui_theme"]'s color palette to every ttk widget via
    ttk.Style, plus the root window background. Populates the
    module-level _THEME dict so plain-tk widgets built later (Canvas,
    Listbox, ScrolledText, Toplevel dialogs) can look up the same colors
    via _theme_color()/_style_toplevel().

    Leaves Tk's default look untouched if CONFIG["gui_theme_enabled"] is
    False, but _THEME is still populated either way so callers don't
    need to branch on the setting themselves.
    """
    global _THEME
    _THEME = dict(_DEFAULT_THEME)
    _THEME.update(CONFIG.get("gui_theme") or {})

    if not CONFIG.get("gui_theme_enabled", True):
        return

    t = _THEME
    root.configure(bg=t["neutral_100"])

    style = ttk.Style(root)
    # "clam" is drawn entirely by Tk, unlike Windows' native "vista"/
    # "winnative" ttk themes, which ignore most color options since
    # those widgets are rendered by the OS theming engine - required
    # for the custom palette below to actually be visible.
    try:
        style.theme_use(CONFIG.get("gui_ttk_theme", "clam"))
    except tk.TclError:
        style.theme_use("clam")

    style.configure(".", background=t["neutral_100"], foreground=t["neutral_900"],
                    fieldbackground=t["neutral_100"])

    style.configure("TFrame", background=t["neutral_100"])
    style.configure("TLabelframe", background=t["neutral_100"], bordercolor=t["neutral_400"])
    style.configure("TLabelframe.Label", background=t["neutral_100"], foreground=t["primary_700"])
    style.configure("TLabel", background=t["neutral_100"], foreground=t["neutral_900"])

    style.configure("TButton", background=t["primary_500"], foreground=t["neutral_100"],
                    bordercolor=t["primary_600"], focuscolor=t["primary_300"], padding=5)
    style.map("TButton",
             background=[("disabled", t["neutral_400"]), ("pressed", t["primary_700"]),
                         ("active", t["primary_400"])],
             foreground=[("disabled", t["neutral_600"])])

    style.configure("TCheckbutton", background=t["neutral_100"], foreground=t["neutral_900"])
    style.map("TCheckbutton",
             indicatorcolor=[("selected", t["accent_500"]), ("!selected", t["neutral_200"])],
             background=[("active", t["neutral_100"])])

    style.configure("TRadiobutton", background=t["neutral_100"], foreground=t["neutral_900"])
    style.map("TRadiobutton",
             indicatorcolor=[("selected", t["accent_500"]), ("!selected", t["neutral_200"])])

    style.configure("TEntry", fieldbackground=t["neutral_100"], foreground=t["neutral_900"],
                    bordercolor=t["neutral_400"])
    style.map("TEntry", bordercolor=[("focus", t["primary_500"])])

    style.configure("TCombobox", fieldbackground=t["neutral_100"], foreground=t["neutral_900"],
                    background=t["neutral_100"], arrowcolor=t["primary_600"])
    style.map("TCombobox",
             fieldbackground=[("readonly", t["neutral_100"])],
             bordercolor=[("focus", t["primary_500"])])

    style.configure("TSpinbox", fieldbackground=t["neutral_100"], foreground=t["neutral_900"],
                    arrowcolor=t["primary_600"])

    style.configure("TNotebook", background=t["neutral_100"], bordercolor=t["neutral_300"])
    style.configure("TNotebook.Tab", background=t["neutral_200"], foreground=t["neutral_800"],
                    padding=(10, 4))
    style.map("TNotebook.Tab",
             background=[("selected", t["primary_500"]), ("active", t["primary_300"])],
             foreground=[("selected", t["neutral_100"]), ("active", t["neutral_900"])])

    style.configure("Treeview", background=t["neutral_100"], fieldbackground=t["neutral_100"],
                    foreground=t["neutral_900"], bordercolor=t["neutral_300"])
    style.configure("Treeview.Heading", background=t["primary_700"], foreground=t["neutral_100"])
    style.map("Treeview.Heading", background=[("active", t["primary_600"])])
    style.map("Treeview",
             background=[("selected", t["primary_300"])],
             foreground=[("selected", t["neutral_900"])])

    style.configure("TProgressbar", background=t["primary_500"], troughcolor=t["neutral_300"],
                    bordercolor=t["neutral_300"])

    style.configure("TScrollbar", background=t["neutral_300"], troughcolor=t["neutral_200"],
                    arrowcolor=t["primary_600"])
    style.map("TScrollbar", background=[("active", t["primary_400"])])

    style.configure("TPanedwindow", background=t["neutral_100"])


class QueueLogHandler(logging.Handler):
    """Logging handler that pushes formatted records into a thread-safe queue."""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


class Tooltip:
    """
    Small delayed popup shown on hover (task #73), used to explain what a
    Stages checkbox actually does without cluttering the tab with a
    permanent label under every one. Attach with Tooltip(widget, "text");
    nothing else needs to reference the returned instance afterward, the
    event bindings on the widget keep it alive for the widget's lifetime.
    """

    def __init__(self, widget, text: str, delay_ms: int = 500):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id = None
        self._tip_window = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self):
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self):
        if self._tip_window is not None:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self._tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify="left", background="#ffffe0",
                         relief="solid", borderwidth=1, wraplength=360, font=("", 9))
        label.pack(ipadx=4, ipady=2)

    def _hide(self, event=None):
        self._cancel()
        if self._tip_window is not None:
            self._tip_window.destroy()
            self._tip_window = None


# Explanatory tooltip text for each Stages checkbox (task #73). Shared
# between both tabs; a given tab only uses the keys for checkboxes it
# actually has (see RunTabController._build's Stages section).
STAGE_TOOLTIPS = {
    "enable_slides": "Detects slide changes in the video by comparing "
                      "perceptual hashes between frames. Automatically "
                      "skipped for audio-only files regardless of this "
                      "setting.",
    "enable_vlm": "Sends each detected slide's snapshot image to a local "
                  "vision-language model (Ollama) to extract a title and "
                  "bullet points. Requires slide detection to be enabled.",
    "enable_diarization": "Identifies and labels distinct speakers "
                          "(SPEAKER_01, SPEAKER_02, ...) using pyannote. "
                          "Requires HF_TOKEN to be set in Settings and the "
                          "pyannote model terms to be accepted on "
                          "HuggingFace.",
    "no_summary": "Skips the LLM summary/notes step entirely: only the "
                  "transcript, .srt, and (Video/Webinar tab) slide report "
                  "are produced.",
    "force_retranscribe": "Ignores any cached transcript, segments, or "
                          "slides already on disk for this file and "
                          "reprocesses it completely from scratch.",
    "enhance_audio": "Runs a conservative ffmpeg cleanup pass over the audio "
                     "before transcription (highpass for rumble/hum, mild "
                     "denoise, loudness normalization between speakers). Off "
                     "by default because audio processing can occasionally "
                     "change what Whisper transcribes. Works on a temporary "
                     "copy: the original file is never modified, and Play "
                     "Sample and voiceprint extraction still read the original.",
    "dry_run": "Runs slide-change detection only, to preview slide "
              "timestamps, without VLM annotation, transcription, or "
              "notes generation.",
}


class PipelineGUI:
    """
    Top-level window: owns the Notebook and the Search/Settings tabs
    directly, and hands the Meeting and Video/Webinar tabs off to two
    independent RunTabController instances (see that class). Shared state
    a RunTabController needs (database path, logging options) lives here
    on self and is reached via each controller's `.app` reference.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Transkription_Notes_Pipeline")
        # 1300x860 (was 980x860): wide enough that the Video/Webinar tab's
        # step 1 "Parameters" row (Whisper model/Language/Output language,
        # then LLM backend/Hash threshold/Slide fps) shows in full without
        # having to manually resize the window first. minsize keeps a
        # user-shrunk window from going back below that point.
        self.root.geometry("1300x860")
        self.root.minsize(1100, 700)

        self.req_queue: queue.Queue = queue.Queue()
        self._notes_result_paths: dict = {}
        self._slides_result_paths: dict = {}
        self._last_check_results: list = []

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)
        self.meeting_tab = ttk.Frame(notebook)
        self.video_tab = ttk.Frame(notebook)
        self.translate_tab = ttk.Frame(notebook)
        self.search_tab = ttk.Frame(notebook)
        self.templates_tab = ttk.Frame(notebook)
        self.settings_tab = ttk.Frame(notebook)
        notebook.add(self.meeting_tab, text="Audio/Meeting")
        notebook.add(self.video_tab, text="Video / Webinar")
        notebook.add(self.translate_tab, text="Translate")
        notebook.add(self.search_tab, text="Search")
        notebook.add(self.templates_tab, text="Templates")
        notebook.add(self.settings_tab, text="Settings")

        # --- Shared Meeting/Video engine settings ---
        # One tkinter Variable per key, created once here and reused by
        # BOTH RunTabController instances below (see RunTabController._build,
        # which points its like-named attributes at these instead of
        # creating its own copy). Editing the value on either the Meeting
        # or the Video/Webinar tab updates the other immediately, and
        # gui_logic.PERSISTED_KEYS_SHARED persists a single copy under
        # gui_state.json's "shared" key - previously each tab kept its own
        # independent copy that could silently drift apart (e.g. Meeting's
        # LLM backend left on "anthropic", Video's still on "ollama").
        self.shared_whisper_model_var = tk.StringVar(value=CONFIG["whisper_model"])
        self.shared_language_var = tk.StringVar(value=CONFIG["whisper_language"])
        self.shared_output_language_var = tk.StringVar(value=CONFIG.get("output_language", "auto"))
        self.shared_llm_backend_var = tk.StringVar(value=CONFIG["llm_backend"])
        self.shared_enable_diarization_var = tk.BooleanVar(value=CONFIG["enable_diarization"])
        self.shared_no_summary_var = tk.BooleanVar(value=False)
        self.shared_force_retranscribe_var = tk.BooleanVar(value=False)
        self.shared_enhance_audio_var = tk.BooleanVar(value=CONFIG.get("enhance_audio", False))

        self._build_search_tab()
        self._build_templates_tab()
        self._build_settings_tab()
        self.meeting_run = RunTabController(self.meeting_tab, "meeting", self)
        self.video_run = RunTabController(self.video_tab, "video", self)
        self.translate_run = TranslateTabController(self.translate_tab, self)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(200, self._poll_req_queue)

    def _on_close(self):
        """Persist both tabs' settings to gui_state.json (task #71)
        before closing: whisper model, language, LLM backend, prompt
        template, thresholds, output folder, and stage checkboxes. Never
        blocks shutdown - save_state_file swallows I/O errors.

        Shared engine settings (whisper model, language, LLM backend,
        diarization, no-summary, force-retranscribe) are written once
        under "shared" rather than duplicated under "meeting"/"video" -
        both tabs' widgets are backed by the same Variable objects, so
        reading them from either tab's snapshot gives the same values."""
        state = {
            "meeting": self.meeting_run._collect_tab_state(),
            "video": self.video_run._collect_tab_state(),
            "shared": self.meeting_run._collect_shared_state(),
        }
        gui_logic.save_state_file(STATE_PATH, state)
        self.root.destroy()

    # -----------------------------------------------------------------
    # Scrollable tab helper (task #63), shared by both RunTabControllers
    # and the Settings tab.
    # -----------------------------------------------------------------

    def _make_scrollable(self, parent) -> ttk.Frame:
        """
        Wrap a Notebook tab Frame in a vertically scrollable canvas, so long
        tab content is never cut off by a small window/screen. Returns an
        inner ttk.Frame: build the tab's actual widgets into that, exactly
        as if it were the tab itself. Mouse-wheel scrolling is only active
        while the pointer is over this tab, so it doesn't hijack scrolling
        on other widgets/tabs.
        """
        canvas = tk.Canvas(parent, highlightthickness=0, bg=_theme_color("neutral_100"))
        vsb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        inner = ttk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfig(inner_id, width=event.width)

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_wheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_wheel(event):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _bind_wheel)
        canvas.bind("<Leave>", _unbind_wheel)

        return inner

    def _open_prompts_folder(self):
        """Open prompts/ in the OS file browser so custom .md templates can
        be added/edited directly, then picked up via "Reload list" above."""
        self._open_path(Path(PROMPTS_DIR))

    # -----------------------------------------------------------------
    # Search tab: FTS5 full-text search over notes + slides
    # -----------------------------------------------------------------

    def _build_search_tab(self):
        parent = self.search_tab
        pad = {"padx": 8, "pady": 4}

        self.search_query_var = tk.StringVar()

        top = ttk.Frame(parent)
        top.pack(fill="x", **pad)
        ttk.Label(top, text="Search notes and slides:").pack(side="left")
        entry = ttk.Entry(top, textvariable=self.search_query_var, width=50)
        entry.pack(side="left", padx=6)
        entry.bind("<Return>", lambda e: self._do_search())
        ttk.Button(top, text="Search", command=self._do_search).pack(side="left")
        ttk.Label(top, text="Double-click a result to open its folder.",
                 foreground="#666").pack(side="left", padx=(12, 0))

        ttk.Label(parent, text="Notes matches:").pack(anchor="w", padx=8)
        notes_frame = ttk.Frame(parent)
        notes_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.notes_results = ttk.Treeview(
            notes_frame, columns=("file", "when", "snippet"), show="headings", height=7)
        for col, text, w in (("file", "File", 220), ("when", "Processed", 150), ("snippet", "Snippet", 420)):
            self.notes_results.heading(col, text=text)
            self.notes_results.column(col, width=w, anchor="w")
        self.notes_results.pack(fill="both", expand=True)
        self.notes_results.bind("<Double-1>", self._open_notes_result)

        ttk.Label(parent, text="Slide matches:").pack(anchor="w", padx=8)
        slides_frame = ttk.Frame(parent)
        slides_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.slides_results = ttk.Treeview(
            slides_frame, columns=("video", "t", "title", "snippet"), show="headings", height=7)
        for col, text, w in (("video", "Video", 200), ("t", "Time", 70),
                             ("title", "Title", 160), ("snippet", "Snippet", 380)):
            self.slides_results.heading(col, text=text)
            self.slides_results.column(col, width=w, anchor="w")
        self.slides_results.pack(fill="both", expand=True)
        self.slides_results.bind("<Double-1>", self._open_slides_result)

    def _do_search(self):
        query = self.search_query_var.get().strip()
        if not query:
            return
        db_path = self.db_path_var.get().strip() or CONFIG["db_path"]
        try:
            results = db.search_all(db_path, query)
        except Exception as exc:
            messagebox.showerror("Search failed", str(exc))
            return

        for row in self.notes_results.get_children():
            self.notes_results.delete(row)
        for row in self.slides_results.get_children():
            self.slides_results.delete(row)
        self._notes_result_paths.clear()
        self._slides_result_paths.clear()

        for r in results["notes"]:
            iid = self.notes_results.insert("", "end", values=(r["file_name"], r["processed_at"], r["snippet"]))
            self._notes_result_paths[iid] = r["file_path"]
        for r in results["slides"]:
            iid = self.slides_results.insert(
                "", "end",
                values=(Path(r["video_path"]).name, f"{r['timestamp_sec']:.1f}s", r["title"], r["snippet"]))
            self._slides_result_paths[iid] = r["video_path"]

        if not results["notes"] and not results["slides"]:
            messagebox.showinfo("No matches", f'No matches for "{query}".')

    def _open_notes_result(self, event):
        sel = self.notes_results.selection()
        if not sel:
            return
        path = self._notes_result_paths.get(sel[0])
        if path:
            self._open_path(Path(path).parent)

    def _open_slides_result(self, event):
        sel = self.slides_results.selection()
        if not sel:
            return
        path = self._slides_result_paths.get(sel[0])
        if path:
            self._open_path(Path(path).parent)

    # -----------------------------------------------------------------
    # Templates tab (task #77): view/import/copy/delete/backup/edit the
    # prompts/*.md files both Run tabs' "Prompt template" dropdown reads
    # from. Every destructive action (Save, Delete) backs the previous
    # content up to prompts/_backup/ first - the originals are never
    # silently lost. gui_logic.sanitize_template_name guards New/Import/
    # Copy against path traversal and blank/README names.
    # -----------------------------------------------------------------

    def _build_templates_tab(self):
        parent = self.templates_tab
        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True, padx=8, pady=8)

        left = ttk.Frame(container)
        left.pack(side="left", fill="y", padx=(0, 8))
        ttk.Label(left, text="Prompt templates\n(prompts/*.md):").pack(anchor="w")
        self.templates_list = tk.Listbox(
            left, width=26, height=22, exportselection=False,
            bg=_theme_color("neutral_100"), fg=_theme_color("neutral_900"),
            selectbackground=_theme_color("primary_400"), selectforeground=_theme_color("neutral_100"))
        self.templates_list.pack(fill="y", pady=(4, 4))
        self.templates_list.bind("<<ListboxSelect>>", self._on_template_selected)

        btns = ttk.Frame(left)
        btns.pack(fill="x")
        ttk.Button(btns, text="New...", command=self._template_new).pack(fill="x", pady=(0, 2))
        ttk.Button(btns, text="Import...", command=self._template_import).pack(fill="x", pady=2)
        ttk.Button(btns, text="Copy...", command=self._template_copy).pack(fill="x", pady=2)
        ttk.Button(btns, text="Delete", command=self._template_delete).pack(fill="x", pady=2)
        ttk.Button(btns, text="Reload list", command=self._refresh_templates_list).pack(
            fill="x", pady=(2, 0))
        ttk.Button(btns, text="Open templates folder...",
                  command=self._open_prompts_folder).pack(fill="x", pady=(8, 0))

        right = ttk.Frame(container)
        right.pack(side="left", fill="both", expand=True)
        self.template_name_label = ttk.Label(right, text="No template selected.", foreground="#444")
        self.template_name_label.pack(anchor="w")
        self.template_editor = scrolledtext.ScrolledText(
            right, wrap="word", height=28, undo=True,
            bg=_theme_color("neutral_100"), fg=_theme_color("neutral_900"),
            insertbackground=_theme_color("neutral_900"),
            selectbackground=_theme_color("primary_300"))
        self.template_editor.pack(fill="both", expand=True, pady=(4, 4))
        self.template_editor.config(state="disabled")

        editor_btns = ttk.Frame(right)
        editor_btns.pack(fill="x")
        self.template_save_btn = ttk.Button(
            editor_btns, text="Save", command=self._template_save, state="disabled")
        self.template_save_btn.pack(side="left")
        self.template_status_label = ttk.Label(editor_btns, text="", foreground="#666")
        self.template_status_label.pack(side="left", padx=(8, 0))
        ttk.Label(right, text="Every Save/Delete backs up the previous content to "
                             "prompts/_backup/ first.",
                 foreground="#666").pack(anchor="w", pady=(4, 0))

        self._current_template_path: Path | None = None
        self._template_paths: dict = {}
        self._refresh_templates_list()

    def _refresh_templates_list(self):
        """Reload the template list from disk and clear the editor pane.
        Also refreshes both Run tabs' "Prompt template" dropdowns, so a
        New/Import/Copy/Delete/rename here is immediately reflected there
        without needing their own "Reload list" button clicked too."""
        self.templates_list.delete(0, "end")
        self._template_paths = {}
        for name, path in notes_mod.list_prompt_templates(PROMPTS_DIR).items():
            self.templates_list.insert("end", name)
            self._template_paths[name] = path

        self.template_editor.config(state="normal")
        self.template_editor.delete("1.0", "end")
        self.template_editor.config(state="disabled")
        self.template_save_btn.config(state="disabled")
        self.template_name_label.config(text="No template selected.")
        self.template_status_label.config(text="")
        self._current_template_path = None

        if hasattr(self, "meeting_run"):
            self.meeting_run._refresh_prompt_templates()
        if hasattr(self, "video_run"):
            self.video_run._refresh_prompt_templates()

    def _select_template_by_name(self, name: str):
        items = list(self.templates_list.get(0, "end"))
        if name in items:
            idx = items.index(name)
            self.templates_list.selection_clear(0, "end")
            self.templates_list.selection_set(idx)
            self.templates_list.see(idx)
            self._on_template_selected()

    def _on_template_selected(self, event=None):
        sel = self.templates_list.curselection()
        if not sel:
            return
        name = self.templates_list.get(sel[0])
        path = self._template_paths.get(name)
        if not path or not path.exists():
            return
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Could not read template", str(exc))
            return
        self._current_template_path = path
        self.template_name_label.config(text=path.name)
        self.template_editor.config(state="normal")
        self.template_editor.delete("1.0", "end")
        self.template_editor.insert("1.0", content)
        self.template_save_btn.config(state="normal")
        self.template_status_label.config(text="")

    def _template_backup(self, path: Path, suffix: str = "") -> Path:
        """Copy path into prompts/_backup/<stem>_<timestamp><suffix>.md
        before it is overwritten or deleted. Returns the backup path."""
        backup_dir = Path(PROMPTS_DIR) / "_backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{path.stem}_{stamp}{suffix}.md"
        shutil.copy2(path, backup_path)
        return backup_path

    def _template_new(self):
        raw = simpledialog.askstring(
            "New template", 'Template name (e.g. "standup"):', parent=self.root)
        if raw is None:
            return
        name = gui_logic.sanitize_template_name(raw)
        if not name:
            messagebox.showwarning(
                "Invalid name",
                "Enter a plain filename with no path separators and not \"readme\" "
                "(a .md extension is added automatically).")
            return
        path = Path(PROMPTS_DIR) / name
        if path.exists():
            messagebox.showwarning("Already exists", f"{name} already exists.")
            return
        try:
            path.write_text("## New Template\n\nDescribe the desired notes structure here.\n",
                            encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Could not create template", str(exc))
            return
        self._refresh_templates_list()
        self._select_template_by_name(path.stem)

    def _template_import(self):
        src = filedialog.askopenfilename(
            title="Import prompt template",
            filetypes=[("Markdown", "*.md"), ("All files", "*.*")])
        if not src:
            return
        dest_name = gui_logic.sanitize_template_name(Path(src).stem)
        if not dest_name:
            messagebox.showwarning("Invalid name", "Selected file has an unusable name.")
            return
        dest = Path(PROMPTS_DIR) / dest_name
        if dest.exists() and not messagebox.askyesno(
                "Overwrite?", f"{dest_name} already exists. Overwrite (previous content is "
                             f"backed up first)?"):
            return
        try:
            if dest.exists():
                self._template_backup(dest)
            shutil.copy2(src, dest)
        except OSError as exc:
            messagebox.showerror("Import failed", str(exc))
            return
        self._refresh_templates_list()
        self._select_template_by_name(dest.stem)

    def _template_copy(self):
        if not self._current_template_path:
            messagebox.showinfo("No template selected", "Select a template to copy first.")
            return
        raw = simpledialog.askstring("Copy template", "New template name:", parent=self.root)
        if raw is None:
            return
        name = gui_logic.sanitize_template_name(raw)
        if not name:
            messagebox.showwarning(
                "Invalid name", "Enter a plain filename with no path separators.")
            return
        dest = Path(PROMPTS_DIR) / name
        if dest.exists():
            messagebox.showwarning("Already exists", f"{name} already exists.")
            return
        try:
            shutil.copy2(self._current_template_path, dest)
        except OSError as exc:
            messagebox.showerror("Copy failed", str(exc))
            return
        self._refresh_templates_list()
        self._select_template_by_name(dest.stem)

    def _template_delete(self):
        if not self._current_template_path:
            messagebox.showinfo("No template selected", "Select a template to delete first.")
            return
        path = self._current_template_path
        if not messagebox.askyesno(
                "Delete template", f"Delete {path.name}? A backup is kept in prompts/_backup/."):
            return
        try:
            self._template_backup(path, suffix="_deleted")
            path.unlink()
        except OSError as exc:
            messagebox.showerror("Delete failed", str(exc))
            return
        self._refresh_templates_list()

    def _template_save(self):
        if not self._current_template_path:
            return
        path = self._current_template_path
        content = self.template_editor.get("1.0", "end-1c")
        try:
            if path.exists():
                backup_path = self._template_backup(path)
            else:
                backup_path = None
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        note = f" (backup: _backup/{backup_path.name})" if backup_path else ""
        self.template_status_label.config(
            text=f"Saved at {datetime.now().strftime('%H:%M:%S')}{note}")
        if hasattr(self, "meeting_run"):
            self.meeting_run._refresh_prompt_templates()
        if hasattr(self, "video_run"):
            self.video_run._refresh_prompt_templates()

    # -----------------------------------------------------------------
    # Settings tab: keys.cfg, requirements, database, logging
    # -----------------------------------------------------------------

    def _build_settings_tab(self):
        parent = self._make_scrollable(self.settings_tab)
        pad = {"padx": 8, "pady": 5}

        current = keys_loader.current_values()

        info = ttk.Label(
            parent,
            text="Edits here are saved to keys.cfg (secrets, git-ignored). "
                 "They take effect on the next run, not the current session. "
                 "For anything else (models, thresholds, output flags), use "
                 "'Open config.py' below.",
            wraplength=820, foreground="#444",
        )
        info.pack(fill="x", **pad)

        form = ttk.LabelFrame(parent, text="keys.cfg")
        form.pack(fill="x", **pad)

        self.llm_backend_setting_var = tk.StringVar(value=current.get("LLM_BACKEND", "ollama"))
        self.ollama_url_var = tk.StringVar(value=current.get("OLLAMA_BASE_URL", "http://localhost:11434"))
        self.ollama_model_var = tk.StringVar(value=current.get("OLLAMA_MODEL", "qwen3:14b"))
        self.ollama_vlm_model_var = tk.StringVar(value=current.get("OLLAMA_VLM_MODEL", "qwen2.5vl:7b"))
        self.anthropic_key_var = tk.StringVar(value=current.get("ANTHROPIC_API_KEY", ""))
        self.hf_token_var = tk.StringVar(value=current.get("HF_TOKEN", ""))

        r = 0
        ttk.Label(form, text="LLM_BACKEND:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Combobox(form, textvariable=self.llm_backend_setting_var, values=LLM_BACKENDS,
                    state="readonly", width=20).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Label(form, text="OLLAMA_BASE_URL:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(form, textvariable=self.ollama_url_var, width=40).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Label(form, text="OLLAMA_MODEL (notes):").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(form, textvariable=self.ollama_model_var, width=40).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Label(form, text="OLLAMA_VLM_MODEL (slides):").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(form, textvariable=self.ollama_vlm_model_var, width=40).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Label(form, text="ANTHROPIC_API_KEY:").grid(row=r, column=0, sticky="w", **pad)
        anth_entry = ttk.Entry(form, textvariable=self.anthropic_key_var, width=40, show="*")
        anth_entry.grid(row=r, column=1, sticky="w")
        ttk.Button(form, text="Show", command=lambda: self._toggle_mask(anth_entry)).grid(
            row=r, column=2, sticky="w")
        r += 1

        ttk.Label(form, text="HF_TOKEN:").grid(row=r, column=0, sticky="w", **pad)
        hf_entry = ttk.Entry(form, textvariable=self.hf_token_var, width=40, show="*")
        hf_entry.grid(row=r, column=1, sticky="w")
        ttk.Button(form, text="Show", command=lambda: self._toggle_mask(hf_entry)).grid(
            row=r, column=2, sticky="w")
        r += 1

        btn_row = ttk.Frame(form)
        btn_row.grid(row=r, column=0, columnspan=3, sticky="w", pady=(10, 4))
        ttk.Button(btn_row, text="Save keys.cfg", command=self._save_keys).pack(side="left", padx=(8, 8))
        self.settings_status = ttk.Label(btn_row, text="")
        self.settings_status.pack(side="left")

        editor_frame = ttk.LabelFrame(parent, text="Everything else")
        editor_frame.pack(fill="x", **pad)
        ttk.Button(editor_frame, text="Open config.py in editor",
                  command=lambda: self._open_in_editor(Path("config.py"))).pack(
            side="left", padx=8, pady=8)
        ttk.Button(editor_frame, text="Open keys.cfg in editor",
                  command=lambda: self._open_in_editor(Path("keys.cfg"))).pack(
            side="left", padx=8, pady=8)
        ttk.Button(editor_frame, text="Open project folder",
                  command=self._open_project_folder).pack(side="left", padx=8, pady=8)

        # --- Requirements ---
        req_frame = ttk.LabelFrame(parent, text="Requirements")
        req_frame.pack(fill="x", **pad)
        req_btn_row = ttk.Frame(req_frame)
        req_btn_row.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Button(req_btn_row, text="Check requirements", command=self._check_requirements).pack(side="left")
        self.install_missing_btn = ttk.Button(
            req_btn_row, text="Install missing", command=self._install_missing, state="disabled")
        self.install_missing_btn.pack(side="left", padx=(8, 0))
        self.req_text = scrolledtext.ScrolledText(
            req_frame, height=9, state="disabled", wrap="word",
            bg=_theme_color("neutral_100"), fg=_theme_color("neutral_900"),
            insertbackground=_theme_color("neutral_900"),
            selectbackground=_theme_color("primary_300"))
        self.req_text.pack(fill="x", padx=8, pady=(0, 8))

        # --- Database ---
        db_frame = ttk.LabelFrame(parent, text="Database")
        db_frame.pack(fill="x", **pad)
        self.db_path_var = tk.StringVar(value=CONFIG["db_path"])
        self.db_browser_path_var = tk.StringVar(value=CONFIG.get("db_browser_path", ""))
        ttk.Label(db_frame, text="Current file:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(db_frame, textvariable=self.db_path_var, width=60).grid(
            row=0, column=1, sticky="we", columnspan=3, padx=8)
        db_frame.columnconfigure(1, weight=1)
        db_btn_row = ttk.Frame(db_frame)
        db_btn_row.grid(row=1, column=0, columnspan=4, sticky="w", padx=8, pady=(4, 4))
        ttk.Button(db_btn_row, text="Open existing...", command=self._db_open).pack(side="left")
        ttk.Button(db_btn_row, text="New database...", command=self._db_new).pack(side="left", padx=(8, 0))
        ttk.Button(db_btn_row, text="Rename current...", command=self._db_rename).pack(side="left", padx=(8, 0))
        ttk.Button(db_btn_row, text="Refresh stats", command=self._refresh_db_stats).pack(side="left", padx=(8, 0))
        ttk.Button(db_btn_row, text="Open in DB Browser for SQLite...",
                  command=self._open_in_db_browser).pack(side="left", padx=(8, 0))
        self.db_stats_label = ttk.Label(db_frame, text="", foreground="#444")
        self.db_stats_label.grid(row=2, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 8))

        # --- Speaker identification (task #82) ---
        speaker_id_frame = ttk.LabelFrame(parent, text="Speaker Identification")
        speaker_id_frame.pack(fill="x", **pad)
        self.enable_speaker_id_var = tk.BooleanVar(value=CONFIG.get("enable_speaker_id", False))
        self.speaker_id_threshold_var = tk.DoubleVar(value=CONFIG.get("speaker_id_threshold", 0.75))
        ttk.Checkbutton(
            speaker_id_frame,
            text="Suggest known speaker names in Rename Speakers (requires diarization)",
            variable=self.enable_speaker_id_var,
        ).grid(row=0, column=0, columnspan=3, sticky="w", **pad)
        ttk.Label(speaker_id_frame, text="Match threshold (cosine similarity):").grid(
            row=1, column=0, sticky="w", **pad)
        ttk.Spinbox(speaker_id_frame, textvariable=self.speaker_id_threshold_var,
                   from_=0.50, to=0.99, increment=0.05, width=6, format="%.2f").grid(
            row=1, column=1, sticky="w")
        ttk.Label(speaker_id_frame,
                 text="Higher = fewer false matches but more new speakers created; "
                      "lower = more auto-matches but more risk of mixing up two people.",
                 foreground="#666", wraplength=600).grid(
            row=2, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 4))
        ttk.Button(speaker_id_frame, text="Manage Known Speakers...",
                  command=self._open_known_speakers_manager).grid(
            row=3, column=0, columnspan=3, sticky="w", padx=8, pady=(4, 8))
        ttk.Label(speaker_id_frame,
                 text="Batch review (task #92): after processing several recordings "
                      "overnight, export every detected speaker across all of them into "
                      "one JSON file, listen to each sample and type in names in the "
                      "Review && Listen window, then apply them all at once - instead of "
                      "opening Rename Speakers per file.",
                 foreground="#666", wraplength=600).grid(
            row=4, column=0, columnspan=3, sticky="w", padx=8, pady=(4, 0))
        roster_btn_row = ttk.Frame(speaker_id_frame)
        roster_btn_row.grid(row=5, column=0, columnspan=3, sticky="w", padx=8, pady=(4, 8))
        ttk.Button(roster_btn_row, text="Export batch speaker roster...",
                  command=self._export_speaker_roster_batch).pack(side="left")
        ttk.Button(roster_btn_row, text="Review && Listen to roster...",
                  command=self._open_speaker_roster_review).pack(side="left", padx=(8, 0))
        ttk.Button(roster_btn_row, text="Apply speaker roster JSON...",
                  command=self._apply_speaker_roster_batch).pack(side="left", padx=(8, 0))

        # --- Logging ---
        log_frame = ttk.LabelFrame(parent, text="Logging")
        log_frame.pack(fill="x", **pad)
        self.log_level_var = tk.StringVar(value=CONFIG.get("log_level", "INFO"))
        self.log_to_file_var = tk.BooleanVar(value=CONFIG.get("log_to_file", False))
        self.log_dir_var = tk.StringVar(value=CONFIG.get("log_dir", ""))
        ttk.Label(log_frame, text="Log level:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Combobox(log_frame, textvariable=self.log_level_var, values=LOG_LEVELS,
                    state="readonly", width=12).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(log_frame, text="Also save this run's log to a file",
                       variable=self.log_to_file_var).grid(row=1, column=0, columnspan=2, sticky="w", **pad)
        ttk.Label(log_frame, text="Log folder:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(log_frame, textvariable=self.log_dir_var, width=45).grid(row=2, column=1, sticky="w")
        ttk.Button(log_frame, text="Browse...", command=self._browse_log_dir).grid(row=2, column=2, sticky="w")

        # --- Backup (task #83) ---
        backup_frame = ttk.LabelFrame(parent, text="Backup")
        backup_frame.pack(fill="x", **pad)
        ttk.Label(backup_frame,
                 text="Zips the database, persisted GUI settings, and prompt templates "
                      "into one file. Also includes keys.cfg (your HuggingFace token and "
                      "Anthropic API key in plain text) - keep the resulting zip as private "
                      "as keys.cfg itself.",
                 foreground="#666", wraplength=700).pack(anchor="w", padx=8, pady=(8, 4))
        ttk.Button(backup_frame, text="Backup all databases + settings to ZIP...",
                  command=self._backup_all).pack(anchor="w", padx=8, pady=(0, 8))

        self._refresh_db_stats()

    def _toggle_mask(self, entry: ttk.Entry):
        entry.config(show="" if entry.cget("show") == "*" else "*")

    def _save_keys(self):
        values = {
            "LLM_BACKEND": self.llm_backend_setting_var.get(),
            "OLLAMA_BASE_URL": self.ollama_url_var.get(),
            "OLLAMA_MODEL": self.ollama_model_var.get(),
            "OLLAMA_VLM_MODEL": self.ollama_vlm_model_var.get(),
            "ANTHROPIC_API_KEY": self.anthropic_key_var.get(),
            "HF_TOKEN": self.hf_token_var.get(),
        }
        try:
            path = keys_loader.save_keys(values)
            self.settings_status.config(text=f"Saved: {path.name} (restart to apply)", foreground="green")
        except Exception as exc:
            self.settings_status.config(text=f"Save failed: {exc}", foreground="red")

    def _open_in_editor(self, path: Path):
        self._open_path(path.resolve())

    def _open_known_speakers_manager(self):
        """Open the global known-speakers roster manager (task #82)."""
        KnownSpeakersDialog(self)

    def _export_speaker_roster_batch(self):
        """Task #92: scan a folder of already-processed recordings and
        write every detected speaker across ALL of them into one JSON
        file (run_pipeline.export_speaker_roster_json), for reviewing an
        overnight batch of several recordings in one sitting instead of
        opening each file's Rename Speakers dialog individually.
        Voiceprint suggestions use the same hf_token/threshold/roster as
        that dialog, and a short .wav sample per speaker is extracted
        next to the JSON (same "Play sample" clip logic as that dialog)
        so voices can still be checked by ear across the whole batch.
        Runs synchronously (like Backup all databases... above) - this
        is an occasional, manual action, not a per-file live operation,
        so a brief wait cursor is enough rather than a background
        thread.

        V1.29: offers to open SpeakerRosterReviewDialog immediately after
        a successful export, so listening to samples and typing names
        can start right away instead of the user having to separately
        open the JSON in a text editor and double-click .wav files in
        Explorer (still possible, just no longer the main path)."""
        folder = filedialog.askdirectory(title="Select folder with processed recordings")
        if not folder:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = filedialog.asksaveasfilename(
            title="Save speaker roster as", initialdir=folder,
            initialfile=f"speaker_roster_{timestamp}.json",
            defaultextension=".json", filetypes=[("JSON", "*.json")],
        )
        if not out_path:
            return
        recursive = messagebox.askyesno(
            "Include subfolders?",
            "Also scan subfolders of the selected folder for processed recordings?")

        hf_token = CONFIG.get("hf_token", "")
        threshold = self.speaker_id_threshold_var.get()
        db_path = self.db_path_var.get().strip() or CONFIG["db_path"]

        self.root.config(cursor="watch")
        self.root.update()
        try:
            result = run_pipeline.export_speaker_roster_json(
                folder, out_path, hf_token, threshold, db_path, recursive=recursive)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        finally:
            self.root.config(cursor="")

        samples_note = (
            f"{result['samples']} sample clip(s) were extracted, ready to play in the "
            f"review window.\n\n"
            if result.get("samples") else
            "No sample clips were extracted (ffmpeg missing, or no source media "
            "found next to the transcripts) - name speakers from labels/context "
            "alone.\n\n"
        )
        review_now = messagebox.askyesno(
            "Speaker roster exported",
            f"{result['files']} file(s), {result['speakers']} speaker(s).\n\n"
            f"{samples_note}"
            f"Review and listen to each speaker now?")
        if review_now:
            self._open_speaker_roster_review(result["path"])

    def _open_speaker_roster_review(self, json_path: str | None = None):
        """Opens SpeakerRosterReviewDialog (V1.29, task #92 batch review
        UI) over json_path, prompting for one via a file picker if not
        given - e.g. called directly from the "Review & Listen to
        roster..." button for a previously-exported roster, rather than
        right after a fresh export."""
        if not json_path:
            json_path = filedialog.askopenfilename(
                title="Select speaker roster JSON",
                filetypes=[("JSON", "*.json"), ("All files", "*.*")])
            if not json_path:
                return
        SpeakerRosterReviewDialog(self, json_path)

    def _apply_speaker_roster_batch(self):
        """Task #92: read back a speaker-roster JSON (see
        _export_speaker_roster_batch, after the user has edited each
        speaker's "name" field) and apply the renames + known-speakers
        roster updates across every file listed in one pass
        (run_pipeline.apply_speaker_roster_json)."""
        json_path = filedialog.askopenfilename(
            title="Select speaker roster JSON", filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if not json_path:
            return
        regenerate_notes = messagebox.askyesno(
            "Regenerate notes?",
            "Also regenerate each file's notes/summary, for files where at least "
            "one speaker was actually renamed?")

        self.root.config(cursor="watch")
        self.root.update()
        try:
            result = run_pipeline.apply_speaker_roster_json(
                json_path, regenerate_notes=regenerate_notes)
        except Exception as exc:
            messagebox.showerror("Apply failed", str(exc))
            return
        finally:
            self.root.config(cursor="")

        msg = (f"{result['files_updated']} file(s) updated, "
               f"{result['speakers_renamed']} segment(s) renamed, "
               f"{result['roster_updates']} roster update(s).")
        if result["errors"]:
            details = "\n".join(f"- {e['file']}: {e['error']}" for e in result["errors"][:10])
            msg += f"\n\n{len(result['errors'])} file(s) failed:\n{details}"
        messagebox.showinfo("Speaker roster applied", msg)

    def _backup_all(self):
        """Zip the database, persisted GUI settings (gui_state.json),
        prompt templates, and keys.cfg into one timestamped archive
        (task #83). keys.cfg is always included per explicit user choice
        made when this feature was designed - it holds the HuggingFace
        token and Anthropic API key in plain text, so the resulting zip
        needs the same handling as keys.cfg itself. prompts/_backup/
        (task #77's own automatic pre-save/delete backups) is excluded -
        those are backups of backups, not source."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"TNP_Backup_{timestamp}.zip"
        out_path = filedialog.asksaveasfilename(
            title="Save backup as", initialfile=default_name,
            defaultextension=".zip", filetypes=[("ZIP archive", "*.zip")],
        )
        if not out_path:
            return
        try:
            import zipfile
            db_path = Path(self.db_path_var.get().strip() or CONFIG["db_path"])
            with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
                if db_path.exists():
                    zf.write(db_path, arcname=db_path.name)
                if STATE_PATH.exists():
                    zf.write(STATE_PATH, arcname=STATE_PATH.name)
                keys_path = Path("keys.cfg")
                if keys_path.exists():
                    zf.write(keys_path, arcname=keys_path.name)
                prompts_dir = Path(PROMPTS_DIR)
                if prompts_dir.exists():
                    for f in prompts_dir.rglob("*"):
                        if not f.is_file():
                            continue
                        if "_backup" in f.relative_to(prompts_dir).parts:
                            continue
                        zf.write(f, arcname=str(Path("prompts") / f.relative_to(prompts_dir)))
            messagebox.showinfo("Backup complete", f"Saved to:\n{out_path}")
        except Exception as exc:
            messagebox.showerror("Backup failed", str(exc))

    def _open_path(self, path: Path):
        if not path.exists():
            messagebox.showwarning("Not found", f"{path} does not exist yet.")
            return
        try:
            if hasattr(os, "startfile"):
                os.startfile(str(path))
            elif sys.platform == "darwin":
                subprocess.run(["open", str(path)])
            else:
                subprocess.run(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("Could not open", str(exc))

    def _play_audio(self, path: Path):
        """Play an audio file (speaker sample preview) via ffplay - a
        plain window-less player, so it doesn't hand the clip off to
        whatever the OS's default app for .wav happens to be (Windows
        Media Player, Groove Music, etc.) with all of that app's own
        startup time and UI. Falls back to _open_path (the OS default
        app) if ffplay isn't available (bundled ffmpeg\\bin\\ or PATH,
        see config.get_ffplay_path).

        V1.28: stops any clip already playing first (_stop_audio) - only
        one preview should ever be audible at a time, so starting a new
        one always supersedes the last rather than overlapping it."""
        if not path.exists():
            messagebox.showwarning("Not found", f"{path} does not exist yet.")
            return
        self._stop_audio()
        ffplay_exe = get_ffplay_path()
        if ffplay_exe:
            try:
                self._current_playback_proc = subprocess.Popen(
                    [ffplay_exe, "-autoexit", "-nodisp", "-loglevel", "error", str(path)])
                return
            except Exception as exc:
                log.warning("ffplay launch failed (%s) - falling back to the "
                            "default player.", exc)
        self._open_path(path)

    def _stop_audio(self):
        """Stop button (V1.28): terminates whatever ffplay process
        _play_audio last launched, if it's still running. Safe to call
        with nothing playing (no-op) - _current_playback_proc may not
        exist yet if _play_audio was never called, and ffplay may have
        already exited on its own (-autoexit, or the clip simply
        finished)."""
        proc = getattr(self, "_current_playback_proc", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception as exc:
                log.warning("Could not stop playback: %s", exc)
        self._current_playback_proc = None

    def _open_project_folder(self):
        self._open_path(Path(__file__).parent.resolve())

    def _browse_log_dir(self):
        path = filedialog.askdirectory(title="Select log folder")
        if path:
            self.log_dir_var.set(path)

    # -----------------------------------------------------------------
    # Requirements check / install
    # -----------------------------------------------------------------

    def _check_requirements(self):
        import requirements_check
        results = requirements_check.check_all()
        self._last_check_results = results
        self._req_set_text(requirements_check.format_report(results))
        missing = requirements_check.missing_required(results) + requirements_check.missing_optional(results)
        self.install_missing_btn.config(state=("normal" if missing else "disabled"))

    def _req_set_text(self, text: str):
        self.req_text.config(state="normal")
        self.req_text.delete("1.0", "end")
        self.req_text.insert("1.0", text)
        self.req_text.config(state="disabled")

    def _req_append(self, line: str):
        self.req_text.config(state="normal")
        self.req_text.insert("end", line + "\n")
        self.req_text.see("end")
        self.req_text.config(state="disabled")

    def _install_missing(self):
        self.install_missing_btn.config(state="disabled")
        import requirements_check
        results = self._last_check_results or requirements_check.check_all()

        def worker():
            ok = requirements_check.install_missing(results, log_fn=lambda m: self.req_queue.put(m))
            self.req_queue.put("__INSTALL_DONE_OK__" if ok else "__INSTALL_DONE_FAIL__")

        threading.Thread(target=worker, daemon=True).start()

    def _poll_req_queue(self):
        try:
            while True:
                line = self.req_queue.get_nowait()
                if line in ("__INSTALL_DONE_OK__", "__INSTALL_DONE_FAIL__"):
                    self._req_append("--- install finished"
                                     f" ({'all OK' if line.endswith('OK__') else 'some failed'}) ---")
                    self._check_requirements()
                    continue
                self._req_append(line)
        except queue.Empty:
            pass
        self.root.after(200, self._poll_req_queue)

    # -----------------------------------------------------------------
    # Database file management
    # -----------------------------------------------------------------

    def _refresh_db_stats(self):
        try:
            stats = db.get_db_stats(self.db_path_var.get().strip())
        except Exception as exc:
            self.db_stats_label.config(text=f"Could not read database: {exc}")
            return
        if not stats.get("exists"):
            self.db_stats_label.config(text="Database not created yet (will be created on first run).")
            return
        size_kb = stats["size_bytes"] / 1024
        fts_note = "" if stats.get("fts5_available", True) else "  [FTS5 unavailable, search uses LIKE fallback]"
        self.db_stats_label.config(
            text=f"{stats['transcripts']} transcript(s), {stats['notes_generated']} with notes, "
                 f"{stats['slides']} slide(s), {stats['runs']} run(s), {size_kb:.0f} KB{fts_note}")

    def _db_open(self):
        path = filedialog.askopenfilename(title="Open database",
                                          filetypes=[("SQLite DB", "*.db"), ("All files", "*.*")])
        if not path:
            return
        self.db_path_var.set(path)
        db.validate_schema(path)
        self._refresh_db_stats()

    def _db_new(self):
        path = filedialog.asksaveasfilename(title="New database", defaultextension=".db",
                                            filetypes=[("SQLite DB", "*.db")])
        if not path:
            return
        try:
            db.create_new_db(path)
        except Exception as exc:
            messagebox.showerror("Could not create database", str(exc))
            return
        self.db_path_var.set(path)
        self._refresh_db_stats()

    def _db_rename(self):
        current = self.db_path_var.get().strip()
        if not current or not Path(current).exists():
            messagebox.showwarning("No database", "Current database file does not exist yet.")
            return
        new_path = filedialog.asksaveasfilename(
            title="Rename database to", defaultextension=".db",
            initialfile=Path(current).name, filetypes=[("SQLite DB", "*.db")])
        if not new_path:
            return
        try:
            db.rename_db(current, new_path)
        except Exception as exc:
            messagebox.showerror("Rename failed", str(exc))
            return
        self.db_path_var.set(new_path)
        self._refresh_db_stats()

    def _find_db_browser(self) -> str:
        """Return a usable "DB Browser for SQLite" executable path, or ""
        if none can be found (task #75). Checks (in order): a path already
        browsed for this session, config.py's db_browser_path, PATH, and
        the standard Windows install locations (both Program Files and
        Program Files (x86), since the official installer offers either)."""
        for candidate in (
            self.db_browser_path_var.get().strip(),
            CONFIG.get("db_browser_path", "").strip(),
        ):
            if candidate and Path(candidate).exists():
                return candidate
        for name in ("DB Browser for SQLite", "sqlitebrowser"):
            found = shutil.which(name)
            if found:
                return found
        for base in (os.environ.get("ProgramFiles", r"C:\Program Files"),
                    os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")):
            default = Path(base) / "DB Browser for SQLite" / "DB Browser for SQLite.exe"
            if default.exists():
                return str(default)
        return ""

    def _open_in_db_browser(self):
        """Launch DB Browser for SQLite (free, MIT-licensed SQLite GUI
        editor, https://sqlitebrowser.org/) against the current database
        file, for manual inspection/editing (task #75). Does not touch the
        pipeline in any way - purely a convenience launcher, same pattern
        as RunTabController._open_in_losslesscut."""
        db_path = self.db_path_var.get().strip()
        if not db_path or not Path(db_path).exists():
            messagebox.showwarning("No database", "Current database file does not exist yet. "
                                   "Run the pipeline once, or use \"New database...\" above.")
            return
        exe = self._find_db_browser()
        if not exe:
            messagebox.showinfo(
                "DB Browser for SQLite not found",
                "Could not auto-detect DB Browser for SQLite. Free download:\n"
                "https://sqlitebrowser.org/dl/\n\n"
                "Select the DB Browser for SQLite executable on the next screen.")
            exe = filedialog.askopenfilename(
                title="Select DB Browser for SQLite executable",
                filetypes=[("DB Browser for SQLite", "*.exe"), ("All files", "*.*")])
            if not exe:
                return
            self.db_browser_path_var.set(exe)
        try:
            subprocess.Popen([exe, db_path])
        except Exception as exc:
            messagebox.showerror("Could not launch DB Browser for SQLite", str(exc))


class RunTabController:
    """
    Self-contained controller for one Run-style tab (Meeting or Video /
    Webinar). Each instance owns its own Mode/path/output fields, batch
    table, parameters, stages, formats, Run/Stop button, and log panel,
    built into the ttk.Frame handed to it by PipelineGUI - one instance
    per tab, so switching tabs never mixes up the file/batch/meeting-info
    being worked on between a meeting run and a video/webinar run.

    Exception: whisper_model_var, language_var, output_language_var,
    llm_backend_var, enable_diarization_var, no_summary_var, and
    force_retranscribe_var are NOT this instance's own Variables - they
    are the same shared objects on both tabs (see PipelineGUI.__init__'s
    shared_*_var and gui_logic.PERSISTED_KEYS_SHARED), so these specific
    transcription/LLM engine settings are always identical on the
    Meeting and Video/Webinar tabs, editable from either.

    kind is "meeting" or "video":
      meeting: no slide detection/VLM/Q&A/slide-report formats/Review-
               Edit-Slides/Re-annotate-failed-slides - a plain transcript
               plus LLM summary only.
      video:   everything, including slide detection, VLM annotation,
               the Q&A section (task #61), slide-report formats, and the
               Review/Edit Slides + Re-annotate failed slides buttons.

    app is the owning PipelineGUI instance, used for the handful of
    things that stay shared across every tab: the database path field
    and logging options (both on the Settings tab), _make_scrollable/
    _open_prompts_folder/_open_path, and the shared_*_var Variables
    described above.
    """

    def __init__(self, tab: ttk.Frame, kind: str, app: PipelineGUI):
        self.tab = tab
        self.kind = kind
        self.app = app
        self.root = app.root

        self.log_queue: queue.Queue = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        # Batch Treeview iids for the run currently in progress, in the same
        # order/index as the rows passed to run_pipeline.run_batch_rows, so
        # a "__BATCH_STATUS__:<i>:<status>" message from the worker thread
        # can be mapped back to the right row (task #74).
        self._batch_run_iids: list = []

        self._build()
        self.root.after(100, self._poll_log_queue)

    # -----------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------

    def _build(self):
        parent = self.app._make_scrollable(self.tab)
        pad = {"padx": 8, "pady": 4}
        is_video = self.kind == "video"

        self.mode_var = tk.StringVar(value=MODES[0])
        self.path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.output_basename_var = tk.StringVar()
        self.use_date_subject_filename_var = tk.BooleanVar(
            value=CONFIG.get("use_date_subject_filename", False))
        self.move_processed_files_var = tk.BooleanVar(
            value=CONFIG.get("move_processed_files", False))
        self.prompt_var = tk.StringVar()

        # Shared with the other Run tab (Meeting <-> Video/Webinar) - see
        # PipelineGUI.__init__ where these are created once and
        # gui_logic.PERSISTED_KEYS_SHARED, which persists them under
        # gui_state.json's "shared" key instead of duplicating them here.
        # Kept as like-named attributes (self.whisper_model_var, etc.) so
        # every widget/override/state_vars reference below is unchanged -
        # only where the Variable comes from is different now.
        self.whisper_model_var = self.app.shared_whisper_model_var
        self.language_var = self.app.shared_language_var
        self.output_language_var = self.app.shared_output_language_var
        self.llm_backend_var = self.app.shared_llm_backend_var
        self.enable_diarization_var = self.app.shared_enable_diarization_var
        self.no_summary_var = self.app.shared_no_summary_var
        self.force_retranscribe_var = self.app.shared_force_retranscribe_var
        self.enhance_audio_var = self.app.shared_enhance_audio_var

        self.losslesscut_path_var = tk.StringVar(value=CONFIG.get("losslesscut_path", ""))

        if is_video:
            self.enable_slides_var = tk.BooleanVar(value=CONFIG["enable_slides"])
            self.enable_vlm_var = tk.BooleanVar(value=CONFIG["enable_vlm"])
            self.dry_run_var = tk.BooleanVar(value=False)
            self.threshold_var = tk.IntVar(value=CONFIG["hash_threshold"])
            self.animation_threshold_var = tk.IntVar(value=CONFIG.get("animation_threshold", 0))
            self.fps_var = tk.IntVar(value=CONFIG["fps"])
            self.min_slide_duration_var = tk.DoubleVar(value=CONFIG["min_slide_duration_sec"])
            self.recording_speed_var = tk.DoubleVar(value=CONFIG.get("recording_speed", 1.0))
            self.convert_video_to_realtime_var = tk.BooleanVar(
                value=CONFIG.get("convert_video_to_realtime", False))
            self.title_slide_image_var = tk.StringVar(value=CONFIG.get("title_slide_image_path", ""))
            self.report_show_image_var = tk.BooleanVar(value=CONFIG.get("report_show_image", True))
            self.report_show_bullets_var = tk.BooleanVar(value=CONFIG.get("report_show_bullets", True))
            self.report_show_transcript_var = tk.BooleanVar(value=CONFIG.get("report_show_transcript", True))
            self.report_transcript_mode_var = tk.StringVar(value=CONFIG.get("report_transcript_mode", "full"))
            self.report_html_var = tk.BooleanVar(value=CONFIG["report_html"])
            self.report_csv_var = tk.BooleanVar(value=CONFIG["report_csv"])
            self.report_json_var = tk.BooleanVar(value=CONFIG["report_json"])
            self.report_srt_var = tk.BooleanVar(value=CONFIG["report_srt"])
            self.report_pdf_var = tk.BooleanVar(value=CONFIG["report_pdf"])
            self.report_slide_timing_var = tk.BooleanVar(value=CONFIG.get("report_slide_timing", True))
            self.zip_snapshots_var = tk.BooleanVar(value=CONFIG.get("zip_snapshots", False))

        self.notes_txt_var = tk.BooleanVar(value=CONFIG["notes_format_txt"])
        self.notes_html_var = tk.BooleanVar(value=CONFIG["notes_format_html"])
        self.notes_pdf_var = tk.BooleanVar(value=CONFIG["notes_format_pdf"])
        self.notes_docx_var = tk.BooleanVar(value=CONFIG["notes_format_docx"])

        top = ttk.Frame(parent)
        top.pack(fill="x", **pad)

        ttk.Label(top, text="Mode:").grid(row=0, column=0, sticky="w")
        mode_box = ttk.Combobox(top, textvariable=self.mode_var, values=MODES,
                                state="readonly", width=32)
        mode_box.grid(row=0, column=1, sticky="w", columnspan=2)
        mode_box.bind("<<ComboboxSelected>>", lambda e: self._update_path_label())

        self.path_label = ttk.Label(top, text="File:")
        self.path_label.grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.path_var, width=60).grid(row=1, column=1, sticky="we")
        ttk.Button(top, text="Browse...", command=self._browse).grid(row=1, column=2, padx=4)

        ttk.Label(top, text="Output folder (optional):").grid(row=2, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.output_dir_var, width=60).grid(row=2, column=1, sticky="we")
        ttk.Button(top, text="Browse...", command=self._browse_output_dir).grid(row=2, column=2, padx=4)
        ttk.Label(top, text="Blank = write next to the source file (default).",
                 foreground="#666").grid(row=3, column=1, sticky="w")
        move_processed_cb = ttk.Checkbutton(
            top, text=f"Move source file to a \"{CONFIG.get('processed_subfolder_name', '_processed')}\" "
                     "subfolder once fully processed",
            variable=self.move_processed_files_var)
        move_processed_cb.grid(row=4, column=0, columnspan=2, sticky="w")
        Tooltip(move_processed_cb,
               "Only the ORIGINAL source audio/video file moves, next to where it already is - "
               "transcript/notes/slide report outputs stay put. Only runs after the file finishes "
               "completely (never on failure or a dry run). Play Sample, \"Improve audio\", and "
               "Rename Speakers keep working for archived files afterward. A folder scan of this "
               "same folder (recursive or not) will not pick the file back up. Off by default.")

        ttk.Label(top, text="Output filename (optional):").grid(row=5, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.output_basename_var, width=60).grid(row=5, column=1, sticky="we")
        ttk.Label(top, text="Blank = use the source filename for every output file (default).",
                 foreground="#666").grid(row=6, column=1, sticky="w")
        date_subject_cb = ttk.Checkbutton(
            top, text="Name files \"YYYYMMDD_Subject\" from Meeting info's Title/Date below",
            variable=self.use_date_subject_filename_var)
        date_subject_cb.grid(row=7, column=1, sticky="w")
        Tooltip(date_subject_cb,
               "Uses the Title/Date fields under \"Meeting info\" (filled manually, from "
               "--ics/\"Load meeting info...\", or from a correlate_calendar_recordings.py "
               "batch CSV) to name every output file e.g. \"20260703_Q3 Kickoff\" instead of "
               "the source filename. Ignored (falls back to the usual naming) for any file "
               "where Title is blank. An explicit \"Output filename\" above always wins over "
               "this.")
        top.columnconfigure(1, weight=1)

        # --- Workflow steps (task #61/#112 GUI remodel) ---
        # The Video/Webinar tab's settings are grouped into an explicit
        # 3-step sequence matching how a recording is actually worked
        # through: transcribe + detect first, then review the detected
        # slides and pin down where Q&A starts, then pick output formats
        # and generate the reports/summary. The Meeting tab has no
        # review phase (no slides to check, no Q&A concept), so it keeps
        # its original flat, single-page layout - step1_target/
        # step2_target/step3_target all just point at "parent" in that
        # case, so every LabelFrame below lands exactly where it always
        # has.
        if is_video:
            ttk.Label(parent,
                     text="Work through the tabs below in order: 1) transcribe and detect "
                          "slides, 2) review what was found and mark where Q&A starts, "
                          "3) generate the summary and reports.",
                     foreground="#666", wraplength=860).pack(fill="x", padx=8, pady=(0, 4))
            self._steps_nb = ttk.Notebook(parent)
            self._steps_nb.pack(fill="both", expand=True, padx=8, pady=(0, 4))
            self._step1_frame = ttk.Frame(self._steps_nb)
            self._step2_frame = ttk.Frame(self._steps_nb)
            self._step3_frame = ttk.Frame(self._steps_nb)
            self._steps_nb.add(self._step1_frame, text="1. Transcribe & Detect")
            self._steps_nb.add(self._step2_frame, text="2. Review & Q&A")
            self._steps_nb.add(self._step3_frame, text="3. Reports & Summary")
            step1_target = self._step1_frame
            step2_target = self._step2_frame
            step3_target = self._step3_frame
        else:
            self._steps_nb = None
            step1_target = step2_target = step3_target = parent

        # --- Recording type preset ---
        rec_types = VIDEO_RECORDING_TYPES if is_video else MEETING_RECORDING_TYPES
        self._recording_types = rec_types
        self.recording_type_var = tk.StringVar(value=next(iter(rec_types)))
        rec_frame = ttk.LabelFrame(step1_target, text="Recording type (preset)")
        rec_frame.pack(fill="x", padx=8, pady=4)
        self._rec_frame = rec_frame
        ttk.Label(rec_frame, text="Type:").pack(side="left", padx=8, pady=8)
        rec_box = ttk.Combobox(rec_frame, textvariable=self.recording_type_var,
                               values=list(rec_types.keys()), state="readonly", width=22)
        rec_box.pack(side="left", pady=8)
        rec_box.bind("<<ComboboxSelected>>", lambda e: self._apply_recording_type())
        ttk.Label(rec_frame, text="Sets the prompt template below; still editable afterward.",
                 foreground="#666").pack(side="left", padx=(12, 0))
        # No more "Copy settings to other tab" button here: whisper model,
        # language, LLM backend, diarization, no-summary, and
        # force-retranscribe are now shared live between the Meeting and
        # Video/Webinar tabs (see PipelineGUI.__init__'s shared_*_var), so
        # there is nothing left to copy - editing either tab already
        # updates both.

        # --- Meeting info (optional) ---
        meeting_frame = ttk.LabelFrame(step1_target, text="Meeting info (optional)")
        self._meeting_frame = meeting_frame
        self.meeting_title_var = tk.StringVar()
        self.meeting_date_var = tk.StringVar()
        self.meeting_comments_var = tk.StringVar()
        # Task #95: attendees loaded alongside Title/Date/Comments from an
        # .ics invite or a .txt/.docx "Invitees:" line - not shown as its
        # own entry field, but offered as the speaker-name pick-list (ahead
        # of a blind guess) when Rename Speakers is opened from this tab.
        # See _load_meeting_info and _get_invitee_names.
        self.meeting_attendees: list = []
        self.meeting_attendees_status_var = tk.StringVar(value="")
        ttk.Label(meeting_frame, text="Title:").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(meeting_frame, textvariable=self.meeting_title_var, width=45).grid(
            row=0, column=1, sticky="w", columnspan=2)
        ttk.Label(meeting_frame, text="Date (YYYY-MM-DD):").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(meeting_frame, textvariable=self.meeting_date_var, width=16).grid(row=1, column=1, sticky="w")
        ttk.Button(meeting_frame, text="Load meeting info...", command=self._load_meeting_info).grid(
            row=1, column=2, sticky="w", padx=(8, 0))
        ttk.Label(meeting_frame, text="Comments:").grid(row=2, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(meeting_frame, textvariable=self.meeting_comments_var, width=60).grid(
            row=2, column=1, sticky="we", columnspan=2)
        ttk.Label(meeting_frame,
                 text="Pre-fills Title/Date/Comments from an Outlook/Google/Teams .ics invite, "
                      "or a .txt/.docx file with \"Title:\"/\"Date:\"/\"Comments:\" lines. Shown "
                      "in the notes header" + (" and used as the report title." if is_video else "."),
                 foreground="#666").grid(row=3, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 4))
        ttk.Label(meeting_frame, textvariable=self.meeting_attendees_status_var,
                 foreground="#666").grid(row=4, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 4))

        # --- Title slide image (optional, video/webinar only) ---
        # Deliberately its OWN frame, not nested in meeting_frame above:
        # meeting_frame gets hidden in batch mode (its fields are all
        # per-row there already - see _update_path_label), and this one
        # must not be, since it is the FALLBACK value for any batch row
        # whose own "Title image" cell is left blank (see batch_columns/
        # gui_logic.build_batch_row) - each webinar can have its own cover
        # image via that per-row column; this field only applies when a
        # row does not set one (or in single-file mode, where there is no
        # batch table at all).
        self._title_slide_frame = None
        if is_video:
            title_slide_frame = ttk.LabelFrame(step1_target, text="Title slide image (optional, default)")
            self._title_slide_frame = title_slide_frame
            ttk.Label(title_slide_frame, text="Image file:").grid(row=0, column=0, sticky="w", padx=8, pady=4)
            ttk.Entry(title_slide_frame, textvariable=self.title_slide_image_var, width=45).grid(
                row=0, column=1, sticky="w")
            ttk.Button(title_slide_frame, text="Browse...", command=self._browse_title_slide_image).grid(
                row=0, column=2, sticky="w", padx=(8, 0))
            ttk.Label(title_slide_frame,
                     text="Optional cover image (jpg/png) shown as a title page before Slide 1 "
                          "in the HTML/PDF slide report. In batch mode, each row's own 'Title "
                          "image' cell (double-click to edit) takes priority; this field is only "
                          "the fallback for rows that leave that cell blank.",
                     foreground="#666").grid(row=1, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 4))

        # --- Q&A section (video only, task #61) ---
        self._qa_frame = None
        if is_video:
            qa_frame = ttk.LabelFrame(step2_target, text="Q&A section (optional)")
            self._qa_frame = qa_frame
            self.qa_start_var = tk.StringVar()
            self.qa_end_var = tk.StringVar()
            self.qa_include_in_summary_var = tk.BooleanVar(
                value=CONFIG.get("qa_include_in_summary", True))
            # CONFIG's qa_autodetect_start is tri-state (None = webinar
            # template only). The checkbox shows whether detection would run
            # for a webinar, which is the case this feature exists for.
            self.qa_autodetect_var = tk.BooleanVar(
                value=CONFIG.get("qa_autodetect_start") is not False)
            ttk.Label(qa_frame, text="Starts at:").grid(row=0, column=0, sticky="w", padx=8, pady=4)
            ttk.Entry(qa_frame, textvariable=self.qa_start_var, width=12).grid(row=0, column=1, sticky="w")
            ttk.Label(qa_frame, text="Ends at (optional):").grid(row=0, column=2, sticky="w", padx=(16, 4), pady=4)
            ttk.Entry(qa_frame, textvariable=self.qa_end_var, width=12).grid(row=0, column=3, sticky="w")
            ttk.Checkbutton(qa_frame, text="Include Q&A in the summary",
                            variable=self.qa_include_in_summary_var).grid(
                row=0, column=4, sticky="w", padx=(16, 4))
            cb = ttk.Checkbutton(qa_frame, text="Auto-detect Q&A start (leave \"Starts at\" blank)",
                                 variable=self.qa_autodetect_var)
            cb.grid(row=1, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 2))
            Tooltip(cb, "Finds the Q&A start from cue phrases in the transcript "
                        "(\"now to the questions and answers\", \"kommen wir zu den "
                        "Fragen\", German and English), searching only the final 40% "
                        "of the recording. Applies to the \"Webinar Transcript\" "
                        "recording type; other types are left alone, since they "
                        "rarely have a formal Q&A block. Anything typed into "
                        "\"Starts at\" always wins. Only a start is ever detected, "
                        "never an end, so the Q&A runs to the end of the recording.")

            ttk.Label(qa_frame,
                     text="For webinars with a Q&A block at the end where only the speakers are shown. "
                          "If set, no new slides are detected from here onward (to Ends at, or the end "
                          "of the recording). \"Include Q&A in the summary\" (checked by default) folds "
                          "that segment into the SAME summary as the rest of the recording, filling in "
                          "its \"QUESTIONS & ANSWERS\" section; uncheck it to leave the Q&A portion out "
                          "of the summary entirely (that section then reads \"No Q&A session\"). One "
                          "prompt, one summary either way. Blank Starts at = disabled. Accepts seconds "
                          "(975), mm:ss (16:15), or hh:mm:ss.",
                     foreground="#666", wraplength=860).grid(
                row=2, column=0, columnspan=5, sticky="w", padx=8, pady=(0, 6))

        # --- Batch table (task #62) ---
        self.batch_frame = ttk.LabelFrame(
            step1_target, text="Batch list: add files/folder, or load a .txt/.csv - double-click a cell to edit")
        if is_video:
            self.batch_columns = ("file", "language", "title", "date", "comments",
                                  "qa_start", "qa_end", "title_image", "srt_path")
        else:
            self.batch_columns = ("file", "language", "title", "date", "comments")
        # "status" (task #74) is a display-only column, not part of
        # batch_columns: it never feeds into gui_logic.build_batch_row and
        # is deliberately excluded from CSV export, so the data schema the
        # batch table round-trips stays unchanged.
        self.batch_display_columns = self.batch_columns + ("status",)
        self.batch_tree = ttk.Treeview(self.batch_frame, columns=self.batch_display_columns,
                                       show="headings", height=8)
        _all_labels = {"file": "File", "language": "Lang", "title": "Title", "date": "Date",
                      "comments": "Comments", "qa_start": "Q&A start", "qa_end": "Q&A end",
                      "title_image": "Title image", "srt_path": "Import .srt", "status": "Status"}
        _all_widths = {"file": 300, "language": 55, "title": 140, "date": 85,
                      "comments": 160, "qa_start": 75, "qa_end": 75, "title_image": 160,
                      "srt_path": 160, "status": 90}
        for col in self.batch_display_columns:
            self.batch_tree.heading(col, text=_all_labels[col])
            self.batch_tree.column(col, width=_all_widths[col], anchor="w")
        self.batch_tree.pack(fill="both", expand=True, padx=8, pady=(6, 4))
        self.batch_tree.bind("<Double-1>", self._batch_on_double_click)

        batch_btn_row = ttk.Frame(self.batch_frame)
        batch_btn_row.pack(fill="x", padx=8, pady=(0, 4))
        ttk.Button(batch_btn_row, text="Add files...", command=self._batch_add_files).pack(side="left")
        ttk.Button(batch_btn_row, text="Add folder...", command=self._batch_add_folder).pack(
            side="left", padx=(6, 0))
        ttk.Button(batch_btn_row, text="Remove selected", command=self._batch_remove_selected).pack(
            side="left", padx=(6, 0))
        ttk.Button(batch_btn_row, text="Clear", command=self._batch_clear).pack(side="left", padx=(6, 0))
        ttk.Button(batch_btn_row, text="Load .txt/.csv...", command=self._batch_load_file).pack(
            side="left", padx=(12, 0))
        ttk.Button(batch_btn_row, text="Save to .csv...", command=self._batch_save_csv).pack(
            side="left", padx=(6, 0))
        ttk.Button(batch_btn_row, text="Correlate calendar...", command=self._batch_correlate_calendar).pack(
            side="left", padx=(12, 0))
        qa_note = " Q&A start/end accept seconds (975), mm:ss (16:15), or hh:mm:ss." if is_video else ""
        ttk.Label(self.batch_frame,
                 text="Language/Title/Date/Comments" + (" /Q&A start/end" if is_video else "") +
                      " are optional per row: blank uses the shared fields above." + qa_note +
                      " Double-click a cell to edit it in place. \"Correlate calendar...\" "
                      "matches a folder of recordings against a .ics calendar export and "
                      "queues them here with Title/Date/Comments pre-filled.",
                 foreground="#666", wraplength=860).pack(fill="x", padx=8, pady=(0, 8))
        # Meeting info/Q&A/batch table visibility is driven by Mode - see
        # _update_path_label(), called once at the end of _build() below.

        # --- Parameters ---
        params = ttk.LabelFrame(step1_target, text="Parameters")
        params.pack(fill="x", **pad)

        ttk.Label(params, text="Whisper model:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Combobox(params, textvariable=self.whisper_model_var, values=WHISPER_MODELS,
                    state="readonly", width=14).grid(row=0, column=1, sticky="w")

        ttk.Label(params, text="Language:").grid(row=0, column=2, sticky="w", **pad)
        ttk.Combobox(params, textvariable=self.language_var, values=LANGUAGES,
                    state="readonly", width=10).grid(row=0, column=3, sticky="w")

        ttk.Label(params, text="Output language:").grid(row=0, column=4, sticky="w", **pad)
        output_language_box = ttk.Combobox(params, textvariable=self.output_language_var, values=LANGUAGES,
                    state="readonly", width=10)
        output_language_box.grid(row=0, column=5, sticky="w")
        Tooltip(output_language_box,
                "Language of the generated notes/summary AND the VLM slide title/bullets, "
                "independent of \"Language\" above (which only affects transcription accuracy). "
                "\"auto\" (default): follows the transcript's/slide's own language, no translation. "
                "Set to e.g. \"de\" to always get German notes even from an English recording. "
                "The verbatim transcript file itself is never translated.")

        ttk.Label(params, text="Prompt template:").grid(row=1, column=0, sticky="w", **pad)
        self.prompt_box = ttk.Combobox(params, textvariable=self.prompt_var,
                                       state="readonly", width=24)
        self.prompt_box.grid(row=1, column=1, sticky="w", columnspan=2)
        ttk.Button(params, text="Reload list", command=self._refresh_prompt_templates).grid(
            row=1, column=3, sticky="w")
        ttk.Button(params, text="Go to prompt folder", command=self.app._open_prompts_folder).grid(
            row=1, column=4, sticky="w", padx=(4, 0))

        ttk.Label(params, text="LLM backend:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Combobox(params, textvariable=self.llm_backend_var, values=LLM_BACKENDS,
                    state="readonly", width=14).grid(row=2, column=1, sticky="w")

        if is_video:
            ttk.Label(params, text="Hash threshold:").grid(row=2, column=2, sticky="w", **pad)
            ttk.Spinbox(params, from_=0, to=64, textvariable=self.threshold_var, width=6).grid(
                row=2, column=3, sticky="w")

            ttk.Label(params, text="Animation threshold:").grid(row=3, column=0, sticky="w", **pad)
            ttk.Spinbox(params, from_=0, to=64, textvariable=self.animation_threshold_var, width=6).grid(
                row=3, column=1, sticky="w")

            ttk.Label(params, text="Slide fps:").grid(row=3, column=2, sticky="w", **pad)
            ttk.Spinbox(params, from_=1, to=10, textvariable=self.fps_var, width=6).grid(
                row=3, column=3, sticky="w")

            ttk.Label(params, text="Min. slide duration (sec):").grid(row=4, column=0, sticky="w", **pad)
            ttk.Spinbox(params, from_=0.0, to=30.0, increment=0.5, textvariable=self.min_slide_duration_var,
                       width=6).grid(row=4, column=1, sticky="w")
            ttk.Label(params, text="How long a slide must stay on screen to count as a real change "
                                  "(filters out animations/transitions). Default: 2.0.",
                     foreground="#666").grid(row=4, column=2, columnspan=3, sticky="w", **pad)

            ttk.Label(params, text="Recording speed:").grid(row=5, column=0, sticky="w", **pad)
            ttk.Spinbox(params, from_=0.1, to=10.0, increment=0.1, textvariable=self.recording_speed_var,
                       width=6).grid(row=5, column=1, sticky="w")
            ttk.Label(params, text="If the recording plays faster/slower than real time, set the factor here: "
                                  "every timestamp everywhere (transcript, .srt, slide report) is converted as "
                                  "real_time = video_time / speed. Default: 1.0 (no change).",
                     foreground="#666").grid(row=5, column=2, columnspan=3, sticky="w", **pad)

            ttk.Checkbutton(params, text="Also create a real-time-speed video copy",
                            variable=self.convert_video_to_realtime_var).grid(
                row=6, column=0, columnspan=2, sticky="w", **pad)
            ttk.Label(params, text="Re-encodes a <stem>_realtime.mp4 that plays at actual 1x speed, so it "
                                  "stays in sync with the already-converted .srt/transcript. Only runs when "
                                  "Recording speed above is not 1.0. Slow for long videos (full ffmpeg re-encode).",
                     foreground="#666").grid(row=6, column=2, columnspan=3, sticky="w", **pad)

            ttk.Label(params, text="0 = disabled (default). When set (must be lower than Hash threshold), a "
                                  "change below Hash threshold but at/above this value is treated as the "
                                  "current slide still building (e.g. bullets appearing one at a time): the "
                                  "slide's snapshot updates to the more complete frame, but its start "
                                  "timestamp stays the same, instead of creating a duplicate slide.",
                     foreground="#666").grid(row=7, column=0, columnspan=5, sticky="w", padx=8, pady=(0, 4))

        # --- Stages ---
        # Each checkbox gets a Tooltip (task #73) explaining what the
        # stage actually does on hover, keyed by STAGE_TOOLTIPS so the
        # explanation text lives in one place shared by both tabs.
        toggles = ttk.LabelFrame(step1_target, text="Stages")
        toggles.pack(fill="x", **pad)
        if is_video:
            cb = ttk.Checkbutton(toggles, text="Slide detection (video files only)",
                                 variable=self.enable_slides_var)
            cb.grid(row=0, column=0, sticky="w", **pad)
            Tooltip(cb, STAGE_TOOLTIPS["enable_slides"])

            cb = ttk.Checkbutton(toggles, text="VLM slide annotation",
                                 variable=self.enable_vlm_var)
            cb.grid(row=0, column=1, sticky="w", **pad)
            Tooltip(cb, STAGE_TOOLTIPS["enable_vlm"])

            cb = ttk.Checkbutton(toggles, text="Speaker diarization",
                                 variable=self.enable_diarization_var)
            cb.grid(row=0, column=2, sticky="w", **pad)
            Tooltip(cb, STAGE_TOOLTIPS["enable_diarization"])

            cb = ttk.Checkbutton(toggles, text="No summary (transcript only)",
                                 variable=self.no_summary_var)
            cb.grid(row=1, column=0, sticky="w", **pad)
            Tooltip(cb, STAGE_TOOLTIPS["no_summary"])

            cb = ttk.Checkbutton(toggles, text="Force re-transcribe / re-process",
                                 variable=self.force_retranscribe_var)
            cb.grid(row=1, column=1, sticky="w", **pad)
            Tooltip(cb, STAGE_TOOLTIPS["force_retranscribe"])

            cb = ttk.Checkbutton(toggles, text="Dry run (slide timestamps only)",
                                 variable=self.dry_run_var)
            cb.grid(row=1, column=2, sticky="w", **pad)
            Tooltip(cb, STAGE_TOOLTIPS["dry_run"])

            cb = ttk.Checkbutton(toggles, text="Improve audio before transcribing",
                                 variable=self.enhance_audio_var)
            cb.grid(row=2, column=0, sticky="w", **pad)
            Tooltip(cb, STAGE_TOOLTIPS["enhance_audio"])
        else:
            cb = ttk.Checkbutton(toggles, text="Speaker diarization",
                                 variable=self.enable_diarization_var)
            cb.grid(row=0, column=0, sticky="w", **pad)
            Tooltip(cb, STAGE_TOOLTIPS["enable_diarization"])

            cb = ttk.Checkbutton(toggles, text="No summary (transcript only)",
                                 variable=self.no_summary_var)
            cb.grid(row=0, column=1, sticky="w", **pad)
            Tooltip(cb, STAGE_TOOLTIPS["no_summary"])

            cb = ttk.Checkbutton(toggles, text="Force re-transcribe / re-process",
                                 variable=self.force_retranscribe_var)
            cb.grid(row=0, column=2, sticky="w", **pad)
            Tooltip(cb, STAGE_TOOLTIPS["force_retranscribe"])

            cb = ttk.Checkbutton(toggles, text="Improve audio before transcribing",
                                 variable=self.enhance_audio_var)
            cb.grid(row=1, column=0, sticky="w", **pad)
            Tooltip(cb, STAGE_TOOLTIPS["enhance_audio"])

        # --- Notes format ---
        notes_format_frame = ttk.LabelFrame(step3_target, text="Notes format")
        notes_format_frame.pack(fill="x", **pad)
        ttk.Checkbutton(notes_format_frame, text="txt", variable=self.notes_txt_var).grid(
            row=0, column=0, sticky="w", **pad)
        ttk.Checkbutton(notes_format_frame, text="html", variable=self.notes_html_var).grid(
            row=0, column=1, sticky="w", **pad)
        ttk.Checkbutton(notes_format_frame, text="pdf", variable=self.notes_pdf_var).grid(
            row=0, column=2, sticky="w", **pad)
        ttk.Checkbutton(notes_format_frame, text="docx", variable=self.notes_docx_var).grid(
            row=0, column=3, sticky="w", **pad)

        if is_video:
            # --- Slide report content ---
            # What's shown ON each slide page, grouped separately from
            # the file-format choices above. "Transcript mode" sits right
            # next to the "transcript" checkbox it modifies (previously
            # two rows apart with a "Report content:"/"Transcript:" label
            # pair that read like two unrelated settings) instead of
            # below it, since it only has any effect when that checkbox
            # is on.
            report_content_frame = ttk.LabelFrame(step3_target, text="Slide report content")
            report_content_frame.pack(fill="x", **pad)
            ttk.Checkbutton(report_content_frame, text="image",
                            variable=self.report_show_image_var).grid(row=0, column=0, sticky="w", **pad)
            ttk.Checkbutton(report_content_frame, text="bullet points",
                            variable=self.report_show_bullets_var).grid(row=0, column=1, sticky="w", **pad)
            ttk.Checkbutton(report_content_frame, text="transcript",
                            variable=self.report_show_transcript_var).grid(row=0, column=2, sticky="w", **pad)
            ttk.Label(report_content_frame, text="Transcript mode:").grid(
                row=0, column=3, sticky="w", padx=(16, 4), pady=4)
            ttk.Combobox(report_content_frame, textvariable=self.report_transcript_mode_var,
                        values=["full", "first_sentence"], state="readonly", width=16).grid(
                row=0, column=4, sticky="w", pady=4)
            ttk.Label(report_content_frame,
                     text="'full' = complete transcript segment per slide. 'first_sentence' = "
                          "first sentence only, so it fits on one page.",
                     foreground="#666", wraplength=860).grid(
                row=1, column=0, columnspan=5, sticky="w", padx=8, pady=(0, 4))

            # --- Slide report format ---
            # Export-format checkboxes on one row, the two "zip snapshots"
            # controls grouped together on their own row below (the
            # checkbox zips THIS run's snapshots as part of the run;
            # the button zips an already-finished run's snapshots folder
            # on demand - related actions, kept next to each other).
            slide_report_frame = ttk.LabelFrame(step3_target, text="Slide report format")
            slide_report_frame.pack(fill="x", **pad)
            ttk.Checkbutton(slide_report_frame, text="html", variable=self.report_html_var).grid(
                row=0, column=0, sticky="w", **pad)
            ttk.Checkbutton(slide_report_frame, text="csv", variable=self.report_csv_var).grid(
                row=0, column=1, sticky="w", **pad)
            ttk.Checkbutton(slide_report_frame, text="json", variable=self.report_json_var).grid(
                row=0, column=2, sticky="w", **pad)
            ttk.Checkbutton(slide_report_frame, text="pdf", variable=self.report_pdf_var).grid(
                row=0, column=3, sticky="w", **pad)
            ttk.Checkbutton(slide_report_frame, text="srt", variable=self.report_srt_var).grid(
                row=0, column=4, sticky="w", **pad)
            ttk.Checkbutton(slide_report_frame, text="timing summary (txt)",
                            variable=self.report_slide_timing_var).grid(
                row=0, column=5, sticky="w", **pad)

            ttk.Checkbutton(slide_report_frame, text="zip snapshots",
                            variable=self.zip_snapshots_var).grid(
                row=1, column=0, sticky="w", padx=8, pady=(0, 8))
            ttk.Button(slide_report_frame, text="Zip an existing snapshots folder...",
                      command=self._zip_existing_snapshots).grid(
                row=1, column=1, columnspan=5, sticky="w", padx=(4, 8), pady=(0, 8))

        # --- Step-specific tools (task #61/#112 GUI remodel) ---
        # On the Video/Webinar tab these four actions used to sit in the
        # run_frame row below; they move into the step they actually
        # belong to, so each tab is a one-stop place for that phase of
        # the workflow. The Meeting tab has no steps, so they stay in
        # run_frame exactly as before.
        if is_video:
            step1_tools = ttk.Frame(step1_target)
            step1_tools.pack(fill="x", padx=8, pady=(0, 8))
            ttk.Button(step1_tools, text="Open in LosslessCut...",
                      command=self._open_in_losslesscut).pack(side="left")
            ttk.Label(step1_tools, text="Trim the source video before processing it (optional).",
                     foreground="#666").pack(side="left", padx=(8, 0))

            step1_tools2 = ttk.Frame(step1_target)
            step1_tools2.pack(fill="x", padx=8, pady=(0, 8))
            ttk.Button(step1_tools2, text="Import .srt as transcript...",
                      command=self._on_import_srt).pack(side="left")
            ttk.Label(step1_tools2,
                     text="Already have captions (e.g. from YouTube)? Import the .srt instead of "
                          "re-transcribing - Run will then skip whisperx and use it directly.",
                     foreground="#666").pack(side="left", padx=(8, 0))

            self._step2_button_row = ttk.Frame(step2_target)
            self._step2_button_row.pack(fill="x", padx=8, pady=(0, 4))
            ttk.Button(self._step2_button_row, text="Review / Edit Slides...",
                      command=self._open_slide_review).pack(side="left")
            ttk.Button(self._step2_button_row, text="Re-annotate failed slides...",
                      command=self._on_reannotate_failed).pack(side="left", padx=(6, 0))
            ttk.Button(self._step2_button_row, text="Rename Speakers...",
                      command=self._on_rename_speakers).pack(side="left", padx=(6, 0))
            ttk.Label(step2_target,
                     text="Omit/merge slides and rebuild the slide report, mark where Q&A "
                          "starts above, or rename speakers - all without re-transcribing.",
                     foreground="#666", wraplength=860).pack(fill="x", padx=8, pady=(0, 8))

            ttk.Label(step3_target,
                     text="To (re)generate just the summary/reports for a recording already "
                          "processed in step 1 - e.g. after adjusting Q&A in step 2, or the "
                          "formats above - set Mode (top of this tab) to \"Notes-only "
                          "(existing transcript)\" and Run.",
                     foreground="#666", wraplength=860).pack(fill="x", padx=8, pady=(8, 4))

        # --- Run button + progress ---
        run_frame = ttk.Frame(parent)
        run_frame.pack(fill="x", **pad)
        self.run_button = ttk.Button(run_frame, text="Run", command=self._on_run)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(run_frame, text="Stop", command=self._on_stop, state="disabled")
        self.stop_button.pack(side="left", padx=(6, 0))
        if not is_video:
            ttk.Button(run_frame, text="Open in LosslessCut...",
                      command=self._open_in_losslesscut).pack(side="left", padx=(6, 0))
            ttk.Button(run_frame, text="Rename Speakers...",
                      command=self._on_rename_speakers).pack(side="left", padx=(6, 0))
            ttk.Button(run_frame, text="Import .srt as transcript...",
                      command=self._on_import_srt).pack(side="left", padx=(6, 0))
        self.status_label = ttk.Label(run_frame, text="Ready.")
        self.status_label.pack(side="left", padx=12)

        progress_frame = ttk.Frame(parent)
        progress_frame.pack(fill="x", **pad)
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", mode="determinate",
                                            maximum=100, variable=self.progress_var, length=320)
        self.progress_bar.pack(side="left")
        self.stage_label = ttk.Label(progress_frame, text="")
        self.stage_label.pack(side="left", padx=10)

        log_frame = ttk.LabelFrame(parent, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log_text = scrolledtext.ScrolledText(
            log_frame, state="disabled", height=16, wrap="word",
            bg=_theme_color("neutral_100"), fg=_theme_color("neutral_900"),
            insertbackground=_theme_color("neutral_900"),
            selectbackground=_theme_color("primary_300"))
        self.log_text.pack(fill="both", expand=True)

        # --- Persisted settings (task #71) ---
        # Maps each gui_logic.persisted_keys_for(self.kind) key to the
        # tkinter Variable holding it, so _apply_persisted_state/
        # _collect_tab_state/_collect_shared_state can read/write them
        # generically. Note some of these Variables (whisper_model,
        # language, output_language, llm_backend, enable_diarization,
        # no_summary, force_retranscribe) are the SAME objects as the
        # other Run tab's - see PipelineGUI.__init__.
        self._state_vars = {
            "recording_type":    self.recording_type_var,
            "prompt_template":   self.prompt_var,
            "whisper_model":     self.whisper_model_var,
            "language":          self.language_var,
            "output_language":   self.output_language_var,
            "llm_backend":       self.llm_backend_var,
            "output_dir":        self.output_dir_var,
            "use_date_subject_filename": self.use_date_subject_filename_var,
            "move_processed_files": self.move_processed_files_var,
            "enable_diarization": self.enable_diarization_var,
            "no_summary":        self.no_summary_var,
            "force_retranscribe": self.force_retranscribe_var,
            "enhance_audio":     self.enhance_audio_var,
        }
        if is_video:
            self._state_vars.update({
                "enable_slides":      self.enable_slides_var,
                "enable_vlm":         self.enable_vlm_var,
                "hash_threshold":     self.threshold_var,
                "animation_threshold": self.animation_threshold_var,
                "fps":                self.fps_var,
                "min_slide_duration_sec": self.min_slide_duration_var,
                "recording_speed":    self.recording_speed_var,
                "convert_video_to_realtime": self.convert_video_to_realtime_var,
                "report_html":        self.report_html_var,
                "report_csv":         self.report_csv_var,
                "report_json":        self.report_json_var,
                "report_srt":         self.report_srt_var,
                "report_pdf":         self.report_pdf_var,
                "report_slide_timing": self.report_slide_timing_var,
                "zip_snapshots":      self.zip_snapshots_var,
                "report_show_image":      self.report_show_image_var,
                "report_show_bullets":    self.report_show_bullets_var,
                "report_show_transcript": self.report_show_transcript_var,
                "report_transcript_mode": self.report_transcript_mode_var,
                "qa_include_in_summary":  self.qa_include_in_summary_var,
            "qa_autodetect_start":    self.qa_autodetect_var,
            })

        self._apply_recording_type()
        self._refresh_prompt_templates()
        self._apply_persisted_state()
        self._update_path_label()

    def _apply_persisted_state(self):
        """Task #71: restore this tab's last-used settings from
        gui_state.json, if any. Called after _apply_recording_type/
        _refresh_prompt_templates so a saved prompt_template correctly
        wins over the recording-type preset's default. Missing file,
        first run, or a corrupt gui_state.json are all silently treated
        as "nothing saved yet" (see gui_logic.load_state_file).

        Reads both this tab's own section AND the shared section (task:
        centralize Meeting/Video common settings) - applying the shared
        values here is idempotent even though both tabs call this, since
        their shared Variables are the same objects and state["shared"]
        is a single dict."""
        state = gui_logic.load_state_file(STATE_PATH)
        saved_tab = state.get(self.kind) or {}
        saved_shared = state.get("shared") or {}
        if not saved_tab and not saved_shared:
            return
        defaults = {k: v.get() for k, v in self._state_vars.items()}
        merged = gui_logic.merge_persisted_state(self.kind, saved_tab, saved_shared, defaults)
        for key, var in self._state_vars.items():
            if key == "recording_type" and merged[key] not in self._recording_types:
                continue
            try:
                var.set(merged[key])
            except Exception:
                pass

    def _collect_tab_state(self) -> dict:
        """Task #71: snapshot this tab's OWN settings for gui_state.json
        (recording_type/prompt_template/output_dir, plus video-only keys
        for the video tab) - excludes the shared keys, see
        _collect_shared_state. Never the path, batch table, or meeting
        info/Q&A fields."""
        raw = {k: v.get() for k, v in self._state_vars.items()}
        return gui_logic.build_tab_state_dict(self.kind, raw)

    def _collect_shared_state(self) -> dict:
        """Snapshot the settings shared with the other Run tab (whisper
        model, language, LLM backend, diarization, no-summary,
        force-retranscribe) for gui_state.json's "shared" key. Either
        tab's values are equally valid here since both are backed by the
        same Variable objects."""
        raw = {k: v.get() for k, v in self._state_vars.items()}
        return gui_logic.build_shared_state_dict(raw)

    def _label(self) -> str:
        return "Video/Webinar" if self.kind == "video" else "Meeting"

    def _apply_recording_type(self):
        preset = self._recording_types.get(self.recording_type_var.get())
        if not preset:
            return
        self.prompt_var.set(preset["prompt_template"])

    def _refresh_prompt_templates(self):
        templates = notes_mod.list_prompt_templates(PROMPTS_DIR)
        names = list(templates.keys())
        self.prompt_box["values"] = names
        if names and self.prompt_var.get() not in names:
            default = "meeting" if "meeting" in names else names[0]
            self.prompt_var.set(default)

    def _update_path_label(self):
        """
        Shows/hides the batch table vs the Meeting info/Q&A sections
        based on Mode: in batch mode, Meeting info/Q&A are per-row in the
        batch table already, so the shared single-file fields would be
        misleading left visible. Meeting info is always re-anchored via
        after=self._rec_frame (both live in step1_target - step1's tab
        for the Video/Webinar tab, "parent" directly for the Meeting
        tab), so toggling never reorders the rest of that tab/step. Q&A
        (Video/Webinar only, lives in step2_target) is re-anchored via
        before=self._step2_button_row instead, for the same reason.
        """
        mode = self.mode_var.get()
        labels = {
            MODES[0]: "File:",
            MODES[1]: "List file (.txt, optional if using the batch table below):",
            MODES[2]: "Folder:",
            MODES[3]: "Transcript file:",
            MODES[4]: "Folder:",
        }
        self.path_label.config(text=labels.get(mode, "File:"))

        if self._steps_nb is not None:
            self._steps_nb.select(
                self._step3_frame if mode in (MODES[3], MODES[4]) else self._step1_frame)

        is_batch = mode == MODES[1]
        self._meeting_frame.pack_forget()
        if self._qa_frame is not None:
            self._qa_frame.pack_forget()
        self.batch_frame.pack_forget()

        # Title slide image (if this tab has one - is_video only) is
        # deliberately NOT pack_forget()-ed above: it has no per-row batch
        # equivalent (unlike meeting_frame's fields), so it must stay
        # visible/enterable in every Mode, batch included. Re-packing an
        # already-packed widget with the same options is a no-op, so this
        # runs safely on every call, not just the first.
        if self._title_slide_frame is not None:
            self._title_slide_frame.pack(fill="x", padx=8, pady=4, after=self._rec_frame)
        meeting_anchor = self._title_slide_frame if self._title_slide_frame is not None else self._rec_frame

        if is_batch:
            self.batch_frame.pack(fill="both", expand=False, padx=8, pady=4, after=meeting_anchor)
        else:
            self._meeting_frame.pack(fill="x", padx=8, pady=4, after=meeting_anchor)
            if self._qa_frame is not None:
                # qa_frame lives in step2_target now (a different Notebook
                # tab than meeting_frame's step1_target), so it can't be
                # anchored to meeting_frame any more - anchor it to the
                # permanent button row within its own tab instead, so it
                # always reinserts above those buttons rather than at
                # whatever position pack_forget()/pack() happens to leave it.
                self._qa_frame.pack(fill="x", padx=8, pady=4, before=self._step2_button_row)

    def _load_meeting_info(self):
        """Task #93: pre-fill Title/Date/Comments from an .ics calendar
        invite, or a .txt/.docx file using the "Title:"/"Date:"/
        "Comments:" labeled-fields format (see ics_utils.parse_labeled_text
        for the exact rules, e.g. Comments can span multiple lines).
        Existing Comments text is appended to (not overwritten by) a
        newly-loaded comments block, since the user may already have
        typed something here; Title/Date are overwritten, matching the
        previous .ics-only behavior."""
        path = filedialog.askopenfilename(
            title="Select .ics / .txt / .docx meeting info file",
            filetypes=[
                ("Meeting info files", "*.ics *.txt *.docx"),
                ("Calendar invite", "*.ics"),
                ("Text file", "*.txt"),
                ("Word document", "*.docx"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            parsed = ics_utils.parse_meeting_info_file(path)
        except Exception as exc:
            messagebox.showerror("Could not read meeting info file", str(exc))
            return
        if parsed.get("title"):
            self.meeting_title_var.set(parsed["title"])
        if parsed.get("date"):
            self.meeting_date_var.set(parsed["date"])
        if parsed.get("comments"):
            existing = self.meeting_comments_var.get().strip()
            combined = f"{existing} {parsed['comments']}" if existing else parsed["comments"]
            self.meeting_comments_var.set(combined)
        # Task #95: Agenda (if present) is folded into Comments too, since
        # there is no separate Agenda entry field in this dialog - it still
        # reaches the notes header/LLM context via meeting_comments.
        if parsed.get("agenda"):
            existing = self.meeting_comments_var.get().strip()
            agenda_line = f"Agenda: {parsed['agenda']}"
            combined = f"{existing}\n{agenda_line}" if existing else agenda_line
            self.meeting_comments_var.set(combined)
        attendees = parsed.get("attendees") or []
        if attendees:
            self.meeting_attendees = attendees
            names = ics_utils.attendee_names(attendees)
            preview = ", ".join(names[:5]) + (", ..." if len(names) > 5 else "")
            self.meeting_attendees_status_var.set(
                f"Loaded {len(names)} invitee(s): {preview} - offered first when naming "
                f"speakers in Rename Speakers.")
        else:
            self.meeting_attendees_status_var.set("")
        if not any(parsed.get(k) for k in ("title", "date", "comments", "agenda")) and not attendees:
            messagebox.showinfo("Nothing found",
                                "No title, date, comments, agenda, or invitees found in this file.")

    def _get_invitee_names(self) -> list:
        """Task #95: the display names from the currently-loaded meeting
        info's attendees (an .ics invite or a "Invitees:" line loaded via
        _load_meeting_info for this tab), for use as a speaker-name
        suggestion/autocomplete pick-list in Rename Speakers - offered
        ahead of a blind guess, since these are the real people invited to
        this recording. Empty list if no meeting info with attendees has
        been loaded for this tab."""
        return ics_utils.attendee_names(self.meeting_attendees)

    def _browse_title_slide_image(self):
        path = filedialog.askopenfilename(
            title="Select title slide cover image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
        )
        if path:
            self.title_slide_image_var.set(path)

    def _browse(self):
        mode = self.mode_var.get()
        if mode == MODES[0]:
            patterns = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS))
            path = filedialog.askopenfilename(
                title="Select audio or video file",
                filetypes=[("Audio/Video", patterns), ("All files", "*.*")],
            )
        elif mode == MODES[1]:
            path = filedialog.askopenfilename(title="Select file list (.txt)",
                                              filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        elif mode in (MODES[2], MODES[4]):
            path = filedialog.askdirectory(title="Select folder")
        else:
            path = filedialog.askopenfilename(title="Select transcript file",
                                              filetypes=[("Transcript", "*_transcript_speakers.txt"),
                                                         ("All files", "*.*")])
        if path:
            self.path_var.set(path)

    def _browse_output_dir(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_dir_var.set(path)

    def _find_losslesscut(self) -> str:
        """Return a usable LosslessCut.exe path, or "" if none can be
        found. Checks (in order): a path already browsed for this
        session, config.py's losslesscut_path, PATH, and the standard
        Windows per-user install location for the LosslessCut installer."""
        for candidate in (
            self.losslesscut_path_var.get().strip(),
            CONFIG.get("losslesscut_path", "").strip(),
        ):
            if candidate and Path(candidate).exists():
                return candidate
        found = shutil.which("LosslessCut")
        if found:
            return found
        default = Path.home() / "AppData" / "Local" / "Programs" / "losslesscut-win" / "LosslessCut.exe"
        if default.exists():
            return str(default)
        return ""

    def _open_in_losslesscut(self):
        """Launch LosslessCut (free, MIT-licensed lossless cut/mute editor,
        https://github.com/mifi/lossless-cut) with the currently selected
        source file preloaded, for manual video editing. Does not touch
        the pipeline in any way - purely a convenience launcher."""
        path = self.path_var.get().strip()
        if not path or not Path(path).exists():
            messagebox.showwarning("No file selected", "Select a source file above first.")
            return
        exe = self._find_losslesscut()
        if not exe:
            messagebox.showinfo(
                "LosslessCut not found",
                "Could not auto-detect LosslessCut.exe. Free download:\n"
                "https://github.com/mifi/lossless-cut/releases\n\n"
                "Select LosslessCut.exe on the next screen.")
            exe = filedialog.askopenfilename(
                title="Select LosslessCut.exe",
                filetypes=[("LosslessCut", "LosslessCut.exe"), ("All files", "*.*")])
            if not exe:
                return
            self.losslesscut_path_var.set(exe)
        try:
            subprocess.Popen([exe, path])
        except Exception as exc:
            messagebox.showerror("Could not launch LosslessCut", str(exc))

    def _on_rename_speakers(self):
        """Open the post-hoc speaker renaming dialog (task #76): pick an
        existing *_transcript_speakers.txt, detect its speaker labels from
        the matching *_segments.json cache, and let the user map each
        label to a real name. See run_pipeline.rename_speakers/
        get_speaker_labels. Available on both tabs - diarization can be
        used for a plain meeting recording just as much as a video."""
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Busy", "A run is already in progress.")
            return
        transcript_path = filedialog.askopenfilename(
            title="Select *_transcript_speakers.txt",
            filetypes=[("Transcript", "*_transcript_speakers.txt"), ("All files", "*.*")],
        )
        if not transcript_path:
            return
        segments_json_path = str(transcript_path).replace("_transcript_speakers.txt", "_segments.json")
        labels = run_pipeline.get_speaker_labels(segments_json_path)
        if not labels:
            messagebox.showwarning(
                "No speaker labels found",
                f"No speaker labels found for:\n{transcript_path}\n\n"
                f"Expected segment cache: {Path(segments_json_path).name}\n\n"
                "Either diarization was not enabled for this run, or the segment "
                "cache is missing (needed for a rename; the already-formatted "
                ".txt file alone is not enough).")
            return
        source_media_path = run_pipeline.find_source_media(segments_json_path)
        SpeakerRenameDialog(self, segments_json_path, labels, source_media_path,
                           invitee_names=self._get_invitee_names())

    def _run_rename_speakers_worker(self, segments_json_path: str, mapping: dict, regenerate_notes: bool,
                                    speaker_suggestions: dict | None = None, db_path: str | None = None):
        handler = QueueLogHandler(self.log_queue)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        level = getattr(logging, self.app.log_level_var.get().upper(), logging.INFO)
        root_logger.setLevel(level)
        try:
            overrides = self._build_overrides()
            if speaker_suggestions:
                try:
                    commit_result = run_pipeline.commit_speaker_identities(
                        db_path or overrides.get("db_path") or CONFIG["db_path"],
                        speaker_suggestions, mapping)
                    if commit_result:
                        for label, info in commit_result.items():
                            self.log_queue.put(
                                f"Speaker roster: {label} -> {info['action']} "
                                f"(known_speakers id {info['speaker_id']})")
                    else:
                        self.log_queue.put(
                            "WARNING: known-speaker roster not updated - none of the "
                            "renamed labels had a computed voiceprint (embedding "
                            "extraction may have failed for those speakers).")
                except Exception as exc:
                    self.log_queue.put(f"WARNING: could not update known-speaker roster: {exc}")
            else:
                self.log_queue.put(
                    "Speaker roster: not updated (no voiceprint suggestions were "
                    "available for this run).")
            result = run_pipeline.rename_speakers(
                segments_json_path, mapping, overrides, regenerate_notes=regenerate_notes)
            self.log_queue.put(f"=== SPEAKER RENAME DONE: {result['renamed_segments']} segment(s) "
                               f"updated ===")
            if regenerate_notes:
                if result.get("notes_regenerated"):
                    self.log_queue.put("=== NOTES REGENERATED ===")
                else:
                    self.log_queue.put("=== NOTES NOT REGENERATED (see log above) ===")
        except Exception as exc:
            self.log_queue.put(f"FATAL ERROR: {exc}")
            import traceback
            self.log_queue.put(traceback.format_exc())
        finally:
            root_logger.removeHandler(handler)
            self.log_queue.put("__RUN_COMPLETE__")

    def _open_slide_review(self):
        """Open the slide review/reprocess window (task #58, video tab
        only): mark detected slides to omit or merge, then rebuild the
        report outputs without re-running transcription/detection/VLM."""
        SlideReviewDialog(self.root, self)

    def _zip_existing_snapshots(self):
        """Zip an already-existing snapshots/ folder on demand (e.g. from
        a run made before the "zip snapshots" checkbox existed), without
        re-running the pipeline."""
        folder = filedialog.askdirectory(title="Select a snapshots folder to zip")
        if not folder:
            return
        folder_path = Path(folder)
        default_name = folder_path.parent.name or "snapshots"
        zip_path = filedialog.asksaveasfilename(
            title="Save zip as", defaultextension=".zip",
            initialfile=f"{default_name}.zip", filetypes=[("Zip archive", "*.zip")])
        if not zip_path:
            return
        ok = run_pipeline.zip_snapshot_dir(folder_path, Path(zip_path))
        if ok:
            messagebox.showinfo("Done", f"Snapshots zipped:\n{zip_path}")
        else:
            messagebox.showwarning("Nothing to zip", f"No files found directly in:\n{folder}")

    # -----------------------------------------------------------------
    # Batch table (task #62)
    # -----------------------------------------------------------------

    def _batch_insert_row(self, **kwargs):
        values = tuple(kwargs.get(col, "") for col in self.batch_columns)
        self.batch_tree.insert("", "end", values=values)

    def _batch_on_double_click(self, event):
        """Inline cell editor: tkinter's Treeview has no built-in editable
        cell, so a temporary Entry is placed exactly over the clicked
        cell's bbox, pre-filled with its current value, and its result is
        written back into the row on Enter/focus-out. "title_image" is the
        one exception (see below): a file path is better picked than
        typed/pasted, so that column opens a file dialog instead."""
        region = self.batch_tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        row_iid = self.batch_tree.identify_row(event.y)
        col = self.batch_tree.identify_column(event.x)
        if not row_iid or not col:
            return
        col_index = int(col.replace("#", "")) - 1
        if col_index >= len(self.batch_columns):
            return
        col_name = self.batch_columns[col_index]

        if col_name == "title_image":
            # Matches the shared field's own "Browse..." button
            # (_browse_title_slide_image) rather than the plain-text
            # inline editor every other column uses - a per-webinar cover
            # image is a file to pick, not a value to type. Leaves the
            # existing cell value untouched if the dialog is cancelled.
            path = filedialog.askopenfilename(
                title="Select title slide cover image",
                filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
            )
            if path:
                # os.path.normpath (task #97): tkinter's file dialogs
                # return forward-slash paths on Windows; normalized here
                # to match the "file" column's own existing convention.
                self.batch_tree.set(row_iid, col_name, os.path.normpath(path))
            return

        if col_name == "srt_path":
            # Same file-picker treatment as title_image (task: batch
            # YouTube import) - a per-row .srt is a file to pick, not a
            # value to type. A blank cell (dialog cancelled, or never
            # set) means this row gets a real transcription like any
            # other - see gui_logic.build_batch_row's srt_import_path key.
            path = filedialog.askopenfilename(
                title="Select .srt subtitle file to import for this row",
                filetypes=[("SRT subtitles", "*.srt"), ("All files", "*.*")],
            )
            if path:
                self.batch_tree.set(row_iid, col_name, os.path.normpath(path))
            return

        bbox = self.batch_tree.bbox(row_iid, col)
        if not bbox:
            return
        x, y, w, h = bbox
        current_value = self.batch_tree.set(row_iid, col_name)

        edit_var = tk.StringVar(value=current_value)
        entry = ttk.Entry(self.batch_tree, textvariable=edit_var)
        entry.place(x=x, y=y, width=w, height=h)
        entry.focus_set()
        entry.select_range(0, "end")

        def save_edit(event=None):
            if entry.winfo_exists():
                self.batch_tree.set(row_iid, col_name, edit_var.get())
                entry.destroy()

        def cancel_edit(event=None):
            if entry.winfo_exists():
                entry.destroy()

        entry.bind("<Return>", save_edit)
        entry.bind("<FocusOut>", save_edit)
        entry.bind("<Escape>", cancel_edit)

    def _batch_rows(self) -> list:
        """Read the batch table into row dicts matching
        run_pipeline.run_batch_rows's expected keys. Blank optional
        fields become None (row falls back to the shared fields above).
        Meeting-tab tables have no qa_start/qa_end columns at all.
        The actual transform is gui_logic.build_batch_row (pure,
        unit-tested); this just walks the Treeview rows."""
        return self._batch_rows_with_iids()[0]

    def _batch_rows_with_iids(self) -> tuple:
        """Like _batch_rows, but also returns the matching Treeview iid for
        each row, in lockstep (same index a row lands at in the returned
        list is the same index its iid lands at). A row with a blank file
        cell is skipped in both lists together, so index i in the rows
        list passed to run_pipeline.run_batch_rows always corresponds to
        iid i in the iids list here - used by the live batch status
        column (task #74) to map a "row N" progress update back to the
        right Treeview row without depending on the row's original
        on-screen position."""
        rows, iids = [], []
        for iid in self.batch_tree.get_children():
            row = gui_logic.build_batch_row(self.batch_columns, self.batch_tree.item(iid, "values"))
            if row is not None:
                rows.append(row)
                iids.append(iid)
        return rows, iids

    def _batch_add_files(self):
        # os.path.normpath (task #97): tkinter's file dialogs return paths
        # with forward slashes on Windows (a Tk/Tcl quirk, not a pathlib
        # one) unless normalized - without this, a file added here would
        # sit in the batch table as "X:/Agilent/.../file.mp4" instead of
        # "X:\Agilent\...\file.mp4", inconsistent with every other path in
        # the table.
        patterns = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS))
        paths = filedialog.askopenfilenames(
            title="Select audio/video files to add",
            filetypes=[("Audio/Video", patterns), ("All files", "*.*")],
        )
        for p in paths:
            self._batch_insert_row(file=os.path.normpath(p))

    def _batch_add_folder(self):
        folder = filedialog.askdirectory(
            title="Select folder (adds every supported file found directly inside)")
        if not folder:
            return
        folder_path = Path(folder)
        files = sorted(p for ext in SUPPORTED_EXTENSIONS for p in folder_path.glob(f"*{ext}"))
        if not files:
            messagebox.showinfo("Nothing found", "No supported audio/video files found in this folder.")
            return
        for f in files:
            self._batch_insert_row(file=os.path.normpath(str(f)))

    def _batch_correlate_calendar(self):
        """Task #96: GUI front-end for correlate_calendar_recordings.py -
        matches every recording in a chosen folder against a .ics
        calendar export's events (by comparing the recording's own
        end-of-file timestamp to each event's end time; see that
        module's docstring for the full matching philosophy) and queues
        every recording here with Title/Date/Comments pre-filled from
        its matched event, exactly like importing that script's
        "..._batch.csv" output via "Load .txt/.csv...", but without
        leaving the GUI or a command line. Unmatched recordings are
        still queued, just with those fields blank. Also writes the two
        CSV reports next to the folder (same as the CLI script) so
        there is still a record to review afterward."""
        folder = filedialog.askdirectory(
            title="Select folder with recordings (and the .ics calendar export)")
        if not folder:
            return
        folder_path = Path(folder)

        ics_path = correlate_calendar_recordings.find_ics_file(folder_path)
        if ics_path is None:
            picked = filedialog.askopenfilename(
                title="Select the .ics calendar file",
                initialdir=folder, filetypes=[("Calendar", "*.ics"), ("All files", "*.*")])
            if not picked:
                return
            ics_path = Path(picked)

        recursive = messagebox.askyesno(
            "Include subfolders?",
            "Also scan subfolders of this folder for recordings?", default="no")

        tolerance = simpledialog.askfloat(
            "Match tolerance",
            "Max minutes between a recording's end time and a calendar event's end "
            "time to still count as a match:",
            initialvalue=correlate_calendar_recordings.DEFAULT_TOLERANCE_MIN, minvalue=0.0,
            parent=self.root)
        if tolerance is None:
            return

        try:
            events = ics_utils.parse_ics_events(str(ics_path))
            recordings = correlate_calendar_recordings.find_recordings(folder_path, recursive=recursive)
            if not recordings:
                messagebox.showinfo(
                    "Nothing found",
                    f"No supported audio/video files found in:\n{folder_path}")
                return
            rows = correlate_calendar_recordings.correlate(events, recordings, tolerance_min=tolerance)

            report_path = folder_path / "calendar_recording_correlation.csv"
            correlate_calendar_recordings.write_report_csv(rows, report_path)
            batch_path = report_path.with_name(report_path.stem + "_batch.csv")
            correlate_calendar_recordings.write_batch_csv(rows, batch_path)
        except Exception as exc:
            messagebox.showerror("Calendar correlation failed", str(exc))
            return

        for r in rows:
            comments_parts = []
            if r.get("description"):
                comments_parts.append(r["description"])
            if r.get("agenda"):
                comments_parts.append(f"Agenda: {r['agenda']}")
            if r.get("attendees"):
                comments_parts.append(f"Invitees: {ics_utils.format_attendees(r['attendees'])}")
            self._batch_insert_row(
                file=os.path.normpath(str(r["recording"])),
                title=r["matched_title"] or "",
                date=r["event_start"].strftime("%Y-%m-%d") if r["event_start"] else "",
                comments="\n".join(comments_parts),
            )

        matched = sum(1 for r in rows if r["status"] == "matched")
        messagebox.showinfo(
            "Calendar correlation done",
            f"{len(events)} calendar event(s), {len(recordings)} recording(s): "
            f"{matched} matched within {tolerance:g} min, {len(rows) - matched} not matched.\n\n"
            f"Queued {len(rows)} row(s) into the batch table below.\n\n"
            f"Reports written to:\n{report_path}\n{batch_path}")

    def _batch_remove_selected(self):
        for iid in self.batch_tree.selection():
            self.batch_tree.delete(iid)

    def _batch_clear(self):
        self.batch_tree.delete(*self.batch_tree.get_children())

    def _batch_load_file(self):
        path = filedialog.askopenfilename(
            title="Load batch list (.txt or .csv)",
            filetypes=[("Text/CSV", "*.txt *.csv"), ("All files", "*.*")],
        )
        if path:
            self._batch_load_path(path)

    def _batch_load_path(self, path: str):
        """Load rows from a .txt (legacy "path|lang" per line) or .csv
        file into the batch table. A .csv with a recognized header (any
        column name matching this tab's batch_columns) fills the
        matching columns; a .csv or .txt without one is treated as plain
        file[, language] pairs, same as the old paste box accepted.
        Header detection and line parsing are gui_logic.csv_header_index/
        parse_list_line (pure, unit-tested)."""
        known_cols = set(self.batch_columns)
        rows = []
        try:
            if path.lower().endswith(".csv"):
                with open(path, newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    first = next(reader, None)
                    if first is None:
                        return
                    idx = gui_logic.csv_header_index(first, known_cols)
                    if idx:
                        def get(row, name):
                            i = idx.get(name)
                            return row[i].strip() if i is not None and i < len(row) else ""

                        for row in reader:
                            if not row or not row[0].strip() or row[0].strip().startswith("#"):
                                continue
                            rows.append({c: get(row, c) for c in self.batch_columns})
                    else:
                        for row in [first] + list(reader):
                            if not row or not row[0].strip() or row[0].strip().startswith("#"):
                                continue
                            file_ = row[0].strip()
                            lang = row[1].strip() if len(row) > 1 else ""
                            rows.append({"file": file_, "language": lang})
            else:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        parsed = gui_logic.parse_list_line(line)
                        if parsed:
                            rows.append(parsed)
        except Exception as exc:
            messagebox.showerror("Could not load file", str(exc))
            return
        for r in rows:
            self._batch_insert_row(**r)

    def _batch_save_csv(self):
        path = filedialog.asksaveasfilename(
            title="Save batch table", defaultextension=".csv",
            filetypes=[("CSV file", "*.csv")],
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(list(self.batch_columns))
                for iid in self.batch_tree.get_children():
                    # Trim off the display-only "status" column (task #74):
                    # the exported CSV must stay in the plain data schema
                    # _batch_load_path already knows how to re-import.
                    writer.writerow(self.batch_tree.item(iid, "values")[:len(self.batch_columns)])
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        messagebox.showinfo("Saved", f"Batch table saved:\n{path}")

    # -----------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------

    def _parse_qa_time(self, text: str) -> float | None:
        """Best-effort wrapper around _parse_time_to_seconds for
        _build_overrides: never raises, so a stray typo here can't crash
        a run started from elsewhere (e.g. Re-annotate failed slides,
        which also calls _build_overrides but has no reason to validate
        Q&A fields). _on_run() does the user-facing validation instead."""
        try:
            return _parse_time_to_seconds(text)
        except ValueError:
            return None

    def _build_overrides(self) -> dict:
        overrides = {
            "prompt_template":    self.prompt_var.get() or None,
            "whisper_model":      self.whisper_model_var.get(),
            "whisper_language":   self.language_var.get(),
            "output_language":    self.output_language_var.get(),
            "llm_backend":        self.llm_backend_var.get(),
            "enable_diarization": self.enable_diarization_var.get(),
            "no_summary":         self.no_summary_var.get(),
            "force_retranscribe": self.force_retranscribe_var.get(),
            "enhance_audio":      self.enhance_audio_var.get(),
            "output_dir_override": self.output_dir_var.get().strip(),
            "notes_format_txt":   self.notes_txt_var.get(),
            "notes_format_html":  self.notes_html_var.get(),
            "notes_format_pdf":   self.notes_pdf_var.get(),
            "notes_format_docx":  self.notes_docx_var.get(),
            "meeting_title":      self.meeting_title_var.get().strip(),
            "meeting_date":       self.meeting_date_var.get().strip(),
            "meeting_comments":   self.meeting_comments_var.get().strip(),
            "output_basename_override": self.output_basename_var.get().strip(),
            "use_date_subject_filename": self.use_date_subject_filename_var.get(),
            "move_processed_files": self.move_processed_files_var.get(),
            "db_path":            self.app.db_path_var.get().strip() or None,
        }
        if self.kind == "video":
            overrides.update({
                "enable_slides":      self.enable_slides_var.get(),
                "enable_vlm":         self.enable_vlm_var.get(),
                "dry_run":            self.dry_run_var.get(),
                "hash_threshold":     self.threshold_var.get(),
                "animation_threshold": self.animation_threshold_var.get(),
                "fps":                self.fps_var.get(),
                "min_slide_duration_sec": self.min_slide_duration_var.get(),
                "qa_start_time_sec":   self._parse_qa_time(self.qa_start_var.get()),
                "qa_end_time_sec":     self._parse_qa_time(self.qa_end_var.get()),
                "qa_include_in_summary": self.qa_include_in_summary_var.get(),
                # Ticked -> None: leave CONFIG's rule in charge, which runs
                # detection for the webinar prompt template only. Unticked ->
                # False: never detect. Passing True here would widen it to
                # every recording type on this tab, including plain video
                # transcripts that have no Q&A block.
                "qa_autodetect_start": (None if self.qa_autodetect_var.get() else False),
                "recording_speed":     self.recording_speed_var.get(),
                "convert_video_to_realtime": self.convert_video_to_realtime_var.get(),
                "title_slide_image_path": self.title_slide_image_var.get().strip(),
                "report_html":        self.report_html_var.get(),
                "report_csv":         self.report_csv_var.get(),
                "report_json":        self.report_json_var.get(),
                "report_srt":         self.report_srt_var.get(),
                "report_pdf":         self.report_pdf_var.get(),
                "report_slide_timing": self.report_slide_timing_var.get(),
                "zip_snapshots":       self.zip_snapshots_var.get(),
                "report_show_image":      self.report_show_image_var.get(),
                "report_show_bullets":    self.report_show_bullets_var.get(),
                "report_show_transcript": self.report_show_transcript_var.get(),
                "report_transcript_mode": self.report_transcript_mode_var.get(),
            })
        else:
            # Meeting tab never runs slide detection/VLM: there is no
            # slide UI here to configure it, regardless of config.py's
            # own default.
            overrides["enable_slides"] = False
            overrides["enable_vlm"] = False
        return overrides

    def _on_run(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Busy", "A run is already in progress.")
            return

        mode = self.mode_var.get()
        path = self.path_var.get().strip()
        batch_rows = None

        self._batch_run_iids = []
        if mode == MODES[1]:
            batch_rows, batch_iids = self._batch_rows_with_iids()
            if not batch_rows and not path:
                messagebox.showwarning(
                    "Missing input",
                    "Add files to the batch table above, or choose an existing list.txt.")
                return
            if batch_rows:
                # Reset the Status column for this run (task #74): a
                # re-run after edits/removals should not show stale
                # "done"/"failed" labels from a previous run.
                self._batch_run_iids = batch_iids
                for iid in batch_iids:
                    self.batch_tree.set(iid, "status", "")
            if batch_rows and self.kind == "video":
                for iid in self.batch_tree.get_children():
                    vals = dict(zip(self.batch_columns, self.batch_tree.item(iid, "values")))
                    file_label = vals.get("file") or "(blank)"
                    for label, key in (("Q&A start", "qa_start"), ("Q&A end", "qa_end")):
                        text = vals.get(key, "")
                        if text:
                            try:
                                _parse_time_to_seconds(text)
                            except ValueError:
                                messagebox.showwarning(
                                    "Invalid Q&A time in batch table",
                                    f"Row '{file_label}': {label} value '{text}' is not valid. "
                                    f"Use seconds (975), mm:ss (16:15), or hh:mm:ss.")
                                return
        elif not path:
            messagebox.showwarning("Missing input", "Please choose a file/folder first.")
            return

        if not self.prompt_var.get():
            messagebox.showwarning("Missing prompt template",
                                   "No prompt template selected. Add a .md file to prompts/ "
                                   "and click 'Reload list'.")
            return

        if self.kind == "video":
            for label, var in (("Q&A 'Starts at'", self.qa_start_var), ("Q&A 'Ends at'", self.qa_end_var)):
                text = var.get().strip()
                if text:
                    try:
                        _parse_time_to_seconds(text)
                    except ValueError:
                        messagebox.showwarning(
                            "Invalid Q&A time",
                            f"{label} value '{text}' is not valid. Use seconds (975), "
                            f"mm:ss (16:15), or hh:mm:ss.")
                        return

        self._clear_log()
        self.run_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.stop_event.clear()
        self.status_label.config(text="Running...")
        self.progress_var.set(0)
        self.stage_label.config(text="")

        overrides = self._build_overrides()

        self.worker_thread = threading.Thread(
            target=self._run_worker, args=(mode, path, overrides, batch_rows), daemon=True
        )
        self.worker_thread.start()

    def _on_stop(self):
        """Request a cooperative stop (see run_pipeline.process_file's
        stop_check parameter). The current pipeline stage or VLM call in
        progress always finishes first; the run then halts at the next
        safe checkpoint instead of terminating mid-write."""
        self.stop_event.set()
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Stopping (finishing current step)...")

    def _on_import_srt(self):
        """Import an existing .srt (e.g. downloaded alongside a YouTube
        video, together with its description.txt) as the transcript for
        the file selected above, instead of running whisperx. See
        run_pipeline.import_srt_transcript: writes the same cache files a
        real transcription run would, so a later Run (Notes-only, or a
        full Video/Webinar run with slides) picks it up via process_file's
        existing cache-aware shortcut and skips re-transcription entirely.
        Refuses to overwrite an existing transcript cache unless "Force
        re-transcribe" (shared with the CLI's --force-retranscribe) is
        checked above."""
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Busy", "A run is already in progress.")
            return
        video_path = self.path_var.get().strip()
        if not video_path or not Path(video_path).exists():
            messagebox.showwarning("No file selected", "Select a source file above first.")
            return
        srt_path = filedialog.askopenfilename(
            title="Select .srt subtitle file to import",
            filetypes=[("SRT subtitles", "*.srt"), ("All files", "*.*")],
        )
        if not srt_path:
            return

        self._clear_log()
        self.run_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.stop_event.clear()
        self.status_label.config(text="Importing .srt as transcript...")
        self.progress_var.set(0)
        self.stage_label.config(text="")

        overrides = self._build_overrides()

        self.worker_thread = threading.Thread(
            target=self._import_srt_worker, args=(video_path, srt_path, overrides), daemon=True
        )
        self.worker_thread.start()

    def _import_srt_worker(self, video_path: str, srt_path: str, overrides: dict):
        handler = QueueLogHandler(self.log_queue)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        level = getattr(logging, self.app.log_level_var.get().upper(), logging.INFO)
        root_logger.setLevel(level)
        try:
            ok = run_pipeline.import_srt_transcript(video_path, srt_path, overrides)
            self.log_queue.put(
                "=== IMPORT .SRT DONE ===" if ok else "=== IMPORT .SRT FAILED (see log above) ===")
        except Exception as exc:
            self.log_queue.put(f"FATAL ERROR: {exc}")
            import traceback
            self.log_queue.put(traceback.format_exc())
        finally:
            root_logger.removeHandler(handler)
            self.log_queue.put("__RUN_COMPLETE__")

    def _on_reannotate_failed(self):
        """Pick an existing *_slides.json and re-run VLM annotation only for
        the slides that failed to parse (empty title/bullets), reusing the
        snapshots already on disk. See run_pipeline.reannotate_failed_slides.
        Video tab only."""
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Busy", "A run is already in progress.")
            return
        path = filedialog.askopenfilename(
            title="Select *_slides.json",
            filetypes=[("Slides JSON", "*_slides.json"), ("All files", "*.*")],
        )
        if not path:
            return

        self._clear_log()
        self.run_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.stop_event.clear()
        self.status_label.config(text="Re-annotating failed slides...")
        self.progress_var.set(0)
        self.stage_label.config(text="")

        overrides = self._build_overrides()

        self.worker_thread = threading.Thread(
            target=self._reannotate_worker, args=(path, overrides), daemon=True
        )
        self.worker_thread.start()

    def _reannotate_worker(self, slides_json_path: str, overrides: dict):
        handler = QueueLogHandler(self.log_queue)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        level = getattr(logging, self.app.log_level_var.get().upper(), logging.INFO)
        root_logger.setLevel(level)
        try:
            retried, still_failed = run_pipeline.reannotate_failed_slides(
                slides_json_path, overrides, stop_check=self.stop_event.is_set)
            self.log_queue.put(f"=== RE-ANNOTATE DONE: {retried} fixed, {still_failed} still failed ===")
        except Exception as exc:
            self.log_queue.put(f"FATAL ERROR: {exc}")
            import traceback
            self.log_queue.put(traceback.format_exc())
        finally:
            root_logger.removeHandler(handler)
            self.log_queue.put("__RUN_COMPLETE__")

    def _run_worker(self, mode: str, path: str, overrides: dict, batch_rows: list | None = None):
        handler = QueueLogHandler(self.log_queue)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        level = getattr(logging, self.app.log_level_var.get().upper(), logging.INFO)
        root_logger.setLevel(level)

        file_handler = None
        if self.app.log_to_file_var.get():
            try:
                log_dir = Path(self.app.log_dir_var.get().strip() or CONFIG.get("log_dir", "logs"))
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
                file_handler = logging.FileHandler(log_path, encoding="utf-8")
                file_handler.setFormatter(
                    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))
                root_logger.addHandler(file_handler)
                self.log_queue.put(f"Logging to file: {log_path}")
            except Exception as exc:
                self.log_queue.put(f"Could not set up file logging: {exc}")

        try:
            if mode == MODES[0]:
                ok = run_pipeline.process_file(path, overrides, stop_check=self.stop_event.is_set)
                self.log_queue.put("=== DONE (success) ===" if ok else "=== DONE (failed/stopped) ===")
            elif mode == MODES[1]:
                if batch_rows:
                    run_pipeline.run_batch_rows(
                        batch_rows, overrides, stop_check=self.stop_event.is_set,
                        progress_callback=lambda i, status: self.log_queue.put(
                            f"__BATCH_STATUS__:{i}:{status}"),
                    )
                else:
                    run_pipeline.run_file_list(path, overrides, stop_check=self.stop_event.is_set)
                self.log_queue.put("=== BATCH DONE ===")
            elif mode == MODES[2]:
                run_pipeline.run_batch_folder(path, overrides, stop_check=self.stop_event.is_set)
                self.log_queue.put("=== BATCH FOLDER DONE ===")
            elif mode == MODES[3]:
                run_pipeline.run_notes_only(path, overrides)
                self.log_queue.put("=== DONE ===")
            else:
                run_pipeline.run_notes_batch(path, overrides)
                self.log_queue.put("=== NOTES BATCH DONE ===")
        except Exception as exc:
            self.log_queue.put(f"FATAL ERROR: {exc}")
            import traceback
            self.log_queue.put(traceback.format_exc())
        finally:
            root_logger.removeHandler(handler)
            if file_handler:
                root_logger.removeHandler(file_handler)
                file_handler.close()
            self.log_queue.put("__RUN_COMPLETE__")

    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    def _poll_log_queue(self):
        try:
            while True:
                line = self.log_queue.get_nowait()
                if line == "__RUN_COMPLETE__":
                    self.run_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.stop_event.clear()
                    self.status_label.config(text="Ready.")
                    self.progress_var.set(0)
                    self.stage_label.config(text="")
                    continue
                if line.startswith("__BATCH_STATUS__:"):
                    # "__BATCH_STATUS__:<i>:<status>" (task #74): <i> is the
                    # row's 1-based position among rows actually passed to
                    # run_pipeline.run_batch_rows, matching self._batch_run_iids
                    # 1:1 (see _batch_rows_with_iids). Never shown in the log
                    # text itself - it is purely a control message for the
                    # Status column.
                    try:
                        _, idx_str, status = line.split(":", 2)
                        idx = int(idx_str) - 1
                        if 0 <= idx < len(self._batch_run_iids):
                            self.batch_tree.set(self._batch_run_iids[idx], "status", status)
                    except (ValueError, IndexError):
                        pass
                    continue
                for marker, pct, label in STAGE_PROGRESS:
                    if marker in line:
                        self.progress_var.set(pct)
                        self.stage_label.config(text=label)
                        break
                self.log_text.config(state="normal")
                self.log_text.insert("end", line + "\n")
                self.log_text.see("end")
                self.log_text.config(state="disabled")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)


class TranslateTabController:
    """
    Controls the "Translate" tab: translate txt/docx/pptx/pdf/image files
    into a chosen target language. Independent of the audio/video
    pipeline (no whisper/notes/slides involved - see translator.py/
    doc_translate.py for the actual translation logic). Deliberately
    much simpler than RunTabController: one mode selector (Single file /
    File list (batch) / Folder (auto-discover)), a target-language
    dropdown, an optional output-folder override, and the same
    worker-thread + queue-log pattern as the Run tabs, minus the
    per-stage progress bar (translation has no distinct stages).
    """

    def __init__(self, tab: ttk.Frame, app):
        self.tab = tab
        self.app = app
        self.log_queue: queue.Queue = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self._batch_run_iids: list = []
        self._build()
        self.app.root.after(100, self._poll_log_queue)

    def _build(self):
        parent = self.app._make_scrollable(self.tab)
        pad = {"padx": 8, "pady": 4}

        self.mode_var = tk.StringVar(value=TRANSLATE_MODES[0])
        self.path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.target_language_var = tk.StringVar(value=CONFIG.get("translate_target_language", "en"))
        self.ocr_language_var = tk.StringVar(value=CONFIG.get("translate_ocr_language", "auto"))

        # --- Mode + source ---
        source = ttk.LabelFrame(parent, text="Source")
        source.pack(fill="x", **pad)
        self._source_frame = source
        ttk.Label(source, text="Mode:").grid(row=0, column=0, sticky="w", **pad)
        mode_box = ttk.Combobox(source, textvariable=self.mode_var, values=TRANSLATE_MODES,
                                state="readonly", width=24)
        mode_box.grid(row=0, column=1, sticky="w")
        mode_box.bind("<<ComboboxSelected>>", lambda e: self._update_mode_visibility())

        self.path_label = ttk.Label(source, text="File:")
        self.path_label.grid(row=1, column=0, sticky="w", **pad)
        self.path_entry = ttk.Entry(source, textvariable=self.path_var, width=68)
        self.path_entry.grid(row=1, column=1, sticky="w")
        ttk.Button(source, text="Browse...", command=self._browse_path).grid(
            row=1, column=2, sticky="w", padx=(4, 0))

        supported = ", ".join(sorted(doc_translate.TRANSLATABLE_EXTENSIONS))
        ttk.Label(source,
                 text=f"Supported: {supported}. Legacy .doc/.ppt are not supported - "
                      "save as .docx/.pptx first. pdf and images are written as a "
                      "translated .docx (no editable layout to translate into).",
                 foreground="#666", wraplength=860).grid(
            row=2, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 4))

        # --- Batch table (File list mode only; hidden otherwise) ---
        self.batch_frame = ttk.LabelFrame(parent, text="Files to translate (one per row)")
        self.batch_columns = ("file",)
        self.batch_tree = ttk.Treeview(self.batch_frame, columns=("file", "status"),
                                       show="headings", height=8)
        self.batch_tree.heading("file", text="File")
        self.batch_tree.heading("status", text="Status")
        self.batch_tree.column("file", width=580, anchor="w")
        self.batch_tree.column("status", width=90, anchor="w")
        self.batch_tree.pack(fill="both", expand=True, padx=4, pady=(4, 4))
        self.batch_tree.bind("<Double-1>", self._on_batch_double_click)

        batch_btn_row = ttk.Frame(self.batch_frame)
        batch_btn_row.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Button(batch_btn_row, text="Add files...", command=self._batch_add_files).pack(side="left")
        ttk.Button(batch_btn_row, text="Remove selected", command=self._batch_remove_selected).pack(
            side="left", padx=(6, 0))
        ttk.Button(batch_btn_row, text="Clear", command=self._batch_clear).pack(side="left", padx=(6, 0))
        ttk.Button(batch_btn_row, text="Load .txt...", command=self._batch_load_path).pack(
            side="left", padx=(12, 0))
        ttk.Button(batch_btn_row, text="Save box to .txt...", command=self._batch_save_path).pack(
            side="left", padx=(6, 0))
        ttk.Label(self.batch_frame,
                 text="Double-click a row to open its containing folder. Leave the table empty "
                      "and use \"File:\" above to point at an existing list.txt instead.",
                 foreground="#666", wraplength=860).pack(fill="x", padx=8, pady=(0, 6))

        # --- Parameters ---
        params = ttk.LabelFrame(parent, text="Parameters")
        params.pack(fill="x", **pad)
        ttk.Label(params, text="Target language:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Combobox(params, textvariable=self.target_language_var, values=TRANSLATE_LANGUAGES,
                    state="readonly", width=10).grid(row=0, column=1, sticky="w")

        ttk.Label(params, text="OCR language (images only):").grid(row=0, column=2, sticky="w", **pad)
        ocr_lang_box = ttk.Combobox(params, textvariable=self.ocr_language_var,
                                    values=["auto"] + TRANSLATE_LANGUAGES, state="readonly", width=10)
        ocr_lang_box.grid(row=0, column=3, sticky="w")
        Tooltip(ocr_lang_box,
                "What script Tesseract should expect to see in an IMAGE input (.jpg/.png/etc.) "
                "before translation - NOT the translation target above. Wrong here silently "
                "garbles non-Latin scripts (e.g. Chinese slides) into fluent-sounding nonsense "
                "once the LLM 'translates' the garbled OCR text. \"auto\" = \"eng+deu\". Set to "
                "the slide's actual language, e.g. \"zh\" for Chinese.")

        ttk.Label(params, text="Output folder (blank = next to source):").grid(
            row=1, column=0, sticky="w", **pad)
        ttk.Entry(params, textvariable=self.output_dir_var, width=50).grid(
            row=1, column=1, sticky="w", columnspan=2)
        ttk.Button(params, text="Browse...", command=self._browse_output_dir).grid(
            row=1, column=3, sticky="w", padx=(4, 0))

        # --- Run button + progress ---
        run_frame = ttk.Frame(parent)
        run_frame.pack(fill="x", **pad)
        self.run_button = ttk.Button(run_frame, text="Translate", command=self._on_run)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(run_frame, text="Stop", command=self._on_stop, state="disabled")
        self.stop_button.pack(side="left", padx=(6, 0))
        self.status_label = ttk.Label(run_frame, text="Ready.")
        self.status_label.pack(side="left", padx=12)

        progress_frame = ttk.Frame(parent)
        progress_frame.pack(fill="x", **pad)
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", mode="determinate",
                                            maximum=100, variable=self.progress_var, length=320)
        self.progress_bar.pack(side="left")

        log_frame = ttk.LabelFrame(parent, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log_text = scrolledtext.ScrolledText(
            log_frame, state="disabled", height=14, wrap="word",
            bg=_theme_color("neutral_100"), fg=_theme_color("neutral_900"),
            insertbackground=_theme_color("neutral_900"),
            selectbackground=_theme_color("primary_300"))
        self.log_text.pack(fill="both", expand=True)

        self._update_mode_visibility()

    # -----------------------------------------------------------------
    # Mode/path helpers
    # -----------------------------------------------------------------

    def _update_mode_visibility(self):
        """Toggle the batch table for File list (batch) mode. Always
        re-anchored via after=self._source_frame (see RunTabController's
        _update_path_label for the same pattern), so showing/hiding it
        never reorders Parameters/Run/Log below it."""
        mode = self.mode_var.get()
        self.batch_frame.pack_forget()
        if mode == TRANSLATE_MODES[1]:  # File list (batch)
            self.path_label.config(text="Or existing list.txt:")
            self.batch_frame.pack(fill="both", expand=True, padx=8, pady=4, after=self._source_frame)
        elif mode == TRANSLATE_MODES[2]:  # Folder (auto-discover)
            self.path_label.config(text="Folder:")
        else:
            self.path_label.config(text="File:")

    def _browse_path(self):
        mode = self.mode_var.get()
        if mode == TRANSLATE_MODES[2]:
            path = filedialog.askdirectory(title="Select folder")
        elif mode == TRANSLATE_MODES[1]:
            path = filedialog.askopenfilename(
                title="Select list.txt", filetypes=[("Text", "*.txt"), ("All files", "*.*")])
        else:
            exts = sorted(doc_translate.TRANSLATABLE_EXTENSIONS)
            pattern = " ".join(f"*{e}" for e in exts)
            path = filedialog.askopenfilename(
                title="Select file to translate",
                filetypes=[("Supported documents", pattern), ("All files", "*.*")])
        if path:
            self.path_var.set(path)

    def _browse_output_dir(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_dir_var.set(path)

    # -----------------------------------------------------------------
    # Batch table helpers (File list (batch) mode)
    # -----------------------------------------------------------------

    def _batch_add_files(self):
        exts = sorted(doc_translate.TRANSLATABLE_EXTENSIONS)
        pattern = " ".join(f"*{e}" for e in exts)
        paths = filedialog.askopenfilenames(
            title="Add files to translate",
            filetypes=[("Supported documents", pattern), ("All files", "*.*")])
        for p in paths:
            self.batch_tree.insert("", "end", values=(p, ""))

    def _batch_remove_selected(self):
        for iid in self.batch_tree.selection():
            self.batch_tree.delete(iid)

    def _batch_clear(self):
        for iid in self.batch_tree.get_children():
            self.batch_tree.delete(iid)

    def _on_batch_double_click(self, event):
        region = self.batch_tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        iid = self.batch_tree.identify_row(event.y)
        if not iid:
            return
        file_path = self.batch_tree.item(iid, "values")[0]
        if file_path:
            self.app._open_path(Path(file_path).parent)

    def _batch_load_path(self):
        path = filedialog.askopenfilename(
            title="Load file list", filetypes=[("Text", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Could not read file", str(exc))
            return
        added = 0
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            self.batch_tree.insert("", "end", values=(line, ""))
            added += 1
        messagebox.showinfo("Loaded", f"Added {added} file(s).")

    def _batch_save_path(self):
        path = filedialog.asksaveasfilename(
            title="Save file list", defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if not path:
            return
        lines = [self.batch_tree.item(iid, "values")[0] for iid in self.batch_tree.get_children()]
        try:
            Path(path).write_text("\n".join(lines), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        messagebox.showinfo("Saved", f"Saved {len(lines)} file(s) to:\n{path}")

    # -----------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------

    def _on_run(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Busy", "A translation is already in progress.")
            return

        mode = self.mode_var.get()
        path = self.path_var.get().strip()
        target_language = self.target_language_var.get()
        output_dir = self.output_dir_var.get().strip()
        overrides = {"translate_ocr_language": self.ocr_language_var.get()}
        if output_dir:
            overrides["output_dir_override"] = output_dir

        batch_paths = None
        self._batch_run_iids = []
        if mode == TRANSLATE_MODES[1]:
            rows = list(self.batch_tree.get_children())
            batch_paths = [self.batch_tree.item(iid, "values")[0] for iid in rows
                          if self.batch_tree.item(iid, "values")[0]]
            if not batch_paths and not path:
                messagebox.showwarning(
                    "Missing input",
                    "Add files to the table above, or choose an existing list.txt.")
                return
            if batch_paths:
                self._batch_run_iids = rows
                for iid in rows:
                    self.batch_tree.set(iid, "status", "")
        elif not path:
            messagebox.showwarning("Missing input", "Please choose a file/folder first.")
            return

        self._clear_log()
        self.run_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.stop_event.clear()
        self.status_label.config(text="Translating...")
        self.progress_var.set(0)

        self.worker_thread = threading.Thread(
            target=self._run_worker, args=(mode, path, target_language, overrides, batch_paths),
            daemon=True,
        )
        self.worker_thread.start()

    def _on_stop(self):
        self.stop_event.set()
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Stopping (finishing current file)...")

    def _run_worker(self, mode: str, path: str, target_language: str, overrides: dict,
                    batch_paths: list | None):
        handler = QueueLogHandler(self.log_queue)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        level = getattr(logging, self.app.log_level_var.get().upper(), logging.INFO)
        root_logger.setLevel(level)
        try:
            if batch_paths:
                self._run_batch_paths(batch_paths, target_language, overrides)
            elif mode == TRANSLATE_MODES[1]:
                run_pipeline.translate_file_list(
                    path, target_language, overrides,
                    stop_check=self.stop_event.is_set, progress_callback=self._on_batch_progress)
            elif mode == TRANSLATE_MODES[2]:
                run_pipeline.translate_batch_folder(
                    path, target_language, overrides,
                    stop_check=self.stop_event.is_set, progress_callback=self._on_batch_progress)
            else:
                dest = run_pipeline.translate_single_file(path, target_language, overrides)
                self.log_queue.put(f"=== TRANSLATION DONE: {dest} ===")
        except Exception as exc:
            self.log_queue.put(f"FATAL ERROR: {exc}")
            import traceback
            self.log_queue.put(traceback.format_exc())
        finally:
            root_logger.removeHandler(handler)
            self.log_queue.put("__RUN_COMPLETE__")

    def _run_batch_paths(self, batch_paths: list, target_language: str, overrides: dict):
        """Translate the GUI batch table's rows directly (not via a
        list.txt written to disk first), mirroring translate_file_list's
        per-file try/except + progress-callback behaviour."""
        cfg = run_pipeline.build_run_config(overrides)
        translate_fn = run_pipeline._make_translate_fn(cfg, target_language)
        ok, errors = 0, 0
        total = len(batch_paths)
        for i, file_path in enumerate(batch_paths, 1):
            if self.stop_event.is_set():
                self.log_queue.put(f"Stop requested - halting after {i - 1}/{total} file(s).")
                break
            self._on_batch_progress(i, "running")
            try:
                dest = doc_translate.translate_file(
                    file_path, target_language, translate_fn,
                    output_dir_override=cfg.get("output_dir_override", ""),
                    ocr_lang=run_pipeline._resolve_ocr_language(cfg))
                self.log_queue.put(f"[{i}/{total}] OK: {Path(file_path).name} -> {dest.name}")
                ok += 1
                self._on_batch_progress(i, "done")
            except Exception as e:
                self.log_queue.put(f"[{i}/{total}] ERROR: {Path(file_path).name} -- {e}")
                errors += 1
                self._on_batch_progress(i, "failed")
        self.log_queue.put(f"=== TRANSLATION BATCH DONE: OK={ok} Errors={errors} ===")

    def _on_batch_progress(self, index: int, status: str):
        self.log_queue.put(f"__BATCH_STATUS__:{index}:{status}")

    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    def _poll_log_queue(self):
        try:
            while True:
                line = self.log_queue.get_nowait()
                if line == "__RUN_COMPLETE__":
                    self.run_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.stop_event.clear()
                    self.status_label.config(text="Ready.")
                    self.progress_var.set(0)
                    continue
                if line.startswith("__BATCH_STATUS__:"):
                    try:
                        _, idx_str, status = line.split(":", 2)
                        idx = int(idx_str) - 1
                        if 0 <= idx < len(self._batch_run_iids):
                            self.batch_tree.set(self._batch_run_iids[idx], "status", status)
                    except (ValueError, IndexError):
                        pass
                    continue
                self.log_text.config(state="normal")
                self.log_text.insert("end", line + "\n")
                self.log_text.see("end")
                self.log_text.config(state="disabled")
        except queue.Empty:
            pass
        self.app.root.after(100, self._poll_log_queue)


class KnownSpeakersDialog(tk.Toplevel):
    """
    Modal window (task #82): manage the global known_speakers roster
    (db.py) used by speaker identification (tasks #79-81). Lists every
    enrolled voiceprint - name, how many confirmed renames contributed to
    it (sample_count), and when it was last updated - and lets the user
    rename or delete an entry. Deleting only removes the stored
    voiceprint; it never touches any already-written transcript, so past
    renames stay exactly as they are - future runs simply stop
    suggesting that name.
    """

    def __init__(self, app):
        super().__init__(app.root)
        _style_toplevel(self)
        self.app = app
        self.title("Manage Known Speakers")
        self.resizable(False, False)
        self.transient(app.root)

        ttk.Label(self, text="Global voiceprint roster used by speaker identification "
                             "suggestions in Rename Speakers.",
                 foreground="#666", wraplength=460).pack(anchor="w", padx=10, pady=(10, 8))

        columns = ("name", "samples", "updated")
        self.tree = ttk.Treeview(self, columns=columns, show="headings", height=10)
        self.tree.heading("name", text="Name")
        self.tree.heading("samples", text="Samples")
        self.tree.heading("updated", text="Last updated")
        self.tree.column("name", width=200)
        self.tree.column("samples", width=70, anchor="center")
        self.tree.column("updated", width=160)
        self.tree.pack(fill="both", expand=True, padx=10)

        btn_row = ttk.Frame(self)
        btn_row.pack(fill="x", padx=10, pady=10)
        ttk.Button(btn_row, text="Rename...", command=self._rename_selected).pack(side="left")
        ttk.Button(btn_row, text="Delete...", command=self._delete_selected).pack(side="left", padx=(8, 0))
        ttk.Button(btn_row, text="Refresh", command=self._refresh).pack(side="left", padx=(8, 0))
        ttk.Button(btn_row, text="Close", command=self.destroy).pack(side="right")

        self._rows = []  # parallel to tree iids: raw db.list_known_speakers rows
        self._refresh()

    def _db_path(self) -> str:
        return self.app.db_path_var.get().strip() or CONFIG["db_path"]

    def _refresh(self):
        self.tree.delete(*self.tree.get_children())
        self._rows = []
        db_path = self._db_path()
        if not Path(db_path).exists():
            return
        conn = db.get_connection(db_path)
        try:
            rows = db.list_known_speakers(conn)
        finally:
            conn.close()
        for row in rows:
            iid = str(len(self._rows))
            self._rows.append(row)
            self.tree.insert("", "end", iid=iid,
                             values=(row["name"], row["sample_count"], row["updated_at"]))

    def _selected_row(self) -> dict | None:
        sel = self.tree.selection()
        if not sel:
            return None
        return self._rows[int(sel[0])]

    def _rename_selected(self):
        row = self._selected_row()
        if not row:
            messagebox.showinfo("No selection", "Select a speaker first.")
            return
        new_name = simpledialog.askstring(
            "Rename Speaker", "New name:", initialvalue=row["name"], parent=self)
        if not new_name or new_name.strip() == row["name"]:
            return
        conn = db.get_connection(self._db_path())
        try:
            db.rename_known_speaker(conn, row["speaker_id"], new_name.strip())
        except Exception as exc:
            messagebox.showerror("Rename failed", str(exc))
            return
        finally:
            conn.close()
        self._refresh()

    def _delete_selected(self):
        row = self._selected_row()
        if not row:
            messagebox.showinfo("No selection", "Select a speaker first.")
            return
        if not messagebox.askyesno(
            "Delete speaker",
            f"Delete the voiceprint for '{row['name']}'?\n\n"
            f"This does not change any transcript already renamed to this "
            f"name - it only stops future runs from suggesting it."):
            return
        conn = db.get_connection(self._db_path())
        try:
            db.delete_known_speaker(conn, row["speaker_id"])
        finally:
            conn.close()
        self._refresh()


class SpeakerRenameDialog(tk.Toplevel):
    """
    Modal window (task #76, extended task #81): post-hoc speaker renaming.
    Lists the distinct speaker labels found in an existing *_segments.json
    cache, lets the user type a replacement name for each, and applies the
    rename on Apply via run_pipeline.rename_speakers (rewrites
    *_segments.json, *_transcript_speakers.txt, *_text.txt,
    *_transcript.srt; optionally also regenerates notes). Runs in the same
    background worker thread/log_queue infrastructure as a normal pipeline
    run, so it can't overlap with one and its progress shows in the same
    log panel.

    Task #81 additions, active only when CONFIG["enable_speaker_id"] is on
    and the original audio/video file is still findable next to the
    transcript (source_media_path): a background thread computes one
    voiceprint per speaker label (speaker_id.py) and matches it against
    the global known_speakers roster, pre-filling the name field when a
    match clears the configured threshold and always showing the
    confidence next to it. Each row also gets a "Play sample" button that
    extracts and plays up to CONFIG["speaker_sample_seconds"] of that
    speaker's speech (speaker_id.extract_speaker_sample_clip, longest
    segments first, concatenated if needed) so the suggestion can be
    confirmed by ear before accepting it. On Apply, any name the user
    settled on is committed back into known_speakers
    (run_pipeline.commit_speaker_identities) in the same worker thread
    that performs the actual local rename.

    V1.27 additions: an "Improve audio" button next to Play sample runs
    the sample clip through the same ffmpeg cleanup filter chain as the
    opt-in pre-transcription enhance_audio pass (highpass + mild denoise
    + loudness normalization, see transcriber.enhance_audio) before
    playing it - useful when a quiet or noisy speaker is hard to judge by
    ear from the raw sample. Both buttons now play via ffplay
    (config.get_ffplay_path: a bundled ffmpeg\\bin\\ or PATH, windowless,
    -autoexit) instead of handing the clip to the OS's default app for
    .wav (_play_audio on the App class), falling back to that default
    app only if ffplay isn't available.

    V1.28 additions:
    - "Stop playback" (bottom button row) stops whatever sample is
      currently playing (App._stop_audio) - also called automatically
      whenever a new sample starts playing, and when this dialog closes.
    - "Edit sample..." per row opens SampleEditDialog, listing the
      chunks speaker_id.select_sample_segments picked for that speaker
      (chronologically, with their time range) so individual ones can be
      removed (e.g. cross-talk, noise, a diarization mix-up) before
      re-rendering (speaker_id.render_sample_clip) and replaying the
      clip. The active (possibly user-edited) segment list per label is
      cached in self._sample_segments so Play sample/Improve audio reuse
      it instead of recomputing the original, unedited selection.

    V1.30 additions (task #95): each row's name field is now a Combobox
    rather than a plain Entry, pre-populated with the real invitee names
    from this recording's loaded meeting info (an .ics calendar invite
    or a "Invitees:" line - see RunTabController._get_invitee_names),
    when any were loaded. This puts the actual people invited to the
    meeting ahead of a blind guess without ever auto-assigning a
    specific name to a specific speaker (that would require knowing who
    spoke, which nothing here can determine acoustically). If a
    voiceprint suggestion's name matches one of the invitees (case-
    insensitively), the prefilled text is snapped to the invitee's exact
    spelling/casing.
    """

    def __init__(self, controller, segments_json_path: str, labels: list,
                 source_media_path: str | None = None,
                 invitee_names: list | None = None):
        super().__init__(controller.root)
        _style_toplevel(self)
        self.controller = controller
        self.segments_json_path = segments_json_path
        self.source_media_path = source_media_path
        # Task #95: real invitee names from the calendar invite/meeting
        # info loaded for this recording (if any) - offered as a pick-list
        # in each row's Combobox, ahead of a blind guess, and used to snap
        # a voiceprint suggestion to its correctly-spelled/-cased form
        # when the two agree (see _poll_suggestions_queue).
        self._invitee_names = invitee_names or []
        self.title("Rename Speakers")
        self.resizable(False, False)
        self.transient(controller.root)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._suggestions: dict = {}
        self._suggestions_queue: queue.Queue = queue.Queue()
        self._sample_dir = tempfile.mkdtemp(prefix="tnp_speaker_samples_")
        # label -> list of {"start", "end"} dicts actually in use for that
        # speaker's preview clip: the original speaker_id.select_sample_segments
        # pick until "Edit sample..." (SampleEditDialog) removes some, at
        # which point this is the edited list instead (V1.28).
        self._sample_segments: dict = {}

        ttk.Label(self, text=f"File: {Path(segments_json_path).name}",
                 foreground="#666").pack(anchor="w", padx=10, pady=(10, 4))
        ttk.Label(self, text="Enter a new name for any speaker you want to rename. "
                             "Leave unchanged to keep the original label.",
                 foreground="#666", wraplength=420).pack(anchor="w", padx=10, pady=(0, 8))

        rows_frame = ttk.Frame(self)
        rows_frame.pack(fill="x", padx=10)
        self._vars = {}
        self._suggestion_vars = {}
        for i, label in enumerate(labels):
            ttk.Label(rows_frame, text=label, width=16).grid(row=i, column=0, sticky="w", pady=2)
            ttk.Label(rows_frame, text="->").grid(row=i, column=1, padx=6)
            var = tk.StringVar(value=label)
            ttk.Combobox(rows_frame, textvariable=var, width=18,
                        values=self._invitee_names).grid(row=i, column=2, sticky="w")
            self._vars[label] = var
            suggestion_var = tk.StringVar(value="")
            ttk.Label(rows_frame, textvariable=suggestion_var, foreground="#666",
                     width=28).grid(row=i, column=3, sticky="w", padx=(8, 0))
            self._suggestion_vars[label] = suggestion_var
            ttk.Button(rows_frame, text="Play sample", width=11,
                      command=lambda l=label: self._play_sample(l)).grid(row=i, column=4, padx=(8, 0))
            ttk.Button(rows_frame, text="Improve audio", width=13,
                      command=lambda l=label: self._improve_audio(l)).grid(row=i, column=5, padx=(6, 0))
            ttk.Button(rows_frame, text="Edit sample...", width=13,
                      command=lambda l=label: self._edit_sample(l)).grid(row=i, column=6, padx=(6, 0))

        self.regenerate_notes_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text="Also regenerate notes with the new names",
                       variable=self.regenerate_notes_var).pack(anchor="w", padx=10, pady=(8, 0))

        self._status_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._status_var, foreground="#666",
                 wraplength=420).pack(anchor="w", padx=10, pady=(4, 0))

        btn_row = ttk.Frame(self)
        btn_row.pack(fill="x", padx=10, pady=10)
        ttk.Button(btn_row, text="Apply", command=self._apply).pack(side="right")
        ttk.Button(btn_row, text="Cancel", command=self._on_close).pack(side="right", padx=(0, 6))
        ttk.Button(btn_row, text="Stop playback",
                  command=self.controller.app._stop_audio).pack(side="left")

        # Tracks whether _compute_suggestions_worker is still running (V1.20),
        # and the error text if it already finished but failed (V1.21).
        # _apply() checks both before committing to known_speakers: without
        # this, clicking Apply either while the background pyannote
        # embedding extraction is still in flight, OR after it already
        # finished with an error (e.g. a gated HuggingFace model not yet
        # accepted), silently skipped roster enrollment entirely
        # (self._suggestions stays {} in both cases), with no warning and
        # no log line, since renaming itself never depended on suggestions
        # being ready.
        self._suggestions_pending = False
        self._suggestions_error: str | None = None

        invitee_note = (f" {len(self._invitee_names)} invitee(s) from the loaded meeting info "
                        f"are offered in each name field's dropdown." if self._invitee_names else "")
        if self.controller.app.enable_speaker_id_var.get():
            if self.source_media_path and Path(self.source_media_path).exists():
                self._suggestions_pending = True
                self._status_var.set("Computing voiceprint suggestions..." + invitee_note)
                threading.Thread(target=self._compute_suggestions_worker, daemon=True).start()
                self.after(200, self._poll_suggestions_queue)
            else:
                self._status_var.set(
                    "Source media not found next to this transcript - voiceprint "
                    "suggestions unavailable, renaming still works." + invitee_note)
        elif self._invitee_names:
            self._status_var.set("Voiceprint speaker ID is off." + invitee_note)

    def _compute_suggestions_worker(self):
        try:
            db_path = self.controller.app.db_path_var.get().strip() or CONFIG["db_path"]
            threshold = self.controller.app.speaker_id_threshold_var.get()
            suggestions = run_pipeline.get_speaker_suggestions(
                self.segments_json_path, self.source_media_path,
                CONFIG.get("hf_token", ""), threshold, db_path)
            self._suggestions_queue.put(("ok", suggestions))
        except Exception as exc:
            self._suggestions_queue.put(("error", str(exc)))

    def _poll_suggestions_queue(self):
        if not self.winfo_exists():
            return
        try:
            status, payload = self._suggestions_queue.get_nowait()
        except queue.Empty:
            self.after(200, self._poll_suggestions_queue)
            return
        self._suggestions_pending = False
        if status == "error":
            self._suggestions_error = payload
            self._status_var.set(f"Voiceprint suggestions failed: {payload}")
            return
        self._suggestions = payload
        threshold = self.controller.app.speaker_id_threshold_var.get()
        invitees_lower = {n.lower(): n for n in self._invitee_names}
        for label, info in payload.items():
            suggestion_var = self._suggestion_vars.get(label)
            if suggestion_var is None:
                continue
            if info["suggested_name"]:
                # Task #95: snap to the invitee's exact spelling/casing
                # when the voiceprint match and a real invitee agree -
                # never invents a name outright, only corrects one
                # already suggested by the roster match.
                suggested_name = invitees_lower.get(info["suggested_name"].lower(), info["suggested_name"])
                suggestion_var.set(f"suggests: {suggested_name} ({info['score']:.2f})")
                entry_var = self._vars.get(label)
                if entry_var is not None and entry_var.get() == label:
                    entry_var.set(suggested_name)
            elif info["score"] > 0:
                suggestion_var.set(f"closest match {info['score']:.2f} (below threshold {threshold:.2f})")
            else:
                if self._invitee_names:
                    suggestion_var.set("no match - pick a name from the invitee list")
                else:
                    suggestion_var.set("no match - looks like a new speaker")
        self._status_var.set(f"Voiceprint suggestions ready for {len(payload)} speaker(s).")

    def _play_sample(self, label: str):
        if not self.source_media_path or not Path(self.source_media_path).exists():
            messagebox.showinfo(
                "No source media",
                "The original audio/video file was not found next to this transcript.")
            return
        threading.Thread(target=self._play_sample_worker, args=(label,), daemon=True).start()

    def _play_sample_worker(self, label: str):
        try:
            picked = self._get_active_segments(label)
            if not picked:
                self.after(0, lambda: messagebox.showinfo(
                    "No sample available",
                    f"Could not extract an audio sample for {label} "
                    f"(no long-enough segment found, or ffmpeg is not installed)."))
                return
            clip_path = str(Path(self._sample_dir) / f"{label}.wav")
            result = speaker_id.render_sample_clip(self.source_media_path, picked, clip_path)
        except Exception as exc:
            # str(exc) must be captured here, not inside the lambda: "except
            # ... as exc" implicitly deletes exc when this block exits, and
            # self.after(0, ...) only runs the lambda later, once tkinter's
            # event loop gets to it - by then exc no longer exists in this
            # scope, raising NameError instead of showing the real error.
            err_msg = str(exc)
            self.after(0, lambda: messagebox.showerror("Playback failed", err_msg))
            return
        if result:
            self.after(0, lambda: self.controller.app._play_audio(Path(result)))
        else:
            self.after(0, lambda: messagebox.showinfo(
                "No sample available",
                f"Could not extract an audio sample for {label} "
                f"(no long-enough segment found, or ffmpeg is not installed)."))

    def _get_active_segments(self, label: str) -> list:
        """Returns the segment list currently in use for label's preview
        clip: the cached, possibly user-edited (SampleEditDialog) list if
        one already exists for this dialog session, else the original
        speaker_id.select_sample_segments pick (computed and cached here
        for the first time). Both Play sample/Improve audio and
        "Edit sample..." go through this so an edit made via one button
        is reflected the next time either of the others is used."""
        if label in self._sample_segments:
            return self._sample_segments[label]
        segments = json.loads(Path(self.segments_json_path).read_text(encoding="utf-8"))
        picked = speaker_id.select_sample_segments(
            segments, label, max_duration=CONFIG.get("speaker_sample_seconds", 15.0))
        self._sample_segments[label] = picked
        return picked

    def _edit_sample(self, label: str):
        """"Edit sample..." button (V1.28): opens SampleEditDialog over
        the segments _get_active_segments picked for label, so individual
        chunks (e.g. cross-talk, noise, a diarization mix-up) can be
        removed before re-rendering the preview clip."""
        if not self.source_media_path or not Path(self.source_media_path).exists():
            messagebox.showinfo(
                "No source media",
                "The original audio/video file was not found next to this transcript.")
            return
        picked = self._get_active_segments(label)
        if not picked:
            messagebox.showinfo(
                "No sample available",
                f"Could not find any usable segment for {label} to edit.")
            return
        SampleEditDialog(self, label, picked)

    def _improve_audio(self, label: str):
        """"Improve audio" button (V1.27): runs the same ffmpeg cleanup
        filter chain used for the opt-in pre-transcription enhance_audio
        pass (highpass + mild denoise + loudness normalization, see
        transcriber.enhance_audio) on that speaker's sample clip instead
        of the raw extract, then plays the cleaned result - useful when a
        quiet/noisy speaker is hard to judge by ear from the raw sample."""
        if not self.source_media_path or not Path(self.source_media_path).exists():
            messagebox.showinfo(
                "No source media",
                "The original audio/video file was not found next to this transcript.")
            return
        threading.Thread(target=self._improve_audio_worker, args=(label,), daemon=True).start()

    def _improve_audio_worker(self, label: str):
        try:
            picked = self._get_active_segments(label)
            if not picked:
                self.after(0, lambda: messagebox.showinfo(
                    "No sample available",
                    f"Could not extract an audio sample for {label} "
                    f"(no long-enough segment found, or ffmpeg is not installed)."))
                return
            # Always re-render from the current active segment list (rather
            # than reusing a cached raw_clip file) so an edit made via
            # "Edit sample..." since the last Improve audio click is
            # reflected here too, not just in Play sample.
            raw_clip = Path(self._sample_dir) / f"{label}.wav"
            result = speaker_id.render_sample_clip(self.source_media_path, picked, str(raw_clip))
            if not result:
                self.after(0, lambda: messagebox.showinfo(
                    "No sample available",
                    f"Could not extract an audio sample for {label} "
                    f"(no long-enough segment found, or ffmpeg is not installed)."))
                return
            enhanced_clip = Path(self._sample_dir) / f"{label}_enhanced.wav"
            enhanced_ok = run_pipeline.transcriber.enhance_audio(str(raw_clip), str(enhanced_clip))
        except Exception as exc:
            # See _play_sample_worker's except block: capture the message
            # now, since "except ... as exc" deletes exc as soon as this
            # block exits, before the deferred self.after(0, ...) lambda
            # ever runs.
            err_msg = str(exc)
            self.after(0, lambda: messagebox.showerror("Improve audio failed", err_msg))
            return
        if enhanced_ok:
            self.after(0, lambda: self.controller.app._play_audio(enhanced_clip))
        else:
            self.after(0, lambda: messagebox.showinfo(
                "Enhancement unavailable",
                "Could not enhance this clip (ffmpeg not found, or the filter pass "
                "failed) - playing the original sample instead."))
            self.after(0, lambda: self.controller.app._play_audio(raw_clip))

    def _on_close(self):
        self.controller.app._stop_audio()
        shutil.rmtree(self._sample_dir, ignore_errors=True)
        self.destroy()

    def _apply(self):
        mapping = {old: var.get().strip() for old, var in self._vars.items()
                  if var.get().strip() and var.get().strip() != old}
        if not mapping:
            messagebox.showinfo("Nothing to rename", "No labels were changed.")
            return
        if self._suggestions_pending:
            proceed = messagebox.askyesno(
                "Voiceprint suggestions still computing",
                "Voiceprint suggestions for this file are still being computed "
                "in the background. The rename itself will work normally "
                "either way, but if you continue now, none of these speakers "
                "will be added to your known-speakers roster this time.\n\n"
                "Continue without waiting?")
            if not proceed:
                return
        elif self._suggestions_error and not self._suggestions:
            proceed = messagebox.askyesno(
                "Voiceprint suggestions failed",
                "Voiceprint suggestions could not be computed for this file, "
                "so none of these speakers will be added to your "
                "known-speakers roster. The rename itself is unaffected and "
                "will still work normally.\n\n"
                f"Error: {self._suggestions_error}\n\n"
                "Continue with the rename anyway?")
            if not proceed:
                return
        regenerate_notes = self.regenerate_notes_var.get()
        segments_json_path = self.segments_json_path
        speaker_suggestions = self._suggestions
        db_path = self.controller.app.db_path_var.get().strip() or CONFIG["db_path"]
        self.controller.app._stop_audio()
        shutil.rmtree(self._sample_dir, ignore_errors=True)
        self.destroy()

        c = self.controller
        c._clear_log()
        c.run_button.config(state="disabled")
        c.stop_button.config(state="disabled")
        c.status_label.config(text="Renaming speakers...")
        c.progress_var.set(0)
        c.stage_label.config(text="")
        c.worker_thread = threading.Thread(
            target=c._run_rename_speakers_worker,
            args=(segments_json_path, mapping, regenerate_notes),
            kwargs={"speaker_suggestions": speaker_suggestions, "db_path": db_path},
            daemon=True,
        )
        c.worker_thread.start()


class SampleEditDialog(tk.Toplevel):
    """
    "Edit sample..." dialog (V1.28), opened from SpeakerRenameDialog:
    lets the user remove one or more of the chunks that make up a
    speaker's preview clip (speaker_id.select_sample_segments's pick, or
    a previously-edited subset of it) before naming them - e.g. a chunk
    that turned out to be cross-talk, background noise, or another
    speaker bleeding through from a diarization mistake.

    Segments are listed chronologically with their time range and
    duration in a multi-select Listbox; "Remove selected" drops them
    from THIS dialog's working copy only (does not touch the transcript/
    segments.json - purely affects the preview clip); "Rebuild & play"
    re-renders the clip from whatever remains (speaker_id.render_sample_clip)
    via App._play_audio, and commits the edited list back into the
    parent SpeakerRenameDialog's _sample_segments cache so Play sample/
    Improve audio pick it up too from then on. Closing without rebuilding
    leaves the parent's cached selection unchanged.
    """

    def __init__(self, parent_dialog, label: str, segments: list):
        super().__init__(parent_dialog)
        _style_toplevel(self)
        self.parent_dialog = parent_dialog
        self.label = label
        self.segments = [dict(s) for s in segments]  # working copy, chronological
        self.title(f"Edit sample - {label}")
        self.resizable(False, False)
        self.transient(parent_dialog)

        ttk.Label(
            self, text="Remove any part(s) you don't want in this speaker's preview "
                       "clip (e.g. cross-talk, noise, another speaker mixed in), then "
                       "rebuild and listen.",
            wraplength=380).pack(anchor="w", padx=10, pady=(10, 6))

        self._listbox = tk.Listbox(self, selectmode="extended", width=48, height=8)
        self._listbox.pack(padx=10, pady=(0, 6), fill="both", expand=True)
        self._reload_listbox()

        self._status_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._status_var, foreground="#666",
                 wraplength=380).pack(anchor="w", padx=10, pady=(0, 4))

        btn_row = ttk.Frame(self)
        btn_row.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(btn_row, text="Remove selected",
                  command=self._remove_selected).pack(side="left")
        ttk.Button(btn_row, text="Stop playback",
                  command=self.parent_dialog.controller.app._stop_audio).pack(side="left", padx=(6, 0))
        ttk.Button(btn_row, text="Close", command=self.destroy).pack(side="right")
        ttk.Button(btn_row, text="Rebuild && play",
                  command=self._rebuild_and_play).pack(side="right", padx=(0, 6))

    def _reload_listbox(self):
        self._listbox.delete(0, "end")
        for seg in self.segments:
            duration = seg["end"] - seg["start"]
            self._listbox.insert(
                "end", f"{seg['start']:.1f}s - {seg['end']:.1f}s   ({duration:.1f}s)")

    def _remove_selected(self):
        idxs = sorted(self._listbox.curselection(), reverse=True)
        if not idxs:
            return
        for i in idxs:
            del self.segments[i]
        self._reload_listbox()
        self._status_var.set(f"{len(self.segments)} part(s) remaining - "
                             f"click \"Rebuild && play\" to hear the result.")

    def _rebuild_and_play(self):
        if not self.segments:
            messagebox.showinfo(
                "Nothing left", "At least one part must remain - close this dialog "
                                "instead if you want to discard the whole sample.")
            return
        self._status_var.set("Rebuilding...")
        threading.Thread(target=self._rebuild_worker, daemon=True).start()

    def _rebuild_worker(self):
        try:
            clip_path = str(Path(self.parent_dialog._sample_dir) / f"{self.label}.wav")
            result = speaker_id.render_sample_clip(
                self.parent_dialog.source_media_path, self.segments, clip_path)
            if result:
                # Invalidate any stale enhanced version so a later "Improve
                # audio" click regenerates it from this edited clip rather
                # than replaying an enhancement of the old, unedited one.
                enhanced = Path(self.parent_dialog._sample_dir) / f"{self.label}_enhanced.wav"
                enhanced.unlink(missing_ok=True)
                self.parent_dialog._sample_segments[self.label] = [dict(s) for s in self.segments]
        except Exception as exc:
            # See SpeakerRenameDialog._play_sample_worker's except block:
            # capture the message now, since "except ... as exc" deletes
            # exc as soon as this block exits, before the deferred
            # self.after(0, ...) lambda ever runs.
            err_msg = str(exc)
            self.after(0, lambda: messagebox.showerror("Rebuild failed", err_msg))
            return
        if result:
            self.after(0, lambda: self.parent_dialog.controller.app._play_audio(Path(result)))
            self.after(0, lambda: self._status_var.set(
                f"Rebuilt from {len(self.segments)} part(s) and playing."))
        else:
            self.after(0, lambda: messagebox.showerror(
                "Rebuild failed", "ffmpeg extraction failed - see the log for details."))


class SpeakerRosterReviewDialog(tk.Toplevel):
    """
    Review & listen to a batch speaker roster JSON (V1.29, task #92
    export via run_pipeline.export_speaker_roster_json) before typing in
    names or applying it. Lists every speaker across every file in one
    scrollable Treeview (recording, label, voiceprint suggestion, sample
    availability, current name) instead of requiring the JSON to be
    hand-edited in a text editor and each .wav double-clicked in
    Explorer separately (still possible as a fallback, just no longer
    the only way).

    Sample clips were already extracted to disk by the export step (one
    "<recording>__<label>.wav" per speaker, path in each entry's
    "sample_clip") - this dialog only ever plays those existing files
    via App._play_audio (ffplay); it never re-renders anything, unlike
    SpeakerRenameDialog/SampleEditDialog which build clips on demand from
    a single file's segments.

    Workflow: select a row (or let "Auto-play on select" do it for you),
    listen, type the name in the Name field, press Enter - the name is
    committed into the loaded JSON structure in memory and the dialog
    advances to the next row automatically, so a whole batch can be
    named without touching the mouse beyond scrolling. "Next unnamed"
    jumps ahead to the next speaker whose name is still blank (or still
    just the raw label), skipping ones already reviewed. "Save" writes
    the edited roster back to its JSON file in the same structure
    apply_speaker_roster_json expects; "Save && Apply now" does that and
    then runs apply_speaker_roster_json immediately, in the background,
    in one step.
    """

    def __init__(self, app, json_path: str):
        super().__init__(app.root)
        _style_toplevel(self)
        self.app = app
        self.json_path = json_path
        self.title(f"Review Speaker Roster - {Path(json_path).name}")
        self.geometry("900x580")
        self.transient(app.root)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._dirty = False
        self._rows: list = []  # parallel to Treeview iids ("0", "1", ...): the actual
                                # speaker-entry dicts from self._payload, edited in place.

        try:
            self._payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not load roster", str(exc), parent=app.root)
            self.destroy()
            return

        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=(10, 4))
        ttk.Label(top, text=f"File: {Path(json_path).name}", foreground="#666").pack(side="left")
        self._status_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self._status_var, foreground="#666").pack(side="right")

        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill="both", expand=True, padx=10, pady=(0, 6))
        columns = ("recording", "label", "suggested", "sample", "name")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=18)
        headers = {
            "recording": ("Recording", 220), "label": ("Label", 100),
            "suggested": ("Suggested (score)", 170), "sample": ("Sample", 60),
            "name": ("Name", 190),
        }
        for col, (text, width) in headers.items():
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor="w")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.tree.tag_configure("named", foreground=_theme_color("primary_700"))
        self.tree.bind("<<TreeviewSelect>>", self._on_row_selected)
        self.tree.bind("<Double-1>", lambda e: self._play_selected())

        name_row = ttk.Frame(self)
        name_row.pack(fill="x", padx=10, pady=(0, 4))
        ttk.Label(name_row, text="Name:").pack(side="left")
        self.name_var = tk.StringVar(value="")
        self.name_entry = ttk.Entry(name_row, textvariable=self.name_var, width=32)
        self.name_entry.pack(side="left", padx=(6, 6))
        self.name_entry.bind("<Return>", lambda e: self._apply_and_advance())
        ttk.Button(name_row, text="Apply", command=self._apply_and_advance).pack(side="left")
        ttk.Button(name_row, text="Play sample", command=self._play_selected).pack(
            side="left", padx=(12, 0))
        ttk.Button(name_row, text="Stop playback", command=self.app._stop_audio).pack(
            side="left", padx=(6, 0))
        self.autoplay_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(name_row, text="Auto-play on select", variable=self.autoplay_var).pack(
            side="left", padx=(12, 0))

        nav_row = ttk.Frame(self)
        nav_row.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Button(nav_row, text="Next unnamed", command=self._select_next_unnamed).pack(side="left")
        ttk.Label(nav_row, text="Jumps to the next speaker whose name is still blank.",
                 foreground="#666").pack(side="left", padx=(8, 0))

        btn_row = ttk.Frame(self)
        btn_row.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(btn_row, text="Close", command=self._on_close).pack(side="right")
        ttk.Button(btn_row, text="Save", command=self._save).pack(side="right", padx=(0, 6))
        ttk.Button(btn_row, text="Save && Apply now", command=self._save_and_apply).pack(
            side="right", padx=(0, 6))

        self._reload_tree()
        if self._rows:
            self._select_index(0)

    # -----------------------------------------------------------------
    # Populating / refreshing the Treeview
    # -----------------------------------------------------------------

    def _reload_tree(self):
        self.tree.delete(*self.tree.get_children())
        self._rows = []
        for file_entry in self._payload.get("files", []):
            recording = (Path(file_entry.get("segments_json", "")).stem.replace("_segments", "")
                        or file_entry.get("output_prefix", ""))
            for speaker_entry in file_entry.get("speakers", []):
                iid = str(len(self._rows))
                self._rows.append(speaker_entry)
                self._insert_row(iid, recording, speaker_entry)
        self._update_status()

    def _insert_row(self, iid: str, recording: str, speaker_entry: dict):
        suggested = speaker_entry.get("suggested_name")
        score = speaker_entry.get("score")
        if suggested and score is not None:
            suggested_text = f"{suggested} ({score:.2f})"
        elif score is not None:
            suggested_text = f"(closest {score:.2f})"
        else:
            suggested_text = ""
        sample_text = "Yes" if speaker_entry.get("sample_clip") else "No"
        name = speaker_entry.get("name", "")
        is_named = bool(name) and name != speaker_entry.get("label")
        self.tree.insert("", "end", iid=iid, values=(
            recording, speaker_entry.get("label", ""), suggested_text, sample_text, name),
            tags=("named",) if is_named else ())

    def _refresh_row(self, idx: int):
        """Re-render one existing row's displayed values/tag from
        self._rows[idx] after an edit, without touching the others or
        losing the current scroll position/selection."""
        iid = str(idx)
        recording = self.tree.item(iid, "values")[0]
        entry = self._rows[idx]
        suggested = entry.get("suggested_name")
        score = entry.get("score")
        if suggested and score is not None:
            suggested_text = f"{suggested} ({score:.2f})"
        elif score is not None:
            suggested_text = f"(closest {score:.2f})"
        else:
            suggested_text = ""
        sample_text = "Yes" if entry.get("sample_clip") else "No"
        name = entry.get("name", "")
        is_named = bool(name) and name != entry.get("label")
        self.tree.item(iid, values=(
            recording, entry.get("label", ""), suggested_text, sample_text, name),
            tags=("named",) if is_named else ())

    def _update_status(self):
        total = len(self._rows)
        named = sum(1 for r in self._rows if r.get("name") and r.get("name") != r.get("label"))
        dirty_marker = "  (unsaved changes)" if self._dirty else ""
        self._status_var.set(f"{named}/{total} named{dirty_marker}")

    # -----------------------------------------------------------------
    # Selection / navigation / playback
    # -----------------------------------------------------------------

    def _current_index(self) -> int | None:
        sel = self.tree.selection()
        if not sel:
            return None
        return int(sel[0])

    def _on_row_selected(self, event=None):
        idx = self._current_index()
        if idx is None:
            return
        entry = self._rows[idx]
        self.name_var.set(entry.get("name", ""))
        if self.autoplay_var.get():
            self._play_selected()

    def _play_selected(self):
        idx = self._current_index()
        if idx is None:
            return
        entry = self._rows[idx]
        clip = entry.get("sample_clip")
        if not clip or not Path(clip).exists():
            self._status_var.set("No sample available for this speaker.")
            return
        self.app._play_audio(Path(clip))

    def _select_index(self, idx: int):
        if idx < 0 or idx >= len(self._rows):
            return
        iid = str(idx)
        self.tree.selection_set(iid)
        self.tree.see(iid)
        self.name_entry.focus_set()

    def _select_next_unnamed(self):
        start = (self._current_index() or -1) + 1
        search_order = list(range(start, len(self._rows))) + list(range(0, start))
        for idx in search_order:
            entry = self._rows[idx]
            name = entry.get("name", "")
            if not name or name == entry.get("label"):
                self._select_index(idx)
                return
        messagebox.showinfo("All named", "Every speaker already has a name.", parent=self)

    # -----------------------------------------------------------------
    # Editing / saving / applying
    # -----------------------------------------------------------------

    def _apply_and_advance(self):
        idx = self._current_index()
        if idx is None:
            return
        entry = self._rows[idx]
        new_name = self.name_var.get().strip()
        if new_name != entry.get("name", ""):
            entry["name"] = new_name
            self._dirty = True
        self._refresh_row(idx)
        self._update_status()
        self._select_index(idx + 1)

    def _save(self) -> bool:
        try:
            Path(self.json_path).write_text(
                json.dumps(self._payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Save failed", str(exc), parent=self)
            return False
        self._dirty = False
        self._update_status()
        return True

    def _save_and_apply(self):
        if not self._save():
            return
        regenerate_notes = messagebox.askyesno(
            "Regenerate notes?",
            "Also regenerate each file's notes/summary, for files where at least "
            "one speaker was actually renamed?", parent=self)
        self.app.root.config(cursor="watch")
        self.app.root.update()
        threading.Thread(
            target=self._apply_worker, args=(regenerate_notes,), daemon=True).start()

    def _apply_worker(self, regenerate_notes: bool):
        try:
            result = run_pipeline.apply_speaker_roster_json(
                self.json_path, regenerate_notes=regenerate_notes)
        except Exception as exc:
            # See SpeakerRenameDialog._play_sample_worker's except block:
            # capture the message now, since "except ... as exc" deletes
            # exc as soon as this block exits, before the deferred
            # self.after(0, ...) lambda ever runs.
            err_msg = str(exc)
            self.after(0, lambda: self._apply_done(error=err_msg))
            return
        self.after(0, lambda: self._apply_done(result=result))

    def _apply_done(self, result: dict | None = None, error: str | None = None):
        self.app.root.config(cursor="")
        if error:
            messagebox.showerror("Apply failed", error, parent=self)
            return
        msg = (f"{result['files_updated']} file(s) updated, "
               f"{result['speakers_renamed']} segment(s) renamed, "
               f"{result['roster_updates']} roster update(s).")
        if result["errors"]:
            details = "\n".join(f"- {e['file']}: {e['error']}" for e in result["errors"][:10])
            msg += f"\n\n{len(result['errors'])} file(s) failed:\n{details}"
        messagebox.showinfo("Speaker roster applied", msg, parent=self)

    def _on_close(self):
        self.app._stop_audio()
        if self._dirty:
            proceed = messagebox.askyesnocancel(
                "Unsaved changes", "Save changes to the roster JSON before closing?",
                parent=self)
            if proceed is None:
                return
            if proceed and not self._save():
                return
        self.destroy()


class SlideReviewDialog(tk.Toplevel):
    """
    Modal window (task #58, video tab only): review the slides detected
    in a run, mark any to omit or merge with the next slide, and rebuild
    the CSV/HTML/PDF/timing-summary reports from those edits without
    re-running transcription/detection/VLM annotation. Works for a
    single video, or for switching between multiple videos from the same
    batch run via the "recently loaded" dropdown. Edits are saved to a
    separate "<stem>_slides_edits.json" next to the original
    *_slides.json, which is never modified - always safe to reload and
    start over. See run_pipeline.apply_slide_edits / rebuild_outputs for
    the edit semantics and rebuild logic.

    Includes a preview panel (task #60): selecting a row shows that
    slide's snapshot image plus title/bullets/transcript, so slides can
    be identified visually before deciding to omit/merge them.

    Task #64: every video's slides loaded this session are kept in
    memory (self._slides_by_video / self._edits_by_video), so switching
    between them via the dropdown never loses in-progress edits, and the
    "Batch overview (all loaded)" checkbox combines all of them into one
    table with a Video column. Also adds "Edit Title..." for a
    non-destructive per-slide title override (edits schema extended with
    title_overrides, see run_pipeline.apply_slide_edits).

    Task #78: "Thumbnail grid view" toggles the list (Treeview) between a
    scrollable grid of slide snapshot thumbnails with a caption per cell
    (video/number/time/title/status), sharing the same underlying
    self._row_refs and edit dicts as the list view - Toggle Omit/Merge/
    Edit Title/Rebuild all keep working unchanged in either view, via
    _selected_refs() reading whichever view is active. Thumbnails are
    decoded lazily: only when grid view is actually switched on, and only
    once per snapshot path (cached in self._grid_photos) - a large batch
    overview never decodes hundreds of images just from opening the
    dialog or staying in list view.
    """

    def __init__(self, master, overrides_source):
        super().__init__(master)
        _style_toplevel(self)
        self.title("Review / Edit Slides")
        self.geometry("1180x560")
        self._overrides_source = overrides_source  # RunTabController (video), for meeting_title/etc.
        self._videos: list = []                # every *_slides.json path loaded this session
        self._slides_by_video: dict = {}        # path -> list of slide dicts
        self._edits_by_video: dict = {}         # path -> {"omit_indices", "merge_next_indices", "title_overrides"}
        self._current_video: str | None = None
        self._batch_mode: bool = False
        self._row_refs: list = []               # tree row position -> (video, slide_idx)
        self._preview_photo = None  # keep a reference, else Tk garbage-collects the image

        # Thumbnail grid view state (task #78).
        self._view_mode_grid: bool = False
        self._grid_photos: dict = {}    # snapshot_path -> PhotoImage cache, never evicted this session
        self._grid_selected: set = set()  # selected indices into self._row_refs, grid mode only
        self._grid_frames: dict = {}    # row index -> its cell Frame, for selection highlighting

        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=8)
        ttk.Button(top, text="Load *_slides.json...", command=self._load_file).pack(side="left")
        self.recent_var = tk.StringVar()
        self.recent_box = ttk.Combobox(top, textvariable=self.recent_var, state="readonly", width=42)
        self.recent_box.pack(side="left", padx=(8, 0))
        self.recent_box.bind("<<ComboboxSelected>>", lambda e: self._switch_to_recent())
        self.batch_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Batch overview (all loaded)", variable=self.batch_mode_var,
                       command=self._toggle_batch_view).pack(side="left", padx=(12, 0))
        self.view_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Thumbnail grid view", variable=self.view_mode_var,
                       command=self._toggle_view_mode).pack(side="left", padx=(12, 0))
        ttk.Label(top, text="Switch between videos loaded this session (batch runs).",
                 foreground="#666").pack(side="left", padx=(8, 0))

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        tree_frame = ttk.Frame(body)
        tree_frame.pack(side="left", fill="both", expand=True)
        self.tree = ttk.Treeview(
            tree_frame, columns=("video", "num", "time", "title", "bullets", "status"),
            show="headings", selectmode="extended", height=16)
        for col, text, w in (("video", "Video", 140), ("num", "#", 40), ("time", "Time", 70),
                             ("title", "Title", 160), ("bullets", "First bullet", 220),
                             ("status", "Status", 130)):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=w, anchor="w")
        self.tree["displaycolumns"] = ("num", "time", "title", "bullets", "status")  # video hidden until batch mode
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # Thumbnail grid view (task #78): built into the same tree_frame,
        # but not packed until "Thumbnail grid view" is checked, so its
        # canvas/scrollbar simply don't exist on screen (and no image is
        # decoded) in the default list view.
        self.grid_canvas = tk.Canvas(tree_frame, highlightthickness=0, bg=_theme_color("neutral_100"))
        self.grid_vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.grid_canvas.yview)
        self.grid_canvas.configure(yscrollcommand=self.grid_vsb.set)
        self.grid_inner = ttk.Frame(self.grid_canvas)
        self._grid_inner_id = self.grid_canvas.create_window((0, 0), window=self.grid_inner, anchor="nw")

        def _on_grid_inner_configure(event):
            self.grid_canvas.configure(scrollregion=self.grid_canvas.bbox("all"))

        def _on_grid_canvas_configure(event):
            self.grid_canvas.itemconfig(self._grid_inner_id, width=event.width)

        self.grid_inner.bind("<Configure>", _on_grid_inner_configure)
        self.grid_canvas.bind("<Configure>", _on_grid_canvas_configure)

        def _on_grid_mousewheel(event):
            self.grid_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.grid_canvas.bind("<Enter>", lambda e: self.grid_canvas.bind_all(
            "<MouseWheel>", _on_grid_mousewheel))
        self.grid_canvas.bind("<Leave>", lambda e: self.grid_canvas.unbind_all("<MouseWheel>"))

        # Preview panel: snapshot image + title + bullets/transcript for
        # whichever row was selected last, so slides can be identified
        # visually before deciding to omit/merge them.
        preview = ttk.Frame(body, width=360)
        preview.pack(side="right", fill="y", padx=(10, 0))
        preview.pack_propagate(False)
        ttk.Label(preview, text="Preview", font=("", 9, "bold")).pack(anchor="w")
        self.preview_image_label = ttk.Label(preview, text="(select a slide to preview)",
                                             foreground="#666", anchor="center",
                                             justify="center", relief="groove")
        self.preview_image_label.pack(fill="x", pady=(4, 6), ipady=40)
        self.preview_title_var = tk.StringVar()
        ttk.Label(preview, textvariable=self.preview_title_var, font=("", 10, "bold"),
                 wraplength=340, justify="left").pack(fill="x", anchor="w")
        self.preview_text = tk.Text(preview, width=42, wrap="word", state="disabled",
                                    relief="flat", bg=self.cget("background"))
        self.preview_text.pack(fill="both", expand=True, pady=(6, 0))

        btn_row = ttk.Frame(self)
        btn_row.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(btn_row, text="Toggle Omit", command=self._toggle_omit).pack(side="left")
        ttk.Button(btn_row, text="Toggle Merge with next", command=self._toggle_merge).pack(
            side="left", padx=(6, 0))
        ttk.Button(btn_row, text="Edit Title...", command=self._edit_title).pack(side="left", padx=(6, 0))
        ttk.Button(btn_row, text="Clear all edits", command=self._clear_edits).pack(side="left", padx=(6, 0))
        ttk.Button(btn_row, text="Rebuild outputs", command=self._rebuild).pack(side="right")
        self.status_label = ttk.Label(btn_row, text="")
        self.status_label.pack(side="right", padx=(0, 12))

        ttk.Label(self, text="Select one or more rows, then Toggle Omit / Toggle Merge with next / "
                            "Edit Title. \"Batch overview\" combines every video loaded this session; "
                            "\"Rebuild outputs\" regenerates CSV/HTML/PDF/timing summary for the "
                            "selected (or last loaded) video only - no re-transcription, "
                            "re-detection, or VLM calls.",
                 foreground="#666", wraplength=860).pack(fill="x", padx=8, pady=(0, 8))

    def _blank_edits(self) -> dict:
        return {"omit_indices": set(), "merge_next_indices": set(), "title_overrides": {}}

    def _edits_path_for(self, path: str) -> Path:
        p = Path(path)
        stem = p.stem.replace("_slides", "")
        return p.parent / f"{stem}_slides_edits.json"

    def _load_file(self):
        path = filedialog.askopenfilename(
            title="Select *_slides.json",
            filetypes=[("Slides JSON", "*_slides.json"), ("All files", "*.*")])
        if not path:
            return
        self._load_path(path)

    def _switch_to_recent(self):
        """Switch the active video. If it was already loaded this
        session, just reactivate it (keeps any in-progress, not-yet-
        rebuilt edits); otherwise load it fresh from disk."""
        path = self.recent_var.get()
        if not path:
            return
        if path in self._slides_by_video:
            self._current_video = path
            self._refresh_tree()
            self.status_label.config(text=f"{len(self._slides_by_video[path])} slide(s) loaded.")
        else:
            self._load_path(path, remember=False)

    def _toggle_batch_view(self):
        self._batch_mode = self.batch_mode_var.get()
        if self._batch_mode:
            self.tree["displaycolumns"] = ("video", "num", "time", "title", "bullets", "status")
        else:
            self.tree["displaycolumns"] = ("num", "time", "title", "bullets", "status")
        self._refresh_tree()

    def _toggle_view_mode(self):
        """Switch between the list (Treeview) and thumbnail grid views
        (task #78). Swaps which widget is packed into tree_frame; the
        underlying data (self._row_refs, self._slides_by_video,
        self._edits_by_video) is shared, so edits made in one view are
        immediately visible after switching to the other."""
        self._view_mode_grid = self.view_mode_var.get()
        if self._view_mode_grid:
            self.tree.pack_forget()
            self.grid_canvas.pack(side="left", fill="both", expand=True)
            self.grid_vsb.pack(side="right", fill="y")
        else:
            self.grid_canvas.pack_forget()
            self.grid_vsb.pack_forget()
            self.tree.pack(fill="both", expand=True)
        self._refresh_tree()

    def _load_path(self, path: str, remember: bool = True):
        try:
            slides = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            messagebox.showerror("Could not load", str(exc))
            return
        self._slides_by_video[path] = slides
        self._current_video = path

        # Load any previously saved edits for this file, so re-opening
        # the review picks up where you left off.
        edits = self._blank_edits()
        edits_path = self._edits_path_for(path)
        if edits_path.exists():
            try:
                saved = json.loads(edits_path.read_text(encoding="utf-8"))
                edits["omit_indices"] = set(saved.get("omit_indices", []) or [])
                edits["merge_next_indices"] = set(saved.get("merge_next_indices", []) or [])
                edits["title_overrides"] = dict(saved.get("title_overrides", {}) or {})
            except Exception:
                pass
        self._edits_by_video[path] = edits

        if remember and path not in self._videos:
            self._videos.append(path)
            self.recent_box["values"] = self._videos
        self.recent_var.set(path)
        self._refresh_tree()
        self.status_label.config(
            text=f"{len(slides)} slide(s) loaded ({len(self._videos)} video(s) total this session).")

    def _refresh_tree(self):
        self.tree.delete(*self.tree.get_children())
        self._row_refs = []
        self._grid_selected = set()
        self._show_preview(None)
        videos = self._videos if self._batch_mode else ([self._current_video] if self._current_video else [])
        for video in videos:
            slides = self._slides_by_video.get(video, [])
            edits = self._edits_by_video.get(video) or self._blank_edits()
            omit = edits["omit_indices"]
            merge_next = edits["merge_next_indices"]
            title_overrides = edits["title_overrides"]
            video_label = Path(video).stem
            for idx, slide in enumerate(slides):
                bullets = slide.get("bullets") or []
                first_bullet = bullets[0] if bullets else ""
                title = title_overrides.get(str(idx), slide.get("title", ""))
                status_parts = []
                if idx in omit:
                    status_parts.append("OMIT")
                if idx in merge_next:
                    status_parts.append("MERGE->next")
                if str(idx) in title_overrides:
                    status_parts.append("TITLE EDITED")
                row_id = str(len(self._row_refs))
                self._row_refs.append((video, idx))
                self.tree.insert("", "end", iid=row_id, values=(
                    video_label, idx + 1, f"{slide.get('timestamp_sec', 0):.1f}s",
                    title, first_bullet, " + ".join(status_parts)))
        if self._view_mode_grid:
            self._refresh_grid()

    def _refresh_grid(self):
        """Rebuild the thumbnail grid (task #78) from self._row_refs -
        the same data the list view's rows come from, computed just above
        in _refresh_tree. Only called while grid view is active: staying
        in (or switching back to) list view never runs this, so a large
        batch overview never decodes images it isn't showing. Thumbnails
        that were already decoded this session are reused from
        self._grid_photos rather than re-read from disk."""
        for child in self.grid_inner.winfo_children():
            child.destroy()
        self._grid_frames = {}
        cols = 4
        for i, (video, idx) in enumerate(self._row_refs):
            slides = self._slides_by_video.get(video, [])
            if idx >= len(slides):
                continue
            slide = slides[idx]
            edits = self._edits_by_video.get(video) or self._blank_edits()
            title_overrides = edits["title_overrides"]
            title = title_overrides.get(str(idx), slide.get("title", "") or "(no title)")
            status_parts = []
            if idx in edits["omit_indices"]:
                status_parts.append("OMIT")
            if idx in edits["merge_next_indices"]:
                status_parts.append("MERGE->next")
            if str(idx) in title_overrides:
                status_parts.append("EDITED")

            cell = ttk.Frame(self.grid_inner, relief="groove", borderwidth=2, padding=4)
            cell.grid(row=i // cols, column=i % cols, padx=4, pady=4, sticky="n")

            photo = self._grid_thumbnail(slide.get("snapshot_path", ""))
            if photo is not None:
                img_label = ttk.Label(cell, image=photo)
            else:
                img_label = ttk.Label(cell, text="(no image)", width=20, anchor="center",
                                      relief="flat")
            img_label.pack()

            caption = f"#{idx + 1}   {slide.get('timestamp_sec', 0):.1f}s"
            if self._batch_mode:
                caption = f"{Path(video).stem}\n{caption}"
            caption += f"\n{title[:44]}"
            if status_parts:
                caption += f"\n[{' + '.join(status_parts)}]"
            cap_label = ttk.Label(cell, text=caption, wraplength=160, justify="center",
                                  font=("", 8))
            cap_label.pack()

            for widget in (cell, img_label, cap_label):
                widget.bind("<Button-1>", lambda e, row=i: self._on_grid_click(row, e))

            self._grid_frames[i] = cell
        self._apply_grid_highlight()

    def _grid_thumbnail(self, snapshot_path: str):
        """Return a cached PhotoImage for snapshot_path, decoding and
        caching it on first use. Returns None if there is no path, the
        file is missing, or it fails to decode (caller shows a text
        placeholder instead)."""
        if not snapshot_path:
            return None
        if snapshot_path in self._grid_photos:
            return self._grid_photos[snapshot_path]
        if not Path(snapshot_path).exists():
            return None
        try:
            with Image.open(snapshot_path) as img:
                img = img.copy()
            img.thumbnail((160, 120))
            photo = ImageTk.PhotoImage(img)
        except Exception:
            return None
        self._grid_photos[snapshot_path] = photo
        return photo

    def _on_grid_click(self, row: int, event=None):
        """Plain click selects only this cell (matches a fresh Treeview
        click); Ctrl-click toggles it into/out of a multi-selection, same
        spirit as extended Treeview selection. Also drives the preview
        panel, same as clicking a list row."""
        ctrl_held = bool(event is not None and (event.state & 0x4))
        if ctrl_held:
            if row in self._grid_selected:
                self._grid_selected.discard(row)
            else:
                self._grid_selected.add(row)
        else:
            self._grid_selected = {row}
        self._apply_grid_highlight()
        if row < len(self._row_refs):
            video, idx = self._row_refs[row]
            slides = self._slides_by_video.get(video, [])
            self._show_preview(slides[idx] if idx < len(slides) else None)

    def _apply_grid_highlight(self):
        for row, frame in self._grid_frames.items():
            frame.configure(relief="solid" if row in self._grid_selected else "groove")

    def _selected_refs(self) -> list:
        """Selected rows as (video_path, slide_index) pairs, valid in
        both single-video and batch-overview mode, and in both the list
        and thumbnail grid views (task #78)."""
        if self._view_mode_grid:
            return [self._row_refs[i] for i in sorted(self._grid_selected) if i < len(self._row_refs)]
        refs = []
        for iid in self.tree.selection():
            i = int(iid)
            if i < len(self._row_refs):
                refs.append(self._row_refs[i])
        return refs

    def _on_select(self, event=None):
        refs = self._selected_refs()
        if not refs:
            self._show_preview(None)
            return
        video, idx = refs[0]
        slides = self._slides_by_video.get(video, [])
        if idx < len(slides):
            self._show_preview(slides[idx])
        else:
            self._show_preview(None)

    def _show_preview(self, slide: dict | None):
        self._preview_photo = None
        if slide is None:
            self.preview_image_label.config(image="", text="(select a slide to preview)")
            self.preview_title_var.set("")
            self._set_preview_text("")
            return

        snap = slide.get("snapshot_path", "")
        loaded = False
        if snap and Path(snap).exists():
            try:
                with Image.open(snap) as img:
                    img = img.copy()
                img.thumbnail((340, 260))
                self._preview_photo = ImageTk.PhotoImage(img)
                self.preview_image_label.config(image=self._preview_photo, text="")
                loaded = True
            except Exception:
                loaded = False
        if not loaded:
            self.preview_image_label.config(image="", text="(snapshot image not found)")

        self.preview_title_var.set(
            f"{slide.get('timestamp_sec', 0):.1f}s   {slide.get('title', '(no title)')}")
        bullets = slide.get("bullets") or []
        body = "\n".join(f"- {b}" for b in bullets)
        transcript = slide.get("transcript_seg", "")
        if transcript:
            body += ("\n\n" if body else "") + "Transcript:\n" + transcript
        self._set_preview_text(body)

    def _set_preview_text(self, text: str):
        self.preview_text.config(state="normal")
        self.preview_text.delete("1.0", "end")
        self.preview_text.insert("1.0", text)
        self.preview_text.config(state="disabled")

    def _toggle_omit(self):
        refs = self._selected_refs()
        if not refs:
            return
        for video, idx in refs:
            edits = self._edits_by_video.setdefault(video, self._blank_edits())
            if idx in edits["omit_indices"]:
                edits["omit_indices"].discard(idx)
            else:
                edits["omit_indices"].add(idx)
        self._refresh_tree()

    def _toggle_merge(self):
        refs = self._selected_refs()
        if not refs:
            return
        for video, idx in refs:
            slides = self._slides_by_video.get(video, [])
            if idx >= len(slides) - 1:
                continue  # last slide can't merge with a next one
            edits = self._edits_by_video.setdefault(video, self._blank_edits())
            if idx in edits["merge_next_indices"]:
                edits["merge_next_indices"].discard(idx)
            else:
                edits["merge_next_indices"].add(idx)
        self._refresh_tree()

    def _edit_title(self):
        """Task #64: store a non-destructive title override for one
        slide (extends the edits schema, see run_pipeline.apply_slide_edits).
        The original *_slides.json is never touched; the override is only
        written out when "Rebuild outputs" saves <stem>_slides_edits.json."""
        refs = self._selected_refs()
        if not refs:
            messagebox.showinfo("No selection", "Select a slide first.")
            return
        if len(refs) > 1:
            messagebox.showinfo("One at a time", "Select a single slide to edit its title.")
            return
        video, idx = refs[0]
        slides = self._slides_by_video.get(video, [])
        if idx >= len(slides):
            return
        edits = self._edits_by_video.setdefault(video, self._blank_edits())
        current = edits["title_overrides"].get(str(idx), slides[idx].get("title", ""))
        new_title = simpledialog.askstring("Edit Title", "New title:", initialvalue=current, parent=self)
        if new_title is None:
            return
        edits["title_overrides"][str(idx)] = new_title
        self._refresh_tree()

    def _clear_edits(self):
        """Clears omit/merge/title edits for the currently active video
        only (the one last loaded or switched to via the dropdown)."""
        if not self._current_video:
            return
        self._edits_by_video[self._current_video] = self._blank_edits()
        self._refresh_tree()

    def _rebuild(self):
        """Rebuilds outputs for the video of the current selection, or
        the currently active video if nothing is selected. In batch
        overview mode, select a row from the video you want to rebuild."""
        target = self._current_video
        refs = self._selected_refs()
        if refs:
            target = refs[0][0]
        if not target:
            messagebox.showwarning("No file loaded", "Load a *_slides.json first.")
            return
        edits = self._edits_by_video.get(target) or self._blank_edits()
        edits_out = {
            "omit_indices": sorted(edits["omit_indices"]),
            "merge_next_indices": sorted(edits["merge_next_indices"]),
            "title_overrides": dict(edits["title_overrides"]),
        }
        overrides = {}
        if hasattr(self._overrides_source, "_build_overrides"):
            try:
                overrides = self._overrides_source._build_overrides()
            except Exception:
                overrides = {}
        try:
            ok = run_pipeline.rebuild_outputs(target, edits_out, overrides)
        except Exception as exc:
            messagebox.showerror("Rebuild failed", str(exc))
            return
        if ok:
            messagebox.showinfo("Done", f"Outputs rebuilt in:\n{Path(target).parent}")
            self.status_label.config(text=f"Rebuild complete: {Path(target).name}")
        else:
            messagebox.showerror("Rebuild failed", "See log for details.")


def launch():
    root = tk.Tk()
    apply_theme(root)
    PipelineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
