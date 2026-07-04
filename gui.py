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
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk

from PIL import Image, ImageTk

from config import CONFIG, PROMPTS_DIR, SUPPORTED_EXTENSIONS
import db
import gui_logic
import ics_utils
import keys_loader
import notes as notes_mod
import run_pipeline

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
LANGUAGES = ["auto", "en", "de", "fr", "es", "it", "ja", "zh", "nl", "uk", "pt"]
LLM_BACKENDS = ["ollama", "anthropic"]
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]
MODES = ["Single file", "File list (batch)", "Folder (auto-discover)",
        "Notes-only (existing transcript)", "Notes-batch (folder)"]

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
        self.root.geometry("980x860")

        self.req_queue: queue.Queue = queue.Queue()
        self._notes_result_paths: dict = {}
        self._slides_result_paths: dict = {}
        self._last_check_results: list = []

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)
        self.meeting_tab = ttk.Frame(notebook)
        self.video_tab = ttk.Frame(notebook)
        self.search_tab = ttk.Frame(notebook)
        self.settings_tab = ttk.Frame(notebook)
        notebook.add(self.meeting_tab, text="Meeting")
        notebook.add(self.video_tab, text="Video / Webinar")
        notebook.add(self.search_tab, text="Search")
        notebook.add(self.settings_tab, text="Settings")

        self._build_search_tab()
        self._build_settings_tab()
        self.meeting_run = RunTabController(self.meeting_tab, "meeting", self)
        self.video_run = RunTabController(self.video_tab, "video", self)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(200, self._poll_req_queue)

    def _on_close(self):
        """Persist both tabs' settings to gui_state.json (task #71)
        before closing: whisper model, language, LLM backend, prompt
        template, thresholds, output folder, and stage checkboxes. Never
        blocks shutdown - save_state_file swallows I/O errors."""
        state = {
            "meeting": self.meeting_run._collect_persisted_state(),
            "video": self.video_run._collect_persisted_state(),
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
        canvas = tk.Canvas(parent, highlightthickness=0)
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
        self.req_text = scrolledtext.ScrolledText(req_frame, height=9, state="disabled", wrap="word")
        self.req_text.pack(fill="x", padx=8, pady=(0, 8))

        # --- Database ---
        db_frame = ttk.LabelFrame(parent, text="Database")
        db_frame.pack(fill="x", **pad)
        self.db_path_var = tk.StringVar(value=CONFIG["db_path"])
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
        self.db_stats_label = ttk.Label(db_frame, text="", foreground="#444")
        self.db_stats_label.grid(row=2, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 8))

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


class RunTabController:
    """
    Self-contained controller for one Run-style tab (Meeting or Video /
    Webinar). Each instance owns its own Mode/path/output fields, batch
    table, parameters, stages, formats, Run/Stop button, and log panel,
    built into the ttk.Frame handed to it by PipelineGUI - one instance
    per tab, so switching tabs never mixes up settings between a meeting
    run and a video/webinar run.

    kind is "meeting" or "video":
      meeting: no slide detection/VLM/Q&A/slide-report formats/Review-
               Edit-Slides/Re-annotate-failed-slides - a plain transcript
               plus LLM summary only.
      video:   everything, including slide detection, VLM annotation,
               the Q&A section (task #61), slide-report formats, and the
               Review/Edit Slides + Re-annotate failed slides buttons.

    app is the owning PipelineGUI instance, used only for the handful of
    things that stay shared across every tab: the database path field
    and logging options (both on the Settings tab), and _make_scrollable/
    _open_prompts_folder/_open_path.
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
        self.whisper_model_var = tk.StringVar(value=CONFIG["whisper_model"])
        self.language_var = tk.StringVar(value=CONFIG["whisper_language"])
        self.prompt_var = tk.StringVar()
        self.llm_backend_var = tk.StringVar(value=CONFIG["llm_backend"])

        self.enable_diarization_var = tk.BooleanVar(value=CONFIG["enable_diarization"])
        self.no_summary_var = tk.BooleanVar(value=False)
        self.force_retranscribe_var = tk.BooleanVar(value=False)

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

        ttk.Label(top, text="Output filename (optional):").grid(row=4, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.output_basename_var, width=60).grid(row=4, column=1, sticky="we")
        ttk.Label(top, text="Blank = use the source filename for every output file (default).",
                 foreground="#666").grid(row=5, column=1, sticky="w")
        top.columnconfigure(1, weight=1)

        # --- Recording type preset ---
        rec_types = VIDEO_RECORDING_TYPES if is_video else MEETING_RECORDING_TYPES
        self._recording_types = rec_types
        self.recording_type_var = tk.StringVar(value=next(iter(rec_types)))
        rec_frame = ttk.LabelFrame(parent, text="Recording type (preset)")
        rec_frame.pack(fill="x", padx=8, pady=4)
        self._rec_frame = rec_frame
        ttk.Label(rec_frame, text="Type:").pack(side="left", padx=8, pady=8)
        rec_box = ttk.Combobox(rec_frame, textvariable=self.recording_type_var,
                               values=list(rec_types.keys()), state="readonly", width=22)
        rec_box.pack(side="left", pady=8)
        rec_box.bind("<<ComboboxSelected>>", lambda e: self._apply_recording_type())
        ttk.Label(rec_frame, text="Sets the prompt template below; still editable afterward.",
                 foreground="#666").pack(side="left", padx=(12, 0))
        other_tab_label = "Video/Webinar" if self.kind == "meeting" else "Meeting"
        ttk.Button(rec_frame, text=f"Copy settings to {other_tab_label} tab...",
                  command=self._copy_settings_to_other_tab).pack(side="right", padx=8)

        # --- Meeting info (optional) ---
        meeting_frame = ttk.LabelFrame(parent, text="Meeting info (optional)")
        self._meeting_frame = meeting_frame
        self.meeting_title_var = tk.StringVar()
        self.meeting_date_var = tk.StringVar()
        self.meeting_comments_var = tk.StringVar()
        ttk.Label(meeting_frame, text="Title:").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(meeting_frame, textvariable=self.meeting_title_var, width=45).grid(
            row=0, column=1, sticky="w", columnspan=2)
        ttk.Label(meeting_frame, text="Date (YYYY-MM-DD):").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(meeting_frame, textvariable=self.meeting_date_var, width=16).grid(row=1, column=1, sticky="w")
        ttk.Button(meeting_frame, text="Load from .ics...", command=self._load_ics).grid(
            row=1, column=2, sticky="w", padx=(8, 0))
        ttk.Label(meeting_frame, text="Comments:").grid(row=2, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(meeting_frame, textvariable=self.meeting_comments_var, width=60).grid(
            row=2, column=1, sticky="we", columnspan=2)
        ttk.Label(meeting_frame,
                 text="Pre-fills Title/Date from an Outlook/Google/Teams .ics invite. Shown in the "
                      "notes header" + (" and used as the report title." if is_video else "."),
                 foreground="#666").grid(row=3, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 4))
        if is_video:
            ttk.Label(meeting_frame, text="Title slide image:").grid(row=4, column=0, sticky="w", padx=8, pady=4)
            ttk.Entry(meeting_frame, textvariable=self.title_slide_image_var, width=45).grid(
                row=4, column=1, sticky="w")
            ttk.Button(meeting_frame, text="Browse...", command=self._browse_title_slide_image).grid(
                row=4, column=2, sticky="w", padx=(8, 0))
            ttk.Label(meeting_frame,
                     text="Optional cover image (jpg/png) shown as a title page before Slide 1 "
                          "in the HTML/PDF slide report.",
                     foreground="#666").grid(row=5, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 4))

        # --- Q&A section (video only, task #61) ---
        self._qa_frame = None
        if is_video:
            qa_frame = ttk.LabelFrame(parent, text="Q&A section (optional)")
            self._qa_frame = qa_frame
            self.qa_start_var = tk.StringVar()
            self.qa_end_var = tk.StringVar()
            ttk.Label(qa_frame, text="Starts at:").grid(row=0, column=0, sticky="w", padx=8, pady=4)
            ttk.Entry(qa_frame, textvariable=self.qa_start_var, width=12).grid(row=0, column=1, sticky="w")
            ttk.Label(qa_frame, text="Ends at (optional):").grid(row=0, column=2, sticky="w", padx=(16, 4), pady=4)
            ttk.Entry(qa_frame, textvariable=self.qa_end_var, width=12).grid(row=0, column=3, sticky="w")
            ttk.Label(qa_frame,
                     text="For webinars with a Q&A block at the end where only the speakers are shown. "
                          "If set, no new slides are detected from here onward (to Ends at, or the end "
                          "of the recording), and this segment gets its own focused Q&A summary "
                          "(prompts/qa_summary.md) appended to the notes instead of the main summary. "
                          "Blank Starts at = disabled. Accepts seconds (975), mm:ss (16:15), or hh:mm:ss.",
                     foreground="#666", wraplength=860).grid(
                row=1, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 6))

        # --- Batch table (task #62) ---
        self.batch_frame = ttk.LabelFrame(
            parent, text="Batch list: add files/folder, or load a .txt/.csv - double-click a cell to edit")
        if is_video:
            self.batch_columns = ("file", "language", "title", "date", "comments", "qa_start", "qa_end")
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
                      "status": "Status"}
        _all_widths = {"file": 300, "language": 55, "title": 140, "date": 85,
                      "comments": 160, "qa_start": 75, "qa_end": 75, "status": 90}
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
        qa_note = " Q&A start/end accept seconds (975), mm:ss (16:15), or hh:mm:ss." if is_video else ""
        ttk.Label(self.batch_frame,
                 text="Language/Title/Date/Comments" + (" /Q&A start/end" if is_video else "") +
                      " are optional per row: blank uses the shared fields above." + qa_note +
                      " Double-click a cell to edit it in place.",
                 foreground="#666", wraplength=860).pack(fill="x", padx=8, pady=(0, 8))
        # Meeting info/Q&A/batch table visibility is driven by Mode - see
        # _update_path_label(), called once at the end of _build() below.

        # --- Parameters ---
        params = ttk.LabelFrame(parent, text="Parameters")
        params.pack(fill="x", **pad)

        ttk.Label(params, text="Whisper model:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Combobox(params, textvariable=self.whisper_model_var, values=WHISPER_MODELS,
                    state="readonly", width=14).grid(row=0, column=1, sticky="w")

        ttk.Label(params, text="Language:").grid(row=0, column=2, sticky="w", **pad)
        ttk.Combobox(params, textvariable=self.language_var, values=LANGUAGES,
                    state="readonly", width=10).grid(row=0, column=3, sticky="w")

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
        toggles = ttk.LabelFrame(parent, text="Stages")
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

        # --- Output formats ---
        formats = ttk.LabelFrame(parent, text="Output formats")
        formats.pack(fill="x", **pad)
        ttk.Label(formats, text="Notes:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Checkbutton(formats, text="txt", variable=self.notes_txt_var).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(formats, text="html", variable=self.notes_html_var).grid(row=0, column=2, sticky="w")
        ttk.Checkbutton(formats, text="pdf", variable=self.notes_pdf_var).grid(row=0, column=3, sticky="w")
        ttk.Checkbutton(formats, text="docx", variable=self.notes_docx_var).grid(row=0, column=4, sticky="w")

        if is_video:
            ttk.Label(formats, text="Slide report:").grid(row=1, column=0, sticky="w", **pad)
            ttk.Checkbutton(formats, text="html", variable=self.report_html_var).grid(row=1, column=1, sticky="w")
            ttk.Checkbutton(formats, text="csv", variable=self.report_csv_var).grid(row=1, column=2, sticky="w")
            ttk.Checkbutton(formats, text="json", variable=self.report_json_var).grid(row=1, column=3, sticky="w")
            ttk.Checkbutton(formats, text="pdf", variable=self.report_pdf_var).grid(row=1, column=4, sticky="w")
            ttk.Checkbutton(formats, text="srt", variable=self.report_srt_var).grid(row=1, column=5, sticky="w")
            ttk.Checkbutton(formats, text="timing summary (txt)",
                            variable=self.report_slide_timing_var).grid(row=1, column=6, sticky="w")
            ttk.Checkbutton(formats, text="zip snapshots",
                            variable=self.zip_snapshots_var).grid(row=1, column=7, sticky="w")
            ttk.Button(formats, text="Zip an existing snapshots folder...",
                      command=self._zip_existing_snapshots).grid(
                row=2, column=0, columnspan=4, sticky="w", padx=8, pady=(4, 0))

            ttk.Label(formats, text="Report content:").grid(row=3, column=0, sticky="w", **pad)
            ttk.Checkbutton(formats, text="image", variable=self.report_show_image_var).grid(
                row=3, column=1, sticky="w")
            ttk.Checkbutton(formats, text="bullet points", variable=self.report_show_bullets_var).grid(
                row=3, column=2, sticky="w")
            ttk.Checkbutton(formats, text="transcript", variable=self.report_show_transcript_var).grid(
                row=3, column=3, sticky="w")
            ttk.Label(formats, text="Transcript:").grid(row=4, column=0, sticky="w", **pad)
            ttk.Combobox(formats, textvariable=self.report_transcript_mode_var,
                        values=["full", "first_sentence"], state="readonly", width=16).grid(
                row=4, column=1, sticky="w", columnspan=2)
            ttk.Label(formats, text="'full' = complete transcript segment per slide. "
                                  "'first_sentence' = first sentence only, so it fits on one page.",
                     foreground="#666").grid(row=5, column=0, columnspan=6, sticky="w", padx=8, pady=(0, 4))

        # --- Run button + progress ---
        run_frame = ttk.Frame(parent)
        run_frame.pack(fill="x", **pad)
        self.run_button = ttk.Button(run_frame, text="Run", command=self._on_run)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(run_frame, text="Stop", command=self._on_stop, state="disabled")
        self.stop_button.pack(side="left", padx=(6, 0))
        if is_video:
            ttk.Button(run_frame, text="Re-annotate failed slides...",
                      command=self._on_reannotate_failed).pack(side="left", padx=(6, 0))
        ttk.Button(run_frame, text="Open in LosslessCut...",
                  command=self._open_in_losslesscut).pack(side="left", padx=(6, 0))
        if is_video:
            ttk.Button(run_frame, text="Review / Edit Slides...",
                      command=self._open_slide_review).pack(side="left", padx=(6, 0))
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
        self.log_text = scrolledtext.ScrolledText(log_frame, state="disabled", height=16, wrap="word")
        self.log_text.pack(fill="both", expand=True)

        # --- Persisted settings (task #71) ---
        # Maps each gui_logic.persisted_keys_for(self.kind) key to the
        # tkinter Variable holding it, so _apply_persisted_state/
        # _collect_persisted_state can read/write them generically.
        self._state_vars = {
            "recording_type":    self.recording_type_var,
            "prompt_template":   self.prompt_var,
            "whisper_model":     self.whisper_model_var,
            "language":          self.language_var,
            "llm_backend":       self.llm_backend_var,
            "output_dir":        self.output_dir_var,
            "enable_diarization": self.enable_diarization_var,
            "no_summary":        self.no_summary_var,
            "force_retranscribe": self.force_retranscribe_var,
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
        as "nothing saved yet" (see gui_logic.load_state_file)."""
        state = gui_logic.load_state_file(STATE_PATH)
        saved = state.get(self.kind) or {}
        if not saved:
            return
        defaults = {k: v.get() for k, v in self._state_vars.items()}
        merged = gui_logic.merge_persisted_state(self.kind, saved, defaults)
        for key, var in self._state_vars.items():
            if key == "recording_type" and merged[key] not in self._recording_types:
                continue
            try:
                var.set(merged[key])
            except Exception:
                pass

    def _collect_persisted_state(self) -> dict:
        """Task #71: snapshot this tab's current settings for
        gui_state.json, filtered to just the keys this tab kind persists
        (gui_logic.build_state_dict) - never the path, batch table, or
        meeting info/Q&A fields."""
        raw = {k: v.get() for k, v in self._state_vars.items()}
        return gui_logic.build_state_dict(self.kind, raw)

    def _copy_settings_to_other_tab(self):
        """Task #72: copy whisper model, language, LLM backend, and the
        diarization/no-summary/force-retranscribe toggles to the other
        tab (Meeting <-> Video/Webinar). Deliberately narrower than
        gui_state.json's persisted keys - NOT the prompt template (each
        tab has its own presets/recording types) and NOT meeting info,
        path, or batch table contents (see gui_logic.COPYABLE_KEYS)."""
        other = self.app.video_run if self.kind == "meeting" else self.app.meeting_run
        raw = {k: v.get() for k, v in self._state_vars.items()}
        settings = gui_logic.extract_copyable_settings(raw)
        for key, value in settings.items():
            var = other._state_vars.get(key)
            if var is not None:
                try:
                    var.set(value)
                except Exception:
                    pass
        other.status_label.config(text=f"Settings copied from the {self._label()} tab.")

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
        misleading left visible. Both frames are always re-anchored via
        after=self._rec_frame, so toggling never reorders the rest of the
        tab (Parameters/Stages/Formats/Run/Log stay put).
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

        is_batch = mode == MODES[1]
        self._meeting_frame.pack_forget()
        if self._qa_frame is not None:
            self._qa_frame.pack_forget()
        self.batch_frame.pack_forget()

        if is_batch:
            self.batch_frame.pack(fill="both", expand=False, padx=8, pady=4, after=self._rec_frame)
        else:
            self._meeting_frame.pack(fill="x", padx=8, pady=4, after=self._rec_frame)
            if self._qa_frame is not None:
                self._qa_frame.pack(fill="x", padx=8, pady=4, after=self._meeting_frame)

    def _load_ics(self):
        path = filedialog.askopenfilename(
            title="Select .ics calendar invite",
            filetypes=[("Calendar invite", "*.ics"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            parsed = ics_utils.parse_ics(path)
        except Exception as exc:
            messagebox.showerror("Could not read .ics file", str(exc))
            return
        if parsed.get("title"):
            self.meeting_title_var.set(parsed["title"])
        if parsed.get("date"):
            self.meeting_date_var.set(parsed["date"])
        if not parsed.get("title") and not parsed.get("date"):
            messagebox.showinfo("Nothing found", "No title or date found in this .ics file.")

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
        written back into the row on Enter/focus-out."""
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
        patterns = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS))
        paths = filedialog.askopenfilenames(
            title="Select audio/video files to add",
            filetypes=[("Audio/Video", patterns), ("All files", "*.*")],
        )
        for p in paths:
            self._batch_insert_row(file=p)

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
            self._batch_insert_row(file=str(f))

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
            "llm_backend":        self.llm_backend_var.get(),
            "enable_diarization": self.enable_diarization_var.get(),
            "no_summary":         self.no_summary_var.get(),
            "force_retranscribe": self.force_retranscribe_var.get(),
            "output_dir_override": self.output_dir_var.get().strip(),
            "notes_format_txt":   self.notes_txt_var.get(),
            "notes_format_html":  self.notes_html_var.get(),
            "notes_format_pdf":   self.notes_pdf_var.get(),
            "notes_format_docx":  self.notes_docx_var.get(),
            "meeting_title":      self.meeting_title_var.get().strip(),
            "meeting_date":       self.meeting_date_var.get().strip(),
            "meeting_comments":   self.meeting_comments_var.get().strip(),
            "output_basename_override": self.output_basename_var.get().strip(),
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
    """

    def __init__(self, master, overrides_source):
        super().__init__(master)
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

    def _selected_refs(self) -> list:
        """Selected rows as (video_path, slide_index) pairs, valid in
        both single-video and batch-overview mode."""
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
    PipelineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
