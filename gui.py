# =============================================================
#  Transkription_Notes_Pipeline - gui.py
#  tkinter parameter GUI, three tabs:
#    Run      - file/batch selection, whisper/notes/slide params,
#               output folder, per-format checkboxes, batch paste
#               box, per-stage progress bar, live log
#    Search   - full-text search (FTS5) over generated notes and
#               slide titles/bullets
#    Settings - keys.cfg as a structured form (masked API keys),
#               requirements check/install, database file
#               management, logging options, "Open config.py"
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
from tkinter import filedialog, messagebox, scrolledtext, ttk

from PIL import Image, ImageTk

from config import CONFIG, PROMPTS_DIR, SUPPORTED_EXTENSIONS
import db
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

# Recording-type presets: picking one sets the prompt template and the
# slide-detection/VLM defaults in one click. All three fields stay
# manually editable afterward, this only fills in a sensible starting
# point. Order here is the order shown in the dropdown.
RECORDING_TYPES = {
    "Audio Transcript":   {"prompt_template": "audio_transcript", "enable_slides": False, "enable_vlm": False},
    "Video Transcript":   {"prompt_template": "video_transcript", "enable_slides": True,  "enable_vlm": True},
    "Meeting Transcript": {"prompt_template": "meeting",          "enable_slides": False, "enable_vlm": False},
    "Webinar Transcript": {"prompt_template": "webinar",          "enable_slides": True,  "enable_vlm": True},
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


class QueueLogHandler(logging.Handler):
    """Logging handler that pushes formatted records into a thread-safe queue."""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


class PipelineGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Transkription_Notes_Pipeline")
        self.root.geometry("900x820")

        self.log_queue: queue.Queue = queue.Queue()
        self.req_queue: queue.Queue = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self._notes_result_paths: dict = {}
        self._slides_result_paths: dict = {}
        self._last_check_results: list = []

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)
        self.run_tab = ttk.Frame(notebook)
        self.search_tab = ttk.Frame(notebook)
        self.settings_tab = ttk.Frame(notebook)
        notebook.add(self.run_tab, text="Run")
        notebook.add(self.search_tab, text="Search")
        notebook.add(self.settings_tab, text="Settings")

        self._build_run_tab()
        self._build_search_tab()
        self._build_settings_tab()
        self._refresh_prompt_templates()
        self.root.after(100, self._poll_log_queue)
        self.root.after(200, self._poll_req_queue)

    # -----------------------------------------------------------------
    # Run tab
    # -----------------------------------------------------------------

    def _build_run_tab(self):
        parent = self.run_tab
        pad = {"padx": 8, "pady": 4}

        self.mode_var = tk.StringVar(value=MODES[0])
        self.path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.recording_type_var = tk.StringVar(value="Meeting Transcript")
        self.whisper_model_var = tk.StringVar(value=CONFIG["whisper_model"])
        self.language_var = tk.StringVar(value=CONFIG["whisper_language"])
        self.prompt_var = tk.StringVar()
        self.llm_backend_var = tk.StringVar(value=CONFIG["llm_backend"])

        self.enable_slides_var = tk.BooleanVar(value=CONFIG["enable_slides"])
        self.enable_vlm_var = tk.BooleanVar(value=CONFIG["enable_vlm"])
        self.enable_diarization_var = tk.BooleanVar(value=CONFIG["enable_diarization"])
        self.no_summary_var = tk.BooleanVar(value=False)
        self.force_retranscribe_var = tk.BooleanVar(value=False)
        self.dry_run_var = tk.BooleanVar(value=False)

        self.threshold_var = tk.IntVar(value=CONFIG["hash_threshold"])
        self.animation_threshold_var = tk.IntVar(value=CONFIG.get("animation_threshold", 0))
        self.fps_var = tk.IntVar(value=CONFIG["fps"])
        self.min_slide_duration_var = tk.DoubleVar(value=CONFIG["min_slide_duration_sec"])
        self.recording_speed_var = tk.DoubleVar(value=CONFIG.get("recording_speed", 1.0))
        self.convert_video_to_realtime_var = tk.BooleanVar(
            value=CONFIG.get("convert_video_to_realtime", False))

        # Title slide cover image (optional, shown before Slide 1 in the report)
        self.title_slide_image_var = tk.StringVar(value=CONFIG.get("title_slide_image_path", ""))

        # External tools: LosslessCut path (session override on top of config.py)
        self.losslesscut_path_var = tk.StringVar(value=CONFIG.get("losslesscut_path", ""))

        # Slide report content selection
        self.report_show_image_var = tk.BooleanVar(value=CONFIG.get("report_show_image", True))
        self.report_show_bullets_var = tk.BooleanVar(value=CONFIG.get("report_show_bullets", True))
        self.report_show_transcript_var = tk.BooleanVar(value=CONFIG.get("report_show_transcript", True))
        self.report_transcript_mode_var = tk.StringVar(value=CONFIG.get("report_transcript_mode", "full"))

        # Notes output formats
        self.notes_txt_var = tk.BooleanVar(value=CONFIG["notes_format_txt"])
        self.notes_html_var = tk.BooleanVar(value=CONFIG["notes_format_html"])
        self.notes_pdf_var = tk.BooleanVar(value=CONFIG["notes_format_pdf"])
        self.notes_docx_var = tk.BooleanVar(value=CONFIG["notes_format_docx"])

        # Slide-report formats (video mode only)
        self.report_html_var = tk.BooleanVar(value=CONFIG["report_html"])
        self.report_csv_var = tk.BooleanVar(value=CONFIG["report_csv"])
        self.report_json_var = tk.BooleanVar(value=CONFIG["report_json"])
        self.report_srt_var = tk.BooleanVar(value=CONFIG["report_srt"])
        self.report_pdf_var = tk.BooleanVar(value=CONFIG["report_pdf"])
        self.report_slide_timing_var = tk.BooleanVar(value=CONFIG.get("report_slide_timing", True))
        self.zip_snapshots_var = tk.BooleanVar(value=CONFIG.get("zip_snapshots", False))

        top = ttk.Frame(parent)
        top.pack(fill="x", **pad)
        self._top_frame = top

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

        self.output_basename_var = tk.StringVar()
        ttk.Label(top, text="Output filename (optional):").grid(row=4, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.output_basename_var, width=60).grid(row=4, column=1, sticky="we")
        ttk.Label(top, text="Blank = use the source filename for every output file (default).",
                 foreground="#666").grid(row=5, column=1, sticky="w")

        top.columnconfigure(1, weight=1)

        # --- Recording type preset ---
        rec_frame = ttk.LabelFrame(parent, text="Recording type (preset)")
        rec_frame.pack(fill="x", padx=8, pady=4)
        self._rec_frame = rec_frame
        ttk.Label(rec_frame, text="Type:").pack(side="left", padx=8, pady=8)
        rec_box = ttk.Combobox(rec_frame, textvariable=self.recording_type_var,
                               values=list(RECORDING_TYPES.keys()), state="readonly", width=22)
        rec_box.pack(side="left", pady=8)
        rec_box.bind("<<ComboboxSelected>>", lambda e: self._apply_recording_type())
        ttk.Label(rec_frame, text="Sets prompt template + slide detection below; still editable afterward.",
                 foreground="#666").pack(side="left", padx=(12, 0))

        # --- Meeting info (optional) ---
        meeting_frame = ttk.LabelFrame(parent, text="Meeting info (optional)")
        meeting_frame.pack(fill="x", padx=8, pady=4)
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
                 text="Pre-fills Title/Date from an Outlook/Google/Teams .ics invite. "
                      "Shown in the notes header and used as the report title. "
                      "Comments are shown in the notes and slide report headers too.",
                 foreground="#666").grid(row=3, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 4))
        ttk.Label(meeting_frame, text="Title slide image:").grid(row=4, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(meeting_frame, textvariable=self.title_slide_image_var, width=45).grid(
            row=4, column=1, sticky="w")
        ttk.Button(meeting_frame, text="Browse...", command=self._browse_title_slide_image).grid(
            row=4, column=2, sticky="w", padx=(8, 0))
        ttk.Label(meeting_frame,
                 text="Optional cover image (jpg/png) shown as a title page before Slide 1 "
                      "in the HTML/PDF slide report.",
                 foreground="#666").grid(row=5, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 4))

        # --- Batch list paste box (File list (batch) mode only) ---
        self.batch_frame = ttk.LabelFrame(
            parent, text="Batch list: paste file paths (one per line, optional |language) "
                        "or load a .txt/.csv")
        self.batch_text = scrolledtext.ScrolledText(self.batch_frame, height=6, wrap="none")
        self.batch_text.pack(fill="both", expand=True, padx=8, pady=(6, 4))
        batch_btn_row = ttk.Frame(self.batch_frame)
        batch_btn_row.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(batch_btn_row, text="Load .txt/.csv...", command=self._load_batch_file).pack(side="left")
        ttk.Button(batch_btn_row, text="Save box to .txt...", command=self._save_batch_box).pack(
            side="left", padx=(8, 0))
        ttk.Button(batch_btn_row, text="Clear", command=self._clear_batch_box).pack(side="left", padx=(8, 0))
        ttk.Label(batch_btn_row, text="Leave empty to use the list file selected above instead.",
                 foreground="#666").pack(side="left", padx=(12, 0))
        # Not packed initially; _update_path_label() shows it only for MODES[1].

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
        ttk.Button(params, text="Go to prompt folder", command=self._open_prompts_folder).grid(
            row=1, column=4, sticky="w", padx=(4, 0))

        ttk.Label(params, text="LLM backend:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Combobox(params, textvariable=self.llm_backend_var, values=LLM_BACKENDS,
                    state="readonly", width=14).grid(row=2, column=1, sticky="w")

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

        # --- Toggles ---
        toggles = ttk.LabelFrame(parent, text="Stages")
        toggles.pack(fill="x", **pad)
        ttk.Checkbutton(toggles, text="Slide detection (video files only)",
                        variable=self.enable_slides_var).grid(row=0, column=0, sticky="w", **pad)
        ttk.Checkbutton(toggles, text="VLM slide annotation",
                        variable=self.enable_vlm_var).grid(row=0, column=1, sticky="w", **pad)
        ttk.Checkbutton(toggles, text="Speaker diarization",
                        variable=self.enable_diarization_var).grid(row=0, column=2, sticky="w", **pad)
        ttk.Checkbutton(toggles, text="No summary (transcript only)",
                        variable=self.no_summary_var).grid(row=1, column=0, sticky="w", **pad)
        ttk.Checkbutton(toggles, text="Force re-transcribe / re-process",
                        variable=self.force_retranscribe_var).grid(row=1, column=1, sticky="w", **pad)
        ttk.Checkbutton(toggles, text="Dry run (slide timestamps only)",
                        variable=self.dry_run_var).grid(row=1, column=2, sticky="w", **pad)

        # --- Output formats ---
        formats = ttk.LabelFrame(parent, text="Output formats")
        formats.pack(fill="x", **pad)
        ttk.Label(formats, text="Notes:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Checkbutton(formats, text="txt", variable=self.notes_txt_var).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(formats, text="html", variable=self.notes_html_var).grid(row=0, column=2, sticky="w")
        ttk.Checkbutton(formats, text="pdf", variable=self.notes_pdf_var).grid(row=0, column=3, sticky="w")
        ttk.Checkbutton(formats, text="docx", variable=self.notes_docx_var).grid(row=0, column=4, sticky="w")

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
                  command=self._zip_existing_snapshots).grid(row=2, column=0, columnspan=4, sticky="w", padx=8, pady=(4, 0))

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
        ttk.Button(run_frame, text="Re-annotate failed slides...",
                  command=self._on_reannotate_failed).pack(side="left", padx=(6, 0))
        ttk.Button(run_frame, text="Open in LosslessCut...",
                  command=self._open_in_losslesscut).pack(side="left", padx=(6, 0))
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

        # --- Log panel ---
        log_frame = ttk.LabelFrame(parent, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log_text = scrolledtext.ScrolledText(log_frame, state="disabled", height=16, wrap="word")
        self.log_text.pack(fill="both", expand=True)

        self._apply_recording_type()
        self._update_path_label()

    def _apply_recording_type(self):
        preset = RECORDING_TYPES.get(self.recording_type_var.get())
        if not preset:
            return
        self.prompt_var.set(preset["prompt_template"])
        self.enable_slides_var.set(preset["enable_slides"])
        self.enable_vlm_var.set(preset["enable_vlm"])

    def _update_path_label(self):
        mode = self.mode_var.get()
        labels = {
            MODES[0]: "File:",
            MODES[1]: "List file (.txt, optional if using the batch box below):",
            MODES[2]: "Folder:",
            MODES[3]: "Transcript file:",
            MODES[4]: "Folder:",
        }
        self.path_label.config(text=labels.get(mode, "File:"))
        if mode == MODES[1]:
            self.batch_frame.pack(fill="both", expand=False, padx=8, pady=4, after=self._meeting_frame)
        else:
            self.batch_frame.pack_forget()

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

    def _refresh_prompt_templates(self):
        templates = notes_mod.list_prompt_templates(PROMPTS_DIR)
        names = list(templates.keys())
        self.prompt_box["values"] = names
        if names and self.prompt_var.get() not in names:
            default = "meeting" if "meeting" in names else names[0]
            self.prompt_var.set(default)

    def _open_prompts_folder(self):
        """Open prompts/ in the OS file browser so custom .md templates can
        be added/edited directly, then picked up via "Reload list" above."""
        self._open_path(Path(PROMPTS_DIR))

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
        """Open the slide review/reprocess window (task #58): mark
        detected slides to omit or merge, then rebuild the report
        outputs without re-running transcription/detection/VLM."""
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
    # Batch paste box (File list (batch) mode)
    # -----------------------------------------------------------------

    def _load_batch_file(self):
        path = filedialog.askopenfilename(
            title="Load batch list (.txt or .csv)",
            filetypes=[("Text/CSV", "*.txt *.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        lines = []
        try:
            if path.lower().endswith(".csv"):
                # CSV columns: filepath[, language]. Converted to the same
                # "path|lang" convention run_pipeline.run_file_list expects.
                with open(path, newline="", encoding="utf-8") as f:
                    for row in csv.reader(f):
                        if not row or not row[0].strip() or row[0].strip().startswith("#"):
                            continue
                        filepath = row[0].strip()
                        lang = row[1].strip() if len(row) > 1 and row[1].strip() else ""
                        lines.append(f"{filepath}|{lang}" if lang else filepath)
            else:
                with open(path, encoding="utf-8") as f:
                    lines = [ln.rstrip("\n") for ln in f]
        except Exception as exc:
            messagebox.showerror("Could not load file", str(exc))
            return
        self.batch_text.delete("1.0", "end")
        self.batch_text.insert("1.0", "\n".join(lines) + ("\n" if lines else ""))

    def _save_batch_box(self):
        path = filedialog.asksaveasfilename(
            title="Save batch list", defaultextension=".txt",
            filetypes=[("Text file", "*.txt")],
        )
        if not path:
            return
        content = self.batch_text.get("1.0", "end")
        try:
            Path(path).write_text(content, encoding="utf-8")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        messagebox.showinfo("Saved", f"Batch list saved:\n{path}")

    def _clear_batch_box(self):
        self.batch_text.delete("1.0", "end")

    # -----------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------

    def _build_overrides(self) -> dict:
        return {
            "prompt_template":    self.prompt_var.get() or None,
            "whisper_model":      self.whisper_model_var.get(),
            "whisper_language":   self.language_var.get(),
            "llm_backend":        self.llm_backend_var.get(),
            "enable_slides":      self.enable_slides_var.get(),
            "enable_vlm":         self.enable_vlm_var.get(),
            "enable_diarization": self.enable_diarization_var.get(),
            "no_summary":         self.no_summary_var.get(),
            "force_retranscribe": self.force_retranscribe_var.get(),
            "dry_run":            self.dry_run_var.get(),
            "hash_threshold":     self.threshold_var.get(),
            "animation_threshold": self.animation_threshold_var.get(),
            "fps":                self.fps_var.get(),
            "min_slide_duration_sec": self.min_slide_duration_var.get(),
            "recording_speed":     self.recording_speed_var.get(),
            "convert_video_to_realtime": self.convert_video_to_realtime_var.get(),
            "title_slide_image_path": self.title_slide_image_var.get().strip(),
            "output_dir_override": self.output_dir_var.get().strip(),
            "notes_format_txt":   self.notes_txt_var.get(),
            "notes_format_html":  self.notes_html_var.get(),
            "notes_format_pdf":   self.notes_pdf_var.get(),
            "notes_format_docx":  self.notes_docx_var.get(),
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
            "meeting_title":      self.meeting_title_var.get().strip(),
            "meeting_date":       self.meeting_date_var.get().strip(),
            "meeting_comments":   self.meeting_comments_var.get().strip(),
            "output_basename_override": self.output_basename_var.get().strip(),
            "db_path":            self.db_path_var.get().strip() or None,
        }

    def _on_run(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Busy", "A run is already in progress.")
            return

        mode = self.mode_var.get()
        path = self.path_var.get().strip()

        if mode == MODES[1]:
            box_content = self.batch_text.get("1.0", "end").strip()
            if box_content:
                tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False, encoding="utf-8", prefix="batch_list_")
                tmp.write(box_content + "\n")
                tmp.close()
                path = tmp.name
            elif not path:
                messagebox.showwarning(
                    "Missing input",
                    "Paste file paths into the batch box, or choose an existing list.txt.")
                return
        elif not path:
            messagebox.showwarning("Missing input", "Please choose a file/folder first.")
            return

        if not self.prompt_var.get():
            messagebox.showwarning("Missing prompt template",
                                   "No prompt template selected. Add a .md file to prompts/ "
                                   "and click 'Reload list'.")
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
            target=self._run_worker, args=(mode, path, overrides), daemon=True
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
        snapshots already on disk. See run_pipeline.reannotate_failed_slides."""
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
        level = getattr(logging, self.log_level_var.get().upper(), logging.INFO)
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

    def _run_worker(self, mode: str, path: str, overrides: dict):
        handler = QueueLogHandler(self.log_queue)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        level = getattr(logging, self.log_level_var.get().upper(), logging.INFO)
        root_logger.setLevel(level)

        file_handler = None
        if self.log_to_file_var.get():
            try:
                log_dir = Path(self.log_dir_var.get().strip() or CONFIG.get("log_dir", "logs"))
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
        parent = self.settings_tab
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


class SlideReviewDialog(tk.Toplevel):
    """
    Modal window (task #58): review the slides detected in a run, mark
    any to omit or merge with the next slide, and rebuild the CSV/HTML/
    PDF/timing-summary reports from those edits without re-running
    transcription/detection/VLM annotation. Works for a single video, or
    for switching between multiple videos from the same batch run via
    the "recently loaded" dropdown. Edits are saved to a separate
    "<stem>_slides_edits.json" next to the original *_slides.json, which
    is never modified - always safe to reload and start over. See
    run_pipeline.apply_slide_edits / rebuild_outputs for the edit
    semantics and rebuild logic.

    Includes a preview panel (task #60): selecting a row shows that
    slide's snapshot image plus title/bullets/transcript, so slides can
    be identified visually before deciding to omit/merge them.
    """

    def __init__(self, master, overrides_source):
        super().__init__(master)
        self.title("Review / Edit Slides")
        self.geometry("1180x560")
        self._overrides_source = overrides_source  # PipelineGUI instance, for meeting_title/etc.
        self.slides_json_path: Path | None = None
        self.slides: list = []
        self.omit_indices: set = set()
        self.merge_next_indices: set = set()
        self._recent_paths: list = []
        self._preview_photo = None  # keep a reference, else Tk garbage-collects the image

        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=8)
        ttk.Button(top, text="Load *_slides.json...", command=self._load_file).pack(side="left")
        self.recent_var = tk.StringVar()
        self.recent_box = ttk.Combobox(top, textvariable=self.recent_var, state="readonly", width=50)
        self.recent_box.pack(side="left", padx=(8, 0))
        self.recent_box.bind("<<ComboboxSelected>>", lambda e: self._switch_to_recent())
        ttk.Label(top, text="Switch between videos loaded this session (batch runs).",
                 foreground="#666").pack(side="left", padx=(8, 0))

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        tree_frame = ttk.Frame(body)
        tree_frame.pack(side="left", fill="both", expand=True)
        self.tree = ttk.Treeview(
            tree_frame, columns=("num", "time", "title", "bullets", "status"),
            show="headings", selectmode="extended", height=16)
        for col, text, w in (("num", "#", 40), ("time", "Time", 70), ("title", "Title", 160),
                             ("bullets", "First bullet", 220), ("status", "Status", 120)):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=w, anchor="w")
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
        ttk.Button(btn_row, text="Clear all edits", command=self._clear_edits).pack(side="left", padx=(6, 0))
        ttk.Button(btn_row, text="Rebuild outputs", command=self._rebuild).pack(side="right")
        self.status_label = ttk.Label(btn_row, text="")
        self.status_label.pack(side="right", padx=(0, 12))

        ttk.Label(self, text="Select one or more rows, then Toggle Omit / Toggle Merge with next. "
                            "\"Rebuild outputs\" regenerates CSV/HTML/PDF/timing summary only - "
                            "no re-transcription, re-detection, or VLM calls.",
                 foreground="#666", wraplength=860).pack(fill="x", padx=8, pady=(0, 8))

    def _load_file(self):
        path = filedialog.askopenfilename(
            title="Select *_slides.json",
            filetypes=[("Slides JSON", "*_slides.json"), ("All files", "*.*")])
        if not path:
            return
        self._load_path(path)

    def _switch_to_recent(self):
        path = self.recent_var.get()
        if path:
            self._load_path(path, remember=False)

    def _load_path(self, path: str, remember: bool = True):
        try:
            slides = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            messagebox.showerror("Could not load", str(exc))
            return
        self.slides = slides
        self.slides_json_path = Path(path)
        self.omit_indices = set()
        self.merge_next_indices = set()

        # Load any previously saved edits for this file, so re-opening
        # the review picks up where you left off.
        edits_path = self._edits_path()
        if edits_path.exists():
            try:
                edits = json.loads(edits_path.read_text(encoding="utf-8"))
                self.omit_indices = set(edits.get("omit_indices", []) or [])
                self.merge_next_indices = set(edits.get("merge_next_indices", []) or [])
            except Exception:
                pass

        if remember and path not in self._recent_paths:
            self._recent_paths.append(path)
            self.recent_box["values"] = self._recent_paths
        self.recent_var.set(path)
        self._refresh_tree()
        self.status_label.config(text=f"{len(self.slides)} slide(s) loaded.")

    def _edits_path(self) -> Path:
        stem = self.slides_json_path.stem.replace("_slides", "")
        return self.slides_json_path.parent / f"{stem}_slides_edits.json"

    def _refresh_tree(self):
        self.tree.delete(*self.tree.get_children())
        self._show_preview(None)
        for idx, slide in enumerate(self.slides):
            bullets = slide.get("bullets") or []
            first_bullet = bullets[0] if bullets else ""
            status_parts = []
            if idx in self.omit_indices:
                status_parts.append("OMIT")
            if idx in self.merge_next_indices:
                status_parts.append("MERGE->next")
            self.tree.insert("", "end", iid=str(idx), values=(
                idx + 1, f"{slide.get('timestamp_sec', 0):.1f}s",
                slide.get("title", ""), first_bullet, " + ".join(status_parts)))

    def _selected_indices(self) -> list:
        return [int(iid) for iid in self.tree.selection()]

    def _on_select(self, event=None):
        sel = self._selected_indices()
        if not sel or sel[0] >= len(self.slides):
            self._show_preview(None)
            return
        self._show_preview(self.slides[sel[0]])

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
        sel = self._selected_indices()
        if not sel:
            return
        for idx in sel:
            if idx in self.omit_indices:
                self.omit_indices.discard(idx)
            else:
                self.omit_indices.add(idx)
        self._refresh_tree()

    def _toggle_merge(self):
        sel = self._selected_indices()
        if not sel:
            return
        for idx in sel:
            if idx >= len(self.slides) - 1:
                continue  # last slide can't merge with a next one
            if idx in self.merge_next_indices:
                self.merge_next_indices.discard(idx)
            else:
                self.merge_next_indices.add(idx)
        self._refresh_tree()

    def _clear_edits(self):
        self.omit_indices = set()
        self.merge_next_indices = set()
        self._refresh_tree()

    def _rebuild(self):
        if not self.slides_json_path:
            messagebox.showwarning("No file loaded", "Load a *_slides.json first.")
            return
        edits = {
            "omit_indices": sorted(self.omit_indices),
            "merge_next_indices": sorted(self.merge_next_indices),
        }
        overrides = {}
        if hasattr(self._overrides_source, "_build_overrides"):
            try:
                overrides = self._overrides_source._build_overrides()
            except Exception:
                overrides = {}
        try:
            ok = run_pipeline.rebuild_outputs(str(self.slides_json_path), edits, overrides)
        except Exception as exc:
            messagebox.showerror("Rebuild failed", str(exc))
            return
        if ok:
            messagebox.showinfo("Done", f"Outputs rebuilt in:\n{self.slides_json_path.parent}")
            self.status_label.config(text="Rebuild complete.")
        else:
            messagebox.showerror("Rebuild failed", "See log for details.")


def launch():
    root = tk.Tk()
    PipelineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
