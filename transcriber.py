# =============================================================
#  Transkription_Notes_Pipeline - transcriber.py
#  Speech-to-text via whisperx with word-level timestamps and
#  optional speaker diarization. Used by both audio-only and
#  video-with-slides runs.
#
#  Three-step pipeline:
#    1. Transcribe (faster-whisper backend, batched)
#    2. Align (wav2vec2 forced alignment for word-level timestamps)
#    3. Diarize (pyannote, optional, requires HuggingFace token)
#
#  Diarization API note (see project CLAUDE.md): whisperx 3.8.x
#  requires `from whisperx.diarize import DiarizationPipeline` with
#  a `token=` keyword argument, NOT `whisperx.DiarizationPipeline(
#  use_auth_token=...)`. Using the wrong form raises a TypeError.
#
#  Diarization model (2026-08): explicitly pinned to
#  pyannote/speaker-diarization-community-1 via DiarizationPipeline's
#  model_name= argument, rather than relying on whisperx's own internal
#  default. Free, fully local/offline (no cloud calls, unlike pyannoteAI's
#  paid "precision-2" model), gated on HuggingFace the same way the
#  previous default (pyannote/speaker-diarization-3.1) was - accept once
#  at https://huggingface.co/pyannote/speaker-diarization-community-1
#  with the same HF_TOKEN already used for 3.1. Benchmarked lower
#  diarization error rate than 3.1 across every dataset pyannote
#  publishes (AMI, AliMeeting, VoxConverse, etc. - see the model card).
#  Note: whisperx>=3.8.6 already defaults to community-1 internally when
#  model_name is omitted, so relying on that default would have picked
#  this up silently anyway; pinning it explicitly here means a future
#  whisperx release changing that default again won't silently change
#  what this pipeline uses without it showing up in a diff.
#
#  Supported languages for alignment (auto-loaded from torchaudio/HF):
#    en, de, fr, es, it, ja, zh, nl, uk, pt
# =============================================================

import gc
import logging
import re
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

try:
    from vocabulary import build_initial_prompt as _build_prompt
    _VOCAB_AVAILABLE = True
except ImportError:
    _VOCAB_AVAILABLE = False

SUPPORTED_LANGUAGES = {"en", "de", "fr", "es", "it", "ja", "zh", "nl", "uk", "pt"}
PRIMARY_LANGUAGES   = {"en", "de"}

# Diarization model - see the module docstring's "Diarization model" note
# for why this is pinned explicitly rather than left to whisperx's own
# internal default.
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


def _build_diarization_pipeline(hf_token: str, device):
    """
    Construct whisperx's DiarizationPipeline with DIARIZATION_MODEL
    explicitly pinned. Split out of transcribe() so the model-name choice
    is unit-testable (monkeypatch whisperx.diarize.DiarizationPipeline)
    without needing a full transcribe() run - see tests/test_transcriber.py.
    """
    from whisperx.diarize import DiarizationPipeline
    return DiarizationPipeline(model_name=DIARIZATION_MODEL, token=hf_token, device=device)


def validate_language(language: str | None) -> str | None:
    """
    Validate the language code before passing it to whisperx.
    Returns the code unchanged if valid, or None (auto-detect) for "auto".
    """
    if language is None or language.lower() == "auto":
        log.info("Language: auto-detect (set 'en' or 'de' for faster results).")
        return None
    code = language.strip().lower()
    if code in PRIMARY_LANGUAGES:
        log.info("Language: %s (primary target, alignment model available).", code)
        return code
    if code in SUPPORTED_LANGUAGES:
        log.info("Language: %s (supported, alignment model available).", code)
        return code
    log.warning(
        "Language code '%s' is not in the confirmed supported list %s. "
        "whisperx will attempt to find an alignment model on HuggingFace.",
        code, sorted(SUPPORTED_LANGUAGES),
    )
    return code


def _import_whisperx():
    """Lazy import with a clear error message if not installed."""
    try:
        import whisperx
        return whisperx
    except ImportError:
        log.error(
            "whisperx not installed. "
            "Run: C:\\Python\\Python311\\python.exe -m pip install whisperx==3.8.5"
        )
        return None


def _resolve_device(device: str) -> tuple[str, str]:
    """
    Resolve 'auto' device to 'cuda' or 'cpu' based on torch availability.
    Returns (device, compute_type) - compute_type switches to int8 on CPU.
    """
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                log.info("GPU detected: using CUDA (%s) with float16.", torch.cuda.get_device_name(0))
                return "cuda", "float16"
        except ImportError:
            pass
        log.info("No GPU detected: falling back to CPU with int8.")
        return "cpu", "int8"
    return device, "float16" if device == "cuda" else "int8"


def transcript_cache_paths(output_prefix: str) -> dict:
    """Return the standard output file paths for a given prefix."""
    return {
        "speakers": output_prefix + "_transcript_speakers.txt",
        "text":     output_prefix + "_text.txt",
        "srt":      output_prefix + "_transcript.srt",
    }


def transcript_cache_exists(output_prefix: str) -> bool:
    """True if a previous transcription already produced the speakers file."""
    return Path(transcript_cache_paths(output_prefix)["speakers"]).exists()


def enhance_audio(input_path: str, output_path: str) -> bool:
    """
    Run a conservative ffmpeg audio-cleanup filter chain and write the
    result to output_path, for use as a temporary pre-transcription pass
    (config: enhance_audio, off by default). Never touches input_path.

    Filter chain, in order:
      highpass=f=100  - cuts rumble, AC hum, and mic-handling noise
                         below the speech range. Very low risk of
                         hurting transcription, this is a safe default.
      afftdn           - mild FFT-based noise reduction (ffmpeg
                         defaults, not tuned aggressively) for steady
                         background hiss or fan noise. Aggressive
                         denoising can introduce artifacts that confuse
                         Whisper, so this deliberately uses ffmpeg's
                         untuned defaults rather than a stronger setting.
      dynaudnorm       - evens out volume between a loud and a quiet
                         speaker on the same recording, and helps the
                         voiceprint/speaker-ID step be more consistent.

    Output is resampled to 16 kHz mono, matching what Whisper expects
    anyway, so this replaces (not adds to) WhisperX's own resampling
    step for the enhanced copy.

    Returns True on success, False if ffmpeg is missing or the filter
    pass fails for any reason - a failed enhancement should never be
    fatal to transcription; callers should fall back to the original
    file.
    """
    import subprocess
    import config

    ffmpeg_exe = config.get_ffmpeg_path()
    if ffmpeg_exe is None:
        log.warning("ffmpeg not found (bundled ffmpeg\\bin\\ or PATH) - skipping "
                    "audio enhancement, using original audio.")
        return False

    cmd = [
        ffmpeg_exe, "-y", "-i", str(input_path),
        "-af", "highpass=f=100,afftdn,dynaudnorm",
        "-ar", "16000", "-ac", "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.warning("Audio enhancement failed: %s", result.stderr[-500:])
        return False
    log.info("Audio enhancement applied: %s", output_path)
    return True


def transcribe(
    file_path: str,
    model_size: str = "large-v3",
    language: str | None = None,
    device: str = "auto",
    batch_size: int = 16,
    compute_type: str = "float16",
    hf_token: str | None = None,
    enable_diarization: bool = False,
    use_vocabulary: bool = True,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[dict]:
    """
    Full whisperx pipeline: transcribe -> align -> (optional) diarize.

    Returns a flat list of segment dicts:
      {start, end, text, speaker (if diarized), words (list of word-level dicts)}

    Parameters:
      model_size         Whisper model: tiny, base, small, medium, large-v2, large-v3
      language            ISO code, "auto"/None for auto-detect
      device              "auto" | "cpu" | "cuda"
      batch_size          Parallel batches; reduce if OOM. CPU: 4, GPU: 8-16.
      compute_type        "int8" (CPU/low VRAM) or "float16" (GPU) - overridden by device resolution
      hf_token             HuggingFace read token; required only for diarization.
      enable_diarization   Run speaker diarization (requires hf_token).
      min_speakers/max_speakers  Optional hints for diarization.
    """
    wx = _import_whisperx()
    if wx is None:
        return []

    audio_path = str(file_path)
    language = validate_language(language)
    device, compute_type = _resolve_device(device)

    # Step 1: Transcribe
    log.info("Loading whisperx model: %s on %s (compute=%s)", model_size, device, compute_type)
    model = wx.load_model(model_size, device, compute_type=compute_type,
                           language=language)
    audio = wx.load_audio(audio_path)

    # Domain vocabulary: TranscriptionOptions is a faster_whisper dataclass -
    # fields are mutable, assign model.options.initial_prompt directly.
    # Do NOT pass initial_prompt to model.transcribe() - not a valid kwarg.
    initial_prompt = None
    if use_vocabulary and _VOCAB_AVAILABLE:
        initial_prompt = _build_prompt(language)
        log.info("Using domain vocabulary (%d words).", len(initial_prompt.split()))
    elif use_vocabulary and not _VOCAB_AVAILABLE:
        log.warning("vocabulary.py not found - transcribing without domain hints.")

    if initial_prompt:
        try:
            model.options.initial_prompt = initial_prompt
            log.info("Vocabulary injected via model.options.initial_prompt.")
        except Exception as e:
            log.warning("Vocabulary injection skipped: %s", e)

    log.info("Transcribing: %s", audio_path)
    result = model.transcribe(audio, batch_size=batch_size)
    detected_lang = result.get("language", "unknown")
    log.info("Detected language: %s  |  segments before alignment: %d",
             detected_lang, len(result.get("segments", [])))

    del model
    gc.collect()

    # Step 2: Align (word-level timestamps)
    try:
        log.info("Loading alignment model for language: %s", detected_lang)
        model_a, metadata = wx.load_align_model(language_code=detected_lang, device=device)
        result = wx.align(
            result["segments"], model_a, metadata, audio, device,
            return_char_alignments=False,
        )
        log.info("Alignment complete: %d segments", len(result.get("segments", [])))
        del model_a
        gc.collect()
    except Exception as exc:
        log.warning(
            "Alignment failed for language '%s': %s. "
            "Proceeding with segment-level timestamps only.",
            detected_lang, exc,
        )

    # Step 3: Diarization (optional)
    if enable_diarization:
        if not hf_token or hf_token.startswith("hf_..."):
            log.error(
                "Diarization requires a HuggingFace token. "
                "Set HF_TOKEN in keys.cfg. "
                "Get a token at https://huggingface.co/settings/tokens"
            )
        else:
            try:
                log.info("Running speaker diarization.")
                diarize_model = _build_diarization_pipeline(hf_token, device)
                diarize_kwargs = {}
                if min_speakers:
                    diarize_kwargs["min_speakers"] = min_speakers
                if max_speakers:
                    diarize_kwargs["max_speakers"] = max_speakers
                diarize_segments = diarize_model(audio, **diarize_kwargs)
                result = wx.assign_word_speakers(diarize_segments, result)
                log.info("Diarization complete.")
            except Exception as exc:
                log.warning("Diarization failed: %s. Continuing without speaker labels.", exc)

    # Flatten to a consistent output format used throughout the pipeline
    segments = result.get("segments", [])
    output = []
    for seg in segments:
        output.append({
            "start":   round(seg.get("start", 0.0), 3),
            "end":     round(seg.get("end", 0.0), 3),
            "text":    seg.get("text", "").strip(),
            "speaker": seg.get("speaker", ""),
            "words":   seg.get("words", []),
        })

    log.info("Transcription pipeline complete: %d segments returned.", len(output))
    return output


def align_transcript_to_slides(
    transcript_segments: list[dict],
    slide_timestamps: list[float],
) -> list[str]:
    """
    For each slide (defined by its start timestamp), collect all transcript
    segments whose time window overlaps with the slide's display window.
    Returns a list of joined text strings, one per slide. Video mode only.
    """
    if not transcript_segments or not slide_timestamps:
        return [""] * len(slide_timestamps)

    windows = []
    for i, ts in enumerate(slide_timestamps):
        end = slide_timestamps[i + 1] if i + 1 < len(slide_timestamps) else float("inf")
        windows.append((ts, end))

    aligned = []
    for (win_start, win_end) in windows:
        parts = []
        for seg in transcript_segments:
            seg_start = seg.get("start", 0.0)
            seg_end   = seg.get("end", 0.0)
            if seg_start < win_end and seg_end > win_start:
                words = seg.get("words", [])
                if words:
                    in_window = [
                        w.get("word", "")
                        for w in words
                        if w.get("start", 0.0) >= win_start and w.get("end", 0.0) <= win_end
                    ]
                    if in_window:
                        parts.append(" ".join(in_window))
                else:
                    parts.append(seg.get("text", ""))
        aligned.append(" ".join(parts).strip())

    return aligned


def get_speaker_map(transcript_segments: list[dict]) -> dict[str, list[float]]:
    """Return a dict mapping speaker label to list of active timestamps."""
    speakers: dict[str, list[float]] = {}
    for seg in transcript_segments:
        spk = seg.get("speaker", "")
        if spk:
            mid = (seg["start"] + seg["end"]) / 2
            speakers.setdefault(spk, []).append(mid)
    return speakers


# ---------------------------------------------------------------------------
# Transcript file writers (speakers.txt, text.txt, srt) - shared by audio
# and video mode. Written next to the source file.
# ---------------------------------------------------------------------------

def _fmt_srt(t: float) -> str:
    h, r = divmod(int(t), 3600)
    m, s = divmod(r, 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_transcript_files(source_file: str, segments: list[dict], output_prefix: str,
                           recording_speed: float = 1.0) -> dict:
    """
    Write *_transcript_speakers.txt, *_text.txt and *_transcript.srt from a
    flat list of segment dicts ({start, end, text, speaker}).
    recording_speed: if not 1.0, every timestamp in these files has already
    been converted (real_time = video_time / recording_speed, see
    run_pipeline._rescale_segments). A note to that effect is written into
    the *_transcript_speakers.txt header only; the plain *_text.txt has no
    timestamps and the *_transcript.srt format has no header/comment field,
    so neither is annotated.
    Returns the dict of written paths (see transcript_cache_paths).
    """
    paths = transcript_cache_paths(output_prefix)

    with open(paths["speakers"], "w", encoding="utf-8") as f:
        f.write(f"Transcript: {Path(source_file).name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        if recording_speed != 1.0:
            f.write(f"Recording speed: {recording_speed}x (all timestamps in this file, "
                    f"the .srt, and the slide report are converted: "
                    f"real_time = video_time / {recording_speed})\n")
        f.write("=" * 60 + "\n\n")
        current_speaker = None
        for seg in segments:
            speaker = seg.get("speaker") or "SPEAKER"
            start, end = seg["start"], seg["end"]
            text = seg["text"].strip()
            ts = f"[{int(start//60):02d}:{int(start%60):02d} - {int(end//60):02d}:{int(end%60):02d}]"
            if speaker != current_speaker:
                f.write(f"\n{speaker}:\n")
                current_speaker = speaker
            f.write(f"  {ts} {text}\n")

    with open(paths["text"], "w", encoding="utf-8") as f:
        current_speaker = None
        for seg in segments:
            speaker = seg.get("speaker") or "SPEAKER"
            if speaker != current_speaker:
                f.write(f"\n{speaker}:\n")
                current_speaker = speaker
            f.write(f"  {seg['text'].strip()}\n")

    with open(paths["srt"], "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            s, e = seg["start"], seg["end"]
            speaker = seg.get("speaker", "")
            text = seg["text"].strip()
            f.write(f"{i}\n{_fmt_srt(s)} --> {_fmt_srt(e)}\n")
            f.write(f"[{speaker}] {text}\n\n" if speaker else f"{text}\n\n")

    log.info("Transcript saved: %s", paths["speakers"])
    return paths


def parse_srt(srt_path: str) -> list[dict]:
    """
    Parse a standard .srt subtitle file into the whisperx segment schema
    ({"start", "end", "text", "speaker", "words"}) used everywhere else in
    this pipeline (see the module docstring's three-step pipeline and
    write_transcript_files above). This lets an already-captioned video
    (e.g. a YouTube download that already has a description.txt + .srt +
    video file) skip whisperx transcription entirely via process_file's
    existing cache-aware shortcut - see run_pipeline.import_srt_transcript,
    which writes this function's output into the same *_segments.json +
    *_transcript_speakers.txt cache a real transcription run produces.

    Handles:
      - the standard "index / HH:MM:SS,mmm --> HH:MM:SS,mmm / text..."
        block form (comma OR period as the fractional separator - both
        appear in the wild, and YouTube's own downloads use comma).
      - multi-line cue text (joined into one line with a single space).
      - inline formatting/positioning tags such as <i>, <b>, <font ...>,
        {\an8}, and the per-word karaoke timing tags like
        <00:00:01,234> that YouTube's auto-generated captions embed -
        all stripped.
      - consecutive cues with byte-identical text after tag-stripping
        (common in YouTube's rolling auto-captions, where each new cue
        repeats the previous line verbatim plus new scrolling words)
        are collapsed into one, keeping the first cue's start time and
        the last's end time.

    speaker is always "" (plain .srt carries no speaker labels) and
    words is always [] (no word-level timing in an .srt) - both are
    already tolerated by every consumer of this schema (see
    align_transcript_to_slides, notes generation): they only ever read
    "start"/"end"/"text"/"speaker" with a get(..., default) fallback for
    the rest.

    Returns [] if the file has no parseable cues (wrong format, empty
    file, or a non-SRT file like a raw .vtt with a "WEBVTT" header and
    no HH:MM:SS,mmm timestamps) - callers must treat that as a hard
    failure, not a valid empty transcript.
    """
    raw = Path(srt_path).read_text(encoding="utf-8-sig")  # tolerate a BOM (common from Windows-saved .srt)
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    time_re = re.compile(r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})")
    tag_re = re.compile(r"<[^>]*>|\{[^}]*\}")

    def _to_sec(groups) -> float:
        h, m, s, ms = (int(x) for x in groups)
        return h * 3600 + m * 60 + s + ms / 1000.0

    segments: list[dict] = []
    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = [ln for ln in block.split("\n") if ln.strip() != ""]
        if not lines:
            continue
        idx = 1 if lines[0].strip().isdigit() else 0  # skip the numeric cue index, if present
        if idx >= len(lines):
            continue
        tm = re.search(
            r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})",
            lines[idx],
        )
        if not tm:
            continue  # not a cue block (e.g. a "WEBVTT" header line) - skip
        start = _to_sec(time_re.match(tm.group(1)).groups())
        end = _to_sec(time_re.match(tm.group(2)).groups())
        text = " ".join(tag_re.sub("", ln).strip() for ln in lines[idx + 1:])
        text = " ".join(text.split())
        if not text:
            continue
        if segments and segments[-1]["text"] == text:
            segments[-1]["end"] = end  # collapse duplicate rolling-caption cue
            continue
        segments.append({
            "start": round(start, 3), "end": round(end, 3),
            "text": text, "speaker": "", "words": [],
        })
    return segments


def read_transcript_text(speakers_file: str) -> str:
    """Read back a *_transcript_speakers.txt file as plain text for the LLM."""
    return Path(speakers_file).read_text(encoding="utf-8")
