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
#  Supported languages for alignment (auto-loaded from torchaudio/HF):
#    en, de, fr, es, it, ja, zh, nl, uk, pt
# =============================================================

import gc
import logging
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
                # Correct API for whisperx 3.8.x - see module docstring.
                from whisperx.diarize import DiarizationPipeline
                diarize_model = DiarizationPipeline(token=hf_token, device=device)
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


def read_transcript_text(speakers_file: str) -> str:
    """Read back a *_transcript_speakers.txt file as plain text for the LLM."""
    return Path(speakers_file).read_text(encoding="utf-8")
