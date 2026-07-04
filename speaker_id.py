# =============================================================
#  Transkription_Notes_Pipeline - speaker_id.py  (task #79)
#
#  Cross-meeting speaker recognition via voice embeddings ("voiceprints").
#  Builds on the existing diarization step (transcriber.py, requires
#  enable_diarization + hf_token): once whisperx/pyannote has split a
#  recording into per-segment SPEAKER_00/SPEAKER_01/... labels, this
#  module computes one embedding vector per local speaker and compares it
#  against a global roster stored in db.py's known_speakers table.
#
#  Library choice: pyannote.audio, NOT SpeechBrain or Resemblyzer. It is
#  already an installed dependency (whisperx pulls it in for diarization,
#  see requirements.txt), so speaker ID adds zero new pip packages. It
#  also reuses the exact hf_token/gated-model-acceptance flow already in
#  place for pyannote/speaker-diarization-3.1 (see config.py, transcriber.py
#  known-issues table): accept the terms once at
#  https://huggingface.co/pyannote/embedding, same as the diarization model.
#
#  Everything in this module that does NOT need torch/pyannote (vector
#  serialization, cosine similarity, threshold matching, running-average
#  updates) is plain numpy and fully unit-testable without either
#  installed - see tests/test_speaker_id.py. Only extract_speaker_embeddings
#  below imports pyannote.audio, and it does so lazily inside the function
#  body (same pattern as transcriber._import_whisperx), so importing this
#  module never fails just because pyannote/torch aren't installed yet.
# =============================================================

import logging

import numpy as np

log = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "pyannote/embedding"


def _import_pyannote_inference():
    """Lazy import with a clear error message if pyannote.audio is
    missing. Returns the Inference/Model classes, or None on failure."""
    try:
        from pyannote.audio import Inference, Model
        return Inference, Model
    except ImportError:
        log.error(
            "pyannote.audio not installed (should have come in automatically "
            "as a whisperx dependency - see requirements.txt). "
            "Run: C:\\Python\\Python311\\python.exe -m pip install pyannote.audio"
        )
        return None, None


_inference_cache: dict = {}


def _get_inference(hf_token: str, device: str = "cpu"):
    """Load (and cache) the pyannote embedding model. Raises RuntimeError
    with a clear message if pyannote.audio is missing or the model can't
    be loaded (most commonly: the gated model's terms haven't been
    accepted on HuggingFace yet, same fix as the diarization model)."""
    cache_key = (hf_token, device)
    if cache_key in _inference_cache:
        return _inference_cache[cache_key]

    Inference, Model = _import_pyannote_inference()
    if Inference is None:
        raise RuntimeError("pyannote.audio is not installed.")

    try:
        model = Model.from_pretrained(EMBEDDING_MODEL_NAME, use_auth_token=hf_token)
        inference = Inference(model, window="whole")
        inference.to_device = device  # best-effort; pyannote handles device internally too
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {EMBEDDING_MODEL_NAME}: {exc}. If this is a 401/403 "
            f"error, accept the model terms at "
            f"https://huggingface.co/{EMBEDDING_MODEL_NAME} with the same "
            f"HuggingFace account as HF_TOKEN, same as the diarization model."
        ) from exc

    _inference_cache[cache_key] = inference
    return inference


def extract_speaker_embeddings(audio: np.ndarray, segments: list[dict],
                                hf_token: str, sample_rate: int = 16000,
                                device: str = "cpu",
                                max_segments_per_speaker: int = 8) -> dict:
    """
    Compute one averaged voiceprint embedding per distinct speaker label
    found in segments, cropped directly out of the already-loaded audio
    waveform (the same array whisperx.load_audio produced for
    transcription/diarization - no second ffmpeg decode).

    segments: list of {start, end, speaker, ...} dicts (whisperx output
    format, see transcriber.transcribe).

    For each speaker, up to max_segments_per_speaker of their longest
    segments are embedded individually and averaged (L2-normalized before
    and after averaging), which is more robust to one noisy/short
    utterance than using a single segment.

    Returns {speaker_label: np.ndarray} for speakers with at least one
    embeddable segment (label omitted if pyannote failed on all its
    segments - see reannotate-style partial-failure tolerance used
    elsewhere in this project). Raises RuntimeError only if the model
    itself cannot be loaded at all (see _get_inference).
    """
    import torch
    from pyannote.core import Segment

    inference = _get_inference(hf_token, device)

    by_speaker: dict = {}
    for seg in segments:
        label = seg.get("speaker")
        if not label:
            continue
        by_speaker.setdefault(label, []).append(seg)

    waveform = torch.from_numpy(audio).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    file_dict = {"waveform": waveform, "sample_rate": sample_rate}
    audio_duration_sec = audio.shape[-1] / float(sample_rate)

    embeddings: dict = {}
    for label, segs in by_speaker.items():
        segs_sorted = sorted(segs, key=lambda s: s["end"] - s["start"], reverse=True)
        vectors = []
        for seg in segs_sorted[:max_segments_per_speaker]:
            start = max(0.0, float(seg["start"]))
            end = min(audio_duration_sec, float(seg["end"]))
            if end - start < 0.3:
                continue  # too short to embed reliably
            try:
                vec = inference.crop(file_dict, Segment(start, end))
                vec = np.asarray(vec).reshape(-1)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vectors.append(vec / norm)
            except Exception as exc:
                log.warning("Embedding failed for %s segment %.1f-%.1fs: %s",
                            label, start, end, exc)
        if vectors:
            mean_vec = np.mean(vectors, axis=0)
            norm = np.linalg.norm(mean_vec)
            embeddings[label] = mean_vec / norm if norm > 0 else mean_vec

    return embeddings


# =============================================================
# Pure numpy logic below - no torch/pyannote import, fully unit-testable.
# =============================================================

def serialize_embedding(vector: np.ndarray) -> bytes:
    """Encode an embedding vector as float32 bytes for the known_speakers
    BLOB column."""
    return np.asarray(vector, dtype=np.float32).tobytes()


def deserialize_embedding(data: bytes, dim: int) -> np.ndarray:
    """Decode a known_speakers BLOB back to a float32 vector. Raises
    ValueError if the byte length doesn't match dim (a mismatched
    embedding_dim column, e.g. after switching embedding models, should
    fail loudly rather than silently comparing incompatible vectors)."""
    vec = np.frombuffer(data, dtype=np.float32)
    if vec.shape[0] != dim:
        raise ValueError(
            f"Embedding blob has {vec.shape[0]} floats, expected {dim}. "
            f"This usually means the embedding model changed; re-enroll speakers."
        )
    return vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 if either vector is all
    zeros (avoids a division-by-zero rather than raising, since a
    degenerate embedding should just never match anything)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def match_speaker(embedding: np.ndarray, known_speakers: list[dict],
                   threshold: float) -> dict:
    """
    Compare embedding against every entry in known_speakers (each a dict
    with at least "speaker_id", "name", "embedding" (bytes),
    "embedding_dim" (int), as returned by db.list_known_speakers) and
    return the best match.

    Returns {"speaker_id": int|None, "name": str|None, "score": float}.
    speaker_id/name are None (score is still the best score found, or 0.0
    if known_speakers is empty) when nothing clears threshold - the
    caller treats that as "new speaker", but can still show the score for
    transparency ("closest match: Anna, 0.61, below your 0.75 threshold").
    Skips any known-speaker row whose embedding_dim doesn't match
    embedding's length instead of raising, so one stale/incompatible
    entry can't block matching against the rest of the roster.
    """
    embedding = np.asarray(embedding, dtype=np.float64)
    best_score = 0.0
    best_id = None
    best_name = None
    for row in known_speakers:
        dim = row.get("embedding_dim")
        if dim is not None and dim != embedding.shape[0]:
            continue
        try:
            known_vec = deserialize_embedding(row["embedding"], dim) if dim is not None \
                else np.frombuffer(row["embedding"], dtype=np.float32)
        except ValueError:
            continue
        score = cosine_similarity(embedding, known_vec)
        if score > best_score:
            best_score = score
            best_id = row.get("speaker_id")
            best_name = row.get("name")
    if best_score >= threshold:
        return {"speaker_id": best_id, "name": best_name, "score": best_score}
    return {"speaker_id": None, "name": None, "score": best_score}


def update_running_average(old_vector: np.ndarray, old_sample_count: int,
                            new_vector: np.ndarray) -> tuple:
    """
    Fold a newly-confirmed sample into an existing known speaker's stored
    voiceprint: a plain running average weighted by how many samples
    already contributed, re-normalized to unit length afterward (the
    vectors are cosine-compared, so keeping them unit-norm keeps
    cosine_similarity well-behaved). Returns (new_vector, new_sample_count).

    old_sample_count must be >= 1 (a known speaker always has at least
    its original enrollment sample).
    """
    if old_sample_count < 1:
        raise ValueError("old_sample_count must be >= 1")
    old_vector = np.asarray(old_vector, dtype=np.float64)
    new_vector = np.asarray(new_vector, dtype=np.float64)
    combined = (old_vector * old_sample_count + new_vector) / (old_sample_count + 1)
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm
    return combined, old_sample_count + 1
