# Transkription_Notes_Pipeline - detector.py
# Perceptual hash-based slide-change detection. Video mode only.
# Primary: imagehash 4.3.2 (phash/dhash/whash/ahash)

import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import imagehash

log = logging.getLogger(__name__)

HASH_FUNCTIONS = {
    "phash": imagehash.phash,
    "dhash": imagehash.dhash,
    "whash": imagehash.whash,
    "ahash": imagehash.average_hash,
}


@dataclass
class SlideChange:
    frame_path: Path
    frame_index: int
    timestamp_sec: float
    hash_value: str
    hamming_distance: int
    is_first: bool = False


def compute_hash(image_path: Path, algorithm: str = "phash") -> imagehash.ImageHash:
    fn = HASH_FUNCTIONS.get(algorithm)
    if fn is None:
        raise ValueError(f"Unknown hash algorithm: {algorithm}. Choose from {list(HASH_FUNCTIONS)}")
    with Image.open(image_path) as img:
        return fn(img)


def detect_slide_changes(
    frames: list[Path],
    fps: int = 1,
    threshold: int = 8,
    algorithm: str = "phash",
    min_slide_duration_sec: float = 2.0,
    animation_threshold: int = 0,
    frame_indices: list[int] | None = None,
) -> list[SlideChange]:
    """
    Walk through frames and return a SlideChange for each detected transition.
    First frame is always included as the opening slide.

    frame_indices (optional): each frame's ORIGINAL position in the extracted
    sequence, same length and order as `frames`. Defaults to
    range(len(frames)), i.e. "the list is the whole sequence".

    Pass it whenever the caller has removed frames from the middle of the
    sequence before calling, which run_pipeline does for a configured Q&A
    range (qa_start_time_sec/qa_end_time_sec). Without it, timestamps are
    derived from the position in the list that was handed over, so every
    slide after an excised block is reported earlier than it really was, by
    exactly the excised duration. That stayed invisible while qa_end was
    normally unset, because then only trailing frames are dropped and no
    surviving frame follows the gap.

    Note that the min_slide_duration_sec debounce below still counts
    OBSERVED frames, not original indices: it asks "did this stay different
    across enough frames I actually looked at", which is the noise question
    it was written to answer. Using original indices there would let a
    candidate spanning an excised block be confirmed on the strength of time
    that was never examined.

    Noise filter (confirm-then-commit debounce): a frame is only accepted as
    a real change once it has stayed visually different from the current
    confirmed baseline for at least min_slide_duration_sec. A transient
    difference that reverts back to the baseline before that (e.g. a brief
    flash of any colour, covering the whole screen, lasting a second or
    less) is discarded as noise and never appears in the output.

    Animation vs. new slide (optional; active when 0 < animation_threshold
    < threshold): once a change is confirmed stable, its distance from the
    baseline decides what it means:
      - distance >= threshold: a genuine new slide. A new SlideChange is
        appended and the baseline resets to this frame (previous behavior).
      - animation_threshold <= distance < threshold: the current slide is
        still building (e.g. bullets appearing one at a time, a build
        animation). No new SlideChange is appended. Instead the CURRENT
        slide's stored snapshot (frame_path, hash_value) is updated to
        this more complete frame, while its recorded start timestamp is
        left unchanged, so the final report shows the fully-built slide
        at the time it first appeared. The baseline updates to this frame
        so later animation steps are measured incrementally from here.
    When animation_threshold is 0 (default) or >= threshold, this
    distinction is disabled and every confirmed change is a new slide,
    identical to the previous behavior.

    Compares every candidate frame against the last CONFIRMED baseline's
    hash (not the immediately preceding frame), so a single-frame outlier
    cannot silently become the new comparison baseline the way frame-to-
    frame comparison would allow.
    """
    if not frames:
        log.warning("No frames provided to detector.")
        return []

    if frame_indices is None:
        frame_numbers = list(range(len(frames)))
    else:
        frame_numbers = list(frame_indices)
        if len(frame_numbers) != len(frames):
            raise ValueError(
                f"frame_indices has {len(frame_numbers)} entries but frames has "
                f"{len(frames)} - they must line up one to one."
            )

    animation_enabled = 0 < animation_threshold < threshold
    low_threshold = animation_threshold if animation_enabled else threshold

    changes: list[SlideChange] = []
    baseline_hash = None
    # (hash, candidate list position, candidate original frame number,
    #  frame path, distance from baseline)
    candidate = None
    min_gap_frames = max(1, int(min_slide_duration_sec * fps))

    for pos, frame_path in enumerate(frames):
        real_idx = frame_numbers[pos]
        timestamp = real_idx / fps

        try:
            current_hash = compute_hash(frame_path, algorithm)
        except Exception as exc:
            log.warning("Skipping frame %d (%s): %s", real_idx, frame_path.name, exc)
            continue

        if baseline_hash is None:
            changes.append(SlideChange(
                frame_path=frame_path,
                frame_index=real_idx,
                timestamp_sec=timestamp,
                hash_value=str(current_hash),
                hamming_distance=0,
                is_first=True,
            ))
            baseline_hash = current_hash
            continue

        distance_to_baseline = current_hash - baseline_hash

        if candidate is None:
            if distance_to_baseline >= low_threshold:
                # Tentative change starts here - not yet confirmed.
                candidate = (current_hash, pos, real_idx, frame_path, distance_to_baseline)
            # else: still matches the confirmed baseline, nothing to do.
            continue

        cand_hash, cand_pos, cand_real_idx, cand_frame_path, cand_distance = candidate

        if distance_to_baseline < low_threshold:
            # Reverted back to the confirmed baseline before the candidate
            # was ever confirmed - it was noise (flash/glitch). Drop it.
            log.debug(
                "Discarded noise candidate starting at frame %d (t=%.1fs): "
                "reverted to previous state after %d frame(s).",
                cand_real_idx, cand_real_idx / fps, pos - cand_pos,
            )
            candidate = None
            continue

        if pos - cand_pos + 1 >= min_gap_frames:
            # Candidate has now persisted long enough to count as real.
            if animation_enabled and cand_distance < threshold and changes:
                # Same slide, still building - update the current slide's
                # snapshot to the more complete frame, but its recorded
                # start timestamp stays unchanged.
                log.debug(
                    "Animation update on current slide at frame %d (t=%.1fs) "
                    "distance=%d (stable for %d frame(s)); start timestamp "
                    "unchanged.",
                    cand_real_idx, cand_real_idx / fps, cand_distance, pos - cand_pos + 1,
                )
                changes[-1].frame_path = cand_frame_path
                changes[-1].hash_value = str(cand_hash)
                baseline_hash = cand_hash
                candidate = None
            else:
                # Genuine new slide. Record it using the FIRST frame of the
                # stable window (when the change actually began), not the
                # confirmation frame, so the reported timestamp matches
                # when the slide actually appeared.
                log.debug(
                    "Slide change confirmed at frame %d (t=%.1fs) distance=%d "
                    "(stable for %d frame(s)).",
                    cand_real_idx, cand_real_idx / fps, cand_distance, pos - cand_pos + 1,
                )
                changes.append(SlideChange(
                    frame_path=cand_frame_path,
                    frame_index=cand_real_idx,
                    timestamp_sec=cand_real_idx / fps,
                    hash_value=str(cand_hash),
                    hamming_distance=cand_distance,
                ))
                baseline_hash = current_hash
                candidate = None
        else:
            # Still waiting for confirmation; track the latest hash so a
            # slow fade-in/transition doesn't get stuck comparing against
            # its own first, possibly unrepresentative, frame.
            candidate = (current_hash, cand_pos, cand_real_idx, cand_frame_path,
                         distance_to_baseline)

    # A candidate still pending when the video ends never had the chance to
    # either revert (proving it was noise) or reach min_slide_duration_sec
    # (proving it was real). It never reverted within the observed frames,
    # so it is trusted and recorded rather than silently dropped - a slide
    # or animation step that appears right at the end of a recording should
    # still show up.
    if candidate is not None:
        cand_hash, cand_pos, cand_real_idx, cand_frame_path, cand_distance = candidate
        if animation_enabled and cand_distance < threshold and changes:
            log.debug(
                "Animation update on current slide at end of video, frame %d "
                "(t=%.1fs) distance=%d (never reverted).",
                cand_real_idx, cand_real_idx / fps, cand_distance,
            )
            changes[-1].frame_path = cand_frame_path
            changes[-1].hash_value = str(cand_hash)
        else:
            log.debug(
                "Slide change at frame %d (t=%.1fs) confirmed at end of video "
                "(never reverted, but video ended before min_slide_duration_sec elapsed).",
                cand_real_idx, cand_real_idx / fps,
            )
            changes.append(SlideChange(
                frame_path=cand_frame_path,
                frame_index=cand_real_idx,
                timestamp_sec=cand_real_idx / fps,
                hash_value=str(cand_hash),
                hamming_distance=cand_distance,
            ))

    log.info("Detected %d slide changes in %d frames", len(changes), len(frames))
    return changes
