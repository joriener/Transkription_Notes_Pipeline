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
) -> list[SlideChange]:
    """
    Walk through frames and return a SlideChange for each detected transition.
    First frame is always included as the opening slide.
    """
    if not frames:
        log.warning("No frames provided to detector.")
        return []

    changes: list[SlideChange] = []
    prev_hash = None
    last_change_frame = -1
    min_gap_frames = max(1, int(min_slide_duration_sec * fps))

    for idx, frame_path in enumerate(frames):
        timestamp = idx / fps

        try:
            current_hash = compute_hash(frame_path, algorithm)
        except Exception as exc:
            log.warning("Skipping frame %d (%s): %s", idx, frame_path.name, exc)
            continue

        if prev_hash is None:
            changes.append(SlideChange(
                frame_path=frame_path,
                frame_index=idx,
                timestamp_sec=timestamp,
                hash_value=str(current_hash),
                hamming_distance=0,
                is_first=True,
            ))
            last_change_frame = idx
            prev_hash = current_hash
            continue

        distance = current_hash - prev_hash
        gap = idx - last_change_frame

        if distance >= threshold and gap >= min_gap_frames:
            log.debug(
                "Slide change at frame %d (t=%.1fs) distance=%d",
                idx, timestamp, distance,
            )
            changes.append(SlideChange(
                frame_path=frame_path,
                frame_index=idx,
                timestamp_sec=timestamp,
                hash_value=str(current_hash),
                hamming_distance=distance,
            ))
            last_change_frame = idx
            prev_hash = current_hash
        else:
            prev_hash = current_hash

    log.info("Detected %d slide changes in %d frames", len(changes), len(frames))
    return changes
