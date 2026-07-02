# Transkription_Notes_Pipeline - extractor.py
# Extracts frames from a video at a given fps using ffmpeg subprocess calls.
# Requires ffmpeg to be available in PATH. Video mode only.

import subprocess
import shutil
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def check_ffmpeg() -> bool:
    """Verify ffmpeg is available before attempting extraction."""
    if shutil.which("ffmpeg") is None:
        log.error(
            "ffmpeg not found in PATH. "
            "Install from https://ffmpeg.org or add to system PATH."
        )
        return False
    return True


def extract_frames(
    video_path: str,
    output_dir: Path,
    fps: int = 1,
    fmt: str = "jpg",
    quality: int = 90,
) -> list[Path]:
    """
    Extract frames from video_path at the given fps.
    Returns a sorted list of extracted frame Paths.
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is required but not found.")

    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / f"frame_%06d.{fmt}"

    # Build ffmpeg command
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vf", f"fps={fps}"]
    if fmt == "jpg":
        cmd += ["-q:v", str(max(2, int(100 - quality) // 3))]  # ffmpeg q:v scale 2-31
    cmd += [str(pattern)]

    log.info("Extracting frames: %s -> %s at %d fps", video_path, output_dir, fps)
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        log.error("ffmpeg error:\n%s", result.stderr[-2000:])
        raise RuntimeError(f"ffmpeg exited with code {result.returncode}")

    frames = sorted(output_dir.glob(f"frame_*.{fmt}"))
    log.info("Extracted %d frames", len(frames))
    return frames


def get_video_duration_sec(video_path: str) -> float:
    """Return video duration in seconds using ffprobe."""
    if shutil.which("ffprobe") is None:
        log.warning("ffprobe not found; duration will be estimated from frame count.")
        return 0.0

    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0
