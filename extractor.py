# Transkription_Notes_Pipeline - extractor.py
# Extracts frames from a video at a given fps using ffmpeg subprocess calls.
# Requires ffmpeg to be available in PATH. Video mode only.

import subprocess
import logging
from pathlib import Path

import config

log = logging.getLogger(__name__)


def check_ffmpeg() -> bool:
    """Verify ffmpeg is available (bundled ffmpeg\\bin\\ or PATH, see
    config.get_ffmpeg_path) before attempting extraction."""
    if config.get_ffmpeg_path() is None:
        log.error(
            "ffmpeg not found. Drop a build into ffmpeg\\bin\\ next to the "
            "project, or install from https://ffmpeg.org / add to system PATH."
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
    cmd = [config.get_ffmpeg_path(), "-y", "-i", str(video_path), "-vf", f"fps={fps}"]
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
    ffprobe_exe = config.get_ffprobe_path()
    if ffprobe_exe is None:
        log.warning("ffprobe not found; duration will be estimated from frame count.")
        return 0.0

    cmd = [
        ffprobe_exe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Real-time (1x) speed conversion (optional, opt-in).
#
# CONFIG["recording_speed"] converts every REPORTED timestamp (transcript,
# .srt, slide report) via real_time = video_time / recording_speed, but the
# source video file itself keeps playing at its own original speed - so a
# viewer watching the raw file alongside the already-converted .srt would
# see them drift out of sync. This section optionally re-encodes the video
# itself so its own internal timeline matches real time too, staying in
# sync with the already-converted subtitles/timestamps without any further
# scaling needed.
# ---------------------------------------------------------------------------

def has_audio_stream(video_path: str) -> bool:
    """True if video_path has at least one audio stream (via ffprobe)."""
    ffprobe_exe = config.get_ffprobe_path()
    if ffprobe_exe is None:
        log.warning("ffprobe not found; assuming an audio stream is present.")
        return True
    cmd = [
        ffprobe_exe, "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return bool(result.stdout.strip())


def _build_atempo_chain(factor: float) -> str:
    """
    Build a chained ffmpeg atempo filter string for an arbitrary speed
    factor. A single atempo instance is only reliably supported for
    [0.5, 2.0] across ffmpeg versions, so factors outside that range are
    decomposed into multiple chained stages whose product equals factor
    (e.g. 4.0 -> "atempo=2,atempo=2").
    """
    if factor <= 0:
        raise ValueError(f"Speed factor must be positive, got {factor}")
    stages: list[float] = []
    remaining = factor
    while remaining > 2.0:
        stages.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        stages.append(0.5)
        remaining /= 0.5
    stages.append(remaining)
    return ",".join(f"atempo={s:.6g}" for s in stages)


def convert_to_realtime_speed(video_path: str, output_path: Path, speed: float) -> bool:
    """
    Re-encode video_path so its own internal timeline matches real time,
    undoing the effect described by CONFIG["recording_speed"]. Since every
    other output already uses real_time = video_time / speed (see
    run_pipeline._rescale_segments / _rescale_slide_changes), this file
    must play `speed` times FASTER than the source: its total duration
    shrinks from D to D/speed, which is exactly the timeline the already-
    converted .srt/transcript/slide report expect. Requires a full
    re-encode (no stream copy is possible when the timing itself changes):
      video: setpts=PTS/speed
      audio: chained atempo filters (pitch-preserving time-stretch)
    Returns True on success, False on failure or if speed == 1.0 (nothing
    to convert).
    """
    if speed == 1.0:
        log.info("recording_speed is 1.0 - nothing to convert to real-time speed.")
        return False
    if not check_ffmpeg():
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_present = has_audio_stream(video_path)

    cmd = [config.get_ffmpeg_path(), "-y", "-i", str(video_path), "-filter:v", f"setpts=PTS/{speed}"]
    if audio_present:
        cmd += ["-filter:a", _build_atempo_chain(speed)]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    if audio_present:
        cmd += ["-c:a", "aac"]
    cmd += [str(output_path)]

    log.info("Converting to real-time (1x) speed: %s -> %s (factor %.4g)",
             video_path, output_path, speed)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        log.error("ffmpeg speed-conversion error:\n%s", result.stderr[-2000:])
        return False

    log.info("Real-time-speed video saved: %s", output_path)
    return True
