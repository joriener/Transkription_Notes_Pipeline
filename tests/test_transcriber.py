# =============================================================
#  Transkription_Notes_Pipeline - tests/test_transcriber.py
#  Functional smoke tests for transcriber.enhance_audio (V1.24).
#  Uses ffmpeg's own lavfi test-source generator to make a short
#  synthetic tone, so these tests need only ffmpeg (already a
#  project dependency) and never touch whisperx/pyannote.
# =============================================================

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

import transcriber

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


def _make_test_tone(path: Path, duration: float = 1.0) -> None:
    """Generate a short synthetic sine-wave WAV via ffmpeg's lavfi input,
    standing in for a real recording without needing any test fixture
    audio file checked into the repo."""
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"sine=frequency=440:duration={duration}",
        str(path),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
class TestEnhanceAudio:
    def test_produces_valid_nonempty_output(self, tmp_path):
        src = tmp_path / "tone.wav"
        out = tmp_path / "tone_enhanced.wav"
        _make_test_tone(src)

        ok = transcriber.enhance_audio(str(src), str(out))

        assert ok is True
        assert out.exists()
        assert out.stat().st_size > 0

    def test_output_is_16k_mono(self, tmp_path):
        src = tmp_path / "tone.wav"
        out = tmp_path / "tone_enhanced.wav"
        _make_test_tone(src)

        transcriber.enhance_audio(str(src), str(out))

        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate,channels",
             "-of", "csv=p=0", str(out)],
            capture_output=True, text=True,
        )
        sample_rate, channels = probe.stdout.strip().split(",")
        assert sample_rate == "16000"
        assert channels == "1"

    def test_never_modifies_input_file(self, tmp_path):
        src = tmp_path / "tone.wav"
        out = tmp_path / "tone_enhanced.wav"
        _make_test_tone(src)
        original_bytes = src.read_bytes()

        transcriber.enhance_audio(str(src), str(out))

        assert src.read_bytes() == original_bytes

    def test_missing_input_returns_false_not_raise(self, tmp_path):
        missing = tmp_path / "does_not_exist.wav"
        out = tmp_path / "out.wav"

        ok = transcriber.enhance_audio(str(missing), str(out))

        assert ok is False

    def test_ffmpeg_missing_returns_false(self, tmp_path, monkeypatch):
        src = tmp_path / "tone.wav"
        out = tmp_path / "tone_enhanced.wav"
        _make_test_tone(src)

        monkeypatch.setattr(shutil, "which", lambda name: None)

        ok = transcriber.enhance_audio(str(src), str(out))

        assert ok is False
