# =============================================================
#  Transkription_Notes_Pipeline - tests/test_transcriber.py
#  Functional smoke tests for transcriber.enhance_audio (V1.24).
#  Uses ffmpeg's own lavfi test-source generator to make a short
#  synthetic tone, so these tests need only ffmpeg (already a
#  project dependency) and never touch whisperx/pyannote.
# =============================================================

import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest

import config
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

        # Mocking shutil.which alone is not enough: config.get_ffmpeg_path
        # checks the project's bundled ffmpeg\bin\ffmpeg.exe FIRST (see
        # config._find_bundled_tool), before ever falling back to PATH via
        # shutil.which - so as long as that bundled copy is actually
        # present in this checkout (it normally is), enhance_audio would
        # still find a usable ffmpeg and this test would not be exercising
        # the "nowhere to find ffmpeg" path it claims to. Patch
        # config.get_ffmpeg_path directly so this test is correct
        # regardless of whether a bundled copy happens to be checked in.
        monkeypatch.setattr(shutil, "which", lambda name: None)
        monkeypatch.setattr(config, "get_ffmpeg_path", lambda: None)

        ok = transcriber.enhance_audio(str(src), str(out))

        assert ok is False


# -----------------------------------------------------------------
# _build_diarization_pipeline: model_name pinning (2026-08 switch from
# pyannote/speaker-diarization-3.1 to community-1, see module docstring).
# whisperx.diarize is stubbed in sys.modules so this never imports the
# real whisperx/torch/pyannote stack - matches this file's own stated
# goal of never touching whisperx/pyannote for real.
# -----------------------------------------------------------------

class TestBuildDiarizationPipeline:
    def test_pins_community_1_model_name(self, monkeypatch):
        captured = {}

        class FakeDiarizationPipeline:
            def __init__(self, model_name=None, token=None, device=None):
                captured["model_name"] = model_name
                captured["token"] = token
                captured["device"] = device

        fake_module = types.SimpleNamespace(DiarizationPipeline=FakeDiarizationPipeline)
        monkeypatch.setitem(sys.modules, "whisperx.diarize", fake_module)

        transcriber._build_diarization_pipeline("hf_realtoken", "cpu")

        assert captured["model_name"] == "pyannote/speaker-diarization-community-1"
        assert captured["model_name"] == transcriber.DIARIZATION_MODEL
        assert captured["token"] == "hf_realtoken"
        assert captured["device"] == "cpu"
from pathlib import Path


# -----------------------------------------------------------------
# parse_srt: pure-Python .srt -> whisperx segment schema parser used by
# run_pipeline.import_srt_transcript (the "Import .srt as transcript"
# CLI flag / GUI button). No whisperx/ffmpeg dependency at all.
# -----------------------------------------------------------------

class TestParseSrt:
    def test_basic_two_cues(self, tmp_path):
        srt = tmp_path / "sample.srt"
        srt.write_text(
            "1\n00:00:00,000 --> 00:00:02,500\nHello there.\n\n"
            "2\n00:00:02,500 --> 00:00:05,000\nSecond line.\n",
            encoding="utf-8",
        )

        segments = transcriber.parse_srt(str(srt))

        assert len(segments) == 2
        assert segments[0] == {"start": 0.0, "end": 2.5, "text": "Hello there.",
                               "speaker": "", "words": []}
        assert segments[1]["text"] == "Second line."
        assert segments[1]["start"] == 2.5

    def test_multiline_cue_joined_with_space(self, tmp_path):
        srt = tmp_path / "sample.srt"
        srt.write_text(
            "1\n00:00:00,000 --> 00:00:03,000\nHello and welcome\nto this tutorial.\n",
            encoding="utf-8",
        )

        segments = transcriber.parse_srt(str(srt))

        assert len(segments) == 1
        assert segments[0]["text"] == "Hello and welcome to this tutorial."

    def test_strips_inline_formatting_and_karaoke_tags(self, tmp_path):
        srt = tmp_path / "sample.srt"
        srt.write_text(
            "1\n00:00:00,000 --> 00:00:02,000\n"
            "<00:00:00,100><c>Today</c> we will <i>learn</i> Python. {\\an8}\n",
            encoding="utf-8",
        )

        segments = transcriber.parse_srt(str(srt))

        assert len(segments) == 1
        assert segments[0]["text"] == "Today we will learn Python."

    def test_collapses_duplicate_rolling_caption_cues(self, tmp_path):
        srt = tmp_path / "sample.srt"
        srt.write_text(
            "1\n00:00:00,000 --> 00:00:02,000\nSame line here.\n\n"
            "2\n00:00:02,000 --> 00:00:04,000\nSame line here.\n\n"
            "3\n00:00:04,000 --> 00:00:06,000\nDifferent line.\n",
            encoding="utf-8",
        )

        segments = transcriber.parse_srt(str(srt))

        assert len(segments) == 2
        assert segments[0] == {"start": 0.0, "end": 4.0, "text": "Same line here.",
                               "speaker": "", "words": []}
        assert segments[1]["text"] == "Different line."

    def test_tolerates_missing_index_lines_and_bom(self, tmp_path):
        srt = tmp_path / "sample.srt"
        # No numeric index line before the timecode, plus a UTF-8 BOM
        # (common from Windows-saved .srt files).
        srt.write_bytes(
            "﻿00:00:00,000 --> 00:00:01,000\nNo index here.\n".encode("utf-8")
        )

        segments = transcriber.parse_srt(str(srt))

        assert len(segments) == 1
        assert segments[0]["text"] == "No index here."

    def test_period_fractional_separator_also_accepted(self, tmp_path):
        srt = tmp_path / "sample.srt"
        srt.write_text(
            "1\n00:00:00.000 --> 00:00:01.500\nPeriod separator.\n",
            encoding="utf-8",
        )

        segments = transcriber.parse_srt(str(srt))

        assert len(segments) == 1
        assert segments[0]["end"] == 1.5

    def test_no_cues_returns_empty_list(self, tmp_path):
        srt = tmp_path / "not_really_srt.srt"
        srt.write_text("WEBVTT\n\nJust some text, no timestamps.\n", encoding="utf-8")

        assert transcriber.parse_srt(str(srt)) == []

    def test_empty_file_returns_empty_list(self, tmp_path):
        srt = tmp_path / "empty.srt"
        srt.write_text("", encoding="utf-8")

        assert transcriber.parse_srt(str(srt)) == []
