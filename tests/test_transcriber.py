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


# -----------------------------------------------------------------
# compute_type resolution and the batch model cache
#
# Both use a fake whisperx installed over transcriber._import_whisperx, so
# no real model is ever loaded and these run fine in an environment with no
# torch or whisperx installed at all.
# -----------------------------------------------------------------

class _FakeModel:
    def __init__(self, calls):
        self._calls = calls
        self.options = types.SimpleNamespace(initial_prompt="stale prompt from a previous file")

    def transcribe(self, audio, batch_size=16):
        self._calls.append(("transcribe", batch_size))
        return {"language": "en",
                "segments": [{"start": 0.0, "end": 1.0, "text": " hello "}]}


def _install_fake_whisperx(monkeypatch, calls):
    """Returns the calls list, which records every load and transcribe, so a
    test can assert how often the weights were actually loaded."""
    def load_model(model_size, device, compute_type=None, language=None):
        calls.append(("load_model", model_size, device, compute_type, language))
        return _FakeModel(calls)

    def load_align_model(language_code=None, device=None):
        calls.append(("load_align_model", language_code, device))
        return ("ALIGN_MODEL", {"language": language_code})

    def align(segments, model_a, metadata, audio, device, return_char_alignments=False):
        calls.append(("align", device))
        return {"segments": segments}

    fake = types.SimpleNamespace(
        load_model=load_model,
        load_audio=lambda path: "AUDIO",
        load_align_model=load_align_model,
        align=align,
    )
    monkeypatch.setattr(transcriber, "_import_whisperx", lambda: fake)
    # Never let a real GPU probe decide the outcome of these tests.
    monkeypatch.setattr(transcriber, "_resolve_device", lambda device: ("cpu", "int8"))
    return calls


class TestResolveDevice:
    def test_explicit_cuda_suggests_float16(self):
        assert transcriber._resolve_device("cuda") == ("cuda", "float16")

    def test_explicit_cpu_suggests_int8(self):
        assert transcriber._resolve_device("cpu") == ("cpu", "int8")


class TestComputeTypeIsHonoured:
    def test_auto_uses_the_device_suggestion(self, tmp_path, monkeypatch):
        calls = _install_fake_whisperx(monkeypatch, [])
        transcriber.transcribe(str(tmp_path / "x.wav"), compute_type="auto",
                               use_vocabulary=False)
        assert next(c for c in calls if c[0] == "load_model")[3] == "int8"

    def test_blank_uses_the_device_suggestion(self, tmp_path, monkeypatch):
        calls = _install_fake_whisperx(monkeypatch, [])
        transcriber.transcribe(str(tmp_path / "x.wav"), compute_type="",
                               use_vocabulary=False)
        assert next(c for c in calls if c[0] == "load_model")[3] == "int8"

    def test_explicit_value_survives(self, tmp_path, monkeypatch):
        """The regression this pins: _resolve_device used to overwrite the
        caller's compute_type unconditionally, so config's
        whisper_compute_type never reached whisperx at all and
        int8_float16 on GPU was unreachable."""
        calls = _install_fake_whisperx(monkeypatch, [])
        transcriber.transcribe(str(tmp_path / "x.wav"), compute_type="int8_float16",
                               use_vocabulary=False)
        assert next(c for c in calls if c[0] == "load_model")[3] == "int8_float16"

    def test_float16_on_cpu_is_honoured_but_warned_about(self, tmp_path, monkeypatch, caplog):
        calls = _install_fake_whisperx(monkeypatch, [])
        with caplog.at_level("WARNING"):
            transcriber.transcribe(str(tmp_path / "x.wav"), compute_type="float16",
                                   use_vocabulary=False)
        assert next(c for c in calls if c[0] == "load_model")[3] == "float16"
        assert "float16 on CPU" in caplog.text


class TestModelCache:
    def _enable(self, monkeypatch):
        """Fresh, isolated caches. monkeypatch restores the real dicts and the
        flag afterwards, so nothing leaks into later tests in the session."""
        monkeypatch.setattr(transcriber, "_ASR_CACHE", {})
        monkeypatch.setattr(transcriber, "_ALIGN_CACHE", {})
        monkeypatch.setattr(transcriber, "_DIARIZE_CACHE", {})
        monkeypatch.setattr(transcriber, "_CACHE_ENABLED", True)

    def test_disabled_by_default_reloads_per_file(self, tmp_path, monkeypatch):
        calls = _install_fake_whisperx(monkeypatch, [])
        monkeypatch.setattr(transcriber, "_CACHE_ENABLED", False)
        for name in ("a.wav", "b.wav"):
            transcriber.transcribe(str(tmp_path / name), use_vocabulary=False)
        assert len([c for c in calls if c[0] == "load_model"]) == 2

    def test_enabled_loads_the_asr_model_once_for_two_files(self, tmp_path, monkeypatch):
        calls = _install_fake_whisperx(monkeypatch, [])
        self._enable(monkeypatch)
        for name in ("a.wav", "b.wav"):
            transcriber.transcribe(str(tmp_path / name), use_vocabulary=False)
        assert len([c for c in calls if c[0] == "load_model"]) == 1
        assert len([c for c in calls if c[0] == "transcribe"]) == 2

    def test_enabled_loads_the_alignment_model_once(self, tmp_path, monkeypatch):
        calls = _install_fake_whisperx(monkeypatch, [])
        self._enable(monkeypatch)
        for name in ("a.wav", "b.wav"):
            transcriber.transcribe(str(tmp_path / name), use_vocabulary=False)
        assert len([c for c in calls if c[0] == "load_align_model"]) == 1

    def test_a_different_model_size_is_cached_separately(self, tmp_path, monkeypatch):
        calls = _install_fake_whisperx(monkeypatch, [])
        self._enable(monkeypatch)
        transcriber.transcribe(str(tmp_path / "a.wav"), model_size="large-v3",
                               use_vocabulary=False)
        transcriber.transcribe(str(tmp_path / "b.wav"), model_size="medium",
                               use_vocabulary=False)
        assert len([c for c in calls if c[0] == "load_model"]) == 2

    def test_a_different_language_is_cached_separately(self, tmp_path, monkeypatch):
        """wx.load_model bakes the language into the model, so a batch mixing
        languages must not reuse the wrong one."""
        calls = _install_fake_whisperx(monkeypatch, [])
        self._enable(monkeypatch)
        transcriber.transcribe(str(tmp_path / "a.wav"), language="de", use_vocabulary=False)
        transcriber.transcribe(str(tmp_path / "b.wav"), language="en", use_vocabulary=False)
        assert len([c for c in calls if c[0] == "load_model"]) == 2

    def test_stale_vocabulary_prompt_is_cleared_on_reuse(self, tmp_path, monkeypatch):
        """A cached model keeps whatever initial_prompt the previous file set.
        Leaving it would silently bias the next transcription, so transcribe
        clears it when no vocabulary is requested."""
        captured = {}
        calls = []

        def load_model(model_size, device, compute_type=None, language=None):
            model = _FakeModel(calls)
            captured["model"] = model
            return model

        fake = types.SimpleNamespace(
            load_model=load_model,
            load_audio=lambda path: "AUDIO",
            load_align_model=lambda language_code=None, device=None: ("M", {}),
            align=lambda *a, **kw: {"segments": []},
        )
        monkeypatch.setattr(transcriber, "_import_whisperx", lambda: fake)
        monkeypatch.setattr(transcriber, "_resolve_device", lambda d: ("cpu", "int8"))
        self._enable(monkeypatch)

        transcriber.transcribe(str(tmp_path / "a.wav"), use_vocabulary=False)
        assert captured["model"].options.initial_prompt is None

    def test_release_models_empties_every_cache(self, tmp_path, monkeypatch):
        _install_fake_whisperx(monkeypatch, [])
        self._enable(monkeypatch)
        transcriber.transcribe(str(tmp_path / "a.wav"), use_vocabulary=False)
        assert transcriber._ASR_CACHE and transcriber._ALIGN_CACHE
        transcriber.release_models()
        assert transcriber._ASR_CACHE == {}
        assert transcriber._ALIGN_CACHE == {}
        assert transcriber._DIARIZE_CACHE == {}

    def test_release_models_is_safe_when_nothing_is_cached(self, monkeypatch):
        self._enable(monkeypatch)
        transcriber.release_models()
        transcriber.release_models()

    def test_enable_model_cache_false_releases(self, tmp_path, monkeypatch):
        _install_fake_whisperx(monkeypatch, [])
        self._enable(monkeypatch)
        transcriber.transcribe(str(tmp_path / "a.wav"), use_vocabulary=False)
        assert transcriber._ASR_CACHE
        transcriber.enable_model_cache(False)
        assert transcriber._ASR_CACHE == {}
