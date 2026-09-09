# =============================================================
#  Transkription_Notes_Pipeline - tests/test_config.py
#  Unit tests for config.py's bundled-ffmpeg resolution helpers (V1.27):
#  get_ffmpeg_path/get_ffplay_path/get_ffprobe_path prefer a portable
#  ffmpeg\bin\ dropped next to the project over a system PATH install,
#  falling back to PATH and finally to None. Monkeypatches
#  config._BUNDLED_FFMPEG_DIR and config.shutil.which so these don't
#  depend on this project's actual ffmpeg\bin\ folder (present or not)
#  or the test host's real PATH.
#
#  Run: pytest tests/test_config.py -v
# =============================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


class TestBundledFfmpegPreferred:
    def test_prefers_bundled_ffmpeg_over_path(self, tmp_path, monkeypatch):
        bundled_dir = tmp_path / "ffmpeg" / "bin"
        bundled_dir.mkdir(parents=True)
        fake_exe = bundled_dir / "ffmpeg.exe"
        fake_exe.write_text("fake")
        monkeypatch.setattr(config, "_BUNDLED_FFMPEG_DIR", bundled_dir)
        monkeypatch.setattr(config.shutil, "which", lambda name: f"/usr/bin/{name}")

        assert config.get_ffmpeg_path() == str(fake_exe)

    def test_bundled_non_exe_name_also_matches(self, tmp_path, monkeypatch):
        # In case a non-Windows build (no .exe suffix) is ever dropped in.
        bundled_dir = tmp_path / "ffmpeg" / "bin"
        bundled_dir.mkdir(parents=True)
        fake_exe = bundled_dir / "ffmpeg"
        fake_exe.write_text("fake")
        monkeypatch.setattr(config, "_BUNDLED_FFMPEG_DIR", bundled_dir)
        monkeypatch.setattr(config.shutil, "which", lambda name: None)

        assert config.get_ffmpeg_path() == str(fake_exe)

    def test_ffplay_and_ffprobe_resolve_independently(self, tmp_path, monkeypatch):
        bundled_dir = tmp_path / "ffmpeg" / "bin"
        bundled_dir.mkdir(parents=True)
        (bundled_dir / "ffplay.exe").write_text("fake")
        (bundled_dir / "ffprobe.exe").write_text("fake")
        monkeypatch.setattr(config, "_BUNDLED_FFMPEG_DIR", bundled_dir)
        monkeypatch.setattr(config.shutil, "which", lambda name: None)

        assert config.get_ffplay_path() == str(bundled_dir / "ffplay.exe")
        assert config.get_ffprobe_path() == str(bundled_dir / "ffprobe.exe")
        # ffmpeg.exe itself was never created in this bundled dir, and
        # PATH is stubbed to nothing - must not accidentally cross-match
        # one tool's file for another.
        assert config.get_ffmpeg_path() is None


class TestFallsBackToPath:
    def test_path_used_when_bundled_missing_entirely(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "_BUNDLED_FFMPEG_DIR", tmp_path / "no_such_dir" / "bin")
        monkeypatch.setattr(config.shutil, "which",
                             lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)

        assert config.get_ffmpeg_path() == "/usr/bin/ffmpeg"

    def test_path_used_when_bundled_dir_exists_but_empty(self, tmp_path, monkeypatch):
        bundled_dir = tmp_path / "ffmpeg" / "bin"
        bundled_dir.mkdir(parents=True)  # exists, but no ffmpeg(.exe) inside
        monkeypatch.setattr(config, "_BUNDLED_FFMPEG_DIR", bundled_dir)
        monkeypatch.setattr(config.shutil, "which",
                             lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)

        assert config.get_ffmpeg_path() == "/usr/bin/ffmpeg"


class TestNeitherAvailable:
    def test_returns_none_when_bundled_and_path_both_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "_BUNDLED_FFMPEG_DIR", tmp_path / "no_such_dir" / "bin")
        monkeypatch.setattr(config.shutil, "which", lambda name: None)

        assert config.get_ffmpeg_path() is None
        assert config.get_ffplay_path() is None
        assert config.get_ffprobe_path() is None
