# =============================================================
#  Transkription_Notes_Pipeline - tests/test_speaker_id.py
#  Unit tests for speaker_id.py's pure-logic helpers (task #79):
#  serialization, cosine similarity, threshold matching, and the
#  running-average enrollment update. Deliberately excludes
#  extract_speaker_embeddings, which needs torch/pyannote.audio and
#  real audio - not available in the CI sandbox, and not pure logic
#  anyway (see module docstring in speaker_id.py).
#
#  Run: pytest tests/test_speaker_id.py -v
# =============================================================

import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import speaker_id


def unit(vec):
    vec = np.asarray(vec, dtype=np.float64)
    return vec / np.linalg.norm(vec)


# -----------------------------------------------------------------
# serialize_embedding / deserialize_embedding
# -----------------------------------------------------------------

class TestSerializeRoundtrip:
    def test_roundtrip_preserves_values(self):
        vec = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
        data = speaker_id.serialize_embedding(vec)
        restored = speaker_id.deserialize_embedding(data, 4)
        assert np.allclose(vec, restored, atol=1e-6)

    def test_serialize_casts_to_float32(self):
        vec = np.array([1, 2, 3], dtype=np.float64)
        data = speaker_id.serialize_embedding(vec)
        assert len(data) == 3 * 4  # float32 = 4 bytes each

    def test_deserialize_wrong_dim_raises(self):
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        data = speaker_id.serialize_embedding(vec)
        with pytest.raises(ValueError):
            speaker_id.deserialize_embedding(data, 4)

    def test_deserialize_correct_dim_no_raise(self):
        vec = np.zeros(192, dtype=np.float32)
        data = speaker_id.serialize_embedding(vec)
        restored = speaker_id.deserialize_embedding(data, 192)
        assert restored.shape == (192,)


# -----------------------------------------------------------------
# cosine_similarity
# -----------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        v = unit([1.0, 2.0, 3.0])
        assert speaker_id.cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert speaker_id.cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_score_negative_one(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert speaker_id.cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero_not_nan(self):
        a = np.zeros(4)
        b = np.array([1.0, 2.0, 3.0, 4.0])
        assert speaker_id.cosine_similarity(a, b) == 0.0

    def test_scale_invariant(self):
        a = np.array([1.0, 1.0])
        b = np.array([2.0, 2.0])
        assert speaker_id.cosine_similarity(a, b) == pytest.approx(1.0)


# -----------------------------------------------------------------
# match_speaker
# -----------------------------------------------------------------

def make_known(speaker_id_val, name, vector):
    vec = unit(vector).astype(np.float32)
    return {
        "speaker_id": speaker_id_val,
        "name": name,
        "embedding": speaker_id.serialize_embedding(vec),
        "embedding_dim": vec.shape[0],
    }


class TestMatchSpeaker:
    def test_empty_roster_returns_no_match(self):
        result = speaker_id.match_speaker(unit([1.0, 0.0, 0.0]), [], threshold=0.75)
        assert result == {"speaker_id": None, "name": None, "score": 0.0}

    def test_exact_match_above_threshold(self):
        roster = [make_known(1, "Anna", [1.0, 0.0, 0.0])]
        result = speaker_id.match_speaker(unit([1.0, 0.0, 0.0]), roster, threshold=0.75)
        assert result["speaker_id"] == 1
        assert result["name"] == "Anna"
        assert result["score"] == pytest.approx(1.0)

    def test_below_threshold_returns_no_match_but_reports_score(self):
        roster = [make_known(1, "Anna", [1.0, 0.0, 0.0])]
        # Orthogonal vector: score 0.0, well below any sane threshold.
        result = speaker_id.match_speaker(unit([0.0, 1.0, 0.0]), roster, threshold=0.75)
        assert result["speaker_id"] is None
        assert result["name"] is None
        assert result["score"] == pytest.approx(0.0)

    def test_picks_best_of_multiple_candidates(self):
        roster = [
            make_known(1, "Anna", [1.0, 0.0, 0.0]),
            make_known(2, "Bert", [0.9, 0.1, 0.0]),
        ]
        result = speaker_id.match_speaker(unit([0.85, 0.15, 0.0]), roster, threshold=0.5)
        assert result["name"] == "Bert"

    def test_mismatched_dim_entry_is_skipped_not_fatal(self):
        good = make_known(1, "Anna", [1.0, 0.0, 0.0])
        bad = make_known(2, "Bert", [1.0, 0.0, 0.0, 0.0])  # 4-dim, won't match a 3-dim query
        result = speaker_id.match_speaker(unit([1.0, 0.0, 0.0]), [bad, good], threshold=0.75)
        assert result["name"] == "Anna"

    def test_threshold_boundary_is_inclusive(self):
        roster = [make_known(1, "Anna", [1.0, 0.0])]
        result = speaker_id.match_speaker(unit([1.0, 0.0]), roster, threshold=1.0)
        assert result["name"] == "Anna"


# -----------------------------------------------------------------
# update_running_average
# -----------------------------------------------------------------

class TestUpdateRunningAverage:
    def test_first_update_averages_two_samples_equally(self):
        old = unit([1.0, 0.0])
        new = unit([0.0, 1.0])
        combined, count = speaker_id.update_running_average(old, 1, new)
        assert count == 2
        # Average of two orthogonal unit vectors, then re-normalized:
        # equal weight -> 45 degrees between them.
        assert combined[0] == pytest.approx(combined[1])
        assert np.linalg.norm(combined) == pytest.approx(1.0)

    def test_result_is_unit_normalized(self):
        old = unit([3.0, 4.0])
        new = unit([1.0, 0.0])
        combined, _ = speaker_id.update_running_average(old, 5, new)
        assert np.linalg.norm(combined) == pytest.approx(1.0)

    def test_later_samples_are_weighted_less(self):
        old = unit([1.0, 0.0])
        new = unit([0.0, 1.0])
        _, count_early = speaker_id.update_running_average(old, 1, new)
        combined_early, _ = speaker_id.update_running_average(old, 1, new)
        combined_late, count_late = speaker_id.update_running_average(old, 20, new)
        # With more prior samples, the update should move less far toward
        # "new": combined_late should stay closer to old than combined_early.
        assert speaker_id.cosine_similarity(combined_late, old) > \
            speaker_id.cosine_similarity(combined_early, old)
        assert count_late == 21

    def test_invalid_sample_count_raises(self):
        with pytest.raises(ValueError):
            speaker_id.update_running_average(unit([1.0, 0.0]), 0, unit([0.0, 1.0]))


# -----------------------------------------------------------------
# extract_speaker_sample_clip (V1.27: longest-segments-first, concatenated
# up to max_duration, instead of only ever the single longest segment)
#
# Unlike extract_speaker_embeddings, this only needs ffmpeg (no torch/
# pyannote), so these run against real, ffmpeg-generated silent audio
# (offline, deterministic, no network/media fixtures needed) rather than
# being mocked - skipped outright if ffmpeg genuinely isn't available on
# the test host.
# -----------------------------------------------------------------

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


def _make_test_audio(path, duration=30.0):
    """Generate a valid mono 16kHz PCM WAV of the given length via
    ffmpeg's lavfi anullsrc (silence) - no real recording needed, just a
    file with genuine duration for extract_speaker_sample_clip to seek/
    trim/concat against."""
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
         "-t", str(duration), str(path)],
        capture_output=True, text=True, check=True)


def _wav_duration_sec(path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not found on PATH")
class TestExtractSpeakerSampleClip:
    def test_single_long_segment_capped_to_max_duration(self, tmp_path):
        src = tmp_path / "source.wav"
        _make_test_audio(src, duration=30.0)
        segments = [{"start": 0.0, "end": 30.0, "speaker": "SPEAKER_00"}]
        out = tmp_path / "clip.wav"

        result = speaker_id.extract_speaker_sample_clip(
            str(src), segments, "SPEAKER_00", str(out), max_duration=15.0)

        assert result == str(out)
        assert _wav_duration_sec(out) == pytest.approx(15.0, abs=0.2)

    def test_multiple_short_segments_concatenated_up_to_max_duration(self, tmp_path):
        src = tmp_path / "source.wav"
        _make_test_audio(src, duration=30.0)
        # Five 3s segments, none alone anywhere close to max_duration -
        # must be concatenated to approach it (the pre-V1.27 behavior
        # would have used only the first 3s segment and stopped there).
        segments = [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 8.0, "speaker": "SPEAKER_00"},
            {"start": 10.0, "end": 13.0, "speaker": "SPEAKER_00"},
            {"start": 15.0, "end": 18.0, "speaker": "SPEAKER_00"},
            {"start": 20.0, "end": 23.0, "speaker": "SPEAKER_00"},
        ]
        out = tmp_path / "clip.wav"

        result = speaker_id.extract_speaker_sample_clip(
            str(src), segments, "SPEAKER_00", str(out), max_duration=12.0)

        assert result == str(out)
        # All 5 segments tie on length, so the stable sort keeps them in
        # chronological order; the cap is reached after 4 of them
        # (4 * 3.0s = 12.0s), so the 5th is never pulled in.
        assert _wav_duration_sec(out) == pytest.approx(12.0, abs=0.3)

    def test_other_speakers_segments_are_excluded(self, tmp_path):
        src = tmp_path / "source.wav"
        _make_test_audio(src, duration=10.0)
        segments = [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"},
        ]
        out = tmp_path / "clip.wav"

        result = speaker_id.extract_speaker_sample_clip(
            str(src), segments, "SPEAKER_01", str(out), max_duration=15.0)

        assert result == str(out)
        assert _wav_duration_sec(out) == pytest.approx(5.0, abs=0.2)

    def test_no_segments_for_label_returns_none(self, tmp_path):
        src = tmp_path / "source.wav"
        _make_test_audio(src, duration=5.0)
        segments = [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}]
        out = tmp_path / "clip.wav"

        result = speaker_id.extract_speaker_sample_clip(
            str(src), segments, "SPEAKER_99", str(out))

        assert result is None
        assert not out.exists()

    def test_segments_under_min_duration_are_skipped(self, tmp_path):
        src = tmp_path / "source.wav"
        _make_test_audio(src, duration=5.0)
        segments = [{"start": 0.0, "end": 0.1, "speaker": "SPEAKER_00"}]  # below 0.3s floor
        out = tmp_path / "clip.wav"

        result = speaker_id.extract_speaker_sample_clip(str(src), segments, "SPEAKER_00", str(out))

        assert result is None

    def test_ffmpeg_unavailable_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "get_ffmpeg_path", lambda: None)
        src = tmp_path / "source.wav"
        _make_test_audio(src, duration=5.0)
        segments = [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}]
        out = tmp_path / "clip.wav"

        result = speaker_id.extract_speaker_sample_clip(str(src), segments, "SPEAKER_00", str(out))

        assert result is None
        assert not out.exists()


# -----------------------------------------------------------------
# select_sample_segments / render_sample_clip (V1.28: split out of
# extract_speaker_sample_clip so the Rename Speakers dialog's "Edit
# sample" feature can show the picked chunks and let the user remove
# some before rendering, instead of the two steps always happening
# together.
# -----------------------------------------------------------------

class TestSelectSampleSegments:
    """Pure logic, no ffmpeg needed - same picking rule
    extract_speaker_sample_clip used inline before V1.28."""

    def test_returns_chronological_order_not_longest_first(self):
        # A later, shorter segment and an earlier, longer one: picking is
        # longest-first internally, but the returned list must read
        # chronologically (by start), not in pick order.
        segments = [
            {"start": 10.0, "end": 12.0, "speaker": "SPEAKER_00"},   # 2s, later
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},     # 5s, earlier
        ]
        picked = speaker_id.select_sample_segments(segments, "SPEAKER_00", max_duration=15.0)
        assert [p["start"] for p in picked] == [0.0, 10.0]

    def test_excludes_other_speakers(self):
        segments = [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"},
        ]
        picked = speaker_id.select_sample_segments(segments, "SPEAKER_01", max_duration=15.0)
        assert len(picked) == 1
        assert picked[0] == {"start": 5.0, "end": 10.0}

    def test_caps_total_at_max_duration(self):
        segments = [{"start": 0.0, "end": 30.0, "speaker": "SPEAKER_00"}]
        picked = speaker_id.select_sample_segments(segments, "SPEAKER_00", max_duration=15.0)
        assert len(picked) == 1
        assert picked[0]["end"] - picked[0]["start"] == pytest.approx(15.0)

    def test_skips_segments_under_min_duration(self):
        segments = [{"start": 0.0, "end": 0.1, "speaker": "SPEAKER_00"}]
        picked = speaker_id.select_sample_segments(segments, "SPEAKER_00")
        assert picked == []

    def test_no_matching_speaker_returns_empty(self):
        segments = [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}]
        picked = speaker_id.select_sample_segments(segments, "SPEAKER_99")
        assert picked == []


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not found on PATH")
class TestRenderSampleClip:
    def test_renders_explicit_single_segment(self, tmp_path):
        src = tmp_path / "source.wav"
        _make_test_audio(src, duration=10.0)
        out = tmp_path / "clip.wav"

        result = speaker_id.render_sample_clip(str(src), [{"start": 2.0, "end": 6.0}], str(out))

        assert result == str(out)
        assert _wav_duration_sec(out) == pytest.approx(4.0, abs=0.2)

    def test_renders_explicit_multi_segment_concat(self, tmp_path):
        src = tmp_path / "source.wav"
        _make_test_audio(src, duration=10.0)
        out = tmp_path / "clip.wav"

        result = speaker_id.render_sample_clip(
            str(src), [{"start": 0.0, "end": 2.0}, {"start": 5.0, "end": 8.0}], str(out))

        assert result == str(out)
        assert _wav_duration_sec(out) == pytest.approx(5.0, abs=0.2)  # 2s + 3s

    def test_removing_a_segment_shortens_the_rendered_clip(self, tmp_path):
        # Simulates the "Edit sample" flow: select, then drop one entry,
        # then re-render - the result must reflect only what remains.
        src = tmp_path / "source.wav"
        _make_test_audio(src, duration=30.0)
        segments = [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 8.0, "speaker": "SPEAKER_00"},
            {"start": 10.0, "end": 13.0, "speaker": "SPEAKER_00"},
        ]
        picked = speaker_id.select_sample_segments(segments, "SPEAKER_00", max_duration=15.0)
        assert len(picked) == 3

        edited = [p for p in picked if p["start"] != 5.0]  # drop the middle chunk
        out = tmp_path / "clip.wav"
        result = speaker_id.render_sample_clip(str(src), edited, str(out))

        assert result == str(out)
        assert _wav_duration_sec(out) == pytest.approx(6.0, abs=0.2)  # 3s + 3s, not 9s

    def test_empty_list_returns_none(self, tmp_path):
        out = tmp_path / "clip.wav"
        result = speaker_id.render_sample_clip("unused.wav", [], str(out))
        assert result is None
        assert not out.exists()

    def test_ffmpeg_unavailable_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "get_ffmpeg_path", lambda: None)
        out = tmp_path / "clip.wav"
        result = speaker_id.render_sample_clip("unused.wav", [{"start": 0.0, "end": 1.0}], str(out))
        assert result is None
