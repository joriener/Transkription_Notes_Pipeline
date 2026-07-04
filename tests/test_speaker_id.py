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

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
