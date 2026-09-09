# =============================================================
#  Transkription_Notes_Pipeline - tests/test_detector.py
#  Unit tests for detector.py's slide-change detection: the first
#  frame, the confirm-then-commit noise debounce, the animation vs.
#  new-slide split, and frame_indices.
#
#  frame_indices is the fix for a real defect: timestamps used to be
#  derived from the position in the list handed to the detector, so when
#  run_pipeline removed frames from the MIDDLE of the sequence for a
#  configured Q&A range, every slide after the gap was reported earlier
#  than it really was, by exactly the excised duration.
#
#  Frames are generated with PIL under tmp_path (real files, real
#  perceptual hashes) - no whisperx/torch, no fixture images in the repo.
#
#  Run: C:\Python\Python311\python.exe -m pytest tests/test_detector.py -v
# =============================================================

from PIL import Image, ImageDraw

import detector


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def write_frame(path, seed: int):
    """Deterministic, visually distinct greyscale frame per seed. Solid
    colours are avoided on purpose: phash is a DCT over luminance, so flat
    images collapse to near-identical hashes regardless of colour."""
    img = Image.new("L", (320, 240), color=250)
    draw = ImageDraw.Draw(img)
    for i in range(6):
        x = (seed * 61 + i * 47) % 260
        y = (seed * 29 + i * 37) % 180
        draw.rectangle([x, y, x + 45, y + 45], fill=(seed * 53 + i * 31) % 200)
    img.save(path)
    return path


def frames_from_seeds(tmp_path, seeds):
    """One frame file per seed, named like ffmpeg's output so the detector
    sees realistic paths. Equal seeds produce byte-identical frames."""
    out = []
    for i, seed in enumerate(seeds):
        out.append(write_frame(tmp_path / f"frame_{i:06d}.png", seed))
    return out


def distance(frame_a, frame_b) -> int:
    return detector.compute_hash(frame_a) - detector.compute_hash(frame_b)


# -----------------------------------------------------------------
# compute_hash
# -----------------------------------------------------------------

class TestComputeHash:
    def test_identical_images_hash_identically(self, tmp_path):
        a, b = frames_from_seeds(tmp_path, [3, 3])
        assert distance(a, b) == 0

    def test_different_images_differ(self, tmp_path):
        a, b = frames_from_seeds(tmp_path, [1, 9])
        assert distance(a, b) > 0

    def test_unknown_algorithm_raises(self, tmp_path):
        frame = write_frame(tmp_path / "f.png", 1)
        try:
            detector.compute_hash(frame, algorithm="nope")
        except ValueError as exc:
            assert "nope" in str(exc)
        else:
            raise AssertionError("expected ValueError for an unknown algorithm")


# -----------------------------------------------------------------
# Basic detection
# -----------------------------------------------------------------

class TestDetectSlideChanges:
    def test_no_frames_returns_empty(self):
        assert detector.detect_slide_changes([]) == []

    def test_first_frame_is_always_the_opening_slide(self, tmp_path):
        frames = frames_from_seeds(tmp_path, [1, 1, 1])
        changes = detector.detect_slide_changes(frames, fps=1, threshold=8)
        assert len(changes) == 1
        assert changes[0].is_first is True
        assert changes[0].frame_index == 0
        assert changes[0].timestamp_sec == 0.0

    def test_a_stable_change_is_detected(self, tmp_path):
        # Four identical frames, then four clearly different ones.
        frames = frames_from_seeds(tmp_path, [1, 1, 1, 1, 9, 9, 9, 9])
        changes = detector.detect_slide_changes(
            frames, fps=1, threshold=1, min_slide_duration_sec=2.0)
        assert len(changes) == 2
        assert changes[1].frame_index == 4      # where the change began
        assert changes[1].timestamp_sec == 4.0

    def test_transient_flash_is_discarded_as_noise(self, tmp_path):
        # One odd frame that reverts immediately must not become a slide.
        frames = frames_from_seeds(tmp_path, [1, 1, 1, 9, 1, 1, 1])
        changes = detector.detect_slide_changes(
            frames, fps=1, threshold=1, min_slide_duration_sec=3.0)
        assert len(changes) == 1                # only the opening slide

    def test_timestamps_follow_fps(self, tmp_path):
        frames = frames_from_seeds(tmp_path, [1, 1, 9, 9])
        changes = detector.detect_slide_changes(
            frames, fps=2, threshold=1, min_slide_duration_sec=0.5)
        assert changes[1].frame_index == 2
        assert changes[1].timestamp_sec == 1.0  # 2 frames at 2 fps


# -----------------------------------------------------------------
# frame_indices  (the Q&A excision fix)
# -----------------------------------------------------------------

class TestFrameIndices:
    def test_defaults_to_list_position(self, tmp_path):
        frames = frames_from_seeds(tmp_path, [1, 1, 9, 9])
        with_default = detector.detect_slide_changes(
            frames, fps=1, threshold=1, min_slide_duration_sec=1.0)
        explicit = detector.detect_slide_changes(
            frames, fps=1, threshold=1, min_slide_duration_sec=1.0,
            frame_indices=[0, 1, 2, 3])
        assert [c.timestamp_sec for c in with_default] == [c.timestamp_sec for c in explicit]

    def test_gap_in_the_middle_does_not_shift_later_timestamps(self, tmp_path):
        """The defect this parameter fixes. Frames 2-51 are excised (a 50
        second Q&A block at 1 fps); the slide that follows really starts at
        frame 52, and must be reported there rather than at list position 2."""
        frames = frames_from_seeds(tmp_path, [1, 1, 9, 9, 9])
        real_indices = [0, 1, 52, 53, 54]
        changes = detector.detect_slide_changes(
            frames, fps=1, threshold=1, min_slide_duration_sec=1.0,
            frame_indices=real_indices)
        assert len(changes) == 2
        assert changes[1].frame_index == 52
        assert changes[1].timestamp_sec == 52.0

    def test_without_frame_indices_the_same_input_is_reported_too_early(self, tmp_path):
        """Companion to the test above, pinning exactly what went wrong: the
        identical frame list with no index mapping reports the change at
        position 2, which is 50 seconds earlier than the truth."""
        frames = frames_from_seeds(tmp_path, [1, 1, 9, 9, 9])
        changes = detector.detect_slide_changes(
            frames, fps=1, threshold=1, min_slide_duration_sec=1.0)
        assert changes[1].timestamp_sec == 2.0

    def test_trailing_gap_is_harmless(self, tmp_path):
        """Q&A at the end of a recording, which is the normal case: only
        trailing frames are dropped, so no surviving frame follows the gap
        and results match the unmapped call. This is why the defect stayed
        invisible for so long."""
        frames = frames_from_seeds(tmp_path, [1, 1, 9, 9])
        mapped = detector.detect_slide_changes(
            frames, fps=1, threshold=1, min_slide_duration_sec=1.0,
            frame_indices=[0, 1, 2, 3])
        assert [c.timestamp_sec for c in mapped] == [0.0, 2.0]

    def test_first_slide_uses_the_mapped_index_too(self, tmp_path):
        frames = frames_from_seeds(tmp_path, [1, 1, 1])
        changes = detector.detect_slide_changes(
            frames, fps=1, threshold=1, frame_indices=[10, 11, 12])
        assert changes[0].frame_index == 10
        assert changes[0].timestamp_sec == 10.0

    def test_length_mismatch_raises(self, tmp_path):
        frames = frames_from_seeds(tmp_path, [1, 1, 9])
        try:
            detector.detect_slide_changes(frames, frame_indices=[0, 1])
        except ValueError as exc:
            assert "line up" in str(exc)
        else:
            raise AssertionError("expected ValueError on a length mismatch")

    def test_debounce_still_counts_observed_frames(self, tmp_path):
        """Deliberate: the min_slide_duration_sec debounce asks "did this stay
        different across enough frames I actually looked at". Counting mapped
        indices instead would confirm a candidate on the strength of an
        excised stretch nobody examined. Two observed frames across a large
        index gap must therefore still be judged as two frames."""
        frames = frames_from_seeds(tmp_path, [1, 1, 9, 9])
        changes = detector.detect_slide_changes(
            frames, fps=1, threshold=1, min_slide_duration_sec=10.0,
            frame_indices=[0, 1, 500, 501])
        # Only 2 observed frames, far short of the 10-frame debounce, so the
        # candidate is confirmed by the end-of-video rule rather than by the
        # index gap - and still lands on its real index.
        assert len(changes) == 2
        assert changes[1].frame_index == 500


# -----------------------------------------------------------------
# Animation vs. new slide
# -----------------------------------------------------------------

class TestAnimationThreshold:
    def test_disabled_when_zero(self, tmp_path):
        frames = frames_from_seeds(tmp_path, [1, 1, 9, 9])
        changes = detector.detect_slide_changes(
            frames, fps=1, threshold=1, min_slide_duration_sec=1.0,
            animation_threshold=0)
        assert len(changes) == 2

    def test_disabled_when_not_below_threshold(self, tmp_path):
        frames = frames_from_seeds(tmp_path, [1, 1, 9, 9])
        changes = detector.detect_slide_changes(
            frames, fps=1, threshold=4, min_slide_duration_sec=1.0,
            animation_threshold=4)
        assert len(changes) == 2

    def test_mid_range_change_updates_the_current_slide_snapshot(self, tmp_path):
        """A change whose distance lands between animation_threshold and
        threshold means "same slide, still building": no new slide, but the
        stored snapshot moves to the more complete frame while the recorded
        start timestamp stays put."""
        frames = frames_from_seeds(tmp_path, [1, 1, 9, 9, 9])
        dist = distance(frames[0], frames[2])
        changes = detector.detect_slide_changes(
            frames, fps=1, threshold=dist + 1, animation_threshold=dist,
            min_slide_duration_sec=1.0)
        assert len(changes) == 1                     # no second slide
        assert changes[0].timestamp_sec == 0.0       # start time unchanged
        assert changes[0].frame_path == frames[2]    # snapshot advanced
