import pytest

from clip_generator.models import LoudRegion, Settings
from clip_generator.segments import build_highlights


def region(start: float, end: float, score: float = 10.0) -> LoudRegion:
    return LoudRegion(start_s=start, end_s=end, score=score, peak_excess_db=score)


def test_no_regions_gives_no_highlights() -> None:
    assert build_highlights([], Settings(), 100.0) == []


def test_single_region_is_padded_on_both_sides() -> None:
    (h,) = build_highlights([region(20, 22)], Settings(), 100.0)
    assert (h.clip_start_s, h.clip_end_s) == (15.0, 27.0)
    assert (h.noise_start_s, h.noise_end_s) == (20.0, 22.0)
    assert h.noise_start_in_clip_s == pytest.approx(5.0)
    assert h.noise_end_in_clip_s == pytest.approx(7.0)
    assert h.clip_duration_s == pytest.approx(12.0)


def test_clip_is_clamped_to_start_of_video() -> None:
    (h,) = build_highlights([region(2, 3)], Settings(), 100.0)
    assert h.clip_start_s == 0.0
    assert h.noise_start_in_clip_s == pytest.approx(2.0)


def test_clip_is_clamped_to_end_of_video() -> None:
    (h,) = build_highlights([region(8, 9)], Settings(), 10.0)
    assert h.clip_end_s == 10.0


def test_overlapping_clips_are_merged() -> None:
    (h,) = build_highlights([region(20, 22, 5), region(28, 30, 9)], Settings(), 100.0)
    assert (h.clip_start_s, h.clip_end_s) == (15.0, 35.0)
    assert (h.noise_start_s, h.noise_end_s) == (20.0, 30.0)
    assert h.score == 9.0  # the higher score wins


def test_touching_clips_are_merged() -> None:
    # first clip ends at 27, second starts at 27
    assert len(build_highlights([region(20, 22), region(32, 34)], Settings(), 100.0)) == 1


def test_far_apart_regions_stay_separate() -> None:
    assert len(build_highlights([region(20, 22), region(40, 42)], Settings(), 100.0)) == 2


def test_long_clip_is_trimmed_around_the_loudest_noise() -> None:
    regions = [region(20, 22, 5), region(28, 30, 9)]
    (h,) = build_highlights(regions, Settings(max_clip_s=10.0), 100.0)
    assert h.clip_duration_s == pytest.approx(10.0)
    assert (h.clip_start_s, h.clip_end_s) == (24.0, 34.0)
    assert (h.noise_start_s, h.noise_end_s) == (28.0, 30.0)  # quieter region dropped
    assert h.noise_start_in_clip_s == pytest.approx(4.0)
    assert h.score == 9.0


def test_trimmed_clip_stays_inside_the_original_clip() -> None:
    regions = [region(20, 22, 9), region(28, 30, 5)]  # loudest is at the start
    (h,) = build_highlights(regions, Settings(max_clip_s=14.0), 100.0)
    assert h.clip_start_s == pytest.approx(15.0)  # centring would give 14, clamped to 15
    assert h.clip_end_s == pytest.approx(29.0)


def test_output_is_in_time_order_with_ranks_by_score() -> None:
    regions = [region(20, 22, 5), region(50, 52, 20), region(80, 82, 10)]
    highlights = build_highlights(regions, Settings(), 200.0)
    assert [h.index for h in highlights] == [1, 2, 3]
    assert [h.clip_start_s for h in highlights] == sorted(h.clip_start_s for h in highlights)
    assert [h.rank for h in highlights] == [3, 1, 2]


def test_top_n_keeps_highest_scores_but_stays_in_time_order() -> None:
    regions = [region(20, 22, 5), region(50, 52, 20), region(80, 82, 10)]
    highlights = build_highlights(regions, Settings(top_n=2), 200.0)
    assert [h.score for h in highlights] == [20.0, 10.0]
    assert [h.rank for h in highlights] == [1, 2]
    assert [h.index for h in highlights] == [1, 2]


def test_invalid_settings_are_rejected() -> None:
    with pytest.raises(ValueError, match="top_n"):
        build_highlights([region(1, 2)], Settings(top_n=0), 10.0)
    with pytest.raises(ValueError, match="max_clip_s"):
        build_highlights([region(1, 2)], Settings(max_clip_s=0), 10.0)
