from pathlib import Path

import numpy as np
import pytest

from clip_generator.audio import LoudnessCurve, compute_loudness, extract_audio
from clip_generator.detect import detect_loud_regions
from clip_generator.models import Settings

HOP, WINDOW = 0.25, 0.5


def make_curve(db: list[float]) -> LoudnessCurve:
    arr = np.array(db, dtype=np.float64)
    times = np.arange(arr.size) * HOP + WINDOW / 2
    return LoudnessCurve(times, arr, WINDOW, HOP)


def curve_with(base: float, loud: dict[tuple[int, int], float], n: int = 60) -> LoudnessCurve:
    """Flat background at `base` dB with loud stretches {(first, last_exclusive): level}."""
    db = [base] * n
    for (a, b), level in loud.items():
        db[a:b] = [level] * (b - a)
    return make_curve(db)


def test_flat_curve_has_no_regions() -> None:
    assert detect_loud_regions(make_curve([-40.0] * 60), Settings()) == []


def test_empty_curve_has_no_regions() -> None:
    assert detect_loud_regions(make_curve([]), Settings()) == []


def test_one_burst_gives_one_region_with_right_times() -> None:
    regions = detect_loud_regions(curve_with(-40, {(10, 16): -10}), Settings())
    assert len(regions) == 1
    assert regions[0].start_s == pytest.approx(2.5)
    assert regions[0].end_s == pytest.approx(4.25)
    assert regions[0].score == pytest.approx(30.0)


def test_constant_loud_music_is_ignored() -> None:
    assert detect_loud_regions(make_curve([-20.0] * 60), Settings()) == []


def test_burst_over_loud_music_is_still_found() -> None:
    regions = detect_loud_regions(curve_with(-20, {(10, 16): -5}), Settings())
    assert len(regions) == 1
    assert regions[0].score == pytest.approx(15.0)


def test_close_bursts_merge_and_far_bursts_do_not() -> None:
    close = curve_with(-40, {(10, 16): -10, (18, 24): -10})  # 0.5 s gap
    far = curve_with(-40, {(10, 16): -10, (30, 36): -10})  # 3.5 s gap
    assert len(detect_loud_regions(close, Settings())) == 1
    assert len(detect_loud_regions(far, Settings())) == 2


def test_short_blip_is_dropped() -> None:
    one_window = curve_with(-40, {(10, 11): -10})  # 0.25 s
    two_windows = curve_with(-40, {(10, 12): -10})  # 0.5 s
    assert detect_loud_regions(one_window, Settings()) == []
    assert len(detect_loud_regions(two_windows, Settings())) == 1


def test_louder_burst_scores_higher() -> None:
    regions = detect_loud_regions(curve_with(-40, {(10, 16): -10, (30, 36): -25}), Settings())
    assert len(regions) == 2
    assert regions[0].score > regions[1].score


def test_lower_sensitivity_finds_quieter_moments() -> None:
    curve = curve_with(-40, {(10, 16): -34})  # only 6 dB above background
    assert detect_loud_regions(curve, Settings(sensitivity_db=8.0)) == []
    assert len(detect_loud_regions(curve, Settings(sensitivity_db=4.0))) == 1


def test_finds_bursts_in_synthetic_video(
    synthetic_video: Path, bursts: list[tuple[float, float]]
) -> None:
    curve = compute_loudness(extract_audio(synthetic_video))
    regions = detect_loud_regions(curve, Settings())
    assert len(regions) == len(bursts)
    for region, (start, end) in zip(regions, bursts, strict=True):
        assert region.start_s == pytest.approx(start, abs=0.75)
        assert region.end_s == pytest.approx(end, abs=0.75)
