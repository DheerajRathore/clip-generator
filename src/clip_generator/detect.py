"""Pure logic: rolling baseline, smoothing, loud-region detection."""

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import percentile_filter

from clip_generator.audio import LoudnessCurve
from clip_generator.models import LoudRegion, Settings


def rolling_baseline(
    db: NDArray[np.float64], hop_s: float, window_s: float, percentile: float
) -> NDArray[np.float64]:
    """Typical loudness around each point, ignoring short loud spikes."""
    size = max(1, round(window_s / hop_s))
    smoothed = percentile_filter(db, percentile=percentile, size=size, mode="nearest")
    return np.asarray(smoothed, dtype=np.float64)


def _true_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Return (start, end) index pairs of consecutive True values; end is exclusive."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def _merge_close_runs(
    runs: list[tuple[int, int]], hop_s: float, merge_gap_s: float
) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if merged and (start - merged[-1][1]) * hop_s <= merge_gap_s:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def detect_loud_regions(curve: LoudnessCurve, settings: Settings) -> list[LoudRegion]:
    """Find stretches that are well above the rolling background level."""
    if curve.db.size == 0:
        return []

    baseline = rolling_baseline(
        curve.db, curve.hop_s, settings.baseline_window_s, settings.baseline_percentile
    )
    excess = curve.db - baseline
    runs = _true_runs(excess >= settings.sensitivity_db)
    runs = _merge_close_runs(runs, curve.hop_s, settings.merge_gap_s)

    regions: list[LoudRegion] = []
    for start, end in runs:
        if (end - start) * curve.hop_s < settings.min_noise_s:
            continue  # too short to be a real moment
        region_excess = excess[start:end]
        regions.append(
            LoudRegion(
                start_s=max(0.0, float(curve.times_s[start] - curve.window_s / 2)),
                end_s=float(curve.times_s[end - 1] + curve.window_s / 2),
                score=float(region_excess.mean()),
                peak_excess_db=float(region_excess.max()),
            )
        )
    return regions
