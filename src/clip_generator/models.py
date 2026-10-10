"""Data models: Settings and LoudRegion. Highlight is added with segments.py."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """All tunable settings in one place."""

    window_s: float = 0.5  # loudness window length
    hop_s: float = 0.25  # step between windows
    baseline_window_s: float = 60.0  # how far around each moment to look for "normal"
    baseline_percentile: float = 50.0  # 50 = median
    sensitivity_db: float = 8.0  # how far above background counts as loud
    min_noise_s: float = 0.5  # shorter noises are ignored
    merge_gap_s: float = 1.0  # loud stretches closer than this become one


@dataclass(frozen=True)
class LoudRegion:
    """One loud stretch in the source video."""

    start_s: float
    end_s: float
    score: float  # mean dB above background
    peak_excess_db: float  # highest dB above background
