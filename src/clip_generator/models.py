"""Data models: Settings, LoudRegion, and Highlight."""

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
    pad_before_s: float = 5.0  # clip starts this long before the noise
    pad_after_s: float = 5.0  # clip ends this long after the noise
    max_clip_s: float = 60.0  # longer clips are trimmed around the loudest noise
    top_n: int | None = None  # keep only the N highest-scoring clips (None = all)


@dataclass(frozen=True)
class LoudRegion:
    """One loud stretch in the source video."""

    start_s: float
    end_s: float
    score: float  # mean dB above background
    peak_excess_db: float  # highest dB above background


@dataclass(frozen=True)
class Highlight:
    """One clip to cut, with the noise position both in the source and inside the clip."""

    index: int  # 1-based, in time order
    rank: int  # 1 = highest score
    clip_start_s: float
    clip_end_s: float
    noise_start_s: float
    noise_end_s: float
    noise_start_in_clip_s: float
    noise_end_in_clip_s: float
    score: float

    @property
    def clip_duration_s(self) -> float:
        return self.clip_end_s - self.clip_start_s
