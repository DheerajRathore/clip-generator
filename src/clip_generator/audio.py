"""Audio extraction via FFmpeg and loudness per time window."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from clip_generator.ffmpeg import FFmpegError, run_ffmpeg

SAMPLE_RATE = 16_000
DB_FLOOR_RMS = 1e-5  # silence is reported as -100 dB instead of -infinity


@dataclass(frozen=True)
class LoudnessCurve:
    """Loudness over time. times_s are window centres; db is RMS level in dBFS."""

    times_s: NDArray[np.float64]
    db: NDArray[np.float64]
    window_s: float
    hop_s: float


def extract_audio(video: Path, sample_rate: int = SAMPLE_RATE) -> NDArray[np.float32]:
    """Decode the audio track of a video into mono float samples in [-1, 1]."""
    if not video.is_file():
        raise FileNotFoundError(f"Video file not found: {video}")
    raw = run_ffmpeg(
        ["-i", str(video), "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1"]
    )
    samples = np.frombuffer(raw, dtype=np.float32)
    if samples.size == 0:
        raise FFmpegError(f"No audio could be read from {video}")
    return samples


def compute_loudness(
    samples: NDArray[np.float32],
    sample_rate: int = SAMPLE_RATE,
    window_s: float = 0.5,
    hop_s: float = 0.25,
) -> LoudnessCurve:
    """Compute RMS loudness (dBFS) over sliding windows.

    window_s must be a whole multiple of hop_s. Audio is split into hop-sized
    blocks, and each window is the sum of consecutive blocks.
    """
    hop = round(hop_s * sample_rate)
    blocks_per_window = round(window_s / hop_s)
    if hop < 1 or blocks_per_window < 1 or abs(window_s - blocks_per_window * hop_s) > 1e-9:
        raise ValueError("window_s must be a positive whole multiple of hop_s")

    n_blocks = samples.size // hop
    if n_blocks < blocks_per_window:
        empty = np.empty(0, dtype=np.float64)
        return LoudnessCurve(empty, empty, window_s, hop_s)

    blocks = samples[: n_blocks * hop].reshape(n_blocks, hop)
    block_energy = np.einsum("ij,ij->i", blocks, blocks, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(block_energy)))
    window_energy = cumulative[blocks_per_window:] - cumulative[:-blocks_per_window]

    mean_square = window_energy / (blocks_per_window * hop)
    rms = np.sqrt(np.maximum(mean_square, 0.0))
    db = 20.0 * np.log10(np.maximum(rms, DB_FLOOR_RMS))

    times = (np.arange(db.size) * hop_s + window_s / 2).astype(np.float64)
    return LoudnessCurve(times, db.astype(np.float64), window_s, hop_s)
