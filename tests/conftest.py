"""Shared fixtures: builds synthetic audio and tiny test videos."""

import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

from clip_generator.ffmpeg import run_ffmpeg

SAMPLE_RATE = 16_000
DURATION_S = 12.0
BURSTS = [(3.0, 4.0), (8.0, 9.0)]  # (start, end) in seconds
BACKGROUND_SIGMA = 0.01  # about -40 dBFS
BURST_SIGMA = 0.3  # about -10 dBFS


@pytest.fixture(scope="session")
def bursts() -> list[tuple[float, float]]:
    return BURSTS


@pytest.fixture(scope="session")
def synthetic_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Quiet noise with loud bursts at known times."""
    rng = np.random.default_rng(seed=42)
    n = int(DURATION_S * SAMPLE_RATE)
    samples = rng.normal(0.0, BACKGROUND_SIGMA, n)
    for start, end in BURSTS:
        a, b = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
        samples[a:b] = rng.normal(0.0, BURST_SIGMA, b - a)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")

    path = tmp_path_factory.mktemp("audio") / "synthetic.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())
    return path


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory, synthetic_wav: Path) -> Path:
    """A tiny black video carrying the synthetic audio."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    path = tmp_path_factory.mktemp("video") / "synthetic.mp4"
    run_ffmpeg(
        [
            "-f", "lavfi", "-i", f"color=c=black:s=64x64:r=10:d={DURATION_S}",
            "-i", str(synthetic_wav),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(path),
        ]
    )  # fmt: skip
    return path
