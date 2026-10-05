from pathlib import Path

import numpy as np
import pytest

from clip_generator.audio import SAMPLE_RATE, compute_loudness, extract_audio


def test_constant_signal_has_expected_level() -> None:
    samples = np.full(SAMPLE_RATE * 3, 0.1, dtype=np.float32)  # RMS 0.1 = -20 dBFS
    curve = compute_loudness(samples)
    assert len(curve.db) == 11
    assert curve.times_s[0] == pytest.approx(0.25)
    assert np.allclose(curve.db, -20.0, atol=0.01)


def test_silence_hits_the_floor() -> None:
    curve = compute_loudness(np.zeros(SAMPLE_RATE * 2, dtype=np.float32))
    assert np.allclose(curve.db, -100.0)


def test_audio_shorter_than_one_window_gives_empty_curve() -> None:
    curve = compute_loudness(np.zeros(100, dtype=np.float32))
    assert curve.db.size == 0


def test_window_must_be_multiple_of_hop() -> None:
    with pytest.raises(ValueError, match="multiple"):
        compute_loudness(np.zeros(SAMPLE_RATE * 2, dtype=np.float32), window_s=0.6, hop_s=0.25)


def test_extract_audio_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_audio(tmp_path / "nope.mp4")


def test_extract_audio_length(synthetic_video: Path) -> None:
    samples = extract_audio(synthetic_video)
    assert samples.size / SAMPLE_RATE == pytest.approx(12.0, abs=0.2)


def test_loudness_spikes_at_known_bursts(
    synthetic_video: Path, bursts: list[tuple[float, float]]
) -> None:
    curve = compute_loudness(extract_audio(synthetic_video))

    def distance_to_burst(t: float) -> float:
        return min(max(start - t, 0.0, t - end) for start, end in bursts)

    for t, db in zip(curve.times_s, curve.db, strict=True):
        if distance_to_burst(t) > 0.5:
            assert db < -30, f"background too loud at {t:.2f}s ({db:.1f} dB)"
        for start, end in bursts:
            if start + 0.25 <= t <= end - 0.25:  # window fully inside the burst
                assert db > -20, f"burst too quiet at {t:.2f}s ({db:.1f} dB)"
