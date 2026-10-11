from pathlib import Path

import numpy as np
import pytest

from clip_generator.audio import SAMPLE_RATE, compute_loudness, extract_audio
from clip_generator.cut import cut_clip, cut_clips, join_reel
from clip_generator.models import Highlight


def highlight(index: int, start: float, end: float) -> Highlight:
    return Highlight(
        index=index,
        rank=index,
        clip_start_s=start,
        clip_end_s=end,
        noise_start_s=start,
        noise_end_s=end,
        noise_start_in_clip_s=0.0,
        noise_end_in_clip_s=end - start,
        score=10.0,
    )


def duration_of(path: Path) -> float:
    return extract_audio(path).size / SAMPLE_RATE


def test_cut_clip_has_expected_length(synthetic_video: Path, tmp_path: Path) -> None:
    out = cut_clip(synthetic_video, highlight(1, 1.0, 6.0), tmp_path / "a.mp4")
    assert out.is_file()
    assert duration_of(out) == pytest.approx(5.0, abs=0.3)


def test_cut_clip_starts_where_asked(synthetic_video: Path, tmp_path: Path) -> None:
    # Source burst is at 3-4 s, so in a clip starting at 1.0 s it sits at 2-3 s.
    out = cut_clip(synthetic_video, highlight(1, 1.0, 6.0), tmp_path / "a.mp4")
    curve = compute_loudness(extract_audio(out))
    at_burst = curve.db[np.argmin(np.abs(curve.times_s - 2.5))]
    before_burst = curve.db[np.argmin(np.abs(curve.times_s - 0.5))]
    assert at_burst > -20
    assert before_burst < -30


def test_cut_clips_names_files_by_index(synthetic_video: Path, tmp_path: Path) -> None:
    highlights = [highlight(1, 0.0, 3.0), highlight(2, 7.0, 10.0)]
    paths = cut_clips(synthetic_video, highlights, tmp_path / "clips")
    assert [p.name for p in paths] == ["clip_01.mp4", "clip_02.mp4"]
    assert all(p.is_file() for p in paths)


def test_join_reel_length_is_sum_of_clips(synthetic_video: Path, tmp_path: Path) -> None:
    highlights = [highlight(1, 0.0, 3.0), highlight(2, 7.0, 10.0)]
    paths = cut_clips(synthetic_video, highlights, tmp_path / "clips")
    reel = join_reel(paths, tmp_path / "reel.mp4")
    assert duration_of(reel) == pytest.approx(6.0, abs=0.4)


def test_join_reel_rejects_empty_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No clips"):
        join_reel([], tmp_path / "reel.mp4")
