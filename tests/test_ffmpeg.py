import shutil

import pytest

from clip_generator.ffmpeg import FFmpegError, find_ffmpeg, run_ffmpeg


def test_find_ffmpeg_missing_gives_friendly_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")
    with pytest.raises(FFmpegError, match="brew install ffmpeg"):
        find_ffmpeg()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_run_ffmpeg_failure_raises() -> None:
    with pytest.raises(FFmpegError):
        run_ffmpeg(["-i", "does-not-exist.mp4", "-f", "null", "-"])
