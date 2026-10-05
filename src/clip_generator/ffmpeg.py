"""Finds FFmpeg, runs commands, and raises friendly errors."""

import shutil
import subprocess


class FFmpegError(RuntimeError):
    """Raised when FFmpeg is missing or a command fails."""


def find_ffmpeg() -> str:
    """Return the path to the ffmpeg binary, or raise with install instructions."""
    path = shutil.which("ffmpeg")
    if path is None:
        raise FFmpegError(
            "FFmpeg was not found on your PATH. Install it with 'brew install ffmpeg' "
            "(macOS) or 'sudo apt-get install ffmpeg' (Ubuntu)."
        )
    return path


def run_ffmpeg(args: list[str]) -> bytes:
    """Run ffmpeg with the given arguments and return its stdout bytes."""
    cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", *args]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0:
        message = result.stderr.decode(errors="replace").strip()
        raise FFmpegError(f"FFmpeg failed (exit code {result.returncode}): {message}")
    return result.stdout
