"""FFmpeg clip cutting and optional reel joining."""

import tempfile
from pathlib import Path

from clip_generator.ffmpeg import run_ffmpeg
from clip_generator.models import Highlight


def cut_clip(video: Path, highlight: Highlight, out_path: Path) -> Path:
    """Cut one clip. Re-encodes so the cut starts exactly at clip_start_s."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-y",
            "-ss", f"{highlight.clip_start_s:.3f}",
            "-i", str(video),
            "-t", f"{highlight.clip_duration_s:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            str(out_path),
        ]
    )  # fmt: skip
    return out_path


def cut_clips(video: Path, highlights: list[Highlight], out_dir: Path) -> list[Path]:
    """Cut every highlight into out_dir as clip_01.mp4, clip_02.mp4, ..."""
    return [cut_clip(video, h, out_dir / f"clip_{h.index:02d}.mp4") for h in highlights]


def join_reel(clips: list[Path], out_path: Path) -> Path:
    """Join clips, in the order given, into one video without re-encoding."""
    if not clips:
        raise ValueError("No clips to join")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        list_file = Path(tmp) / "clips.txt"
        lines = []
        for clip in clips:
            escaped = str(clip.resolve()).replace("'", "'\\''")
            lines.append(f"file '{escaped}'")
        list_file.write_text("\n".join(lines) + "\n")
        run_ffmpeg(
            ["-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(out_path)]
        )
    return out_path
