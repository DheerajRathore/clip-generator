"""Pure logic: padding, merging, capping, and ranking segments."""

from dataclasses import replace

from clip_generator.models import Highlight, LoudRegion, Settings


def _group_overlapping(regions: list[LoudRegion], settings: Settings) -> list[list[LoudRegion]]:
    """Group regions whose padded clips overlap or touch."""
    groups: list[list[LoudRegion]] = []
    group_end = 0.0
    for region in sorted(regions, key=lambda r: r.start_s):
        if groups and region.start_s - settings.pad_before_s <= group_end:
            groups[-1].append(region)
            group_end = max(group_end, region.end_s + settings.pad_after_s)
        else:
            groups.append([region])
            group_end = region.end_s + settings.pad_after_s
    return groups


def _trim_around_peak(
    clip_start: float, clip_end: float, peak: LoudRegion, max_len: float
) -> tuple[float, float]:
    """Slide a max_len window so it is centred on the peak, kept inside the clip."""
    middle = (peak.start_s + peak.end_s) / 2
    start = min(max(middle - max_len / 2, clip_start), clip_end - max_len)
    return start, start + max_len


def _make_highlight(group: list[LoudRegion], settings: Settings, duration_s: float) -> Highlight:
    clip_start = max(0.0, min(r.start_s for r in group) - settings.pad_before_s)
    clip_end = min(duration_s, max(r.end_s for r in group) + settings.pad_after_s)

    if clip_end - clip_start > settings.max_clip_s:
        peak = max(group, key=lambda r: r.score)
        clip_start, clip_end = _trim_around_peak(clip_start, clip_end, peak, settings.max_clip_s)
        group = [r for r in group if r.end_s > clip_start and r.start_s < clip_end]

    noise_start = max(min(r.start_s for r in group), clip_start)
    noise_end = min(max(r.end_s for r in group), clip_end)
    return Highlight(
        index=0,  # filled in by build_highlights
        rank=0,
        clip_start_s=clip_start,
        clip_end_s=clip_end,
        noise_start_s=noise_start,
        noise_end_s=noise_end,
        noise_start_in_clip_s=noise_start - clip_start,
        noise_end_in_clip_s=noise_end - clip_start,
        score=max(r.score for r in group),
    )


def build_highlights(
    regions: list[LoudRegion], settings: Settings, video_duration_s: float
) -> list[Highlight]:
    """Turn loud regions into a clip plan, returned in time order."""
    if settings.top_n is not None and settings.top_n < 1:
        raise ValueError("top_n must be at least 1")
    if settings.max_clip_s <= 0:
        raise ValueError("max_clip_s must be positive")
    if not regions:
        return []

    groups = _group_overlapping(regions, settings)
    clips = [_make_highlight(g, settings, video_duration_s) for g in groups]

    ranked = sorted(clips, key=lambda h: (-h.score, h.clip_start_s))
    if settings.top_n is not None:
        ranked = ranked[: settings.top_n]
    ranked = [replace(h, rank=i) for i, h in enumerate(ranked, start=1)]

    in_time_order = sorted(ranked, key=lambda h: h.clip_start_s)
    return [replace(h, index=i) for i, h in enumerate(in_time_order, start=1)]
