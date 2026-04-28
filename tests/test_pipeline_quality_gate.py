from humeo.pipeline import _filter_weak_hook_clips, _normalize_layout_for_render
from humeo_core.schemas import BoundingBox, Clip, LayoutInstruction, LayoutKind, RenderTheme


def _clip(
    clip_id: str,
    *,
    start: float,
    duration: float = 60.0,
    hook_start: float | None = None,
    hook_end: float | None = None,
) -> Clip:
    payload = {
        "clip_id": clip_id,
        "topic": f"topic {clip_id}",
        "start_time_sec": start,
        "end_time_sec": start + duration,
        "virality_score": 0.8,
    }
    if hook_start is not None:
        payload["hook_start_sec"] = hook_start
    if hook_end is not None:
        payload["hook_end_sec"] = hook_end
    return Clip.model_validate(payload)


def test_filter_weak_hook_clips_drops_late_openers():
    clips = [
        _clip("001", start=0.0, hook_start=7.2, hook_end=10.5),
        _clip("002", start=100.0, hook_start=1.4, hook_end=4.0),
    ]
    transcript = {
        "segments": [
            {"start": 7.2, "end": 10.5, "text": "This changes the whole market."},
            {"start": 101.4, "end": 104.0, "text": "Robots cut delivery costs 90%."},
        ]
    }

    kept = _filter_weak_hook_clips(clips, transcript, min_kept=1)

    assert [clip.clip_id for clip in kept] == ["002"]


def test_filter_weak_hook_clips_drops_filler_phrase_openers():
    clips = [
        _clip("001", start=0.0, hook_start=0.0, hook_end=2.2),
        _clip("002", start=100.0, hook_start=0.0, hook_end=2.2),
    ]
    transcript = {
        "segments": [
            {"start": 0.0, "end": 2.2, "text": "You know, AI is changing everything."},
            {"start": 100.0, "end": 102.2, "text": "AI is cutting delivery under $1."},
        ]
    }

    kept = _filter_weak_hook_clips(clips, transcript, min_kept=1)

    assert [clip.clip_id for clip in kept] == ["002"]


def test_normalize_layout_for_render_prefers_person_first_in_native_highlight():
    instruction = LayoutInstruction(
        clip_id="001",
        layout=LayoutKind.SPLIT_CHART_PERSON,
        person_x_norm=0.75,
        split_chart_region=BoundingBox(x1=0.02, y1=0.04, x2=0.67, y2=0.81),
        split_person_region=BoundingBox(x1=0.60, y1=0.35, x2=0.999, y2=1.0),
    )

    normalized = _normalize_layout_for_render(
        instruction,
        render_theme=RenderTheme.NATIVE_HIGHLIGHT,
    )

    assert normalized.layout == LayoutKind.SIT_CENTER
    assert normalized.split_chart_region is None
    assert normalized.split_person_region is None


def test_normalize_layout_for_render_keeps_top_anchored_split_in_native_highlight():
    instruction = LayoutInstruction(
        clip_id="001",
        layout=LayoutKind.SPLIT_CHART_PERSON,
        person_x_norm=0.75,
        split_chart_region=BoundingBox(x1=0.02, y1=0.03, x2=0.58, y2=0.70),
        split_person_region=BoundingBox(x1=0.585, y1=0.0, x2=0.995, y2=0.62),
        top_band_ratio=0.436,
    )

    normalized = _normalize_layout_for_render(
        instruction,
        render_theme=RenderTheme.NATIVE_HIGHLIGHT,
    )

    assert normalized.layout == LayoutKind.SPLIT_CHART_PERSON
    assert normalized.split_chart_region == instruction.split_chart_region
    assert normalized.split_person_region == instruction.split_person_region


def test_normalize_layout_for_render_keeps_split_when_chart_is_not_dominant():
    instruction = LayoutInstruction(
        clip_id="001",
        layout=LayoutKind.SPLIT_CHART_PERSON,
        person_x_norm=0.75,
        split_chart_region=BoundingBox(x1=0.02, y1=0.04, x2=0.67, y2=0.60),
        split_person_region=BoundingBox(x1=0.55, y1=0.22, x2=0.999, y2=0.95),
    )

    normalized = _normalize_layout_for_render(
        instruction,
        render_theme=RenderTheme.NATIVE_HIGHLIGHT,
    )

    assert normalized.layout == LayoutKind.SPLIT_CHART_PERSON
