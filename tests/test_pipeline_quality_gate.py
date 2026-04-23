from humeo.pipeline import _normalize_layout_for_render
from humeo_core.schemas import BoundingBox, LayoutInstruction, LayoutKind, RenderTheme


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
