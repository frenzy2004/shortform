"""Tests for trim/hook → ffmpeg source window."""

from humeo.render_window import clip_for_render, effective_export_bounds
from humeo_core.schemas import Clip, ClipRenderSpan


def _clip(**kwargs) -> Clip:
    base = dict(
        clip_id="1",
        topic="t",
        start_time_sec=100.0,
        end_time_sec=130.0,
    )
    base.update(kwargs)
    return Clip.model_validate(base)


def test_trim_only():
    c = _clip(trim_start_sec=5.0, trim_end_sec=3.0)
    lo, hi = effective_export_bounds(c)
    assert lo == 105.0
    assert hi == 127.0


def test_hook_does_not_shorten_export_window():
    c = _clip(
        trim_start_sec=0.0,
        trim_end_sec=0.0,
        hook_start_sec=2.0,
        hook_end_sec=8.0,
    )
    lo, hi = effective_export_bounds(c)
    assert lo == 100.0
    assert hi == 130.0


def test_clip_for_render_clears_timing_fields():
    c = _clip(
        trim_start_sec=0.0,
        trim_end_sec=0.0,
        hook_start_sec=2.0,
        hook_end_sec=10.0,
    )
    r = clip_for_render(c)
    assert r.start_time_sec == 100.0
    assert r.end_time_sec == 130.0
    assert r.trim_start_sec == 0.0
    assert r.hook_start_sec is None


def test_render_spans_override_contiguous_trim_window():
    c = _clip(
        trim_start_sec=5.0,
        trim_end_sec=5.0,
        render_spans=[
            ClipRenderSpan(start_time_sec=101.0, end_time_sec=110.0),
            ClipRenderSpan(start_time_sec=115.0, end_time_sec=120.0),
        ],
    )
    lo, hi = effective_export_bounds(c)
    assert lo == 101.0
    assert hi == 120.0
