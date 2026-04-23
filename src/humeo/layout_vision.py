"""Per-clip layout + bbox via Gemini vision (no pixel heuristics in the product pipeline)."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import struct
import subprocess
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from openai import OpenAI

from humeo_core.schemas import (
    BoundingBox,
    LayoutInstruction,
    LayoutKind,
    Scene,
    SceneClassification,
    SceneRegions,
    TimedCenterPoint,
)
from humeo_core.primitives.vision import layout_instruction_from_regions

from humeo.config import GEMINI_MODEL, GEMINI_VISION_MODEL, PipelineConfig
from humeo.env import (
    OPENROUTER_BASE_URL,
    current_llm_provider,
    model_name_for_provider,
    openrouter_default_headers,
    resolve_gemini_api_key,
    resolve_llm_provider,
    resolve_openrouter_api_key,
)
from humeo.gemini_generate import gemini_generate_config

logger = logging.getLogger(__name__)

LAYOUT_VISION_CACHE_VERSION = 5
LAYOUT_VISION_META = "layout_vision.meta.json"
LAYOUT_VISION_JSON = "layout_vision.json"
TRACKING_SAMPLE_FRACTIONS = tuple(i / 10.0 for i in range(1, 10))
TRACKING_MIN_SPREAD_NORM = 0.08
TRACKING_OUTLIER_DELTA_NORM = 0.16
TRACKING_OUTLIER_NEIGHBOR_MAX_NORM = 0.10
TRACKING_DEADBAND_NORM = 0.025
TRACKING_MIN_USABLE_POINTS = 5
TRACKING_UNSTABLE_JUMP_NORM = 0.18
REPLICATE_SAM2_VIDEO_PINNED = (
    "meta/sam-2-video:2d7219877ca847f463d749d9b224e62f7b078fe035d60a74b58889b455d5cbad"
)

GEMINI_LAYOUT_VISION_PROMPT = """You are framing a vertical short (9:16) from a 16:9 video frame.

HARD RULE: the final short shows AT MOST TWO on-screen items. An "item" is one
of person (a human speaker) or chart (slide, graph, data visual, screenshare).
That gives exactly five layouts to choose from.

Return ONLY a JSON object with this exact shape:
{
  "layout": "zoom_call_center" | "sit_center" | "split_chart_person" | "split_two_persons" | "split_two_charts",
  "person_bbox":        {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0} | null,
  "face_bbox":          {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0} | null,
  "chart_bbox":         {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0} | null,
  "second_person_bbox": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0} | null,
  "second_face_bbox":   {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0} | null,
  "second_chart_bbox":  {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0} | null,
  "reason": "short rationale"
}

Bbox rules:
- All bbox coordinates are normalized 0..1 (left/top = 0, right/bottom = 1). Require x2 > x1 and y2 > y1 when a bbox is non-null.
- person_bbox / second_person_bbox: tight box around each speaker's head AND upper body. If two speakers are visible, ``person_bbox`` is the LEFT speaker and ``second_person_bbox`` is the RIGHT speaker (by x-center).
- face_bbox / second_face_bbox: TIGHT box around the SPEAKER'S FACE ONLY (forehead to chin, ear to ear). This is NOT the full body — exclude torso, arms, shoulders, tank top, mug, table. The face bbox drives horizontal framing in the 9:16 crop, so putting torso or arms in it will push the face off-screen.
  * If the subject is shown in profile, the face_bbox still surrounds only the visible half of the head (ear to nose, forehead to chin). It should be roughly square-ish, not a tall body rectangle.
  * ``face_bbox`` matches ``person_bbox`` (same speaker), ``second_face_bbox`` matches ``second_person_bbox``.
  * Set face bbox to null ONLY if no face is visible at all (back of head, occluded, off-frame).
- chart_bbox / second_chart_bbox: slide, chart, graph, or large on-screen graphic. If two charts are visible, ``chart_bbox`` is the LEFT chart and ``second_chart_bbox`` is the RIGHT chart.
- The two bboxes of the same kind must not overlap meaningfully; they should partition the source frame into distinct regions.

Layout selection (pick exactly one):
- zoom_call_center: ONE person, tight webcam / video-call headshot filling much of the frame. person_bbox + face_bbox set; others null.
- sit_center: ONE person, interview / seated framing, or when unsure. person_bbox + face_bbox set; others null.
- split_chart_person: ONE chart + ONE person in distinct regions (webinar / explainer). person_bbox + face_bbox + chart_bbox set; second_* null.
- split_two_persons: TWO visible speakers (interview two-up, podcast panel). person_bbox + face_bbox AND second_person_bbox + second_face_bbox set; chart bboxes null.
- split_two_charts: TWO charts / slides side-by-side. chart_bbox AND second_chart_bbox set; person/face bboxes null.

When in doubt prefer ``sit_center``. Never output more than two of {person, chart} items in total.
No markdown. JSON only."""

ACTIVE_SPEAKER_VISION_PROMPT = """You are analyzing a single frame from a two-person talking video.

Return ONLY a JSON object:
{
  "speaker": "left" | "right" | "both" | "unclear",
  "reason": "short rationale"
}

Rules:
- "left" means the LEFT visible person appears to be the one speaking in this exact frame.
- "right" means the RIGHT visible person appears to be the one speaking in this exact frame.
- Use visible cues only: open mouth mid-word, facial expression while talking, hand gesture timing, body engagement.
- If both appear to be talking at once, return "both".
- If it is impossible to tell from this frame, return "unclear".
- No markdown. JSON only."""


def _openai_message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _clips_fingerprint(clips_path: Path) -> str:
    if not clips_path.is_file():
        return ""
    return hashlib.sha256(clips_path.read_bytes()).hexdigest()


def layout_cache_valid(
    work_dir: Path,
    *,
    transcript_fp: str,
    clips_fp: str,
    vision_model: str,
    segmentation_provider: str = "off",
    segmentation_model: str = "meta/sam-2-video",
) -> bool:
    meta_path = work_dir / LAYOUT_VISION_META
    data_path = work_dir / LAYOUT_VISION_JSON
    if not meta_path.is_file() or not data_path.is_file():
        return False
    try:
        meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        meta.get("layout_vision_cache_version") == LAYOUT_VISION_CACHE_VERSION
        and
        meta.get("transcript_sha256") == transcript_fp
        and meta.get("clips_sha256") == clips_fp
        and meta.get("gemini_vision_model") == vision_model
        and meta.get("segmentation_provider", "off") == segmentation_provider
        and meta.get("segmentation_model", "meta/sam-2-video") == segmentation_model
        and (
            current_llm_provider() is None
            or (
                current_llm_provider() == "google"
                and meta.get("llm_backend") in (None, "google")
            )
            or meta.get("llm_backend") == current_llm_provider()
        )
    )


def load_layout_cache(work_dir: Path) -> dict[str, dict[str, Any]] | None:
    p = work_dir / LAYOUT_VISION_JSON
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    clips = data.get("clips")
    return clips if isinstance(clips, dict) else None


def write_layout_cache(
    work_dir: Path,
    *,
    transcript_fp: str,
    clips_fp: str,
    vision_model: str,
    clips_payload: dict[str, dict[str, Any]],
    segmentation_provider: str = "off",
    segmentation_model: str = "meta/sam-2-video",
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "layout_vision_cache_version": LAYOUT_VISION_CACHE_VERSION,
        "transcript_sha256": transcript_fp,
        "clips_sha256": clips_fp,
        "gemini_vision_model": vision_model,
        "llm_backend": current_llm_provider() or "google",
        "segmentation_provider": segmentation_provider,
        "segmentation_model": segmentation_model,
    }
    (work_dir / LAYOUT_VISION_META).write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    (work_dir / LAYOUT_VISION_JSON).write_text(
        json.dumps({"clips": clips_payload}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote %s and %s", LAYOUT_VISION_META, LAYOUT_VISION_JSON)


def _png_dims(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as f:
            head = f.read(24)
        if head[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        width, height = struct.unpack(">II", head[16:24])
        return int(width), int(height)
    except Exception:
        return None


def _jpeg_dims(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as f:
            if f.read(2) != b"\xff\xd8":
                return None
            sof_markers = {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }
            while True:
                marker_start = f.read(1)
                if not marker_start:
                    return None
                if marker_start != b"\xff":
                    continue
                marker = f.read(1)
                while marker == b"\xff":
                    marker = f.read(1)
                if not marker:
                    return None
                marker_byte = marker[0]
                if marker_byte in (0xD8, 0xD9, 0x01) or 0xD0 <= marker_byte <= 0xD7:
                    continue
                seg_len_bytes = f.read(2)
                if len(seg_len_bytes) != 2:
                    return None
                seg_len = struct.unpack(">H", seg_len_bytes)[0]
                if seg_len < 2:
                    return None
                if marker_byte in sof_markers:
                    frame_header = f.read(5)
                    if len(frame_header) != 5:
                        return None
                    _, height, width = struct.unpack(">BHH", frame_header)
                    return int(width), int(height)
                f.seek(seg_len - 2, 1)
    except Exception:
        return None


def _keyframe_dimensions(keyframe_path: str) -> tuple[int, int] | None:
    path = Path(keyframe_path)
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as img:
            width, height = img.size
        return int(width), int(height)
    except Exception:
        pass

    png_dims = _png_dims(path)
    if png_dims is not None:
        return png_dims
    return _jpeg_dims(path)


def _normalize_bbox_payload(
    raw: dict[str, Any], image_size: tuple[int, int] | None
) -> dict[str, Any]:
    if image_size is None:
        return dict(raw)

    width, height = image_size
    normalized = dict(raw)
    x_values = [
        float(normalized[key])
        for key in ("x1", "x2")
        if isinstance(normalized.get(key), (int, float))
    ]
    y_values = [
        float(normalized[key])
        for key in ("y1", "y2")
        if isinstance(normalized.get(key), (int, float))
    ]

    if not x_values and not y_values:
        return normalized

    use_thousand_grid = False
    if any(v > 1.0 for v in x_values + y_values):
        max_coord = max(x_values + y_values)
        fits_image_pixels = (
            all(v <= float(width) for v in x_values)
            and all(v <= float(height) for v in y_values)
        )
        if max_coord <= 1000.0 and not fits_image_pixels:
            use_thousand_grid = True

    x_scale = 1000.0 if use_thousand_grid else float(width)
    y_scale = 1000.0 if use_thousand_grid else float(height)

    axis_scales = {
        "x1": x_scale,
        "x2": x_scale,
        "y1": y_scale,
        "y2": y_scale,
    }
    for key, axis_scale in axis_scales.items():
        value = normalized.get(key)
        if not isinstance(value, (int, float)):
            continue
        coord = float(value)
        if coord > 1.0 and axis_scale > 0.0:
            coord = coord / axis_scale
        normalized[key] = max(0.0, min(coord, 1.0))
    return normalized


def _parse_bbox(
    raw: object, *, image_size: tuple[int, int] | None = None
) -> BoundingBox | None:
    if not raw or not isinstance(raw, dict):
        return None
    try:
        return BoundingBox.model_validate(_normalize_bbox_payload(raw, image_size))
    except Exception:
        return None


def _instruction_from_gemini_json(
    scene_id: str,
    data: dict[str, Any],
    *,
    image_size: tuple[int, int] | None = None,
) -> LayoutInstruction:
    """Translate Gemini's JSON into a validated :class:`LayoutInstruction`.

    Falls back to ``sit_center`` whenever the LLM returns something the
    contract doesn't support, so a bad vision call can never crash the
    pipeline. Also downgrades "two-item" layouts when the second bbox is
    missing -- e.g. ``split_two_persons`` with only one person_bbox drops
    to ``sit_center`` rather than rendering a silently-broken split.
    """

    layout_str = str(data.get("layout", "sit_center")).strip()
    try:
        kind = LayoutKind(layout_str)
    except ValueError:
        kind = LayoutKind.SIT_CENTER

    pb = _parse_bbox(data.get("person_bbox"), image_size=image_size)
    fb = _parse_bbox(data.get("face_bbox"), image_size=image_size)
    cb = _parse_bbox(data.get("chart_bbox"), image_size=image_size)
    p2 = _parse_bbox(data.get("second_person_bbox"), image_size=image_size)
    f2 = _parse_bbox(data.get("second_face_bbox"), image_size=image_size)
    c2 = _parse_bbox(data.get("second_chart_bbox"), image_size=image_size)
    reason = str(data.get("reason", ""))[:400]

    # Downgrade any split that is missing its required bboxes, so we never
    # emit a split layout that will render as garbage.
    if kind == LayoutKind.SPLIT_CHART_PERSON and (pb is None or cb is None):
        kind = LayoutKind.SIT_CENTER if pb is not None else LayoutKind.SIT_CENTER
    if kind == LayoutKind.SPLIT_TWO_PERSONS and (pb is None or p2 is None):
        kind = LayoutKind.SIT_CENTER
    if kind == LayoutKind.SPLIT_TWO_CHARTS and (cb is None or c2 is None):
        kind = LayoutKind.SIT_CENTER

    regions = SceneRegions(
        scene_id=scene_id, person_bbox=pb, chart_bbox=cb, raw_reason=reason
    )
    classification = SceneClassification(
        scene_id=scene_id, layout=kind, confidence=1.0, reason=reason
    )
    instr = layout_instruction_from_regions(
        regions, classification, clip_id=scene_id
    )

    updates: dict[str, Any] = {}

    # CENTERING FIX: the single-person 9:16 crop is driven by ``person_x_norm``.
    # A ``person_bbox`` that spans head + torso + arms is fine for framing
    # *extent* but its center_x can drift far from the actual face when the
    # subject is in profile or asymmetric (one arm up, mug on the table, etc).
    # Prefer the tight ``face_bbox`` center when the model gave us one so the
    # face lands in the visual center of the vertical crop instead of the
    # torso doing.
    face_center = _face_center_x(fb, pb)
    if face_center is not None:
        updates["person_x_norm"] = face_center

    if kind == LayoutKind.SPLIT_CHART_PERSON and pb is not None and cb is not None:
        updates["split_chart_region"] = cb
        updates["split_person_region"] = pb
    elif kind == LayoutKind.SPLIT_TWO_PERSONS and pb is not None and p2 is not None:
        # Order by x-center so ``split_person_region`` is always the LEFT speaker.
        left, right = sorted((pb, p2), key=lambda b: b.center_x)
        updates["split_person_region"] = left
        updates["split_second_person_region"] = right
    elif kind == LayoutKind.SPLIT_TWO_CHARTS and cb is not None and c2 is not None:
        left, right = sorted((cb, c2), key=lambda b: b.center_x)
        updates["split_chart_region"] = left
        updates["split_second_chart_region"] = right

    if updates:
        instr = instr.model_copy(update=updates)
    return instr


def _face_center_x(
    face: BoundingBox | None, person: BoundingBox | None
) -> float | None:
    """Pick a horizontal center to aim the 9:16 crop at.

    Priority:
    1. ``face`` bbox center when it looks reasonable (narrow, plausibly
       inside the matching person bbox).
    2. No override (caller keeps the person-bbox center, or the default 0.5
       when neither was provided).

    We sanity-check the face box because Gemini sometimes echoes the full
    person bbox into ``face_bbox``. If the face bbox is as wide as the
    person bbox, it gives us nothing new; fall back to the person center
    rather than pretending we have a tighter signal.
    """
    if face is None:
        return None
    face_w = max(0.0, face.x2 - face.x1)
    if face_w <= 0.0:
        return None
    # A real face in a 16:9 frame is rarely wider than ~35% of frame width,
    # even for tight webcam framing. A face "bbox" that's wider than that
    # almost certainly includes torso and is no better than person_bbox.
    if face_w > 0.40:
        return None
    # If we have a person bbox too, require the face center to sit inside it
    # — otherwise the model got confused and matched the wrong subject.
    if person is not None:
        if not (person.x1 - 0.02 <= face.center_x <= person.x2 + 0.02):
            return None
    return float(face.center_x)


def _person_center_x_from_data(
    data: dict[str, Any], image_size: tuple[int, int] | None = None
) -> float | None:
    person_bbox = _parse_bbox(data.get("person_bbox"), image_size=image_size)
    face_bbox = _parse_bbox(data.get("face_bbox"), image_size=image_size)
    face_center = _face_center_x(face_bbox, person_bbox)
    if face_center is not None:
        return face_center
    if person_bbox is not None:
        return float(person_bbox.center_x)
    return None


def _tracking_sample_times(duration_sec: float) -> list[float]:
    seen: set[float] = set()
    out: list[float] = []
    for fraction in TRACKING_SAMPLE_FRACTIONS:
        t_sec = max(0.0, min(duration_sec, duration_sec * fraction))
        key = round(t_sec, 3)
        if key in seen:
            continue
        seen.add(key)
        out.append(t_sec)
    return out


def _tracking_points_from_centers(
    duration_sec: float, centers: list[tuple[float, float]]
) -> list[TimedCenterPoint]:
    deduped: list[tuple[float, float]] = []
    for t_sec, x_norm in sorted(centers, key=lambda item: item[0]):
        clamped_t = max(0.0, min(duration_sec, float(t_sec)))
        clamped_x = max(0.0, min(1.0, float(x_norm)))
        if deduped and abs(clamped_t - deduped[-1][0]) < 1e-6:
            deduped[-1] = (clamped_t, clamped_x)
        else:
            deduped.append((clamped_t, clamped_x))

    if len(deduped) < 2:
        return []

    filtered = list(deduped)
    for idx in range(1, len(filtered) - 1):
        prev_x = filtered[idx - 1][1]
        curr_t, curr_x = filtered[idx]
        next_x = filtered[idx + 1][1]
        if (
            abs(prev_x - next_x) <= TRACKING_OUTLIER_NEIGHBOR_MAX_NORM
            and abs(curr_x - prev_x) >= TRACKING_OUTLIER_DELTA_NORM
            and abs(curr_x - next_x) >= TRACKING_OUTLIER_DELTA_NORM
        ):
            filtered[idx] = (curr_t, (prev_x + next_x) / 2.0)

    smoothed = list(filtered)
    for idx in range(1, len(filtered) - 1):
        prev_x = filtered[idx - 1][1]
        curr_t, curr_x = filtered[idx]
        next_x = filtered[idx + 1][1]
        median_x = sorted((prev_x, curr_x, next_x))[1]
        if abs(curr_x - median_x) > TRACKING_DEADBAND_NORM:
            smoothed[idx] = (curr_t, median_x)

    filtered = list(smoothed)
    for idx in range(1, len(filtered)):
        prev_t, prev_x = filtered[idx - 1]
        curr_t, curr_x = filtered[idx]
        if abs(curr_x - prev_x) < TRACKING_DEADBAND_NORM:
            filtered[idx] = (curr_t, prev_x)

    spread = max(x for _, x in filtered) - min(x for _, x in filtered)
    if spread < TRACKING_MIN_SPREAD_NORM:
        stable_x = sum(x for _, x in filtered) / len(filtered)
        return [
            TimedCenterPoint(t_sec=0.0, x_norm=stable_x),
            TimedCenterPoint(t_sec=duration_sec, x_norm=stable_x),
        ]

    if filtered[0][0] > 0.0:
        filtered.insert(0, (0.0, filtered[0][1]))
    else:
        filtered[0] = (0.0, filtered[0][1])

    if filtered[-1][0] < duration_sec:
        filtered.append((duration_sec, filtered[-1][1]))
    else:
        filtered[-1] = (duration_sec, filtered[-1][1])

    return [TimedCenterPoint(t_sec=t_sec, x_norm=x_norm) for t_sec, x_norm in filtered]


def _tracking_is_unstable(points: list[TimedCenterPoint]) -> bool:
    if len(points) < TRACKING_MIN_USABLE_POINTS:
        return True
    return any(
        abs(points[idx].x_norm - points[idx - 1].x_norm) > TRACKING_UNSTABLE_JUMP_NORM
        for idx in range(1, len(points))
    )


def _interpolate_tracking_x(points: list[TimedCenterPoint], t_sec: float) -> float | None:
    if not points:
        return None
    if t_sec <= points[0].t_sec:
        return float(points[0].x_norm)
    if t_sec >= points[-1].t_sec:
        return float(points[-1].x_norm)
    for idx in range(1, len(points)):
        left = points[idx - 1]
        right = points[idx]
        if right.t_sec < t_sec:
            continue
        span = right.t_sec - left.t_sec
        if span <= 1e-6:
            return float(right.x_norm)
        alpha = (t_sec - left.t_sec) / span
        return float(left.x_norm + (right.x_norm - left.x_norm) * alpha)
    return float(points[-1].x_norm)


def _speaker_seed_boxes(
    data: dict[str, Any], image_size: tuple[int, int] | None
) -> tuple[BoundingBox, BoundingBox] | None:
    first_person = _parse_bbox(data.get("person_bbox"), image_size=image_size)
    first_face = _parse_bbox(data.get("face_bbox"), image_size=image_size)
    second_person = _parse_bbox(data.get("second_person_bbox"), image_size=image_size)
    second_face = _parse_bbox(data.get("second_face_bbox"), image_size=image_size)
    left = first_face or first_person
    right = second_face or second_person
    if left is None or right is None:
        return None
    ordered = sorted((left, right), key=lambda box: box.center_x)
    return ordered[0], ordered[1]


def _speaker_follow_sample_times(duration_sec: float) -> list[float]:
    seen: set[float] = set()
    out: list[float] = []
    for t_sec in [0.0, *_tracking_sample_times(duration_sec), duration_sec]:
        key = round(max(0.0, min(duration_sec, t_sec)), 3)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _resolve_speaker_focus_samples(
    samples: list[tuple[float, str]],
    *,
    default_side: str = "left",
) -> list[tuple[float, str]]:
    resolved: list[list[float | str | None]] = []
    prev_side: str | None = None
    for t_sec, side in samples:
        normalized = side if side in ("left", "right") else None
        if normalized is None and prev_side is not None:
            normalized = prev_side
        if normalized is not None:
            prev_side = normalized
        resolved.append([t_sec, normalized])

    next_side: str | None = None
    for idx in range(len(resolved) - 1, -1, -1):
        if resolved[idx][1] is None:
            resolved[idx][1] = next_side
        else:
            next_side = str(resolved[idx][1])

    out: list[tuple[float, str]] = []
    for t_sec, side in resolved:
        out.append((float(t_sec), str(side or default_side)))
    return out


def _extract_frame_at_time(source_path: Path, time_sec: float, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{time_sec:.3f}",
            "-i",
            str(source_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    return output_path


def _probe_video_fps(source_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(source_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    rate = (result.stdout or "").strip()
    if "/" in rate:
        num, den = rate.split("/", 1)
        try:
            return max(1.0, float(num) / max(float(den), 1.0))
        except ValueError:
            return 30.0
    try:
        return max(1.0, float(rate))
    except ValueError:
        return 30.0


def _mask_center_x_from_url(mask_url: str) -> float | None:
    try:
        import httpx
        from PIL import Image  # type: ignore
    except ImportError:
        return None

    response = httpx.get(mask_url, timeout=120.0)
    response.raise_for_status()
    with Image.open(BytesIO(response.content)) as image:
        image = image.convert("L")
        width, height = image.size
        pixels = image.load()
        xs: list[int] = []
        for y in range(height):
            for x in range(width):
                if pixels[x, y] > 16:
                    xs.append(x)
        if not xs or width <= 0:
            return None
        return float(sum(xs) / len(xs) / width)


def _segmentation_mask_urls(output: object) -> list[str]:
    def _coerce_urls(items: Iterable[object]) -> list[str]:
        urls: list[str] = []
        for item in items:
            if item is None:
                continue
            if isinstance(item, (str, Path)):
                text = str(item).strip()
            else:
                url = getattr(item, "url", None)
                text = str(url).strip() if isinstance(url, str) else str(item).strip()
            if text:
                urls.append(text)
        return urls

    if isinstance(output, dict):
        for key in ("black_white_masks", "masks", "output"):
            value = output.get(key)
            if isinstance(value, (str, bytes, bytearray)) or value is None:
                continue
            try:
                urls = _coerce_urls(value)
            except TypeError:
                continue
            if urls:
                return urls
        return []
    if isinstance(output, (str, bytes, bytearray)) or output is None:
        return []
    try:
        return _coerce_urls(output)
    except TypeError:
        return []


def _infer_person_tracking_with_segmentation(
    scene: Scene,
    *,
    source_video: Path,
    segmentation_model: str,
    initial_data: dict[str, Any] | None = None,
    initial_image_size: tuple[int, int] | None = None,
    seed_bbox: BoundingBox | None = None,
    object_id: str = "speaker",
) -> tuple[list[TimedCenterPoint], dict[str, Any] | None]:
    token = (os.environ.get("REPLICATE_API_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("REPLICATE_API_TOKEN is not set")
    if initial_image_size is None:
        raise RuntimeError("Segmentation fallback requires the keyframe dimensions")
    if seed_bbox is None and initial_data is None:
        raise RuntimeError("Segmentation fallback requires an initial vision bbox")

    if seed_bbox is None:
        face_bbox = _parse_bbox(initial_data.get("face_bbox"), image_size=initial_image_size)
        person_bbox = _parse_bbox(initial_data.get("person_bbox"), image_size=initial_image_size)
        seed_bbox = face_bbox or person_bbox
    if seed_bbox is None:
        raise RuntimeError("No seed bbox available for segmentation fallback")

    try:
        import replicate
    except ImportError as exc:
        raise RuntimeError("replicate package is not installed") from exc

    width, height = initial_image_size
    fps = _probe_video_fps(source_video)
    midpoint_frame = max(0, int(round((scene.duration / 2.0) * fps)))
    output_frame_interval = max(1, int(round(max(1.0, scene.duration * fps) / 10.0)))
    click_x = int(round(seed_bbox.center_x * width))
    click_y = int(round(seed_bbox.center_y * height))
    prompt_frames = [0]
    if midpoint_frame > 0:
        prompt_frames.append(midpoint_frame)
    prompt_coordinates = ",".join(f"[{click_x},{click_y}]" for _ in prompt_frames)
    prompt_labels = ",".join("1" for _ in prompt_frames)
    prompt_frame_str = ",".join(str(frame_idx) for frame_idx in prompt_frames)
    prompt_object_ids = ",".join(object_id for _ in prompt_frames)
    run_input = {
        "input_video": None,
        "click_coordinates": prompt_coordinates,
        "click_labels": prompt_labels,
        "click_frames": prompt_frame_str,
        "click_object_ids": prompt_object_ids,
        "mask_type": "binary",
        "annotation_type": "mask",
        "output_video": False,
        "output_format": "png",
        "output_frame_interval": output_frame_interval,
    }

    with source_video.open("rb") as handle:
        client = replicate.Client(api_token=token)
        run_input["input_video"] = handle
        try:
            output = client.run(segmentation_model, input=run_input)
            resolved_model = segmentation_model
        except Exception as exc:
            if ":" in segmentation_model or "404" not in str(exc):
                raise
            handle.seek(0)
            output = client.run(REPLICATE_SAM2_VIDEO_PINNED, input=run_input)
            resolved_model = REPLICATE_SAM2_VIDEO_PINNED

    urls = _segmentation_mask_urls(output)
    if not urls:
        raise RuntimeError("Segmentation fallback returned no masks")

    centers: list[tuple[float, float]] = []
    for idx, mask_url in enumerate(urls):
        center_x = _mask_center_x_from_url(mask_url)
        if center_x is None:
            continue
        rel_time = min(scene.duration, (idx * output_frame_interval) / fps)
        centers.append((rel_time, center_x))

    points = _tracking_points_from_centers(scene.duration, centers)
    detail = {
        "provider": "replicate",
        "model": resolved_model,
        "seed_point_px": [click_x, click_y],
        "seed_frame": midpoint_frame,
        "prompt_frames": prompt_frames,
        "output_frame_interval": output_frame_interval,
        "mask_count": len(urls),
    }
    return points, detail


def _infer_two_speaker_focus_tracking_with_segmentation(
    scene: Scene,
    *,
    source_video: Path,
    tracking_dir: Path,
    model_name: str,
    segmentation_model: str,
    initial_data: dict[str, Any],
    initial_image_size: tuple[int, int] | None,
) -> tuple[list[TimedCenterPoint], dict[str, Any] | None]:
    seeds = _speaker_seed_boxes(initial_data, initial_image_size)
    if seeds is None:
        raise RuntimeError("Two-speaker SAM follow requires both speaker bboxes")

    left_seed, right_seed = seeds
    left_points, left_detail = _infer_person_tracking_with_segmentation(
        scene,
        source_video=source_video,
        segmentation_model=segmentation_model,
        initial_data=initial_data,
        initial_image_size=initial_image_size,
        seed_bbox=left_seed,
        object_id="left_speaker",
    )
    right_points, right_detail = _infer_person_tracking_with_segmentation(
        scene,
        source_video=source_video,
        segmentation_model=segmentation_model,
        initial_data=initial_data,
        initial_image_size=initial_image_size,
        seed_bbox=right_seed,
        object_id="right_speaker",
    )
    if not left_points or not right_points:
        raise RuntimeError("Two-speaker SAM follow did not return both speaker tracks")

    focus_dir = tracking_dir / scene.scene_id / "speaker_focus"
    focus_samples: list[dict[str, Any]] = []
    focus_choices: list[tuple[float, str]] = []
    for rel_time in _speaker_follow_sample_times(max(0.0, scene.duration)):
        abs_time = scene.start_time + rel_time
        frame_path = focus_dir / f"{scene.scene_id}_{int(round(rel_time * 1000)):06d}.jpg"
        try:
            _extract_frame_at_time(source_video, abs_time, frame_path)
            data = _call_active_speaker_vision(str(frame_path), model_name)
            speaker = str(data.get("speaker", "unclear")).strip().lower()
            if speaker not in ("left", "right", "both", "unclear"):
                speaker = "unclear"
            focus_choices.append((rel_time, speaker))
            focus_samples.append(
                {
                    "time_sec": rel_time,
                    "frame_path": str(frame_path),
                    "speaker": speaker,
                    "raw": data,
                }
            )
        except Exception as exc:
            focus_choices.append((rel_time, "unclear"))
            focus_samples.append(
                {
                    "time_sec": rel_time,
                    "frame_path": str(frame_path),
                    "speaker": "unclear",
                    "error": str(exc),
                }
            )

    resolved_focus = _resolve_speaker_focus_samples(focus_choices, default_side="left")
    centers: list[tuple[float, float]] = []
    for rel_time, speaker in resolved_focus:
        x_norm = (
            _interpolate_tracking_x(left_points, rel_time)
            if speaker == "left"
            else _interpolate_tracking_x(right_points, rel_time)
        )
        if x_norm is None:
            x_norm = left_seed.center_x if speaker == "left" else right_seed.center_x
        centers.append((rel_time, x_norm))

    points = _tracking_points_from_centers(scene.duration, centers)
    detail = {
        "mode": "two_speaker_follow",
        "left_segmentation": left_detail,
        "right_segmentation": right_detail,
        "focus_samples": focus_samples,
        "resolved_focus": [
            {"time_sec": rel_time, "speaker": speaker} for rel_time, speaker in resolved_focus
        ],
    }
    return points, detail


def _infer_person_tracking(
    scene: Scene,
    *,
    source_video: Path,
    tracking_dir: Path,
    model_name: str,
    initial_data: dict[str, Any] | None = None,
    initial_image_size: tuple[int, int] | None = None,
) -> tuple[list[TimedCenterPoint], list[dict[str, Any]]]:
    duration_sec = max(0.0, scene.duration)
    if duration_sec <= 0.0:
        return [], []

    midpoint_rel = duration_sec / 2.0
    centers: list[tuple[float, float]] = []
    samples: list[dict[str, Any]] = []

    if initial_data is not None:
        center_x = _person_center_x_from_data(initial_data, image_size=initial_image_size)
        samples.append(
            {
                "sample_kind": "midpoint_keyframe",
                "time_sec": midpoint_rel,
                "frame_path": scene.keyframe_path,
                "center_x_norm": center_x,
                "raw": initial_data,
            }
        )
        if center_x is not None:
            centers.append((midpoint_rel, center_x))

    scene_tracking_dir = tracking_dir / scene.scene_id
    for rel_time in _tracking_sample_times(duration_sec):
        if abs(rel_time - midpoint_rel) < 1e-3:
            continue
        abs_time = scene.start_time + rel_time
        frame_path = scene_tracking_dir / f"{scene.scene_id}_{int(round(rel_time * 1000)):06d}.jpg"
        try:
            _extract_frame_at_time(source_video, abs_time, frame_path)
            data = _call_gemini_vision(str(frame_path), model_name)
            image_size = _keyframe_dimensions(str(frame_path))
            center_x = _person_center_x_from_data(data, image_size=image_size)
            samples.append(
                {
                    "sample_kind": "tracking_frame",
                    "time_sec": rel_time,
                    "frame_path": str(frame_path),
                    "center_x_norm": center_x,
                    "raw": data,
                }
            )
            if center_x is not None:
                centers.append((rel_time, center_x))
        except Exception as e:
            logger.warning(
                "Speaker tracking sample failed for %s at %.2fs: %s",
                scene.scene_id,
                rel_time,
                e,
            )
            samples.append(
                {
                    "sample_kind": "tracking_frame",
                    "time_sec": rel_time,
                    "frame_path": str(frame_path),
                    "error": str(e),
                }
            )

    return _tracking_points_from_centers(duration_sec, centers), samples


def _call_vision_json(keyframe_path: str, model_name: str, prompt: str) -> dict[str, Any]:
    path = Path(keyframe_path)
    data = path.read_bytes()
    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    provider = resolve_llm_provider()
    resolved_model = model_name_for_provider(model_name, provider)

    if provider == "google":
        client = genai.Client(api_key=resolve_gemini_api_key())
        response = client.models.generate_content(
            model=resolved_model,
            contents=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=data, mime_type=mime),
            ],
            config=gemini_generate_config(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
        if not response.text:
            raise RuntimeError("Gemini vision returned empty response")
        return json.loads(response.text)

    client = OpenAI(
        api_key=resolve_openrouter_api_key(),
        base_url=OPENROUTER_BASE_URL,
        default_headers=openrouter_default_headers(),
    )
    data_url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    response = client.chat.completions.create(
        model=resolved_model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this keyframe and return only JSON."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    text = _openai_message_text(response.choices[0].message.content)
    if not text:
        raise RuntimeError("OpenRouter vision returned empty response")
    return json.loads(text)


def _call_gemini_vision(keyframe_path: str, model_name: str) -> dict[str, Any]:
    return _call_vision_json(keyframe_path, model_name, GEMINI_LAYOUT_VISION_PROMPT)


def _call_active_speaker_vision(frame_path: str, model_name: str) -> dict[str, Any]:
    return _call_vision_json(frame_path, model_name, ACTIVE_SPEAKER_VISION_PROMPT)


def infer_layout_instructions(
    scenes: list[Scene],
    *,
    gemini_vision_model: str,
    source_video: Path,
    tracking_dir: Path,
    source_videos_by_scene: dict[str, Path] | None = None,
    segmentation_provider: str = "off",
    segmentation_model: str = "meta/sam-2-video",
) -> tuple[dict[str, LayoutInstruction], dict[str, dict[str, Any]]]:
    """Return ``(clip_id -> LayoutInstruction, clip_id -> raw_gemini_json)``."""

    out: dict[str, LayoutInstruction] = {}
    raw_by_clip: dict[str, dict[str, Any]] = {}
    model_name = gemini_vision_model.strip()

    for s in scenes:
        sid = s.scene_id
        if not s.keyframe_path:
            logger.warning("No keyframe for %s; using sit_center.", sid)
            out[sid] = LayoutInstruction(clip_id=sid, layout=LayoutKind.SIT_CENTER)
            raw_by_clip[sid] = {"error": "no keyframe", "layout": "sit_center"}
            continue
        try:
            data = _call_gemini_vision(s.keyframe_path, model_name)
            image_size = _keyframe_dimensions(s.keyframe_path)
            instr = _instruction_from_gemini_json(
                sid,
                data,
                image_size=image_size,
            )
            raw_data = dict(data)
            speaker_follow_applied = False
            tracking_source = (
                source_videos_by_scene.get(sid, source_video)
                if source_videos_by_scene
                else source_video
            )
            if instr.layout == LayoutKind.SPLIT_TWO_PERSONS and segmentation_provider == "replicate":
                try:
                    focus_points, focus_detail = _infer_two_speaker_focus_tracking_with_segmentation(
                        s,
                        source_video=tracking_source,
                        tracking_dir=tracking_dir,
                        model_name=model_name,
                        segmentation_model=segmentation_model,
                        initial_data=data,
                        initial_image_size=image_size,
                    )
                    if focus_detail:
                        raw_data["speaker_follow_tracking"] = focus_detail
                    if focus_points:
                        instr = LayoutInstruction(
                            clip_id=sid,
                            layout=LayoutKind.SIT_CENTER,
                            person_x_norm=focus_points[0].x_norm,
                            person_tracking=focus_points,
                        )
                        speaker_follow_applied = True
                except Exception as exc:
                    raw_data["speaker_follow_tracking"] = {"error": str(exc)}
            if instr.layout in (LayoutKind.SIT_CENTER, LayoutKind.ZOOM_CALL_CENTER):
                if speaker_follow_applied:
                    raw_by_clip[sid] = raw_data
                    out[sid] = instr
                    continue
                tracking_points: list[TimedCenterPoint] = []
                tracking_samples: list[dict[str, Any]] = []
                attempted_segmentation = False

                if segmentation_provider == "replicate":
                    attempted_segmentation = True
                    try:
                        sam_points, sam_detail = _infer_person_tracking_with_segmentation(
                            s,
                            source_video=tracking_source,
                            segmentation_model=segmentation_model,
                            initial_data=data,
                            initial_image_size=image_size,
                        )
                        if sam_points:
                            tracking_points = sam_points
                        if sam_detail:
                            raw_data["segmentation_tracking"] = sam_detail
                    except Exception as exc:
                        raw_data["segmentation_tracking"] = {"error": str(exc)}
                if not tracking_points:
                    tracking_points, tracking_samples = _infer_person_tracking(
                        s,
                        source_video=tracking_source,
                        tracking_dir=tracking_dir,
                        model_name=model_name,
                        initial_data=data,
                        initial_image_size=image_size,
                    )
                    if (
                        segmentation_provider == "replicate"
                        and _tracking_is_unstable(tracking_points)
                        and not attempted_segmentation
                    ):
                        try:
                            sam_points, sam_detail = _infer_person_tracking_with_segmentation(
                                s,
                                source_video=tracking_source,
                                segmentation_model=segmentation_model,
                                initial_data=data,
                                initial_image_size=image_size,
                            )
                            if sam_points:
                                tracking_points = sam_points
                            if sam_detail:
                                raw_data["segmentation_tracking"] = sam_detail
                        except Exception as exc:
                            raw_data.setdefault("segmentation_tracking", {"error": str(exc)})
                if tracking_points:
                    instr = instr.model_copy(update={"person_tracking": tracking_points})
                if tracking_samples:
                    raw_data["person_tracking_samples"] = tracking_samples
            raw_by_clip[sid] = raw_data
            out[sid] = instr
        except Exception as e:
            logger.warning("Gemini vision failed for %s: %s — defaulting sit_center", sid, e)
            out[sid] = LayoutInstruction(clip_id=sid, layout=LayoutKind.SIT_CENTER)
            raw_by_clip[sid] = {"error": str(e), "layout": "sit_center"}

    return out, raw_by_clip


def resolved_vision_model(config: PipelineConfig) -> str:
    if config.gemini_vision_model:
        return config.gemini_vision_model.strip()
    if GEMINI_VISION_MODEL:
        return GEMINI_VISION_MODEL
    return (config.gemini_model or GEMINI_MODEL).strip()


def run_layout_vision_stage(
    work_dir: Path,
    scenes: list[Scene],
    *,
    source_video: Path,
    source_videos_by_scene: dict[str, Path] | None = None,
    transcript_fp: str,
    clips_path: Path,
    config: PipelineConfig,
) -> dict[str, LayoutInstruction]:
    """Load cache or call Gemini vision for each keyframe; persist JSON artifacts."""
    clips_fp = _clips_fingerprint(clips_path)
    vm = resolved_vision_model(config)

    if (
        not config.force_layout_vision
        and layout_cache_valid(
            work_dir,
            transcript_fp=transcript_fp,
            clips_fp=clips_fp,
            vision_model=vm,
            segmentation_provider=config.segmentation_provider,
            segmentation_model=config.segmentation_model,
        )
    ):
        cached = load_layout_cache(work_dir)
        if cached:
            logger.info("Layout vision cache hit; skipping Gemini vision calls.")
            return {
                k: LayoutInstruction.model_validate(v["instruction"])
                for k, v in cached.items()
                if isinstance(v, dict) and "instruction" in v
            }

    instructions, raw_by_clip = infer_layout_instructions(
        scenes,
        gemini_vision_model=vm,
        source_video=source_video,
        tracking_dir=work_dir / "layout_tracking",
        source_videos_by_scene=source_videos_by_scene,
        segmentation_provider=config.segmentation_provider,
        segmentation_model=config.segmentation_model,
    )

    payload: dict[str, dict[str, Any]] = {}
    for sid, instr in instructions.items():
        payload[sid] = {
            "instruction": json.loads(instr.model_dump_json()),
            "raw": raw_by_clip.get(sid, {}),
        }
    write_layout_cache(
        work_dir,
        transcript_fp=transcript_fp,
        clips_fp=clips_fp,
        vision_model=vm,
        clips_payload=payload,
        segmentation_provider=config.segmentation_provider,
        segmentation_model=config.segmentation_model,
    )
    return instructions
