"""Per-clip layout + bbox via Gemini vision (no pixel heuristics in the product pipeline)."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import struct
import subprocess
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

LAYOUT_VISION_CACHE_VERSION = 4
LAYOUT_VISION_META = "layout_vision.meta.json"
LAYOUT_VISION_JSON = "layout_vision.json"
TRACKING_SAMPLE_FRACTIONS = tuple(i / 10.0 for i in range(1, 10))
TRACKING_MIN_SPREAD_NORM = 0.08
TRACKING_OUTLIER_DELTA_NORM = 0.16
TRACKING_OUTLIER_NEIGHBOR_MAX_NORM = 0.10

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
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "layout_vision_cache_version": LAYOUT_VISION_CACHE_VERSION,
        "transcript_sha256": transcript_fp,
        "clips_sha256": clips_fp,
        "gemini_vision_model": vision_model,
        "llm_backend": current_llm_provider() or "google",
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


def _bbox_unit_mode(raw: object) -> str | None:
    if not isinstance(raw, dict):
        return None
    values = [float(v) for v in raw.values() if isinstance(v, (int, float))]
    if not values:
        return None
    has_normalized = any(v <= 1.0 for v in values)
    has_pixelish = any(v > 1.0 for v in values)
    if has_normalized and has_pixelish:
        return "mixed"
    if has_pixelish:
        return "pixelish"
    return "normalized"


def _reject_mixed_scale_split_bbox(raw: object) -> object:
    if _bbox_unit_mode(raw) == "mixed":
        return None
    return raw


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

    person_bbox_raw = data.get("person_bbox")
    face_bbox_raw = data.get("face_bbox")
    second_person_bbox_raw = data.get("second_person_bbox")
    second_face_bbox_raw = data.get("second_face_bbox")
    if kind == LayoutKind.SPLIT_TWO_PERSONS:
        person_bbox_raw = _reject_mixed_scale_split_bbox(person_bbox_raw)
        second_person_bbox_raw = _reject_mixed_scale_split_bbox(second_person_bbox_raw)

    pb = _parse_bbox(person_bbox_raw, image_size=image_size)
    fb = _parse_bbox(face_bbox_raw, image_size=image_size)
    cb = _parse_bbox(data.get("chart_bbox"), image_size=image_size)
    p2 = _parse_bbox(second_person_bbox_raw, image_size=image_size)
    f2 = _parse_bbox(second_face_bbox_raw, image_size=image_size)
    c2 = _parse_bbox(data.get("second_chart_bbox"), image_size=image_size)
    reason = str(data.get("reason", ""))[:400]

    # Downgrade any split that is missing its required bboxes, so we never
    # emit a split layout that will render as garbage.
    if kind == LayoutKind.SPLIT_CHART_PERSON and (pb is None or cb is None):
        kind = LayoutKind.SIT_CENTER if pb is not None else LayoutKind.SIT_CENTER
    if kind == LayoutKind.SPLIT_TWO_PERSONS and (pb is None or p2 is None):
        if pb is None and p2 is not None:
            pb, fb = p2, f2
        elif p2 is None and pb is not None:
            pass
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


def _tracking_sample_center_x(
    data: dict[str, Any],
    *,
    image_size: tuple[int, int] | None,
    midpoint_layout: LayoutKind,
) -> float | None:
    """Return a tracking center only for samples compatible with midpoint layout.

    Ticket C keeps the sampling cadence itself unchanged; it only rejects
    sample outputs whose layout family disagrees with a one-person midpoint
    classification. That prevents a split-style outlier from seeding the
    opening crop of an otherwise single-speaker clip.
    """
    sample_layout_raw = str(data.get("layout", "")).strip()
    try:
        sample_layout = LayoutKind(sample_layout_raw)
    except ValueError:
        sample_layout = None

    if midpoint_layout in (LayoutKind.SIT_CENTER, LayoutKind.ZOOM_CALL_CENTER):
        if sample_layout not in (LayoutKind.SIT_CENTER, LayoutKind.ZOOM_CALL_CENTER):
            return None

    return _person_center_x_from_data(data, image_size=image_size)


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

    spread = max(x for _, x in filtered) - min(x for _, x in filtered)
    if spread < TRACKING_MIN_SPREAD_NORM:
        return []

    if filtered[0][0] > 0.0:
        filtered.insert(0, (0.0, filtered[0][1]))
    else:
        filtered[0] = (0.0, filtered[0][1])

    if filtered[-1][0] < duration_sec:
        filtered.append((duration_sec, filtered[-1][1]))
    else:
        filtered[-1] = (duration_sec, filtered[-1][1])

    return [TimedCenterPoint(t_sec=t_sec, x_norm=x_norm) for t_sec, x_norm in filtered]


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


def _infer_person_tracking(
    scene: Scene,
    *,
    source_video: Path,
    tracking_dir: Path,
    model_name: str,
    midpoint_layout: LayoutKind,
    midpoint_center_x: float | None,
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
        center_x = midpoint_center_x
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
            center_x = _tracking_sample_center_x(
                data,
                image_size=image_size,
                midpoint_layout=midpoint_layout,
            )
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


def _call_gemini_vision(keyframe_path: str, model_name: str) -> dict[str, Any]:
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
                types.Part.from_text(text=GEMINI_LAYOUT_VISION_PROMPT),
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
            {"role": "system", "content": GEMINI_LAYOUT_VISION_PROMPT},
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


def infer_layout_instructions(
    scenes: list[Scene],
    *,
    gemini_vision_model: str,
    source_video: Path,
    tracking_dir: Path,
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
            if instr.layout in (LayoutKind.SIT_CENTER, LayoutKind.ZOOM_CALL_CENTER):
                tracking_points, tracking_samples = _infer_person_tracking(
                    s,
                    source_video=source_video,
                    tracking_dir=tracking_dir,
                    model_name=model_name,
                    midpoint_layout=instr.layout,
                    midpoint_center_x=instr.person_x_norm,
                    initial_data=data,
                    initial_image_size=image_size,
                )
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
    transcript_fp: str,
    clips_path: Path,
    config: PipelineConfig,
) -> dict[str, LayoutInstruction]:
    """Load cache or call Gemini vision for each keyframe; persist JSON artifacts."""
    clips_fp = _clips_fingerprint(clips_path)
    vm = resolved_vision_model(config)

    if (
        not config.force_layout_vision
        and layout_cache_valid(work_dir, transcript_fp=transcript_fp, clips_fp=clips_fp, vision_model=vm)
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
    )
    return instructions
