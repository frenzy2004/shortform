"""Per-clip layout + bbox via Gemini vision (no pixel heuristics in the product pipeline)."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
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

LAYOUT_VISION_META = "layout_vision.meta.json"
LAYOUT_VISION_JSON = "layout_vision.json"

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


def _parse_bbox(raw: object) -> BoundingBox | None:
    if not raw or not isinstance(raw, dict):
        return None
    try:
        return BoundingBox.model_validate(raw)
    except Exception:
        return None


def _instruction_from_gemini_json(
    scene_id: str, data: dict[str, Any]
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

    pb = _parse_bbox(data.get("person_bbox"))
    fb = _parse_bbox(data.get("face_bbox"))
    cb = _parse_bbox(data.get("chart_bbox"))
    p2 = _parse_bbox(data.get("second_person_bbox"))
    f2 = _parse_bbox(data.get("second_face_bbox"))
    c2 = _parse_bbox(data.get("second_chart_bbox"))
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
            raw_by_clip[sid] = data
            out[sid] = _instruction_from_gemini_json(sid, data)
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

    instructions, raw_by_clip = infer_layout_instructions(scenes, gemini_vision_model=vm)

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
