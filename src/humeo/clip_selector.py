"""
Step 2 - Clip Selection: Gemini-only LLM for viral clip identification.

Uses the unified ``google-genai`` SDK (``from google import genai``). See:
https://github.com/googleapis/python-genai
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, TypeVar

from google import genai
from openai import OpenAI

from humeo.gemini_generate import gemini_generate_config

from humeo_core.schemas import Clip, ClipPlan

from humeo.config import (
    GEMINI_MODEL,
    MAX_CLIP_DURATION_SEC,
    MIN_CLIP_DURATION_SEC,
    TARGET_CLIP_COUNT,
)
from humeo.env import (
    OPENROUTER_BASE_URL,
    model_name_for_provider,
    openrouter_default_headers,
    resolve_gemini_api_key,
    resolve_llm_provider,
    resolve_openrouter_api_key,
)
from humeo.prompt_loader import clip_selection_prompts

logger = logging.getLogger(__name__)

T = TypeVar("T")

LLM_MAX_ATTEMPTS = 3
LLM_RETRY_DELAY_SEC = 2.0

# Over-generation defaults (also exposed via PipelineConfig so callers can
# override per-run without touching code). Rationale:
#
# - Ask Gemini for a *pool* of ~12 candidates at temperature 0.7 so the model
#   considers a wider slice of the transcript instead of locking onto the
#   first 5 obvious ones. More candidates -> more chance the actual gold
#   nugget is in the list.
# - Then rank by ``virality_score`` and keep everything >= threshold, but
#   always keep at least ``min_kept`` and at most ``max_kept`` clips. This
#   lets a single strong clip survive a weak transcript ("keep the best 5
#   even if no one clears the bar") AND lets an exceptionally rich
#   transcript ship 7-8 strong shorts instead of artificially capping at 5.
DEFAULT_CANDIDATE_COUNT = 12
DEFAULT_QUALITY_THRESHOLD = 0.70
DEFAULT_MIN_KEPT = TARGET_CLIP_COUNT
DEFAULT_MAX_KEPT = 8
# Higher than the old 0.3 so the pool is meaningfully different from
# "the same five most-obvious clips every run". Still well below 1.0 so we
# do not get word-salad IDs or timestamps.
DEFAULT_CANDIDATE_TEMPERATURE = 0.7


def _openai_message_text(content: object) -> str:
    """Normalize OpenAI-compatible message content into plain text."""
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


def _retry_llm(name: str, fn: Callable[[], T], attempts: int = LLM_MAX_ATTEMPTS) -> T:
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            if i < attempts - 1:
                logger.warning("%s attempt %d/%d failed: %s", name, i + 1, attempts, e)
                time.sleep(LLM_RETRY_DELAY_SEC * (i + 1))
    assert last is not None
    raise last


def build_prompt(
    transcript: dict,
    *,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    steering_notes: list[str] | None = None,
) -> tuple[str, str]:
    """Return ``(system_prompt, user_message)`` for the clip-selector LLM call.

    ``candidate_count`` is the size of the candidate POOL we ask Gemini for.
    A downstream ranker (``rank_and_filter_clips``) then keeps the top
    clips that clear the quality threshold. Defaults preserve the previous
    visible output (5 clips) when the pool is narrow.
    """
    lines = []
    for seg in transcript.get("segments", []):
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "").strip()
        lines.append(f"[{start:.1f}s - {end:.1f}s] {text}")

    transcript_text = "\n".join(lines)

    system, user = clip_selection_prompts(
        transcript_text=transcript_text,
        min_dur=MIN_CLIP_DURATION_SEC,
        max_dur=MAX_CLIP_DURATION_SEC,
        count=candidate_count,
        steering_notes=steering_notes,
    )
    return system, user


def rank_and_filter_clips(
    clips: list[Clip],
    *,
    threshold: float = DEFAULT_QUALITY_THRESHOLD,
    min_kept: int = DEFAULT_MIN_KEPT,
    max_kept: int = DEFAULT_MAX_KEPT,
) -> list[Clip]:
    """Rank ``clips`` by ``virality_score`` and apply the threshold+floor+cap.

    Rules (in order, with clear precedence):

    1. Sort descending by ``virality_score``.
    2. Keep clips with ``virality_score >= threshold`` (or ``needs_review``
       cleared). Reviewed-out clips (``needs_review=True``) are always sent
       to the back of the priority queue.
    3. If fewer than ``min_kept`` clips passed the threshold, fill up from
       the remaining clips in rank order until we reach ``min_kept`` (or
       run out of candidates).
    4. Cap the final list at ``max_kept`` entries.
    5. Renumber ``clip_id`` to ``001``, ``002``, ... so downstream artifacts
       (keyframes, subtitles, output filenames) stay dense and ordered.

    This is the "threshold with a floor" policy the user asked for: quality
    first, but never ship zero shorts when the transcript is weak.
    """
    if not clips:
        return []

    def _priority(c: Clip) -> tuple[float, float]:
        # needs_review clips fall behind same-score non-reviewed ones.
        review_penalty = 0.5 if c.needs_review else 0.0
        return (c.virality_score - review_penalty, c.virality_score)

    ordered = sorted(clips, key=_priority, reverse=True)

    strong = [c for c in ordered if c.virality_score >= threshold and not c.needs_review]
    kept = list(strong)

    if len(kept) < min_kept:
        backfill = [c for c in ordered if c not in kept]
        for c in backfill:
            if len(kept) >= min_kept:
                break
            kept.append(c)

    if len(kept) > max_kept:
        kept = kept[:max_kept]

    # Renumber clip_ids so consumers (filenames, layout vision, subtitles)
    # always see 001..NNN in rank order regardless of what the LLM returned.
    renumbered: list[Clip] = []
    for i, c in enumerate(kept, start=1):
        new_id = f"{i:03d}"
        renumbered.append(c if c.clip_id == new_id else c.model_copy(update={"clip_id": new_id}))

    dropped = len(ordered) - len(kept)
    logger.info(
        "Clip ranking: kept %d / %d candidates (threshold=%.2f, min=%d, max=%d, dropped=%d).",
        len(renumbered),
        len(ordered),
        threshold,
        min_kept,
        max_kept,
        dropped,
    )
    for c in renumbered:
        logger.info(
            "  [%s] score=%.2f %s %s",
            c.clip_id,
            c.virality_score,
            "(review)" if c.needs_review else "",
            c.topic,
        )
    return renumbered


def select_clips(
    transcript: dict,
    *,
    gemini_model: str | None = None,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    min_kept: int = DEFAULT_MIN_KEPT,
    max_kept: int = DEFAULT_MAX_KEPT,
    temperature: float = DEFAULT_CANDIDATE_TEMPERATURE,
    steering_notes: list[str] | None = None,
) -> tuple[list[Clip], str]:
    """
    Call Gemini to select clips. Returns ``(clips, raw_json)`` for caching / debugging.

    The returned clip list has already been ranked + filtered by
    :func:`rank_and_filter_clips`. ``raw_json`` is the untouched LLM
    response so the cache artifact reflects the entire candidate pool for
    audit / re-ranking without another LLM call.

    Uses ``google.genai.Client`` and ``GenerateContentConfig`` (see Google
    Gen AI SDK for Python).
    """
    provider = resolve_llm_provider()
    model_name = model_name_for_provider((gemini_model or GEMINI_MODEL).strip(), provider)
    system_prompt, user_text = build_prompt(
        transcript,
        candidate_count=candidate_count,
        steering_notes=steering_notes,
    )

    def _call() -> str:
        logger.info(
            "%s clip selection (model=%s, candidate_pool=%d, temp=%.2f)...",
            provider,
            model_name,
            candidate_count,
            temperature,
        )
        if provider == "google":
            client = genai.Client(api_key=resolve_gemini_api_key())
            response = client.models.generate_content(
                model=model_name,
                contents=user_text,
                config=gemini_generate_config(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    response_mime_type="application/json",
                ),
            )
            if not response.text:
                raise RuntimeError("Gemini returned empty response text")
            return response.text

        client = OpenAI(
            api_key=resolve_openrouter_api_key(),
            base_url=OPENROUTER_BASE_URL,
            default_headers=openrouter_default_headers(),
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        text = _openai_message_text(response.choices[0].message.content)
        if not text:
            raise RuntimeError("OpenRouter returned empty response text")
        return text

    raw = _retry_llm("Gemini clip selection", _call)
    candidates = _parse_clips(raw)
    # The ranker can only backfill from the pool Gemini returned. If Gemini
    # under-delivered (e.g. returned 2 of a requested 12), the min_kept floor
    # is unenforceable -- warn loudly so we do not silently ship fewer shorts
    # than the caller expected.
    if len(candidates) < min_kept:
        logger.warning(
            "Clip selection: Gemini returned only %d candidates (requested %d, floor %d). "
            "Output will be capped at %d shorts -- check prompt or transcript length.",
            len(candidates),
            candidate_count,
            min_kept,
            len(candidates),
        )
    elif len(candidates) < candidate_count:
        logger.info(
            "Clip selection: Gemini returned %d of %d requested candidates "
            "(pool still >= floor of %d).",
            len(candidates),
            candidate_count,
            min_kept,
        )
    clips = rank_and_filter_clips(
        candidates,
        threshold=quality_threshold,
        min_kept=min_kept,
        max_kept=max_kept,
    )
    return clips, raw


def _parse_clips(raw_json: str) -> list[Clip]:
    """Parse and validate the LLM's JSON response into Clip objects."""
    data = json.loads(raw_json)
    clips_data = data.get("clips", data) if isinstance(data, dict) else data

    clips: list[Clip] = []
    for item in clips_data:
        payload = dict(item)
        payload.pop("duration_sec", None)
        clip = Clip.model_validate(payload)

        actual_dur = clip.end_time_sec - clip.start_time_sec
        stated_dur = item.get("duration_sec")
        if stated_dur is not None and abs(actual_dur - float(stated_dur)) > 1.0:
            logger.warning(
                "Clip %s: stated duration %.1fs doesn't match (%.1f-%.1f = %.1f).",
                clip.clip_id, float(stated_dur),
                clip.start_time_sec, clip.end_time_sec, actual_dur,
            )
        clips.append(clip)

    logger.info("Parsed %d clips from LLM response", len(clips))
    return clips


def save_clips(clips: list[Clip], output_path: Path) -> Path:
    """Persist clips to a JSON file using the shared Pydantic schema."""
    plan = ClipPlan(source_path="", clips=list(clips))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(plan.model_dump_json(indent=2))
    logger.info("Saved %d clips to %s", len(clips), output_path)
    return output_path


def load_clips(clips_path: Path) -> list[Clip]:
    """Load clips from a previously saved JSON file."""
    with open(clips_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "clips" in data:
        return [Clip.model_validate(c) for c in data["clips"]]
    return [Clip.model_validate(c) for c in data]
