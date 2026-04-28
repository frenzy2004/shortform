"""
Step 2 - Clip Selection: Gemini-only LLM for viral clip identification.

Uses the unified ``google-genai`` SDK (``from google import genai``). See:
https://github.com/googleapis/python-genai
"""

from __future__ import annotations

import json
import logging
import re
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
    TEXT_AXIS_WEIGHTS,
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
from humeo.hook_library import (
    format_hook_examples,
    retrieve_hook_examples,
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
_TITLE_SMALL_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "vs",
    "with",
}
_TITLE_DROP_WORDS = {
    "actually",
    "entirely",
    "just",
    "next",
    "really",
    "still",
    "that",
    "their",
    "these",
    "this",
    "those",
    "very",
    "will",
    "your",
}
_TITLE_BLAND_WORDS = {
    "big",
    "future",
    "important",
    "lesson",
    "matter",
    "matters",
    "opportunity",
    "reason",
    "soon",
    "story",
    "thing",
}
_GENERIC_TITLE_PATTERNS = (
    "big opportunity",
    "future of",
    "important lesson",
    "start a business with ai",
    "why this matters",
    "what this means",
)
_TITLE_TOKEN_REPLACEMENTS = {
    "ai": "AI",
    "agi": "AGI",
    "api": "API",
    "btc": "BTC",
    "ev": "EV",
    "evs": "EVs",
    "us": "US",
}
_POWER_TITLE_TOKENS = {"$", "%", "under", "beats", "fewer", "more", "less", "vs"}
_FILLER_OPENERS = {
    "actually",
    "basically",
    "i",
    "kind",
    "look",
    "listen",
    "now",
    "okay",
    "ok",
    "right",
    "so",
    "sort",
    "well",
    "yeah",
    "you",
}
_FILLER_OPENING_PHRASES = {
    "i mean",
    "kind of",
    "sort of",
    "you know",
}
_PREFERRED_MAX_DURATION_SEC = 72.0


def _has_valid_duration(clip: Clip) -> bool:
    """Return True when the clip window satisfies the product duration contract."""
    return MIN_CLIP_DURATION_SEC <= clip.duration_sec <= MAX_CLIP_DURATION_SEC


def _text_composite_score(clip: Clip) -> float:
    """Weighted composite from the text-axis breakdown, falling back to virality_score.

    Cache compatibility note:
    - New Ticket 3 clips use the three-axis rubric (message_wow / hook_emotion / catchy).
    - Older caches may still contain legacy rule-name ``score_breakdown`` maps from the
      pre-Ticket-3 prompt. If none of the expected axes are present, fall back cleanly
      to ``virality_score`` instead of treating the legacy shape as three missing axes.
    """
    if not clip.score_breakdown:
        return clip.virality_score

    present_expected_axes = [axis for axis in TEXT_AXIS_WEIGHTS if axis in clip.score_breakdown]
    if not present_expected_axes:
        return clip.virality_score

    total = 0.0
    missing: list[str] = []
    for axis, weight in TEXT_AXIS_WEIGHTS.items():
        value = clip.score_breakdown.get(axis)
        if value is None:
            missing.append(axis)
            continue
        total += value * weight

    if missing:
        logger.warning(
            "Clip %s score_breakdown missing axis(es) %s; treating as 0.0.",
            clip.clip_id,
            ", ".join(missing),
        )
    return total


def _title_quality_penalty(clip: Clip) -> float:
    title = _tighten_overlay_title_text(clip.suggested_overlay_title or "")
    if not title:
        return 0.0
    penalty = 0.0
    if _looks_generic_title(title):
        penalty += 0.18
    tokens = [token for token in _normalized_title(title).split() if token]
    if len(tokens) < 2 or len(tokens) > 6:
        penalty += 0.05
    if not any(token in title.lower() for token in _POWER_TITLE_TOKENS) and not any(
        ch.isdigit() for ch in title
    ):
        penalty += 0.03
    return min(0.22, penalty)


def _hook_quality_penalty(clip: Clip) -> float:
    penalty = 0.0
    if clip.hook_start_sec is not None and clip.hook_start_sec > 5.0:
        penalty += min(0.18, 0.06 + (clip.hook_start_sec - 5.0) * 0.025)
    opener = " ".join((clip.viral_hook or clip.transcript or "").split()).lower()
    if opener:
        first_words = opener.split()
        first_word = first_words[0] if first_words else ""
        opening_phrase = " ".join(first_words[:2])
        if first_word in _FILLER_OPENERS:
            penalty += 0.14
        if opening_phrase in _FILLER_OPENING_PHRASES:
            penalty += 0.06
        if len(first_words) >= 12:
            penalty += 0.03
    return min(0.24, penalty)


def _duration_quality_penalty(clip: Clip) -> float:
    if clip.duration_sec <= _PREFERRED_MAX_DURATION_SEC:
        return 0.0
    drift = clip.duration_sec - _PREFERRED_MAX_DURATION_SEC
    return min(0.14, 0.03 + drift * 0.01)


def clip_quality_penalty(clip: Clip) -> float:
    return min(
        0.42,
        _title_quality_penalty(clip)
        + _hook_quality_penalty(clip)
        + _duration_quality_penalty(clip),
    )


def clip_quality_priority_score(clip: Clip) -> float:
    review_penalty = 0.5 if clip.needs_review else 0.0
    composite = _text_composite_score(clip)
    return composite - review_penalty - clip_quality_penalty(clip)


def renumber_clips_dense(clips: list[Clip]) -> list[Clip]:
    renumbered: list[Clip] = []
    for idx, clip in enumerate(clips, start=1):
        new_id = f"{idx:03d}"
        renumbered.append(clip if clip.clip_id == new_id else clip.model_copy(update={"clip_id": new_id}))
    return renumbered


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


def _headline_case_title(text: str) -> str:
    words = text.split()
    if not words:
        return ""
    out: list[str] = []
    for idx, word in enumerate(words):
        if any(ch.isdigit() for ch in word) or word.startswith("$"):
            out.append(word)
            continue
        raw = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", word)
        lower = raw.lower()
        if lower in _TITLE_TOKEN_REPLACEMENTS:
            out.append(word.replace(raw, _TITLE_TOKEN_REPLACEMENTS[lower]))
            continue
        if idx not in (0, len(words) - 1) and lower in _TITLE_SMALL_WORDS:
            out.append(word.replace(raw, lower))
            continue
        out.append(word.replace(raw, raw.capitalize()))
    return " ".join(out)


def _normalized_title(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9$% ]+", " ", (text or "").lower())).strip()


def _looks_generic_title(text: str) -> bool:
    normalized = _normalized_title(text)
    if not normalized:
        return True
    if any(pattern in normalized for pattern in _GENERIC_TITLE_PATTERNS):
        return True
    tokens = [token for token in normalized.split() if token]
    bland_count = sum(token in _TITLE_BLAND_WORDS for token in tokens)
    return bland_count >= 2


def _tighten_overlay_title_text(text: str) -> str:
    title = " ".join((text or "").replace("—", "-").split()).strip(" .,!?:;-")
    if not title:
        return ""
    title = re.sub(r"\bwill cost less than\b", "under", title, flags=re.IGNORECASE)
    title = re.sub(r"\bless than\b", "under", title, flags=re.IGNORECASE)
    title = re.sub(r"\bmade your\b", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\bis still\b", "is", title, flags=re.IGNORECASE)
    title = re.sub(r"\bis creating\b", "creates", title, flags=re.IGNORECASE)
    title = re.sub(r"\bthere are\b", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\bentirely\b", "", title, flags=re.IGNORECASE)
    words = title.split()
    while len(words) > 6:
        filtered = [word for word in words if word.lower() not in _TITLE_DROP_WORDS]
        if len(filtered) == len(words):
            break
        words = filtered
    if len(words) > 4:
        words = [word for word in words if word.lower() not in {"your", "next"} or len(words) <= 4]
    if len(words) > 6 and words[0].lower() in {"why", "how", "when"}:
        words = words[1:]
    if len(words) > 6:
        words = words[:6]
    return _headline_case_title(" ".join(words).strip(" .,!?:;-"))


def _polish_overlay_title(clip: Clip) -> str:
    current = _tighten_overlay_title_text(clip.suggested_overlay_title or "")
    if current and not _looks_generic_title(current):
        return current
    for candidate in (clip.viral_hook or "", clip.topic or ""):
        polished = _tighten_overlay_title_text(candidate)
        if polished and not _looks_generic_title(polished):
            return polished
    return current


def _polish_clip_metadata(clip: Clip) -> Clip:
    title = _polish_overlay_title(clip)
    if not title or title == clip.suggested_overlay_title:
        return clip
    return clip.model_copy(update={"suggested_overlay_title": title})


def build_prompt(
    transcript: dict,
    *,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    steering_notes: list[str] | None = None,
    hook_library_path: Path | None = None,
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

    hook_examples = format_hook_examples(
        retrieve_hook_examples(
            transcript_text[:8000],
            path=hook_library_path,
            limit=8,
        )
    )

    system, user = clip_selection_prompts(
        transcript_text=transcript_text,
        min_dur=MIN_CLIP_DURATION_SEC,
        max_dur=MAX_CLIP_DURATION_SEC,
        count=candidate_count,
        steering_notes=steering_notes,
        hook_examples=hook_examples,
    )
    return system, user


def rank_and_filter_clips(
    clips: list[Clip],
    *,
    threshold: float = DEFAULT_QUALITY_THRESHOLD,
    min_kept: int = DEFAULT_MIN_KEPT,
    max_kept: int = DEFAULT_MAX_KEPT,
) -> list[Clip]:
    """Rank ``clips`` by text composite (or legacy ``virality_score``) and apply
    the threshold+floor+cap.

    Rules (in order, with clear precedence):

    1. Sort descending by the text composite score when the Ticket 3
       three-axis ``score_breakdown`` is present; otherwise fall back to the
       legacy ``virality_score``.
    2. Keep clips whose active score signal is ``>= threshold`` (or
       ``needs_review`` cleared). Reviewed-out clips (``needs_review=True``)
       are always sent to the back of the priority queue.
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

    score_signal = {id(c): _text_composite_score(c) for c in clips}
    priority_signal = {id(c): clip_quality_priority_score(c) for c in clips}

    def _priority(c: Clip) -> tuple[float, float]:
        return (priority_signal[id(c)], score_signal[id(c)])

    valid: list[Clip] = []
    invalid: list[Clip] = []
    for clip in clips:
        if _has_valid_duration(clip):
            valid.append(clip)
        else:
            invalid.append(clip)
            logger.warning(
                "Clip %s dropped before ranking: duration %.1fs is outside [%ds, %ds] - %s",
                clip.clip_id,
                clip.duration_sec,
                MIN_CLIP_DURATION_SEC,
                MAX_CLIP_DURATION_SEC,
                clip.topic,
            )

    if not valid:
        logger.warning(
            "Clip ranking: 0 valid candidates remain after duration filtering (dropped=%d).",
            len(invalid),
        )
        return []

    ordered = sorted(valid, key=_priority, reverse=True)

    strong = [c for c in ordered if priority_signal[id(c)] >= threshold and not c.needs_review]
    kept = list(strong)

    if len(kept) < min_kept:
        backfill = [c for c in ordered if c not in kept]
        for c in backfill:
            if len(kept) >= min_kept:
                break
            kept.append(c)

    if len(kept) < min_kept:
        logger.warning(
            "Clip ranking: only %d valid candidates remain after duration filtering; "
            "cannot satisfy min_kept=%d without invalid clips.",
            len(kept),
            min_kept,
        )

    if len(kept) > max_kept:
        kept = kept[:max_kept]

    # Renumber clip_ids so consumers (filenames, layout vision, subtitles)
    # always see 001..NNN in rank order regardless of what the LLM returned.
    renumbered = renumber_clips_dense(kept)

    dropped = len(valid) - len(kept) + len(invalid)
    logger.info(
        "Clip ranking: kept %d / %d candidates (threshold=%.2f, min=%d, max=%d, dropped=%d).",
        len(renumbered),
        len(clips),
        threshold,
        min_kept,
        max_kept,
        dropped,
    )
    for c in renumbered:
        logger.info(
            "  [%s] score=%.2f priority=%.2f penalty=%.2f %s %s",
            c.clip_id,
            c.virality_score,
            clip_quality_priority_score(c),
            clip_quality_penalty(c),
            "(review)" if c.needs_review else "",
            c.topic,
        )
    return renumbered


def select_clips(
    transcript: dict,
    *,
    gemini_model: str | None = None,
    hook_library_path: Path | None = None,
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
        hook_library_path=hook_library_path,
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
        clip = _polish_clip_metadata(Clip.model_validate(payload))

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
