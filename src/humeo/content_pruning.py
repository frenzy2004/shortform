"""Stage 2.5 - Content pruning inside each selected clip.

This is the HIVE "irrelevant content pruning" sub-task, applied at the
*inner-clip* scale rather than the scene scale. After the clip selector has
chosen 5 x 50-90s windows, we ask Gemini to tighten each window by dropping
weak lead-in (throat-clears, false starts, slow setup) and weak tail content
(trailing ramble, fade-out talk).

Design choices kept deliberately minimal:

- **No schema changes.** The existing ``Clip.trim_start_sec`` /
  ``Clip.trim_end_sec`` fields already feed ``humeo.render_window`` and
  ``humeo_core.primitives.compile`` via ``-ss`` / ``-t``. Writing the pruned
  in / out points into those fields tightens the exported window for free.
- **Contiguous trimming only** (V1). We move the in-point forward and the
  out-point backward; we do not cut in the middle. That keeps subtitles and
  layout vision untouched.
- **Strict clamping** after the LLM returns, so the final duration always
  respects ``MIN_CLIP_DURATION_SEC`` and any declared hook window is
  preserved.
- **Never fatal.** Any failure (API error, malformed JSON, missing clip_id)
  degrades to no-op trims (0.0 / 0.0) for that clip. The pipeline still
  produces output identical to the pre-Stage-2.5 behaviour.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar

from google import genai
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from humeo_core.schemas import Clip

from humeo.config import (
    GEMINI_MODEL,
    MAX_CLIP_DURATION_SEC,
    MIN_CLIP_DURATION_SEC,
    PipelineConfig,
)
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
from humeo.prompt_loader import content_pruning_system_prompt

logger = logging.getLogger(__name__)

T = TypeVar("T")

PRUNE_META_VERSION = 1
PRUNE_META_FILENAME = "prune.meta.json"
PRUNE_RAW_FILENAME = "prune_raw.json"
PRUNE_ARTIFACT_FILENAME = "prune.json"

LLM_MAX_ATTEMPTS = 3
LLM_RETRY_DELAY_SEC = 2.0

PruneLevel = Literal["off", "conservative", "balanced", "aggressive"]

VALID_LEVELS: tuple[PruneLevel, ...] = ("off", "conservative", "balanced", "aggressive")


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

# The clip-selection prompt uses `[0.0, 3.0]` as an example / fallback hook
# window. Gemini frequently copies this placeholder verbatim instead of
# localising the real hook, which silently disables Stage 2.5 start-trims for
# every clip (the hook clamp below refuses to trim past `hook_start_sec`, so
# any `trim_start_sec > 0` returned by the prune LLM gets zeroed).
#
# Treat this exact fingerprint as "no real hook" for clamp purposes. The real
# fix is the Stage 2.25 hook detector (``humeo.hook_detector``) which
# overwrites the clip's hook fields with a localised window before pruning
# runs. This constant is the belt-and-suspenders guard for the case where
# hook detection is disabled, fails, or cache-hits stale data.
_DEFAULT_HOOK_FINGERPRINT: tuple[float, float] = (0.0, 3.0)
_DEFAULT_HOOK_EPS: float = 1e-3


def _looks_like_default_hook(hook_start: float | None, hook_end: float | None) -> bool:
    """True when the hook window matches the prompt's 0-3s placeholder.

    This is intentionally a narrow, exact-match check so a real hook that
    happens to open at t=0 with a 3.0s window is still respected.
    """
    if hook_start is None or hook_end is None:
        return False
    return (
        abs(hook_start - _DEFAULT_HOOK_FINGERPRINT[0]) < _DEFAULT_HOOK_EPS
        and abs(hook_end - _DEFAULT_HOOK_FINGERPRINT[1]) < _DEFAULT_HOOK_EPS
    )

# Per-level cap on the fraction of the original clip the LLM is allowed to
# trim. Even if the LLM tries to be more eager, we clamp. Final duration is
# additionally clamped to ``MIN_CLIP_DURATION_SEC``.
_MAX_TOTAL_TRIM_PCT: dict[PruneLevel, float] = {
    "off": 0.0,
    "conservative": 0.10,
    "balanced": 0.20,
    "aggressive": 0.35,
}


class _PruneDecision(BaseModel):
    """Per-clip decision returned by Gemini (clip-relative seconds)."""

    clip_id: str
    trim_start_sec: float = Field(default=0.0, ge=0.0)
    trim_end_sec: float = Field(default=0.0, ge=0.0)
    reason: str = ""


class _PruneResponse(BaseModel):
    decisions: list[_PruneDecision] = Field(default_factory=list)


@dataclass
class _ClampStats:
    """Diagnostics for why a returned trim got reshaped."""

    clamped_start: bool = False
    clamped_end: bool = False
    hook_protected: bool = False
    min_duration_protected: bool = False
    max_pct_protected: bool = False


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


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------


def _clamp_decision(
    clip: Clip,
    trim_start: float,
    trim_end: float,
    *,
    level: PruneLevel,
) -> tuple[float, float, _ClampStats]:
    """Clamp a raw (trim_start, trim_end) pair so the resulting clip is legal.

    Guarantees:
    - ``trim_start`` and ``trim_end`` are non-negative.
    - Final duration (``clip.duration_sec - trim_start - trim_end``) is at
      least ``MIN_CLIP_DURATION_SEC`` (or the original duration, whichever is
      smaller - we never *extend* a clip that was already too short).
    - Combined trim does not exceed the level's allowed fraction of the
      original duration.
    - If ``hook_start_sec`` / ``hook_end_sec`` are set on the clip, the hook
      window stays fully inside the result.
    """
    stats = _ClampStats()
    duration = clip.duration_sec

    ts = max(0.0, float(trim_start))
    te = max(0.0, float(trim_end))
    if ts != trim_start:
        stats.clamped_start = True
    if te != trim_end:
        stats.clamped_end = True

    max_pct = _MAX_TOTAL_TRIM_PCT.get(level, 0.0)
    max_total_trim = duration * max_pct
    if ts + te > max_total_trim:
        scale = max_total_trim / max(ts + te, 1e-9)
        ts = ts * scale
        te = te * scale
        stats.max_pct_protected = True

    # Only protect the hook when the clip carries a *real* localised hook
    # window. The clip-selection LLM frequently echoes the prompt's
    # 0.0-3.0s placeholder, which would otherwise lock ``trim_start`` to 0
    # for every clip and silently disable the entire pruning stage. See
    # ``_looks_like_default_hook`` for the fingerprint rationale.
    hook_is_real = (
        clip.hook_start_sec is not None
        and clip.hook_end_sec is not None
        and not _looks_like_default_hook(clip.hook_start_sec, clip.hook_end_sec)
    )
    if hook_is_real:
        hook_lo = clip.hook_start_sec  # type: ignore[assignment]
        hook_hi = clip.hook_end_sec  # type: ignore[assignment]
        if ts > max(0.0, hook_lo - 0.25):
            ts = max(0.0, hook_lo - 0.25)
            stats.hook_protected = True
        if te > max(0.0, duration - hook_hi - 0.25):
            te = max(0.0, duration - hook_hi - 0.25)
            stats.hook_protected = True

    min_final = min(float(MIN_CLIP_DURATION_SEC), duration)
    max_total_by_min = max(0.0, duration - min_final)
    if ts + te > max_total_by_min:
        overflow = ts + te - max_total_by_min
        te_cut = min(te, overflow)
        te -= te_cut
        overflow -= te_cut
        if overflow > 0:
            ts = max(0.0, ts - overflow)
        stats.min_duration_protected = True

    ts = max(0.0, min(ts, duration))
    te = max(0.0, min(te, duration - ts))
    return ts, te, stats


# Tolerance used when snapping trim boundaries to WhisperX segment edges. A
# 3s window comfortably covers "finish the current sentence" cases without
# materially deviating from what the LLM asked for. Tuned on the reported
# mid-sentence cut in clip 001 of the ``PdVv_vLkUgk`` run (6.38s trim vs a
# sentence that ended ~1.5s later).
_SEGMENT_SNAP_TOLERANCE_SEC: float = 3.0
_BOUNDARY_GAP_SEC: float = 0.5
_BOUNDARY_TIME_EPS_SEC: float = 0.12
_START_BOUNDARY_WINDOW_SEC: float = 3.0
_END_BOUNDARY_WINDOW_SEC: float = 2.0
_TERMINAL_PUNCT: tuple[str, ...] = (".", "?", "!")
_WEAK_START_WORDS: frozenset[str] = frozenset({"and", "but", "so", "or", "then", "because"})


@dataclass(frozen=True)
class _BoundaryCandidate:
    """A possible snapped boundary on the source timeline."""

    time_sec: float
    clean: bool
    reason: str
    source: str
    weak_start: bool = False


def _snap_trims_to_segment_boundaries(
    clip: Clip,
    transcript: dict,
    *,
    level: PruneLevel,
    tolerance_sec: float = _SEGMENT_SNAP_TOLERANCE_SEC,
) -> tuple[float, float]:
    """Snap an already-clamped ``(trim_start, trim_end)`` to phrase boundaries.

    WhisperX segments correspond to natural phrase / sentence groupings.
    Landing cuts on segment edges eliminates the "this could be..." class of
    mid-sentence truncation, even when the LLM rounds to an arbitrary
    syllable.

    Direction preference:

    - ``trim_start``: prefer the nearest segment START at-or-after the
      current in-point (trim a hair more to drop lead-in filler). Fallback
      is the nearest segment start behind, within tolerance.
    - ``trim_end``: prefer the nearest segment END at-or-after the current
      out-point (let the sentence finish, keeping MORE content). Fallback
      is the nearest segment end before, within tolerance.

    Safety: the snapped pair is reverted if it would violate
    ``MIN_CLIP_DURATION_SEC``, exceed the level's ``max_pct`` trim cap, or
    eat into a real (non-placeholder) hook window. Snapping can only
    *improve* a decision, never break it.
    """
    ts0 = float(clip.trim_start_sec)
    te0 = float(clip.trim_end_sec)
    if ts0 < 0.05 and te0 < 0.05:
        return ts0, te0

    segs = _segments_within_clip(transcript, clip)
    if not segs:
        return ts0, te0

    duration = clip.duration_sec
    seg_starts = [float(s["start"]) for s in segs]
    seg_ends = [float(s["end"]) for s in segs]

    new_ts = ts0
    if ts0 >= 0.05:
        forward = [s for s in seg_starts if s >= ts0 and (s - ts0) <= tolerance_sec]
        backward = [s for s in seg_starts if s < ts0 and (ts0 - s) <= tolerance_sec]
        if forward:
            new_ts = min(forward)
        elif backward:
            new_ts = max(backward)

    new_te = te0
    if te0 >= 0.05:
        out0 = duration - te0
        forward = [e for e in seg_ends if e >= out0 and (e - out0) <= tolerance_sec]
        backward = [e for e in seg_ends if e < out0 and (out0 - e) <= tolerance_sec]
        if forward:
            new_out = min(forward)
        elif backward:
            new_out = max(backward)
        else:
            new_out = out0
        new_te = max(0.0, duration - new_out)

    new_ts = max(0.0, min(new_ts, duration))
    new_te = max(0.0, min(new_te, duration - new_ts))

    min_final = min(float(MIN_CLIP_DURATION_SEC), duration)
    if duration - new_ts - new_te < min_final - 1e-6:
        return ts0, te0

    max_pct = _MAX_TOTAL_TRIM_PCT.get(level, 0.0)
    if max_pct > 0.0 and (new_ts + new_te) > duration * max_pct + 1e-6:
        return ts0, te0

    if (
        clip.hook_start_sec is not None
        and clip.hook_end_sec is not None
        and not _looks_like_default_hook(clip.hook_start_sec, clip.hook_end_sec)
    ):
        hook_lo = float(clip.hook_start_sec)
        hook_hi = float(clip.hook_end_sec)
        if new_ts > max(0.0, hook_lo - 0.25) + 1e-6:
            return ts0, te0
        if duration - new_te < hook_hi + 0.25 - 1e-6:
            return ts0, te0

    return new_ts, new_te


def _flatten_transcript_words(transcript: dict) -> list[dict[str, float | str]]:
    words: list[dict[str, float | str]] = []
    for seg in transcript.get("segments", []):
        for word in seg.get("words", []):
            if "start" not in word or "end" not in word:
                continue
            try:
                start = float(word["start"])
                end = float(word["end"])
            except (TypeError, ValueError):
                continue
            words.append(
                {
                    "word": str(word.get("word", "")),
                    "start": start,
                    "end": end,
                }
            )
    return words


def _normalized_last_char(text: str) -> str:
    stripped = text.rstrip()
    return stripped[-1] if stripped else ""


def _segment_start_hint(
    segments: list[dict[str, Any]],
    words: list[dict[str, float | str]],
    time_sec: float,
) -> tuple[bool, str, bool]:
    for idx, seg in enumerate(segments):
        seg_start = float(seg.get("start", 0.0))
        if abs(seg_start - time_sec) > _BOUNDARY_TIME_EPS_SEC:
            continue
        seg_words = seg.get("words") or []
        first_word = ""
        if seg_words:
            first_word = str(seg_words[0].get("word", "")).strip().lower()
        weak_start = first_word in _WEAK_START_WORDS
        if idx == 0:
            return True, "first transcript segment", weak_start
        prev_text = str(segments[idx - 1].get("text", "")).rstrip()
        if _normalized_last_char(prev_text) in _TERMINAL_PUNCT:
            return True, "previous segment ends with terminal punctuation", weak_start
        break

    for idx, word in enumerate(words):
        start = float(word["start"])
        if abs(start - time_sec) > _BOUNDARY_TIME_EPS_SEC:
            continue
        weak_start = str(word["word"]).strip().lower() in _WEAK_START_WORDS
        if idx == 0:
            return True, "first transcript word", weak_start
        gap_before = start - float(words[idx - 1]["end"])
        if gap_before >= _BOUNDARY_GAP_SEC:
            return True, f"silence gap before boundary ({gap_before:.2f}s)", weak_start
        return False, "no terminal punctuation or >=0.5s silence before boundary", weak_start

    return False, "no matching transcript boundary", False


def _segment_end_hint(
    segments: list[dict[str, Any]],
    words: list[dict[str, float | str]],
    time_sec: float,
) -> tuple[bool, str]:
    for seg in segments:
        seg_end = float(seg.get("end", 0.0))
        if abs(seg_end - time_sec) > _BOUNDARY_TIME_EPS_SEC:
            continue
        text = str(seg.get("text", "")).rstrip()
        if _normalized_last_char(text) in _TERMINAL_PUNCT:
            return True, "segment ends with terminal punctuation"
        break

    for idx, word in enumerate(words):
        end = float(word["end"])
        if abs(end - time_sec) > _BOUNDARY_TIME_EPS_SEC:
            continue
        if idx == len(words) - 1:
            return True, "last transcript word"
        gap_after = float(words[idx + 1]["start"]) - end
        if gap_after >= _BOUNDARY_GAP_SEC:
            return True, f"silence gap after boundary ({gap_after:.2f}s)"
        return False, "no terminal punctuation or >=0.5s silence after boundary"

    return False, "no matching transcript boundary"


def _candidate_key(time_sec: float) -> float:
    return round(time_sec, 3)


def _gather_start_candidates(
    clip: Clip,
    current_start: float,
    transcript: dict,
) -> list[_BoundaryCandidate]:
    low = current_start - _START_BOUNDARY_WINDOW_SEC
    high = current_start + _START_BOUNDARY_WINDOW_SEC
    segments = list(transcript.get("segments", []))
    words = _flatten_transcript_words(transcript)

    by_time: dict[float, _BoundaryCandidate] = {}

    def add_candidate(time_sec: float, source: str) -> None:
        clean, reason, weak = _segment_start_hint(segments, words, time_sec)
        candidate = _BoundaryCandidate(
            time_sec=float(time_sec),
            clean=clean,
            reason=reason,
            source=source,
            weak_start=weak,
        )
        key = _candidate_key(candidate.time_sec)
        existing = by_time.get(key)
        if existing is None:
            by_time[key] = candidate
            return
        if candidate.clean and not existing.clean:
            by_time[key] = candidate
            return
        if candidate.clean == existing.clean and not candidate.weak_start and existing.weak_start:
            by_time[key] = candidate

    add_candidate(current_start, "current")
    add_candidate(clip.start_time_sec, "raw")

    for seg in segments:
        seg_start = float(seg.get("start", 0.0))
        if low <= seg_start <= high:
            add_candidate(seg_start, "segment")
    for word in words:
        word_start = float(word["start"])
        if low <= word_start <= high:
            add_candidate(word_start, "word")

    return list(by_time.values())


def _gather_end_candidates(
    clip: Clip,
    current_end: float,
    transcript: dict,
) -> list[_BoundaryCandidate]:
    low = current_end - _END_BOUNDARY_WINDOW_SEC
    high = current_end + _END_BOUNDARY_WINDOW_SEC
    segments = list(transcript.get("segments", []))
    words = _flatten_transcript_words(transcript)

    by_time: dict[float, _BoundaryCandidate] = {}

    def add_candidate(time_sec: float, source: str) -> None:
        clean, reason = _segment_end_hint(segments, words, time_sec)
        candidate = _BoundaryCandidate(
            time_sec=float(time_sec),
            clean=clean,
            reason=reason,
            source=source,
        )
        key = _candidate_key(candidate.time_sec)
        existing = by_time.get(key)
        if existing is None or (candidate.clean and not existing.clean):
            by_time[key] = candidate

    add_candidate(current_end, "current")
    add_candidate(clip.end_time_sec, "raw")

    for seg in segments:
        seg_end = float(seg.get("end", 0.0))
        if low <= seg_end <= high:
            add_candidate(seg_end, "segment")
    for word in words:
        word_end = float(word["end"])
        if low <= word_end <= high:
            add_candidate(word_end, "word")

    return list(by_time.values())


def _candidate_priority(current_time: float, candidate: _BoundaryCandidate) -> tuple[int, int, int, float]:
    source_rank = {"current": 0, "raw": 1, "segment": 2, "word": 3}.get(candidate.source, 9)
    weak_rank = 1 if candidate.weak_start else 0
    clean_rank = 0 if candidate.clean else 1
    return (clean_rank, weak_rank, source_rank, abs(candidate.time_sec - current_time))


def _pair_priority(
    current_start: float,
    current_end: float,
    start_candidate: _BoundaryCandidate,
    end_candidate: _BoundaryCandidate,
) -> tuple[int, int, int, float]:
    good_start = start_candidate.clean and not start_candidate.weak_start
    good_end = end_candidate.clean
    return (
        -(int(good_start) + int(good_end)),
        1 if start_candidate.weak_start else 0,
        0 if (good_start or good_end) else 1,
        abs(start_candidate.time_sec - current_start) + abs(end_candidate.time_sec - current_end),
    )


def snap_render_windows_to_sentence_boundaries(
    clips: list[Clip],
    transcript: dict,
) -> list[Clip]:
    """Snap render windows to nearby complete-thought boundaries.

    This runs after Stage 2.5 pruning and operates on the *actual* render
    window (`start + trim_start`, `end - trim_end`). Unlike trim snapping, it
    can undo a harmful trim or move slightly beyond the original selected
    window, as long as the final duration still satisfies the hard
    `[MIN_CLIP_DURATION_SEC, MAX_CLIP_DURATION_SEC]` contract.
    """
    if not transcript.get("segments"):
        return clips

    snapped: list[Clip] = []
    for clip in clips:
        current_start = clip.start_time_sec + clip.trim_start_sec
        current_end = clip.end_time_sec - clip.trim_end_sec
        start_candidates = sorted(
            _gather_start_candidates(clip, current_start, transcript),
            key=lambda c: _candidate_priority(current_start, c),
        )
        end_candidates = sorted(
            _gather_end_candidates(clip, current_end, transcript),
            key=lambda c: _candidate_priority(current_end, c),
        )

        current_start_candidate = next(c for c in start_candidates if c.source == "current")
        current_end_candidate = next(c for c in end_candidates if c.source == "current")
        current_pair = (current_start_candidate, current_end_candidate)
        best_pair = current_pair
        best_priority = _pair_priority(
            current_start,
            current_end,
            current_start_candidate,
            current_end_candidate,
        )

        for start_candidate in start_candidates:
            for end_candidate in end_candidates:
                if end_candidate.time_sec <= start_candidate.time_sec:
                    continue
                duration = end_candidate.time_sec - start_candidate.time_sec
                if duration < MIN_CLIP_DURATION_SEC or duration > MAX_CLIP_DURATION_SEC:
                    continue
                priority = _pair_priority(
                    current_start,
                    current_end,
                    start_candidate,
                    end_candidate,
                )
                if priority < best_priority:
                    best_pair = (start_candidate, end_candidate)
                    best_priority = priority

        start_candidate, end_candidate = best_pair
        start_improved = best_pair[0] is not current_pair[0]
        end_improved = best_pair[1] is not current_pair[1]
        if start_improved or end_improved:
            logger.info(
                "Clip %s: render window snapped %.2f-%.2f -> %.2f-%.2f "
                "(start=%s; end=%s).",
                clip.clip_id,
                current_start,
                current_end,
                start_candidate.time_sec,
                end_candidate.time_sec,
                start_candidate.reason,
                end_candidate.reason,
            )
            snapped.append(
                clip.model_copy(
                    update={
                        "start_time_sec": start_candidate.time_sec,
                        "end_time_sec": end_candidate.time_sec,
                        "trim_start_sec": 0.0,
                        "trim_end_sec": 0.0,
                        "hook_start_sec": None,
                        "hook_end_sec": None,
                    }
                )
            )
            continue

        warnings: list[str] = []
        if not current_start_candidate.clean or current_start_candidate.weak_start:
            warnings.append(f"start@{current_start:.2f}s")
        if not current_end_candidate.clean:
            warnings.append(f"end@{current_end:.2f}s")
        if warnings:
            logger.warning(
                "Clip %s: no valid clean sentence boundary found for %s; leaving render window unchanged.",
                clip.clip_id,
                ", ".join(warnings),
            )
        snapped.append(clip)

    return snapped


def apply_prune_decisions(
    clips: list[Clip],
    decisions: list[_PruneDecision],
    *,
    level: PruneLevel,
    transcript: dict | None = None,
) -> list[Clip]:
    """Return new clips with trim_start / trim_end set from LLM decisions.

    Clips whose ``clip_id`` is missing from ``decisions`` are returned with
    trims of 0 / 0 (no-op). Decisions are always clamped; no exception is
    raised if the model returned invalid numbers.

    When ``transcript`` is provided, each clamped trim pair is additionally
    snapped to the nearest WhisperX segment boundary (see
    :func:`_snap_trims_to_segment_boundaries`) so cuts never land
    mid-sentence. The clamp is authoritative -- snapping only ever produces
    an equally-safe boundary, never a looser one.
    """
    by_id = {d.clip_id: d for d in decisions}
    out: list[Clip] = []
    for clip in clips:
        d = by_id.get(clip.clip_id)
        if d is None or level == "off":
            out.append(clip.model_copy(update={"trim_start_sec": 0.0, "trim_end_sec": 0.0}))
            continue
        ts, te, stats = _clamp_decision(
            clip, d.trim_start_sec, d.trim_end_sec, level=level
        )
        # Surface every non-trivial clamp so silent degradations (e.g. a
        # fake hook nuking every trim) are visible in INFO logs, not just
        # buried in ``prune_raw.json``.
        requested = d.trim_start_sec + d.trim_end_sec
        applied = ts + te
        reshaped = (
            stats.hook_protected
            or stats.min_duration_protected
            or stats.max_pct_protected
            or (requested > 0.0 and abs(applied - requested) > 0.05)
        )
        if reshaped:
            logger.info(
                "Clip %s: prune decision clamped (hook=%s min=%s cap=%s) "
                "requested %.2f/%.2f -> applied %.2f/%.2f",
                clip.clip_id,
                stats.hook_protected,
                stats.min_duration_protected,
                stats.max_pct_protected,
                d.trim_start_sec,
                d.trim_end_sec,
                ts,
                te,
            )
        candidate = clip.model_copy(update={"trim_start_sec": ts, "trim_end_sec": te})
        if transcript is not None:
            snapped_ts, snapped_te = _snap_trims_to_segment_boundaries(
                candidate, transcript, level=level
            )
            if abs(snapped_ts - ts) > 1e-3 or abs(snapped_te - te) > 1e-3:
                logger.info(
                    "Clip %s: prune boundaries snapped to segment edges "
                    "%.2f/%.2f -> %.2f/%.2f",
                    clip.clip_id,
                    ts,
                    te,
                    snapped_ts,
                    snapped_te,
                )
                candidate = candidate.model_copy(
                    update={"trim_start_sec": snapped_ts, "trim_end_sec": snapped_te}
                )
        out.append(candidate)
    return out


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _segments_within_clip(transcript: dict, clip: Clip) -> list[dict]:
    """Return transcript segments that overlap the clip window, with times
    expressed as seconds relative to the clip start.
    """
    s0 = clip.start_time_sec
    s1 = clip.end_time_sec
    lines: list[dict] = []
    for seg in transcript.get("segments", []):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        if end <= s0 or start >= s1:
            continue
        rel_start = max(0.0, start - s0)
        rel_end = min(clip.duration_sec, end - s0)
        if rel_end <= rel_start:
            continue
        lines.append(
            {
                "start": rel_start,
                "end": rel_end,
                "text": (seg.get("text") or "").strip(),
            }
        )
    return lines


def _build_user_message(clips: list[Clip], transcript: dict) -> str:
    """Render a compact textual view of every clip for the LLM user turn."""
    blocks: list[str] = []
    for clip in clips:
        seg_lines = _segments_within_clip(transcript, clip)
        header = (
            f"clip_id: {clip.clip_id}\n"
            f"duration_sec: {clip.duration_sec:.2f}\n"
            f"topic: {clip.topic}"
        )
        if clip.hook_start_sec is not None and clip.hook_end_sec is not None:
            header += (
                f"\nhook_window_sec: [{clip.hook_start_sec:.2f}, {clip.hook_end_sec:.2f}]"
            )
        body = "\n".join(
            f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}" for seg in seg_lines
        )
        if not body:
            body = "(no segments overlap this clip window)"
        blocks.append(f"{header}\n---\n{body}")
    return "\n\n===\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _clips_fingerprint(clips: list[Clip]) -> str:
    """Fingerprint the clip *windows* (not trims, so the cache ignores previous
    prune results when deciding whether to re-ask the LLM).
    """
    payload = json.dumps(
        [
            {
                "id": c.clip_id,
                "s": round(c.start_time_sec, 3),
                "e": round(c.end_time_sec, 3),
                "hs": c.hook_start_sec,
                "he": c.hook_end_sec,
            }
            for c in clips
        ],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolved_gemini_model(config: PipelineConfig) -> str:
    return (config.gemini_model or GEMINI_MODEL).strip()


def _prune_meta(
    *,
    transcript_fp: str,
    clips_fp: str,
    config: PipelineConfig,
    level: PruneLevel,
) -> dict[str, Any]:
    return {
        "version": PRUNE_META_VERSION,
        "transcript_sha256": transcript_fp,
        "clips_sha256": clips_fp,
        "gemini_model": _resolved_gemini_model(config),
        "prune_level": level,
        "llm_backend": current_llm_provider() or "google",
    }


def _load_cached_clips(work_dir: Path, clips: list[Clip]) -> list[Clip] | None:
    artifact = work_dir / PRUNE_ARTIFACT_FILENAME
    if not artifact.is_file():
        return None
    try:
        with open(artifact, "r", encoding="utf-8") as f:
            data = json.load(f)
        cached = {item["clip_id"]: item for item in data.get("clips", [])}
    except Exception as e:
        logger.warning("Prune cache artifact unreadable (%s); re-running.", e)
        return None
    out: list[Clip] = []
    for clip in clips:
        cached_c = cached.get(clip.clip_id)
        if cached_c is None:
            return None
        out.append(
            clip.model_copy(
                update={
                    "trim_start_sec": float(cached_c.get("trim_start_sec", 0.0)),
                    "trim_end_sec": float(cached_c.get("trim_end_sec", 0.0)),
                }
            )
        )
    return out


def _write_cache(
    work_dir: Path,
    *,
    pruned: list[Clip],
    meta: dict[str, Any],
    raw_response: str,
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "clips": [
            {
                "clip_id": c.clip_id,
                "trim_start_sec": c.trim_start_sec,
                "trim_end_sec": c.trim_end_sec,
            }
            for c in pruned
        ]
    }
    (work_dir / PRUNE_ARTIFACT_FILENAME).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (work_dir / PRUNE_RAW_FILENAME).write_text(raw_response, encoding="utf-8")
    with open(work_dir / PRUNE_META_FILENAME, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    logger.info(
        "Wrote %s, %s and %s",
        PRUNE_META_FILENAME,
        PRUNE_ARTIFACT_FILENAME,
        PRUNE_RAW_FILENAME,
    )


def _prune_cache_valid(
    work_dir: Path,
    *,
    transcript_fp: str,
    clips_fp: str,
    config: PipelineConfig,
    level: PruneLevel,
) -> bool:
    meta_path = work_dir / PRUNE_META_FILENAME
    if not meta_path.is_file():
        return False
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return False
    if meta.get("version") != PRUNE_META_VERSION:
        return False
    if meta.get("transcript_sha256") != transcript_fp:
        return False
    if meta.get("clips_sha256") != clips_fp:
        return False
    current_provider = current_llm_provider()
    meta_provider = meta.get("llm_backend")
    if current_provider == "openrouter":
        if meta_provider != "openrouter":
            return False
    elif current_provider == "google":
        if meta_provider not in (None, "google"):
            return False
    if meta.get("gemini_model") != _resolved_gemini_model(config):
        return False
    if meta.get("prune_level") != level:
        return False
    return True


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------


def _parse_decisions(raw_json: str) -> list[_PruneDecision]:
    """Parse a raw JSON response into decisions; bare arrays accepted too."""
    data = json.loads(raw_json)
    if isinstance(data, dict) and "decisions" in data:
        try:
            return _PruneResponse.model_validate(data).decisions
        except ValidationError as e:
            logger.warning("Prune response failed validation: %s", e)
            return []
    if isinstance(data, list):
        decisions: list[_PruneDecision] = []
        for item in data:
            try:
                decisions.append(_PruneDecision.model_validate(item))
            except ValidationError:
                continue
        return decisions
    return []


def request_prune_decisions(
    clips: list[Clip],
    transcript: dict,
    *,
    level: PruneLevel,
    gemini_model: str | None = None,
) -> tuple[list[_PruneDecision], str]:
    """Call Gemini for (potentially) one decision per clip.

    Returns ``(decisions, raw_response)``. ``raw_response`` is the literal
    string Gemini returned (cached to ``prune_raw.json`` for audit). On
    transport or parse failure this raises; callers should catch and treat as
    no-op.
    """
    if level == "off" or not clips:
        return [], "{\"decisions\": []}"

    system = content_pruning_system_prompt(
        min_dur=MIN_CLIP_DURATION_SEC,
        max_dur=MAX_CLIP_DURATION_SEC,
        level=level,
    )
    user_text = _build_user_message(clips, transcript)

    provider = resolve_llm_provider()
    model_name = model_name_for_provider((gemini_model or GEMINI_MODEL).strip(), provider)

    def _call() -> str:
        logger.info(
            "%s content pruning (model=%s, level=%s, clips=%d)...",
            provider,
            model_name,
            level,
            len(clips),
        )
        if provider == "google":
            client = genai.Client(api_key=resolve_gemini_api_key())
            response = client.models.generate_content(
                model=model_name,
                contents=user_text,
                config=gemini_generate_config(
                    system_instruction=system,
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )
            if not response.text:
                raise RuntimeError("Gemini returned empty response text for content pruning")
            return response.text

        client = OpenAI(
            api_key=resolve_openrouter_api_key(),
            base_url=OPENROUTER_BASE_URL,
            default_headers=openrouter_default_headers(),
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        text = _openai_message_text(response.choices[0].message.content)
        if not text:
            raise RuntimeError("OpenRouter returned empty response text for content pruning")
        return text

    raw = _retry_llm("Gemini content pruning", _call)
    decisions = _parse_decisions(raw)
    return decisions, raw


# ---------------------------------------------------------------------------
# Public stage entrypoint (used by pipeline.run_pipeline)
# ---------------------------------------------------------------------------


def run_content_pruning_stage(
    work_dir: Path,
    clips: list[Clip],
    transcript: dict,
    *,
    transcript_fp: str,
    config: PipelineConfig,
) -> list[Clip]:
    """Apply Stage 2.5 pruning to ``clips`` and return the new list.

    - When ``config.prune_level == "off"``, this is a cheap no-op: returns a
      copy of the clips with trim_start/end zeroed.
    - Otherwise, tries the cache first, then calls Gemini. A failing call
      degrades to no-op (the pipeline is never killed by Stage 2.5).
    """
    level = _validated_level(config.prune_level)
    if level == "off":
        logger.info("Content pruning disabled (prune_level=off); skipping Stage 2.5.")
        return [
            clip.model_copy(update={"trim_start_sec": 0.0, "trim_end_sec": 0.0})
            for clip in clips
        ]

    clips_fp = _clips_fingerprint(clips)

    if not config.force_content_pruning and _prune_cache_valid(
        work_dir,
        transcript_fp=transcript_fp,
        clips_fp=clips_fp,
        config=config,
        level=level,
    ):
        cached = _load_cached_clips(work_dir, clips)
        if cached is not None:
            logger.info(
                "Content pruning cache hit (level=%s, %d clips); skipping LLM.",
                level,
                len(clips),
            )
            return cached

    try:
        decisions, raw = request_prune_decisions(
            clips, transcript, level=level, gemini_model=config.gemini_model
        )
    except Exception as e:
        logger.warning(
            "Content pruning call failed (%s); continuing with un-pruned clips.", e
        )
        return [
            clip.model_copy(update={"trim_start_sec": 0.0, "trim_end_sec": 0.0})
            for clip in clips
        ]

    pruned = apply_prune_decisions(
        clips, decisions, level=level, transcript=transcript
    )
    _log_prune_summary(pruned, clips)

    meta = _prune_meta(
        transcript_fp=transcript_fp,
        clips_fp=clips_fp,
        config=config,
        level=level,
    )
    try:
        _write_cache(work_dir, pruned=pruned, meta=meta, raw_response=raw)
    except Exception as e:
        logger.warning("Failed to write prune cache (%s); continuing.", e)
    return pruned


def _validated_level(level: str | None) -> PruneLevel:
    lvl = (level or "balanced").strip().lower()
    if lvl not in VALID_LEVELS:
        logger.warning("Unknown prune_level=%r; falling back to 'balanced'.", level)
        return "balanced"
    return lvl  # type: ignore[return-value]


def _log_prune_summary(pruned: list[Clip], original: list[Clip]) -> None:
    total_before = sum(c.duration_sec for c in original)
    total_after = sum(
        max(0.0, c.duration_sec - c.trim_start_sec - c.trim_end_sec) for c in pruned
    )
    removed = total_before - total_after
    pct = (removed / total_before * 100.0) if total_before > 0 else 0.0
    logger.info(
        "Content pruning done: removed %.1fs across %d clips (%.1f%% of total).",
        removed,
        len(pruned),
        pct,
    )
    for c in pruned:
        if c.trim_start_sec > 0 or c.trim_end_sec > 0:
            final = c.duration_sec - c.trim_start_sec - c.trim_end_sec
            logger.info(
                "  [%s] trim=%.2fs/%.2fs  %.1fs -> %.1fs",
                c.clip_id,
                c.trim_start_sec,
                c.trim_end_sec,
                c.duration_sec,
                final,
            )
