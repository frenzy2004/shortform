"""Tests for the candidate-pool ranking/filtering in :mod:`humeo.clip_selector`.

Covers the "over-generate -> threshold with a floor" policy:

- scores are the primary rank signal
- clips above ``threshold`` are always kept
- if fewer than ``min_kept`` clear the threshold, backfill from the next
  best candidates so the pipeline never ships zero shorts
- the final list is capped at ``max_kept``
- ``needs_review`` clips are demoted but still eligible as a backfill
- ``clip_id`` is renumbered densely in rank order
- empty input is handled without raising
"""

from __future__ import annotations

import logging

from humeo.clip_selector import build_prompt, rank_and_filter_clips
from humeo.config import MAX_CLIP_DURATION_SEC, MIN_CLIP_DURATION_SEC
from humeo_core.schemas import Clip


def _clip(
    clip_id: str,
    *,
    score: float,
    start: float = 0.0,
    duration: float = 60.0,
    needs_review: bool = False,
    topic: str = "t",
    score_breakdown: dict[str, float] | None = None,
) -> Clip:
    # Give each clip a unique, legal window. Score is the only field these
    # tests care about beyond needs_review / clip_id.
    return Clip.model_validate(
        {
            "clip_id": clip_id,
            "topic": topic,
            "start_time_sec": start,
            "end_time_sec": start + duration,
            "virality_score": score,
            "score_breakdown": score_breakdown,
            "needs_review": needs_review,
        }
    )


def test_empty_candidates_returns_empty_list():
    assert rank_and_filter_clips([]) == []


def test_strong_candidates_are_kept_and_sorted():
    candidates = [
        _clip("a", score=0.80, start=0),
        _clip("b", score=0.92, start=100),
        _clip("c", score=0.75, start=200),
    ]
    kept = rank_and_filter_clips(
        candidates, threshold=0.70, min_kept=1, max_kept=5
    )
    # All above threshold, sorted desc, renumbered 001..003.
    assert [c.virality_score for c in kept] == [0.92, 0.80, 0.75]
    assert [c.clip_id for c in kept] == ["001", "002", "003"]


def test_weak_pool_backfills_to_min_kept():
    """Even when no candidate clears the bar, we ship at least ``min_kept``."""
    candidates = [
        _clip("a", score=0.40),
        _clip("b", score=0.55, start=100),
        _clip("c", score=0.62, start=200),
    ]
    kept = rank_and_filter_clips(
        candidates, threshold=0.80, min_kept=2, max_kept=5
    )
    assert len(kept) == 2
    assert [c.virality_score for c in kept] == [0.62, 0.55]


def test_composite_ranking_uses_weighted_breakdown():
    candidates = [
        _clip(
            "a",
            score=0.20,
            score_breakdown={"message_wow": 0.70, "hook_emotion": 0.70, "catchy": 0.70},
        ),
        _clip(
            "b",
            score=0.99,
            start=100,
            score_breakdown={"message_wow": 1.0, "hook_emotion": 0.0, "catchy": 0.0},
        ),
        _clip(
            "c",
            score=0.50,
            start=200,
            score_breakdown={"message_wow": 0.0, "hook_emotion": 1.0, "catchy": 0.0},
        ),
    ]

    kept = rank_and_filter_clips(candidates, threshold=0.00, min_kept=3, max_kept=5)

    assert [c.start_time_sec for c in kept] == [0.0, 100.0, 200.0]


def test_legacy_clips_fallback_to_virality_score():
    candidates = [
        _clip("a", score=0.55, score_breakdown=None),
        _clip("b", score=0.80, start=100, score_breakdown=None),
    ]

    kept = rank_and_filter_clips(candidates, threshold=0.00, min_kept=2, max_kept=5)

    assert [c.virality_score for c in kept] == [0.80, 0.55]


def test_mixed_pool_composite_and_legacy_compared_correctly():
    candidates = [
        _clip(
            "a",
            score=0.60,
            score_breakdown={"message_wow": 0.80, "hook_emotion": 0.80, "catchy": 0.40},
        ),
        _clip("b", score=0.60, start=100, score_breakdown=None),
    ]

    kept = rank_and_filter_clips(candidates, threshold=0.00, min_kept=2, max_kept=5)

    assert [c.start_time_sec for c in kept] == [0.0, 100.0]


def test_single_axis_composite_math(caplog):
    candidates = [
        _clip(
            "a",
            score=0.99,
            score_breakdown={"message_wow": 1.0},
        ),
        _clip("b", score=0.39, start=100, score_breakdown=None),
    ]

    with caplog.at_level(logging.WARNING):
        kept = rank_and_filter_clips(candidates, threshold=0.00, min_kept=2, max_kept=5)

    assert kept[0].start_time_sec == 0.0
    assert "missing axis(es) hook_emotion, catchy" in caplog.text


def test_all_three_axes_max_scores_1_0():
    candidates = [
        _clip(
            "a",
            score=0.01,
            score_breakdown={"message_wow": 1.0, "hook_emotion": 1.0, "catchy": 1.0},
        ),
        _clip("b", score=0.99, start=100, score_breakdown=None, needs_review=True),
    ]

    kept = rank_and_filter_clips(candidates, threshold=0.00, min_kept=2, max_kept=5)

    assert kept[0].start_time_sec == 0.0


def test_needs_review_penalty_still_applies_on_composite():
    candidates = [
        _clip(
            "a",
            score=0.10,
            score_breakdown={"message_wow": 0.90, "hook_emotion": 0.90, "catchy": 0.90},
            needs_review=True,
        ),
        _clip(
            "b",
            score=0.10,
            start=100,
            score_breakdown={"message_wow": 0.60, "hook_emotion": 0.60, "catchy": 0.60},
        ),
    ]

    kept = rank_and_filter_clips(candidates, threshold=0.00, min_kept=2, max_kept=5)

    assert kept[0].start_time_sec == 100.0


def test_missing_axes_logs_warning(caplog):
    candidates = [
        _clip(
            "a",
            score=0.90,
            score_breakdown={"message_wow": 0.70, "catchy": 0.40},
        ),
        _clip("b", score=0.30, start=100, score_breakdown=None),
    ]

    with caplog.at_level(logging.WARNING):
        rank_and_filter_clips(candidates, threshold=0.00, min_kept=2, max_kept=5)

    assert "Clip a score_breakdown missing axis(es) hook_emotion; treating as 0.0." in caplog.text


def test_legacy_point_based_score_breakdown_falls_back_to_virality_score():
    candidates = [
        _clip(
            "a",
            score=0.91,
            score_breakdown={"counter_intuitive_claim": 1.0, "quotable_phrasing": 1.0},
        ),
        _clip("b", score=0.60, start=100, score_breakdown=None),
    ]

    kept = rank_and_filter_clips(candidates, threshold=0.00, min_kept=2, max_kept=5)

    assert kept[0].start_time_sec == 0.0


def test_invalid_short_clips_excluded():
    candidates = [
        _clip("a", score=0.95, duration=MIN_CLIP_DURATION_SEC - 1.0),
        _clip("b", score=0.80, start=100),
        _clip("c", score=0.70, start=200),
    ]

    kept = rank_and_filter_clips(candidates, threshold=0.00, min_kept=1, max_kept=5)

    assert [round(c.duration_sec, 1) for c in kept] == [60.0, 60.0]


def test_invalid_long_clips_excluded():
    candidates = [
        _clip("a", score=0.95, duration=MAX_CLIP_DURATION_SEC + 5.0),
        _clip("b", score=0.80, start=100),
        _clip("c", score=0.70, start=200),
    ]

    kept = rank_and_filter_clips(candidates, threshold=0.00, min_kept=1, max_kept=5)

    assert [round(c.duration_sec, 1) for c in kept] == [60.0, 60.0]


def test_backfill_never_uses_invalid():
    candidates = [
        _clip("a", score=0.92, start=0),
        _clip("b", score=0.81, start=100),
        _clip("c", score=0.70, start=200, duration=MIN_CLIP_DURATION_SEC - 2.0),
        _clip("d", score=0.68, start=300, duration=MAX_CLIP_DURATION_SEC + 2.0),
    ]

    kept = rank_and_filter_clips(candidates, threshold=0.85, min_kept=4, max_kept=5)

    assert len(kept) == 2
    assert [c.virality_score for c in kept] == [0.92, 0.81]


def test_all_invalid_returns_empty():
    candidates = [
        _clip(f"c{i}", score=0.90 - i * 0.01, start=i * 100, duration=MIN_CLIP_DURATION_SEC - 5.0)
        for i in range(5)
    ]

    assert rank_and_filter_clips(candidates, threshold=0.00, min_kept=5, max_kept=8) == []


def test_fewer_valid_than_min_kept_logs_warning(caplog):
    candidates = [
        _clip("a", score=0.92, start=0),
        _clip("b", score=0.81, start=100),
        _clip("c", score=0.70, start=200, duration=MIN_CLIP_DURATION_SEC - 2.0),
        _clip("d", score=0.68, start=300, duration=MIN_CLIP_DURATION_SEC - 10.0),
        _clip("e", score=0.66, start=400, duration=MAX_CLIP_DURATION_SEC + 10.0),
    ]

    with caplog.at_level(logging.WARNING):
        kept = rank_and_filter_clips(candidates, threshold=0.85, min_kept=5, max_kept=8)

    assert len(kept) == 2
    assert "cannot satisfy min_kept=5 without invalid clips" in caplog.text


def test_rich_pool_capped_at_max_kept():
    candidates = [_clip(f"c{i}", score=0.90 - i * 0.01, start=i * 100) for i in range(10)]
    kept = rank_and_filter_clips(
        candidates, threshold=0.50, min_kept=5, max_kept=7
    )
    assert len(kept) == 7
    # Renumbered, monotonically decreasing score.
    assert [c.clip_id for c in kept] == [f"{i:03d}" for i in range(1, 8)]
    scores = [c.virality_score for c in kept]
    assert scores == sorted(scores, reverse=True)


def test_needs_review_clip_is_demoted_but_usable_as_backfill():
    """Reviewed clips lose ground to same-score non-reviewed, but survive when
    they're the only way to hit ``min_kept``.
    """
    candidates = [
        _clip("a", score=0.95, needs_review=True, start=0),
        _clip("b", score=0.60, needs_review=False, start=100),
    ]
    # threshold too high for the clean clip and the reviewed clip loses a
    # heavy penalty; still, min_kept=2 must be satisfied via backfill.
    kept = rank_and_filter_clips(
        candidates, threshold=0.90, min_kept=2, max_kept=5
    )
    assert len(kept) == 2
    # The non-reviewed, lower-score clip should rank *first* because the
    # reviewed one gets a 0.5 priority penalty.
    assert kept[0].clip_id == "001"
    assert kept[0].needs_review is False
    assert kept[1].needs_review is True


def test_threshold_keeps_all_qualifying_clips_even_above_min():
    """When many clips clear the threshold, we keep ALL of them (up to max_kept),
    not just min_kept. That's the "one exceptionally rich transcript ships
    more shorts" behavior the user asked for.
    """
    candidates = [_clip(f"c{i}", score=0.85, start=i * 100) for i in range(6)]
    kept = rank_and_filter_clips(
        candidates, threshold=0.70, min_kept=5, max_kept=8
    )
    assert len(kept) == 6  # all clear threshold, below max_kept


def test_prompt_asks_for_full_candidate_pool_not_just_strong_ones():
    """Regression test: the system prompt must tell Gemini to return the
    full ``count`` pool (with ``needs_review`` on weak ones) so the
    downstream ranker has enough candidates to backfill up to ``min_kept``.

    The previous wording ("Return up to N, fewer stronger clips beat a
    padded list") let Gemini return only 2 of 12 requested for a short
    transcript, which made the min_kept=5 floor unenforceable. The fixed
    prompt must explicitly ask for ``exactly {count}`` and must instruct
    the LLM to flag weak clips rather than omit them.
    """
    transcript = {
        "segments": [
            {"start": i * 5.0, "end": (i + 1) * 5.0, "text": f"line {i}"}
            for i in range(10)
        ]
    }
    system, _user = build_prompt(transcript, candidate_count=12)

    # The prompt must request the full pool, not "up to" / "fewer is better".
    assert "exactly 12" in system
    assert "fewer, stronger clips beat" not in system.lower()
    assert "up to 12" not in system

    # The prompt must instruct the LLM to keep weak clips with a review
    # flag, not drop them.
    assert "needs_review" in system
    assert "never drop" in system.lower() or "do not drop" in system.lower() or "do not self-censor" in system.lower()


def test_clip_ids_are_renumbered_in_rank_order():
    candidates = [
        _clip("zzz", score=0.50, start=0),
        _clip("aaa", score=0.90, start=100),
        _clip("mmm", score=0.70, start=200),
    ]
    kept = rank_and_filter_clips(
        candidates, threshold=0.00, min_kept=3, max_kept=5
    )
    assert [c.clip_id for c in kept] == ["001", "002", "003"]
    assert [c.virality_score for c in kept] == [0.90, 0.70, 0.50]


def test_valid_clips_still_sorted_and_renumbered():
    candidates = [
        _clip("zzz", score=0.82, start=0),
        _clip("aaa", score=0.91, start=100),
        _clip("mmm", score=0.77, start=200),
    ]

    kept = rank_and_filter_clips(candidates, threshold=0.00, min_kept=3, max_kept=5)

    assert [c.clip_id for c in kept] == ["001", "002", "003"]
    assert [c.virality_score for c in kept] == [0.91, 0.82, 0.77]


def test_generic_title_is_penalized_against_specific_title():
    candidates = [
        _clip(
            "a",
            score=0.80,
            score_breakdown={"message_wow": 0.8, "hook_emotion": 0.8, "catchy": 0.8},
            topic="topic a",
        ).model_copy(
            update={
                "suggested_overlay_title": "Big Opportunity",
                "viral_hook": "This changes everything",
            }
        ),
        _clip(
            "b",
            score=0.80,
            start=100,
            score_breakdown={"message_wow": 0.8, "hook_emotion": 0.8, "catchy": 0.8},
            topic="topic b",
        ).model_copy(
            update={
                "suggested_overlay_title": "Robots Cut Costs 90%",
                "viral_hook": "Robots cut costs by 90%",
            }
        ),
    ]

    kept = rank_and_filter_clips(candidates, threshold=0.00, min_kept=2, max_kept=5)

    assert [c.virality_score for c in kept] == [0.80, 0.80]
    assert [c.start_time_sec for c in kept] == [100.0, 0.0]


def test_late_hook_is_penalized_against_early_hook():
    candidates = [
        _clip(
            "a",
            score=0.84,
            score_breakdown={"message_wow": 0.84, "hook_emotion": 0.84, "catchy": 0.84},
        ).model_copy(
            update={
                "suggested_overlay_title": "Robots Cut Costs 90%",
                "viral_hook": "Robots cut costs by 90%",
                "hook_start_sec": 8.0,
                "hook_end_sec": 11.0,
            }
        ),
        _clip(
            "b",
            score=0.84,
            start=100,
            score_breakdown={"message_wow": 0.84, "hook_emotion": 0.84, "catchy": 0.84},
        ).model_copy(
            update={
                "suggested_overlay_title": "Robots Cut Costs 90%",
                "viral_hook": "Robots cut costs by 90%",
                "hook_start_sec": 1.0,
                "hook_end_sec": 4.0,
            }
        ),
    ]

    kept = rank_and_filter_clips(candidates, threshold=0.00, min_kept=2, max_kept=5)

    assert [c.start_time_sec for c in kept] == [100.0, 0.0]
