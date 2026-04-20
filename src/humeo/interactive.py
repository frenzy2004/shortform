"""Plain stdin interactive gates for the pipeline."""

from __future__ import annotations

from pathlib import Path

from humeo_core.schemas import ApprovalResult, Clip, RatingFeedback

_ISSUE_MAP = {
    "a": "wrong_moments",
    "b": "bad_cuts",
    "c": "boring",
    "d": "confusing",
    "e": "wrong_layout",
    "f": "length_off",
    "g": "other",
}


def _preview(text: str, limit: int = 100) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def approve_clips(clips: list[Clip]) -> ApprovalResult:
    """Prompt the user to approve or refine the selected clips."""
    clip_ids = [clip.clip_id for clip in clips]

    for clip in clips:
        print(
            f'[{clip.clip_id}] score={clip.virality_score:.2f}  '
            f'duration={clip.duration_sec:.1f}s  "{clip.topic}"'
        )
        print(f'       "{_preview(clip.transcript)}"')

    print()
    print("Actions:")
    print("  numbers in order (e.g. '3,1,5')  — select these clips to proceed")
    print("  'all'                            — accept all clips as-is")
    print("  'refine <note>'                  — re-run selection with steering")
    print("  'quit'                           — abort pipeline")
    print()

    while True:
        raw = input("> ").strip()
        lowered = raw.lower()

        if lowered == "all":
            return ApprovalResult(action="accept_all", selected_ids=list(clip_ids))
        if lowered == "quit":
            return ApprovalResult(action="quit")
        if lowered.startswith("refine"):
            note = raw[6:].strip()
            if not note:
                print("Refine requires a note. Try: refine more emotional clips")
                continue
            return ApprovalResult(action="refine", steering_note=note)

        tokens = [token.strip() for token in raw.split(",") if token.strip()]
        if not tokens:
            print("Enter clip numbers, 'all', 'refine <note>', or 'quit'.")
            continue

        selected_ids: list[str] = []
        seen_ids: set[str] = set()
        invalid = False
        for token in tokens:
            clip_id: str | None = None
            if token in clip_ids:
                clip_id = token
            elif token.isdigit():
                idx = int(token)
                if 1 <= idx <= len(clips):
                    clip_id = clips[idx - 1].clip_id

            if clip_id is None:
                print(f"Unknown clip selection: {token}")
                invalid = True
                break
            if clip_id in seen_ids:
                print(f"Duplicate clip selection: {token}")
                invalid = True
                break
            seen_ids.add(clip_id)
            selected_ids.append(clip_id)

        if invalid:
            continue

        return ApprovalResult(action="proceed", selected_ids=selected_ids)


def rate_output(outputs: list[Path]) -> RatingFeedback:
    """Prompt the user to rate the rendered outputs."""
    print("Outputs:")
    for path in outputs:
        print(f"  {path}")
    print()
    print("Watch them, then rate:")
    print("  1. slop   2. good   3. great")

    while True:
        rating_raw = input("> ").strip()
        if rating_raw in {"1", "2", "3"}:
            rating = int(rating_raw)
            break
        print("Enter 1, 2, or 3.")

    if rating == 3:
        return RatingFeedback(rating=3)

    print("What's wrong? (space-separated letters, or empty for skip):")
    print("  [a] wrong_moments  [b] bad_cuts  [c] boring  [d] confusing")
    print("  [e] wrong_layout   [f] length_off  [g] other (free text)")

    while True:
        issues_raw = input("> ").strip()
        if not issues_raw:
            return RatingFeedback(rating=rating)

        tokens = issues_raw.lower().split()
        issues: list[str] = []
        invalid = [token for token in tokens if token not in _ISSUE_MAP]
        if invalid:
            print(f"Unknown issue selection: {' '.join(invalid)}")
            continue

        for token in tokens:
            issue = _ISSUE_MAP[token]
            if issue not in issues:
                issues.append(issue)

        free_text = None
        if "other" in issues:
            other_text = input("> ").strip()
            free_text = other_text or None

        return RatingFeedback(rating=rating, issues=issues, free_text=free_text)
