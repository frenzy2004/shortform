"""Load Jinja2 prompt templates (editable; override dir via HUMEO_PROMPTS_DIR)."""

from __future__ import annotations

import os
from pathlib import Path

import jinja2


def _prompt_loader() -> jinja2.BaseLoader:
    override = (os.environ.get("HUMEO_PROMPTS_DIR") or "").strip()
    if override:
        return jinja2.FileSystemLoader(str(Path(override).expanduser()))
    return jinja2.PackageLoader("humeo", "prompts")


def clip_selection_prompts(
    *,
    transcript_text: str,
    min_dur: float,
    max_dur: float,
    count: int,
    steering_notes: list[str] | None = None,
) -> tuple[str, str]:
    """Return ``(system_instruction, user_message)`` for Gemini clip selection."""
    env = jinja2.Environment(loader=_prompt_loader(), autoescape=False, trim_blocks=True)
    ctx = {
        "min_dur": min_dur,
        "max_dur": max_dur,
        "count": count,
        "transcript_text": transcript_text,
        "steering_notes": steering_notes or [],
    }
    system = env.get_template("clip_selection_system.jinja2").render(**ctx)
    user = env.get_template("clip_selection_user.jinja2").render(**ctx)
    return system, user


def hook_detection_system_prompt() -> str:
    """Return the system prompt for Stage 2.25 hook detection.

    The user message is built in :mod:`humeo.hook_detector` because the
    segment listing is dynamic per-clip.
    """
    env = jinja2.Environment(loader=_prompt_loader(), autoescape=False, trim_blocks=True)
    return env.get_template("hook_detection_system.jinja2").render()


def content_pruning_system_prompt(
    *,
    min_dur: float,
    max_dur: float,
    level: str,
) -> str:
    """Return the system prompt for Stage 2.5 content pruning.

    The user message is built in ``humeo.content_pruning`` from the list of
    candidate clips (clip-relative segment lines) since it is not static text.
    """
    env = jinja2.Environment(loader=_prompt_loader(), autoescape=False, trim_blocks=True)
    return env.get_template("content_pruning_system.jinja2").render(
        min_dur=min_dur, max_dur=max_dur, level=level
    )
