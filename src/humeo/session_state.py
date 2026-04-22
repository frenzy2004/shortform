"""Persist and load interactive session state."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from humeo.video_cache import extract_youtube_video_id, normalize_local_source_path
from humeo_core.schemas import SessionState

logger = logging.getLogger(__name__)

SESSION_STATE_FILENAME = "session_state.json"


def source_key_for_url(youtube_url: str) -> str:
    video_id = extract_youtube_video_id(youtube_url)
    if video_id:
        return f"youtube:{video_id}"
    local_path = normalize_local_source_path(youtube_url)
    if local_path is not None:
        return f"local:{local_path}"
    return f"url:{youtube_url}"


def fresh_state(youtube_url: str) -> SessionState:
    return SessionState(source_key=source_key_for_url(youtube_url))


def load_state(work_dir: Path, youtube_url: str) -> SessionState:
    """Load session state, resetting on corruption or source mismatch."""
    path = work_dir / SESSION_STATE_FILENAME
    state = fresh_state(youtube_url)
    if not path.is_file():
        return state

    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        loaded = SessionState.model_validate(payload)
    except Exception as exc:
        logger.warning("Session state at %s is invalid. Starting fresh. Error: %s", path, exc)
        return state

    if loaded.source_key and loaded.source_key != state.source_key:
        logger.warning(
            "Session state at %s belongs to %s, not %s. Starting fresh.",
            path,
            loaded.source_key,
            state.source_key,
        )
        return state

    if not loaded.source_key:
        loaded.source_key = state.source_key
    return loaded


def save_state(work_dir: Path, state: SessionState) -> Path:
    """Persist session state to the work dir."""
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / SESSION_STATE_FILENAME
    with open(path, "w", encoding="utf-8") as f:
        f.write(state.model_dump_json(indent=2))
        f.write("\n")
    return path
