"""Persist Gemini clip-selection output and skip re-inference when transcript matches."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from humeo.config import GEMINI_MODEL, PipelineConfig
from humeo.env import current_llm_provider

logger = logging.getLogger(__name__)

# v2: Gemini-only meta (no llm_provider). v1 legacy supported in cache_valid.
CURRENT_META_VERSION = 2
META_FILENAME = "clips.meta.json"
RAW_FILENAME = "clip_selection_raw.json"


def transcript_fingerprint(transcript: dict) -> str:
    payload = json.dumps(transcript, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolved_gemini_model(config: PipelineConfig) -> str:
    return (config.gemini_model or GEMINI_MODEL).strip()


def load_meta(work_dir: Path) -> dict[str, Any] | None:
    path = work_dir / META_FILENAME
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cache_valid(meta: dict[str, Any], fingerprint: str, config: PipelineConfig) -> bool:
    if meta.get("transcript_sha256") != fingerprint:
        return False
    gm = resolved_gemini_model(config)
    current_provider = current_llm_provider()
    meta_provider = meta.get("llm_backend")
    if current_provider == "openrouter":
        if meta_provider != "openrouter":
            return False
    elif current_provider == "google":
        if meta_provider not in (None, "google"):
            return False
    ver = meta.get("version", 1)
    if ver >= CURRENT_META_VERSION:
        return meta.get("gemini_model") == gm
    # Legacy v1: had llm_provider + model fields
    if meta.get("llm_provider") == "openai":
        return False
    return meta.get("gemini_model") == gm


def write_artifacts(
    work_dir: Path,
    *,
    transcript: dict,
    config: PipelineConfig,
    raw_response: str,
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    fp = transcript_fingerprint(transcript)
    meta: dict[str, Any] = {
        "version": CURRENT_META_VERSION,
        "transcript_sha256": fp,
        "gemini_model": resolved_gemini_model(config),
        "llm_backend": current_llm_provider() or "google",
    }
    (work_dir / RAW_FILENAME).write_text(raw_response, encoding="utf-8")
    with open(work_dir / META_FILENAME, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    logger.info("Wrote %s and %s", META_FILENAME, RAW_FILENAME)
