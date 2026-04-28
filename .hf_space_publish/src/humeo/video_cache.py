"""Video ingest cache: YouTube id → work directory + manifest on disk."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from humeo.env import default_humeo_cache_root

logger = logging.getLogger(__name__)

# Typical watch / short / embed URLs (11-char id).
_YOUTUBE_ID_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/)([a-zA-Z0-9_-]{11})"
)

MANIFEST_VERSION = 1
MANIFEST_NAME = "video_cache_manifest.json"
LOCAL_SOURCE_INFO_NAME = "source.local.json"


class VideoCacheEntry(BaseModel):
    """One row in the global cache manifest (machine-checkable, Pydantic-only)."""

    video_id: str
    url: str = ""
    title: str = ""
    channel: str = ""
    work_dir: str
    source_mp4: str
    transcript_json: str
    downloaded_at: str = ""  # ISO 8601 UTC when ingest completed


class VideoCacheManifest(BaseModel):
    version: int = MANIFEST_VERSION
    entries: dict[str, VideoCacheEntry] = Field(default_factory=dict)


def extract_youtube_video_id(url: str) -> str | None:
    """Return the 11-character video id, or None if not a recognized YouTube URL."""
    m = _YOUTUBE_ID_RE.search(url)
    return m.group(1) if m else None


def looks_like_local_source(source: str) -> bool:
    """Return True when ``source`` should be treated as a local file path."""
    if extract_youtube_video_id(source):
        return False
    return "://" not in source


def normalize_local_source_path(source: str) -> Path | None:
    """Return an absolute local path for ``source`` when it is file-like."""
    if not looks_like_local_source(source):
        return None
    return Path(source).expanduser().resolve(strict=False)


def local_source_cache_key(source: str) -> str | None:
    """Return a stable cache key for a local source path."""
    path = normalize_local_source_path(source)
    if path is None:
        return None
    stem = re.sub(r"[^a-zA-Z0-9]+", "-", path.stem).strip("-").lower() or "video"
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
    return f"{stem}-{digest}"


def _local_source_info_path(work_dir: Path) -> Path:
    return work_dir / LOCAL_SOURCE_INFO_NAME


def read_local_source_info(work_dir: Path) -> dict[str, str]:
    """Read ``source.local.json`` when present."""
    path = _local_source_info_path(work_dir)
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def write_local_source_info(work_dir: Path, source_path: Path) -> Path:
    """Persist the original local source path used for ``source.mp4``."""
    work_dir.mkdir(parents=True, exist_ok=True)
    path = _local_source_info_path(work_dir)
    payload = {"local_source_path": str(Path(source_path).expanduser().resolve(strict=False))}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return path


def local_source_matches(work_dir: Path, source: str) -> bool:
    """Return True when ``work_dir`` already contains the same local source."""
    path = normalize_local_source_path(source)
    if path is None:
        return False
    info = read_local_source_info(work_dir)
    return info.get("local_source_path") == str(path)


def manifest_path(cache_root: Path | None = None) -> Path:
    root = cache_root if cache_root is not None else default_humeo_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    return root / MANIFEST_NAME


def load_manifest(cache_root: Path | None = None) -> VideoCacheManifest:
    path = manifest_path(cache_root)
    if not path.exists():
        return VideoCacheManifest()
    with open(path, encoding="utf-8") as f:
        data: Any = json.load(f)
    return VideoCacheManifest.model_validate(data)


def save_manifest(manifest: VideoCacheManifest, cache_root: Path | None = None) -> Path:
    path = manifest_path(cache_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))
    return path


def resolve_work_directory(
    *,
    youtube_url: str,
    explicit_work_dir: Path | None,
    use_video_cache: bool,
    cache_root: Path | None,
) -> Path:
    """Pick the directory for ``source.mp4``, ``transcript.json``, ``clips.json``, etc.

    - If ``explicit_work_dir`` is set (CLI ``--work-dir``), use it.
    - Else if video cache is disabled, use ``.humeo_work``.
    - Else if the source is a local file path, use ``<cache_root>/local/<source_key>/``.
    - Else if the source has no YouTube id, use ``.humeo_work``.
    - Else use ``<cache_root>/videos/<video_id>/`` (creates parents as needed).
    """
    if explicit_work_dir is not None:
        p = Path(explicit_work_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    if not use_video_cache:
        p = Path(".humeo_work").resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    local_key = local_source_cache_key(youtube_url)
    if local_key:
        root = cache_root if cache_root is not None else default_humeo_cache_root()
        p = (root / "local" / local_key).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    vid = extract_youtube_video_id(youtube_url)
    if not vid:
        p = Path(".humeo_work").resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    root = cache_root if cache_root is not None else default_humeo_cache_root()
    p = (root / "videos" / vid).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def ingest_complete(work_dir: Path, source: str | None = None) -> bool:
    """Return True if both video and transcript exist and match the current source."""
    complete = (work_dir / "source.mp4").is_file() and (work_dir / "transcript.json").is_file()
    if not complete:
        return False
    if source is None:
        return True
    local_path = normalize_local_source_path(source)
    if local_path is None:
        return True
    return local_source_matches(work_dir, source)


def read_youtube_info_json(work_dir: Path) -> dict[str, Any]:
    """Read ``source.info.json`` written by yt-dlp ``--write-info-json``."""
    p = work_dir / "source.info.json"
    if not p.is_file():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def upsert_manifest_from_info(
    *,
    work_dir: Path,
    youtube_url: str,
    info: dict[str, Any],
    cache_root: Path | None = None,
) -> None:
    """Merge or add a manifest entry after successful ingest."""
    vid = (info.get("id") or extract_youtube_video_id(youtube_url) or "").strip()
    if not vid:
        logger.debug("No video id for manifest; skipping.")
        return

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    wd = work_dir.resolve()
    entry = VideoCacheEntry(
        video_id=vid,
        url=str(info.get("webpage_url") or youtube_url),
        title=str(info.get("title") or ""),
        channel=str(info.get("channel") or info.get("uploader") or ""),
        work_dir=str(wd),
        source_mp4=str((wd / "source.mp4").resolve()),
        transcript_json=str((wd / "transcript.json").resolve()),
        downloaded_at=now,
    )

    manifest = load_manifest(cache_root)
    manifest.entries[vid] = entry
    path = save_manifest(manifest, cache_root)
    logger.info("Updated video cache manifest: %s", path)
