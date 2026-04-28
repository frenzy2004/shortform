"""Curate a small review pack from a larger batch render."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from humeo.clip_selector import clip_quality_priority_score
from humeo_core.schemas import Clip

_SHORT_FILENAME_RE = re.compile(r"^short_(?P<clip_id>\d+)\.mp4$", re.IGNORECASE)


def _load_clip_map(work_dir: Path) -> dict[str, Clip]:
    for filename in ("clips.json", "assembled_clips.json"):
        path = work_dir / filename
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data.get("clips", data) if isinstance(data, dict) else data
        return {
            clip["clip_id"]: Clip.model_validate(clip)
            for clip in items
            if isinstance(clip, dict) and clip.get("clip_id")
        }
    return {}


def _default_work_dir_for_source(source_dir: Path, repo_root: Path) -> Path:
    match = re.fullmatch(r"videoplayback_(\d+)", source_dir.name)
    if match:
        return repo_root / f".humeo_batch_videoplayback{match.group(1)}"
    return repo_root / f".humeo_{source_dir.name}"


def build_best_of_review_pack(
    batch_root: Path,
    destination_dir: Path,
    *,
    per_source: int = 2,
    repo_root: Path | None = None,
    work_dir_map: dict[str, Path] | None = None,
) -> list[Path]:
    batch_root = Path(batch_root)
    destination_dir = Path(destination_dir)
    repo_root = Path(repo_root) if repo_root is not None else batch_root.parent
    destination_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    manifest: list[dict[str, object]] = []
    for source_dir in sorted(path for path in batch_root.iterdir() if path.is_dir()):
        work_dir = (
            work_dir_map[source_dir.name]
            if work_dir_map is not None and source_dir.name in work_dir_map
            else _default_work_dir_for_source(source_dir, repo_root)
        )
        clip_map = _load_clip_map(work_dir)
        ranked: list[tuple[float, Path, str, Clip | None]] = []
        for mp4_path in sorted(source_dir.glob("short_*.mp4")):
            match = _SHORT_FILENAME_RE.match(mp4_path.name)
            if not match:
                continue
            clip_id = match.group("clip_id")
            clip = clip_map.get(clip_id)
            score = clip_quality_priority_score(clip) if clip is not None else 0.0
            ranked.append((score, mp4_path, clip_id, clip))

        ranked.sort(
            key=lambda item: (
                item[0],
                item[3].virality_score if item[3] is not None else 0.0,
                -(item[3].duration_sec if item[3] is not None else 0.0),
            ),
            reverse=True,
        )
        for rank, (score, mp4_path, clip_id, clip) in enumerate(ranked[: max(1, per_source)], start=1):
            target_path = destination_dir / f"{source_dir.name}__pick{rank:02d}__{mp4_path.name}"
            shutil.copy2(mp4_path, target_path)
            copied.append(target_path)
            manifest.append(
                {
                    "source": source_dir.name,
                    "rank": rank,
                    "score": round(score, 4),
                    "output_path": str(target_path),
                    "original_path": str(mp4_path),
                    "clip_id": clip.clip_id if clip is not None else clip_id,
                    "title": clip.suggested_overlay_title if clip is not None else "",
                    "topic": clip.topic if clip is not None else "",
                }
            )

    (destination_dir / "best_of_manifest.json").write_text(
        json.dumps({"clips": manifest}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return copied
