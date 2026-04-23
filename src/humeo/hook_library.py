"""Parse and retrieve viral hook examples from a local zip or directory."""

from __future__ import annotations

import hashlib
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from humeo.config import PipelineConfig

_ENTRY_RE = re.compile(
    r"^\s*\d+\.\s*Hook:\s*(?P<hook>.+?)<br>Example:\s*(?P<example>.+?)<br>Psychology:\s*(?P<psychology>.+?)\s*$",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class HookExample:
    category: str
    hook: str
    example: str
    psychology: str


_LIB_CACHE: dict[str, list[HookExample]] = {}


def resolve_hook_library_path(config: PipelineConfig | None = None) -> Path | None:
    if config is not None and config.hook_library_path is not None:
        return Path(config.hook_library_path)
    raw = (os.environ.get("HUMEO_HOOK_LIBRARY_PATH") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return None


def require_hook_library_path(config: PipelineConfig | None = None) -> Path:
    path = resolve_hook_library_path(config)
    if path is None:
        raise FileNotFoundError(
            "HUMEO_HOOK_LIBRARY_PATH is required for the hook retrieval workflow."
        )
    if not path.exists():
        raise FileNotFoundError(f"Hook library path does not exist: {path}")
    return path


def hook_library_fingerprint(path: Path | None) -> str:
    if path is None:
        return ""
    if not path.exists():
        return ""
    hasher = hashlib.sha256()
    if path.is_file():
        hasher.update(path.read_bytes())
        return hasher.hexdigest()

    for md_path in sorted(p for p in path.rglob("*.md") if p.is_file()):
        hasher.update(str(md_path.relative_to(path)).encode("utf-8"))
        hasher.update(md_path.read_bytes())
    return hasher.hexdigest()


def _tokenize(text: str) -> set[str]:
    return {m.group(0) for m in _TOKEN_RE.finditer(text.lower()) if len(m.group(0)) > 2}


def _ordered_tokens(text: str) -> list[str]:
    return [m.group(0) for m in _TOKEN_RE.finditer(text.lower()) if len(m.group(0)) > 2]


def _iter_markdown_files(path: Path) -> Iterable[tuple[str, str]]:
    if path.is_file():
        with zipfile.ZipFile(path) as zf:
            for name in sorted(n for n in zf.namelist() if n.endswith(".md")):
                yield name, zf.read(name).decode("utf-8", errors="replace")
        return

    for md_path in sorted(p for p in path.rglob("*.md") if p.is_file()):
        yield str(md_path.relative_to(path)).replace("\\", "/"), md_path.read_text(
            encoding="utf-8", errors="replace"
        )


def _category_from_name(name: str) -> str:
    stem = Path(name).stem
    stem = stem.replace("_Hooks", "").replace("_", " ").strip()
    return stem


def _parse_examples(path: Path) -> list[HookExample]:
    examples: list[HookExample] = []
    for name, content in _iter_markdown_files(path):
        category = _category_from_name(name)
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or not line[0].isdigit():
                continue
            match = _ENTRY_RE.match(line)
            if not match:
                continue
            examples.append(
                HookExample(
                    category=category,
                    hook=match.group("hook").strip(),
                    example=match.group("example").strip(),
                    psychology=match.group("psychology").strip(),
                )
            )
    return examples


def load_hook_library(path: Path | None) -> list[HookExample]:
    if path is None:
        return []
    fingerprint = hook_library_fingerprint(path)
    if not fingerprint:
        return []
    cached = _LIB_CACHE.get(fingerprint)
    if cached is not None:
        return cached
    parsed = _parse_examples(path)
    _LIB_CACHE[fingerprint] = parsed
    return parsed


def retrieve_hook_examples(
    query_text: str,
    *,
    topic: str = "",
    path: Path | None,
    limit: int = 8,
) -> list[HookExample]:
    items = load_hook_library(path)
    if not items:
        return []

    query_tokens = _tokenize(f"{topic} {query_text}")
    query_phrases = [
        " ".join(pair)
        for pair in zip(_ordered_tokens(f"{topic} {query_text}"), _ordered_tokens(f"{topic} {query_text}")[1:])
    ]
    if not query_tokens:
        return items[:limit]

    scored: list[tuple[tuple[int, int, int], HookExample]] = []
    for item in items:
        hook_tokens = _tokenize(item.hook)
        example_tokens = _tokenize(item.example)
        category_tokens = _tokenize(item.category)
        hook_overlap = len(query_tokens & hook_tokens)
        example_overlap = len(query_tokens & example_tokens)
        category_overlap = len(query_tokens & category_tokens)
        overlap = hook_overlap + example_overlap + category_overlap
        if overlap == 0:
            continue
        psychology_overlap = len(query_tokens & _tokenize(item.psychology))
        phrase_bonus = sum(1 for phrase in query_phrases if phrase in item.example.lower())
        scored.append(
            (
                (
                    phrase_bonus * 5 + example_overlap * 3 + hook_overlap + category_overlap,
                    phrase_bonus,
                    example_overlap,
                    category_overlap + psychology_overlap,
                ),
                item,
            )
        )

    if not scored:
        return items[:limit]

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:limit]]


def format_hook_examples(examples: list[HookExample]) -> str:
    if not examples:
        return ""
    lines: list[str] = []
    for idx, item in enumerate(examples, start=1):
        lines.append(
            f"{idx}. [{item.category}] Hook: {item.hook}\n"
            f"   Example: {item.example}\n"
            f"   Psychology: {item.psychology}"
        )
    return "\n".join(lines)
