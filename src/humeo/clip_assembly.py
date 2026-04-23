"""Hard-cut filler/silence cleanup by assembling multiple kept spans."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from humeo_core.schemas import Clip, ClipPlan, ClipRenderSpan

from humeo.render_window import effective_export_bounds

logger = logging.getLogger(__name__)

_SPAN_BREAK_MIN_GAP_SEC = 0.55
_SPAN_EDGE_PAD_SEC = 0.05
_SPAN_MIN_DURATION_SEC = 0.30
_SEGMENT_BREAK_MIN_GAP_SEC = 0.65
_SEGMENT_MAX_DURATION_SEC = 6.0
_SEGMENT_MAX_WORDS = 18


@dataclass(frozen=True)
class AssembledClip:
    source_path: Path
    clip: Clip
    transcript: dict
    spans: list[ClipRenderSpan]


def _iter_words(transcript: dict) -> list[dict]:
    words: list[dict] = []
    for seg in transcript.get("segments", []) or []:
        for raw in seg.get("words", []) or []:
            try:
                word = {
                    "word": str(raw.get("word", "")).strip(),
                    "start": float(raw["start"]),
                    "end": float(raw["end"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
            if not word["word"] or word["end"] <= word["start"]:
                continue
            words.append(word)
    return words


def derive_render_spans(clip: Clip, transcript: dict) -> list[ClipRenderSpan]:
    if clip.render_spans:
        return list(clip.render_spans)

    start_sec, end_sec = effective_export_bounds(clip)
    words = [
        word
        for word in _iter_words(transcript)
        if word["end"] > start_sec and word["start"] < end_sec
    ]
    if not words:
        return [ClipRenderSpan(start_time_sec=start_sec, end_time_sec=end_sec)]

    spans: list[ClipRenderSpan] = []
    span_start = max(start_sec, float(words[0]["start"]) - _SPAN_EDGE_PAD_SEC)
    prev_end = float(words[0]["end"])

    for word in words[1:]:
        word_start = float(word["start"])
        word_end = float(word["end"])
        if word_start - prev_end >= _SPAN_BREAK_MIN_GAP_SEC:
            span_end = min(end_sec, prev_end + _SPAN_EDGE_PAD_SEC)
            if span_end - span_start >= _SPAN_MIN_DURATION_SEC:
                spans.append(ClipRenderSpan(start_time_sec=span_start, end_time_sec=span_end))
            span_start = max(start_sec, word_start - _SPAN_EDGE_PAD_SEC)
        prev_end = word_end

    final_end = min(end_sec, prev_end + _SPAN_EDGE_PAD_SEC)
    if final_end - span_start >= _SPAN_MIN_DURATION_SEC:
        spans.append(ClipRenderSpan(start_time_sec=span_start, end_time_sec=final_end))

    if not spans:
        spans.append(ClipRenderSpan(start_time_sec=start_sec, end_time_sec=end_sec))
    return spans


def apply_render_spans(clips: list[Clip], transcript: dict) -> list[Clip]:
    out: list[Clip] = []
    for clip in clips:
        spans = derive_render_spans(clip, transcript)
        out.append(clip.model_copy(update={"render_spans": spans}))
    return out


def _segment_local_words(words: list[dict], *, language: str) -> dict:
    segments: list[dict] = []
    chunk: list[dict] = []

    def flush() -> None:
        if not chunk:
            return
        segments.append(
            {
                "start": chunk[0]["start"],
                "end": chunk[-1]["end"],
                "text": " ".join(str(word["word"]) for word in chunk).strip(),
                "words": list(chunk),
            }
        )
        chunk.clear()

    for word in words:
        if chunk:
            gap = float(word["start"]) - float(chunk[-1]["end"])
            dur = float(word["end"]) - float(chunk[0]["start"])
            if (
                gap >= _SEGMENT_BREAK_MIN_GAP_SEC
                or dur >= _SEGMENT_MAX_DURATION_SEC
                or len(chunk) >= _SEGMENT_MAX_WORDS
            ):
                flush()
        chunk.append(word)
    flush()
    return {"segments": segments, "language": language}


def build_assembled_transcript(clip: Clip, transcript: dict) -> dict:
    words = _iter_words(transcript)
    local_words: list[dict] = []
    current_offset = 0.0
    for span in derive_render_spans(clip, transcript):
        for word in words:
            if word["end"] <= span.start_time_sec or word["start"] >= span.end_time_sec:
                continue
            local_words.append(
                {
                    "word": word["word"],
                    "start": max(word["start"], span.start_time_sec) - span.start_time_sec + current_offset,
                    "end": min(word["end"], span.end_time_sec) - span.start_time_sec + current_offset,
                }
            )
        current_offset += span.duration_sec
    language = str(transcript.get("language") or "en")
    return _segment_local_words(local_words, language=language)


def _ffmpeg_concat_filter(spans: list[ClipRenderSpan]) -> str:
    parts: list[str] = []
    for idx, span in enumerate(spans):
        parts.append(
            f"[0:v]trim=start={span.start_time_sec:.3f}:end={span.end_time_sec:.3f},setpts=PTS-STARTPTS[v{idx}]"
        )
        parts.append(
            f"[0:a]atrim=start={span.start_time_sec:.3f}:end={span.end_time_sec:.3f},asetpts=PTS-STARTPTS[a{idx}]"
        )
    concat_inputs = "".join(f"[v{idx}][a{idx}]" for idx in range(len(spans)))
    parts.append(f"{concat_inputs}concat=n={len(spans)}:v=1:a=1[vout][aout]")
    return ";".join(parts)


def assemble_clip(
    source_path: Path,
    clip: Clip,
    transcript: dict,
    output_dir: Path,
) -> AssembledClip:
    spans = derive_render_spans(clip, transcript)
    output_dir.mkdir(parents=True, exist_ok=True)
    assembled_path = output_dir / f"clip_{clip.clip_id}.mp4"

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH")

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source_path),
        "-filter_complex",
        _ffmpeg_concat_filter(spans),
        "-map",
        "[vout]",
        "-map",
        "[aout]",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        str(assembled_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    assembled_transcript = build_assembled_transcript(clip, transcript)
    assembled_transcript_path = output_dir / f"clip_{clip.clip_id}.transcript.json"
    assembled_transcript_path.write_text(
        json.dumps(assembled_transcript, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    timeline_path = output_dir / f"clip_{clip.clip_id}.timeline.json"
    timeline_path.write_text(
        json.dumps(
            {
                "clip_id": clip.clip_id,
                "source_spans": [span.model_dump() for span in spans],
                "assembled_duration_sec": sum(span.duration_sec for span in spans),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    assembled_duration = sum(span.duration_sec for span in spans)
    assembled_clip = clip.model_copy(
        update={
            "start_time_sec": 0.0,
            "end_time_sec": assembled_duration,
            "trim_start_sec": 0.0,
            "trim_end_sec": 0.0,
            "hook_start_sec": None,
            "hook_end_sec": None,
            "render_spans": [],
        }
    )
    logger.info(
        "Assembled clip %s into %d span(s): %.1fs -> %.1fs",
        clip.clip_id,
        len(spans),
        clip.duration_sec,
        assembled_duration,
    )
    return AssembledClip(
        source_path=assembled_path,
        clip=assembled_clip,
        transcript=assembled_transcript,
        spans=spans,
    )


def write_clip_plan(path: Path, clips: list[Clip]) -> Path:
    path.write_text(
        ClipPlan(source_path="", clips=clips).model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return path
