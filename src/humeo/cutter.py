"""Subtitle helpers for the product pipeline."""

import logging
import os
from pathlib import Path

from humeo_core.schemas import Clip, RenderTheme

from humeo.transcript_align import (
    clip_subtitle_words,
    clip_words_to_srt_lines,
    format_ass,
    format_srt,
    group_words_to_cue_chunks,
)

logger = logging.getLogger(__name__)

_NATIVE_HIGHLIGHT_FONT_NAME = "Arial"
_NATIVE_HIGHLIGHT_PURPLE = "&H00F65C8B"


def _balance_reference_caption(text: str) -> str:
    words = text.split()
    if len(words) <= 5 and len(text) <= 28:
        return text
    best_idx = 1
    best_delta = 10**9
    for idx in range(1, len(words)):
        left = " ".join(words[:idx])
        right = " ".join(words[idx:])
        line_penalty = 0
        if len(words[:idx]) < 2 or len(words[idx:]) < 2:
            line_penalty += 1000
        delta = abs(len(left) - len(right)) + abs(len(words[:idx]) - len(words[idx:])) * 6 + line_penalty
        if delta < best_delta:
            best_delta = delta
            best_idx = idx
    return " ".join(words[:best_idx]) + "\n" + " ".join(words[best_idx:])


def _split_native_highlight_lines(words):
    if len(words) <= 3 and len(" ".join(word.word for word in words)) <= 22:
        return [list(words)]
    if len(words) < 2:
        return [list(words)]
    best_idx = 1
    best_delta = 10**9
    for idx in range(1, len(words)):
        left_words = words[:idx]
        right_words = words[idx:]
        left = " ".join(word.word for word in left_words)
        right = " ".join(word.word for word in right_words)
        line_penalty = 0
        if len(left_words) < 2 or len(right_words) < 2:
            line_penalty += 800
        delta = abs(len(left) - len(right)) + abs(len(left_words) - len(right_words)) * 7 + line_penalty
        if delta < best_delta:
            best_delta = delta
            best_idx = idx
    return [list(words[:best_idx]), list(words[best_idx:])]


def _native_highlight_font_path() -> Path | None:
    windows_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    for filename in ("arialbd.ttf", "Arialbd.ttf", "ARIALBD.TTF", "arial.ttf"):
        path = windows_fonts / filename
        if path.is_file():
            return path
    return None


def _text_width(font, text: str) -> float:
    if not text:
        return 0.0
    if hasattr(font, "getlength"):
        return float(font.getlength(text))
    bbox = font.getbbox(text)
    return float(bbox[2] - bbox[0])


def _text_height(font) -> int:
    bbox = font.getbbox("Ag")
    return max(1, int(round(bbox[3] - bbox[1])))


def _escape_ass_text(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\N")
    )


def _fmt_ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    whole = int(secs)
    cs = int(round((secs - whole) * 100))
    if cs >= 100:
        cs = 99
    return f"{hours:d}:{minutes:02d}:{whole:02d}.{cs:02d}"


def _format_native_highlight_ass(
    cue_chunks,
    *,
    play_res_x: int,
    play_res_y: int,
    font_size: int,
    margin_v: int,
    font_name: str,
) -> str:
    from PIL import ImageFont

    font_path = _native_highlight_font_path()
    if font_path is not None:
        font = ImageFont.truetype(str(font_path), size=font_size)
    else:
        font = ImageFont.load_default()

    line_height = max(font_size, _text_height(font) + 6)
    line_gap = max(8, int(round(font_size * 0.08)))
    bottom_anchor = play_res_y - margin_v
    center_x = play_res_x / 2.0
    space_width = _text_width(font, " ")

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {play_res_x}\n"
        f"PlayResY: {play_res_y}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: None\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Base,{font_name},{font_size},&H00FFFFFF,&H000000FF,&H00101010,&H00000000,-1,0,0,0,100,100,-1,0,1,4,0,8,0,0,0,0\n"
        f"Style: Highlight,{font_name},{font_size},&H00FFFFFF,&H000000FF,&H00000000,{_NATIVE_HIGHLIGHT_PURPLE},-1,0,0,0,100,100,-1,0,4,0,10,7,0,0,0,0\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events: list[str] = []
    for cue_words in cue_chunks:
        if not cue_words:
            continue
        lines = _split_native_highlight_lines(cue_words)
        block_height = len(lines) * line_height + max(0, len(lines) - 1) * line_gap
        block_top = bottom_anchor - block_height
        cue_start = cue_words[0].start_time
        cue_end = cue_words[-1].end_time
        for line_idx, line_words in enumerate(lines):
            if not line_words:
                continue
            line_text = " ".join(word.word for word in line_words)
            line_top = block_top + line_idx * (line_height + line_gap)
            line_left = (play_res_x - _text_width(font, line_text)) / 2.0
            events.append(
                "Dialogue: 0,"
                f"{_fmt_ass_time(cue_start)},{_fmt_ass_time(cue_end)},Base,,0,0,0,,"
                f"{{\\an7\\pos({line_left:.1f},{line_top:.1f})}}{_escape_ass_text(line_text)}"
            )
            x_cursor = line_left
            for word_idx, word in enumerate(line_words):
                if word_idx < len(line_words) - 1:
                    next_start = line_words[word_idx + 1].start_time
                elif line_idx < len(lines) - 1 and lines[line_idx + 1]:
                    next_start = lines[line_idx + 1][0].start_time
                else:
                    next_start = cue_end
                highlight_end = min(cue_end, max(word.end_time, next_start))
                highlight_end = max(word.start_time + 0.05, highlight_end)
                events.append(
                    "Dialogue: 1,"
                    f"{_fmt_ass_time(word.start_time)},{_fmt_ass_time(highlight_end)},Highlight,,0,0,0,,"
                    f"{{\\an7\\pos({x_cursor:.1f},{line_top:.1f})\\blur0.8}}{_escape_ass_text(word.word)}"
                )
                x_cursor += _text_width(font, word.word) + space_width

    return header + "\n".join(events) + ("\n" if events else "")


def generate_srt(
    clip: Clip,
    transcript: dict,
    output_dir: Path,
    *,
    max_words_per_cue: int = 8,
    max_cue_sec: float = 4.0,
) -> Path:
    """
    Build an SRT file from word-level ASR aligned to this clip's timeline.

    ``transcript`` is the persisted ``transcript.json`` (segments with optional
    per-word timestamps). Times are shifted so 0 = clip in-point.
    """
    srt_path = output_dir / f"clip_{clip.clip_id}.srt"
    aligned = clip_subtitle_words(transcript, clip)
    lines = clip_words_to_srt_lines(
        aligned.words,
        max_words_per_cue=max_words_per_cue,
        max_cue_sec=max_cue_sec,
    )
    srt_path.write_text(format_srt(lines), encoding="utf-8")
    logger.info("Generated SRT: %s (%d cues)", srt_path, len(lines))
    return srt_path


def generate_ass(
    clip: Clip,
    transcript: dict,
    output_dir: Path,
    *,
    max_words_per_cue: int = 4,
    max_cue_sec: float = 2.2,
    play_res_x: int = 1080,
    play_res_y: int = 1920,
    font_size: int = 48,
    margin_v: int = 160,
    margin_h: int = 60,
    font_name: str = "Arial",
    render_theme: RenderTheme = RenderTheme.LEGACY,
) -> Path:
    """Generate an ASS caption file tuned for direct libass rendering.

    Unlike SRT → libass (default PlayResY=288), an ASS file with
    ``PlayResY = output_height`` means libass' scale factor is 1.0, so the
    ``font_size`` / ``margin_v`` arguments below are honest output pixels.

    This is the root-cause fix for the "captions rendering in the middle of
    the frame, four times too large" bug the user reported.
    """
    ass_path = output_dir / f"clip_{clip.clip_id}.ass"
    aligned = clip_subtitle_words(transcript, clip)
    cue_words = max_words_per_cue
    cue_sec = max_cue_sec
    cue_font_size = font_size
    cue_margin_v = margin_v
    prefer_break_on_punctuation = False
    min_words_before_break = 1
    if render_theme == RenderTheme.REFERENCE_LOWER_THIRD:
        cue_words = max(max_words_per_cue, 7)
        cue_sec = max(max_cue_sec, 2.6)
        cue_font_size = max(font_size, 52)
        cue_margin_v = min(margin_v, 136)
        prefer_break_on_punctuation = True
        min_words_before_break = 5
    elif render_theme == RenderTheme.NATIVE_HIGHLIGHT:
        cue_words = 6
        cue_sec = 2.4
        cue_font_size = max(font_size, 86)
        cue_margin_v = max(margin_v, 300)
        prefer_break_on_punctuation = True
        min_words_before_break = 4

    cue_chunks = group_words_to_cue_chunks(
        aligned.words,
        max_words_per_cue=cue_words,
        max_cue_sec=cue_sec,
        prefer_break_on_punctuation=prefer_break_on_punctuation,
        min_words_before_break=min_words_before_break,
    )
    lines = [
        (chunk[0].start_time, chunk[-1].end_time, " ".join(word.word for word in chunk))
        for chunk in cue_chunks
    ]
    if render_theme == RenderTheme.REFERENCE_LOWER_THIRD:
        lines = [(start, end, _balance_reference_caption(text)) for start, end, text in lines]
        ass_text = format_ass(
            lines,
            play_res_x=play_res_x,
            play_res_y=play_res_y,
            font_size=cue_font_size,
            margin_v=cue_margin_v,
            margin_h=margin_h,
            font_name="Source Sans 3",
            render_theme=render_theme,
        )
    elif render_theme == RenderTheme.NATIVE_HIGHLIGHT:
        ass_text = _format_native_highlight_ass(
            cue_chunks,
            play_res_x=play_res_x,
            play_res_y=play_res_y,
            font_size=cue_font_size,
            margin_v=cue_margin_v,
            font_name=_NATIVE_HIGHLIGHT_FONT_NAME,
        )
    else:
        ass_text = format_ass(
            lines,
            play_res_x=play_res_x,
            play_res_y=play_res_y,
            font_size=cue_font_size,
            margin_v=cue_margin_v,
            margin_h=margin_h,
            font_name=font_name,
            render_theme=render_theme,
        )
    ass_path.write_text(ass_text, encoding="utf-8")
    logger.info("Generated ASS: %s (%d cues)", ass_path, len(lines))
    return ass_path
