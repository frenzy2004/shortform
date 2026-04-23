"""Map source-timeline ASR words to per-clip subtitle timings (t=0 at clip in-point)."""

from __future__ import annotations

from humeo_core.schemas import Clip, ClipSubtitleWords, RenderTheme, TranscriptWord

# Whisper / WhisperX / OpenAI-normalized segment shapes
_MAX_WORDS_PER_CUE = 8
_MAX_CUE_SEC = 4.0
_PUNCTUATION_BREAK_CHARS = (".", "?", "!", ";", ":")
_SENTENCE_RESTART_WORDS = frozenset(
    {
        "And",
        "But",
        "Did",
        "Now",
        "So",
        "That",
        "Then",
        "This",
        "Those",
        "What",
        "When",
        "Where",
        "Why",
    }
)


def _iter_words_from_segments(transcript: dict) -> list[TranscriptWord]:
    out: list[TranscriptWord] = []
    for seg in transcript.get("segments", []) or []:
        words = seg.get("words") or []
        if words:
            for raw in words:
                w = str(raw.get("word", "")).strip()
                if not w:
                    continue
                out.append(
                    TranscriptWord(
                        word=w,
                        start_time=float(raw["start"]),
                        end_time=float(raw["end"]),
                    )
                )
            continue
        # Segment-level only (no word list): treat whole segment as one token
        text = str(seg.get("text", "")).strip()
        if text:
            out.append(
                TranscriptWord(
                    word=text,
                    start_time=float(seg.get("start", 0.0)),
                    end_time=float(seg.get("end", 0.0)),
                )
            )
    return out


def clip_subtitle_words(transcript: dict, clip: Clip) -> ClipSubtitleWords:
    """Words overlapping ``clip`` with times shifted to start at 0 (clip-local)."""
    clip_start = clip.start_time_sec
    clip_end = clip.end_time_sec
    words = _iter_words_from_segments(transcript)
    local: list[TranscriptWord] = []
    for w in words:
        if w.end_time <= clip_start or w.start_time >= clip_end:
            continue
        t0 = max(w.start_time, clip_start) - clip_start
        t1 = min(w.end_time, clip_end) - clip_start
        if t1 <= t0:
            continue
        local.append(TranscriptWord(word=w.word, start_time=t0, end_time=t1))

    if local:
        return ClipSubtitleWords(words=local)

    return ClipSubtitleWords(words=_fallback_even_words(clip))


def _fallback_even_words(clip: Clip) -> list[TranscriptWord]:
    """Even split over clip duration when no word timestamps exist."""
    text = (clip.transcript or "").strip()
    if not text:
        return []
    parts = text.split()
    if not parts:
        return []
    d = clip.duration_sec
    step = d / len(parts)
    out: list[TranscriptWord] = []
    for i, p in enumerate(parts):
        out.append(
            TranscriptWord(
                word=p,
                start_time=i * step,
                end_time=(i + 1) * step if i < len(parts) - 1 else d,
            )
        )
    return out


def _looks_like_sentence_restart(prev_word: str, next_word: str) -> bool:
    prev = prev_word.rstrip("\"')]}")
    nxt = next_word.lstrip("\"'([{")
    if not prev or not nxt:
        return False
    if nxt in _SENTENCE_RESTART_WORDS:
        return True
    return any(ch.isdigit() for ch in prev) and nxt[0].isupper()


def clip_words_to_srt_lines(
    words: list[TranscriptWord],
    *,
    max_words_per_cue: int = _MAX_WORDS_PER_CUE,
    max_cue_sec: float = _MAX_CUE_SEC,
    prefer_break_on_punctuation: bool = False,
    min_words_before_break: int = 1,
) -> list[tuple[float, float, str]]:
    """Group words into SRT cues: max N words and max duration per cue."""
    chunks = group_words_to_cue_chunks(
        words,
        max_words_per_cue=max_words_per_cue,
        max_cue_sec=max_cue_sec,
        prefer_break_on_punctuation=prefer_break_on_punctuation,
        min_words_before_break=min_words_before_break,
    )
    return [
        (chunk[0].start_time, chunk[-1].end_time, " ".join(w.word for w in chunk))
        for chunk in chunks
    ]


def group_words_to_cue_chunks(
    words: list[TranscriptWord],
    *,
    max_words_per_cue: int = _MAX_WORDS_PER_CUE,
    max_cue_sec: float = _MAX_CUE_SEC,
    prefer_break_on_punctuation: bool = False,
    min_words_before_break: int = 1,
) -> list[list[TranscriptWord]]:
    """Group words into timed cue chunks while preserving per-word timings."""
    if not words:
        return []
    max_words_per_cue = max(1, int(max_words_per_cue))
    max_cue_sec = max(0.2, float(max_cue_sec))
    min_words_before_break = max(1, int(min_words_before_break))
    chunks_out: list[list[TranscriptWord]] = []
    i = 0
    n = len(words)
    while i < n:
        chunk: list[TranscriptWord] = [words[i]]
        t0 = words[i].start_time
        end_t = words[i].end_time
        j = i + 1
        while j < n:
            w = words[j]
            if len(chunk) >= max_words_per_cue:
                break
            if w.start_time - t0 > max_cue_sec:
                break
            if (
                prefer_break_on_punctuation
                and (len(chunk) >= 2 or end_t - t0 >= 0.45)
                and _looks_like_sentence_restart(chunk[-1].word, w.word)
            ):
                break
            chunk.append(w)
            end_t = w.end_time
            j += 1
            if (
                prefer_break_on_punctuation
                and len(chunk) >= min_words_before_break
                and chunk[-1].word.rstrip("\"')]}").endswith(_PUNCTUATION_BREAK_CHARS)
            ):
                break
        chunks_out.append(chunk)
        i = j
    return chunks_out


def format_srt(lines: list[tuple[float, float, str]]) -> str:
    blocks: list[str] = []
    for idx, (start, end, text) in enumerate(lines, start=1):
        blocks.append(
            f"{idx}\n{_fmt_time(start)} --> {_fmt_time(end)}\n{text}\n"
        )
    return "\n".join(blocks)


def _fmt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
    if millis >= 1000:
        millis = 999
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# ---------------------------------------------------------------------------
# ASS / SubStation Alpha output (the format libass natively renders)
# ---------------------------------------------------------------------------


def _fmt_ass_time(seconds: float) -> str:
    """ASS time format: ``H:MM:SS.cs`` (centiseconds)."""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    whole = int(secs)
    cs = int(round((secs - whole) * 100))
    if cs >= 100:
        cs = 99
    return f"{hours:d}:{minutes:02d}:{whole:02d}.{cs:02d}"


def _escape_ass_text(text: str) -> str:
    """Escape characters that are significant to the ASS dialogue parser."""
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\N")
    )


def format_ass(
    lines: list[tuple[float, float, str]],
    *,
    play_res_x: int,
    play_res_y: int,
    font_size: int,
    margin_v: int,
    margin_h: int = 60,
    font_name: str = "Arial",
    render_theme: RenderTheme = RenderTheme.LEGACY,
) -> str:
    """Render captions as an ASS script whose PlayRes matches the output video.

    Why this exists: libass' font/margin scaling multiplies every pixel-ish
    value by ``video_height / PlayResY``. The default ``PlayResY=288`` blew
    ``FontSize=48`` up to ~320 output pixels and pushed ``MarginV`` to the
    middle of the frame. Pinning ``PlayResY`` to the actual output height
    makes that scale factor exactly 1.0, so ``font_size`` and ``margin_v``
    below are honest output pixel values.
    """

    if render_theme == RenderTheme.REFERENCE_LOWER_THIRD:
        style_line = (
            f"Style: Default,{font_name},{font_size},&H00FFFFFF,&H000000FF,"
            "&H00000000,&H00000000,-1,0,0,0,100,100,-1,0,1,3,0,2,"
            f"{margin_h},{margin_h},{margin_v},0\n"
        )
    else:
        style_line = (
            f"Style: Default,{font_name},{font_size},&H00FFFFFF,&H000000FF,"
            f"&H00000000,&H70000000,-1,0,0,0,100,100,0,0,4,0,0,2,"
            f"{margin_h},{margin_h},{margin_v},0\n"
        )

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
        + style_line +
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events = []
    for start, end, text in lines:
        events.append(
            f"Dialogue: 0,{_fmt_ass_time(start)},{_fmt_ass_time(end)},Default,,"
            f"0,0,0,,{_escape_ass_text(text)}"
        )
    return header + "\n".join(events) + ("\n" if events else "")
