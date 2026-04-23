"""Word-level subtitle alignment for the product pipeline."""

import pytest

from humeo.transcript_align import (
    clip_subtitle_words,
    clip_words_to_srt_lines,
    format_srt,
)
from humeo_core.schemas import Clip, TranscriptWord


def test_clip_subtitle_words_shifts_to_clip_local():
    transcript = {
        "segments": [
            {
                "start": 100.0,
                "end": 102.0,
                "text": "one two",
                "words": [
                    {"word": "one", "start": 100.0, "end": 100.5},
                    {"word": "two", "start": 100.6, "end": 101.2},
                ],
            }
        ]
    }
    clip = Clip(
        clip_id="1",
        topic="t",
        start_time_sec=100.0,
        end_time_sec=101.5,
        transcript="one two",
    )
    aligned = clip_subtitle_words(transcript, clip)
    assert len(aligned.words) == 2
    assert aligned.words[0].word == "one"
    assert aligned.words[0].start_time == 0.0
    assert aligned.words[1].start_time == pytest.approx(0.6)


def test_clip_words_to_srt_lines_groups():
    words = [
        TranscriptWord(word=str(i), start_time=i * 0.1, end_time=(i + 1) * 0.1)
        for i in range(10)
    ]
    lines = clip_words_to_srt_lines(words)
    assert len(lines) == 2
    assert lines[0][2].startswith("0 1 2 3 4 5 6 7")


def test_format_srt_roundtrip_single_line():
    s = format_srt([(0.0, 1.0, "hello")])
    assert "00:00:00,000 --> 00:00:01,000" in s
    assert "hello" in s


def test_clip_words_to_srt_lines_can_prefer_punctuation_breaks():
    words = [
        TranscriptWord(word="The", start_time=0.0, end_time=0.2),
        TranscriptWord(word="market", start_time=0.2, end_time=0.4),
        TranscriptWord(word="shrinks", start_time=0.4, end_time=0.6),
        TranscriptWord(word="fast.", start_time=0.6, end_time=0.9),
        TranscriptWord(word="Robotaxis", start_time=1.0, end_time=1.2),
        TranscriptWord(word="change", start_time=1.2, end_time=1.4),
        TranscriptWord(word="everything", start_time=1.4, end_time=1.7),
    ]

    lines = clip_words_to_srt_lines(
        words,
        max_words_per_cue=7,
        max_cue_sec=3.0,
        prefer_break_on_punctuation=True,
        min_words_before_break=4,
    )

    assert len(lines) == 2
    assert lines[0][2] == "The market shrinks fast."
    assert lines[1][2] == "Robotaxis change everything"


def test_clip_words_to_srt_lines_breaks_on_likely_sentence_restart():
    words = [
        TranscriptWord(word="delivery", start_time=0.0, end_time=0.2),
        TranscriptWord(word="costs", start_time=0.2, end_time=0.4),
        TranscriptWord(word="by", start_time=0.4, end_time=0.6),
        TranscriptWord(word="60", start_time=0.6, end_time=0.8),
        TranscriptWord(word="Drones", start_time=0.9, end_time=1.1),
        TranscriptWord(word="and", start_time=1.1, end_time=1.3),
        TranscriptWord(word="robots", start_time=1.3, end_time=1.5),
        TranscriptWord(word="Now", start_time=1.6, end_time=1.8),
        TranscriptWord(word="they", start_time=1.8, end_time=2.0),
        TranscriptWord(word="scale", start_time=2.0, end_time=2.2),
    ]

    lines = clip_words_to_srt_lines(
        words,
        max_words_per_cue=10,
        max_cue_sec=4.0,
        prefer_break_on_punctuation=True,
        min_words_before_break=5,
    )

    assert [line[2] for line in lines] == [
        "delivery costs by 60",
        "Drones and robots",
        "Now they scale",
    ]


def test_clip_words_to_srt_lines_breaks_single_word_tail_before_sentence_restart():
    words = [
        TranscriptWord(word="dramatically", start_time=0.0, end_time=0.6),
        TranscriptWord(word="Did", start_time=0.7, end_time=0.9),
        TranscriptWord(word="you", start_time=0.9, end_time=1.1),
        TranscriptWord(word="know", start_time=1.1, end_time=1.3),
    ]

    lines = clip_words_to_srt_lines(
        words,
        max_words_per_cue=10,
        max_cue_sec=4.0,
        prefer_break_on_punctuation=True,
        min_words_before_break=5,
    )

    assert [line[2] for line in lines] == [
        "dramatically",
        "Did you know",
    ]
