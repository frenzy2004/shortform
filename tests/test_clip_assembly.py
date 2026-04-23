from humeo.clip_assembly import build_assembled_transcript, derive_render_spans
from humeo_core.schemas import Clip


def _clip() -> Clip:
    return Clip(
        clip_id="001",
        topic="topic",
        start_time_sec=10.0,
        end_time_sec=20.0,
        trim_start_sec=1.0,
        trim_end_sec=1.0,
    )


def test_derive_render_spans_breaks_on_internal_silence():
    transcript = {
        "language": "en",
        "segments": [
            {
                "start": 10.9,
                "end": 12.0,
                "text": "first part",
                "words": [
                    {"word": "first", "start": 10.9, "end": 11.2},
                    {"word": "part", "start": 11.3, "end": 12.0},
                ],
            },
            {
                "start": 14.0,
                "end": 16.0,
                "text": "second part",
                "words": [
                    {"word": "second", "start": 14.0, "end": 14.5},
                    {"word": "part", "start": 14.6, "end": 16.0},
                ],
            },
        ],
    }

    spans = derive_render_spans(_clip(), transcript)

    assert len(spans) == 2
    assert spans[0].start_time_sec >= 10.95
    assert spans[1].start_time_sec >= 13.9


def test_build_assembled_transcript_rebases_times_after_hard_cuts():
    transcript = {
        "language": "en",
        "segments": [
            {
                "start": 11.0,
                "end": 12.0,
                "text": "first part",
                "words": [
                    {"word": "first", "start": 11.0, "end": 11.2},
                    {"word": "part", "start": 11.3, "end": 12.0},
                ],
            },
            {
                "start": 14.0,
                "end": 15.0,
                "text": "second part",
                "words": [
                    {"word": "second", "start": 14.0, "end": 14.4},
                    {"word": "part", "start": 14.5, "end": 15.0},
                ],
            },
        ],
    }

    assembled = build_assembled_transcript(_clip(), transcript)

    words = [word for seg in assembled["segments"] for word in seg["words"]]
    assert words[0]["start"] == 0.0
    assert words[-1]["end"] < 3.5
