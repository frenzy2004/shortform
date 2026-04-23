import json
from unittest.mock import patch

from humeo.ingest import (
    _normalize_elevenlabs_response,
    _merge_transcripts,
    _offset_transcript_timestamps,
    _plan_openai_chunk_ranges,
    _write_transcript,
    resolved_transcribe_settings,
    stage_local_video,
    transcript_cache_valid,
    transcribe_whisperx,
)


def test_plan_openai_chunk_ranges_single_chunk_when_under_limit():
    ranges = _plan_openai_chunk_ranges(duration_sec=600.0, file_size_bytes=10 * 1024 * 1024)
    assert ranges == [(0.0, 600.0)]


def test_plan_openai_chunk_ranges_splits_large_file():
    ranges = _plan_openai_chunk_ranges(duration_sec=3600.0, file_size_bytes=80 * 1024 * 1024)
    assert len(ranges) >= 2
    assert ranges[0][0] == 0.0
    total_duration = sum(duration for _, duration in ranges)
    assert abs(total_duration - 3600.0) < 0.01


def test_offset_transcript_timestamps_shifts_segments_and_words():
    transcript = {
        "language": "en",
        "segments": [
            {
                "start": 1.0,
                "end": 3.0,
                "text": "hello world",
                "words": [
                    {"word": "hello", "start": 1.0, "end": 1.5},
                    {"word": "world", "start": 1.5, "end": 2.0},
                ],
            }
        ],
    }

    shifted = _offset_transcript_timestamps(transcript, 120.0)
    segment = shifted["segments"][0]
    assert segment["start"] == 121.0
    assert segment["end"] == 123.0
    assert segment["words"][0]["start"] == 121.0
    assert segment["words"][1]["end"] == 122.0


def test_merge_transcripts_concatenates_segments():
    merged = _merge_transcripts(
        [
            {"language": "en", "segments": [{"start": 0.0, "end": 1.0, "text": "a", "words": []}]},
            {"language": "en", "segments": [{"start": 1.0, "end": 2.0, "text": "b", "words": []}]},
        ]
    )
    assert merged["language"] == "en"
    assert len(merged["segments"]) == 2


def test_transcribe_provider_openai_calls_openai_api(monkeypatch, tmp_path):
    """When HUMEO_TRANSCRIBE_PROVIDER=openai, do not require whisperx."""
    monkeypatch.setenv("HUMEO_TRANSCRIBE_PROVIDER", "openai")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    out = {"segments": [], "language": "en"}
    with patch("humeo.ingest._transcribe_openai_api", return_value=out) as m:
        r = transcribe_whisperx(audio, tmp_path)
    m.assert_called_once_with(audio)
    assert r == out
    assert (tmp_path / "transcript.json").read_text(encoding="utf-8").strip()


def test_stage_local_video_copies_source_and_records_marker(tmp_path):
    source = tmp_path / "downloads" / "episode.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"video-bytes")

    staged = stage_local_video(source, tmp_path / "work")

    assert staged == tmp_path / "work" / "source.mp4"
    assert staged.read_bytes() == b"video-bytes"
    marker = json.loads((tmp_path / "work" / "source.local.json").read_text(encoding="utf-8"))
    assert marker["local_source_path"] == str(source.resolve())


def test_normalize_elevenlabs_response_builds_word_segments():
    normalized = _normalize_elevenlabs_response(
        {
            "language_code": "en",
            "words": [
                {"text": "Hello", "start": 0.0, "end": 0.3, "type": "word"},
                {"text": "world", "start": 0.35, "end": 0.7, "type": "word"},
                {"text": "(laughter)", "start": 1.8, "end": 2.0, "type": "audio_event"},
                {"text": "again", "start": 2.1, "end": 2.5, "type": "word"},
            ],
        }
    )
    assert normalized["language"] == "en"
    assert len(normalized["segments"]) == 2
    assert normalized["segments"][0]["text"] == "Hello world"
    assert normalized["segments"][1]["words"][0]["word"] == "again"


def test_transcript_cache_tracks_provider_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("HUMEO_TRANSCRIBE_PROVIDER", "elevenlabs")
    monkeypatch.setenv("ELEVENLABS_NO_VERBATIM", "true")
    _write_transcript(tmp_path, {"segments": [], "language": "en"})
    assert transcript_cache_valid(tmp_path)

    monkeypatch.setenv("ELEVENLABS_NO_VERBATIM", "false")
    assert not transcript_cache_valid(tmp_path)


def test_resolved_transcribe_settings_defaults_to_elevenlabs(monkeypatch):
    monkeypatch.delenv("HUMEO_TRANSCRIBE_PROVIDER", raising=False)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    settings = resolved_transcribe_settings()
    assert settings["provider"] == "elevenlabs"
    assert settings["no_verbatim"] is True
