"""
Step 1 - Ingestion: Download video and generate word-level transcript.

Responsibilities:
  - Download source video from YouTube using yt-dlp.
  - Extract audio track for transcription.
  - Generate word-level timestamped transcript.
"""

import json
import logging
import os
import shutil
import subprocess
from math import ceil
from pathlib import Path

import httpx

from humeo.video_cache import local_source_matches, write_local_source_info

logger = logging.getLogger(__name__)

OPENAI_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
OPENAI_TARGET_UPLOAD_BYTES = 20 * 1024 * 1024
OPENAI_MIN_CHUNK_SEC = 300.0
ELEVENLABS_TRANSCRIBE_URL = "https://api.elevenlabs.io/v1/speech-to-text"
TRANSCRIPT_META_FILENAME = "transcript.meta.json"
ELEVENLABS_SCRIBE_MODEL = "scribe_v2"
_ELEVENLABS_SEGMENT_MAX_GAP_SEC = 0.65
_ELEVENLABS_SEGMENT_MAX_DURATION_SEC = 6.0
_ELEVENLABS_SEGMENT_MAX_WORDS = 18


def stage_local_video(source: str | Path, output_dir: Path) -> Path:
    """
    Copy a local source video into ``output_dir/source.mp4`` for cacheable reruns.
    """
    source_path = Path(source).expanduser().resolve(strict=False)
    if not source_path.is_file():
        raise FileNotFoundError(f"Local source video does not exist: {source_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    staged_path = output_dir / "source.mp4"
    staged_resolved = staged_path.resolve(strict=False)

    if source_path == staged_resolved:
        logger.info("Using local source video in place: %s", source_path)
        write_local_source_info(output_dir, source_path)
        return staged_path

    if staged_path.exists() and local_source_matches(output_dir, str(source_path)):
        logger.info("Local source already staged at: %s", staged_path)
        return staged_path

    if source_path.suffix.lower() != ".mp4":
        logger.warning(
            "Local source uses %s; staging it as source.mp4 anyway.",
            source_path.suffix or "<no extension>",
        )

    action = "Replacing" if staged_path.exists() else "Staging"
    logger.info("%s local video: %s -> %s", action, source_path, staged_path)
    shutil.copy2(source_path, staged_path)
    write_local_source_info(output_dir, source_path)
    return staged_path


def download_video(youtube_url: str, output_dir: Path) -> Path:
    """
    Download the best quality video+audio from YouTube.

    Returns the path to the downloaded MP4 file.
    """
    output_template = str(output_dir / "source.%(ext)s")
    cmd = [
        "yt-dlp",
        "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--output", output_template,
        "--no-playlist",
        "--write-info-json",
        "--quiet",
        youtube_url,
    ]

    logger.info("Downloading video: %s", youtube_url)
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if result.stderr:
        logger.warning(result.stderr.strip())

    # yt-dlp should produce source.mp4
    video_path = output_dir / "source.mp4"
    if not video_path.exists():
        # Fallback: find any mp4 in the output dir
        mp4_files = list(output_dir.glob("source.*"))
        if mp4_files:
            video_path = mp4_files[0]
        else:
            raise FileNotFoundError(f"Download failed - no output found in {output_dir}")

    logger.info("Downloaded to: %s", video_path)
    return video_path


def extract_audio(video_path: Path, output_dir: Path) -> Path:
    """
    Extract audio track from video as WAV (required by most ASR models).
    """
    audio_path = output_dir / "source_audio.wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                        # no video
        "-acodec", "pcm_s16le",       # raw PCM
        "-ar", "16000",               # 16kHz sample rate (standard for ASR)
        "-ac", "1",                   # mono
        str(audio_path),
    ]

    logger.info("Extracting audio to: %s", audio_path)
    subprocess.run(cmd, check=True, capture_output=True)
    return audio_path


def _resolve_elevenlabs_api_key() -> str:
    key = (os.environ.get("ELEVENLABS_API_KEY") or "").strip()
    if key:
        return key
    raise ValueError("Set ELEVENLABS_API_KEY to use ElevenLabs Scribe v2 transcription.")


def _elevenlabs_no_verbatim_enabled() -> bool:
    raw = (os.environ.get("ELEVENLABS_NO_VERBATIM") or "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def resolved_transcribe_settings() -> dict[str, object]:
    provider = (os.environ.get("HUMEO_TRANSCRIBE_PROVIDER") or "elevenlabs").strip().lower()
    if provider in ("", "auto"):
        if (os.environ.get("ELEVENLABS_API_KEY") or "").strip():
            provider = "elevenlabs"
        else:
            provider = "openai"

    if provider in ("api",):
        provider = "openai"
    if provider in ("local",):
        provider = "whisperx"

    settings: dict[str, object] = {"provider": provider}
    if provider == "elevenlabs":
        settings.update(
            {
                "model_id": ELEVENLABS_SCRIBE_MODEL,
                "no_verbatim": _elevenlabs_no_verbatim_enabled(),
            }
        )
    return settings


def transcript_cache_valid(output_dir: Path) -> bool:
    transcript_path = output_dir / "transcript.json"
    meta_path = output_dir / TRANSCRIPT_META_FILENAME
    if not transcript_path.is_file() or not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return meta == resolved_transcribe_settings()


def _write_transcript(output_dir: Path, transcript: dict) -> None:
    transcript_path = output_dir / "transcript.json"
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    with open(output_dir / TRANSCRIPT_META_FILENAME, "w", encoding="utf-8") as f:
        json.dump(resolved_transcribe_settings(), f, indent=2, ensure_ascii=False)
        f.write("\n")


def _normalize_elevenlabs_word(raw_word: dict) -> dict | None:
    if not isinstance(raw_word, dict):
        return None
    if str(raw_word.get("type", "word")).strip().lower() not in {"word", ""}:
        return None
    text = str(raw_word.get("text", raw_word.get("word", ""))).strip()
    if not text:
        return None
    try:
        start = float(raw_word["start"])
        end = float(raw_word["end"])
    except (KeyError, TypeError, ValueError):
        return None
    if end <= start:
        return None
    return {"word": text, "start": start, "end": end}


def _segment_words_into_transcript(words: list[dict], *, language: str) -> dict:
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
                gap >= _ELEVENLABS_SEGMENT_MAX_GAP_SEC
                or dur >= _ELEVENLABS_SEGMENT_MAX_DURATION_SEC
                or len(chunk) >= _ELEVENLABS_SEGMENT_MAX_WORDS
            ):
                flush()
        chunk.append(word)
    flush()
    return {"segments": segments, "language": language}


def _normalize_elevenlabs_response(data: dict) -> dict:
    words = [
        word
        for raw_word in data.get("words", []) or []
        if (word := _normalize_elevenlabs_word(raw_word)) is not None
    ]
    language = str(
        data.get("language_code") or data.get("language") or "en"
    ).strip() or "en"
    return _segment_words_into_transcript(words, language=language)


def _transcribe_elevenlabs_scribe(audio_path: Path) -> dict:
    headers = {"xi-api-key": _resolve_elevenlabs_api_key()}
    form = {
        "model_id": ELEVENLABS_SCRIBE_MODEL,
        "timestamps_granularity": "word",
        "diarize": "false",
        "tag_audio_events": "false",
        "file_format": "pcm_s16le_16",
        "no_verbatim": "true" if _elevenlabs_no_verbatim_enabled() else "false",
    }
    with audio_path.open("rb") as handle:
        files = {"file": (audio_path.name, handle, "audio/wav")}
        response = httpx.post(
            ELEVENLABS_TRANSCRIBE_URL,
            headers=headers,
            data=form,
            files=files,
            timeout=600.0,
        )
    response.raise_for_status()
    return _normalize_elevenlabs_response(response.json())


def _transcribe_whisperx_local(audio_path: Path) -> dict:
    """Word-level transcript via WhisperX (local). Raises ImportError if not installed."""
    import whisperx

    logger.info("Transcribing with WhisperX...")
    device = "cpu"  # Use "cuda" if GPU available
    model = whisperx.load_model("base", device=device, compute_type="int8")
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=16)

    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, device,
        return_char_alignments=False,
    )

    logger.info("Transcription complete: %d segments", len(result["segments"]))
    return result


def transcribe_whisperx(audio_path: Path, output_dir: Path) -> dict:
    """
    Transcribe audio for word-level timestamps.

    Provider is controlled by **HUMEO_TRANSCRIBE_PROVIDER** (default ``auto``):

    - ``auto`` — WhisperX if installed, else OpenAI Whisper API.
    - ``openai`` / ``api`` — OpenAI Whisper API (uses ``OPENAI_API_KEY``), even when WhisperX is installed.
    - ``whisperx`` / ``local`` — WhisperX only; fails clearly if not installed.

    The result is written to ``output_dir / "transcript.json"``. Re-runs with an
    existing transcript are skipped by the pipeline before this function runs.
    """
    settings = resolved_transcribe_settings()
    provider = str(settings["provider"])

    if provider == "elevenlabs":
        logger.info(
            "Transcribing with ElevenLabs Scribe v2 (no_verbatim=%s).",
            bool(settings.get("no_verbatim", False)),
        )
        result = _transcribe_elevenlabs_scribe(audio_path)
    elif provider == "openai":
        logger.info(
            "Transcribing with OpenAI Whisper API (HUMEO_TRANSCRIBE_PROVIDER=%s).",
            provider,
        )
        result = _transcribe_openai_api(audio_path)
    elif provider == "whisperx":
        try:
            result = _transcribe_whisperx_local(audio_path)
        except ImportError as e:
            raise RuntimeError(
                "WhisperX requested (HUMEO_TRANSCRIBE_PROVIDER=whisperx) but whisperx is not installed. "
                "Install with: uv sync --extra whisper"
            ) from e
    else:
        raise RuntimeError(
            f"Unknown HUMEO_TRANSCRIBE_PROVIDER={provider!r}. "
            "Use elevenlabs, openai, or whisperx."
        )

    _write_transcript(output_dir, result)

    return result


def _transcribe_openai_api(audio_path: Path) -> dict:
    """
    Fallback transcription using OpenAI's Whisper API.
    Requires OPENAI_API_KEY environment variable.
    """
    from openai import OpenAI

    client = OpenAI()

    work_dir = audio_path.parent / "openai_transcribe"
    work_dir.mkdir(parents=True, exist_ok=True)
    duration_sec = _probe_media_duration(audio_path)
    chunk_ranges = _plan_openai_chunk_ranges(
        duration_sec=duration_sec,
        file_size_bytes=audio_path.stat().st_size,
    )

    if len(chunk_ranges) == 1:
        return _transcribe_openai_file(client, audio_path)

    logger.info("Audio exceeds OpenAI upload limit; transcribing in %d chunks.", len(chunk_ranges))
    chunk_transcripts: list[dict] = []
    for idx, (offset_sec, chunk_duration_sec) in enumerate(chunk_ranges, start=1):
        chunk_path = work_dir / f"{audio_path.stem}_part_{idx:03d}.wav"
        if not chunk_path.exists():
            _extract_openai_audio_chunk(
                input_path=audio_path,
                output_path=chunk_path,
                offset_sec=offset_sec,
                duration_sec=chunk_duration_sec,
            )
        logger.info(
            "Transcribing chunk %d/%d (%.1fs-%.1fs)",
            idx,
            len(chunk_ranges),
            offset_sec,
            offset_sec + chunk_duration_sec,
        )
        chunk_transcript = _transcribe_openai_file(client, chunk_path)
        chunk_transcripts.append(_offset_transcript_timestamps(chunk_transcript, offset_sec))

    return _merge_transcripts(chunk_transcripts)


def _extract_openai_audio_chunk(
    input_path: Path,
    output_path: Path,
    offset_sec: float,
    duration_sec: float,
) -> Path:
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{offset_sec:.3f}",
        "-t",
        f"{duration_sec:.3f}",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def _probe_media_duration(media_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(media_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def _plan_openai_chunk_ranges(
    *,
    duration_sec: float,
    file_size_bytes: int,
    max_upload_bytes: int = OPENAI_MAX_UPLOAD_BYTES,
    target_upload_bytes: int = OPENAI_TARGET_UPLOAD_BYTES,
) -> list[tuple[float, float]]:
    if file_size_bytes <= max_upload_bytes:
        return [(0.0, duration_sec)]

    chunk_sec = max(
        OPENAI_MIN_CHUNK_SEC,
        duration_sec * (target_upload_bytes / file_size_bytes),
    )
    chunk_count = max(2, ceil(duration_sec / chunk_sec))
    exact_chunk_sec = duration_sec / chunk_count

    ranges: list[tuple[float, float]] = []
    for idx in range(chunk_count):
        start = idx * exact_chunk_sec
        end = min(duration_sec, (idx + 1) * exact_chunk_sec)
        ranges.append((round(start, 3), round(end - start, 3)))
    return ranges


def _transcribe_openai_file(client, audio_path: Path) -> dict:
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )
    return _normalize_openai_response(response)


def _normalize_openai_response(response: object) -> dict:
    data = response.model_dump() if hasattr(response, "model_dump") else response
    if not isinstance(data, dict):
        raise TypeError(f"Unexpected transcription payload type: {type(data)!r}")

    top_words = [_normalize_word(word) for word in data.get("words", []) or []]
    segments: list[dict] = []
    word_index = 0

    for raw_segment in data.get("segments", []) or []:
        segment = raw_segment.model_dump() if hasattr(raw_segment, "model_dump") else raw_segment
        if not isinstance(segment, dict):
            continue

        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        text = str(segment.get("text", "")).strip()

        segment_words = [_normalize_word(word) for word in segment.get("words", []) or []]
        if not segment_words and top_words:
            while word_index < len(top_words) and top_words[word_index]["end"] <= start:
                word_index += 1

            probe_index = word_index
            while probe_index < len(top_words) and top_words[probe_index]["start"] < end:
                word = top_words[probe_index]
                if word["end"] > start:
                    segment_words.append(word)
                probe_index += 1
            word_index = probe_index

        segments.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "words": segment_words,
            }
        )

    if not segments and top_words:
        segments.append(
            {
                "start": top_words[0]["start"],
                "end": top_words[-1]["end"],
                "text": " ".join(word["word"] for word in top_words).strip(),
                "words": top_words,
            }
        )

    return {
        "segments": segments,
        "language": str(data.get("language", "en") or "en"),
    }


def _normalize_word(raw_word: object) -> dict:
    word = raw_word.model_dump() if hasattr(raw_word, "model_dump") else raw_word
    if not isinstance(word, dict):
        return {"word": "", "start": 0.0, "end": 0.0}
    return {
        "word": str(word.get("word", "")).strip(),
        "start": float(word.get("start", 0.0)),
        "end": float(word.get("end", 0.0)),
    }


def _offset_transcript_timestamps(transcript: dict, offset_sec: float) -> dict:
    shifted_segments = []
    for segment in transcript.get("segments", []):
        shifted_segments.append(
            {
                "start": float(segment["start"]) + offset_sec,
                "end": float(segment["end"]) + offset_sec,
                "text": segment["text"],
                "words": [
                    {
                        "word": word["word"],
                        "start": float(word["start"]) + offset_sec,
                        "end": float(word["end"]) + offset_sec,
                    }
                    for word in segment.get("words", [])
                ],
            }
        )
    return {
        "segments": shifted_segments,
        "language": transcript.get("language", "en"),
    }


def _merge_transcripts(transcripts: list[dict]) -> dict:
    merged_segments = []
    language = "en"
    for transcript in transcripts:
        merged_segments.extend(transcript.get("segments", []))
        if transcript.get("language"):
            language = transcript["language"]
    return {
        "segments": merged_segments,
        "language": language,
    }
