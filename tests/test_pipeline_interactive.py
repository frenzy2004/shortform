from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

from humeo.config import PipelineConfig
from humeo.pipeline import run_pipeline
from humeo.session_state import save_state
from humeo_core.schemas import ApprovalResult, Clip, LayoutInstruction, LayoutKind, RatingFeedback, Scene, SessionState


def _clip(clip_id: str = "001", *, start: float = 0.0, topic: str = "topic") -> Clip:
    return Clip.model_validate(
        {
            "clip_id": clip_id,
            "topic": topic,
            "start_time_sec": start,
            "end_time_sec": start + 60.0,
            "virality_score": 0.9,
            "transcript": f"{topic} transcript",
        }
    )


def _prepare_inputs(tmp_path: Path) -> None:
    (tmp_path / "source.mp4").write_text("video", encoding="utf-8")
    transcript = {
        "segments": [
            {"start": 0.0, "end": 5.0, "text": "hello"},
            {"start": 5.0, "end": 10.0, "text": "world"},
        ]
    }
    (tmp_path / "transcript.json").write_text(json.dumps(transcript), encoding="utf-8")


def _config(
    tmp_path: Path,
    *,
    interactive: bool,
    youtube_url: str = "https://www.youtube.com/watch?v=PdVv_vLkUgk",
) -> PipelineConfig:
    return PipelineConfig(
        youtube_url=youtube_url,
        work_dir=tmp_path,
        output_dir=tmp_path / "output",
        interactive=interactive,
        gemini_model="gemini-test",
    )


def _patch_pipeline(monkeypatch, tmp_path: Path, clips: list[Clip]) -> list[list[str]]:
    import humeo.pipeline as pipeline_mod

    _prepare_inputs(tmp_path)
    call_notes: list[list[str]] = []

    def fake_select_clips(*_args, **kwargs):
        call_notes.append(list(kwargs.get("steering_notes") or []))
        return list(clips), '{"clips": []}'

    monkeypatch.setattr(pipeline_mod, "select_clips", fake_select_clips)
    monkeypatch.setattr(pipeline_mod, "run_hook_detection_stage", lambda *_args, **_kwargs: list(clips))
    monkeypatch.setattr(pipeline_mod, "run_content_pruning_stage", lambda *_args, **_kwargs: list(clips))
    monkeypatch.setattr(
        pipeline_mod,
        "extract_keyframes",
        lambda *_args, **_kwargs: [
            Scene(
                scene_id=clip.clip_id,
                start_time=clip.start_time_sec,
                end_time=clip.end_time_sec,
                keyframe_path="frame.jpg",
            )
            for clip in clips
        ],
    )
    monkeypatch.setattr(
        pipeline_mod,
        "run_layout_vision_stage",
        lambda *_args, **_kwargs: {
            clip.clip_id: LayoutInstruction(clip_id=clip.clip_id, layout=LayoutKind.SIT_CENTER)
            for clip in clips
        },
    )
    monkeypatch.setattr(pipeline_mod, "generate_ass", lambda *_args, **_kwargs: tmp_path / "subtitles" / "clip.ass")

    def fake_reframe(*, output_path, **_kwargs):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("rendered", encoding="utf-8")

    monkeypatch.setattr(pipeline_mod, "reframe_clip_ffmpeg", fake_reframe)
    return call_notes


def test_pipeline_noninteractive_does_not_prompt(monkeypatch, tmp_path):
    clips = [_clip("001")]
    _patch_pipeline(monkeypatch, tmp_path, clips)
    approve = MagicMock(side_effect=AssertionError("approve_clips should not be called"))
    rate = MagicMock(side_effect=AssertionError("rate_output should not be called"))
    monkeypatch.setattr("humeo.pipeline.interactive.approve_clips", approve)
    monkeypatch.setattr("humeo.pipeline.interactive.rate_output", rate)

    outputs = run_pipeline(_config(tmp_path, interactive=False))

    assert len(outputs) == 1
    approve.assert_not_called()
    rate.assert_not_called()


def test_pipeline_interactive_accept_all_and_rating_three(monkeypatch, tmp_path):
    clips = [_clip("001"), _clip("002", start=70.0)]
    _patch_pipeline(monkeypatch, tmp_path, clips)
    monkeypatch.setattr(
        "humeo.pipeline.interactive.approve_clips",
        lambda _clips: ApprovalResult(action="accept_all", selected_ids=["001", "002"]),
    )
    monkeypatch.setattr(
        "humeo.pipeline.interactive.rate_output",
        lambda _outputs: RatingFeedback(rating=3),
    )

    outputs = run_pipeline(_config(tmp_path, interactive=True))
    state = json.loads((tmp_path / "session_state.json").read_text(encoding="utf-8"))

    assert len(outputs) == 2
    assert state["last_selected_ids"] == ["001", "002"]
    assert state["last_rating"]["rating"] == 3


def test_pipeline_refine_increments_iteration_and_adds_steering(monkeypatch, tmp_path):
    clips = [_clip("001")]
    call_notes = _patch_pipeline(monkeypatch, tmp_path, clips)
    approvals = iter(
        [
            ApprovalResult(action="refine", steering_note="more emotional clips please"),
            ApprovalResult(action="accept_all", selected_ids=["001"]),
        ]
    )
    monkeypatch.setattr("humeo.pipeline.interactive.approve_clips", lambda _clips: next(approvals))
    monkeypatch.setattr("humeo.pipeline.interactive.rate_output", lambda _outputs: RatingFeedback(rating=3))

    outputs = run_pipeline(_config(tmp_path, interactive=True))
    state = json.loads((tmp_path / "session_state.json").read_text(encoding="utf-8"))

    assert len(outputs) == 1
    assert call_notes == [[], ["more emotional clips please"]]
    assert state["iteration"] == 1
    assert state["steering_notes"] == ["more emotional clips please"]


def test_pipeline_iteration_cap_stops_recursion(monkeypatch, tmp_path):
    clips = [_clip("001")]
    call_notes = _patch_pipeline(monkeypatch, tmp_path, clips)
    monkeypatch.setattr(
        "humeo.pipeline.interactive.approve_clips",
        lambda _clips: ApprovalResult(action="refine", steering_note="keep trying"),
    )
    monkeypatch.setattr("humeo.pipeline.interactive.rate_output", lambda _outputs: RatingFeedback(rating=3))

    outputs = run_pipeline(_config(tmp_path, interactive=True, youtube_url="https://www.youtube.com/watch?v=PdVv_vLkUgk"))
    state = json.loads((tmp_path / "session_state.json").read_text(encoding="utf-8"))

    assert len(outputs) == 1
    assert len(call_notes) == 5
    assert state["iteration"] == 5
    assert state["steering_notes"] == ["keep trying"] * 5


def test_pipeline_restart_with_same_source_reenters_gate_one(monkeypatch, tmp_path):
    clips = [_clip("001")]
    call_notes = _patch_pipeline(monkeypatch, tmp_path, clips)
    save_state(
        tmp_path,
        SessionState(
            source_key="youtube:PdVv_vLkUgk",
            iteration=2,
            steering_notes=["be punchier"],
        ),
    )
    approve = MagicMock(return_value=ApprovalResult(action="accept_all", selected_ids=["001"]))
    rate = MagicMock(return_value=RatingFeedback(rating=3))
    monkeypatch.setattr("humeo.pipeline.interactive.approve_clips", approve)
    monkeypatch.setattr("humeo.pipeline.interactive.rate_output", rate)

    outputs = run_pipeline(_config(tmp_path, interactive=True))

    assert len(outputs) == 1
    approve.assert_called_once()
    rate.assert_called_once()
    assert call_notes == [["be punchier"]]


def test_pipeline_different_source_in_same_workdir_ignores_old_state(monkeypatch, tmp_path):
    clips = [_clip("001")]
    call_notes = _patch_pipeline(monkeypatch, tmp_path, clips)
    save_state(
        tmp_path,
        SessionState(
            source_key="youtube:oldvideoid1",
            iteration=2,
            steering_notes=["old note"],
        ),
    )
    monkeypatch.setattr(
        "humeo.pipeline.interactive.approve_clips",
        lambda _clips: ApprovalResult(action="accept_all", selected_ids=["001"]),
    )
    monkeypatch.setattr("humeo.pipeline.interactive.rate_output", lambda _outputs: RatingFeedback(rating=3))

    outputs = run_pipeline(
        _config(tmp_path, interactive=True, youtube_url="https://www.youtube.com/watch?v=PdVv_vLkUgk")
    )
    state = json.loads((tmp_path / "session_state.json").read_text(encoding="utf-8"))

    assert len(outputs) == 1
    assert call_notes == [[]]
    assert state["source_key"] == "youtube:PdVv_vLkUgk"
    assert state["steering_notes"] == []


def test_pipeline_local_source_stages_video_without_downloading(monkeypatch, tmp_path):
    import humeo.pipeline as pipeline_mod

    local_source = tmp_path / "downloads" / "episode.mp4"
    local_source.parent.mkdir(parents=True, exist_ok=True)
    local_source.write_bytes(b"video")
    transcript = {
        "segments": [
            {"start": 0.0, "end": 5.0, "text": "hello"},
            {"start": 5.0, "end": 10.0, "text": "world"},
        ]
    }
    (tmp_path / "transcript.json").write_text(json.dumps(transcript), encoding="utf-8")

    clips = [_clip("001")]
    monkeypatch.setattr(
        pipeline_mod,
        "download_video",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("download_video should not be called")),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "select_clips",
        lambda *_args, **_kwargs: (list(clips), '{"clips": []}'),
    )
    monkeypatch.setattr(pipeline_mod, "extract_audio", lambda *_args, **_kwargs: tmp_path / "source_audio.wav")
    monkeypatch.setattr(pipeline_mod, "transcribe_whisperx", lambda *_args, **_kwargs: transcript)
    monkeypatch.setattr(pipeline_mod, "run_hook_detection_stage", lambda *_args, **_kwargs: list(clips))
    monkeypatch.setattr(pipeline_mod, "run_content_pruning_stage", lambda *_args, **_kwargs: list(clips))
    monkeypatch.setattr(
        pipeline_mod,
        "extract_keyframes",
        lambda *_args, **_kwargs: [
            Scene(scene_id="001", start_time=0.0, end_time=60.0, keyframe_path="frame.jpg")
        ],
    )
    monkeypatch.setattr(
        pipeline_mod,
        "run_layout_vision_stage",
        lambda *_args, **_kwargs: {
            "001": LayoutInstruction(clip_id="001", layout=LayoutKind.SIT_CENTER)
        },
    )
    monkeypatch.setattr(pipeline_mod, "generate_ass", lambda *_args, **_kwargs: tmp_path / "subtitles" / "clip.ass")

    def fake_reframe(*, output_path, **_kwargs):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("rendered", encoding="utf-8")

    monkeypatch.setattr(pipeline_mod, "reframe_clip_ffmpeg", fake_reframe)

    outputs = run_pipeline(
        PipelineConfig(
            youtube_url=str(local_source),
            work_dir=tmp_path,
            output_dir=tmp_path / "output",
            interactive=False,
            gemini_model="gemini-test",
        )
    )

    assert len(outputs) == 1
    assert (tmp_path / "source.mp4").read_bytes() == b"video"
    assert json.loads((tmp_path / "source.local.json").read_text(encoding="utf-8"))["local_source_path"] == str(
        local_source.resolve()
    )


def test_pipeline_guardrail_drops_render_invalid_clips(monkeypatch, tmp_path, caplog):
    import humeo.pipeline as pipeline_mod

    valid = _clip("001")
    invalid_after_trim = Clip.model_validate(
        {
            "clip_id": "002",
            "topic": "too short after trim",
            "start_time_sec": 100.0,
            "end_time_sec": 160.0,
            "virality_score": 0.95,
            "transcript": "too short after trim transcript",
            "trim_start_sec": 25.0,
            "trim_end_sec": 25.0,
        }
    )
    clips = [valid, invalid_after_trim]
    _patch_pipeline(monkeypatch, tmp_path, clips)

    approve = MagicMock(return_value=ApprovalResult(action="accept_all", selected_ids=["001"]))
    rate = MagicMock(return_value=RatingFeedback(rating=3))
    monkeypatch.setattr("humeo.pipeline.interactive.approve_clips", approve)
    monkeypatch.setattr("humeo.pipeline.interactive.rate_output", rate)

    with caplog.at_level(logging.WARNING):
        outputs = run_pipeline(_config(tmp_path, interactive=True))

    assert len(outputs) == 1
    assert approve.call_count == 1
    approved_clips = approve.call_args.args[0]
    assert [clip.clip_id for clip in approved_clips] == ["001"]
    assert "Stage 2.5 guardrail: dropping clip 002" in caplog.text
