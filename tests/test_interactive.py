from pathlib import Path

from humeo.interactive import approve_clips, rate_output
from humeo_core.schemas import Clip


def _clip(clip_id: str, *, score: float = 0.9, topic: str = "topic", transcript: str = "hello") -> Clip:
    return Clip.model_validate(
        {
            "clip_id": clip_id,
            "topic": topic,
            "start_time_sec": 0.0,
            "end_time_sec": 60.0,
            "virality_score": score,
            "transcript": transcript,
        }
    )


def test_approve_clips_accepts_numeric_selection(monkeypatch):
    inputs = iter(["3,1,5"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
    clips = [_clip(f"{i:03d}") for i in range(1, 6)]

    result = approve_clips(clips)

    assert result.action == "proceed"
    assert result.selected_ids == ["003", "001", "005"]


def test_approve_clips_accepts_zero_padded_selection(monkeypatch):
    inputs = iter(["003,001,005"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
    clips = [_clip(f"{i:03d}") for i in range(1, 6)]

    result = approve_clips(clips)

    assert result.action == "proceed"
    assert result.selected_ids == ["003", "001", "005"]


def test_approve_clips_invalid_id_reprompts(monkeypatch, capsys):
    inputs = iter(["9", "1,2"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
    clips = [_clip("001"), _clip("002")]

    result = approve_clips(clips)

    assert result.action == "proceed"
    assert result.selected_ids == ["001", "002"]
    assert "Unknown clip selection" in capsys.readouterr().out


def test_approve_clips_all(monkeypatch):
    inputs = iter(["all"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
    clips = [_clip("001"), _clip("002")]

    result = approve_clips(clips)

    assert result.action == "accept_all"
    assert result.selected_ids == ["001", "002"]


def test_approve_clips_refine(monkeypatch):
    inputs = iter(["refine my note"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    result = approve_clips([_clip("001")])

    assert result.action == "refine"
    assert result.steering_note == "my note"


def test_approve_clips_quit(monkeypatch):
    inputs = iter(["quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    result = approve_clips([_clip("001")])

    assert result.action == "quit"


def test_rate_output_rating_three(monkeypatch):
    inputs = iter(["3"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    result = rate_output([Path("output/short_001.mp4")])

    assert result.rating == 3
    assert result.issues == []


def test_rate_output_rating_one_with_issues(monkeypatch):
    inputs = iter(["1", "a c"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    result = rate_output([Path("output/short_001.mp4")])

    assert result.rating == 1
    assert result.issues == ["wrong_moments", "boring"]


def test_rate_output_rating_two_with_other(monkeypatch):
    inputs = iter(["2", "g", "needs more context"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    result = rate_output([Path("output/short_001.mp4")])

    assert result.rating == 2
    assert result.issues == ["other"]
    assert result.free_text == "needs more context"


def test_rate_output_rating_two_with_empty_issues(monkeypatch):
    inputs = iter(["2", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    result = rate_output([Path("output/short_001.mp4")])

    assert result.rating == 2
    assert result.issues == []
    assert result.free_text is None
