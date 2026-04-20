import json

from humeo.session_state import SESSION_STATE_FILENAME, load_state, save_state, source_key_for_url
from humeo_core.schemas import RatingFeedback, SessionState


def test_source_key_prefers_youtube_id():
    assert source_key_for_url("https://www.youtube.com/watch?v=PdVv_vLkUgk") == "youtube:PdVv_vLkUgk"


def test_source_key_falls_back_to_url():
    assert source_key_for_url("https://example.com/video") == "url:https://example.com/video"


def test_save_and_load_roundtrip(tmp_path):
    state = SessionState(
        source_key="youtube:PdVv_vLkUgk",
        iteration=2,
        steering_notes=["more emotional"],
        last_rating=RatingFeedback(rating=2, issues=["boring"]),
        last_selected_ids=["001", "003"],
    )

    save_state(tmp_path, state)
    loaded = load_state(tmp_path, "https://www.youtube.com/watch?v=PdVv_vLkUgk")

    assert loaded == state


def test_load_corrupt_json_warns_and_resets(tmp_path, caplog):
    path = tmp_path / SESSION_STATE_FILENAME
    path.write_text("{not json", encoding="utf-8")

    loaded = load_state(tmp_path, "https://www.youtube.com/watch?v=PdVv_vLkUgk")

    assert loaded.source_key == "youtube:PdVv_vLkUgk"
    assert loaded.iteration == 0
    assert "Starting fresh" in caplog.text


def test_save_creates_directory(tmp_path):
    nested = tmp_path / "nested" / "dir"
    save_state(nested, SessionState(source_key="youtube:PdVv_vLkUgk"))

    assert (nested / SESSION_STATE_FILENAME).is_file()


def test_mismatched_source_key_resets_state(tmp_path, caplog):
    path = tmp_path / SESSION_STATE_FILENAME
    path.write_text(
        json.dumps({"source_key": "youtube:oldvideoid1", "iteration": 3, "steering_notes": ["old"]}),
        encoding="utf-8",
    )

    loaded = load_state(tmp_path, "https://www.youtube.com/watch?v=PdVv_vLkUgk")

    assert loaded.source_key == "youtube:PdVv_vLkUgk"
    assert loaded.iteration == 0
    assert loaded.steering_notes == []
    assert "belongs to" in caplog.text
