"""Tests for product clip selection (Gemini API key and google-genai client)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_resolve_gemini_api_key_prefers_google_over_gemini(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "from-google")
    monkeypatch.setenv("GEMINI_API_KEY", "from-gemini")
    from humeo.env import resolve_gemini_api_key

    assert resolve_gemini_api_key() == "from-google"


def test_resolve_gemini_api_key_falls_back_to_gemini(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "only-gemini")
    from humeo.env import resolve_gemini_api_key

    assert resolve_gemini_api_key() == "only-gemini"


def test_resolve_gemini_api_key_strips_whitespace(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "  key  ")
    from humeo.env import resolve_gemini_api_key

    assert resolve_gemini_api_key() == "key"


def test_resolve_gemini_api_key_missing_raises(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from humeo.env import resolve_gemini_api_key

    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        resolve_gemini_api_key()


def test_resolve_llm_provider_prefers_google_over_openrouter(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "from-google")
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-openrouter")
    from humeo.env import resolve_llm_provider

    assert resolve_llm_provider() == "google"


def test_resolve_llm_provider_falls_back_to_openrouter(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    from humeo.env import resolve_llm_provider

    assert resolve_llm_provider() == "openrouter"


def test_model_name_for_provider_normalizes_openrouter_prefix():
    from humeo.env import model_name_for_provider

    assert (
        model_name_for_provider("gemini-3.1-flash-lite-preview", "openrouter")
        == "google/gemini-3.1-flash-lite-preview"
    )
    assert (
        model_name_for_provider("google/gemini-3.1-flash-lite-preview", "google")
        == "gemini-3.1-flash-lite-preview"
    )


@patch("humeo.clip_selector.genai.Client")
def test_select_clips_uses_gemini_client(mock_client_cls, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
    mock_inst = MagicMock()
    mock_client_cls.return_value = mock_inst
    mock_inst.models.generate_content.return_value = MagicMock(text='{"clips": []}')

    from humeo.clip_selector import select_clips

    select_clips({"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}]})

    mock_client_cls.assert_called_once_with(api_key="test-api-key")
    mock_inst.models.generate_content.assert_called_once()
    call_kw = mock_inst.models.generate_content.call_args
    assert call_kw[1]["model"]  # model name set


@patch("humeo.clip_selector.OpenAI")
def test_select_clips_uses_openrouter_when_only_router_key(mock_openai_cls, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    mock_inst = MagicMock()
    mock_openai_cls.return_value = mock_inst
    mock_inst.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"clips": []}'))]
    )

    from humeo.clip_selector import select_clips

    select_clips({"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}]})

    mock_openai_cls.assert_called_once()
    init_kwargs = mock_openai_cls.call_args.kwargs
    assert init_kwargs["api_key"] == "router-key"
    assert init_kwargs["base_url"] == "https://openrouter.ai/api/v1"

    call_kw = mock_inst.chat.completions.create.call_args.kwargs
    assert call_kw["model"] == "google/gemini-3.1-flash-lite-preview"
    assert call_kw["response_format"] == {"type": "json_object"}


@patch("humeo.clip_selector.genai.Client")
def test_select_clips_raises_without_key(mock_client_cls, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    from humeo.clip_selector import select_clips

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        select_clips({"segments": []})

    mock_client_cls.assert_not_called()


def test_parse_clips_preserves_score_breakdown_and_reasoning():
    from humeo.clip_selector import _parse_clips

    raw_json = json.dumps(
        {
            "clips": [
                {
                    "clip_id": "123",
                    "topic": "topic",
                    "start_time_sec": 10.0,
                    "end_time_sec": 25.0,
                    "duration_sec": 15.0,
                    "score_breakdown": {"message_wow": 0.8, "hook_emotion": 0.6},
                    "reasoning": "Strong payoff and clear takeaway.",
                }
            ]
        }
    )

    clips = _parse_clips(raw_json)

    assert len(clips) == 1
    assert clips[0].score_breakdown == {"message_wow": 0.8, "hook_emotion": 0.6}
    assert clips[0].reasoning == "Strong payoff and clear takeaway."


def test_parse_clips_clamps_legacy_point_based_score_breakdown():
    from humeo.clip_selector import _parse_clips

    raw_json = json.dumps(
        {
            "clips": [
                {
                    "clip_id": "123",
                    "topic": "topic",
                    "start_time_sec": 10.0,
                    "end_time_sec": 25.0,
                    "duration_sec": 15.0,
                    "score_breakdown": {
                        "counter_intuitive_claim": 3,
                        "quotable_phrasing": 2,
                    },
                }
            ]
        }
    )

    clips = _parse_clips(raw_json)

    assert len(clips) == 1
    assert clips[0].score_breakdown == {
        "counter_intuitive_claim": 1.0,
        "quotable_phrasing": 1.0,
    }


def test_load_clips_legacy_fixture_defaults_new_fields():
    from humeo.clip_selector import load_clips

    clips = load_clips(Path("tests/fixtures/legacy_clips.json"))

    assert clips
    for clip in clips:
        assert clip.origin == "text"
        assert clip.score_breakdown is None
        assert clip.visual_notes is None
        assert clip.reasoning is None


def test_build_prompt_passes_through_steering_notes():
    from humeo.clip_selector import build_prompt

    system, _user = build_prompt(
        {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]},
        steering_notes=["prefer standalone moments"],
    )

    assert "prefer standalone moments" in system
