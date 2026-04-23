"""Clip selection cache fingerprint (Gemini-only meta)."""

from humeo.clip_selection_cache import (
    CURRENT_META_VERSION,
    cache_valid,
    load_meta,
    transcript_fingerprint,
    write_artifacts,
)
from humeo.config import PipelineConfig


def test_transcript_fingerprint_stable():
    t1 = {"segments": [{"start": 1.0, "end": 2.0, "text": "a"}]}
    t2 = {"segments": [{"start": 1.0, "end": 2.0, "text": "a"}]}
    assert transcript_fingerprint(t1) == transcript_fingerprint(t2)


def test_transcript_fingerprint_key_order():
    t1 = {"a": 1, "b": 2}
    t2 = {"b": 2, "a": 1}
    assert transcript_fingerprint(t1) == transcript_fingerprint(t2)


def test_cache_roundtrip_v2(tmp_path):
    tr = {"segments": []}
    cfg = PipelineConfig(youtube_url="https://youtu.be/x", gemini_model="m")
    write_artifacts(tmp_path, transcript=tr, config=cfg, raw_response='{"clips":[]}')
    meta = load_meta(tmp_path)
    assert meta is not None
    assert meta.get("version") == CURRENT_META_VERSION
    assert cache_valid(meta, transcript_fingerprint(tr), cfg)


def test_cache_invalidates_on_model_change(tmp_path):
    tr = {"segments": []}
    cfg = PipelineConfig(youtube_url="https://youtu.be/x", gemini_model="m")
    write_artifacts(tmp_path, transcript=tr, config=cfg, raw_response="{}")
    meta = load_meta(tmp_path)
    cfg2 = PipelineConfig(youtube_url="https://youtu.be/x", gemini_model="other")
    assert not cache_valid(meta, transcript_fingerprint(tr), cfg2)


def test_legacy_v1_openai_meta_invalidates(tmp_path):
    """Old meta from --provider openai runs must not hit cache."""
    tr = {"segments": []}
    cfg = PipelineConfig(youtube_url="https://youtu.be/x", gemini_model="gemini-2.0-flash")
    meta = {
        "version": 1,
        "transcript_sha256": transcript_fingerprint(tr),
        "llm_provider": "openai",
        "openai_model": "gpt-4o",
    }
    assert not cache_valid(meta, transcript_fingerprint(tr), cfg)


def test_openrouter_backend_invalidates_legacy_google_cache(monkeypatch):
    tr = {"segments": []}
    cfg = PipelineConfig(youtube_url="https://youtu.be/x", gemini_model="m")
    meta = {
        "version": CURRENT_META_VERSION,
        "transcript_sha256": transcript_fingerprint(tr),
        "gemini_model": "m",
    }
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")

    assert not cache_valid(meta, transcript_fingerprint(tr), cfg)


def test_openrouter_backend_roundtrip(monkeypatch, tmp_path):
    tr = {"segments": []}
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    cfg = PipelineConfig(youtube_url="https://youtu.be/x", gemini_model="m")

    write_artifacts(tmp_path, transcript=tr, config=cfg, raw_response='{"clips":[]}')
    meta = load_meta(tmp_path)

    assert meta is not None
    assert meta.get("llm_backend") == "openrouter"
    assert cache_valid(meta, transcript_fingerprint(tr), cfg)


def test_cache_invalidates_when_hook_library_changes(monkeypatch, tmp_path):
    tr = {"segments": []}
    hook_dir = tmp_path / "hooks"
    hook_dir.mkdir()
    (hook_dir / "Educational_Hooks.md").write_text(
        "1. Hook: A<br>Example: B<br>Psychology: C\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HUMEO_HOOK_LIBRARY_PATH", str(hook_dir))
    cfg = PipelineConfig(youtube_url="https://youtu.be/x", gemini_model="m")
    write_artifacts(tmp_path, transcript=tr, config=cfg, raw_response='{"clips":[]}')
    meta = load_meta(tmp_path)
    assert meta is not None
    assert cache_valid(meta, transcript_fingerprint(tr), cfg)

    (hook_dir / "Educational_Hooks.md").write_text(
        "1. Hook: A<br>Example: Changed<br>Psychology: C\n",
        encoding="utf-8",
    )
    assert not cache_valid(meta, transcript_fingerprint(tr), cfg)
