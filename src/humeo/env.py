"""Environment bootstrap (``.env``) and cache path helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

_BOOTSTRAPPED = False
LLMProvider = Literal["google", "openrouter"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def bootstrap_env() -> None:
    """Load ``.env`` from the process cwd (non-fatal if missing). Safe to call twice."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    _BOOTSTRAPPED = True


def default_humeo_cache_root() -> Path:
    """Default cache root: ``~/.cache/humeo`` on Unix; ``%LOCALAPPDATA%/humeo`` on Windows."""
    override = (os.environ.get("HUMEO_CACHE_ROOT") or "").strip()
    if override:
        return Path(override)
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        return base / "humeo"
    return Path.home() / ".cache" / "humeo"


def resolve_gemini_api_key() -> str:
    """Return an API key for Gemini, or raise if none is configured.

    Prefer ``GOOGLE_API_KEY``; fall back to ``GEMINI_API_KEY``. Values are read from
    the environment after ``bootstrap_env()`` (``.env`` in cwd).

    We require an explicit key so we do not fall back to Application Default
    Credentials (e.g. ``gcloud auth application-default login``), which often
    lack the Generative Language API scope and produce
    ``403 ACCESS_TOKEN_SCOPE_INSUFFICIENT``.
    """
    bootstrap_env()
    for env_name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        val = (os.environ.get(env_name) or "").strip()
        if val:
            return val
    raise ValueError(
        "Set GOOGLE_API_KEY or GEMINI_API_KEY for Gemini clip selection. "
        "See docs/ENVIRONMENT.md. Without an API key the client may use ADC and fail "
        "with insufficient scopes (403)."
    )


def resolve_openrouter_api_key() -> str:
    """Return the OpenRouter API key, or raise if missing."""
    bootstrap_env()
    val = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if val:
        return val
    raise ValueError(
        "Set OPENROUTER_API_KEY to use OpenRouter as the backend for the Gemini stages. "
        "See docs/ENVIRONMENT.md."
    )


def current_llm_provider() -> LLMProvider | None:
    """Best-effort active backend detection from the environment.

    Preference order preserves the historical behavior: if a Google Gemini key is
    configured, use Google's SDK. OpenRouter is the fallback only when no Google
    Gemini key is present.
    """
    bootstrap_env()
    if (os.environ.get("GOOGLE_API_KEY") or "").strip():
        return "google"
    if (os.environ.get("GEMINI_API_KEY") or "").strip():
        return "google"
    if (os.environ.get("OPENROUTER_API_KEY") or "").strip():
        return "openrouter"
    return None


def resolve_llm_provider() -> LLMProvider:
    """Return the active backend for Gemini-like stages, or raise if none is configured."""
    provider = current_llm_provider()
    if provider is not None:
        return provider
    raise ValueError(
        "Set GOOGLE_API_KEY or GEMINI_API_KEY for the Google Gemini SDK, "
        "or set OPENROUTER_API_KEY to route these stages through OpenRouter."
    )


def model_name_for_provider(model_name: str, provider: LLMProvider) -> str:
    """Normalize model identifiers between Google Gemini SDK and OpenRouter.

    - Google SDK expects bare Gemini ids like ``gemini-3.1-flash-lite-preview``.
    - OpenRouter expects provider-qualified ids like
      ``google/gemini-3.1-flash-lite-preview``.
    """
    name = model_name.strip()
    if provider == "openrouter":
        if "/" not in name and name.startswith(("gemini-", "gemma-")):
            return f"google/{name}"
        return name
    if provider == "google" and name.startswith("google/"):
        return name.split("/", 1)[1]
    return name


def openrouter_default_headers() -> dict[str, str]:
    """Headers that help identify Humeo traffic to OpenRouter."""
    return {
        "HTTP-Referer": "https://github.com/frenzy2004/shortform",
        "X-Title": "Humeo",
    }
