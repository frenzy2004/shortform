"""Configuration for the product pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from humeo.env import bootstrap_env

bootstrap_env()

# ---------------------------------------------------------------------------
# Video Output
# ---------------------------------------------------------------------------
TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
TARGET_ASPECT = 9 / 16

# ---------------------------------------------------------------------------
# Clip Selection
# ---------------------------------------------------------------------------
# Clip length bounds for Gemini (also referenced in prompts/clip_selection_system.jinja2).
MIN_CLIP_DURATION_SEC = 50
MAX_CLIP_DURATION_SEC = 90
TARGET_CLIP_COUNT = 5
TEXT_AXIS_WEIGHTS: dict[str, float] = {
    "message_wow": 0.4,
    "hook_emotion": 0.35,
    "catchy": 0.25,
}

# Gemini model id (override with GEMINI_MODEL in .env or shell). See docs/ENVIRONMENT.md.
GEMINI_MODEL = (os.environ.get("GEMINI_MODEL") or "gemini-3.1-flash-lite-preview").strip() or "gemini-3.1-flash-lite-preview"
# Optional *only* when layout vision should use a different id than clip selection
# (e.g. cheaper model per keyframe). Empty unset → ``resolved_vision_model`` uses
# ``GEMINI_MODEL`` / ``PipelineConfig.gemini_model`` (same multimodal stack).
GEMINI_VISION_MODEL = (os.environ.get("GEMINI_VISION_MODEL") or "").strip() or None

# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """Runtime configuration for a single pipeline run."""

    youtube_url: str
    output_dir: Path = field(default_factory=lambda: Path("output"))
    # None = auto: per-video dir under the cache root (see docs/ENVIRONMENT.md).
    work_dir: Path | None = None
    use_video_cache: bool = True
    # None = default from env (HUMEO_CACHE_ROOT) or platform default.
    cache_root: Path | None = None

    # None = use GEMINI_MODEL from env / module default (Gemini-only clip selection).
    gemini_model: str | None = None
    # None = GEMINI_VISION_MODEL env or same as gemini_model (per-keyframe layout + bbox).
    gemini_vision_model: str | None = None
    # When True, always re-run clip-selection LLM (ignore clips.meta.json match).
    force_clip_selection: bool = False
    # When True, always re-run Gemini vision for layouts (ignore layout_vision.meta.json).
    force_layout_vision: bool = False
    # When True, use an isolated work dir and force all stages to recompute.
    clean_run: bool = False
    # When True, render stage overwrites existing output files.
    overwrite_outputs: bool = False
    # When True, pause after clip selection and after render for human approval.
    interactive: bool = False
    # Interactive steering notes injected into the clip-selection prompt on reruns.
    steering_notes: list[str] = field(default_factory=list)
    # Hard cap on interactive reruns.
    max_iterations: int = 5

    # Stage 2.25 - hook detection. The clip selector is unreliable at
    # localising the hook sentence and tends to echo the 0.0-3.0s placeholder
    # from the prompt verbatim. This dedicated stage reads each candidate
    # window and returns a real hook window per clip, which Stage 2.5 then
    # uses to clamp pruning safely. When False, the clip-selection hook
    # (possibly a placeholder) is carried through unchanged.
    detect_hooks: bool = True
    # When True, re-run the hook-detection LLM even when hooks.meta.json matches.
    force_hook_detection: bool = False

    # Stage 2.5 - inner-clip content pruning (HIVE "irrelevant content pruning"
    # applied at clip scale). One of: off | conservative | balanced | aggressive.
    # See ``src/humeo/content_pruning.py`` for the caps and the prompt.
    prune_level: str = "balanced"
    # When True, re-run the pruning LLM even when prune.meta.json matches.
    force_content_pruning: bool = False

    # Stage 2 - candidate over-generation. The selector now asks Gemini for a
    # pool of candidates (``clip_selection_candidate_count``), scores them,
    # and keeps the top ones that pass ``clip_selection_quality_threshold``.
    # We always keep at least ``clip_selection_min_kept`` clips even when
    # none pass the threshold, so rendering never blocks on a weak transcript.
    # See ``src/humeo/clip_selector.py`` for the ranking logic.
    clip_selection_candidate_count: int = 12
    clip_selection_quality_threshold: float = 0.70
    clip_selection_min_kept: int = 5
    clip_selection_max_kept: int = 8

    # Subtitle rendering / cue shaping.
    # Values are in **output pixels** for a 1080x1920 short: libass is pinned to
    # the output resolution via ``original_size``, so ``FontSize`` and ``MarginV``
    # mean what they say. 48px font with a 160px bottom margin lands the caption
    # in the lower third with a readable-but-not-shouting size.
    subtitle_font_size: int = 48
    subtitle_margin_v: int = 160
    subtitle_max_words_per_cue: int = 4
    subtitle_max_cue_sec: float = 2.2

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.cache_root is not None:
            self.cache_root = Path(self.cache_root)
        if self.work_dir is not None:
            self.work_dir = Path(self.work_dir)
            self.work_dir.mkdir(parents=True, exist_ok=True)
