"""CLI entry point for the Humeo pipeline."""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from humeo.config import PipelineConfig
from humeo.pipeline import run_pipeline


def setup_logging(verbose: bool = False):
    """Configure logging with a clean format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="humeo",
        description="Humeo - Automated podcast-to-shorts pipeline from YouTube or local MP4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  humeo --long-to-shorts "https://youtube.com/watch?v=abc123"
  humeo --long-to-shorts "C:\\Videos\\episode.mp4"
  humeo --long-to-shorts "https://youtube.com/watch?v=abc123" --work-dir .humeo_work
  humeo --long-to-shorts "https://youtube.com/watch?v=abc123" --gemini-model gemini-2.0-flash
        """,
    )

    parser.add_argument(
        "--long-to-shorts",
        metavar="SOURCE",
        required=True,
        help="YouTube video URL or local MP4 path to process",
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("output"),
        help="Output directory for final shorts (default: ./output)",
    )

    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Working directory for intermediate files. Default: per-video folder under the "
        "cache root (see docs/ENVIRONMENT.md). Use this to force e.g. ./.humeo_work.",
    )

    parser.add_argument(
        "--no-video-cache",
        action="store_true",
        help="Do not use per-video cache dirs; use ./.humeo_work unless --work-dir is set.",
    )

    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Override cache root for manifests and per-video ingest (env: HUMEO_CACHE_ROOT).",
    )

    parser.add_argument(
        "--gemini-model",
        default=None,
        help="Gemini model id for clip selection (default: GEMINI_MODEL env; see humeo.config).",
    )

    parser.add_argument(
        "--force-clip-selection",
        action="store_true",
        help="Re-run clip-selection LLM even when clips.meta.json matches the transcript.",
    )

    parser.add_argument(
        "--gemini-vision-model",
        default=None,
        help="Gemini model for per-keyframe layout + bbox (default: GEMINI_VISION_MODEL env or --gemini-model).",
    )

    parser.add_argument(
        "--force-layout-vision",
        action="store_true",
        help="Re-run Gemini vision for layouts even when layout_vision.meta.json matches.",
    )

    parser.add_argument(
        "--prune-level",
        choices=["off", "conservative", "balanced", "aggressive"],
        default="balanced",
        help=(
            "Stage 2.5 inner-clip content pruning aggressiveness. "
            "'off' skips pruning entirely; 'conservative' trims <=10%%, "
            "'balanced' <=20%%, 'aggressive' <=35%% of each clip "
            "(always clamped to the MIN_CLIP_DURATION_SEC floor). Default: balanced."
        ),
    )

    parser.add_argument(
        "--force-content-pruning",
        action="store_true",
        help="Re-run content-pruning LLM even when prune.meta.json matches.",
    )

    parser.add_argument(
        "--no-hook-detection",
        action="store_true",
        help=(
            "Skip Stage 2.25 hook detection. The selector's hook window "
            "(possibly the 0.0-3.0s placeholder) will be carried through. "
            "Stage 2.5 content pruning still treats that exact placeholder "
            "as 'no hook' so pruning is not disabled."
        ),
    )

    parser.add_argument(
        "--force-hook-detection",
        action="store_true",
        help="Re-run hook-detection LLM even when hooks.meta.json matches.",
    )

    parser.add_argument(
        "--clean-run",
        action="store_true",
        help=(
            "Run with a fresh work dir and no cache reuse. Implies --no-video-cache, "
            "--force-clip-selection, --force-layout-vision, and overwrite existing outputs."
        ),
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Pause after clip selection and after render for human approval.",
    )

    parser.add_argument(
        "--subtitle-font-size",
        type=int,
        default=48,
        help=(
            "Caption font size in output pixels. libass is pinned to "
            "original_size=1080x1920, so this is a true pixel value. "
            "(default: 48)"
        ),
    )

    parser.add_argument(
        "--subtitle-margin-v",
        type=int,
        default=160,
        help="Caption bottom margin in output pixels (default: 160)",
    )

    parser.add_argument(
        "--subtitle-max-words",
        type=int,
        default=4,
        help="Max words per subtitle cue (default: 4)",
    )

    parser.add_argument(
        "--subtitle-max-cue-sec",
        type=float,
        default=2.2,
        help="Max subtitle cue duration in seconds (default: 2.2)",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    return parser


def main():
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    use_video_cache = not args.no_video_cache
    force_clip_selection = args.force_clip_selection
    force_layout_vision = args.force_layout_vision
    force_content_pruning = args.force_content_pruning
    force_hook_detection = args.force_hook_detection
    detect_hooks = not args.no_hook_detection
    overwrite_outputs = False
    work_dir = args.work_dir

    if args.clean_run:
        use_video_cache = False
        force_clip_selection = True
        force_layout_vision = True
        force_content_pruning = True
        force_hook_detection = True
        overwrite_outputs = True
        if work_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            work_dir = Path(f".humeo_work_clean_{stamp}")

    config = PipelineConfig(
        youtube_url=args.long_to_shorts,
        output_dir=args.output,
        work_dir=work_dir,
        use_video_cache=use_video_cache,
        cache_root=args.cache_root,
        gemini_model=args.gemini_model,
        gemini_vision_model=args.gemini_vision_model,
        force_clip_selection=force_clip_selection,
        force_layout_vision=force_layout_vision,
        clean_run=args.clean_run,
        overwrite_outputs=overwrite_outputs,
        interactive=args.interactive,
        prune_level=args.prune_level,
        force_content_pruning=force_content_pruning,
        detect_hooks=detect_hooks,
        force_hook_detection=force_hook_detection,
        subtitle_font_size=args.subtitle_font_size,
        subtitle_margin_v=args.subtitle_margin_v,
        subtitle_max_words_per_cue=args.subtitle_max_words,
        subtitle_max_cue_sec=args.subtitle_max_cue_sec,
    )

    try:
        outputs = run_pipeline(config)
        print(f"\nDone. {len(outputs)} shorts generated in: {config.output_dir}")
        for p in outputs:
            print(f"   -> {p}")
    except KeyboardInterrupt:
        print("\nPipeline interrupted.")
        sys.exit(1)
    except Exception as e:
        logging.getLogger(__name__).error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
