"""End-to-end product pipeline."""

import dataclasses
import json
import logging
from pathlib import Path

from humeo_core.primitives.ingest import extract_keyframes
from humeo_core.schemas import LayoutInstruction, LayoutKind, RatingFeedback, Scene

from humeo import interactive, session_state
from humeo.clip_selection_cache import cache_valid, load_meta, transcript_fingerprint, write_artifacts
from humeo.clip_selector import load_clips, save_clips, select_clips
from humeo.config import MAX_CLIP_DURATION_SEC, MIN_CLIP_DURATION_SEC, PipelineConfig
from humeo.content_pruning import run_content_pruning_stage
from humeo.cutter import generate_ass
from humeo.hook_detector import run_hook_detection_stage
from humeo.ingest import download_video, extract_audio, stage_local_video, transcribe_whisperx
from humeo.layout_vision import run_layout_vision_stage
from humeo.render_window import clip_for_render
from humeo.reframe_ffmpeg import reframe_clip_ffmpeg
from humeo.video_cache import (
    extract_youtube_video_id,
    ingest_complete,
    normalize_local_source_path,
    read_youtube_info_json,
    resolve_work_directory,
    upsert_manifest_from_info,
)

logger = logging.getLogger(__name__)


def _rerun_config(config: PipelineConfig, steering_notes: list[str]) -> PipelineConfig:
    return dataclasses.replace(
        config,
        steering_notes=list(steering_notes),
        force_clip_selection=True,
        overwrite_outputs=True,
    )


def _build_steering_from_feedback(feedback: RatingFeedback) -> str:
    parts: list[str] = []
    if "wrong_moments" in feedback.issues:
        parts.append("Previous selection picked the wrong moments. Reselect with different candidates.")
    if "bad_cuts" in feedback.issues:
        parts.append(
            "Clip boundaries were bad. Prefer clips starting on clean sentence beginnings and ending on completed thoughts."
        )
    if "boring" in feedback.issues:
        parts.append("Previous selection lacked energy. Bias strongly toward high-emotion, high-hook moments.")
    if "confusing" in feedback.issues:
        parts.append("Previous clips needed too much context. Pick moments that make sense standalone.")
    if "wrong_layout" in feedback.issues:
        logger.warning("Received wrong_layout feedback, but layout overrides are not available until Gate 2 ships.")
    if "length_off" in feedback.issues:
        parts.append("Clip durations felt off. Respect the duration bounds strictly.")
    if "other" in feedback.issues and feedback.free_text:
        parts.append(feedback.free_text)
    return " ".join(parts).strip()


def _ensure_work_dir(config: PipelineConfig) -> None:
    """Resolve ``config.work_dir`` when unset (per-video cache) or ensure it exists."""
    if config.work_dir is not None:
        return
    config.work_dir = resolve_work_directory(
        youtube_url=config.youtube_url,
        explicit_work_dir=None,
        use_video_cache=config.use_video_cache,
        cache_root=config.cache_root,
    )


def _filter_render_valid_clips(clips: list, *, stage_label: str) -> list:
    """Drop clips whose actual render window violates the duration contract."""
    valid: list = []
    dropped = 0
    for clip in clips:
        render_clip = clip_for_render(clip)
        render_duration = render_clip.duration_sec
        if MIN_CLIP_DURATION_SEC <= render_duration <= MAX_CLIP_DURATION_SEC:
            valid.append(clip)
            continue
        dropped += 1
        logger.warning(
            "%s: dropping clip %s because render-window duration %.1fs is outside [%ds, %ds] "
            "(trim_start=%.1fs trim_end=%.1fs).",
            stage_label,
            clip.clip_id,
            render_duration,
            MIN_CLIP_DURATION_SEC,
            MAX_CLIP_DURATION_SEC,
            clip.trim_start_sec,
            clip.trim_end_sec,
        )
    if dropped:
        logger.warning("%s: dropped %d invalid render-window clip(s).", stage_label, dropped)
    return valid


def run_pipeline(config: PipelineConfig) -> list[Path]:
    """
    Execute the full podcast-to-shorts pipeline.

    Args:
        config: Pipeline configuration.

    Returns:
        List of paths to the final short-form MP4 files.
    """
    logger.info("=" * 60)
    logger.info("HUMEO PIPELINE START")
    logger.info("Source: %s", config.youtube_url)
    logger.info("Output: %s", config.output_dir)
    logger.info("=" * 60)

    _ensure_work_dir(config)
    assert config.work_dir is not None

    state = None
    if config.interactive:
        state = session_state.load_state(config.work_dir, config.youtube_url)
        if config.steering_notes:
            if list(config.steering_notes) != state.steering_notes:
                state.steering_notes = list(config.steering_notes)
                session_state.save_state(config.work_dir, state)
        elif state.steering_notes:
            config = dataclasses.replace(
                config,
                steering_notes=list(state.steering_notes),
                force_clip_selection=True,
                overwrite_outputs=True,
            )
            logger.info(
                "Loaded %d steering note(s) from session state for this source.",
                len(state.steering_notes),
            )

    # ------------------------------------------------------------------
    # Stage 1: Ingest
    # ------------------------------------------------------------------
    logger.info("--- STAGE 1: INGESTION ---")

    source_video = config.work_dir / "source.mp4"
    transcript_path = config.work_dir / "transcript.json"
    local_source_path = normalize_local_source_path(config.youtube_url)
    reuse_ingest = ingest_complete(config.work_dir, config.youtube_url)

    if reuse_ingest:
        logger.info("Cached ingest found for this source (reusing source + transcript).")
    elif local_source_path is not None:
        source_video = stage_local_video(local_source_path, config.work_dir)
    elif source_video.exists():
        logger.info("Source video already downloaded, skipping download.")
    else:
        source_video = download_video(config.youtube_url, config.work_dir)

    if reuse_ingest or (transcript_path.exists() and local_source_path is None):
        logger.info("Transcript already exists, loading.")
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)
    else:
        if transcript_path.exists() and local_source_path is not None:
            logger.info("Transcript exists but belongs to a different local source; regenerating.")
        audio_path = extract_audio(source_video, config.work_dir)
        transcript = transcribe_whisperx(audio_path, config.work_dir)

    if local_source_path is None:
        vid = extract_youtube_video_id(config.youtube_url)
        info = read_youtube_info_json(config.work_dir)
        if not info and vid:
            info = {"id": vid, "webpage_url": config.youtube_url}
        if info:
            upsert_manifest_from_info(
                work_dir=config.work_dir,
                youtube_url=config.youtube_url,
                info=info,
                cache_root=config.cache_root,
            )

    # ------------------------------------------------------------------
    # Stage 2: Clip Selection
    # ------------------------------------------------------------------
    logger.info("--- STAGE 2: CLIP SELECTION ---")

    clips_path = config.work_dir / "clips.json"
    fp = transcript_fingerprint(transcript)
    meta = load_meta(config.work_dir)
    cache_hit = (
        clips_path.is_file()
        and not config.force_clip_selection
        and meta is not None
        and cache_valid(meta, fp, config)
    )

    if cache_hit:
        clips = load_clips(clips_path)
        logger.info("Clip selection cache hit (transcript + provider/model unchanged); skipping LLM.")
    else:
        clips, raw = select_clips(
            transcript,
            gemini_model=config.gemini_model,
            candidate_count=config.clip_selection_candidate_count,
            quality_threshold=config.clip_selection_quality_threshold,
            min_kept=config.clip_selection_min_kept,
            max_kept=config.clip_selection_max_kept,
            steering_notes=config.steering_notes,
        )
        save_clips(clips, clips_path)
        write_artifacts(
            config.work_dir,
            transcript=transcript,
            config=config,
            raw_response=raw,
        )

    logger.info("Selected %d clips:", len(clips))
    for clip in clips:
        logger.info(
            "  [%s] %.1fs-%.1fs (%.1fs) score=%.2f - %s",
            clip.clip_id,
            clip.start_time_sec,
            clip.end_time_sec,
            clip.duration_sec,
            clip.virality_score,
            clip.topic,
        )

    # ------------------------------------------------------------------
    # Stage 2.25: Hook Detection
    # ------------------------------------------------------------------
    # The clip selector is unreliable at localising the hook sentence and
    # tends to return the 0.0-3.0s placeholder verbatim, which would disable
    # start-trim in Stage 2.5. This stage asks Gemini to localise the real
    # hook per clip so Stage 2.5 can clamp against a real window.
    logger.info("--- STAGE 2.25: HOOK DETECTION (enabled=%s) ---", config.detect_hooks)
    clips = run_hook_detection_stage(
        config.work_dir,
        clips,
        transcript,
        transcript_fp=fp,
        config=config,
    )

    # ------------------------------------------------------------------
    # Stage 2.5: Content Pruning (HIVE-style inner-clip tightening)
    # ------------------------------------------------------------------
    # Tightens each candidate window by writing trim_start_sec / trim_end_sec
    # on the Clip models. keyframe extraction and layout vision below both
    # consume ``clip_for_render(clip)`` so they automatically operate on the
    # pruned window without further changes.
    logger.info("--- STAGE 2.5: CONTENT PRUNING (level=%s) ---", config.prune_level)
    clips = run_content_pruning_stage(
        config.work_dir,
        clips,
        transcript,
        transcript_fp=fp,
        config=config,
    )
    clips = _filter_render_valid_clips(clips, stage_label="Stage 2.5 guardrail")

    if config.interactive and state is not None:
        result = interactive.approve_clips(clips)
        if result.action == "quit":
            logger.info("Aborted by user at Gate 1.")
            return []
        if result.action == "refine":
            state.iteration += 1
            if result.steering_note:
                state.steering_notes.append(result.steering_note)
            state.last_selected_ids = None
            session_state.save_state(config.work_dir, state)
            if state.iteration >= config.max_iterations:
                logger.warning("Iteration cap hit. Proceeding with current clips.")
            else:
                return run_pipeline(_rerun_config(config, state.steering_notes))
        elif result.action == "proceed":
            selected_ids = list(result.selected_ids or [])
            state.last_selected_ids = selected_ids
            session_state.save_state(config.work_dir, state)
            clip_by_id = {clip.clip_id: clip for clip in clips}
            clips = [clip_by_id[clip_id] for clip_id in selected_ids]
        elif result.action == "accept_all":
            state.last_selected_ids = [clip.clip_id for clip in clips]
            session_state.save_state(config.work_dir, state)

    # ------------------------------------------------------------------
    # Stage 3: Clip layouts
    # ------------------------------------------------------------------
    logger.info("--- STAGE 3: CLIP LAYOUTS ---")

    keyframes_dir = config.work_dir / "keyframes"
    clip_scenes: list[Scene] = []
    for clip in clips:
        rw = clip_for_render(clip)
        clip_scenes.append(
            Scene(scene_id=clip.clip_id, start_time=rw.start_time_sec, end_time=rw.end_time_sec)
        )
    clip_scenes = extract_keyframes(str(source_video), clip_scenes, str(keyframes_dir))
    layout_instructions = run_layout_vision_stage(
        config.work_dir,
        clip_scenes,
        source_video=source_video,
        transcript_fp=fp,
        clips_path=clips_path,
        config=config,
    )

    # ------------------------------------------------------------------
    # Stage 4: Render
    # ------------------------------------------------------------------
    logger.info("--- STAGE 4: RENDER ---")

    final_outputs: list[Path] = []
    subtitles_dir = config.work_dir / "subtitles"
    subtitles_dir.mkdir(parents=True, exist_ok=True)

    for clip in clips:
        instr = layout_instructions.get(clip.clip_id)
        if instr is None:
            hint = clip.layout_hint or LayoutKind.SIT_CENTER
            instr = LayoutInstruction(clip_id=clip.clip_id, layout=hint)
        clip.layout = instr.layout
        rclip = clip_for_render(clip)
        # ASS (not SRT) so the caption file's PlayResY matches the output
        # resolution and libass' font/margin scaling is 1:1.
        subtitle_path = generate_ass(
            rclip,
            transcript,
            subtitles_dir,
            max_words_per_cue=config.subtitle_max_words_per_cue,
            max_cue_sec=config.subtitle_max_cue_sec,
            play_res_x=1080,
            play_res_y=1920,
            font_size=config.subtitle_font_size,
            margin_v=config.subtitle_margin_v,
        )
        final_path = config.output_dir / f"short_{clip.clip_id}.mp4"
        if final_path.exists() and not config.overwrite_outputs:
            logger.info("Clip %s already rendered, skipping.", clip.clip_id)
            final_outputs.append(final_path)
            continue
        if final_path.exists() and config.overwrite_outputs:
            logger.info("Clip %s exists; overwriting due to clean-run settings.", clip.clip_id)

        # Font size and margin are already baked into the ASS file at
        # PlayResY=1920, so the compile primitive does not need to override
        # them -- but it still does, harmlessly, for single-source overrides.
        reframe_clip_ffmpeg(
            input_path=source_video,
            output_path=final_path,
            clip=rclip,
            layout_instruction=instr,
            subtitle_path=subtitle_path,
            subtitle_font_size=config.subtitle_font_size,
            subtitle_margin_v=config.subtitle_margin_v,
            title_text=clip.suggested_overlay_title,
        )
        final_outputs.append(final_path)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE - %d shorts generated:", len(final_outputs))
    for p in final_outputs:
        logger.info("  -> %s", p)
    logger.info("=" * 60)

    if config.interactive and final_outputs and state is not None:
        feedback = interactive.rate_output(final_outputs)
        state.last_rating = feedback
        session_state.save_state(config.work_dir, state)
        if feedback.rating == 3:
            logger.info("Rated Great. Shipped.")
            return final_outputs

        steering = _build_steering_from_feedback(feedback)
        if not steering:
            logger.warning("Interactive feedback recorded, but it is not actionable until a later gate ships.")
            return final_outputs

        state.iteration += 1
        state.steering_notes.append(steering)
        session_state.save_state(config.work_dir, state)
        if state.iteration >= config.max_iterations:
            logger.warning("Iteration cap hit. Source may not have a strong short.")
            return final_outputs
        return run_pipeline(_rerun_config(config, state.steering_notes))

    return final_outputs
