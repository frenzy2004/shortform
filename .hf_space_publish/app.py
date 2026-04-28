from __future__ import annotations

import logging
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path


def _bootstrap_local_paths() -> None:
    repo_root = Path(__file__).resolve().parent
    for candidate in (repo_root / "src", repo_root / "humeo-core" / "src"):
        candidate_str = str(candidate)
        if candidate.is_dir() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


_bootstrap_local_paths()
os.environ.setdefault("HUMEO_TRANSCRIBE_PROVIDER", "openai")

import gradio as gr

from humeo.config import PipelineConfig
from humeo.pipeline import run_pipeline


APP_TITLE = "Humeo"
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
STAGE_UPDATES: list[tuple[str, float, str]] = [
    ("STAGE 1: INGESTION", 0.18, "Stage 1/4: Ingestion"),
    ("STAGE 2: CLIP SELECTION", 0.34, "Stage 2/4: Clip selection"),
    ("STAGE 2.25: HOOK DETECTION", 0.46, "Stage 2.25/4: Hook detection"),
    ("STAGE 2.5: CONTENT PRUNING", 0.58, "Stage 2.5/4: Content pruning"),
    ("STAGE 3: CLIP LAYOUTS", 0.74, "Stage 3/4: Layout vision"),
    ("STAGE 4: RENDER", 0.88, "Stage 4/4: Render"),
]
LLM_KEY_NAMES = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY")


class QueueLogHandler(logging.Handler):
    def __init__(self, sink: queue.Queue[str]):
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._sink.put_nowait(self.format(record))
        except Exception:
            pass


def _validate_inputs(uploaded_file: str | None) -> Path:
    if not uploaded_file:
        raise gr.Error("Upload an MP4 file before starting a run.")

    path = Path(uploaded_file)
    if not path.is_file():
        raise gr.Error("The uploaded file is no longer available. Please upload the MP4 again.")
    if path.suffix.lower() != ".mp4":
        raise gr.Error("Only .mp4 uploads are supported in this Space.")
    if not any((os.environ.get(name) or "").strip() for name in LLM_KEY_NAMES):
        raise gr.Error(
            "Missing LLM credentials. Set GOOGLE_API_KEY, GEMINI_API_KEY, or OPENROUTER_API_KEY in Space secrets."
        )
    if not (os.environ.get("OPENAI_API_KEY") or "").strip():
        raise gr.Error(
            "Missing OPENAI_API_KEY. This Space uses OpenAI Whisper for transcription."
        )
    return path


def _install_log_handler(message_queue: queue.Queue[str], verbose: bool) -> tuple[logging.Handler, int, dict[str, int]]:
    handler = QueueLogHandler(message_queue)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S"))

    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    previous_logger_levels: dict[str, int] = {}
    for logger_name in ("urllib3", "httpx"):
        logger = logging.getLogger(logger_name)
        previous_logger_levels[logger_name] = logger.level
        logger.setLevel(logging.WARNING)

    return handler, previous_level, previous_logger_levels


def _remove_log_handler(
    handler: logging.Handler,
    previous_root_level: int,
    previous_logger_levels: dict[str, int],
) -> None:
    root_logger = logging.getLogger()
    root_logger.removeHandler(handler)
    root_logger.setLevel(previous_root_level)

    for logger_name, level in previous_logger_levels.items():
        logging.getLogger(logger_name).setLevel(level)


def _stage_update_for_line(line: str) -> tuple[float, str] | None:
    for needle, progress_value, label in STAGE_UPDATES:
        if needle in line:
            return progress_value, label
    if "PIPELINE COMPLETE" in line:
        return 1.00, "Complete"
    return None


def _run_pipeline_job(
    config: PipelineConfig,
    verbose: bool,
    message_queue: queue.Queue[str],
    result: dict[str, object],
) -> None:
    handler, previous_root_level, previous_logger_levels = _install_log_handler(message_queue, verbose)
    try:
        outputs = run_pipeline(config)
        result["outputs"] = [str(Path(output).resolve()) for output in outputs if Path(output).exists()]
    except Exception as exc:
        result["error"] = str(exc)
        for line in traceback.format_exc().splitlines():
            if line.strip():
                message_queue.put_nowait(line)
    finally:
        _remove_log_handler(handler, previous_root_level, previous_logger_levels)
        result["done"] = True


def run_space(
    uploaded_file: str | None,
    prune_level: str,
    subtitle_font_size: int,
    subtitle_margin_v: int,
    verbose: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    upload_path = _validate_inputs(uploaded_file)
    run_root = Path(tempfile.mkdtemp(prefix="humeo-space-"))
    staged_source = run_root / "input.mp4"
    work_dir = run_root / "work"
    output_dir = run_root / "output"

    shutil.copy2(upload_path, staged_source)

    log_lines = [
        f"Prepared upload: {upload_path.name}",
        f"Run directory: {run_root}",
    ]
    status_text = "Queued"
    yield status_text, "\n".join(log_lines), []

    config = PipelineConfig(
        source=str(staged_source),
        output_dir=output_dir,
        work_dir=work_dir,
        use_video_cache=False,
        clean_run=True,
        interactive=False,
        prune_level=prune_level,
        subtitle_font_size=int(subtitle_font_size),
        subtitle_margin_v=int(subtitle_margin_v),
    )

    message_queue: queue.Queue[str] = queue.Queue()
    result: dict[str, object] = {"done": False, "outputs": [], "error": None}
    worker = threading.Thread(
        target=_run_pipeline_job,
        args=(config, verbose, message_queue, result),
        daemon=True,
    )
    worker.start()

    seen_progress_values: set[float] = set()
    last_status = "Running"

    while worker.is_alive() or not message_queue.empty():
        updated = False
        while True:
            try:
                line = message_queue.get_nowait()
            except queue.Empty:
                break
            log_lines.append(line)
            stage_update = _stage_update_for_line(line)
            if stage_update is not None:
                progress_value, label = stage_update
                last_status = label
                if progress_value not in seen_progress_values:
                    progress(progress_value, desc=label)
                    seen_progress_values.add(progress_value)
            updated = True

        output_files = result["outputs"] if result["done"] and not result["error"] else []
        yield last_status, "\n".join(log_lines), output_files

        if not updated:
            time.sleep(0.25)

    error_text = result["error"]
    if error_text:
        last_status = f"Failed: {error_text}"
        log_lines.append(last_status)
        yield last_status, "\n".join(log_lines), []
        return

    if 1.00 not in seen_progress_values:
        progress(1.00, desc="Complete")
    output_files = result["outputs"]
    if output_files:
        last_status = "Complete"
    else:
        last_status = "Complete - no clips generated"
    yield last_status, "\n".join(log_lines), output_files


with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(
        """
        # Humeo

        Upload one MP4 and run the podcast-to-shorts pipeline inside a Hugging Face Docker Space.
        This demo streams Humeo's pipeline logs live and shows stage progress while rendering.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            source_file = gr.File(
                label="Source MP4",
                file_count="single",
                file_types=[".mp4"],
                type="filepath",
            )
            prune_level = gr.Dropdown(
                label="Prune level",
                choices=["off", "conservative", "balanced", "aggressive"],
                value="balanced",
            )
            subtitle_font_size = gr.Slider(
                label="Subtitle font size",
                minimum=32,
                maximum=72,
                value=48,
                step=1,
            )
            subtitle_margin_v = gr.Slider(
                label="Subtitle bottom margin",
                minimum=100,
                maximum=260,
                value=160,
                step=1,
            )
            verbose = gr.Checkbox(label="Verbose logging", value=False)
            run_button = gr.Button("Generate Shorts", variant="primary")

        with gr.Column(scale=1):
            status_box = gr.Textbox(label="Status", value="Idle", interactive=False)
            logs_box = gr.Textbox(label="Run logs", value="", lines=20, interactive=False)
            files_box = gr.Files(label="Rendered clips")

    run_button.click(
        fn=run_space,
        inputs=[source_file, prune_level, subtitle_font_size, subtitle_margin_v, verbose],
        outputs=[status_box, logs_box, files_box],
    )

demo.queue(default_concurrency_limit=1, max_size=1)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
    )
