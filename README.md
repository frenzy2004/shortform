---
title: Humeo
sdk: docker
app_port: 7860
---

# Humeo

Current default preset:

- `native_highlight` captions
- OpenRouter + `google/gemini-2.5-pro` for Gemini-like stages
- Replicate SAM speaker-lock when `REPLICATE_API_TOKEN` is available
- ElevenLabs Scribe v2 transcription when `ELEVENLABS_API_KEY` is set

Long podcast or interview → vertical 9:16 shorts. Pipeline: download, transcribe, Gemini (clip selection, hook detection, content pruning, layout vision), ffmpeg render.

**Architecture (static HTML, GitHub Pages):**  
[https://bryanthelai.github.io/long-to-shorts/hive_architecture_visualization.html](https://bryanthelai.github.io/long-to-shorts/hive_architecture_visualization.html)

## Hugging Face Space

This repo includes a Hugging Face Docker Space entrypoint in `app.py`.

- Upload one local MP4
- Watch live pipeline logs and stage progress
- Download rendered `short_*.mp4` clips from the UI

Required Space secrets:

- `GOOGLE_API_KEY` or `GEMINI_API_KEY`, or `OPENROUTER_API_KEY`
- `OPENAI_API_KEY` or `ELEVENLABS_API_KEY`

The Docker image pins `HUMEO_TRANSCRIBE_PROVIDER=openai` for the Space demo.

## Repo layout

| Path | Role |
|------|------|
| `src/humeo/` | CLI, pipeline, ingest, Gemini prompts, render adapters |
| `humeo-core/` | Schemas, ffmpeg compile, primitives, optional MCP server |

## Pipeline (actual order)

```text
YouTube URL
  → ingest (source.mp4, transcript.json)
  → clip selection (Gemini → clips.json)
  → hook detection (Gemini → hooks.json)
  → content pruning (Gemini → prune.json)
  → keyframes + layout vision (Gemini vision → layout_vision.json)
  → ASS subtitles + humeo-core ffmpeg render → short_<id>.mp4
```

Details: **`docs/PIPELINE.md`**.

## Five layouts

A short shows at most two on-screen items (`person` or `chart`). That yields five layout modes (see **`TERMINOLOGY.md`**).

## Requirements

- **Python** ≥ 3.10  
- **`uv`** — install: [astral.sh/uv](https://docs.astral.sh/uv/)  
- **`ffmpeg`** — on `PATH` for extract/render  
- **API keys** — see **`docs/ENVIRONMENT.md`**  
  - `GOOGLE_API_KEY` or `GEMINI_API_KEY` — preferred for Gemini stages  
  - `OPENROUTER_API_KEY` — supported fallback for those same Gemini-like stages when Google keys are unavailable  
  - `OPENAI_API_KEY` — if using OpenAI Whisper API (`HUMEO_TRANSCRIBE_PROVIDER=openai`)

Copy **`.env.example`** → **`.env`** (never commit `.env`).

## Install

```bash
uv venv
uv sync
```

Optional local WhisperX (heavy; Windows often uses OpenAI API instead):

```bash
uv sync --extra whisper
```

## Run

```bash
humeo --long-to-shorts "https://www.youtube.com/watch?v=VIDEO_ID"
humeo --long-to-shorts "C:\path\to\video.mp4"
```

Use **`--work-dir`** or **`--no-video-cache`** to control where `source.mp4` and intermediates live (see **`docs/ENVIRONMENT.md`**).

## CLI guide (all flags)

Use `humeo --help` for the live source of truth. This table matches `src/humeo/cli.py`.

### Required

| Flag | Meaning |
|------|---------|
| `--long-to-shorts SOURCE` | YouTube URL or local MP4 path to process (required). |

### Paths and cache behavior

| Flag | Meaning |
|------|---------|
| `--output`, `-o` | Output directory for final `short_*.mp4` (default: `./output`). |
| `--work-dir PATH` | Directory for intermediate artifacts (`source.mp4`, `transcript.json`, caches). |
| `--no-video-cache` | Disable per-video cache dirs; uses `./.humeo_work` unless `--work-dir` is set. |
| `--cache-root PATH` | Override cache root (env equivalent: `HUMEO_CACHE_ROOT`). |
| `--clean-run` | Fresh run: disables video cache, forces all model stages, overwrites outputs, and auto-creates a timestamped work dir if `--work-dir` is not provided. |

### Model selection and stage forcing

| Flag | Meaning |
|------|---------|
| `--gemini-model MODEL_ID` | Gemini model for clip selection / text stages (default from env/config). |
| `--gemini-vision-model MODEL_ID` | Gemini model for keyframe layout vision (defaults to `GEMINI_VISION_MODEL` or clip model). |
| `--force-clip-selection` | Re-run clip selection even if `clips.meta.json` cache matches. |
| `--force-hook-detection` | Re-run Stage 2.25 hook detection even if `hooks.meta.json` cache matches. |
| `--force-content-pruning` | Re-run Stage 2.5 pruning even if `prune.meta.json` cache matches. |
| `--force-layout-vision` | Re-run layout vision even if `layout_vision.meta.json` cache matches. |
| `--no-hook-detection` | Skip Stage 2.25 hook detection (pruning still runs with fallback behavior). |

### Pruning and subtitles

| Flag | Meaning |
|------|---------|
| `--prune-level {off,conservative,balanced,aggressive}` | Stage 2.5 aggressiveness (default: `balanced`). |
| `--subtitle-font-size INT` | Subtitle font size in output pixels (default: `48`). |
| `--subtitle-margin-v INT` | Bottom subtitle margin in output pixels (default: `160`). |
| `--subtitle-max-words INT` | Max words per subtitle cue (default: `4`). |
| `--subtitle-max-cue-sec FLOAT` | Max subtitle cue duration in seconds (default: `2.2`). |

### Logging

| Flag | Meaning |
|------|---------|
| `--verbose`, `-v` | Enable debug logging. |

### Common command recipes

```bash
# Basic run
humeo --long-to-shorts "https://www.youtube.com/watch?v=VIDEO_ID"

# Local MP4
humeo --long-to-shorts "C:\path\to\video.mp4"

# Full fresh run for debugging / prompt tuning
humeo --long-to-shorts "https://www.youtube.com/watch?v=VIDEO_ID" --clean-run --verbose

# Re-run only clip selection after prompt edits
humeo --long-to-shorts "https://www.youtube.com/watch?v=VIDEO_ID" --force-clip-selection

# Keep intermediates in a fixed local folder
humeo --long-to-shorts "https://www.youtube.com/watch?v=VIDEO_ID" --work-dir .humeo_work

# Compare different prune levels on same source
humeo --long-to-shorts "https://www.youtube.com/watch?v=VIDEO_ID" --prune-level conservative
humeo --long-to-shorts "https://www.youtube.com/watch?v=VIDEO_ID" --prune-level aggressive
```

## Documentation

| Doc | Purpose |
|-----|---------|
| **`docs/README.md`** | Index of all files under `docs/` |
| **`docs/STUDY_ORDER.md`** | Read order for onboarding |
| **`docs/PIPELINE.md`** | Stages, caches, JSON contracts |
| **`docs/ENVIRONMENT.md`** | Keys, env vars, cache layout |
| **`docs/SHARING.md`** | How to share logs/docs/video without bloating git |
| **`docs/TARGET_VIDEO_ANALYSIS.md`** | Reference input analysis example |
| **`docs/full_run_output.txt`** | Example full run log (text) |
| **`docs/hive-paper/PAPER_BREAKDOWN.md`** | HIVE paper, file mapping §9 |
| **`docs/hive-paper/hive_paper_blunt_guide.md`** | Short HIVE recap |
| **`docs/TODO.md`** | Backlog |
| **`docs/KNOWN_LIMITATIONS_AND_PROMPT_CONTRACT_GAP.md`** | Prompt vs code (ranking, hooks, unused fields, scene detect) |
| **`docs/SOLUTIONS.md`** | Design rationale |
| **`TERMINOLOGY.md`** | Glossary |

## Tests

```bash
uv sync --extra dev
uv run pytest
```

## Sharing outputs

`output/`, `*.mp4`, and `keyframes/` are **gitignored**. Put rendered shorts on **YouTube** or **GitHub Releases**; keep the repo for source and docs. See **`docs/SHARING.md`**.

## License

See **`LICENSE`** (root) and **`humeo-core/LICENSE`**.
