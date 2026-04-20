# Environment variables

This is the single reference for how Humeo reads configuration from the environment and from a project `.env` file.

## Loading order

1. On import, `humeo.config` runs `humeo.env.bootstrap_env()`, which calls `python-dotenv`’s `load_dotenv()` for the **current working directory** (typically your repo root).
2. Values already set in the process environment **win** over `.env` (dotenv default).

Copy `.env.example` to `.env` and fill in secrets. `.env` is gitignored.

## Gemini (clip selection)

Clip selection uses the **Google Gen AI SDK for Python** (`google-genai` package): `from google import genai`. Upstream docs: [python-genai](https://github.com/googleapis/python-genai) (Gemini Developer API and Vertex AI). The older `google-generativeai` package is not used.

| Variable | Used for |
|----------|----------|
| **`GOOGLE_API_KEY`** | **Preferred** API key for Gemini. Get a key from [Google AI Studio](https://aistudio.google.com/apikey). The SDK also recognizes **`GEMINI_API_KEY`** in the environment when using `genai.Client()` without an explicit key. |
| **`GEMINI_API_KEY`** | Fallback only if `GOOGLE_API_KEY` is unset (same kind of key as AI Studio). |
| **`OPENROUTER_API_KEY`** | Optional fallback backend for the same Gemini-like stages when no Google Gemini key is available. Humeo routes clip selection, hook detection, content pruning, and layout vision through OpenRouter's OpenAI-compatible chat completions API. |

Gemini-like stages **must** use an explicit API key. Humeo prefers the Google Gemini SDK when `GOOGLE_API_KEY` / `GEMINI_API_KEY` is present. If those are absent and `OPENROUTER_API_KEY` is set, Humeo falls back to OpenRouter instead. Without any of those keys, the pipeline cannot run the LLM stages.

| Variable | Default | Meaning |
|----------|---------|---------|
| **`GEMINI_MODEL`** | `gemini-3.1-flash-lite-preview` | Gemini model id for clip selection. Override per run with `--gemini-model`. |
| **`GEMINI_VISION_MODEL`** | *(unset)* | Optional separate model id for per-keyframe layout + bbox. If unset, the effective clip-selection model is used. Override per run with `--gemini-vision-model`. |

## Clip selection prompts (Jinja2)

Templates live under `src/humeo/prompts/` in the repo (`clip_selection_system.jinja2`, `clip_selection_user.jinja2`) and are shipped with the package.

| Variable | Used for |
|----------|----------|
| **`HUMEO_PROMPTS_DIR`** | If set to a directory path, Humeo loads those `.jinja2` files instead of the built-ins (custom prompts without editing the package). |

Clip duration bounds for the LLM match `MIN_CLIP_DURATION_SEC` / `MAX_CLIP_DURATION_SEC` in `humeo.config` (defaults **50–90** seconds). Edit `config.py` or your forked templates to change what Gemini is asked for.

### Per-clip layout (product pipeline)

After clip selection, the pipeline extracts one keyframe per clip and calls **Gemini vision** with a fixed JSON schema (`layout`, `person_bbox`, `chart_bbox`, `reason`). That produces a full **`LayoutInstruction`** per clip (including optional normalized split regions). Cached artifacts: **`layout_vision.meta.json`** and **`layout_vision.json`** under the work directory. Use **`--force-layout-vision`** to ignore that cache. Full detail: **`docs/PIPELINE.md`**.

## OpenAI (transcription only)

| Variable | Used for |
|----------|----------|
| **`OPENAI_API_KEY`** | OpenAI **Whisper** HTTP API when you choose it (see below) or when WhisperX is not installed. Not used for clip selection. |
| **`HUMEO_TRANSCRIBE_PROVIDER`** | `auto` (default), `openai`, or `whisperx`. With `uv sync --extra whisper`, WhisperX wins unless you set **`openai`** — then **`OPENAI_API_KEY`** is used and you avoid local WhisperX/torch stack noise on Windows. Same video directory still reuses **`transcript.json`** after the first successful run. |

Transcription output is always normalized to **`transcript.json`** in the work directory. Re-running the pipeline on the same cached video **skips** ASR when that file already exists (same as skipping re-download of **`source.mp4`**).

## Clip selection cache (LLM skip)

When **`clips.json`** and **`clips.meta.json`** exist under the work directory and the meta file’s **`transcript_sha256`** matches the current **`transcript.json`** (canonical JSON hash), and **`gemini_model`** matches, the pipeline **skips** the clip-selection call and reuses **`clips.json`**.

| Artifact | Role |
|----------|------|
| **`clips.meta.json`** | Version (v2 = Gemini-only), transcript hash, `gemini_model` |
| **`clip_selection_raw.json`** | Exact raw JSON string returned by Gemini (debug / audit) |

Use **`--force-clip-selection`** to always re-run. Legacy v1 meta files that used OpenAI for clip selection are treated as **cache miss**.

## Layout vision cache (Gemini multimodal)

When **`layout_vision.json`** and **`layout_vision.meta.json`** exist and **`transcript_sha256`**, **`clips_sha256`**, and **`gemini_vision_model`** match the current run, the pipeline **skips** per-keyframe vision calls.

| Artifact | Role |
|----------|------|
| **`layout_vision.meta.json`** | Transcript hash, SHA256 of **`clips.json`**, vision model id |
| **`layout_vision.json`** | Per-clip `instruction` (serialized `LayoutInstruction`) + `raw` (Gemini JSON or error) |

Use **`--force-layout-vision`** to always re-run. Changing **`clips.json`** (any byte) invalidates the layout cache.

## Video cache

Repeat runs for the same YouTube video id reuse **`source.mp4`** and **`transcript.json`** when they already exist in the resolved work directory.

| Variable | Meaning |
|----------|---------|
| **`HUMEO_CACHE_ROOT`** | Root directory for the cache layout and the manifest file. Default: `~/.cache/humeo` on Unix, `%LOCALAPPDATA%/humeo` on Windows. |

### Layout

- **Default work directory** (no `--work-dir`): `<cache_root>/videos/<11-char-video-id>/`
- **Global manifest** (index of ids → paths and metadata): `<cache_root>/video_cache_manifest.json`
- **`--no-video-cache`**: use `./.humeo_work` unless you pass `--work-dir`
- **`--work-dir PATH`**: use that folder for all intermediates (disables the default per-id path)

After a successful download, yt-dlp writes **`source.info.json`** next to **`source.mp4`**. The pipeline merges that metadata into the manifest.

## CLI cross-reference

| Flag | Env equivalent / notes |
|------|-------------------------|
| `--gemini-model` | `GEMINI_MODEL` |
| `--gemini-vision-model` | `GEMINI_VISION_MODEL` |
| `--cache-root` | `HUMEO_CACHE_ROOT` |
| `--no-video-cache` | Disables per-video cache dirs |
| `--force-clip-selection` | Ignores clip-selection cache |
| `--force-layout-vision` | Ignores layout vision cache |

## See also

- **[`docs/README.md`](README.md)** — Index of all documentation under `docs/`.
- **[`docs/PIPELINE.md`](PIPELINE.md)** — Stages and cache invalidation (overlaps CLI flags above only by cross-reference; no duplicate tables maintained here).
- **[`docs/SHARING.md`](SHARING.md)** — Public repo vs raw file URLs vs GitHub Pages; why large media is not committed.
