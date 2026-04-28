# humeo-core

**Reusable-rocket MCP server for long-video → 9:16 shorts.**

First-principles design, from the HIVE paper + Bryan's rocket analogy:
we don't build doors and windows (general subject-tracker UI, retraining
models). We build the **container** (schemas), **landing gear** (deterministic
local extraction), and **five thrusters** (the five 9:16 layouts this video
format actually uses). Everything else is pluggable.

## The rocket, in one picture

```
            ┌──────────────────────────────────────────┐
            │         Control panel  (MCP tools)       │   <- any MCP client
            └───────────────────┬──────────────────────┘
                                │ strict JSON
   ┌────────────────┬───────────┼────────────────┬─────────────────┐
   ▼                ▼           ▼                ▼                 ▼
 ingest       classify_scenes  select_clips   plan_layout       render_clip
(scenes +     (5-way layout   (clip picker,   (5 thrusters,    (ffmpeg compile,
 keyframes +   classifier)     heuristic +     pure filter      dry-run safe)
 transcript)                   LLM-ready)      math)
                                                 │
                                                 ▼
                                       ┌────────────────────┐
                                       │   LayoutKind       │
                                       │  ────────────────  │
                                       │  zoom_call_center  │
                                       │  sit_center        │
                                       │  split_chart_person│
                                       │  split_two_persons │
                                       │  split_two_charts  │
                                       └────────────────────┘
```

Only the classifier and clip-selector have optional LLM hooks; everything
else is deterministic, local, and cheap.

## Why five layouts? (the "max 2 items" rule)

The hard constraint for this format: **a short shows at most two on-screen
items** — where an "item" is a `person` (a human speaker) or a `chart`
(slide, graph, data visual, screenshare). That gives exactly five recipes:

1. **`zoom_call_center`** — 1 person, tight zoom-call / webcam framing.
2. **`sit_center`** — 1 person, interview / seated framing.
3. **`split_chart_person`** — 1 chart + 1 person, stacked vertically
   (default: **even 50/50** top/bottom, chart on top).
4. **`split_two_persons`** — 2 speakers, stacked vertically.
5. **`split_two_charts`** — 2 charts, stacked vertically.

Because the geometry is bounded, we do NOT need a general subject-tracker
ML model or a drag-to-highlight UI. We need five small, correct pieces of
crop/compose math. That is exactly what `src/humeo_core/primitives/layouts.py`
is.

See [`TERMINOLOGY.md`](../TERMINOLOGY.md) for the full glossary of terms
used across these docs (subject, crop, band, seam, bbox, layout, etc.).

## Install

```bash
uv venv
uv sync
```

External requirements: `ffmpeg` and `ffprobe` on PATH.

`scenedetect` requires OpenCV. Install `opencv-python-headless` or
`opencv-python` alongside `scenedetect`.

## Use it as an MCP server

```bash
humeo-core         # stdio transport (primary console script)
# humeo-mcp        # same entrypoint — kept so existing MCP configs keep working
```

Example Cursor/Claude Desktop config:

```json
{
  "mcpServers": {
    "humeo": { "command": "humeo-core" }
  }
}
```

Tools exposed:

| Tool                              | Purpose                                                                     |
| --------------------------------- | --------------------------------------------------------------------------- |
| `list_layouts`                    | Enumerate the 5 supported layouts.                                          |
| `ingest`                          | Scene detection + keyframe extraction (+ optional transcript).              |
| `classify_scenes`                 | Pixel-heuristic per-scene layout classification.                            |
| `detect_scene_regions`            | Return the bbox prompt + per-scene jobs (agent runs its own vision model).  |
| `classify_scenes_with_vision`     | Classify scenes from already-gathered `SceneRegions` bbox JSON + build layout instructions. |
| `select_clips`                    | Heuristic clip picker over a word-level transcript.                         |
| `plan_layout`                     | Return the exact `ffmpeg -filter_complex` for a layout.                     |
| `build_render_cmd`                | Build the ffmpeg command (no execution) — review before spend.              |
| `render_clip`                     | Build + run ffmpeg to produce a 9:16 MP4.                                   |

Resource: `humeo://layouts` (JSON listing of the 5 layouts).

### Three interchangeable region detectors

All three emit the same `SceneRegions` schema, so the layout planner and renderer don't care which one you used:

```
classify.py   (pixel variance, no ML)
face_detect.py (MediaPipe, local)            ──► SceneRegions ──► SceneClassification ──► LayoutInstruction ──► ffmpeg
vision.py     (multimodal LLM + OCR bboxes)
```

## JSON contracts (non-negotiable)

All tools take and return Pydantic-validated JSON. The contracts live in
[`src/humeo_core/schemas.py`](src/humeo_core/schemas.py):

- `Scene`                     `{scene_id, start_time, end_time, keyframe_path?}`
- `TranscriptWord`            `{word, start_time, end_time}`
- `IngestResult`              `{source_path, duration_sec, scenes[], transcript_words[], keyframes_dir?}`
- `SceneClassification`       `{scene_id, layout, confidence, reason}`
- `BoundingBox`               `{x1, y1, x2, y2, label, confidence}`  (all coords normalized)
- `SceneRegions`              `{scene_id, person_bbox?, chart_bbox?, ocr_text, raw_reason}`
- `Clip`                      `{clip_id, topic, start_time_sec, end_time_sec, viral_hook, virality_score, transcript, suggested_overlay_title, layout?}`
- `ClipPlan`                  `{source_path, clips[]}`
- `LayoutInstruction`         `{clip_id, layout, zoom, person_x_norm, chart_x_norm, split_chart_region?, split_person_region?, split_second_chart_region?, split_second_person_region?, top_band_ratio, focus_stack_order}`
- `RenderRequest` / `RenderResult`

## First-principles decisions (what we intentionally did NOT build)

- **No giant subject-tracker ML.** The video format has 5 fixed layouts
  (with a hard "max 2 items" rule); pixel-level tracking is not needed.
- **No drag-and-highlight UI.** An MCP tool is a better "UI" for an
  agent-first workflow. If a human wants to override, they pass a
  `LayoutInstruction` with their own `person_x_norm` / `chart_x_norm` /
  `zoom`.
- **No end-to-end video→video model.** The HIVE paper's core insight is
  that decomposed orchestration beats monolithic generation. We reify
  that insight as six small composable tools.

## Extending the pilot

- Plug a real multimodal model into `classify_scenes_with_llm(vision_fn)`.
- Plug a real reasoning model into `select_clips_with_llm(text_fn)`.
- Plug a real vision-LLM into `detect_regions_with_llm(scenes, vision_fn)`
  to get per-scene bboxes + OCR text, then feed the results back through
  `classify_scenes_with_vision`. This is the scene-change → v3 images →
  LLM+OCR → bbox path; see `../docs/SOLUTIONS.md §4` for rationale.
- All enforce strict JSON outputs, so bad model output can't corrupt
  downstream stages.

## Testing

```bash
python -m pytest
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for deeper rationale.

## License

MIT
