# Architecture — Reusable Rocket

> *"We don't need to build the door or windows — just a container with landing
> gear and thrusters that move in different directions."*
> — Bryan

That analogy maps exactly onto this MCP:

| Rocket part     | Codebase                                                         | Purpose                                                                 |
| --------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Container       | `src/humeo_core/schemas.py`                                       | Strict JSON contracts every stage reads/writes.                         |
| Landing gear    | `src/humeo_core/primitives/ingest.py`                             | Deterministic local extraction (scenes, keyframes, transcript).         |
| Thrusters (×5)  | `src/humeo_core/primitives/layouts.py`                            | Five fixed 9:16 crop/compose recipes (max 2 on-screen items).           |
| Pilot           | `primitives/classify.py` + `primitives/select_clips.py`          | Heuristic + LLM-ready decision makers.                                  |
| Compiler        | `src/humeo_core/primitives/compile.py`                            | Deterministic ffmpeg assembly.                                          |
| Control panel   | `src/humeo_core/server.py`                                        | MCP tools exposing every primitive.                                     |
| Control surface | `src/humeo_core/server.py`                                        | MCP tool surface for agents and clients.                                |

## First-principles reasoning

The HIVE paper's core insight is that good short-video editing requires
**staged reasoning with strict intermediate artifacts**, not a single
giant model call. Three consequences flow from that:

1. **Extraction must be local and deterministic.** No model call should
   ever touch raw video bytes. `ingest.py` runs ffprobe + PySceneDetect
   + ffmpeg + (optional) faster-whisper. Everything it emits is JSON or
   a file path.

2. **Reasoning must be decomposed into narrow sub-tasks.** Classifying a
   scene's layout is a completely different task from selecting a viral
   clip. Each has its own schema, its own prompt, its own validation.
   This is why `primitives/` is five files instead of one.

3. **Every model call must emit schema-validated JSON.** Free-form model
   output is not allowed to enter the pipeline. `classify_scenes_with_llm`
   and `select_clips_with_llm` both `model_validate(...)` the raw output
   before returning; parse failures degrade gracefully to `SIT_CENTER` +
   low confidence, not crashes.

## Why only five layouts?

The hard rule for this format: **a short shows at most two on-screen
items**, where an "item" is a `person` or a `chart`. That gives exactly
five recipes — all implemented as pure functions from
`LayoutInstruction` to an ffmpeg filtergraph string in `layouts.py`:

| Layout                 | Items           | Recipe                                        |
| ---------------------- | --------------- | --------------------------------------------- |
| `zoom_call_center`     | 1 person        | tight centered 9:16 crop (zoom ≥ 1.25).       |
| `sit_center`           | 1 person        | wider centered 9:16 crop.                     |
| `split_chart_person`   | 1 chart + person| source partitioned L/R by bboxes, stacked.    |
| `split_two_persons`    | 2 persons       | L/R speakers, stacked top/bottom.             |
| `split_two_charts`     | 2 charts        | L/R charts, stacked top/bottom.               |

A general subject-tracker ML model is orders of magnitude more expensive
and less reliable than five hand-written crop recipes. If a new geometry
ever shows up in future source videos, adding a sixth thruster is
strictly additive: write a new `plan_*` function, add it to `_DISPATCH`,
add an enum variant. No existing code has to change.

## 9:16 layout math

Source is assumed 16:9 (1920×1080 by default, but probed per-clip).
Target is 1080×1920. For each layout:

### `zoom_call_center` and `sit_center`

Standard centered aspect-ratio crop to 9:16, then scale to 1080×1920:

```
crop=cw:ch:x:y,scale=1080:1920:flags=lanczos,setsar=1[vout]
```

`cw`, `ch` are the largest 9:16 window that fits in the source, divided
by `zoom`. `x`, `y` center the window on `person_x_norm` / 0.5.
Dimensions are rounded to even values so libx264 is happy. The window is
clamped inside the source so a high `person_x_norm` never crops outside.

### Split layouts (`split_chart_person`, `split_two_persons`, `split_two_charts`)

All three splits share one recipe — only the items differ:

1. **Horizontal partition.** The source is cut at a single vertical seam
   so the two source strips are **complementary** (no overlap, no gap).
   When both bboxes are set (Gemini vision), the seam is the midpoint
   between `left.x2` and `right.x1`. Otherwise the seam defaults to
   either an even 50/50 (two-of-a-kind splits) or a 2/3 | 1/3 split
   (legacy `split_chart_person` fallback).
2. **Vertical crop.** Each strip's vertical extent comes from the
   corresponding bbox when provided, so each item **fills** its output
   band instead of being lost in full-height source context.
3. **Cover-scale to the band.** Each strip is scaled with
   `force_original_aspect_ratio=increase` + center-cropped to the band
   dimensions. Bands are always fully painted; no letterbox bars.
4. **Stack.** Two branches produced by `split=2` are `vstack`-ed into
   the final 1080×1920.

**Band heights** are controlled by `LayoutInstruction.top_band_ratio`,
which defaults to **0.5** (even 50/50 — the symmetric look Bryan asked
for after the uneven Cathy Wood shorts). Legacy 60/40 is still reachable
by setting `top_band_ratio=0.6`.

**Stack order** (for `split_chart_person`) is controlled by
`focus_stack_order`: chart-on-top (default) or person-on-top.

## Extensibility story

- **Smarter classifier:** implement `LLMVisionFn` with any multimodal
  model and pass it to `classify_scenes_with_llm`. The fallback heuristic
  stays available for offline runs and tests.
- **Smarter clip selector:** same pattern, `LLMTextFn` → `select_clips_with_llm`.
- **New layout:** add a `plan_*` planner, register in `_DISPATCH`, add a
  `LayoutKind` variant. Tests in `test_layouts.py` automatically iterate
  over all `LayoutKind`s, so the dispatch coverage test will catch a
  missing registration immediately.

## What we intentionally did NOT build

- Drag-and-highlight subject-selector UI.
- A general ML subject-tracker.
- A monolithic video-in-video-out model.
- Any network calls in the core library. The MCP server is stdio-only;
  the CLI runs fully offline.

This keeps the rocket **reusable**: the same primitives power the MCP
server, the CLI, a Python library, and (soon) a web UI if that's ever
warranted.
