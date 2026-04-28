# Using humeo-core from an MCP client

The installed console command is **`humeo-core`**. For backward compatibility,
**`humeo-mcp`** is also registered (same entrypoint); either works in
`"command": ...` if both are on `PATH` from the same install.

## 1. Add to your client

`claude_desktop_config.json` or `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "humeo": {
      "command": "humeo-core"
    }
  }
}
```

## 2. A typical agent plan

```
→ humeo.list_layouts()
    # discover the 5 layouts (max 2 on-screen items per short)

→ humeo.ingest(source_path="/abs/long.mp4", work_dir="/abs/work", with_transcript=true)
    # IngestResult: scenes[], keyframes, transcript_words[]

→ humeo.classify_scenes(scenes=<IngestResult.scenes>)
    # SceneClassification[] — one layout per scene

→ humeo.select_clips(
      source_path=..., transcript_words=..., duration_sec=...,
      target_count=5, min_sec=30, max_sec=60
  )
    # ClipPlan — top non-overlapping clips

# For each clip, pick the layout of the scene its midpoint falls in,
# build a LayoutInstruction, and:

→ humeo.build_render_cmd(request={...})
    # dry-run: returns the exact ffmpeg argv, no execution

→ humeo.render_clip(request={..., "mode": "normal"})
    # actually renders the 9:16 MP4
```

## 3. Strict JSON all the way

Every request/response is validated against the schemas in
[`schemas.py`](../src/humeo_core/schemas.py). Invalid input is rejected
*before* ffmpeg is touched, so a confused agent can't accidentally
rm-rf your disk or burn GPU hours.

## 4. Override knobs

`LayoutInstruction` accepts:

- `zoom`, `person_x_norm`, `chart_x_norm` — single-subject knobs.
- `split_chart_region`, `split_person_region`,
  `split_second_chart_region`, `split_second_person_region` —
  normalized bboxes that drive split-layout cropping.
- `top_band_ratio` — fraction of output height used by the top band
  (default 0.5 = even 50/50, the symmetric look).
- `focus_stack_order` — for `split_chart_person`, chart-on-top vs
  person-on-top.

Example: chart + person with a precise bbox crop and an even split.

```json
{
  "clip_id": "001",
  "layout": "split_chart_person",
  "split_chart_region":  {"x1": 0.00, "y1": 0.10, "x2": 0.52, "y2": 0.95},
  "split_person_region": {"x1": 0.55, "y1": 0.05, "x2": 1.00, "y2": 1.00},
  "top_band_ratio": 0.5,
  "focus_stack_order": "chart_then_person"
}
```

Example: two-speaker interview.

```json
{
  "clip_id": "002",
  "layout": "split_two_persons",
  "split_person_region":        {"x1": 0.02, "y1": 0.05, "x2": 0.48, "y2": 0.95},
  "split_second_person_region": {"x1": 0.52, "y1": 0.05, "x2": 0.98, "y2": 0.95}
}
```

## 5. When to stay in dry-run

- You want to show an approval UI before spending CPU.
- You want to diff the planned ffmpeg commands against a previous run.
- You're building tests.

`mode="dry_run"` is always safe, never writes output, and returns the
exact argv list.
