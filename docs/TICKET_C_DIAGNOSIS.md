# Ticket C Diagnosis

This diagnosis uses the post-Ticket-B rerun artifacts:

- `.humeo_ticketb_videoplayback3`
- `.humeo_ticketb_videoplayback4`
- `.humeo_ticketb_videoplayback5`

The goal was to determine whether the remaining layout failures come from one
bug family or multiple distinct failures before changing code.

## Mixed-scale Hypothesis Scan

First pass: scan `layout_vision.json` across the 9 evaluation clips and classify
the raw bbox coordinate scale per box:

- `normalized`: every coordinate is in `[0.0, 1.0]`
- `pixelish`: every coordinate is `> 1.0` (or `0` where zero can still be a pixel edge)
- `mixed`: the same bbox contains both normalized-style and pixel-style values

Results:

- `videoplayback (3) / short_002`
  - `raw.person_bbox = {"x1": 0.228, "y1": 234, "x2": 0.438, "y2": 0.945}` -> **mixed**
  - `raw.second_person_bbox = {"x1": 0.595, "y1": 0.234, "x2": 0.825, "y2": 0.935}` -> normalized
- `videoplayback (4) / short_001`
  - `raw.person_bbox = {"x1": 0.0, "y1": 198, "x2": 339, "y2": 896}` -> **mixed**
  - `raw.second_person_bbox = {"x1": 734, "y1": 198, "x2": 951, "y2": 771}` -> pixelish
- All other clips in the 9-clip set have only `pixelish` single-person boxes or normalized split regions after instruction construction.

Pattern:

- The two failing `split_two_persons` clips are the only clips in the 9-clip set
  with **mixed-scale raw bboxes**.
- The working controls do not show mixed-scale raw bboxes.
- This strongly suggests **Mode A root cause = bbox coordinate normalization /
  validation** rather than a generic split-layout renderer failure.

## videoplayback (3) / short_002 - "The 'Crab in a Tub' Ecosystem"

Failure mode: **Mode A** (bad split regions)

Rendered failure evidence:
  start frame shows: the top half is mostly chair / torso / empty lower-band crop rather than a clean speaker face crop
  5s frame shows: the stacked layout persists; the bottom half shows the right speaker while the top half still feels anchored too low
  why this is a layout failure: the clip content is readable, but the split layout frames the left speaker from the lower body band instead of the face / upper body

Layout decision (`layout_vision.json`):
  chosen layout: `split_two_persons`
  top_band_ratio: `0.5`
  person_x_norm: `0.29`
  split regions:
    - `split_person_region = {"x1": 0.228, "y1": 0.65, "x2": 0.438, "y2": 0.945}`
    - `split_second_person_region = {"x1": 0.595, "y1": 0.234, "x2": 0.825, "y2": 0.935}`
  raw bbox values:
    - `person_bbox = {"x1": 0.228, "y1": 234, "x2": 0.438, "y2": 0.945}`
    - `second_person_bbox = {"x1": 0.595, "y1": 0.234, "x2": 0.825, "y2": 0.935}`
  coordinate scales:
    - `person_bbox` is mixed (`y1` pixel-scale, others normalized-style)
    - `second_person_bbox` is normalized

Keyframe evidence:
  midpoint keyframe shows: a normal two-person podcast setup with the left speaker seated on the left and the right speaker seated on the right
  does the chosen layout match the keyframe: yes in coarse class (two people), no in region geometry

Tracking evidence:
  no person-tracking is used for `split_two_persons`

Renderer path:
  code path used: `layout_vision._instruction_from_gemini_json` -> `layouts.plan_split_two_persons` -> `_bbox_strip`
  does renderer honor stored regions literally: yes
  if yes, are the stored regions themselves invalid: yes; the left split region is normalized to the lower 35% of frame height (`y1 = 0.65`), which explains the torso/chair crop

Stage that introduced the failure:
  `bbox normalization / layout acceptance`

## videoplayback (4) / short_001 - "The AI self-improvement loop"

Failure mode: **Mode A** (bad split regions)

Rendered failure evidence:
  start frame shows: huge blue empty band in the middle with only thin left/right edge slices of people
  5s frame shows: same split composition, still dominated by empty stage and cropped speaker edges
  why this is a layout failure: the split layout is faithfully rendering two narrow edge regions that omit the actual speaking subject area

Layout decision (`layout_vision.json`):
  chosen layout: `split_two_persons`
  top_band_ratio: `0.5`
  person_x_norm: `0.090625`
  split regions:
    - `split_person_region = {"x1": 0.0, "y1": 0.198, "x2": 0.339, "y2": 0.896}`
    - `split_second_person_region = {"x1": 0.734, "y1": 0.198, "x2": 0.951, "y2": 0.771}`
  raw bbox values:
    - `person_bbox = {"x1": 0.0, "y1": 198, "x2": 339, "y2": 896}`
    - `second_person_bbox = {"x1": 734, "y1": 198, "x2": 951, "y2": 771}`
  coordinate scales:
    - `person_bbox` is mixed (`x1 = 0.0`, the rest pixel-style)
    - `second_person_bbox` is pixelish on a 1000-grid assumption

Keyframe evidence:
  midpoint keyframe shows: a live panel with left participant, center moderator, and right participant visible
  does the chosen layout match the keyframe: only partially; `split_two_persons` chooses the left and right edge participants and excludes the actual center stage composition

Tracking evidence:
  no person-tracking is used for `split_two_persons`

Renderer path:
  code path used: `layout_vision._instruction_from_gemini_json` -> `layouts.plan_split_two_persons` -> `_compute_seam` / `_bbox_strip`
  does renderer honor stored regions literally: yes
  if yes, are the stored regions themselves invalid: effectively yes; the accepted regions produce narrow left/right strips that render as empty-band slop in 9:16

Stage that introduced the failure:
  `bbox normalization / layout acceptance`

## videoplayback (4) / short_003 - "AI and the job market"

Failure mode: **Mode B** (tracking contamination)

Rendered failure evidence:
  start frame shows: audience / empty stage dominates the crop instead of the speaker
  5s frame shows: the opening remains off-target before the crop later recovers
  why this is a layout failure: the clip is classified as a single-speaker `sit_center` scene, but the crop opens hard-left on a non-speaker region

Layout decision (`layout_vision.json`):
  chosen layout: `sit_center`
  top_band_ratio: `0.5`
  person_x_norm: `0.435`
  person_tracking samples:
    - midpoint keyframe at `41.638s` -> `layout = sit_center`, `center_x_norm = 0.435`
    - tracking frame at `8.328s` -> `layout = split_two_persons`, `center_x_norm = 0.14`
    - later tracking frames from `16.655s` onward mostly return `sit_center` with centers around `0.425-0.52`
  split regions: none in the final instruction
  raw bbox values:
    - midpoint `person_bbox = {"x1": 218, "y1": 59, "x2": 842, "y2": 1000}`

Keyframe evidence:
  midpoint keyframe shows: one clear primary speaker centered in a panel setting
  does the chosen layout match the keyframe: yes

Tracking evidence:
  sample at `t = 8.328s` chose `split_two_persons` with `center_x_norm = 0.14`
  midpoint decision: `sit_center` with `center_x_norm = 0.435`
  outlier samples that disagree with midpoint:
    - the earliest tracking sample at `8.328s` is split-style and far left
  does an outlier sample explain the bad opening crop: yes; `_tracking_points_from_centers` keeps the earliest sample and then inserts it at `t = 0`, so the opening crop is seeded from the wrong layout class

Renderer path:
  code path used: `layout_vision._infer_person_tracking` -> `_tracking_points_from_centers` -> `layouts.plan_sit_center`
  does renderer honor stored tracking literally: yes
  if yes, are the stored tracking points themselves invalid: yes for the opening segment; the first accepted tracking point comes from a split-style sample that disagrees with the midpoint scene class

Stage that introduced the failure:
  `tracking acceptance`

## videoplayback (4) / short_002 - "Technological adolescence and risks" (control)

Failure mode: control

Rendered evidence:
  start frame shows: centered speaker crop
  5s frame shows: still centered on the same speaker; no obvious opening slop
  why this is a control: this clip is postable after Ticket B and the layout does not visibly fail

Layout decision (`layout_vision.json`):
  chosen layout: `sit_center`
  midpoint keyframe: `center_x_norm = 0.5`
  tracking samples:
    - most samples are `sit_center` near `0.50-0.58`
    - two later samples (`23.04s` and `46.08s`) return `split_two_persons`-style detections (`0.316`, `0.300`)

Key takeaway:
  the existence of a split-style tracking outlier alone is not enough to break a clip
  the bad case in `(4)/003` is specifically that the earliest accepted sample is the outlier and becomes the inserted `t=0` anchor

## videoplayback (5) / short_003 - "White Collar Bloodbath" (control)

Failure mode: control

Rendered evidence:
  start frame shows: stable centered speaker crop
  5s frame shows: still on speaker, no audience slop
  why this is a control: the clip remains postable and the layout works despite one noisy tracking sample

Layout decision (`layout_vision.json`):
  chosen layout: `sit_center`
  midpoint keyframe: `center_x_norm = 0.42`
  tracking samples:
    - one split-style sample at `13.247s` (`center_x_norm = 0.3046875`)
    - all earlier / later anchor points are normal single-speaker detections

Key takeaway:
  later split-style tracking noise is survivable when the first accepted sample matches the midpoint scene class

## Pattern Summary

### Mode A Analysis

- Affected clips in this evaluation set: `videoplayback (3) / short_002`, `videoplayback (4) / short_001`
- Shared signature:
  - final layout is `split_two_persons`
  - the renderer faithfully applies the stored split regions
  - the accepted split regions are visibly wrong for 9:16 framing
  - the raw vision payload includes mixed-scale bbox values
- Important nuance:
  - the renderer is not inventing the bad crop
  - `layout_vision._normalize_bbox_payload` accepts mixed-scale raw boxes and clamps them into valid normalized `BoundingBox` objects
  - `_instruction_from_gemini_json` then happily emits `split_two_persons` because both parsed boxes are non-null

### Mode B Analysis

- Affected clip in this evaluation set: `videoplayback (4) / short_003`
- Shared signature:
  - final midpoint decision is `sit_center`
  - an early tracking sample returns `split_two_persons`
  - `_infer_person_tracking` accepts the sample anyway because it only asks `_person_center_x_from_data(...)` for a center and ignores the sample's layout class
  - `_tracking_points_from_centers` does not guard the first sample, then inserts that first sample's x-position at `t = 0`
- Control evidence:
  - `(4)/002` and `(5)/003` also contain split-style tracking outliers
  - they stay usable because the first accepted sample is sane, so the opening crop is not poisoned

## Mode A Root Cause Conclusion

Mode A is an **upstream layout-acceptance bug**, not a renderer bug. The pipeline accepts mixed-scale split bboxes as if they were trustworthy, normalizes them, and emits `split_two_persons` instructions that are internally valid but semantically wrong. The visible empty-band render is just the literal consequence of those bad accepted regions.

## Mode A Fix Proposal

Validate split-layout bbox scale consistency before accepting the layout:

1. If a split-person bbox mixes normalized-style and pixel-style coordinates inside the same bbox, reject it.
2. If a `split_two_persons` scene is missing a valid left or right bbox after validation, downgrade the layout to `sit_center`.
3. Prefer **detection + fallback** over guessing at auto-normalization. When the model gives contradictory coordinate scales, the honest thing is to reject the split layout.

Fix point:
  `layout_vision` parsing / instruction construction, not the renderer.

## Mode B Root Cause Conclusion

Mode B is a **tracking-sample acceptance bug**. For one-person layouts, the tracking pipeline accepts centers from samples whose raw layout class disagrees with the midpoint class. When that disagreeing sample is the first accepted sample, `_tracking_points_from_centers` seeds the opening crop from the wrong layout family and the clip opens on audience / empty-stage slop.

## Mode B Fix Proposal

Filter tracking samples before they become `TimedCenterPoint`s:

1. For midpoint-classified `sit_center` / `zoom_call_center` clips, reject tracking samples whose raw layout is a split layout.
2. Keep the existing tracking sampler itself unchanged; only change which samples are admitted.
3. Leave later inlier single-speaker samples intact so tracking still works on legitimate reframes.

Fix point:
  tracking-sample acceptance inside `layout_vision`, not render-time tracking consumption.
