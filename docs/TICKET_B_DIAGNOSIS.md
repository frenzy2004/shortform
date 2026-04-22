# Ticket B Diagnosis

This diagnosis uses the saved evidence runs:

- `.humeo_realuse_videoplayback3`
- `.humeo_realuse_videoplayback4`
- `.humeo_realuse_videoplayback5`

The goal was to find which stage actually causes the cut-quality failures seen in the
9-clip evaluation set before changing code.

## videoplayback (3) / short_001 — "The Founder's Doom Loop"

Rejected boundary evidence: `"to load up your"` (start) / `"makes you yeah sure"` (end)

Raw selection (clips.json):
  start_time_sec: 2208.00
  end_time_sec: 2262.30
  transcript at start: `"like it stacks up You stack momentum [Now] the thing about like going after like"`
  transcript at end: `"all in line it makes you yeah [sure] Okay that's human psychology right Like people"`
  raw boundary quality: mid-sentence / mid-sentence

Hook detection (hooks.json):
  hook_start_sec (clip-relative): 10.41
  hook_end_sec (clip-relative): 14.15
  absolute timestamp at hook start: 2218.41
  transcript at hook start: `"is that it gives the founder confidence [If] the founder only has one VC who's"`
  hook detector moved anchor: improved internally, but hook metadata does not move export boundaries

Pruning (prune.json):
  trim_start_sec: 4.30
  trim_end_sec: 0.00
  final render start: 2212.30
  final render end: 2262.30
  transcript at final render start: `"or 10 more ways to be able [to] load up your fundraising odds of success"`
  transcript at final render end: `"all in line it makes you yeah [sure] Okay that's human psychology right Like people"`
  pruning effect on boundaries: worse; the clip already started badly and the inward-only start trim made it more fragmentary

Stage that introduced the bad boundary: selection

## videoplayback (3) / short_002 — "The Crab Bucket Theory"

Rejected boundary evidence: `"And like oh we've"` (start) / `"of this"` (end)

Raw selection (clips.json):
  start_time_sec: 1306.70
  end_time_sec: 1393.90
  transcript at start: `"just going to tear each other down [It's] like when I was 15 years old"`
  transcript at end: `"crab We all be happy crabs right [So] okay Yeah there you go Okay please"`
  raw boundary quality: mid-sentence / clean

Hook detection (hooks.json):
  hook_start_sec (clip-relative): 26.21
  hook_end_sec (clip-relative): 31.91
  absolute timestamp at hook start: 1332.91
  transcript at hook start: `"Okay let's put the crabs in a [That] night every crab started breaking each other"`
  hook detector moved anchor: improved internally, but still only as metadata

Pruning (prune.json):
  trim_start_sec: 11.55
  trim_end_sec: 5.05
  final render start: 1318.25
  final render end: 1388.85
  transcript at final render start: `"put a bunch of crabs into this [And] we're like oh we've caught so many"`
  transcript at final render end: `"you know Cause there's this abundance of [this] and this beach beautiful nature You want"`
  pruning effect on boundaries: worse; the raw bad start stayed bad and the raw clean end became a clipped fragment

Stage that introduced the bad boundary: selection

## videoplayback (3) / short_003 — "The Ladder Strategy"

Rejected boundary evidence: `"There's this concept that"` (start) / `"have too much time"` (end)

Raw selection (clips.json):
  start_time_sec: 2282.20
  end_time_sec: 2365.10
  transcript at start: `"I can give you one last tip [This] is for everyone for fundraising okay There's"`
  transcript at end: `"just take up on this thing right [So] that VC would be like wow you"`
  raw boundary quality: mid-sentence / mid-sentence

Hook detection (hooks.json):
  hook_start_sec (clip-relative): 2.03
  hook_end_sec (clip-relative): 7.71
  absolute timestamp at hook start: 2284.23
  transcript at hook start: `"This is for everyone for fundraising okay [There's] this concept that I and we developed"`
  hook detector moved anchor: unchanged-to-worse for boundary quality; the hook itself still begins inside a running thought

Pruning (prune.json):
  trim_start_sec: 1.78
  trim_end_sec: 2.61
  final render start: 2283.98
  final render end: 2362.49
  transcript at final render start: `"tip This is for everyone for fundraising [okay] There's this concept that I and we"`
  transcript at final render end: `"but we don't have too much time [But] I just you know wanted to just"`
  pruning effect on boundaries: worse; both ends remained incomplete and the end got cut earlier

Stage that introduced the bad boundary: selection

## videoplayback (4) / short_001 — "The AI Self-Improvement Loop"

Rejected boundary evidence: `"So the mechanism whereby"` (start) / `"longer than that"` (end)

Raw selection (clips.json):
  start_time_sec: 80.20
  end_time_sec: 148.90
  transcript at start: `"turn out to be that far off [So] the mechanism whereby I imagined it would"`
  transcript at end: `"guess that this goes faster than people [imagine] And that key element of code and"`
  raw boundary quality: clean / mid-sentence

Hook detection (hooks.json):
  hook_start_sec (clip-relative): 18.28
  hook_end_sec (clip-relative): 26.20
  absolute timestamp at hook start: 98.48
  transcript at hook start: `"would increase the speed of model development [We] are now in terms of the models"`
  hook detector moved anchor: improved internally, but export still uses the raw window

Pruning (prune.json):
  trim_start_sec: 0.00
  trim_end_sec: 5.66
  final render start: 80.20
  final render end: 143.24
  transcript at final render start: `"turn out to be that far off [So] the mechanism whereby I imagined it would"`
  transcript at final render end: `"see how it could take longer than [that] But if I had to guess I"`
  pruning effect on boundaries: worse; the bad tail was shortened but still ends before the thought resolves

Stage that introduced the bad boundary: selection

## videoplayback (4) / short_002 — "Technological Adolescence Risks"

Rejected boundary evidence: `"you there's this scene"` (start) / `"misuse them"` (end)

Raw selection (clips.json):
  start_time_sec: 656.70
  end_time_sec: 733.50
  transcript at start: `"And the way I framed it was [you] there's this scene from Carl Sagan's Contact"`
  transcript at end: `"we make sure that individuals don't misuse [them] Right I have worries about things like"`
  raw boundary quality: mid-sentence / mid-sentence

Hook detection (hooks.json):
  hook_start_sec (clip-relative): 27.52
  hook_end_sec (clip-relative): 33.24
  absolute timestamp at hook start: 684.22
  transcript at hook start: `"would ask how did you do it [How] did you manage to get through this"`
  hook detector moved anchor: improved internally, but no boundary repair was applied

Pruning (prune.json):
  trim_start_sec: 0.00
  trim_end_sec: 0.00
  final render start: 656.70
  final render end: 733.50
  transcript at final render start: `"And the way I framed it was [you] there's this scene from Carl Sagan's Contact"`
  transcript at final render end: `"we make sure that individuals don't misuse [them] Right I have worries about things like"`
  pruning effect on boundaries: unchanged; the bad raw boundaries pass straight through

Stage that introduced the bad boundary: selection

## videoplayback (4) / short_003 — "The Fermi Paradox & AI"

Rejected boundary evidence: `"me is the Fermi"` (start) / `"going to next"` (end)

Raw selection (clips.json):
  start_time_sec: 1709.10
  end_time_sec: 1766.80
  transcript at start: `"sort of strongest argument for Doomerism to [me] is the Fermi paradox the idea that"`
  transcript at end: `"write as humanity what's going to happen [next] This could be a great discussion but"`
  raw boundary quality: mid-sentence / mid-sentence

Hook detection (hooks.json):
  hook_start_sec (clip-relative): 0.00
  hook_end_sec (clip-relative): 2.93
  absolute timestamp at hook start: 1709.10
  transcript at hook start: `"sort of strongest argument for Doomerism to [me] is the Fermi paradox the idea that"`
  hook detector moved anchor: no change; it simply re-affirmed the already-bad opening

Pruning (prune.json):
  trim_start_sec: 0.00
  trim_end_sec: 0.00
  final render start: 1709.10
  final render end: 1766.80
  transcript at final render start: `"sort of strongest argument for Doomerism to [me] is the Fermi paradox the idea that"`
  transcript at final render end: `"write as humanity what's going to happen [next] This could be a great discussion but"`
  pruning effect on boundaries: unchanged; the bad raw boundaries pass straight through

Stage that introduced the bad boundary: selection

## videoplayback (5) / short_001 — "AI's exponential growth"

Rejected boundary evidence: `"the amount of power"` (start) / `"all Claude He's"` (end)

Raw selection (clips.json):
  start_time_sec: 50.60
  end_time_sec: 105.00
  transcript at start: `"it is this very smooth exponential process [Just] like in the 90s you saw Moore's"`
  transcript at end: `"in the last two It's all Claude [He's] edited it he's looked at it but"`
  raw boundary quality: mid-sentence / mid-sentence

Hook detection (hooks.json):
  hook_start_sec (clip-relative): 8.08
  hook_end_sec (clip-relative): 12.84
  absolute timestamp at hook start: 58.68
  transcript at hook start: `"double every 12 months or every 18 [months] We have a Moore's law like law"`
  hook detector moved anchor: unchanged-to-worse for outer boundary quality; the localized hook still sits inside a running explanation

Pruning (prune.json):
  trim_start_sec: 4.40
  trim_end_sec: 0.00
  final render start: 55.00
  final render end: 105.00
  transcript at final render start: `"the 90s you saw Moore's law that [the] amount of computing power would double every"`
  transcript at final render end: `"in the last two It's all Claude [He's] edited it he's looked at it but"`
  pruning effect on boundaries: worse; the start became a smaller fragment because the min-duration floor forced an inward trim

Stage that introduced the bad boundary: selection

## videoplayback (5) / short_002 — "AI and national security"

Rejected boundary evidence: `"the national security implications"` (start) / `"advised"` (end)

Raw selection (clips.json):
  start_time_sec: 380.00
  end_time_sec: 433.90
  transcript at start: `"a big mistake to ship these chips [You] know the analogy I thought of if"`
  transcript at end: `"Akin to  Yeah it's not well [advised] Another issue in this you know you"`
  raw boundary quality: clean / mid-sentence

Hook detection (hooks.json):
  hook_start_sec (clip-relative): 17.20
  hook_end_sec (clip-relative): 22.54
  absolute timestamp at hook start: 397.20
  transcript at hook start: `"country of geniuses and a data center [right] So imagine 100 million people smarter than"`
  hook detector moved anchor: improved internally, but the export still starts before and ends after that cleaner beat

Pruning (prune.json):
  trim_start_sec: 3.90
  trim_end_sec: 0.00
  final render start: 383.90
  final render end: 433.90
  transcript at final render start: `"I thought of if you think about [the] national security implications of building models that"`
  transcript at final render end: `"Akin to  Yeah it's not well [advised] Another issue in this you know you"`
  pruning effect on boundaries: worse; pruning turned a clean raw opening into a fragment while leaving the bad ending untouched

Stage that introduced the bad boundary: selection

## videoplayback (5) / short_003 — "The AI economic bubble"

Rejected boundary evidence: `"And you don't want"` (start) / `"process"` (end)

Raw selection (clips.json):
  start_time_sec: 605.60
  end_time_sec: 662.90
  transcript at start: `"buy compute to serve all that revenue [And] you don't want to buy too much"`
  transcript at end: `"in all of the companies to this [process] Whenever you have an upside whose amount"`
  raw boundary quality: mid-sentence / mid-sentence

Hook detection (hooks.json):
  hook_start_sec (clip-relative): 0.00
  hook_end_sec (clip-relative): 4.94
  absolute timestamp at hook start: 605.60
  transcript at hook start: `"buy compute to serve all that revenue [And] you don't want to buy too much"`
  hook detector moved anchor: no change; it re-used the already-bad opening

Pruning (prune.json):
  trim_start_sec: 0.00
  trim_end_sec: 0.00
  final render start: 605.60
  final render end: 662.90
  transcript at final render start: `"buy compute to serve all that revenue [And] you don't want to buy too much"`
  transcript at final render end: `"in all of the companies to this [process] Whenever you have an upside whose amount"`
  pruning effect on boundaries: unchanged; the bad raw boundaries pass straight through

Stage that introduced the bad boundary: selection

## Pattern Summary

- Selection is the main culprit. All 9 evaluation clips already had at least one broken boundary in `clips.json` before hook detection or pruning touched them.
- Hook detection is not introducing the final bad cuts. In several clips it found a cleaner internal beat, but it only writes metadata for Stage 2.5 clamp protection and never changes the outer export window.
- Pruning improves 0 of 9 clips.
  - 3 of 9 (`videoplayback (4) / short_002`, `videoplayback (4) / short_003`, `videoplayback (5) / short_003`) pass the bad raw boundaries through unchanged with zero trims.
  - 6 of 9 apply inward trims, but those trims either preserve the broken raw window or make one side worse.
- The worst cases share the same structural limitation:
  - the clip already starts or ends inside a running thought
  - Stage 2.5 can only trim inward
  - the existing Stage 2.5 segment snap only snaps trims that already exist, and it reverts when snapping would violate the 50s minimum
  - so the system has no way to expand slightly earlier or later to recover the nearest complete thought
- `transcript.json` is usable for snapping, but not in the naive "word punctuation" sense:
  - `segments[].words[]` gives precise word timings and silence gaps
  - punctuation is more reliable in `segments[].text` than in the individual `word` tokens
  - this means boundary snapping should use transcript timing plus segment punctuation / silence-gap cues, not raw word tokens alone

## Root Cause Conclusion

The dominant failure is not ranking and not layout. It is that Stage 2 selection frequently chooses semantically interesting windows whose export boundaries are already inside running thoughts. Downstream stages cannot repair that because hook detection only annotates metadata and content pruning only trims inward. In several clips, pruning makes the start worse because the 50s floor blocks snapping to the next clean segment boundary, while the current pipeline has no ability to extend the opposite edge to compensate.

## Fix Proposal

Implement a small post-pruning boundary snap inside the existing Stage 2.5 flow that operates on the final render window, not just on inward trims:

1. Start from the actual render bounds (`start_time_sec + trim_start_sec`, `end_time_sec - trim_end_sec`).
2. Search nearby transcript boundaries for a cleaner start and end.
3. Allow both inward and outward movement within the ticket's search windows while preserving the 50-90s duration contract.
4. Use transcript timings plus segment punctuation / silence-gap cues to score candidate boundaries.
5. If no valid in-bounds boundary exists, leave the clip unchanged and log a warning.

This is the smallest change that matches the evidence. A prompt rewrite to hook detection or pruning would still leave the system unable to recover context that sits just outside the selected window.

## Post-fix Follow-up

`videoplayback (5) / short_002` (`"The AI Exponential"`) remained a mid-thought opener after Ticket B because there was no clean start boundary inside the allowed `+-3s` snap window. The pre-snap render window from `.humeo_ticketb_videoplayback5/clips.json` plus `.humeo_ticketb_videoplayback5/prune.json` was `46.30s-122.80s`, and replaying `snap_render_windows_to_sentence_boundaries(...)` against the saved transcript leaves the post-snap window unchanged at `46.30s-122.80s` while logging `no valid clean sentence boundary found for start@46.30s`. The nearby start-side transcript segments at `43.340-46.260` (`"Anthropos co-founders were among the first to document it,"`) and `46.260-50.220` (`"is this very smooth exponential process."`) are both continuations of an earlier sentence, not valid sentence starts: neither has terminal punctuation nor a `>=0.5s` silence gap before it. The end boundary is already clean at `122.80s` (`"almost entirely with Claude Code."`). Conclusion: this is outcome (1), not a Ticket B bug. No recoverable clean start existed in range, so the snap correctly left the clip unchanged.
