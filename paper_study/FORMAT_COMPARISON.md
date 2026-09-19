# Output-format comparison — does the asked field set change detection quality?

**Status: complete (2026-09-19).** Three output formats, one model, one
testbed, one effort level — only the per-fruit field set varies. All three
runs PASS `verify_run.py` (1479 checks each); 420/420 calls parsed
schema-valid; controls 0-2.2 px.

## Design

| format | per-fruit fields | style name | rationale |
| --- | --- | --- | --- |
| `full9` | bbox, polygon, redness_pct, occlusion_pct, calyx_visible, peduncle_visible, graspable, confidence_pct, description | `fruit_inventory_segmentation` | the strawdi_segmentation census — the blog study's standard; forces the model to look at and verbalise every fruit |
| `seg3` | bbox, polygon, confidence_pct | `fruit_box_polygon` | the minimal format that fits a SEGMENTATION task |
| `box2` | bbox, confidence_pct | `fruit_box_minimal` | the minimal format that fits a pure DETECTION task |

Held constant across variants: model (glm-5.3-flash via claude CLI), reasoning
effort (max), census rules ("EVERY fruit, whatever its colour, do not filter
by ripeness, include partly occluded, largest first, do not invent"),
coordinate-system preamble, per-dataset scene/fruit wording, delivery path,
and the testbed. The prompts differ ONLY in the field list, the field-count
sentence, and the JSON example. All three prompts are recorded verbatim in
each run's `responses.jsonl`.

**Testbed:** `manifest20.json` — 20 frames per dataset (140 total) from each
dataset's own test split, same deterministic selection rule as the 3-frame
testbed (this manifest is TEMPORARY, built for this comparison; the committed
default stays `manifest.json` at 3/dataset).

**Comparison metric:** box F1@IoU-0.5 per dataset (definable for every
format), with the IoU ladder (0.25/0.75) and AP recorded per run; mask
F1@0.5 for `full9`/`seg3` on the three mask-GT datasets (strawdi,
minneapple-mirror, wgisd). `box2` has no polygons, so mask metrics are n/a
by design. Parse/schema rates and tokens/cost compared as secondary
outcomes.

## Results

Run dirs (`paper_study/runs/`): full9 `20260918-232733-…-cmp20`, seg3
`20260919-004023-…-seg3-cmp20`, box2 `20260919-013223-…-box2-cmp20`.
Box F1@IoU-0.5 (P/R), 20 frames per dataset:

| dataset | box F1@0.5 full9 | box F1@0.5 seg3 | box F1@0.5 box2 | mask F1@0.5 full9 | mask F1@0.5 seg3 |
| --- | --- | --- | --- | --- | --- |
| strawdi | 0.628 (0.69/0.58) | **0.694** (0.76/0.64) | **0.694** (0.76/0.64) | 0.601 | **0.640** |
| wgisd | 0.438 (0.58/0.35) | 0.450 (0.58/0.37) | **0.465** (0.56/0.40) | 0.283 | **0.304** |
| acfr_mangoes | 0.415 (0.50/0.35) | 0.408 (0.51/0.34) | **0.449** (0.54/0.38) | n/a | n/a |
| kfuji | 0.384 (0.49/0.31) | **0.420** (0.57/0.33) | 0.407 (0.53/0.33) | n/a | n/a |
| acfr_apples | 0.209 (0.28/0.17) | **0.218** (0.28/0.18) | 0.154 (0.20/0.13) | n/a | n/a |
| minneapple | 0.193 (0.22/0.17) | 0.198 (0.21/0.19) | **0.216** (0.20/0.23) | 0.160 | **0.172** |
| acfr_almonds | **0.030** (0.04/0.02) | 0.025 (0.04/0.02) | 0.006 (0.01/0.00) | n/a | n/a |
| **all 140 frames** | 0.285 (0.35/0.24) | 0.297 (0.35/0.26) | **0.300** (0.33/0.27) | | |

Secondary outcomes:

| secondary | full9 | seg3 | box2 |
| --- | --- | --- | --- |
| calls ok / total | 140/140 (2 in-run retries) | 140/140 (0) | 140/140 (0) |
| tokens (in+out, non-cached) | 942,908 | 748,622 (−21%) | 400,116 (−58%) |
| billed cost | $19.78 | $15.05 (−24%) | $7.06 (−64%) |
| wall time (jobs 3) | 70 min | 53 min | 36 min |

Cost note: usage is the CLI-reported token count (input+output, cache reads
excluded from the total, per convention); `cost_usd` is the endpoint's own
per-call charge — the two agree exactly at $5/$0.5/$25 per MTok
in/cached/out (same audit as `DATASETS.md`).

## Reading rules (written before the numbers, so they cannot be fitted)

1. A format difference is only meaningful when it exceeds the run-to-run
   variance of this model. Reference points: the blog study observed ~10 px
   picking-point jitter between identical calls, and the 3-frame testbed vs
   the 100-frame val batch differed by several F1 points on StrawDI — treat
   single-digit F1 deltas as suggestive, not conclusive, at 20 frames/dataset.
2. MinneApple FPs are structurally inflated (GT deliberately omits
   background-row/ground apples while the prompt demands every apple) —
   compare its PRECISION across formats, not against other datasets.
3. Mask F1 compares only `full9` vs `seg3` (same polygon contract); the
   polygon-fidelity ceiling applies to both equally.
4. Cheaper formats that score the same are strictly better for pipeline
   purposes; the cost row is therefore part of the verdict, not a footnote.

## Verdict

1. **The asked field set does not materially change detection quality.**
   Overall box F1@0.5 moves 0.285 → 0.297 → 0.300 (full9 → seg3 → box2) —
   a +1.5-point spread, below the pre-registered noise bar at 20
   frames/dataset. The nine census fields neither help nor hurt the boxes
   the model reports.
2. **Where a difference shows, the minimal formats are ahead.** StrawDI is
   the clearest: 0.628 full9 vs 0.694 for BOTH minimal formats (+6.6 pts —
   suggestive, not conclusive); mask F1 likewise 0.601 → 0.640 (seg3).
   seg3 edges full9 on mask F1 on all three mask datasets. Nothing anywhere
   favours full9 beyond noise (almonds full9 0.030 vs box2 0.006 is a
   difference between near-zero and near-zero — 3 vs 2 matched boxes).
3. **Minimal formats are much cheaper and faster** (box2: −64% cost, −58%
   tokens, half the wall time of full9; seg3 in between) with ZERO parse or
   schema failures. For measurement purposes they are the better
   instrument: same or better numbers, less money, less exposure to
   output-budget truncation.
4. **What full9 buys is content, not accuracy**: redness/occlusion/graspable
   and free-text descriptions — the picking-oriented reporting the blog
   study is about. That is a product decision, not a scoring one.

**Recommendation:** headline paper numbers on the minimal format that fits
each dataset's task — `seg3` where the task is segmentation (strawdi,
minneapple, wgisd: masks stay scoreable), `box2` for the box-only datasets
(ACFR ×3, KFuji); run `full9` only where the qualitative census fields are
themselves the deliverable. Follow-up worth running once: repeat the strawdi
+6.6-pt observation on a different frame set to confirm it is real and not
selection noise.
