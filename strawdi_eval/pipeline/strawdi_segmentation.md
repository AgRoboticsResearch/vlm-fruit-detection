# strawdi_segmentation — full-scene strawberry inventory with visible-surface polygons, scored as segmentation on StrawDI_Db1

Standalone, **model-agnostic** spec of the third perception pipeline. Its
siblings are
[`../../vlm_eval/pipeline/full_detection.md`](../../vlm_eval/pipeline/full_detection.md)
(the SROI picking-point pipeline) and
[`strawdi_detection.md`](strawdi_detection.md) (the StrawDI box pipeline);
this one keeps the detection pipeline's **output standard intact** — the same
unbiased eight-field census — and adds a ninth field, `polygon`: an ordered
vertex outline of each fruit's **visible surface**. The answer is scored as
multi-instance **segmentation** against the raw StrawDI ground-truth masks
(the label id-map PNGs the box pipeline only used to derive boxes), with the
reported bboxes scored by the detection scorer unchanged as a cross-check.

**Status: v0.1.** Validated on the full val batch (glm-5.3-flash via claude
CLI, 2026-09-16, 100/100 calls parsed, verifier PASS over 1418 checks); see
§6.

Implementation lives in `../seg/` (`run_segmentation_eval.py`, `lib/prompt.py`,
`lib/seg_scoring.py`, `lib/render.py`, `schema/inventory_segmentation_schema.json`,
`verify_strawdi_seg_run.py`, `test_seg_scoring.py`) — deliberately **outside
the detection harness's fingerprint glob** (top-level `strawdi_eval/*.py`,
`lib/`, `schema/`), same trick as `agent_bridge/`, so the two pipelines'
provenance stay independent. The model-facing machinery — provider stack,
reply parser, synthetic control, measurement invariants — is imported
unchanged from `../../vlm_eval/`; the manifest is the detection pipeline's,
reused as-is. Every record states **both** harness fingerprints.

---

## 1. What this pipeline does

One StrawDI frame in → one complete, **unbiased** inventory of **every**
strawberry visible in it, whatever its colour, as **nine** per-fruit fields
(the detection eight plus `polygon`) — then that inventory is **scored as
segmentation**: every reported polygon is rasterised and matched against the
ground-truth instance masks of that frame. The model works from pixels
alone: no detector, segmenter, tracker or any other model participates.

What is scored and what is not:

* **Scored, primary** — the polygons, as multi-instance segmentation (P/R/F1
  at mask-IoU thresholds, mask AP, count error, size-stratified recall).
* **Scored, cross-check** — the reported `bbox`es, by the detection
  pipeline's scorer **unchanged** (`strawdi_eval/lib/scoring.py`), so numbers
  line up with the box pipeline for the same model.
* **Asked, never scored** — `redness_pct`, `occlusion_pct`, `calyx_visible`,
  `peduncle_visible`, `graspable`, `confidence_pct`, `description` (same
  role as in the box pipeline: unbiased census + report diagnostics).
* **Not asked at all** — `picking_point` and `target_index`; a reply that
  volunteers either is rejected wholesale by the schema
  (`additionalProperties: false`).

## 2. Input / output contract

**Input:** the box pipeline's manifest, reused as-is — StrawDI_Db1 `val`
frames (1008×756, snapshotted in `data/frames/`, delivered unchanged inside
the 1280 px no-resize ceiling) plus the GT provenance. The seg pipeline adds
no manifest step of its own.

**Ground truth:** the raw label id-map PNGs (grayscale, 0 = background,
1..N = instance, one bool mask per id, sorted by id — the same order as the
manifest's `gt_boxes`). Masks are read **after** the model call,
sha256-guarded against the manifest's `label_sha256`; the verifier re-reads
and re-derives everything and fails on drift. The label PNGs are never
inputs and never leave the removable mount.

**Output:** per frame, one JSON object with a single `strawberries` array —
**nine fields per fruit, no more, no fewer**: the detection pipeline's eight
fields verbatim (`bbox` whole-fruit box, `redness_pct`, `occlusion_pct`,
`calyx_visible`, `peduncle_visible`, `graspable`, `confidence_pct`,
`description`; see the sibling spec's table) plus:

| field | type | meaning |
| --- | --- | --- |
| `polygon` | 3–32 pairs `[x,y]` ints | ordered outline of the fruit's **visible surface**, walked once (cw or ccw, start anywhere); follows occluder edges — never extrapolates the hidden shape (unlike `bbox`); same fruit-body rule as `bbox`; extent must agree with the visible fruit |

**Prompt:** `inventory_segmentation` (`seg/lib/prompt.py`) — the detection
prompt with the polygon field inserted after `bbox` and "eight" → "nine",
otherwise verbatim, parameterised to the frame size, printed in every
report.

**Run statuses:** `ok` · `schema_invalid` / `parse_error` / `empty` /
`refused` / `exec_error` — explicit failures carrying **no** mask or box
numbers, never zeros. Retry policy mirrors the box pipeline exactly: one
recorded resample at the same effort for `schema_invalid`, one low-effort
retry for `empty`. A record whose label PNG cannot be loaded/verified keeps
its answer but carries `gt_mask_error` and no mask numbers (the box
cross-check still scores — its GT rides in the record).

## 3. Scoring

* **Rasterisation:** each `polygon` fills a bool HxW mask (PIL polygon
  semantics, whole-pixel vertices). A vertex outside `[0,W)×[0,H)` or an
  empty raster excludes the prediction from matching and counts it as FP —
  the box scorer's out-of-frame/degenerate rule, transplanted.
* **Matching:** greedy one-to-one; predictions ranked `confidence_pct` desc
  (ties → inventory order); each prediction claims the unmatched GT mask
  with the highest **mask IoU** (ties → lowest GT index); match at IoU ≥ t;
  recomputed independently at t ∈ {0.25, 0.5, 0.75}.
* **Headline: F1@mask-IoU-0.5** (with P and R). Mask AP@50 / mAP@[.50:.95]
  secondary (same confidence-clumping caveat as the box pipeline).
* **Cross-check:** the detection scorer, unchanged, on the asked `bbox`es —
  its whole-fruit-vs-visible-surface semantics gap applies there (and is
  stated); **the polygon metric has no such gap** — the prompt asks for the
  visible surface and GT annotates the visible surface.
* **Diagnostics:** polygon-vs-bbox extent IoU (internal consistency of the
  answer); out-of-frame/degenerate polygon counts; recall by COCO size
  strata; matched-mask-IoU split by reported `occlusion_pct`; count error.

**The polygon-fidelity caveat (stated in every report):** straight segments
between ≤32 whole-pixel vertices bound how faithfully any outline can follow
a strawberry boundary, and the raster boundary vs the annotated pixel set
adds small systematic noise — expect a mask-IoU ceiling a fraction below 1
even for a perfect outliner. It is never "fixed" by loosening the metric.

## 4. How to run it (any model the provider CLIs can route)

```bash
# via Claude Code (default: glm-5.3-flash, the multimodal GLM-5.3):
python3 strawdi_eval/seg/run_segmentation_eval.py --dry-run        # plan + prompt, 0 calls
python3 strawdi_eval/seg/run_segmentation_eval.py --limit 1 --tag smoke
python3 strawdi_eval/seg/run_segmentation_eval.py --sample-id 108 --tag one
python3 strawdi_eval/seg/run_segmentation_eval.py --jobs 3 --tag full   # 1 control + 100 calls

# acceptance gate — must print PASS (needs the label mount in place):
python3 strawdi_eval/seg/verify_strawdi_seg_run.py strawdi_eval/seg/runs/<run dir>

# scorer self-checks (no dataset needed):
python3 strawdi_eval/seg/test_seg_scoring.py
```

The claude CLI child needs network and write access outside a read-only
sandbox (escalate). `--provider codex` reuses the vision catalog override
(regenerate `vlm_eval/model_catalog_vision.json` after any catalog change);
`--provider agy` delivers via `view_file` on the staged frame and the
harness scales **every polygon vertex** back from the 800×600 delivered
space (plus the bboxes), with the usual delivery-path handicap caveat — see
the sibling spec's provider section before judging those numbers.

The base-harness gates all carry over unchanged:

1. **Vision-delivery control first** — fails the batch as `Vision delivery: NO`.
2. **All tool surfaces off** — any tool attempt fails acceptance.
3. **No annotated image is ever an input** — label masks are read only after
   the call, sha256-guarded.
4. **Inputs ≤1280 px max edge** (StrawDI is 1008; delivered unchanged).
5. **Provenance** — report + every record state harness, base harness, model,
   effort; the run directory name carries
   `<model>-<effort>-<cli>-strawdi_seg`.

## 5. Artefacts

`strawdi_eval/seg/runs/<ts>-<model>-<effort>-<cli>-strawdi_seg[-<tag>/`:
`report.md` (verdict banner, control, method + caveats, mask headline, box
cross-check, size strata, occlusion diagnostic, per-image table, error
analysis, provenance) · `responses.jsonl` (raw + parsed answers, usage,
prompt, full mask block + nested `box_metrics` — GT boxes/areas ride inside
each record so the box cross-check re-scores without the dataset) ·
`metrics.csv` / `seg_summary.csv` / `size_strata.csv` / `ap.csv` · per-frame
overlays (white translucent fill = GT instance mask — a missed one carries a
red `MISS` label at its GT-box position; green polygon = TP, red polygon =
FP; label text takes its polygon colour except the redness-coloured
`red xx%` segment; small GT/TP/FP legend bottom-left) · chunked contact
sheets + `miss_gallery.jpg` · `manifest.json` + `catalog.json` provenance
snapshots · `control.json`. Overlays and sheets are JPG (quality 90); model
inputs and label masks stay PNG.

## 6. Reference points (examples, not the contract)

| date | model (cli) | calls | control | mask F1@0.5 (P / R) | box F1@0.5 | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-09-16 | glm-5.3-flash (claude) | 1 | 2.2 px | **0.333** (0.40 / 0.286) | 0.333 | v0.1 prototype smoke, frame `1002` (7 GT): parse+schema 1/1 first attempt; mask IoU@0.25→0.75 F1 0.667→0.167; matched mask IoU 0.724; polygon-vs-bbox extent IoU 0.938; misses = white unripe fruit + occluded + border-clipped; the 3 FPs fragment one occluded fruit into overlapping pieces; $0.11, 78 s |
| 2026-09-16 | glm-5.3-flash (claude) | 100 | 1.0 px | **0.665** (0.745 / 0.601) | 0.700 | **v0.1 full val batch (harness v0.1.0)** — parse + schema-valid 100/100, zero retries; mask F1 ladder 0.816@0.25 → 0.665@0.5 → 0.300@0.75 (P@0.25 0.913 — polygons land on the right fruit, strict thresholds cost fidelity); matched mask IoU 0.724; mask AP@50 0.581, mAP 0.246; box cross-check AP@50 0.619, mAP 0.329; recall large/medium/small = 0.99/0.61/**0.00** (128 small GT); occlusion split 0.73 vs 0.67 (the box pipeline's gap, shrunk); count bias −1.1; 2 out-of-frame polygons; $10.47, 44 min |

## 7. Known behaviours

* **Mask re-scoring needs the label mount.** GT boxes ride in every record
  (box cross-check rebuilds offline); mask numbers are recomputed only when
  the label PNGs verify — a `--rebuild-report` without the mount **keeps**
  the stored mask block rather than wiping it, and mask AP is only reported
  when **every** parsed record re-derived (never a number over a silent
  subset).
* **First-image failure shape (glm-5.3-flash, frame 1002):** clean exposed
  fruit outlines well (matched IoU 0.72); white unripe fruit is missed
  outright (looks like flowers — the unbiased-census challenge, same as the
  box pipeline's small/white recall loss); one partly occluded fruit came
  back as THREE overlapping fragment polygons instead of one outline
  following the occluder edge — duplicate-FP fragmentation, not offset
  error; border-clipped fruit missed.
* **Batch-level failure shape (100-frame run):** recall is the binding
  constraint, not placement — P@0.25 is 0.91 (found fruit get polygons on
  the right object) while the ladder falls to 0.30@0.75 (boundary fidelity).
  The small stratum is **0.00 recall over 128 GT instances** (same wall as
  the box pipeline's 0.02); worst-miss frames are dense (7-9 unfound
  instances). The occlusion IoU split is 0.73 vs 0.67 — the box pipeline's
  semantics gap, shrunk to a real occlusion effect as expected when prompt
  and GT agree on visible-surface semantics.
* **Polygon-vs-bbox extent IoU** (0.94 on the smoke frame) is the answer's
  internal consistency check — the model's boxes and polygon extents agree,
  so box and mask failures trace to the same fruits, not to incoherent
  geometry.
* **Zero-fruit inventories are valid answers** and score as all-missed;
  distinct from "could not parse", which carries no numbers at all.
* The dataset lives on a removable mount (`/media/zfei/GLOWAY/...`); the
  runner needs it for GT masks at score time and the verifier needs it for
  GT + scorer reproducibility (sha256-guarded).
* This pipeline's fingerprint covers `strawdi_eval/seg/` only; the detection
  pipeline's glob does not see this directory and vice versa.
