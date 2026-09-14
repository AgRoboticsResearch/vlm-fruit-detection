# strawdi_detection — full-scene strawberry inventory, scored as detection on StrawDI_Db1

Standalone, **model-agnostic** spec of the second perception pipeline. It
sibling spec is [`../vlm_eval/full_detection.md`](../vlm_eval/full_detection.md)
(the SROI picking-point pipeline); this one reuses that pipeline's **output
standard** verbatim and changes the *scene* and the *scoring*: the frames come
from the public StrawDI_Db1 dataset, and the answer is scored as multi-instance
**detection** against per-instance ground-truth boxes derived from the
dataset's segmentation masks.

Implementation lives in this directory (`run_detection_eval.py`,
`build_manifest.py`, `lib/scoring.py`, `lib/render.py`,
`verify_strawdi_run.py`). The model-facing machinery — provider stack, prompt
text, reply parser, synthetic control, measurement invariants — is imported
unchanged from `../vlm_eval/`; every record states **both** harness
fingerprints, because the numbers depend on both codebases.

---

## 1. What this pipeline does

One StrawDI frame in → one complete, **unbiased** inventory of **every**
strawberry visible in it, in exactly the `full_detection` format (the nine
per-fruit fields + `target_index`) — then that inventory is **scored as
detection**: every reported `bbox` is matched against the ground-truth boxes of
that frame. The model works from pixels alone: no detector, tracker, segmenter
or any other model participates.

What is scored and what is not:

* **Scored** — the bounding boxes, as multi-instance detection (P/R/F1 at IoU
  thresholds, AP, count error, size-stratified recall).
* **Recorded, never scored** — `redness_pct`, `occlusion_pct`,
  `calyx_visible`, `peduncle_visible`, `graspable`, `picking_point`,
  `target_index`. StrawDI has no picking-point or ripeness ground truth, so
  these fields exist because the output standard demands them, and nothing
  more.

## 2. Input / output contract

**Input:** one RGB frame from StrawDI_Db1 `val` (1008×756, real field
photographs). 1008 < 1280, so frames are delivered **unchanged** — byte-for-byte
copies snapshotted into `data/frames/` — and ground-truth coordinates map 1:1
onto the delivered image with no resampling. The builder hard-errors on any
frame over the 1280 px ceiling rather than silently resizing.

**Ground truth:** the dataset ships per-instance segmentation masks (grayscale
PNG, 0 = background, 1..N = instance). One tight, axis-aligned, **half-open**
box `[x1, y1, x2, y2]` is derived per instance (`x2 = mask_xmax + 1`,
`y2 = mask_ymax + 1`), **every** instance kept however small (smallest in val:
59 px ≈ 8×8; 572 instances over 100 frames, mean 5.72/image). Ground truth is
always *re-derived from the masks* — the verifier re-runs the same function and
fails on any drift; nothing is ever hand-edited.

**Output:** per frame, one JSON object exactly as in `full_detection.md`:

```json
{"strawberries": [ { ...nine fields, no more, no fewer... }, ... ],
 "target_index": 0}
```

**Prompt:** `inventory_plain` verbatim from the base harness
(`vlm_eval/prompts.py`), parameterised to 1008×756, printed in every report.

**Run statuses:** `ok` · `no_pick_point` (both = the answer parsed; both are
scored for detection here — the nomination has no GT, so it must never gate
the boxes) · `schema_invalid` / `parse_error` / `out_of_frame` / `empty` /
`refused` / `exec_error` (explicit failures, never silently treated as
success, and they carry **no** detection numbers — never zeros).

## 3. Scoring

* **Matching:** greedy one-to-one; predictions ranked `confidence_pct` desc
  (ties → inventory order); each prediction claims the unmatched GT with the
  highest IoU (ties → lowest GT index); match at IoU ≥ t; recomputed
  independently at t ∈ {0.25, 0.5, 0.75}. Boxes outside the frame or
  degenerate never match and count as FP.
* **Headline: F1@IoU-0.5** (with P and R). AP@50 and mAP@[.50:.95] are
  reported but secondary — inventory confidences clump near 100, under which
  ranking is arbitrary and AP collapses toward the F1 point.
* **Secondary criterion:** symmetric centre-containment (`center_hit`) — see
  the semantics caveat below.
* **Diagnostics:** recall by COCO size strata (small/medium/large on GT area);
  matched-IoU split by the prediction's reported `occlusion_pct` (<25 vs ≥25);
  count-error mean/MAE/bias.

**The bbox-semantics caveat (stated in every report):** the prompt asks for
the box of the WHOLE fruit including hidden parts, while StrawDI masks
annotate the VISIBLE surface only. Predictions on partly occluded fruit are
expected to *exceed* the GT box and lose IoU — a one-directional bias, worst
on heavily occluded fruit. IoU@0.5 stays primary (standard, prompt-faithful);
centre-containment is reported alongside as a semantics-robust presence
check; the occlusion-split diagnostic quantifies the gap. It is never fixed
by quietly loosening the metric.

## 4. How to run it (any model the provider CLIs can route)

```bash
# via Claude Code (default: glm-5.3-flash, the multimodal GLM-5.3):
python3 strawdi_eval/build_manifest.py                      # masks -> GT + frames
python3 strawdi_eval/run_detection_eval.py --dry-run        # plan + prompt, 0 calls
python3 strawdi_eval/run_detection_eval.py --limit 1 --tag smoke
python3 strawdi_eval/run_detection_eval.py --jobs 3 --tag full   # 1 control + 100 calls

# acceptance gate — must print PASS:
python3 strawdi_eval/verify_strawdi_run.py strawdi_eval/runs/<run dir>
```

The claude CLI child needs network and write access outside a read-only
sandbox (escalate). Codex is wired (`--provider codex`) but not validated on
this dataset. The scorer self-checks live in `test_scoring.py` (includes an
agreement test against the base harness's `_iou`).

The base-harness gates all carry over unchanged:

1. **Vision-delivery control first** — fails the batch as `Vision delivery: NO`.
2. **All tool surfaces off** — any tool attempt fails acceptance.
3. **No annotated image is ever an input** — masks are read only after the call.
4. **Inputs ≤1280 px max edge** (StrawDI is 1008; delivered unchanged).
5. **Provenance** — report + every record state harness, base harness, model,
   effort; the run directory name carries `<model>-<effort>-<cli>-strawdi_eval`.

## 5. Artefacts

`strawdi_eval/runs/<ts>-<model>-<effort>-<cli>-strawdi_eval[-<tag>/`:
`report.md` (verdict banner, control, method + caveats, headline metrics, size
strata, occlusion diagnostic, per-image table, error analysis, provenance) ·
`responses.jsonl` (raw + parsed answers, usage, prompt, full detection block —
GT boxes ride inside each record so `--rebuild-report` re-scores without the
dataset) · `metrics.csv` / `detection_summary.csv` / `size_strata.csv` /
`ap.csv` · per-frame overlays (green = GT, red inset = missed GT,
redness-ramp = predictions, white corner ticks = TP, cyan = nominated target)
· chunked contact sheets + `miss_gallery.png` · `manifest.json` +
`catalog.json` provenance snapshots · `control.json`.

## 6. Reference points (examples, not the contract)

| date | model (cli) | calls | control | F1@0.5 (P / R) | mAP@[.50:.95] | notes |
| --- | --- | --- | --- | --- | --- | --- |
| — | — | — | — | — | — | first batch pending |

## 7. Known behaviours

* **Zero-fruit inventories are valid answers** and score as all-missed; they
  are distinct from "could not parse", which carries no numbers at all.
* **`no_pick_point` is a parse success here** and is fully scored — unlike the
  SROI pipeline, where it suppresses point scoring.
* **Tiny instances (down to 59 px)** make the small stratum brutally hard;
  expect near-zero recall there. That is a capability finding, not a bug — the
  report stratifies so it cannot poison the headline.
* **The bbox-semantics gap** (§3) shows up as: matched IoU visibly lower for
  predictions reporting `occlusion_pct ≥ 25`, and phantom "duplicate" FPs
  where the model boxes the whole fruit around a sliver GT box.
* The dataset lives on a removable mount (`/media/zfei/GLOWAY/...`); frames
  are snapshotted in-repo, but the verifier needs the label masks in place
  (sha256-guarded) to re-derive ground truth.
