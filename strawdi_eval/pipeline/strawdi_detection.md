# strawdi_detection — full-scene strawberry inventory, scored as detection on StrawDI_Db1

Standalone, **model-agnostic** spec of the second perception pipeline. Its
sibling spec is
[`../../vlm_eval/pipeline/full_detection.md`](../../vlm_eval/pipeline/full_detection.md)
(the SROI picking-point pipeline); this one takes that pipeline's **output
standard** and narrows it to what a detection eval needs: the same unbiased
per-fruit census, **without** the picking-oriented parts — no `picking_point`
per fruit, no `target_index` nomination (since v0.2; the v0.1 batch ran the
full nine-field prompt for comparability with the sibling pipeline). The
frames come from the public StrawDI_Db1 dataset, and the answer is scored as
multi-instance **detection** against per-instance ground-truth boxes derived
from the dataset's segmentation masks.

Implementation lives in the parent directory (`run_detection_eval.py`,
`build_manifest.py`, `lib/prompt.py`, `lib/scoring.py`, `lib/render.py`,
`verify_strawdi_run.py`). The model-facing machinery — provider stack, reply
parser, synthetic control, measurement invariants — is imported unchanged from
`../vlm_eval/`; the prompt is the base inventory minus the picking parts
(`lib/prompt.py`). Every record states **both** harness fingerprints, because
the numbers depend on both codebases.

---

## 1. What this pipeline does

One StrawDI frame in → one complete, **unbiased** inventory of **every**
strawberry visible in it, whatever its colour, as **eight** per-fruit fields —
then that inventory is **scored as detection**: every reported `bbox` is
matched against the ground-truth boxes of that frame. The model works from
pixels alone: no detector, tracker, segmenter or any other model participates.

What is scored and what is not:

* **Scored** — the bounding boxes, as multi-instance detection (P/R/F1 at IoU
  thresholds, AP, count error, size-stratified recall).
* **Asked, never scored** — `redness_pct`, `occlusion_pct`, `calyx_visible`,
  `peduncle_visible`, `graspable`, `confidence_pct`, `description`. StrawDI
  has no ripeness or graspability ground truth; these fields make the census
  unbiased and give the report its diagnostics (the occlusion-split).
* **Not asked at all** — `picking_point` and `target_index`. A reply that
  volunteers either is rejected wholesale by the schema (`additionalProperties:
  false`), exactly like any other extra field.

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

**Output:** per frame, one JSON object with a single `strawberries` array —
**eight fields per fruit, no more, no fewer**:

| field | type | meaning |
| --- | --- | --- |
| `bbox` | 4 ints `[x1,y1,x2,y2]` | whole fruit incl. hidden parts; **fruit body only** — calyx/stem excluded when they sit apart from the fruit, a calyx lying on the fruit is inside |
| `redness_pct` | int 0–100 | continuous: how red the visible surface is |
| `occlusion_pct` | int 0–100 | continuous: how much of the fruit is hidden |
| `calyx_visible` | bool | green sepal ring visible |
| `peduncle_visible` | bool | stem visible where it meets the fruit |
| `graspable` | bool | could be picked right now, from what is visible |
| `confidence_pct` | int 0–100 | confidence this is a strawberry |
| `description` | string (optional value) | 1–2 sentences in the model's own words |

**Prompt:** `inventory_detection` (`lib/prompt.py`) — the base harness's
`inventory_plain` minus the `picking_point` field line and the nomination
paragraph, otherwise verbatim, parameterised to 1008×756, printed in every
report.

**Run statuses:** `ok` (= the answer parsed) · `schema_invalid` /
`parse_error` / `out_of_frame` / `empty` / `refused` / `exec_error` (explicit
failures, never silently treated as success, and they carry **no** detection
numbers — never zeros). `no_pick_point` can no longer occur with the v0.2
prompt; it stays in the accepted vocabulary only so v0.1 runs still rebuild
and verify. Since harness v0.2.1 a `schema_invalid` reply gets **exactly one
recorded resample at the same effort** (both attempts kept in the record,
`fallback_used`/`fallback_reason` set); an `empty` reply keeps the base
harness's one retry at low effort. A retry that still fails stands as a
failure — no other failure mode is ever retried.

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
sandbox (escalate). Codex (`--provider codex`) is validated (k3, 2026-09-15;
regenerate `vlm_eval/model_catalog_vision.json` with `make_catalog.py` after
any catalog change). The scorer self-checks live in `test_scoring.py`
(includes an agreement test against the base harness's `_iou`).

### `--provider agy` (Antigravity CLI, e.g. Gemini 3.8 Flash)

```bash
python3 strawdi_eval/run_detection_eval.py --provider agy --jobs 3 --tag full
# default model gemini-3.8-flash-high; override with --agy-model / --model
```

agy changes ONE invariant fundamentally, and every agy report says so: it has
**no headless image attachment** (stream-json input is text-only, `@path`
mentions are literal text) and **no API-level tool-off**. Delivery therefore
goes through the built-in `view_file` tool — the frame is staged as the ONLY
file of a fresh per-call temp workspace (no `.agents` up-tree), the prompt
names that exact path, and the model must call `view_file` once. Discipline
is enforced by the workspace (anything but a workspace read is denied
headlessly) and verified by the event-stream scan: any tool step other than
`view_file` on the staged frame, and any denied action, lands in
`tool_attempts` and fails the record. Expect occasional agentic detours
(a denied `run_command` and an empty reply); they are recorded as failures,
never hidden.

Measured on agy 1.2.3: `view_file` shows the model the frame **resampled to
800×600**. The prompt therefore speaks the 800×600 coordinate space (same
text, parameterised), and the harness scales every reported bbox back by
(1008/800, 756/600) = 1.26 before scoring. The synthetic control gates this
whole chain per run (code + circle + square, after the same rescale). Fine
detail is softer than in the codex/claude runs — a delivery-path handicap,
worst on the small stratum; compare like with like.

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
`ap.csv` · per-frame overlays (white = GT box — a missed one carries a red
`MISS` label below it; green = TP prediction, red = FP prediction; label text
takes its box colour except the redness-coloured `red xx%` segment; small
GT/TP/FP legend bottom-left)
· chunked contact sheets + `miss_gallery.jpg` · `manifest.json` +
`catalog.json` provenance snapshots · `control.json`. Overlays and sheets are
saved as JPG (quality 90, 4:4:4) since 2026-09-15; model inputs, label masks
and the synthetic control stay PNG.

## 6. Reference points (examples, not the contract)

| date | model (cli) | calls | control | F1@0.5 (P / R) | mAP@[.50:.95] | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-09-14 | glm-5.3-flash (claude) | 100 | 1.0 px | **0.698** (0.775 / 0.635) | 0.283 | **v0.1 nine-field prompt (superseded)** — parse rate 82% (8× transport errors, 9× field-level schema violations, 1× arithmetic-in-bbox); centre-F1 0.861; count bias −1.0; recall large/medium/small = 0.99/0.70/0.04; $7.54, 49 min |
| 2026-09-15 | glm-5.3-flash (claude) | 100 | 1.4 px | **0.733** (0.815 / 0.666) | 0.352 | **v0.2 eight-field prompt + retry policy (harness v0.2.2)** — parse rate 99%: 5 schema-invalid replies, 4 recovered by the one-shot same-effort retry (frame 1367 failed twice on an invented `occlusion_note`); centre-F1 0.856; AP@50 0.653; count bias −1.04; recall large/medium/small = 0.99/0.75/0.02; matched IoU 0.79 (occl <25) vs 0.70 (≥25); $6.24, 35 min |
| 2026-09-15 | k3 (codex) | 100 | 0.0 px | **0.7493** (0.824 / 0.687) | 0.4269 | harness v0.2.4 — parse rate 100% (4 retried, all recovered); centre-F1 0.877; AP@50 0.664; count bias −0.95; recall large/medium/small = 1.00/0.77/0.05; matched IoU 0.86 (occl <25) vs 0.73 (≥25); 37 min |
| 2026-09-16 | gemini-3.8-flash-high (agy) | 100 | 0.0 px | **0.2569** (0.294 / 0.228) | 0.0525 | **first `--provider agy` batch** — delivery via the sanctioned `view_file` call on a staged single-file workspace, frame resampled to 800×600, bboxes rescaled ×1.26 harness-side (see §4 provider section before comparing). Parse rate 73%: 16 malformed-JSON `parse_error` (e.g. a stray `"label": "redness_pct"` key — never retried, per policy) + 11 `empty` on a mid-run auth transport outage (`oauth2 userinfo EOF`, both attempts failed); 0 tool attempts across all 100 calls. On the 73 parsed frames: centre-F1 0.456; count bias −1.19; recall large/medium/small = 0.40/0.19/0.00; matched IoU 0.85 — boxes land near fruit but loosely (the 800×600 resample + weaker localisation), which is what sinks IoU-thresholded metrics; 76 min |

## 6b. The 2026-09-16 agy failure-retry chain (how retries are handled)

Failures are retried as SEPARATE runs over a subset manifest
(`retry_failures_*.json`, built from the failed records, `retry_of` recorded);
a run directory is never edited after the fact. Each retry run gates on its
own control and verifies on its own. The agy full run's 27 failures (16
malformed-JSON `parse_error` + 11 `empty` from an agy auth/eligibility outage:
`oauth2 userinfo EOF`, later `loadCodeAssist` POST failures — a transport
problem, retried per policy and recorded) went through four retry batches:

* retry1 (VALID): 12 of 27 recovered; the outage returned mid-run (14 empty).
* retry2 (VALID): 4 more recovered (89/100 frames validly parsed in total).
* retry3/retry4 (**INVALID — control gate failed**): between 12:51 and 13:37
  the endpoint started answering the control with a deterministic 40 px
  red-circle miss (`[700,100]` vs true `[650,100]` delivered, byte-identical
  across batches) while still reading the code, the green square, and the
  1008×756 probe EXACTLY — a silent served-side behaviour change, not a
  delivery-transform change (the probe re-verified the 800×600 resample
  minutes before). The gate did its job; both batches are INVALID. Every one
  of the 100 frames HAS parsed at least once — but 11 only inside those two
  INVALID batches, so valid coverage stays 89/100 until the endpoint passes
  the control again.

## 7. Known behaviours

* **Zero-fruit inventories are valid answers** and score as all-missed; they
  are distinct from "could not parse", which carries no numbers at all.
* **`no_pick_point` is a v0.1 relic**: the v0.2 prompt asks for no nomination,
  so every parsed run is `ok`. The status remains accepted (records, rebuild,
  verify) so the v0.1 batch stays reproducible.
* **Tiny instances (down to 59 px)** make the small stratum brutally hard;
  expect near-zero recall there. That is a capability finding, not a bug — the
  report stratifies so it cannot poison the headline.
* **The bbox-semantics gap** (§3) shows up as: matched IoU visibly lower for
  predictions reporting `occlusion_pct ≥ 25`, and phantom "duplicate" FPs
  where the model boxes the whole fruit around a sliver GT box. Measured on
  the 2026-09-14 batch: matched IoU 0.77 (occl < 25) vs 0.68 (occl ≥ 25), and
  unmatched predictions report ~36% occlusion vs ~15% for matched ones.
* **Three distinct failure modes** on the first batch (18 of 100 calls):
  (a) 8 **transport errors** — 5× HTTP 529/code 1305 (endpoint overload) +
  3× HTTP 429/code 1302 (account rate limit); the CLI's error text (whose
  bracketed code extracts as a JSON array) lands as `schema_invalid`, and the
  true cause rides in `cli_errors`; (b) 9 **field-level schema violations** on
  otherwise-valid inventories — 4× a required field dropped (`confidence_pct`
  ×3, `picking_point` ×1) and 5× an invented extra field (`occlusion_note`,
  `peduncle_hidden_by_leaves`, …) rejected by `additionalProperties: false`,
  always on a mid-array fruit of a dense frame; (c) 1 **malformed JSON** — the
  model emitted arithmetic inside a bbox (`[638, 188, 268 + 370, 268]`). No
  response was truncated. All are recorded as explicit failures carrying no
  detection numbers, excluded from every metric. Since harness v0.2.1 the
  `schema_invalid` cases (including the transport-error-text variant) get
  exactly one recorded resample at the same effort — both attempts are kept,
  and the retry either recovers the frame or the failure stands. `parse_error`
  and every other failure mode are never retried.
* The dataset lives on a removable mount (`/media/zfei/GLOWAY/...`); frames
  are snapshotted in-repo, but the verifier needs the label masks in place
  (sha256-guarded) to re-derive ground truth.
