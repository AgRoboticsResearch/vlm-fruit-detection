# chaos_strawberry_detection — detect + segment on curated chaotic scenes

Standalone, **model-agnostic** spec of the fourth perception pipeline in this
harness. Its siblings are
[`full_detection.md`](full_detection.md) (the shunba-only picking-point
census), [`full_segmentation.md`](full_segmentation.md) (the vlm_eval
picking census plus visible-surface polygons) and the StrawDI pipelines
under `strawdi_eval/pipeline/` (where detection and segmentation are scored
against ground truth). This one uses the **`strawdi_segmentation` output
standard and prompt, byte for byte** — the StrawDI NINE-field census (`bbox`
= **detection**, `polygon` = **visible-surface segmentation**; no
`picking_point`, no `target_index`) — and changes only the INPUT: a curated
list of deliberately **chaotic** strawberry scenes (dense fruit, heavy
clutter, deep occlusion), currently the shunba `sb04` scene and hand-held
photographs normalised to 720p (EXIF-uprighted).

**The defining constraint: chaos scenes have no ground truth of any kind.**
No picking point, no fruit box, no mask — so nothing is scored, ever. What
the pipeline produces: per-scene overlays, the complete per-fruit inventory,
and answer-internal diagnostics. Its purpose is eyeballing the **density
limit** of the census — where fruit counts stop being trustworthy, whether
occluded fruit fragments into overlapping polygons, whether unripe fruit
disappears — on the hardest scenes the other pipelines' selections contain.

**Status: v0.2.** Validated 2026-09-17 on the StrawDI nine-field standard
(glm-5.3-flash via claude CLI, verifier PASS) — see §6. v0.1 briefly used
the vlm_eval-seg ten-field prompt (picking fields included) before the
switch to the StrawDI standard.

Implementation lives in [`../chaos/`](../chaos/)
(`build_chaos_manifest.py`, `run_chaos_eval.py`, `lib/prompt.py` — the
frozen StrawDI prompt — `lib/render.py`,
`schema/inventory_segmentation_schema.json`; plus `verify_chaos_run.py`,
an optional integrity checker that is NOT part of the workflow) — outside
the base harness's fingerprint glob, and itself outside the seg subtree's.
The prompt and
schema are **frozen byte-identical copies** of the StrawDI seg pipeline's
(diffed at creation and re-diffable any time), not imports: a pipeline's
fingerprint must cover the exact prompt text it sends, and an edit to one
pipeline's prompt must never silently change another's measurement. Every
record states the three-link provenance chain:

* `base_harness` = `vlm_eval` — provider stack, reply parser, synthetic
  vision-delivery control, all eight measurement invariants;
* `seg_harness` = `vlm_seg` (`vlm_eval/seg/`) — the polygon scorer this
  pipeline imports unchanged (`seg_scoring.py`);
* `harness` = `vlm_chaos` (`vlm_eval/chaos/`) — the curated manifest, the
  runner, and the frozen StrawDI prompt/schema + renderer.

---

## 1. What this pipeline does

One curated chaos scene in → one complete, **unbiased** inventory of
**every** strawberry visible in it, as the StrawDI nine-field census (see
[`../../strawdi_eval/pipeline/strawdi_segmentation.md`](../../strawdi_eval/pipeline/strawdi_segmentation.md)
§2 for the field table): `bbox` covers the whole fruit (detection),
`polygon` traces its visible surface (segmentation), plus the census
attributes. There is **no picking point and no nomination** — the StrawDI
standard drops both, so a reply volunteering either is rejected wholesale.
The model works from pixels alone: no detector, segmenter, tracker or any
other model participates.

Nothing is scored against ground truth (there is none). Reported per scene:

* **overlays** — redness-ramped polygons (the ramp legend is drawn
  bottom-left; no target/picking-point/GT elements exist in this standard);
* **the full per-fruit inventory** in the report (every fruit: bbox, polygon
  vertex count, redness, occlusion, flags, description) — with ≤ a handful
  of curated scenes this is the pipeline's primary output, meant to be read
  against the overlay;
* **diagnostics** — polygon validity counts, vertex budget,
  polygon-vs-bbox extent IoU.

## 2. Input / output contract

**Input:** the curated chaos list in `vlm_eval/chaos/manifest.json`, built
by `python3 vlm_eval/chaos/build_chaos_manifest.py`. Each entry names the
ORIGINAL image and a delivery rule; the builder materialises the delivered
frame as `vlm_eval/data/frames/chaos__<sample_id>.png` (model inputs stay
PNG) and records the sha256 of both original and delivered file (the
optional integrity checker can re-verify a delivered frame against its
hash at any time).

| sample | original | delivered | rule |
| --- | --- | --- | --- |
| `sb04` | shunba `0000093.jpg` (1920×1080, via `data/frames/sb04.png`) | 1280×720 | none — already 720p (the base pipeline's max-edge-1280 normalisation); the densest shunba frame (19 fruits in the 2026-09-16 vlm_seg run) |
| `IMG_7665` | `data/frames/IMG_7665.jpeg` (stored 4032×3024, EXIF orientation 6 → upright portrait 3024×4032) | 540×720 | EXIF transpose to upright first, then the **720P rule**: fit inside 1280×720 preserving aspect (16:9 → exactly 1280×720; 4:3 → 960×720; this portrait → 540×720), LANCZOS |

The EXIF step matters: hand-held photos carry an orientation tag that PIL
does **not** apply on open, so without an explicit transpose a portrait
photo arrives at the model sideways (caught on the first IMG_7665 build,
2026-09-17 — that run was discarded and redelivered upright). The manifest
records the tag, the stored shape and the transposed shape per scene.

Adding a scene = one entry in `CHAOS_FRAMES` (or `--frame NAME=PATH`, which
applies the 720P rule) + rebuild. The synthetic control is the base
harness's, regenerated deterministically by the builder.

**Output:** per scene, the nine-field StrawDI inventory — `strawberries`
only, no `target_index`. **Run statuses:** `ok` · `schema_invalid` /
`parse_error` / `empty` / `refused` / `exec_error` (no `no_pick_point`: the
standard has no picking point to withhold). Retry policy mirrors the seg
pipelines (one low-effort retry for `empty`, one same-effort resample for
`schema_invalid` — dense scenes produce the longest inventories, and the
documented extra-note-field decoration lands here).

**Prompt:** `inventory_segmentation`, a **frozen byte-identical copy** of
the StrawDI seg prompt (`chaos/lib/prompt.py`, diffed against
`strawdi_eval/seg/lib/prompt.py` at creation; if the StrawDI prompt ever
changes, this copy is re-frozen deliberately, never auto-inherited) —
parameterised to the delivered frame size, printed verbatim in every
report. The answer schema (`chaos/schema/`) is the same frozen copy.

## 3. How to run it (any model the provider CLIs can route)

```bash
python3 vlm_eval/chaos/build_chaos_manifest.py
python3 vlm_eval/chaos/run_chaos_eval.py --dry-run              # plan + prompt, 0 calls
python3 vlm_eval/chaos/run_chaos_eval.py --tag full             # scenes + 1 control call
python3 vlm_eval/chaos/run_chaos_eval.py --sample-id IMG_7665 --tag one
```

**No acceptance gate — deliberate (decided 2026-09-17).** Unlike its
siblings, this pipeline is a fast qualitative iteration loop (curate a
scene, run, eyeball the overlay) and runs WITHOUT a verifier step.
`verify_chaos_run.py` stays in the tree as an optional integrity checker
(frame hashes, purity, diagnostics reproduction) that can be run by hand,
but it is not part of the workflow and run reports do not depend on it.

The claude CLI child needs network and write access outside a read-only
sandbox (escalate). `--provider codex` reuses the vision catalog override;
`--provider agy` delivers via `view_file` with harness-side rescale of every
bbox and polygon vertex, with the usual delivery-path handicap
caveat.

The base-harness gates all carry over unchanged:

1. **Vision-delivery control first** — fails the batch as `Vision delivery: NO`.
2. **All tool surfaces off** — any tool attempt fails acceptance.
3. **No annotated image is ever an input** (overlays are post-hoc only).
4. **Inputs inside the no-resize ceiling** (the 720P rule guarantees it).
5. **Provenance** — report + every record state all three fingerprints
   (base → seg → chaos); the run directory name carries
   `<model>-<effort>-<cli>-vlm_chaos`.

## 4. Artefacts

`vlm_eval/chaos/runs/<ts>-<model>-<effort>-<cli>-vlm_chaos[-<tag>/`:
`report.md` (verdict banner, control, method + no-GT caveat, polygon
diagnostics, per-scene table, **per-fruit detail**, every run, tokens/time/
cost, overlays legend, exact prompt, three-fingerprint provenance) ·
`responses.jsonl` (raw + parsed answers, usage, prompt, diagnostics) ·
`metrics.csv` / `chaos_summary.csv` · per-scene overlays (redness-ramped
polygons + ramp legend) + `contact_sheets/chaos_sheet.jpg` (mixed aspect
ratios padded onto one canvas) · `manifest.json` + `catalog.json`
provenance snapshots · `control.json`. Overlays and sheets are JPG (quality
90); model inputs stay PNG.

## 5. Known behaviours

* **Nothing is scored, by construction.** No scored-looking key
  (error/IoU/PCK/mask numbers) is ever written to a record — the exact
  analogue of the base harness's unlabelled-purity rule. The optional
  integrity checker asserts this if run, but no gate enforces it.
* **Chaos is where the census degrades first.** Expect the shunba failure
  shapes at their worst: fragment polygons on heavily occluded fruit,
  missed unripe/white fruit, border-clipped fruit, and schema-invalid first
  attempts (extra note fields) on the longest inventories — the same-effort
  resample usually recovers them.
* **Fruit counts on dense scenes are an estimate, not a measurement** —
  without GT, the count is the model's claim; cross-check the per-fruit
  list against the overlay before quoting a number.
* This pipeline's fingerprint covers `vlm_eval/chaos/` only; neither the
  base harness's nor the seg subtree's glob sees this directory, and vice
  versa.

## 6. Reference points (examples, not the contract)

| date | model (cli) | scenes | control | fruit / valid polys | notes |
| --- | --- | --- | --- | --- | --- |
| 2026-09-17 | glm-5.3-flash (claude) | 2 | 0.0 px | 31 / 30 (1 oof, 0 degenerate) | **v0.2 first batch on the StrawDI nine-field standard** (harness v0.2.0): IMG_7665 11 fruit / 10 valid (1 border-clipped oof), sb04 20 fruit / 20 valid; extent-vs-own-bbox IoU 0.72 (IMG_7665) vs 0.79 (sb04); mean vertices 13.3 / 9.5; polygons on-fruit (15×/52× red-pixel enrichment); parse+schema 2/2 first attempt; verifier PASS over 40 checks; $0.55, 4 min. (Two same-day earlier runs were superseded: one on a sideways EXIF-untransposed IMG_7665, one on the v0.1 ten-field vlm_eval-seg prompt — 28 fruit there; both discarded with the standard switch.) |
