# full_segmentation — full-scene strawberry inventory with visible-surface polygons (vlm_eval scenes)

Standalone, **model-agnostic** spec of the second perception pipeline in this
harness. Its siblings are
[`full_detection.md`](full_detection.md) (the SROI picking-point pipeline this
one extends) and
[`../../strawdi_eval/pipeline/strawdi_segmentation.md`](../../strawdi_eval/pipeline/strawdi_segmentation.md)
(the StrawDI pipeline where polygons are scored as segmentation against GT
masks). This one keeps the **full_detection output standard intact** — the
same unbiased nine-field census with `picking_point` and the separate
`target_index` nomination — and adds a tenth field, `polygon`: an ordered
vertex outline of each fruit's **visible surface**.

**The defining constraint: no vlm_eval source has mask ground truth.** The
SROI frames carry one recomputed picking point each (plus the upstream rough
60×60 box derived from it); the shunba scenes are unlabelled. So the polygon
here is **never scored as segmentation** — no mask IoU, no mask P/R/F1, no
mask AP exists in this pipeline, and the verifier fails the run if any such
number appears. What the pipeline does with the polygon:

* **rendered** — per-frame overlays and contact sheets (redness-ramped
  outlines, cyan nominated target, magenta picking points);
* **diagnosed on every frame** — validity counts (out-of-frame /
  degenerate), vertex budget, polygon-vs-bbox extent IoU (internal
  consistency of the answer), picking-point distance to its own outline;
* **cross-checked on the GT frames only** — the nominated target is scored
  exactly like `full_detection` (picking-point error / dx / dy / PCK), plus
  the polygon extent's IoU against the same upstream rough box behind the
  approximate bbox-IoU, plus the GT point's distance to the target outline.

**Status: v0.1.** Validated on a full 15-frame batch (glm-5.3-flash via
claude CLI, 2026-09-16, verifier PASS) — see §6.

Implementation lives in [`../seg/`](../seg/) (`run_segmentation_eval.py`,
`lib/prompt.py`, `lib/seg_scoring.py`, `lib/render.py`,
`schema/inventory_segmentation_schema.json`, `verify_vlm_seg_run.py`,
`test_seg_scoring.py`) — deliberately **outside the base harness's
fingerprint glob** (top-level `vlm_eval/*.py`, `lib/`, `schema/`), the same
independence trick as `strawdi_eval/seg/`, so the two vlm_eval pipelines'
provenance stay independent. The model-facing machinery — provider stack,
reply parser, synthetic control, measurement invariants — is imported
unchanged from the parent harness; the manifest is `full_detection`'s,
reused as-is. Every record states **both** harness fingerprints.

---

## 1. What this pipeline does

One vlm_eval frame in → one complete, **unbiased** inventory of **every**
strawberry visible in it, whatever its colour, as **ten** per-fruit fields
(the full_detection nine plus `polygon`) plus the unchanged single
`target_index` nomination — then that answer is scored as follows:

* **Scored, GT frames only** — the nominated target's `picking_point`
  against the recomputed GT point (error / dx / dy / PCK@5/10/20), the
  direct `full_detection` cross-check; `bbox` and polygon-extent IoU against
  the upstream rough 60×60 box (both approximate — same caveat as the box
  pipeline's bbox-IoU); GT-point-to-target-outline distance (a diagnostic:
  the grasp point sits on the peduncle **above** the calyx, so it should
  land just outside the fruit-body outline, not inside it).
* **Diagnosed, every frame** — polygon validity, vertex budget,
  polygon-vs-bbox extent IoU, picking-point-to-own-outline distance.
* **Never scored** — `redness_pct`, `occlusion_pct`, `calyx_visible`,
  `peduncle_visible`, `graspable`, `confidence_pct`, `description` (same
  role as in `full_detection`: unbiased census + report diagnostics).
* **Not computable here** — mask IoU / mask AP / mask P-R-F1: there is no
  mask ground truth on any vlm_eval source. The StrawDI pipeline is where
  that comparison lives; this pipeline exists to get the same
  visible-surface polygons on the SROI/shunba scenes and to keep the
  picking-point semantics that only this dataset has.

The model works from pixels alone: no detector, segmenter, tracker or any
other model participates.

## 2. Input / output contract

**Input:** the base pipeline's manifest, reused as-is — under the base
pipeline's **shunba-only default (since 2026-09-17)** that is the ten
`shunba` scenes (normalised to ≤1280 px, unlabelled, qualitative only). The
GT-frame cross-check path needs an explicitly SROI-inclusive experiment
build (`python3 vlm_eval/build_manifest.py --with-sroi`, ideally to a
separate `--out` passed on via this runner's `--manifest`) plus
`--sources validation occluded shunba`; rebuild the shunba-only manifest
afterwards to restore the pipeline default. The 2026-09-16 reference run
(§6) used the pre-scope-change manifest, snapshotted inside its run
directory.

**Output:** per frame, one JSON object with a `strawberries` array and a
`target_index` — **ten fields per fruit, no more, no fewer**: the
`full_detection` nine fields verbatim (`bbox` whole-fruit box,
`redness_pct`, `occlusion_pct`, `calyx_visible`, `peduncle_visible`,
`graspable`, `picking_point`, `confidence_pct`, `description`; see the
sibling spec's table) plus:

| field | type | meaning |
| --- | --- | --- |
| `polygon` | 3–32 pairs `[x,y]` ints | ordered outline of the fruit's **visible surface**, walked once (cw or ccw, start anywhere); follows occluder edges — never extrapolates the hidden shape (unlike `bbox`); fruit body only (excludes calyx/stem where they sit apart); extent must agree with the visible fruit |

**Prompt:** `inventory_segmentation` (`seg/lib/prompt.py`) — the
`inventory_plain` prompt with the polygon field inserted after `bbox`,
"nine" → "ten", the task line gaining "and trace the outline of its visible
surface", and the JSON example gaining the polygon key; **otherwise
verbatim** (byte-diffed, the four edits above are the complete delta). The
polygon paragraph is self-contained where the StrawDI prompt refers back to
its bbox's fruit-body rule — `inventory_plain`'s bbox field carries no such
rule, so the rule is stated inline.

**Run statuses:** `ok` · `no_pick_point` (valid inventory, nominated fruit
has no visible peduncle → answer kept, no point number) · `schema_invalid` /
`parse_error` / `empty` / `refused` / `exec_error` (explicit failures
carrying **no** numbers, never zeros). Retry policy mirrors the base
harnesses exactly: one recorded resample at the same effort for
`schema_invalid`, one low-effort retry for `empty`.

## 3. Scoring

* **Nomination cross-check (GT frames):** the nominated target's
  `picking_point` error against the recomputed GT point — median / mean
  error, dx / dy, PCK@5/10/20; `target_scored` is false when the nominee
  offers no point (`no_pick_point`) or the frame is unlabelled.
* **Rough-box IoUs (GT frames, approximate):** `bbox_iou_rough` and
  `polygon_extent_iou_rough` (the tight box of the polygon's vertices) both
  compare against the upstream rough 60×60 box centred 30 px below the GT
  point — the only fruit-body reference this dataset has; both inherit the
  base pipeline's "rough reference + whole-fruit-vs-visible-surface"
  caveats. The polygon version additionally crosses semantics (visible
  extent vs whole-fruit rough box).
* **GT-point-to-outline (GT frames, diagnostic):** analytic distance from
  the GT point to the nominated polygon's nearest edge, plus a containment
  flag. Expected shape: a few px, usually **outside** — the grasp point is
  on the peduncle above the calyx. Never a headline.
* **Diagnostics (every parsed frame):** polygon validity per the box
  scorer's out-of-frame/degenerate rules (a vertex outside
  `[0,W)×[0,H)` or an empty raster excludes the polygon and counts it);
  mean vertex count; polygon-vs-bbox extent IoU; picking-point distance to
  its own outline (a grasp point filed against another fruit's polygon
  shows up here).
* **Unlabelled purity (I7):** shunba records carry no scored value of any
  kind; the verifier fails the run on any leak.

## 4. How to run it (any model the provider CLIs can route)

```bash
# via Claude Code (default: glm-5.3-flash, the multimodal GLM-5.3):
python3 vlm_eval/seg/run_segmentation_eval.py --dry-run          # plan + prompt, 0 calls
python3 vlm_eval/seg/run_segmentation_eval.py --limit 1 --tag smoke
python3 vlm_eval/seg/run_segmentation_eval.py --jobs 3 --tag full  # pipeline scope: 10 shunba calls

# the GT cross-check is an experiment build (the base pipeline is shunba-only):
python3 vlm_eval/build_manifest.py --with-sroi --out vlm_eval/manifest_sroi.json
python3 vlm_eval/seg/run_segmentation_eval.py --manifest vlm_eval/manifest_sroi.json \
    --sources validation occluded shunba --jobs 3 --tag scored    # 15 calls incl. the 5 scored frames

# acceptance gate — must print PASS:
python3 vlm_eval/seg/verify_vlm_seg_run.py vlm_eval/seg/runs/<run dir>

# scorer self-checks (no dataset needed):
python3 vlm_eval/seg/test_seg_scoring.py
```

The claude CLI child needs network and write access outside a read-only
sandbox (escalate). `--provider codex` reuses the vision catalog override;
`--provider agy` delivers via `view_file` and the harness scales **every
point, bbox and polygon vertex** back from the 800×600 delivered space, with
the usual delivery-path handicap caveat.

The base-harness gates all carry over unchanged:

1. **Vision-delivery control first** — fails the batch as `Vision delivery: NO`.
2. **All tool surfaces off** — any tool attempt fails acceptance.
3. **No annotated image is ever an input** — overlays and GT-marker
   renderings are post-hoc only; GT is recomputed by the manifest builder,
   never pasted.
4. **Inputs ≤1280 px max edge** (manifest-enforced + belt-and-braces gate).
5. **Provenance** — report + every record state harness, base harness,
   model, effort; the run directory name carries
   `<model>-<effort>-<cli>-vlm_seg`.

## 5. Artefacts

`vlm_eval/seg/runs/<ts>-<model>-<effort>-<cli>-vlm_seg[-<tag>/`:
`report.md` (verdict banner, control, method + the no-mask-GT caveat,
polygon diagnostics, nomination cross-check, per-image table, error
analysis, exact prompt, provenance) · `responses.jsonl` (raw + parsed
answers, usage, prompt, diagnostics + scored block — GT rides inside each
record so the cross-check re-scores offline) · `metrics.csv` /
`seg_summary.csv` · per-frame overlays (polygon outline + light fill
coloured by the fruit's **continuous redness ramp**, cyan polygon =
nominated target, magenta ring = reported picking point; on GT frames the
cyan crosshair + rough ellipse are the reference, drawn last; out-of-frame /
degenerate polygons are counted, not drawn) · chunked contact sheets
(`__unlabelled` suffixed for the no-GT frames) · `manifest.json` +
`catalog.json` provenance snapshots · `control.json`. Overlays and sheets
are JPG (quality 90); model inputs stay PNG.

## 6. Reference points (examples, not the contract)

| date | model (cli) | calls | control | parse+schema | nomination median err | polygons ok/oof/degen | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-16 | glm-5.3-flash (claude) | 15 | 0.0 px | 15/15 after 3 same-effort resamples (all recovered) | **11.74 px** (5 scored frames; mean 10.84, PCK@5/10/20 = 0.4/0.4/1.0) | 95/1/0 | v0.1 full batch (harness v0.1.0, all three sources): 96 polygons over 15 frames, 11.2 vertices mean; extent-vs-own-bbox IoU 0.79 (the whole-fruit-bbox vs visible-surface-polygon semantics gap, by design); pick-to-own-outline 15.3 px mean; rough-box IoU polygon-extent 0.52 vs bbox 0.52; GT-point-to-outline 11.5 px mean, containment 0/5 (expected shape — grasp point above the fruit body); the 1 oof polygon is a border-clipped sb01 fruit whose bbox runs to x=1280; `no_pick_point` on sb07/sb10 (the base pipeline's known sparse-scene shape); 147k tokens, $2.64, 13 min; verifier PASS over 178 checks |

## 7. Known behaviours

* **The polygons are qualitative here.** They can be eyeballed (overlays)
  and internally cross-checked, but no number on this dataset says how well
  they trace fruit. For scored segmentation use the StrawDI pipeline; for
  the same prompt's scored behaviour on GT masks, its val-batch F1@0.5 of
  0.665 (glm-5.3-flash) is the closest available reference.
* **Nomination errors and polygon placement fail together.** A bad
  nominated target means a bad polygon on the same fruit — the failure is
  in perception, not in geometry bookkeeping.
* **Extent-vs-own-bbox IoU runs ~0.79 here vs ~0.94 on StrawDI — by
  design, not drift.** This prompt asks the bbox for the WHOLE fruit and
  the polygon for the VISIBLE surface, so occluded fruit legitimately
  disagree between the two; on StrawDI both the box pipeline's eight-field
  prompt and the polygon share visible-surface-adjacent semantics against
  GT masks. A LOW value here flags incoherent answers; a high value on
  unoccluded fruit is the sanity case.
* **`no_pick_point` is a valid answer** (nominee's peduncle hidden) and
  keeps its diagnostics; it yields no point number, never a zero. Zero-fruit
  inventories are valid and score as nothing-found, distinct from "could not
  parse".
* **The GT-point-to-outline containment is expected to be low** — the grasp
  point sits above the fruit body by definition; the distance diagnostic,
  not containment, is the informative one.
* This pipeline's fingerprint covers `vlm_eval/seg/` only; the base
  harness's glob does not see this directory and vice versa.
