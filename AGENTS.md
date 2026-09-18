# strawberry_detection

Workspace notes for agents working in this repo.

## Layout

- `blog_study/` — the VLM fruit-detection study behind the blog: all four VLM
  eval pipelines (`vlm_eval/`, `vlm_eval/seg/`, `vlm_eval/chaos/`,
  `strawdi_eval/`, `strawdi_eval/seg/`) plus the blog itself (`blogs/`).
  Everything inside keeps its original relative layout — **run the commands
  below from `blog_study/`**, not from the repo root.
- `small_detector/` — the unrelated YOLO inference pipeline
  (`infer_videos.py`, `infer_first_frames.py`, `weights/`, run logs).
- `paper_study/` — placeholder for the extended paper version of the blog
  study (empty for now).

## VLM raw-capability eval (`blog_study/vlm_eval/`)

If your task involves the VLM picking-point eval — running it, extending it,
changing the prompt styles, adding frames, or reporting its numbers — **read
[`blog_study/vlm_eval/AGENTS.md`](blog_study/vlm_eval/AGENTS.md) first and follow it.** It is a binding
runbook, not background reading: it records the invariants that separate a valid
measurement from a believable but worthless one (image delivery depends on a
catalog override that silently breaks, the model must not be given tools, ground
truth is recomputed rather than hard-coded).

Short version of the workflow, from `blog_study/`:

```bash
python3 vlm_eval/make_catalog.py                          # after ANY catalog change
python3 vlm_eval/build_manifest.py
python3 vlm_eval/run_vlm_eval.py --jobs 3 --tag full     # needs escalation + network
python3 vlm_eval/verify_run.py vlm_eval/runs/<timestamp>-full   # must print PASS
```

## VLM segmentation eval (`blog_study/vlm_eval/seg/`)

A second pipeline on the same scenes: the full_detection output standard
(nine per-fruit fields incl. `picking_point` + `target_index`) plus a tenth,
`polygon` — a vertex outline of each fruit's VISIBLE surface. **No vlm_eval
source has mask ground truth**, so the polygons are qualitative +
internal-consistency diagnostics here (rendered, validity counts, extent vs
own bbox, pick-point vs own outline); on SROI GT frames (explicit
`--with-sroi` experiment manifests only — the base pipeline is shunba-only)
the nominated target is scored exactly like full_detection, plus
polygon-extent IoU vs the rough box. The verifier **fails** any run
carrying mask-matching
numbers — they cannot be measured on this data. Scored segmentation lives
in `strawdi_eval/seg/`. Spec:
[`blog_study/vlm_eval/pipeline/full_segmentation.md`](blog_study/vlm_eval/pipeline/full_segmentation.md);
lives in `vlm_eval/seg/`, outside the base harness's fingerprint glob, same
invariants, same provider defaults (`--provider claude` → glm-5.3-flash).

```bash
python3 vlm_eval/seg/run_segmentation_eval.py --limit 1 --tag smoke   # escalation + network
python3 vlm_eval/seg/run_segmentation_eval.py --jobs 3 --tag full     # default scope: shunba, qualitative
# scored cross-check needs an SROI experiment manifest (see the spec):
#   python3 vlm_eval/build_manifest.py --with-sroi --out vlm_eval/manifest_sroi.json
#   python3 vlm_eval/seg/run_segmentation_eval.py --manifest vlm_eval/manifest_sroi.json \
#       --sources validation occluded shunba --jobs 3 --tag scored
python3 vlm_eval/seg/verify_vlm_seg_run.py vlm_eval/seg/runs/<dir>    # must print PASS
python3 vlm_eval/seg/test_seg_scoring.py                              # scorer self-checks
```

## Chaos strawberry detection (`blog_study/vlm_eval/chaos/`)

A fourth pipeline: the **StrawDI segmentation standard** — the
`strawdi_segmentation` nine-field census (bbox = detection, polygon =
visible-surface seg; no picking point, no nomination; prompt + schema are
frozen byte-identical copies of the StrawDI ones) — run on a **curated list
of chaotic scenes** (dense fruit, heavy clutter — currently shunba `sb04` +
the `IMG_7665` hand-held photo, EXIF-uprighted and resized to fit 1280×720).
**No ground truth exists on any chaos scene** — purely qualitative:
overlays, per-fruit inventory and answer-internal diagnostics. **No
verifier step** — the pipeline is a fast iteration loop by design.
Imports the
base provider stack and the vlm_seg polygon scorer unchanged; every record
carries the three-fingerprint provenance chain (vlm_eval → vlm_seg →
vlm_chaos). Spec:
[`blog_study/vlm_eval/pipeline/chaos_strawberry_detection.md`](blog_study/vlm_eval/pipeline/chaos_strawberry_detection.md).

```bash
python3 vlm_eval/chaos/build_chaos_manifest.py                        # curate + 720p-normalise frames
python3 vlm_eval/chaos/run_chaos_eval.py --tag full                   # scenes + 1 control (escalation + network)
# no acceptance gate for this pipeline — deliberate (2026-09-17); see the spec's §3
```

## StrawDI detection eval (`blog_study/strawdi_eval/`)

The second VLM pipeline: the unbiased inventory output standard narrowed to
detection (eight per-fruit fields — no picking point, no target nomination
since v0.2), run on the public StrawDI_Db1 dataset and **scored as
multi-instance detection** against mask-derived GT boxes. Its spec is
[`blog_study/strawdi_eval/pipeline/strawdi_detection.md`](blog_study/strawdi_eval/pipeline/strawdi_detection.md); it
imports the provider stack, prompt, parser and control from `vlm_eval/`
unchanged (do not modify `vlm_eval/` for it), and inherits the same
invariants — control first, tools off, no annotated inputs, ≤1280 px,
provenance with both harness fingerprints.

```bash
python3 strawdi_eval/build_manifest.py
python3 strawdi_eval/run_detection_eval.py --jobs 3 --tag full   # needs escalation + network
python3 strawdi_eval/verify_strawdi_run.py strawdi_eval/runs/<dir>   # must print PASS
python3 strawdi_eval/test_scoring.py                             # scorer self-checks
```

Providers: `--provider claude` (default, glm-5.3-flash), `--provider codex`
(e.g. k3 — regenerate `vlm_eval/model_catalog_vision.json` after any catalog
change), `--provider agy` (Antigravity CLI, gemini-3.8-flash-high — image
reaches the model via the sanctioned `view_file` call on a staged
single-file workspace, resampled to 800×600 with harness-side coordinate
rescale; see the spec's provider section before judging the numbers).

Same rule as above: no detector in the loop — this measures the VLM alone.

## StrawDI segmentation eval (`blog_study/strawdi_eval/seg/`)

The third VLM pipeline (v0.1, validated on the full val batch 2026-09-16):
the detection pipeline's nine-field inventory — the eight detection fields
plus `polygon`, a vertex outline of each fruit's VISIBLE surface — scored as
multi-instance **segmentation** against the raw StrawDI GT masks (label
id-map PNGs, sha256-guarded, read only after the call), with the asked
bboxes scored by the detection scorer unchanged as a cross-check. Spec:
[`blog_study/strawdi_eval/pipeline/strawdi_segmentation.md`](blog_study/strawdi_eval/pipeline/strawdi_segmentation.md).
Lives in `strawdi_eval/seg/`, deliberately outside the detection harness's
fingerprint glob; reuses the detection manifest as-is; same invariants.

```bash
python3 strawdi_eval/seg/run_segmentation_eval.py --limit 1 --tag smoke   # escalation + network
python3 strawdi_eval/seg/run_segmentation_eval.py --jobs 3 --tag full
python3 strawdi_eval/seg/verify_strawdi_seg_run.py strawdi_eval/seg/runs/<dir>   # must print PASS
python3 strawdi_eval/seg/test_seg_scoring.py                              # scorer self-checks
```

Same provider set and defaults as the detection eval (`--provider claude`
→ glm-5.3-flash), plus `--sample-id` to run a single frame by id. Mask
scoring needs the label mount in place. Same rule again: no detector in the
loop.

## Small detector (`small_detector/`)

`infer_videos.py`, `infer_first_frames.py` and `weights/yolov11-m-best.pt`
are a separate YOLO inference pipeline (default paths are anchored to the
script's own directory, so run them from anywhere). Nothing in `blog_study/`
may import, call or otherwise depend on a detector — that is the whole point
of the eval.

## Paper study (`paper_study/`)

Reserved for the extended paper version of the blog study. Empty for now —
add its own spec/AGENTS.md here when work starts.
