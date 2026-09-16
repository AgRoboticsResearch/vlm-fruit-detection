# strawberry_detection

Workspace notes for agents working in this repo.

## VLM raw-capability eval (`vlm_eval/`)

If your task involves the VLM picking-point eval — running it, extending it,
changing the prompt styles, adding frames, or reporting its numbers — **read
[`vlm_eval/AGENTS.md`](vlm_eval/AGENTS.md) first and follow it.** It is a binding
runbook, not background reading: it records the invariants that separate a valid
measurement from a believable but worthless one (image delivery depends on a
catalog override that silently breaks, the model must not be given tools, ground
truth is recomputed rather than hard-coded).

Short version of the workflow, from this directory:

```bash
python3 vlm_eval/make_catalog.py                          # after ANY catalog change
python3 vlm_eval/build_manifest.py
python3 vlm_eval/run_vlm_eval.py --jobs 3 --tag full     # needs escalation + network
python3 vlm_eval/verify_run.py vlm_eval/runs/<timestamp>-full   # must print PASS
```

## StrawDI detection eval (`strawdi_eval/`)

The second VLM pipeline: the unbiased inventory output standard narrowed to
detection (eight per-fruit fields — no picking point, no target nomination
since v0.2), run on the public StrawDI_Db1 dataset and **scored as
multi-instance detection** against mask-derived GT boxes. Its spec is
[`strawdi_eval/pipeline/strawdi_detection.md`](strawdi_eval/pipeline/strawdi_detection.md); it
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

## StrawDI segmentation eval (`strawdi_eval/seg/`)

The third VLM pipeline (v0.1, validated on the full val batch 2026-09-16):
the detection pipeline's nine-field inventory — the eight detection fields
plus `polygon`, a vertex outline of each fruit's VISIBLE surface — scored as
multi-instance **segmentation** against the raw StrawDI GT masks (label
id-map PNGs, sha256-guarded, read only after the call), with the asked
bboxes scored by the detection scorer unchanged as a cross-check. Spec:
[`strawdi_eval/pipeline/strawdi_segmentation.md`](strawdi_eval/pipeline/strawdi_segmentation.md).
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

## Unrelated to the eval

`infer_videos.py`, `infer_first_frames.py` and `weights/yolov11-m-best.pt` are a
separate YOLO inference pipeline. Nothing in `vlm_eval/` may import, call or
otherwise depend on a detector — that is the whole point of the eval.
