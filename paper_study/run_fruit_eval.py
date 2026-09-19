#!/usr/bin/env python3
"""paper_study mixed-testbed fruit detection/segmentation eval.

One frame in -> one complete, unbiased nine-field fruit inventory (the
strawdi_segmentation output standard, fruit-generic wording) -> scored against
that dataset's own ground truth and protocol:

* EVERY source scores BOX detection (F1/P/R at IoU 0.5, ladder 0.25/0.5/0.75,
  COCO-style AP@50 + mAP@[.50:.95]) with the strawdi detection scorer unchanged.
* Sources with INSTANCE masks (strawdi id-maps, wgisd npz clusters, minneapple
  COCO polygons) additionally score the polygons as segmentation (mask F1@0.5,
  AP ladder) with the strawdi seg scorer unchanged.
* ACFR sources are box-only (the apples' pixel masks are semantic, not
  instance) — polygons asked, unscored, exactly like the base vlm_seg pipeline
  treats unmaskable GT.

Model: glm-5.3-flash via the claude CLI (``--effort max`` thinking for the
reference runs). All base-harness invariants carry over: vision-delivery
control first, every tool surface off, no annotated image ever an input (GT
read only after the call, sha256-guarded), inputs <= 1280 px, full provenance
(own + base harness fingerprints in every record and the run dir name).

Usage (from the repo root):

    python3 paper_study/build_testbed_manifest.py      # first
    python3 paper_study/run_fruit_eval.py --dry-run
    python3 paper_study/run_fruit_eval.py --jobs 3 --tag full --reasoning-effort max
    python3 paper_study/verify_run.py paper_study/runs/<run dir>
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent          # paper_study
REPO = HERE.parent                              # repo root
sys.path.insert(0, str(REPO / "blog_study"))
sys.path.insert(0, str(REPO))

from vlm_eval.lib import imaging, parse  # noqa: E402
import vlm_eval.run_vlm_eval as vre  # noqa: E402
from strawdi_eval.lib import scoring as det_scoring  # noqa: E402
from strawdi_eval.seg.lib import seg_scoring  # noqa: E402
from strawdi_eval.seg.lib import render as seg_render  # noqa: E402
from paper_study.lib import gtload  # noqa: E402
from paper_study.lib import prompt as fruit_prompt  # noqa: E402

HARNESS_NAME = "paper_fruit_seg"
HARNESS_VERSION = "0.1.1"
DEFAULT_RUNS_DIR = HERE / "runs"
DEFAULT_JOBS = 3
FORMAT_KEYS = tuple(fruit_prompt.FORMATS)          # ("full9", "seg3", "box2")

POLYGON_FIDELITY_NOTE = (
    "Straight segments between <= 32 whole-pixel vertices (PIL raster "
    "semantics) bound polygon fidelity; expect a mask-IoU ceiling below 1 "
    "even for a perfect outliner — same caveat as the strawdi seg pipeline."
)
CONFIDENCE_TIE_NOTE = (
    "AP ranks predictions by the model's confidence_pct, which clumps near "
    "100; AP then collapses toward the single F1 operating point. F1@IoU-0.5 "
    "is the headline; AP is secondary (deterministic tie-breaks: confidence, "
    "then inventory order)."
)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def harness_info() -> dict:
    """Fingerprint of THIS pipeline's code only (never the base harness's)."""
    digest = hashlib.sha256()
    files = (sorted(HERE.glob("*.py")) + sorted((HERE / "lib").glob("*.py"))
             + sorted((HERE / "schema").glob("*.json")))
    for path in files:
        digest.update(path.relative_to(HERE).as_posix().encode())
        digest.update(path.read_bytes())
    return {
        "name": HARNESS_NAME,
        "version": HARNESS_VERSION,
        "fingerprint": digest.hexdigest()[:12],
        "path": str(HERE),
        "files_hashed": len(files),
    }


def run_dir_name(stamp: str, model: str, effort: str | None,
                 tag: str = "", cli: str = "", fmt: str = "full9") -> str:
    parts = [stamp, vre.slugify(model), vre.slugify(effort or "default")]
    if cli:
        parts.append(vre.slugify(cli))
    parts.append(vre.slugify(HARNESS_NAME))
    if fmt != "full9":
        parts.append(vre.slugify(fmt))
    if tag:
        parts.append(vre.slugify(tag))
    return "-".join(parts)


# ---------------------------------------------------------------------------
# GT masks (read AFTER the call, sha256-guarded — never an input)
# ---------------------------------------------------------------------------

def load_gt_masks(sample: dict):
    """Per-instance bool masks in DELIVERED space, or None when box-only.

    Raises on any drift (sha mismatch, shape mismatch, count mismatch) — the
    caller records the error and leaves the mask block unscored, never zeros.
    """
    spec = sample.get("gt_masks") or {}
    kind = spec.get("kind")
    wh = tuple(reversed(sample["image_shape_hw"]))       # (w, h)
    if kind == "strawdi_idmap":
        label = Path(spec["label_image"])
        if gtload.sha256_file(label) != spec["label_sha256"]:
            raise RuntimeError(f"{label.name}: sha256 drift vs manifest")
        mask = np.array(Image.open(label))
        ids = sorted(int(v) for v in np.unique(mask) if v != 0)
        masks = [mask == i for i in ids]
        if spec.get("scale", 1.0) < 1.0:
            masks = [gtload.resize_mask(m, wh) for m in masks]
        return masks
    if kind == "wgisd_npz":
        npz_path = Path(spec["npz_image"])
        if gtload.sha256_file(npz_path) != spec["npz_sha256"]:
            raise RuntimeError(f"{npz_path.name}: sha256 drift vs manifest")
        src_wh = tuple(reversed(sample["source_image_shape_hw"]))
        masks = gtload.wgisd_masks(npz_path, wh, src_wh)
        if len(masks) != sample["n_gt"]:
            raise RuntimeError(f"{npz_path.name}: {len(masks)} masks vs "
                               f"{sample['n_gt']} boxes")
        return masks
    if kind == "coco_polygons":
        if gtload.sha256_file(Path(spec["coco_json"])) != spec["coco_json_sha256"]:
            raise RuntimeError("minneapple COCO json sha256 drift vs manifest")
        rings = spec["polygons"]
        masks = [gtload.minneapple_instance_mask(r, wh) for r in rings]
        return masks
    return None                                     # box-only source


def mask_gt_areas(masks) -> list[float]:
    return [float(int(m.sum())) for m in masks]


# ---------------------------------------------------------------------------
# Parsing / scoring
# ---------------------------------------------------------------------------

def classify_fruit_inventory(last_message: str, schema: dict) -> dict:
    """Parse a fruit inventory ('fruits' key; bare list wrapped).

    Format-agnostic: bbox is normalised, every OTHER schema-allowed field is
    copied through verbatim (polygon, confidence_pct, ... — whichever the
    format asks for; the schema already rejected anything else).
    """
    result = {"status": None, "json_method": None, "schema_valid": None,
              "schema_error": None, "fruits": [], "n_fruits": 0}
    text = (last_message or "").strip()
    if not text:
        result["status"] = parse.EMPTY_RESPONSE
        return result
    obj, method = parse.extract_json(text)
    result["json_method"] = method
    if obj is None:
        result["status"] = parse.REFUSED if parse.looks_like_refusal(text) \
            else parse.PARSE_ERROR
        return result
    payload = {"fruits": obj} if isinstance(obj, list) else obj
    valid, error = parse.validate(payload, schema)
    result["schema_valid"] = valid
    result["schema_error"] = error
    if not valid:
        result["status"] = parse.SCHEMA_INVALID
        return result
    fruits = []
    for entry in payload["fruits"]:
        box = parse.normalise_box(entry.get("bbox"))
        fruit = {"bbox": list(box) if box else None}
        fruit.update({key: value for key, value in entry.items()
                      if key != "bbox"})
        fruits.append(fruit)
    result["fruits"] = fruits
    result["n_fruits"] = len(fruits)
    result["status"] = parse.OK
    return result


def mask_block(status, inventory, gt_masks, gt_areas, frame_w, frame_h, n_gt,
               asks_polygons: bool = True) -> dict:
    if gt_masks is None or status != parse.OK or not asks_polygons:
        reason = ("format asks no polygons" if not asks_polygons else None)
        return {**seg_scoring.empty_masks(), "n_gt": n_gt,
                "mask_scored": False, **({"mask_unscored_reason": reason}
                                         if reason else {})}
    block = seg_scoring.score_masks_image(inventory, gt_masks, gt_areas,
                                          frame_w, frame_h)
    return {**block, "mask_scored": True}


def box_metrics_block(status, inventory, gt_boxes, gt_areas, frame_w, frame_h) -> dict:
    if status != parse.OK:
        return {**det_scoring.empty_detection(), "n_gt": len(gt_boxes)}
    return det_scoring.score_image(inventory, gt_boxes, gt_areas, frame_w, frame_h)


# ---------------------------------------------------------------------------
# One scored run
# ---------------------------------------------------------------------------

def run_one(args, manifest, sample, run_ctx) -> dict:
    image = HERE / sample["images"]["raw"]
    h, w = sample["image_shape_hw"]
    fmt = run_ctx["fmt_spec"]
    style = fmt["style"]
    prompt = fmt["build"](w, h, sample["source"])
    schema = json.loads((HERE / "schema" / fmt["schema"]).read_text())

    model = run_ctx["model"]
    effort = run_ctx["effort"]
    stem = f"{style}__{sample['sample_id']}"
    attempts: list[dict] = []
    fallback_used = False
    fallback_reason = None

    message_path = run_ctx["tmp"] / f"{stem}.last.txt"
    attempt = vre.call_model(args, manifest, image, prompt, message_path,
                             None, model, effort, run_ctx["agent_cwd"],
                             args.timeout)
    attempts.append(attempt)

    scored = classify_fruit_inventory(attempt["last_message"], schema)
    do_retry = False
    retry_effort = None
    if scored["status"] == parse.EMPTY_RESPONSE and effort != "low":
        retry_effort = "low"
        fallback_reason = "empty response at configured reasoning effort"
        do_retry = True
    elif scored["status"] == parse.SCHEMA_INVALID:
        retry_effort = effort
        fallback_reason = "schema-invalid reply, one resample at same effort"
        do_retry = True
    if do_retry:
        fallback_used = True
        message_path = run_ctx["tmp"] / f"{stem}.retry.last.txt"
        retry = vre.call_model(args, manifest, image, prompt, message_path,
                               None, model, retry_effort, run_ctx["agent_cwd"],
                               args.timeout)
        retry["fallback_of_attempt"] = 1
        attempts.append(retry)
        scored = classify_fruit_inventory(retry["last_message"], schema)

    totals = vre.merge_usage(attempts)
    inventory = scored["fruits"] if scored["status"] == parse.OK else None

    # GT masks are read only NOW (after the call) and verified — never inputs.
    gt_masks, gt_mask_error = None, None
    try:
        gt_masks = load_gt_masks(sample)
    except Exception as exc:
        gt_mask_error = f"{type(exc).__name__}: {exc}"

    gt_areas_masks = mask_gt_areas(gt_masks) if gt_masks is not None else None
    asks_polygons = bool(fmt["polygons"])
    record = {
        "run_id": stem,
        "style": style,
        "format": run_ctx["fmt"],
        "sample_id": sample["sample_id"],
        "source": sample["source"],
        "split": sample["split"],
        "has_gt": True,
        "status": scored["status"],
        "json_method": scored["json_method"],
        "schema_valid": scored["schema_valid"],
        "schema_error": scored["schema_error"],
        "model": model,
        "harness": HARNESS_NAME,
        "harness_version": HARNESS_VERSION,
        "harness_fingerprint": run_ctx["harness"]["fingerprint"],
        "base_harness": run_ctx["base_harness"]["name"],
        "base_harness_version": run_ctx["base_harness"]["version"],
        "base_harness_fingerprint": run_ctx["base_harness"]["fingerprint"],
        "provider": args.provider,
        "reasoning_effort": effort,
        "used_output_schema": False,
        "output_shape": "inventory",
        "image": sample["images"]["raw"],
        "image_kind": "raw",
        "frame_w": w,
        "frame_h": h,
        "downscaled": sample.get("downscaled", False),
        "prompt": prompt,
        "inventory": inventory,
        "n_fruits": scored["n_fruits"] if scored["status"] == parse.OK else None,
        # Ground truth rides in the record (box cross-check rescored offline);
        # mask rescoring needs the mount.
        "gt_boxes": sample["gt_boxes"],
        "gt_areas": sample["gt_areas"],
        "gt_mask_kind": sample["gt_masks"]["kind"],
        "gt_masks_loaded": gt_masks is not None,
        "gt_mask_error": gt_mask_error,
        **mask_block(scored["status"], inventory, gt_masks,
                     gt_areas_masks or sample["gt_areas"], w, h, sample["n_gt"],
                     asks_polygons=asks_polygons),
        "box_metrics": box_metrics_block(scored["status"], inventory,
                                         sample["gt_boxes"], sample["gt_areas"],
                                         w, h),
        **totals,
        "attempts": len(attempts),
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason if fallback_used else None,
        "wall_s": round(sum(a["wall_s"] for a in attempts), 3),
        "returncodes": [a["returncode"] for a in attempts],
        "timed_out": any(a["timed_out"] for a in attempts),
        "tool_attempts": sorted({t for a in attempts for t in a["tool_attempts"]}),
        "cli_errors": [e for a in attempts for e in a["errors"]],
        "response_text": attempts[-1]["last_message"],
        "response_excerpt": (attempts[-1]["last_message"] or "")[:600],
        "cost_usd": (round(sum(a.get("api_cost_usd") or 0.0 for a in attempts), 6)
                     or None),
        "attempts_detail": [
            {"index": i, "returncode": a["returncode"], "wall_s": a["wall_s"],
             "timed_out": a["timed_out"], "usage": vre.token_totals(a["usage"]),
             "tool_attempts": a["tool_attempts"], "cli_errors": a["errors"],
             "message_chars": len(a["last_message"] or "")}
            for i, a in enumerate(attempts, start=1)],
    }

    title = (f"{sample['source']} | {sample['sample_id']} | "
             f"{sample['n_gt']}gt | {scored['status']}")
    rel = Path("overlays") / sample["source"] / f"{stem}.jpg"
    try:
        raw = imaging.load_rgb(image)
        if asks_polygons:
            # Colour polygons by MASK matches where masks exist, else BOX matches.
            if record.get("mask_scored"):
                matches = record.get("matches_50") or []
                fn_idx = record.get("fn_gt_indices_50") or []
            else:
                matches = record["box_metrics"].get("matches_50") or []
                fn_idx = record["box_metrics"].get("fn_gt_indices_50") or []
            overlay = seg_render.draw_segmentation_overlay(
                raw, inventory, gt_masks, sample["gt_boxes"], matches, fn_idx, title)
        else:
            # No polygons asked: colour the predicted BOXES by box matching.
            matches = record["box_metrics"].get("matches_50") or []
            fn_idx = record["box_metrics"].get("fn_gt_indices_50") or []
            overlay = draw_box_overlay(raw, inventory, sample["gt_boxes"],
                                       matches, fn_idx, title)
        imaging.save_jpg(overlay, run_ctx["run_dir"] / rel)
        record["overlay"] = str(rel)
    except Exception as exc:  # a drawing bug must not throw away a paid answer
        record["overlay"] = None
        record["render_error"] = f"{type(exc).__name__}: {exc}"
    return record


def draw_box_overlay(image_rgb, inventory, gt_boxes, matches_50, fn_gt_indices,
                     title: str, header_px: int = 34):
    """Diagnostic overlay for polygon-less formats: GT fills + pred boxes.

    Same visual language as the segmentation overlay where it transfers:
    white translucent GT mask fills (or outlines when no mask GT exists),
    green boxes = TP, red boxes = FP, red MISS labels on missed GT.
    """
    from PIL import Image as _Image, ImageDraw as _ImageDraw
    h, w = image_rgb.shape[:2]
    frame = _Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).copy()
    canvas = _Image.new("RGB", (w, h + header_px), (0, 0, 0))
    canvas.paste(frame, (0, header_px))
    draw = _ImageDraw.Draw(canvas)
    scale = max(1.0, min(h, w) / 720.0)
    width = max(2, int(round(3 * scale)))
    tp_preds = {m["pred_index"] for m in (matches_50 or [])}
    for index, fruit in enumerate(inventory or []):
        box = fruit.get("bbox") if isinstance(fruit, dict) else None
        if not box:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        colour = (60, 230, 90) if index in tp_preds else (255, 60, 60)
        draw.rectangle([x1, y1 + header_px, x2, y2 + header_px],
                       outline=(0, 0, 0), width=width + 2)
        draw.rectangle([x1, y1 + header_px, x2, y2 + header_px],
                       outline=colour, width=width)
    for gt_index in sorted(set(fn_gt_indices or [])):
        if gt_index >= len(gt_boxes):
            continue
        x1, y1, x2, y2 = [float(v) for v in gt_boxes[gt_index]]
        draw.rectangle([x1, y1 + header_px, x2, y2 + header_px],
                       outline=(255, 255, 255), width=width)
        draw.text((x1 + 2, y2 + header_px + 2), "MISS", fill=(255, 60, 60))
    n_pred = len(inventory or [])
    tp = len(matches_50 or [])
    draw.text((6, header_px // 2 - 8), title, fill=(255, 255, 255))
    draw.text((w - 260, header_px // 2 - 8),
              f"pred {n_pred} / gt {len(gt_boxes)}  TP {tp} "
              f"FP {n_pred - tp} FN {len(set(fn_gt_indices or []))}",
              fill=(255, 255, 255))
    return np.array(canvas)


def failure_record(sample: dict, exc: BaseException, run_ctx) -> dict:
    return {
        "run_id": f"{run_ctx['fmt_spec']['style']}__{sample['sample_id']}",
        "style": run_ctx["fmt_spec"]["style"],
        "format": run_ctx["fmt"],
        "sample_id": sample["sample_id"],
        "source": sample["source"],
        "split": sample["split"],
        "has_gt": True,
        "status": parse.EXEC_ERROR,
        "json_method": None, "schema_valid": None, "schema_error": None,
        "model": run_ctx["model"],
        "harness": HARNESS_NAME,
        "harness_version": HARNESS_VERSION,
        "harness_fingerprint": run_ctx["harness"]["fingerprint"],
        "base_harness": run_ctx["base_harness"]["name"],
        "base_harness_version": run_ctx["base_harness"]["version"],
        "base_harness_fingerprint": run_ctx["base_harness"]["fingerprint"],
        "provider": run_ctx["args"].provider,
        "reasoning_effort": run_ctx["effort"],
        "image": sample["images"]["raw"], "image_kind": "raw",
        "frame_w": sample["image_shape_hw"][1], "frame_h": sample["image_shape_hw"][0],
        "prompt": None, "inventory": None, "n_fruits": None,
        "gt_boxes": sample["gt_boxes"], "gt_areas": sample["gt_areas"],
        "gt_mask_kind": sample["gt_masks"]["kind"],
        "gt_masks_loaded": False,
        "gt_mask_error": f"{type(exc).__name__}: {exc}",
        **{**seg_scoring.empty_masks(), "n_gt": sample["n_gt"],
           "mask_scored": False},
        "box_metrics": {**det_scoring.empty_detection(), "n_gt": sample["n_gt"]},
        "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0,
        "reasoning_output_tokens": 0, "total_tokens": 0,
        "attempts": 0, "fallback_used": False, "fallback_reason": None,
        "wall_s": 0.0, "returncodes": [], "timed_out": False,
        "tool_attempts": [], "cli_errors": [f"{type(exc).__name__}: {exc}"],
        "response_text": "", "response_excerpt": "", "cost_usd": None,
        "attempts_detail": [], "overlay": None,
        "harness_error": f"{type(exc).__name__}: {exc}",
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def per_source_summary(records, manifest=None) -> list[dict]:
    """Headline numbers per dataset: box F1@0.5 (+AP), mask F1@0.5 (+AP).

    Mask AP needs the GT masks a second time; they are re-loaded from the
    manifest (mount present) and the AP is only reported when EVERY mask-scored
    record of the source re-derives — never a number over a silent subset.
    """
    by_sample = {s["sample_id"]: s for s in (manifest or {}).get("samples", [])}
    rows = []
    for source in sorted({r["source"] for r in records}):
        recs = [r for r in records if r["source"] == source]
        ok = [r for r in recs if r["status"] == parse.OK]

        def agg(block, key):
            return sum((r[block] if block else r).get(key) or 0 for r in ok)

        tp, fp, fn = (agg("box_metrics", "tp_50"), agg("box_metrics", "fp_50"),
                      agg("box_metrics", "fn_50"))
        p, rcl, f1 = det_scoring.precision_recall_f1(tp, fp, fn)

        box_ap = None
        if ok:
            box_ap = det_scoring.ap_metrics([
                {"sample_id": r["sample_id"],
                 "preds": det_scoring.prepare_predictions(
                     r["inventory"], r["frame_w"], r["frame_h"]),
                 "gt_boxes": r["gt_boxes"]} for r in ok])

        mask_scored = bool(ok) and all(r.get("mask_scored") for r in ok)
        mp = mr = mf1 = None
        mask_ap = None
        if mask_scored:
            mtp, mfp, mfn = (sum(r["tp_50"] for r in ok),
                             sum(r["fp_50"] for r in ok),
                             sum(r["fn_50"] for r in ok))
            mp, mr, mf1 = det_scoring.precision_recall_f1(mtp, mfp, mfn)
            per_image = []
            for r in ok:
                sample = by_sample.get(r["sample_id"])
                if sample is None:
                    per_image = None
                    break
                try:
                    masks = load_gt_masks(sample)
                except Exception:
                    masks = None
                if masks is None:
                    per_image = None
                    break
                per_image.append({
                    "sample_id": r["sample_id"],
                    "preds": seg_scoring.prepare_predictions_masks(
                        r["inventory"], r["frame_w"], r["frame_h"]),
                    "gt_masks": masks})
            if per_image:
                mask_ap = seg_scoring.ap_metrics_masks(per_image)

        ious = [r["box_metrics"]["mean_matched_iou_50"] for r in ok
                if r["box_metrics"].get("mean_matched_iou_50") is not None]
        rows.append({
            "source": source,
            "frames": len(recs),
            "ok": len(ok),
            "n_gt": sum(r["box_metrics"]["n_gt"] for r in recs),
            "count_bias": (round(sum(r["box_metrics"]["count_error"] for r in ok)
                                 / len(ok), 2) if ok else None),
            "box_f1_50": f1, "box_p_50": p, "box_r_50": rcl,
            "box_ap_50": box_ap["ap_50"] if box_ap else None,
            "box_map_50_95": box_ap["map_50_95"] if box_ap else None,
            "mask_scored": mask_scored,
            "mask_f1_50": mf1, "mask_p_50": mp, "mask_r_50": mr,
            "mask_ap_50": mask_ap["ap_50"] if mask_ap else None,
            "mask_map_50_95": mask_ap["map_50_95"] if mask_ap else None,
            "mean_matched_box_iou_50": round(float(np.mean(ious)), 4) if ious else None,
        })
    return rows


# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------

def write_report(run_dir: Path, manifest, control, records, started,
                 elapsed, run_summary) -> dict:
    summary_rows = per_source_summary(records, manifest)
    n_calls = len(records)

    lines = []
    verdict_ok = (control is None or control.get("images_delivered")) and \
        all(r["status"] == parse.OK for r in records) and \
        not any(r["tool_attempts"] for r in records)
    lines.append("# paper_fruit_seg mixed-testbed report"
                 + (" — ALL OK" if verdict_ok else " — CHECK FAILURES"))
    lines.append("")
    lines.append(f"* generated: {started}  ·  wall {elapsed:.0f}s  ·  "
                 f"{n_calls} calls")
    lines.append(f"* model: **{run_summary['model']}** via {run_summary['provider']} "
                 f"(effort {run_summary['effort'] or 'config default'})")
    lines.append(f"* harness: {HARNESS_NAME} v{HARNESS_VERSION} "
                 f"fingerprint `{run_summary['harness']['fingerprint']}` · "
                 f"base {run_summary['base_harness']['name']} "
                 f"v{run_summary['base_harness']['version']} fingerprint "
                 f"`{run_summary['base_harness']['fingerprint']}`")
    if control is not None:
        lines.append(f"* vision control: "
                     f"{'DELIVERED' if control['images_delivered'] else 'NOT DELIVERED'} "
                     f"(code {control.get('prediction', {}).get('code')!r}, "
                     f"circle err {control.get('circle_error_px')}px, "
                     f"tool_attempts {control.get('tool_attempts')})")
    else:
        lines.append("* vision control: SKIPPED (unverified)")
    lines.append("")

    lines.append("## Tracking table — dataset | task | metrics")
    lines.append("")
    protocols = manifest["dataset"]["protocols"]
    lines.append("| dataset | frames | task (per its protocol) | box F1@0.5 (P/R) | "
                 "box AP@50 / mAP | mask F1@0.5 (P/R) | mask AP@50 / mAP |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in summary_rows:
        proto = protocols.get(row["source"], {})
        task = proto.get("task", "?")
        box = (f"{row['box_f1_50']:.3f} ({row['box_p_50']:.3f}/{row['box_r_50']:.3f})"
               if row["box_f1_50"] is not None else "—")
        boxap = (f"{row['box_ap_50']:.3f} / {row['box_map_50_95']:.3f}"
                 if row["box_ap_50"] is not None else "—")
        if row["mask_scored"] and row["mask_f1_50"] is not None:
            mask = (f"{row['mask_f1_50']:.3f} ({row['mask_p_50']:.3f}/"
                    f"{row['mask_r_50']:.3f})")
            map_ = (f"{row['mask_ap_50']:.3f} / {row['mask_map_50_95']:.3f}"
                    if row["mask_ap_50"] is not None else "—")
        else:
            mask = map_ = "n/a (box-only GT)"
        lines.append(f"| `{row['source']}` | {row['frames']} | {task} | "
                     f"{box} | {boxap} | {mask} | {map_} |")
    lines.append("")

    lines.append("## Per-frame results")
    lines.append("")
    lines.append("| sample | status | n_gt | n_pred | box TP/FP/FN@0.5 | "
                 "mask TP/FP/FN@0.5 | tokens |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for rec in sorted(records, key=lambda r: (r["source"], r["sample_id"])):
        bm = rec["box_metrics"]
        mtp = rec.get("tp_50"); mfp = rec.get("fp_50"); mfn = rec.get("fn_50")
        mask_cell = (f"{mtp}/{mfp}/{mfn}"
                     if rec.get("mask_scored") else "n/a")
        lines.append(
            f"| {rec['sample_id']} | {rec['status']} | {bm['n_gt']} | "
            f"{bm['n_pred']} | {bm['tp_50']}/{bm['fp_50']}/{bm['fn_50']} | "
            f"{mask_cell} | {rec.get('total_tokens', '—')} |")
    lines.append("")

    lines.append("## Method and caveats")
    fmt = run_summary.get("format", "full9")
    fmt_desc = {
        "full9": "the strawdi_segmentation nine-field census (bbox, polygon, "
                 "redness, occlusion, calyx, peduncle, graspable, confidence, "
                 "description)",
        "seg3": "bbox + polygon + confidence_pct — the minimal format for a "
                "segmentation task",
        "box2": "bbox + confidence_pct — the minimal format for a pure "
                "detection task (no polygons; mask metrics n/a by design)",
    }[fmt]
    lines.append(f"* Output format `{fmt}`: {fmt_desc}. Scene/fruit wording "
                 "parameterised per dataset; JSON key `fruits`. The prompt is "
                 "printed in full in responses.jsonl.")
    lines.append(f"* {POLYGON_FIDELITY_NOTE}")
    lines.append(f"* {CONFIDENCE_TIE_NOTE}")
    lines.append("* Per-dataset GT semantics (verified against the data): "
                 "ACFR circles→square boxes (the apples' pixel masks are "
                 "semantic, not instance — polygons asked, unscored); WGISD "
                 "delivered at max-edge 1280 (boxes+NEAREST mask rescale); "
                 "MinneApple GT from the HF COCO mirror (official test labels "
                 "are withheld on CodaLab — mirror annotations, not official); "
                 "prompt asks WHOLE-fruit boxes while WGISD/MinneApple/StrawDI "
                 "GT annotates the visible surface, so occluded-fruit boxes "
                 "lose IoU by construction (the polygon metric has no such gap "
                 "where masks are scored).")
    lines.append("* Testbed: 3 frames per dataset from each dataset's TEST "
                 "split, deterministic even-spaced selection — a development "
                 "diagnostic, NOT a benchmark number per dataset protocol.")
    lines.append("")
    lines.append("## Provenance")
    lines.append("```json")
    lines.append(json.dumps({k: run_summary[k] for k in
                             ("model", "effort", "provider", "harness",
                              "base_harness")}, indent=1))
    lines.append("```")

    (run_dir / "report.md").write_text("\n".join(lines) + "\n")

    # metrics.csv: one row per record, flat
    import csv
    with (run_dir / "metrics.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        header = ["sample_id", "source", "status", "n_gt", "n_pred",
                  "box_tp_50", "box_fp_50", "box_fn_50",
                  "box_mean_iou_50", "count_error",
                  "mask_scored", "mask_tp_50", "mask_fp_50", "mask_fn_50",
                  "total_tokens", "cost_usd"]
        writer.writerow(header)
        for rec in sorted(records, key=lambda r: (r["source"], r["sample_id"])):
            bm = rec["box_metrics"]
            writer.writerow([
                rec["sample_id"], rec["source"], rec["status"],
                bm["n_gt"], bm["n_pred"], bm["tp_50"], bm["fp_50"], bm["fn_50"],
                bm.get("mean_matched_iou_50"), bm.get("count_error"),
                rec.get("mask_scored", False),
                rec.get("tp_50"), rec.get("fp_50"), rec.get("fn_50"),
                rec.get("total_tokens"), rec.get("cost_usd")])

    # summary_by_source.csv
    with (run_dir / "summary_by_source.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    return {"per_source": summary_rows}


def write_artifacts(run_dir, manifest, control, records, started,
                    elapsed, run_summary) -> None:
    with (run_dir / "responses.jsonl").open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    summary = write_report(run_dir, manifest, control, records, started,
                           elapsed, run_summary)
    print(f"per-source:")
    for row in summary["per_source"]:
        box = row["box_f1_50"]
        mask = row["mask_f1_50"]
        print(f"  {row['source']:14s} box F1@0.5 {box if box is not None else '—'}"
              f"   mask F1@0.5 {mask if mask is not None else '—'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    ap.add_argument("--out", type=Path, default=DEFAULT_RUNS_DIR)
    ap.add_argument("--tag", default="")
    ap.add_argument("--format", choices=FORMAT_KEYS, default="full9",
                    help="output format variant: full9 (nine-field census), "
                         "seg3 (bbox+polygon+confidence), box2 (bbox+"
                         "confidence)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sources", nargs="*", default=None,
                    help="subset of sources (default: all in the manifest)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--reasoning-effort", default=None,
                    help="thinking effort; the reference runs use max")
    ap.add_argument("--provider", choices=("claude", "codex", "agy"),
                    default="claude")
    ap.add_argument("--claude-model", default="glm-5.3-flash")
    ap.add_argument("--agy-model", default=None)
    ap.add_argument("--catalog", type=Path,
                    default=REPO / "blog_study" / "vlm_eval" / "model_catalog_vision.json")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    ap.add_argument("--control-image", type=Path, default=None)
    ap.add_argument("--skip-control", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    args.manifest = args.manifest.resolve()
    args.out = args.out.resolve()
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    manifest = json.loads(args.manifest.read_text())

    model = args.claude_model or args.model or "glm-5.3-flash"
    effort = args.reasoning_effort
    catalog_info = vre.claude_catalog_info(args, model)

    samples = manifest["samples"]
    if args.sources:
        known = {s["source"] for s in samples}
        unknown = sorted(set(args.sources) - known)
        if unknown:
            raise SystemExit(f"unknown source(s): {unknown}; known: {sorted(known)}")
        samples = [s for s in samples if s["source"] in set(args.sources)]
    if args.limit:
        samples = samples[: args.limit]

    if args.dry_run:
        s = samples[0]
        w, h = s["image_shape_hw"][1], s["image_shape_hw"][0]
        print(f"plan: {len(samples)} frames x 1 prompt = {len(samples)} calls + 1 control")
        print(f"model: {model}  provider={args.provider}  effort={effort or 'default'}  "
              f"format={args.format} ({fruit_prompt.FORMATS[args.format]['style']})")
        print("\n--- first prompt "
              f"({s['source']}, {w}x{h}) " + "-" * 30)
        print(fruit_prompt.build_prompt(args.format, w, h, s["source"]))
        return

    info = harness_info()
    base = vre.harness_info()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    cli_name = {"claude": "claude", "agy": "agy"}.get(args.provider, "codex")
    run_dir = args.out / run_dir_name(stamp, model, effort, args.tag,
                                      cli=cli_name, fmt=args.format)
    run_dir.mkdir(parents=True, exist_ok=False)
    tmp = run_dir / ".tmp"
    tmp.mkdir()
    agent_cwd = HERE / "agent_cwd"
    agent_cwd.mkdir(exist_ok=True)

    shutil.copyfile(args.manifest, run_dir / "manifest.json")
    (run_dir / "catalog.json").write_text(json.dumps(catalog_info, indent=2) + "\n")

    run_ctx = {"run_dir": run_dir, "tmp": tmp, "agent_cwd": agent_cwd,
               "model": model, "effort": effort, "harness": info,
               "base_harness": base, "args": args, "fmt": args.format,
               "fmt_spec": fruit_prompt.FORMATS[args.format]}

    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()

    vre.check_input_sizes(samples, None)

    print(f"run dir : {run_dir}")
    print(f"model   : {model}  provider={args.provider}  "
          f"effort={effort or 'config default'}  format={args.format}")
    print(f"plan    : {len(samples)} frames x 1 prompt "
          f"({run_ctx['fmt_spec']['style']}) "
          f"= {len(samples)} calls + 1 control")

    if not args.control_image:
        args.control_image = HERE / manifest["control"]["image"]

    control = None
    if args.skip_control:
        print("control : SKIPPED (results will be marked unverified)")
    else:
        print("control : running synthetic vision-delivery check ...", flush=True)
        control = vre.run_control(args, manifest, run_ctx)
        verdict = "DELIVERED" if control["images_delivered"] else "NOT DELIVERED"
        print(f"control : {verdict}  code={control.get('prediction', {}).get('code')!r} "
              f"(expected {control['truth']['code']!r})  "
              f"circle_err={control['circle_error_px']}px  "
              f"tool_attempts={control['tool_attempts']}")
        (run_dir / "control.json").write_text(json.dumps(control, indent=2) + "\n")

    records: list[dict] = []
    write_lock = threading.Lock()
    partial_path = run_dir / "responses.partial.jsonl"

    def collect(record: dict) -> None:
        with write_lock:
            records.append(record)
            with partial_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")

    total = len(samples)
    done = 0

    def guarded(sample):
        try:
            return run_one(args, manifest, sample, run_ctx)
        except Exception as exc:
            return failure_record(sample, exc, run_ctx)

    if args.jobs <= 1:
        for sample in samples:
            record = guarded(sample)
            collect(record)
            done += 1
            _progress(done, total, record)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(guarded, sample): sample for sample in samples}
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                collect(record)
                done += 1
                _progress(done, total, record)

    records.sort(key=lambda r: (r["source"], r["sample_id"]))
    elapsed = time.perf_counter() - t0

    run_summary = {"model": model, "effort": effort, "provider": args.provider,
                   "format": args.format,
                   **vre.cli_identity(args.provider), "catalog": catalog_info,
                   "harness": info, "base_harness": base}
    write_artifacts(run_dir, manifest, control, records, started, elapsed,
                    run_summary)
    partial_path.unlink(missing_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\ndone in {elapsed:.0f}s -> {run_dir}")


def _progress(done: int, total: int, record: dict) -> None:
    bm = record["box_metrics"]
    kind = "ok" if record["status"] == parse.OK else record["status"]
    print(f"[{done}/{total}] {record['sample_id']:44s} {kind:14s} "
          f"box {bm['tp_50']}tp/{bm['fp_50']}fp/{bm['fn_50']}fn "
          f"tok={record.get('total_tokens', '—')}", flush=True)


if __name__ == "__main__":
    main()
