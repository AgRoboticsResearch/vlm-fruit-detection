#!/usr/bin/env python3
"""StrawDI segmentation eval — VLM instance masks over StrawDI_Db1.

Executes the ONE prompt of the seg pipeline (``inventory_segmentation``: the
detection pipeline's nine-field inventory — the eight detection fields plus a
``polygon`` tracing each fruit's VISIBLE surface) on manifest samples, then
scores the polygons as multi-instance segmentation against the raw StrawDI
ground-truth masks (the label id-map PNGs the detection pipeline only used to
derive boxes). The reported bboxes are additionally scored with the detection
scorer unchanged, as a cross-check that stays comparable with the
``strawdi_eval`` box pipeline.

The base harness (``vlm_eval/``) supplies everything model-facing: the
provider stack (claude/codex CLIs, tool surfaces off, images as base64
blocks), the reply parser and the synthetic vision-delivery control. This
script adds only the mask scoring, the GT-mask overlays and its own report.
The base-harness invariants carry over unchanged:

1. the synthetic vision-delivery control runs FIRST and invalidates the batch
   on failure;
2. all tool surfaces stay disabled (``--tools "" --safe-mode
   --strict-mcp-config --no-session-persistence``, empty agent_cwd; the event
   stream is scanned for tool_use and any attempt fails acceptance);
3. no annotated image is ever an input (label masks are only read AFTER the
   call, sha256-guarded against the manifest snapshot);
4. inputs stay inside the no-resize ceiling (the manifest builder enforces it);
5. every record and report states harness, base harness, model and effort.

This pipeline lives in ``strawdi_eval/seg/`` — deliberately outside the
detection harness's fingerprint glob (top-level ``strawdi_eval/*.py``,
``lib/``, ``schema/``) so the two pipelines' provenance stay independent,
same trick as ``agent_bridge/``.

Usage (repo root):
    python3 strawdi_eval/seg/run_segmentation_eval.py --limit 1 --tag smoke
    python3 strawdi_eval/seg/verify_strawdi_seg_run.py strawdi_eval/seg/runs/<dir>
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import re
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent          # strawdi_eval/seg
STRAWDI = HERE.parent                            # strawdi_eval
REPO = STRAWDI.parent                            # repo root
sys.path.insert(0, str(REPO))

from vlm_eval.lib import imaging, parse  # noqa: E402
import vlm_eval.run_vlm_eval as vre  # noqa: E402
from strawdi_eval.lib import scoring as det_scoring  # noqa: E402
from strawdi_eval.seg.lib import prompt as seg_prompt  # noqa: E402
from strawdi_eval.seg.lib import render, seg_scoring  # noqa: E402

HARNESS_NAME = "strawdi_seg"
HARNESS_VERSION = "0.1.0"

# The pipeline's single prompt and its answer schema: the detection
# inventory's eight fields PLUS a polygon of the visible surface (nine
# fields). See lib/prompt.py for the contract.
STYLE_NAME = seg_prompt.STYLE_NAME
SCHEMA_PATH = HERE / "schema" / "inventory_segmentation_schema.json"

DEFAULT_RUNS_DIR = HERE / "runs"
DEFAULT_JOBS = 3
CONTACT_SHEET_CHUNK = 24          # per sheet, 3 columns

# Printed in every report so a headline number can never be quoted without
# the semantics that produced it.
POLYGON_FIDELITY_NOTE = (
    "The polygon contract bounds fidelity: straight segments between <= 32 "
    "whole-pixel vertices, rasterised with PIL polygon semantics. A "
    "strawberry outline needs perimeter-level detail the vertex budget "
    "mostly covers, and the raster boundary vs the annotated pixel set adds "
    "small systematic noise — expect a mask-IoU ceiling a fraction below 1 "
    "even for a perfect outliner."
)
BBOX_SEMANTICS_NOTE = (
    "The box cross-check inherits the detection pipeline's semantics gap: "
    "the prompt asks for the WHOLE-fruit box while StrawDI ground truth "
    "annotates the VISIBLE mask surface, so occluded-fruit boxes lose IoU "
    "by construction. The polygon metric has NO such gap — it asks for the "
    "visible surface and is scored against the visible surface."
)
CONFIDENCE_TIE_NOTE = (
    "AP ranks predictions by the model's confidence_pct. Inventory confidences "
    "tend to clump near 100, under which the ranking is arbitrary and AP "
    "collapses toward the single F1 operating point. F1@mask-IoU-0.5 is the "
    "headline; AP is secondary and its tie-breaks (confidence, then inventory "
    "order) are deterministic."
)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def harness_info() -> dict:
    """Fingerprint of THIS pipeline's code (not the base harness's).

    Hashes the seg subtree only — adding files to the detection pipeline
    must not shift this fingerprint, and vice versa.
    """
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
                 tag: str = "", cli: str = "") -> str:
    """``<ts>-<model>-<effort>-<cli>-strawdi_seg[-<tag>]`` (base convention)."""
    parts = [stamp, vre.slugify(model), vre.slugify(effort or "default")]
    if cli:
        parts.append(vre.slugify(cli))
    parts.append(vre.slugify(HARNESS_NAME))
    if tag:
        parts.append(vre.slugify(tag))
    return "-".join(parts)


# ---------------------------------------------------------------------------
# Ground truth masks (read AFTER the call, sha256-guarded — never an input)
# ---------------------------------------------------------------------------

def load_gt_masks(sample: dict) -> list[np.ndarray]:
    """Per-instance bool masks from the label id-map PNG, manifest-verified.

    Same derivation order as ``build_manifest.derive_instances`` (sorted
    instance ids), so masks align 1:1 with the manifest's ``gt_boxes`` /
    ``gt_areas``. Raises on any drift or damage — the caller records the
    error rather than scoring against unverified ground truth.
    """
    label_path = Path(sample["label_image"])
    digest = hashlib.sha256(label_path.read_bytes()).hexdigest()
    if digest != sample["label_sha256"]:
        raise RuntimeError(f"{label_path.name}: sha256 drifted from the "
                           f"manifest snapshot (label file changed?)")
    with Image.open(label_path) as label_img:
        mask = np.array(label_img)
    h, w = sample["image_shape_hw"]
    if mask.dtype != np.uint8 or mask.ndim != 2 or mask.shape != (h, w):
        raise RuntimeError(f"{label_path.name}: unexpected label array "
                           f"{mask.dtype} {mask.shape} (uint8 {h}x{w} expected)")
    ids = sorted(int(v) for v in np.unique(mask) if v != 0)
    return [mask == instance_id for instance_id in ids]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=STRAWDI / "manifest.json",
                    help="the StrawDI manifest (frames + GT provenance); the seg "
                         "pipeline reuses the detection pipeline's manifest as-is")
    ap.add_argument("--out", type=Path, default=DEFAULT_RUNS_DIR,
                    help="runs root; a timestamped subdirectory is created inside it")
    ap.add_argument("--tag", default="", help="suffix for the run directory name")
    ap.add_argument("--limit", type=int, default=None,
                    help="max samples (manifest order)")
    ap.add_argument("--sample-id", default=None,
                    help="run a single manifest sample by id (e.g. 108)")
    ap.add_argument("--model", default=None, help="model slug")
    ap.add_argument("--reasoning-effort", default=None,
                    help="reasoning effort override; default keeps the provider's")
    ap.add_argument("--provider", choices=("claude", "codex", "agy"), default="claude")
    ap.add_argument("--agy-model", default=None,
                    help="model slug for --provider agy "
                         "(default gemini-3.8-flash-high)")
    ap.add_argument("--claude-model", default="glm-5.3-flash",
                    help="model slug for --provider claude (default glm-5.3-flash, "
                         "the multimodal GLM-5.3; the flagship slug rejects images)")
    ap.add_argument("--catalog", type=Path,
                    default=REPO / "vlm_eval" / "model_catalog_vision.json",
                    help="codex-only: vision-enabled catalog override")
    ap.add_argument("--catalog-source", type=Path,
                    default=vre.codex_home() / "cc-switch-model-catalog.json",
                    help="codex-only: catalog generation source, for provenance")
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds per call")
    ap.add_argument("--jobs", type=int, default=DEFAULT_JOBS, help="parallel calls")
    ap.add_argument("--price-in-per-mtok", type=float, default=None)
    ap.add_argument("--price-out-per-mtok", type=float, default=None)
    ap.add_argument("--control-image", type=Path, default=None,
                    help="override the prepared control image path")
    ap.add_argument("--skip-control", action="store_true",
                    help="skip the vision control (results marked unverified)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the exact first prompt, no calls")
    ap.add_argument("--rebuild-report", action="store_true",
                    help="re-derive artefacts from a run's responses.jsonl; "
                         "pass the run directory via --out")
    args = ap.parse_args(argv)
    args.manifest = args.manifest.resolve()
    args.out = args.out.resolve()
    args.catalog = args.catalog.resolve()
    return args


# ---------------------------------------------------------------------------
# One scored call
# ---------------------------------------------------------------------------

def build_prompt(args, sample: dict) -> str:
    h, w = sample["image_shape_hw"]
    # agy's view_file resamples the frame to 800x600, so the prompt speaks that
    # coordinate space; the delivery notice (with the staged path) is appended
    # by the provider inside call_model_agy, replacing the no-tools notice.
    pw, ph = vre.prompt_frame_size(args, w, h)
    prompt = seg_prompt.build_inventory_segmentation(frame_w=pw, frame_h=ph)
    if args.provider == "agy":
        return prompt
    return f"{prompt}\n\n{vre.TOOL_NOTICE}"


# The per-fruit attributes kept from the inventory (the eight detection
# fields plus the polygon; the scorer reads bbox/polygon separately).
SEG_ATTRS = ("redness_pct", "occlusion_pct", "calyx_visible",
             "peduncle_visible", "graspable", "confidence_pct",
             "description", "polygon")


def classify_segmentation(last_message: str, schema: dict) -> dict:
    """Parse a segmentation inventory (nine fields, polygon included).

    Mirrors the detection harness's ``classify_detection`` with the polygon
    normalised alongside the bbox. A bare top-level list is accepted and
    wrapped.
    """
    result = {
        "status": None, "json_method": None, "schema_valid": None,
        "schema_error": None, "strawberries": [], "n_strawberries": 0,
    }
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

    payload = {"strawberries": obj} if isinstance(obj, list) else obj
    valid, error = parse.validate(payload, schema)
    result["schema_valid"] = valid
    result["schema_error"] = error
    if not valid:
        result["status"] = parse.SCHEMA_INVALID
        return result

    berries = []
    for entry in payload["strawberries"]:
        box = parse.normalise_box(entry.get("bbox"))
        berry = {"bbox": list(box) if box else None}
        berry.update({key: entry.get(key) for key in SEG_ATTRS})
        berries.append(berry)

    result["strawberries"] = berries
    result["n_strawberries"] = len(berries)
    result["status"] = parse.OK
    return result


def mask_block(status: str, inventory, gt_masks, gt_areas, frame_w: int,
               frame_h: int, n_gt: int) -> dict:
    """Score a parsed record's polygons against its GT instance masks.

    Failure statuses — and any record whose GT masks could not be loaded and
    verified — carry no mask *numbers* at all, never zeros (``n_gt`` is kept:
    a property of the sample). A parsed inventory reporting ZERO fruit is a
    valid answer and scores as all-missed.
    """
    if status not in (parse.OK, parse.NO_PICK_POINT) or gt_masks is None:
        return {**seg_scoring.empty_masks(), "n_gt": n_gt}
    return seg_scoring.score_masks_image(inventory, gt_masks, gt_areas,
                                         frame_w, frame_h)


def box_metrics_block(status: str, inventory, gt_boxes, gt_areas,
                      frame_w: int, frame_h: int) -> dict:
    """The detection scorer, unchanged, on the ASKED bboxes (cross-check)."""
    if status not in (parse.OK, parse.NO_PICK_POINT):
        return {**det_scoring.empty_detection(), "n_gt": len(gt_boxes)}
    return det_scoring.score_image(inventory, gt_boxes, gt_areas,
                                   frame_w, frame_h)


def run_one(args, manifest, sample, run_ctx) -> dict:
    """Execute (and if needed retry) one model call, then score it."""
    image = STRAWDI / sample["images"]["raw"]
    h, w = sample["image_shape_hw"]
    prompt = build_prompt(args, sample)
    schema = json.loads(SCHEMA_PATH.read_text())

    model = run_ctx["model"]
    effort = run_ctx["effort"]
    stem = f"{STYLE_NAME}__{sample['sample_id']}"
    attempts: list[dict] = []
    fallback_used = False
    fallback_reason = None

    message_path = run_ctx["tmp"] / f"{stem}.last.txt"
    attempt = vre.call_model(args, manifest, image, prompt, message_path,
                             None, model, effort, run_ctx["agent_cwd"],
                             args.timeout)
    attempts.append(attempt)

    scored = classify_segmentation(attempt["last_message"], schema)
    do_retry = False
    retry_effort = None
    if scored["status"] == parse.EMPTY_RESPONSE and effort != "low":
        # Same recorded fallback as the base harness: reasoning can exhaust the
        # output budget, leaving no message at all.
        retry_effort = "low"
        fallback_reason = "empty response at configured reasoning effort"
        do_retry = True
    elif scored["status"] == parse.SCHEMA_INVALID:
        # One recorded resample at the SAME effort (polygon replies are long;
        # a vertex over the 32 cap or a malformed pair is a contract break,
        # not a reasoning-budget problem).
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
        scored = classify_segmentation(retry["last_message"], schema)

    totals = vre.merge_usage(attempts)
    parsed_ok = scored["status"] in (parse.OK, parse.NO_PICK_POINT)
    inventory = scored["strawberries"] if parsed_ok else None
    if inventory and args.provider == "agy":
        # The model answered in the delivered 800x600 frame; scale every bbox
        # and polygon vertex back to the original coordinate space before
        # scoring. Everything downstream sees original coords.
        def scale_point(pt):
            return [int(round(v)) for v in vre.agy_scale_point(pt, w, h)]
        inventory = [dict(item,
                          bbox=vre.agy_scale_box(item["bbox"], w, h)
                          if item.get("bbox") else None,
                          polygon=[scale_point(pt) for pt in item["polygon"]]
                          if item.get("polygon") else None)
                     for item in inventory]

    # GT masks are read only NOW (after the call) and verified against the
    # manifest's sha256 — the label PNGs must never be inputs. A load failure
    # keeps the answer but leaves the mask block unscored (never zeros).
    gt_masks, gt_mask_error = None, None
    try:
        gt_masks = load_gt_masks(sample)
    except Exception as exc:
        gt_mask_error = f"{type(exc).__name__}: {exc}"

    record = {
        "run_id": stem,
        "style": STYLE_NAME,
        "sample_id": sample["sample_id"],
        "source": sample["source"],
        "split": sample["split"],
        "has_gt": True,
        "task": sample.get("task", "full_inventory"),
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
        **({"delivery": {
            "mechanism": "agy view_file on a staged single-file workspace "
                         "(the one sanctioned tool call)",
            "delivered_wh": [vre.AGY_DELIVERED_W, vre.AGY_DELIVERED_H],
            "coord_rescale": [w / vre.AGY_DELIVERED_W, h / vre.AGY_DELIVERED_H],
            "view_file_calls": sum(a.get("view_file_calls", 0) for a in attempts),
        }} if args.provider == "agy" else {}),
        "used_output_schema": False,
        "output_shape": "inventory",
        "image": sample["images"]["raw"],
        "image_kind": "raw",
        "frame_w": w,
        "frame_h": h,
        "prompt": prompt,
        "inventory": inventory,
        "n_strawberries": scored["n_strawberries"] if parsed_ok else None,
        **vre._inventory_summary(inventory),
        # Ground truth rides inside the record so the run is self-contained
        # and --rebuild-report can re-score the BOX cross-check without the
        # dataset. Mask re-scoring needs the label PNGs (mount present).
        "gt_boxes": sample["gt_boxes"],
        "gt_areas": sample["gt_areas"],
        "label_image": sample["label_image"],
        "label_sha256": sample["label_sha256"],
        "gt_masks_loaded": gt_masks is not None,
        "gt_mask_error": gt_mask_error,
        **mask_block(scored["status"], inventory, gt_masks,
                     sample["gt_areas"], w, h, sample["n_gt"]),
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
        # Prefer explicit --price flags; otherwise report what the endpoint
        # charged (only the claude provider returns a per-call cost).
        "cost_usd": (
            vre._cost(totals, args)
            if args.price_in_per_mtok is not None
            and args.price_out_per_mtok is not None
            else (round(sum(a.get("api_cost_usd") or 0.0 for a in attempts), 6) or None)
        ),
        "attempts_detail": [
            {
                "index": i,
                "returncode": a["returncode"],
                "wall_s": a["wall_s"],
                "timed_out": a["timed_out"],
                "usage": vre.token_totals(a["usage"]),
                "tool_attempts": a["tool_attempts"],
                "cli_errors": a["errors"],
                "message_chars": len(a["last_message"] or ""),
            }
            for i, a in enumerate(attempts, start=1)
        ],
    }

    title = f"{STYLE_NAME} | {sample['sample_id']} | {sample['n_gt']}gt | {scored['status']}"
    rel = Path("overlays") / f"{stem}.jpg"
    try:
        raw = imaging.load_rgb(image)
        overlay = render.draw_segmentation_overlay(
            raw, inventory, gt_masks, sample["gt_boxes"],
            record["matches_50"], record["fn_gt_indices_50"], title)
        imaging.save_jpg(overlay, run_ctx["run_dir"] / rel)
        record["overlay"] = str(rel)
    except Exception as exc:  # a drawing bug must not throw away a paid-for answer
        record["overlay"] = None
        record["render_error"] = f"{type(exc).__name__}: {exc}"
    return record


def failure_record(sample: dict, exc: BaseException, run_ctx) -> dict:
    """A run that blew up in the harness itself, recorded rather than raised."""
    return {
        "run_id": f"{STYLE_NAME}__{sample['sample_id']}",
        "style": STYLE_NAME, "sample_id": sample["sample_id"],
        "source": sample["source"], "split": sample["split"],
        "has_gt": True, "task": sample.get("task", "full_inventory"),
        "status": parse.EXEC_ERROR, "json_method": None,
        "schema_valid": None, "schema_error": None,
        "model": run_ctx["model"],
        "harness": HARNESS_NAME, "harness_version": HARNESS_VERSION,
        "harness_fingerprint": run_ctx["harness"]["fingerprint"],
        "base_harness": run_ctx["base_harness"]["name"],
        "base_harness_version": run_ctx["base_harness"]["version"],
        "base_harness_fingerprint": run_ctx["base_harness"]["fingerprint"],
        "provider": None, "reasoning_effort": None,
        "used_output_schema": False, "output_shape": "inventory",
        "image": sample["images"]["raw"], "image_kind": "raw",
        "frame_w": sample["image_shape_hw"][1],
        "frame_h": sample["image_shape_hw"][0],
        "prompt": None, "inventory": None, "n_strawberries": None,
        **vre._inventory_summary(None),
        "gt_boxes": sample["gt_boxes"], "gt_areas": sample["gt_areas"],
        "label_image": sample["label_image"],
        "label_sha256": sample["label_sha256"],
        "gt_masks_loaded": False,
        "gt_mask_error": f"{type(exc).__name__}: {exc}",
        **{**seg_scoring.empty_masks(), "n_gt": sample["n_gt"]},
        "box_metrics": {**det_scoring.empty_detection(), "n_gt": sample["n_gt"]},
        "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0,
        "reasoning_output_tokens": 0, "total_tokens": 0,
        "attempts": 0, "fallback_used": False, "fallback_reason": None,
        "wall_s": 0.0, "returncodes": [], "timed_out": False,
        "tool_attempts": [], "cli_errors": [],
        "response_text": "", "response_excerpt": "", "cost_usd": None,
        "attempts_detail": [], "overlay": None,
        "harness_error": f"{type(exc).__name__}: {exc}",
    }


# ---------------------------------------------------------------------------
# Batch aggregation
# ---------------------------------------------------------------------------

def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(float(np.mean(values)), 4) if values else None


def seg_summary(records: list[dict], mask_ap: dict | None) -> dict:
    """Every batch-level number the report prints, derived from records only."""
    parsed = [r for r in records
              if r["status"] in (parse.OK, parse.NO_PICK_POINT)]
    # Mask metrics only over records that actually scored against verified GT
    # masks (gt_masks_loaded); everything else carries None, never zeros.
    scored = [r for r in parsed if r.get("tp_50") is not None]
    failed = [r for r in records if r["status"] in parse.FAILURE_STATUSES]

    summary = {
        "calls": len(records),
        "parsed": len(parsed),
        "parse_rate": round(len(parsed) / len(records), 4) if records else None,
        "schema_valid_rate": _rate(records, lambda r: r.get("schema_valid") is True),
        "ok": sum(1 for r in parsed if r["status"] == parse.OK),
        "no_pick_point": sum(1 for r in parsed if r["status"] == parse.NO_PICK_POINT),
        "mask_scored": len(scored),
        "mask_unscored_gt_error": sum(1 for r in parsed if r.get("tp_50") is None),
        "failures_by_status": {
            status: sum(1 for r in failed if r["status"] == status)
            for status in sorted({r["status"] for r in failed})
        },
        "retried_calls": sum(1 for r in records if r.get("fallback_used")),
        "retries_recovered": sum(
            1 for r in records if r.get("fallback_used")
            and r["status"] in (parse.OK, parse.NO_PICK_POINT)),
    }

    # Primary metric: the mask-IoU ladder.
    for threshold in seg_scoring.IOU_THRESHOLDS:
        key = int(round(threshold * 100))
        tp = sum(r.get(f"tp_{key}") or 0 for r in scored)
        fp = sum(r.get(f"fp_{key}") or 0 for r in scored)
        fn = sum(r.get(f"fn_{key}") or 0 for r in scored)
        p, recall, f1 = det_scoring.precision_recall_f1(tp, fp, fn)
        summary[f"tp_{key}"], summary[f"fp_{key}"], summary[f"fn_{key}"] = tp, fp, fn
        summary[f"precision_{key}"] = p
        summary[f"recall_{key}"] = recall
        summary[f"f1_{key}"] = f1

    matched_ious = [m["iou"] for r in scored for m in (r.get("matches_50") or [])]
    summary["matched_pairs_50"] = len(matched_ious)
    summary["mean_matched_iou_50"] = _mean(matched_ious)
    summary["median_matched_iou_50"] = (round(float(np.median(matched_ious)), 4)
                                        if matched_ious else None)
    summary["polygon_bbox_iou_mean"] = _mean(
        [r.get("polygon_bbox_iou_mean") for r in scored])
    summary["n_polygons_out_of_frame"] = sum(
        r.get("n_polygons_out_of_frame") or 0 for r in scored)
    summary["n_polygons_degenerate"] = sum(
        r.get("n_polygons_degenerate") or 0 for r in scored)

    count_errors = [r["count_error"] for r in scored if r.get("count_error") is not None]
    summary["count_error_mean"] = _mean(count_errors)
    summary["count_error_mae"] = _mean([abs(e) for e in count_errors])
    summary["gt_total"] = sum(r.get("n_gt") or 0 for r in scored)
    summary["pred_total"] = sum(r.get("n_pred") or 0 for r in scored)

    # Mask AP over the records whose masks re-derived at write time.
    summary["mask_ap_50"] = (mask_ap or {}).get("ap_50")
    summary["mask_map_50_95"] = (mask_ap or {}).get("map_50_95")
    summary["mask_ap_available"] = mask_ap is not None

    # Cross-check: the detection scorer on the ASKED bboxes, unchanged —
    # directly comparable with the strawdi_eval box pipeline.
    for threshold in det_scoring.IOU_THRESHOLDS:
        key = int(round(threshold * 100))
        tp = sum((r.get("box_metrics") or {}).get(f"tp_{key}") or 0 for r in parsed)
        fp = sum((r.get("box_metrics") or {}).get(f"fp_{key}") or 0 for r in parsed)
        fn = sum((r.get("box_metrics") or {}).get(f"fn_{key}") or 0 for r in parsed)
        p, recall, f1 = det_scoring.precision_recall_f1(tp, fp, fn)
        summary[f"box_tp_{key}"] = tp
        summary[f"box_f1_{key}"] = f1
        if abs(threshold - 0.5) < 1e-9:
            summary["box_precision_50"], summary["box_recall_50"] = p, recall
    box_per_image = [{
        "sample_id": r["sample_id"],
        "preds": det_scoring.prepare_predictions(r.get("inventory"),
                                                 r["frame_w"], r["frame_h"]),
        "gt_boxes": r["gt_boxes"],
    } for r in parsed]
    box_aps = det_scoring.ap_metrics(box_per_image)
    summary["box_ap_50"] = box_aps["ap_50"]
    summary["box_map_50_95"] = box_aps["map_50_95"]

    summary["size_strata"] = det_scoring.size_stratified(scored)
    summary["occlusion_split"] = det_scoring.occlusion_split(scored)
    summary["total_input_tokens"] = sum(r.get("input_tokens") or 0 for r in records)
    summary["total_output_tokens"] = sum(r.get("output_tokens") or 0 for r in records)
    summary["total_tokens"] = sum(r.get("total_tokens") or 0 for r in records)
    summary["wall_s_total"] = round(sum(r.get("wall_s") or 0 for r in records), 1)
    summary["cost_usd_total"] = round(sum(r.get("cost_usd") or 0 for r in records), 4) \
        if any(r.get("cost_usd") is not None for r in records) else None
    return summary


def _rate(records: list[dict], predicate) -> float | None:
    if not records:
        return None
    return round(sum(1 for r in records if predicate(r)) / len(records), 4)


# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------

CSV_DROP = ("prompt", "response_text", "attempts_detail", "box_metrics")


def write_seg_csvs(run_dir: Path, records: list[dict], summary: dict) -> None:
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in CSV_DROP}
                       for r in records])
    for column in df.columns:
        df[column] = df[column].map(
            lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)
    df.to_csv(run_dir / "metrics.csv", index=False)

    flat = {k: v for k, v in summary.items()
            if not isinstance(v, (list, dict))}
    pd.DataFrame([flat]).to_csv(run_dir / "seg_summary.csv", index=False)
    pd.DataFrame(summary["size_strata"]).to_csv(run_dir / "size_strata.csv",
                                                index=False)
    pd.DataFrame([
        {"metric": "mask_ap_50", "value": summary.get("mask_ap_50")},
        {"metric": "mask_map_50_95", "value": summary.get("mask_map_50_95")},
        {"metric": "box_ap_50", "value": summary.get("box_ap_50")},
        {"metric": "box_map_50_95", "value": summary.get("box_map_50_95")},
        {"metric": "box_f1_50", "value": summary.get("box_f1_50")},
    ]).to_csv(run_dir / "ap.csv", index=False)


def _render_all(run_dir: Path, records: list[dict],
                masks_by_sample: dict[str, list[np.ndarray]]) -> None:
    """(Re)render every overlay + contact sheets from the records themselves."""
    overlays = run_dir / "overlays"
    shutil.rmtree(overlays, ignore_errors=True)
    overlays.mkdir(exist_ok=True)
    sheets = run_dir / "contact_sheets"
    shutil.rmtree(sheets, ignore_errors=True)
    sheets.mkdir(exist_ok=True)

    rendered: list[tuple[dict, Path]] = []
    for record in records:
        sample_id = record["sample_id"]
        title = (f"{record['style']} | {sample_id} | "
                 f"{record.get('n_gt')}gt | {record['status']}")
        rel = Path("overlays") / f"{record['run_id']}.jpg"
        try:
            raw = imaging.load_rgb(STRAWDI / record["image"])
            overlay = render.draw_segmentation_overlay(
                raw, record.get("inventory"),
                masks_by_sample.get(sample_id), record.get("gt_boxes"),
                record.get("matches_50"), record.get("fn_gt_indices_50"), title)
            imaging.save_jpg(overlay, run_dir / rel)
            record["overlay"] = str(rel)
            record.pop("render_error", None)
        except Exception as exc:
            record["overlay"] = None
            record["render_error"] = f"{type(exc).__name__}: {exc}"
            continue
        rendered.append((record, run_dir / rel))

    for start in range(0, len(rendered), CONTACT_SHEET_CHUNK):
        chunk = rendered[start:start + CONTACT_SHEET_CHUNK]
        images = [imaging.load_rgb(path) for _, path in chunk]
        titles = [rec["sample_id"] for rec, _ in chunk]
        sheet = imaging.contact_sheet(images, titles, cols=3)
        imaging.save_jpg(sheet, sheets / f"sheet_{start // CONTACT_SHEET_CHUNK + 1:02d}.jpg")

    # Miss gallery: the images where the most GT area went unfound.
    misses = sorted((r for r in records if r.get("fn_gt_areas_50")),
                    key=lambda r: (sum(r["fn_gt_areas_50"]),
                                   len(r["fn_gt_areas_50"])), reverse=True)[:12]
    if misses:
        images = [imaging.load_rgb(run_dir / r["overlay"]) for r in misses
                  if r.get("overlay")]
        if images:
            imaging.save_jpg(imaging.contact_sheet(
                images, [f"{r['sample_id']} miss{len(r['fn_gt_areas_50'])}"
                         for r in misses if r.get("overlay")], cols=3),
                sheets / "miss_gallery.jpg")


def write_report(run_dir: Path, args, manifest, control, records: list[dict],
                 summary: dict, started: str, elapsed: float,
                 run_summary: dict, rebuilt_with: str | None = None) -> None:
    info = run_summary["harness"]
    base = run_summary["base_harness"]
    model = run_summary["model"]
    effort = run_summary["effort"]
    lines: list[str] = []
    add = lines.append

    delivered = None if control is None else bool(control["images_delivered"])
    verdict = ("BATCH VALID" if delivered else
               "BATCH INVALID (vision delivery failed)" if delivered is False
               else "BATCH UNVERIFIED (control skipped)")
    headline = summary.get("f1_50")
    add(f"# StrawDI segmentation eval — {verdict}")
    add("")
    failed_note = (f"; {summary['calls'] - summary['parsed']} call(s) failed to parse "
                   f"and are excluded from every metric"
                   if summary["calls"] != summary["parsed"] else "")
    unscored_note = (f"; {summary['mask_unscored_gt_error']} parsed call(s) could not "
                     f"be scored against GT masks (label PNG unavailable/drifted)"
                     if summary["mask_unscored_gt_error"] else "")
    add(f"Headline: **F1@mask-IoU0.5 = {headline}** "
        f"(P {summary.get('precision_50')}, R {summary.get('recall_50')}) "
        f"on {summary['mask_scored']} of {summary['calls']} scored frames "
        f"({summary['gt_total']} GT instances){failed_note}{unscored_note}.")
    add(f"Box cross-check (detection scorer, unchanged): "
        f"F1@IoU0.5 = {summary.get('box_f1_50')} "
        f"(P {summary.get('box_precision_50')}, R {summary.get('box_recall_50')}).")
    add("")
    add(f"- **Harness:** {info['name']} v{info['version']} "
        f"(fingerprint `{info['fingerprint']}`)")
    add(f"- **Base harness:** {base['name']} v{base['version']} "
        f"(fingerprint `{base['fingerprint']}`) — provider stack, parser and control")
    add(f"- **Model:** `{model}` via {args.provider} CLI "
        f"({run_summary.get('cli_version', '')})")
    add(f"- **Reasoning effort:** {effort or 'provider default'}")
    if rebuilt_with:
        add(f"- ⚠ artefacts rebuilt with harness fingerprint `{rebuilt_with}`; "
            f"records were produced by `{records[0].get('harness_fingerprint')}`.")
    add("")

    # Control
    add("## Vision delivery")
    if control is None:
        add("**Vision delivery: UNVERIFIED** — control skipped via --skip-control.")
    else:
        add("**Vision delivery: YES**" if delivered else "**Vision delivery: NO**")
        add("")
        add("| check | result |")
        add("| --- | --- |")
        add(f"| code | {control.get('prediction', {}).get('code')!r} "
            f"(expected {control['truth']['code']!r}) |")
        add(f"| red circle error | {control.get('circle_error_px')} px "
            f"(tolerance {control['tolerance_px']} px) |")
        add(f"| green square error | {control.get('green_square_error_px')} px |")
        add(f"| tool attempts | {control.get('tool_attempts') or '[]'} |")
        if not delivered:
            add("")
            add("Every number below is INVALID: the model did not receive the image.")
    add("")

    # Method
    ds = manifest["dataset"]
    add("## Method")
    add(f"- Dataset: StrawDI_Db1 `{ds['split']}` ({ds['n_selected']} of "
        f"{ds['n_total']} frames), {ds['root']}")
    if args.provider == "agy":
        fw = manifest['samples'][0]['image_shape_hw'][1]
        fh = manifest['samples'][0]['image_shape_hw'][0]
        add(f"- Frames delivered via `agy -p`: staged as the ONLY file of a fresh "
            f"per-call temp workspace, fetched with the built-in `view_file` — the "
            f"one sanctioned tool call. `view_file` resamples the {fw}x{fh} frame "
            f"to {vre.AGY_DELIVERED_W}x{vre.AGY_DELIVERED_H}; the prompt spoke that "
            f"space and the harness scaled every bbox AND polygon vertex back "
            f"before scoring. Fine detail is softer than in the codex/claude "
            f"runs — a delivery-path handicap, worst on small fruit.")
    else:
        add(f"- Frames delivered unchanged at "
            f"{manifest['samples'][0]['image_shape_hw'][1]}x"
            f"{manifest['samples'][0]['image_shape_hw'][0]} "
            f"(inside the 1280 px no-resize ceiling; GT maps 1:1).")
    add("- Prompt: `inventory_segmentation` — the detection inventory's EIGHT "
        "fields PLUS a ninth, `polygon`: an ordered vertex outline of the "
        "fruit's VISIBLE surface (follows occluder edges, never extrapolates "
        "the hidden shape; 8-20 vertices typical, 32 max; extent agrees with "
        "the reported bbox).")
    add("- Matching: greedy, confidence-descending, one GT mask per polygon, "
        "recomputed per threshold at mask IoU 0.25 / 0.5 / 0.75. Predictions "
        "rasterise with PIL polygon semantics; a vertex outside the frame or "
        "an empty raster is an automatic FP (the box scorer's rule).")
    add("- GT masks: the StrawDI label id-map PNGs (0 = background, 1..N = "
        "instance), read AFTER the call, sha256-guarded against the manifest "
        "snapshot; one bool mask per instance id, same order as the "
        "manifest's GT boxes.")
    add("- Box cross-check: the reported bboxes scored by the detection "
        "pipeline's scorer, unchanged — the bridge to `strawdi_eval` numbers.")
    add("")
    add(f"> {POLYGON_FIDELITY_NOTE}")
    add("")
    add(f"> {BBOX_SEMANTICS_NOTE}")
    add("")
    add(f"> {CONFIDENCE_TIE_NOTE}")
    add("")

    # Headline metrics
    add("## Headline metrics (mask IoU)")
    add("| metric | value |")
    add("| --- | --- |")
    add(f"| calls | {summary['calls']} |")
    add(f"| parsed (ok / no_pick_point) | {summary['parsed']} "
        f"({summary['ok']} / {summary['no_pick_point']}) |")
    add(f"| scored against GT masks | {summary['mask_scored']}"
        + (f" ({summary['mask_unscored_gt_error']} unscored)" if summary["mask_unscored_gt_error"] else "")
        + " |")
    add(f"| parse rate | {summary['parse_rate']} |")
    add(f"| schema-valid rate | {summary['schema_valid_rate']} |")
    for label, extra in (("IoU 0.25", "25"), ("IoU 0.50", "50"), ("IoU 0.75", "75")):
        add(f"| P / R / F1 @{label} | {summary.get(f'precision_{extra}')} / "
            f"{summary.get(f'recall_{extra}')} / **{summary.get(f'f1_{extra}')}** "
            f"(TP {summary[f'tp_{extra}']} FP {summary[f'fp_{extra}']} "
            f"FN {summary[f'fn_{extra}']}) |")
    add(f"| mean / median matched mask IoU@0.5 | {summary['mean_matched_iou_50']} / "
        f"{summary['median_matched_iou_50']} over {summary['matched_pairs_50']} pairs |")
    add(f"| AP@0.50 / mAP@[.50:.95] (mask) | {summary.get('mask_ap_50')} / "
        f"{summary.get('mask_map_50_95')}"
        + ("" if summary.get("mask_ap_available") else " (not recomputed — GT masks "
           "unavailable at artefact time)") + " |")
    add(f"| polygon-vs-bbox extent IoU (mean) | {summary['polygon_bbox_iou_mean']} |")
    add(f"| polygons out-of-frame / degenerate | {summary['n_polygons_out_of_frame']} / "
        f"{summary['n_polygons_degenerate']} |")
    add(f"| count error (pred − gt) | mean {summary['count_error_mean']}, "
        f"MAE {summary['count_error_mae']} |")
    add(f"| totals | {summary['gt_total']} GT, {summary['pred_total']} predicted |")
    add("")

    add("## Box cross-check (detection scorer on the asked bboxes)")
    add("| metric | value |")
    add("| --- | --- |")
    for label, extra in (("IoU 0.25", "25"), ("IoU 0.50", "50"), ("IoU 0.75", "75")):
        add(f"| F1 @{label} | {summary.get(f'box_f1_{extra}')} "
            f"(TP {summary[f'box_tp_{extra}']}) |")
    add(f"| AP@0.50 / mAP@[.50:.95] | {summary['box_ap_50']} / "
        f"{summary['box_map_50_95']} |")
    add("")
    if summary["failures_by_status"]:
        add(f"failures by status: {summary['failures_by_status']}")
        add("")
    if summary.get("retried_calls"):
        add(f"retried calls (recovered): {summary['retried_calls']} "
            f"({summary['retries_recovered']} parsed after retry; "
            f"empty→low-effort or schema-invalid→same-effort, all recorded)")
        add("")

    # Size strata
    add("## Recall by GT size (COCO bands, mask IoU@0.5)")
    add("| stratum | area px² | n GT | matched | recall | mean IoU@0.5 |")
    add("| --- | --- | --- | --- | --- | --- |")
    for row in summary["size_strata"]:
        add(f"| {row['stratum']} | {row['area_band_px']} | {row['n_gt']} | "
            f"{row['n_matched']} | {row['recall']} | {row['mean_matched_iou_50']} |")
    add("")

    # Occlusion diagnostic
    occ = summary["occlusion_split"]
    add("## Occlusion diagnostic (mask IoU vs reported occlusion)")
    add("| quantity | value |")
    add("| --- | --- |")
    add(f"| matched mask IoU@0.5, pred occlusion < 25% | {occ['matched_iou_occl_lt25_mean']} "
        f"(n={occ['n_matched_occl_lt25']}) |")
    add(f"| matched mask IoU@0.5, pred occlusion ≥ 25% | {occ['matched_iou_occl_ge25_mean']} "
        f"(n={occ['n_matched_occl_ge25']}) |")
    add(f"| mean reported occlusion, matched vs unmatched preds | "
        f"{occ['mean_reported_occlusion_matched']} vs "
        f"{occ['mean_reported_occlusion_unmatched']} |")
    add("")

    # Per-image table
    add("## Per-image results")
    add("| sample | status | gt | pred | TP | FP | FN | mask IoU@0.5 | box TP@0.5 | Δcount | tok |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in records:
        tok = r.get("total_tokens")
        box_tp = (r.get("box_metrics") or {}).get("tp_50")
        add(f"| {r['sample_id']} | {r['status']} | {r.get('n_gt')} | "
            f"{r.get('n_pred')} | {r.get('tp_50')} | {r.get('fp_50')} | "
            f"{r.get('fn_50')} | {r.get('mean_matched_iou_50')} | "
            f"{box_tp} | {r.get('count_error')} | {tok} |")
    add("")

    # Error analysis
    scored = [r for r in records if r.get("tp_50") is not None]
    worst_miss = sorted(
        (r for r in scored if r.get("fn_gt_areas_50")),
        key=lambda r: (sum(r["fn_gt_areas_50"]), len(r["fn_gt_areas_50"])),
        reverse=True)[:10]
    high_conf_fp = []
    for r in scored:
        fp_indices = set(r.get("fp_pred_indices_50") or [])
        for i in fp_indices:
            inventory = r.get("inventory") or []
            conf = (inventory[i].get("confidence_pct")
                    if i < len(inventory) and isinstance(inventory[i], dict) else None)
            high_conf_fp.append((conf if isinstance(conf, (int, float)) else -1,
                                 r["sample_id"], i))
    high_conf_fp = sorted(high_conf_fp, reverse=True)[:10]
    add("## Error analysis")
    add("")
    add("**Worst misses by GT area** (largest unfound instances first): "
        + (", ".join(f"{r['sample_id']} (miss {len(r['fn_gt_areas_50'])}, "
                     f"area sum {sum(r['fn_gt_areas_50'])})"
                     for r in worst_miss) or "none"))
    add("")
    add("**Highest-confidence false positives**: "
        + (", ".join(f"{sid}#{idx} (conf {conf})" for conf, sid, idx in high_conf_fp)
           or "none"))
    add("")

    # Runs + usage
    add("## Every run")
    add("| run | status | attempts | wall s | tokens | cost USD |")
    add("| --- | --- | --- | --- | --- | --- |")
    for r in records:
        add(f"| {r['run_id']} | {r['status']} | {r.get('attempts')} | "
            f"{r.get('wall_s')} | {r.get('total_tokens')} | "
            f"{r.get('cost_usd') if r.get('cost_usd') is not None else '–'} |")
    add("")
    add("## Tokens / time / cost")
    add(f"- input tokens: {summary['total_input_tokens']:,}, "
        f"output tokens: {summary['total_output_tokens']:,} "
        f"(total {summary['total_tokens']:,})")
    add(f"- summed call wall time: {summary['wall_s_total']} s; batch elapsed: "
        f"{elapsed:.0f} s at --jobs {args.jobs}")
    cost = summary.get("cost_usd_total")
    add(f"- cost: {'$%.2f' % cost if cost is not None else 'not reported by endpoint'}")
    add(f"- started {started}")
    add("")

    # Overlays
    add("## Overlays")
    add("Per-run overlays in `overlays/` (white translucent fill = GT instance "
        "mask — a missed one carries a red `MISS` label at its GT-box "
        "position; green polygon = prediction matched as TP, red = unmatched "
        "prediction (FP); label text takes its polygon colour except the "
        "`red xx%` segment, which is redness-coloured; small legend at the "
        "bottom-left); contact sheets in `contact_sheets/` (chunks of 24) and "
        "`contact_sheets/miss_gallery.jpg`. Overlays and sheets are JPG "
        "(quality 90); the model inputs are untouched PNG.")
    add("")

    # Exact prompt
    add("## Exact prompt (verbatim, first frame)")
    add("")
    add("```text")
    add(records[0]["prompt"] if records and records[0].get("prompt")
        else build_prompt(args, manifest["samples"][0]))
    add("```")
    add("")

    add("## Provenance")
    add(f"- run directory name encodes model/effort/CLI/harness: "
        f"`{run_dir.name}`")
    add(f"- base-harness fingerprint covers the provider stack, parser and "
        f"control the numbers depend on: `{base['fingerprint']}`")
    add(f"- this harness's fingerprint covers the seg subtree only "
        f"(`strawdi_eval/seg/`): `{info['fingerprint']}` — the detection "
        f"pipeline's glob does not see this directory and vice versa.")
    add(f"- manifest snapshot + catalog snapshot in this directory; "
        f"records in `responses.jsonl` carry the full identity per call.")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")


def write_artifacts(run_dir: Path, args, manifest, control, records: list[dict],
                    started: str, elapsed: float, run_summary: dict,
                    rebuilt_with: str | None = None) -> None:
    samples = {s["sample_id"]: s for s in manifest["samples"]}

    # Re-derive the scored fields from the stored inventories so a rebuild
    # after a scoring fix re-scores every old run. The box cross-check is
    # always offline-safe (GT boxes ride in the record); the mask block needs
    # the label PNGs — without the mount a rebuild keeps the stored mask
    # numbers rather than wiping them.
    masks_by_sample: dict[str, list[np.ndarray]] = {}
    for record in records:
        sample = samples.get(record["sample_id"], {})
        try:
            masks_by_sample[record["sample_id"]] = load_gt_masks(sample)
        except Exception as exc:
            record.setdefault("gt_mask_error", f"{type(exc).__name__}: {exc}")
            record["gt_masks_loaded"] = False
        if record["sample_id"] in masks_by_sample:
            record["gt_masks_loaded"] = True
            record.pop("gt_mask_error", None)
            record.update(mask_block(record["status"], record.get("inventory"),
                                     masks_by_sample[record["sample_id"]],
                                     sample["gt_areas"], record["frame_w"],
                                     record["frame_h"], sample["n_gt"]))
        record["box_metrics"] = box_metrics_block(
            record["status"], record.get("inventory"),
            record.get("gt_boxes"), record.get("gt_areas"),
            record["frame_w"], record["frame_h"])
    records.sort(key=lambda r: r["sample_id"])

    # Mask AP needs rasterised predictions + GT masks; compute it only when
    # EVERY parsed record re-derived against its masks (else it would be a
    # number over a subset, silently).
    parsed = [r for r in records
              if r["status"] in (parse.OK, parse.NO_PICK_POINT)]
    mask_ap = None
    if parsed and all(r["sample_id"] in masks_by_sample for r in parsed):
        mask_ap = seg_scoring.ap_metrics_masks([{
            "sample_id": r["sample_id"],
            "preds": seg_scoring.prepare_predictions_masks(
                r.get("inventory"), r["frame_w"], r["frame_h"]),
            "gt_masks": masks_by_sample[r["sample_id"]],
        } for r in parsed])

    # Render BEFORE persisting: _render_all updates each record's overlay
    # field, and those updates must land in the written artefacts.
    _render_all(run_dir, records, masks_by_sample)
    (run_dir / "responses.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records))
    summary = seg_summary(records, mask_ap)
    write_seg_csvs(run_dir, records, summary)
    write_report(run_dir, args, manifest, control, records, summary, started,
                 elapsed, run_summary, rebuilt_with)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def select_samples(args, manifest) -> list[dict]:
    samples = manifest["samples"]
    if args.sample_id:
        samples = [s for s in samples if s["sample_id"] == args.sample_id]
        if not samples:
            raise SystemExit(f"no manifest sample with sample_id {args.sample_id!r}")
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit("no samples selected")
    return samples


def dry_run(args, manifest, model) -> None:
    samples = select_samples(args, manifest)
    print(f"dry run : {len(samples)} frame(s), 1 prompt ({STYLE_NAME}), "
          f"1 control call; provider={args.provider} model={model}")
    print(f"plan    : {len(samples) + 1} model call(s) if run for real")
    sample = samples[0]
    h, w = sample["image_shape_hw"]
    try:
        load_gt_masks(sample)
        gt = f"{sample['n_gt']} GT (label masks verified)"
    except Exception as exc:
        gt = f"{sample['n_gt']} GT (WARNING: label masks unavailable: {exc})"
    print(f"first   : {sample['sample_id']} ({w}x{h}, {gt})")
    print("\n--- prompt (verbatim) " + "-" * 40)
    print(build_prompt(args, sample))
    print("--- end prompt " + "-" * 46)


def rebuild_report(args) -> None:
    run_dir = args.out
    records = [json.loads(line) for line in
               (run_dir / "responses.jsonl").read_text().splitlines() if line.strip()]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    control = (json.loads((run_dir / "control.json").read_text())
               if (run_dir / "control.json").exists() else None)
    catalog = json.loads((run_dir / "catalog.json").read_text())
    info = harness_info()
    old = next((r.get("harness_fingerprint") for r in records if r.get("harness_fingerprint")), None)
    run_summary = {
        "model": records[0].get("model"),
        "effort": records[0].get("reasoning_effort"),
        "provider": records[0].get("provider"),
        "catalog": catalog,
        "harness": info,
        "base_harness": {
            "name": records[0].get("base_harness"),
            "version": records[0].get("base_harness_version"),
            "fingerprint": records[0].get("base_harness_fingerprint"),
        },
    }
    rebuilt_with = info["fingerprint"] if old and old != info["fingerprint"] else None
    write_artifacts(run_dir, args, manifest, control, records,
                    started=records[0].get("started", ""),
                    elapsed=0.0, run_summary=run_summary,
                    rebuilt_with=rebuilt_with)
    print(f"rebuilt : {run_dir / 'report.md'}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.rebuild_report:
        rebuild_report(args)
        return

    manifest = json.loads(args.manifest.read_text())

    if args.provider == "agy":
        model = args.agy_model or args.model or vre.AGY_DEFAULT_MODEL
        effort = args.reasoning_effort
        catalog_info = vre.agy_catalog_info(args, model)
    elif args.provider == "claude":
        model = args.claude_model or args.model or "glm-5.3-flash"
        effort = args.reasoning_effort
        if re.split(r"[\[]", model)[0].strip() == "glm-5.3":
            print("warning: the flagship glm-5.3 slug rejects image content on this "
                  "key; the vision control will fail and the batch will be reported "
                  "INVALID. glm-5.3-flash is the multimodal GLM-5.3.", file=sys.stderr)
        catalog_info = vre.claude_catalog_info(args, model)
    else:
        model = args.model or "gpt-5.4-codex"
        effort = args.reasoning_effort
        catalog_info = vre.catalog_entry(args, model)

    samples = select_samples(args, manifest)

    if args.dry_run:
        dry_run(args, manifest, model)
        return

    info = harness_info()
    base = vre.harness_info()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    cli_name = {"claude": "claude", "agy": "agy"}.get(args.provider, "codex")
    run_dir = args.out / run_dir_name(stamp, model, effort, args.tag, cli=cli_name)
    run_dir.mkdir(parents=True, exist_ok=False)
    tmp = run_dir / ".tmp"
    tmp.mkdir()
    agent_cwd = HERE / "agent_cwd"
    agent_cwd.mkdir(exist_ok=True)

    shutil.copyfile(args.manifest, run_dir / "manifest.json")
    (run_dir / "catalog.json").write_text(json.dumps(catalog_info, indent=2) + "\n")

    run_ctx = {
        "run_dir": run_dir,
        "tmp": tmp,
        "agent_cwd": agent_cwd,
        "model": model,
        "effort": effort,
        "harness": info,
        "base_harness": base,
    }

    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()

    # The manifest builder enforces the ceiling; this is the belt-and-braces
    # gate from the base harness (identical function, verbatim).
    vre.check_input_sizes(samples, None)

    print(f"run dir : {run_dir}")
    print(f"model   : {model}  provider={args.provider}  "
          f"effort={effort or 'config default'}")
    print(f"plan    : {len(samples)} frames x 1 prompt ({STYLE_NAME}) "
          f"= {len(samples)} calls + 1 control")

    # The control image path must be resolved against the STRAWDI directory
    # (manifest paths are relative to it; the base run_control defaults to
    # its own HERE).
    if not args.control_image:
        args.control_image = STRAWDI / manifest["control"]["image"]

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
        """Persist each finished call immediately: a crash later in the batch
        can never throw away completed work."""
        with write_lock:
            records.append(record)
            with partial_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")

    total = len(samples)
    done = 0

    def guarded(sample):
        try:
            return run_one(args, manifest, sample, run_ctx)
        except Exception as exc:  # keep the batch alive
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

    records.sort(key=lambda r: r["sample_id"])
    elapsed = time.perf_counter() - t0

    run_summary = {"model": model, "effort": effort, "provider": args.provider,
                   **vre.cli_identity(args.provider), "catalog": catalog_info,
                   "harness": info, "base_harness": base}
    write_artifacts(run_dir, args, manifest, control, records, started, elapsed,
                    run_summary)

    partial_path.unlink(missing_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)
    summary = seg_summary(records, None)
    print(f"\ndone in {elapsed:.0f}s -> {run_dir}")
    print(f"report  : {run_dir / 'report.md'}")
    print(f"metrics : mask F1@0.5 {summary.get('f1_50')}  "
          f"P {summary.get('precision_50')}  R {summary.get('recall_50')}  "
          f"| box cross-check F1@0.5 {summary.get('box_f1_50')}")


def _progress(done: int, total: int, record: dict) -> None:
    tp = record.get("tp_50")
    fp = record.get("fp_50")
    fn = record.get("fn_50")
    det = (f"TP{tp} FP{fp} FN{fn}" if tp is not None else "–")
    print(f"[{done}/{total}] {record['run_id']:<44} {record['status']:<13} "
          f"{det:<12} tok={record.get('total_tokens')}", flush=True)


if __name__ == "__main__":
    main()
