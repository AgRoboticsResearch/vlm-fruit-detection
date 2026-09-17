#!/usr/bin/env python3
"""VLM segmentation eval on the vlm_eval scenes (SROI + shunba).

Executes the ONE prompt of the seg pipeline (``inventory_segmentation``: the
full_detection pipeline's nine-field inventory + pick nomination, plus a
``polygon`` tracing each fruit's VISIBLE surface — ten fields) on manifest
samples. No vlm_eval source has mask ground truth (the SROI frames carry one
recomputed picking point each; the shunba scenes are unlabelled), so the
polygons are never scored as segmentation here — that comparison lives in
the StrawDI pipeline (``strawdi_eval/seg/``). This pipeline instead:

* renders and diagnoses the polygons everywhere (validity counts, vertex
  budget, polygon-vs-bbox extent IoU, picking-point-to-own-outline distance);
* on the ground-truthed SROI frames (`--sources validation occluded shunba`),
  scores the nominated target exactly like ``full_detection`` (picking-point
  error / PCK — the cross-check that lines the two pipelines up), plus the
  polygon extent's IoU against the upstream rough box and the GT point's
  distance to the target outline.

The base harness (``vlm_eval/``) supplies everything model-facing: the
provider stack (claude/codex/agy CLIs, tool surfaces off, images as base64
blocks), the reply parser and the synthetic vision-delivery control. The
base-harness invariants carry over unchanged:

1. the synthetic vision-delivery control runs FIRST and invalidates the batch
   on failure;
2. all tool surfaces stay disabled (``--tools "" --safe-mode
   --strict-mcp-config --no-session-persistence``, empty agent_cwd; the event
   stream is scanned for tool_use and any attempt fails acceptance);
3. no annotated image is ever an input (the GT marker renderings are never
   inputs; GT is recomputed by the manifest builder, not pasted);
4. inputs stay inside the no-resize ceiling (the manifest builder enforces it);
5. every record and report states harness, base harness, model and effort.

This pipeline lives in ``vlm_eval/seg/`` — deliberately outside the base
harness's fingerprint glob (top-level ``vlm_eval/*.py``, ``lib/``,
``schema/``), the same independence trick as ``strawdi_eval/seg/``.

Usage (repo root):
    python3 vlm_eval/seg/run_segmentation_eval.py --dry-run
    python3 vlm_eval/seg/run_segmentation_eval.py --limit 1 --tag smoke
    python3 vlm_eval/seg/run_segmentation_eval.py --sources validation occluded shunba --jobs 3 --tag full
    python3 vlm_eval/seg/verify_vlm_seg_run.py vlm_eval/seg/runs/<dir>
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

HERE = Path(__file__).resolve().parent          # vlm_eval/seg
VLM = HERE.parent                                # vlm_eval
REPO = VLM.parent                                # repo root
sys.path.insert(0, str(REPO))

from vlm_eval.lib import imaging, parse  # noqa: E402
import vlm_eval.run_vlm_eval as vre  # noqa: E402
from vlm_eval.seg.lib import prompt as seg_prompt  # noqa: E402
from vlm_eval.seg.lib import render, seg_scoring  # noqa: E402

HARNESS_NAME = "vlm_seg"
HARNESS_VERSION = "0.1.0"

# Same default scope as the base harness: the unlabelled shunba scenes only.
# The ground-truthed SROI frames are opt-in via --sources, exactly like
# run_vlm_eval.py.
DEFAULT_SOURCES = ("shunba",)

# The pipeline's single prompt and its answer schema: the full_detection
# inventory's nine fields PLUS a polygon of the visible surface (ten fields)
# and the unchanged target_index nomination. See lib/prompt.py.
STYLE_NAME = seg_prompt.STYLE_NAME
SCHEMA_PATH = HERE / "schema" / "inventory_segmentation_schema.json"

DEFAULT_RUNS_DIR = HERE / "runs"
DEFAULT_JOBS = 3
CONTACT_SHEET_CHUNK = 24          # per sheet, 3 columns

# Printed in every report: the polygons here are qualitative + internally
# consistent, never scored against a mask — no vlm_eval source has one.
NO_MASK_GT_NOTE = (
    "No vlm_eval source has mask ground truth: the SROI frames carry ONE "
    "recomputed picking point each (plus the upstream rough 60x60 box "
    "derived from it) and the shunba scenes are unlabelled. The polygons "
    "are therefore NEVER scored as segmentation on these scenes — no mask "
    "IoU, no mask P/R/F1, no mask AP exists in this pipeline. Scored "
    "segmentation lives in the StrawDI pipeline (strawdi_eval/seg). What "
    "this run measures: the polygons' internal consistency (extent vs the "
    "answer's own bbox, picking point vs own outline) and, on the "
    "ground-truthed frames, the nominated target's picking point (the "
    "full_detection cross-check) plus the polygon extent vs the rough box."
)
POLYGON_FIDELITY_NOTE = (
    "The polygon contract bounds fidelity: straight segments between <= 32 "
    "whole-pixel vertices, rasterised with PIL polygon semantics. Without "
    "mask ground truth this cannot be quantified here — on StrawDI, where "
    "GT masks exist, the same contract measured a mask-IoU ceiling a "
    "fraction below 1 for a perfect outliner."
)
ROUGH_BOX_NOTE = (
    "polygon_extent_iou_rough and bbox_iou_rough are APPROXIMATE: they "
    "compare against the upstream rough 60x60 box centred 30 px below the "
    "GT picking point (the only fruit-body reference this dataset has — it "
    "has no hand-annotated fruit boxes), and the polygon traces the VISIBLE "
    "surface while the rough box approximates the whole fruit."
)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def harness_info() -> dict:
    """Fingerprint of THIS pipeline's code (not the base harness's).

    Hashes the seg subtree only — changing the base harness must not shift
    this fingerprint, and vice versa (the base harness's glob does not see
    this directory).
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
    """``<ts>-<model>-<effort>-<cli>-vlm_seg[-<tag>]`` (base convention)."""
    parts = [stamp, vre.slugify(model), vre.slugify(effort or "default")]
    if cli:
        parts.append(vre.slugify(cli))
    parts.append(vre.slugify(HARNESS_NAME))
    if tag:
        parts.append(vre.slugify(tag))
    return "-".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=VLM / "manifest.json",
                    help="the vlm_eval manifest (frames + GT provenance); the seg "
                         "pipeline reuses the base pipeline's manifest as-is")
    ap.add_argument("--sources", nargs="*", default=None,
                    help="only these manifest sources. Default: "
                         f"{' '.join(DEFAULT_SOURCES)} (unlabelled, qualitative). "
                         "Pass 'validation occluded shunba' to include the "
                         "ground-truthed SROI frames and the scored cross-check.")
    ap.add_argument("--out", type=Path, default=DEFAULT_RUNS_DIR,
                    help="runs root; a timestamped subdirectory is created inside it")
    ap.add_argument("--tag", default="", help="suffix for the run directory name")
    ap.add_argument("--limit", type=int, default=None,
                    help="max samples (manifest order)")
    ap.add_argument("--sample-id", default=None,
                    help="run a single manifest sample by id (e.g. sb03)")
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
                    default=VLM / "model_catalog_vision.json",
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


def select_samples(args, manifest) -> list[dict]:
    """Apply --sources then --sample-id then --limit, in manifest order."""
    samples = manifest["samples"]
    requested = set(args.sources) if args.sources else set(DEFAULT_SOURCES)
    known = {s["source"] for s in samples}
    unknown = sorted(requested - known)
    if unknown:
        raise SystemExit(f"unknown source(s): {unknown}; known: {sorted(known)}")
    samples = [s for s in samples if s["source"] in requested]
    if args.sample_id:
        samples = [s for s in samples if s["sample_id"] == args.sample_id]
        if not samples:
            raise SystemExit(f"no manifest sample with sample_id {args.sample_id!r}")
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit("no samples selected")
    return samples


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


# The per-fruit attributes kept from the inventory besides bbox/polygon/
# picking_point (the scorer reads those three separately).
SEG_ATTRS = ("redness_pct", "occlusion_pct", "calyx_visible",
             "peduncle_visible", "graspable", "confidence_pct", "description")


def classify_segmentation(last_message: str, schema: dict,
                          frame_w: int, frame_h: int) -> dict:
    """Parse a segmentation inventory (ten fields + target nomination).

    Mirrors the base harness's ``classify_inventory`` with the polygon
    normalised alongside the bbox. A bare top-level list is accepted and
    treated as an inventory with no nomination (target_index -1).
    """
    result = {
        "status": None, "json_method": None, "schema_valid": None,
        "schema_error": None, "strawberries": [], "n_strawberries": 0,
        "target_index": None, "target": None, "target_valid": None,
        "n_out_of_frame": 0,
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

    payload = ({"strawberries": obj, "target_index": -1} if isinstance(obj, list)
               else obj)
    valid, error = parse.validate(payload, schema)
    result["schema_valid"] = valid
    result["schema_error"] = error
    if not valid:
        result["status"] = parse.SCHEMA_INVALID
        return result

    berries = []
    for entry in payload["strawberries"]:
        box = parse.normalise_box(entry.get("bbox"))
        point = parse.normalise_point(entry.get("picking_point"))
        berry = {"bbox": list(box) if box else None,
                 "picking_point": list(point) if point else None}
        berry.update({key: entry.get(key) for key in ("polygon", *SEG_ATTRS)})
        berries.append(berry)

    result["strawberries"] = berries
    result["n_strawberries"] = len(berries)
    index = payload.get("target_index")
    if isinstance(index, int) and 0 <= index < len(berries):
        result["target_index"] = index
        result["target"] = berries[index]
        result["target_valid"] = True
    else:
        result["target_index"] = -1 if index in (-1, None) else index
        result["target_valid"] = index in (-1, None)
    result["n_out_of_frame"] = sum(
        1 for b in berries
        if b["picking_point"] is not None
        and not (0 <= b["picking_point"][0] < frame_w
                 and 0 <= b["picking_point"][1] < frame_h)
    )
    # Same rule as the base harness: a parsed inventory whose nominee offers
    # no grasp point (the peduncle is hidden) is a valid answer that yields
    # no point number — its own status, never a failure and never a zero.
    result["status"] = (parse.OK if (result["target"] or {}).get("picking_point")
                        else parse.NO_PICK_POINT)
    return result


def diag_block(status: str, inventory, frame_w: int, frame_h: int) -> dict:
    """Polygon diagnostics — parsed answers only, never zeros from failures."""
    if status not in (parse.OK, parse.NO_PICK_POINT):
        return seg_scoring.empty_diag()
    return seg_scoring.polygon_diag_block(inventory, frame_w, frame_h)


def scored_block(status: str, inventory, target_index, sample,
                 frame_w: int, frame_h: int) -> dict:
    """The full_detection nomination cross-check + polygon-vs-rough-box IoU.

    Ground-truthed frames only; unlabelled frames and failure statuses carry
    None everywhere (I7: unlabelled runs carry no scored value at all).
    """
    if not sample.get("has_gt") or status not in (parse.OK, parse.NO_PICK_POINT):
        return seg_scoring.empty_scored()
    return seg_scoring.score_gt_frame(inventory, target_index,
                                      sample.get("gt_uv"),
                                      sample.get("gt_rough_box"),
                                      frame_w, frame_h)


def run_one(args, manifest, sample, run_ctx) -> dict:
    """Execute (and if needed retry) one model call, then score it."""
    image = VLM / sample["images"]["raw"]
    h, w = sample["image_shape_hw"]
    has_gt = bool(sample.get("has_gt"))
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

    scored = classify_segmentation(attempt["last_message"], schema, w, h)
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
        scored = classify_segmentation(retry["last_message"], schema, w, h)

    totals = vre.merge_usage(attempts)
    parsed_ok = scored["status"] in (parse.OK, parse.NO_PICK_POINT)
    inventory = scored["strawberries"] if parsed_ok else None
    if inventory and args.provider == "agy":
        # The model answered in the delivered 800x600 frame; scale every point,
        # bbox and polygon vertex back to the original coordinate space before
        # scoring. Everything downstream sees original coords.
        def scale_point(pt):
            return [int(round(v)) for v in vre.agy_scale_point(pt, w, h)]
        inventory = [dict(item,
                          bbox=vre.agy_scale_box(item["bbox"], w, h)
                          if item.get("bbox") else None,
                          picking_point=scale_point(item["picking_point"])
                          if item.get("picking_point") else None,
                          polygon=[scale_point(pt) for pt in item["polygon"]]
                          if item.get("polygon") else None)
                     for item in inventory]
        if scored.get("target") is not None:
            scored["target"] = inventory[scored["target_index"]]

    target = scored.get("target") if parsed_ok else None
    pred = (target or {}).get("picking_point")
    error_px = None
    scored_fields = scored_block(scored["status"], inventory,
                                 scored.get("target_index"), sample, w, h)
    if scored_fields["error_px"] is not None:
        error_px = scored_fields["error_px"]

    record = {
        "run_id": stem,
        "style": STYLE_NAME,
        "sample_id": sample["sample_id"],
        "source": sample["source"],
        "session": sample["session"],
        "episode": sample["episode"],
        "has_gt": has_gt,
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
        "target_index": scored.get("target_index") if parsed_ok else None,
        "target_valid": scored.get("target_valid") if parsed_ok else None,
        "target_scored": bool(has_gt and parsed_ok and pred is not None),
        "n_points_out_of_frame": scored.get("n_out_of_frame") if parsed_ok else None,
        **vre._inventory_summary(inventory),
        # Ground truth rides inside the record so the run is self-contained
        # and --rebuild-report re-scores the cross-check without the dataset.
        "gt_status": sample.get("gt_status") if has_gt else None,
        "gt_uv": sample["gt_uv"] if has_gt else None,
        "gt_uv_rounded": sample["gt_uv_rounded"] if has_gt else None,
        "gt_z_m": sample["gt_z_m"] if has_gt else None,
        "gt_rough_box": sample["gt_rough_box"] if has_gt else None,
        "picking_point": pred,
        "bbox": (target or {}).get("bbox"),
        "error_px": error_px,
        "dx_px": scored_fields["dx_px"],
        "dy_px": scored_fields["dy_px"],
        "error_pct_width": None if error_px is None
        else round(error_px / float(w) * 100.0, 4),
        "bbox_iou_rough": scored_fields["bbox_iou_rough"],
        "polygon_extent_iou_rough": scored_fields["polygon_extent_iou_rough"],
        "polygon_contains_gt": scored_fields["polygon_contains_gt"],
        "gt_to_target_polygon_px": scored_fields["gt_to_target_polygon_px"],
        **{f"pck_{t}": None if error_px is None else bool(error_px <= t)
           for t in seg_scoring.PCK_THRESHOLDS},
        **diag_block(scored["status"], inventory, w, h),
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

    title = (f"{STYLE_NAME} | {sample['sample_id']} | {sample['episode']} | "
             f"{scored['status']}")
    rel = Path("overlays") / f"{stem}.jpg"
    try:
        raw = imaging.load_rgb(image)
        gt = None
        if has_gt and sample.get("gt_uv") is not None:
            gt = {"uv": sample["gt_uv"], "rough_box": sample.get("gt_rough_box"),
                  "error_px": error_px}
        overlay = render.draw_segmentation_overlay(
            raw, inventory, scored.get("target_index"), gt, title)
        imaging.save_jpg(overlay, run_ctx["run_dir"] / rel)
        record["overlay"] = str(rel)
    except Exception as exc:  # a drawing bug must not throw away a paid-for answer
        record["overlay"] = None
        record["render_error"] = f"{type(exc).__name__}: {exc}"
    return record


def failure_record(sample: dict, exc: BaseException, run_ctx) -> dict:
    """A run that blew up in the harness itself, recorded rather than raised."""
    has_gt = bool(sample.get("has_gt"))
    return {
        "run_id": f"{STYLE_NAME}__{sample['sample_id']}",
        "style": STYLE_NAME, "sample_id": sample["sample_id"],
        "source": sample["source"], "session": sample["session"],
        "episode": sample["episode"], "has_gt": has_gt,
        "task": sample.get("task", "full_inventory"),
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
        "target_index": None, "target_valid": None, "target_scored": False,
        "n_points_out_of_frame": None,
        **vre._inventory_summary(None),
        "gt_status": sample.get("gt_status") if has_gt else None,
        "gt_uv": sample["gt_uv"] if has_gt else None,
        "gt_uv_rounded": sample["gt_uv_rounded"] if has_gt else None,
        "gt_z_m": sample["gt_z_m"] if has_gt else None,
        "gt_rough_box": sample["gt_rough_box"] if has_gt else None,
        "picking_point": None, "bbox": None,
        "error_px": None, "dx_px": None, "dy_px": None, "error_pct_width": None,
        "bbox_iou_rough": None, "polygon_extent_iou_rough": None,
        "polygon_contains_gt": None, "gt_to_target_polygon_px": None,
        **{f"pck_{t}": None for t in seg_scoring.PCK_THRESHOLDS},
        **seg_scoring.empty_diag(),
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


def _rate(records: list[dict], predicate) -> float | None:
    if not records:
        return None
    return round(sum(1 for r in records if predicate(r)) / len(records), 4)


def seg_summary(records: list[dict]) -> dict:
    """Every batch-level number the report prints, derived from records only."""
    parsed = [r for r in records
              if r["status"] in (parse.OK, parse.NO_PICK_POINT)]
    scored = [r for r in parsed if r.get("error_px") is not None]
    failed = [r for r in records if r["status"] in parse.FAILURE_STATUSES]

    summary = {
        "calls": len(records),
        "parsed": len(parsed),
        "parse_rate": round(len(parsed) / len(records), 4) if records else None,
        "schema_valid_rate": _rate(records, lambda r: r.get("schema_valid") is True),
        "ok": sum(1 for r in parsed if r["status"] == parse.OK),
        "no_pick_point": sum(1 for r in parsed if r["status"] == parse.NO_PICK_POINT),
        "gt_frames": sum(1 for r in records if r.get("has_gt")),
        "unlabelled_frames": sum(1 for r in records if not r.get("has_gt")),
        "scored_runs": len(scored),
        "failures_by_status": {
            status: sum(1 for r in failed if r["status"] == status)
            for status in sorted({r["status"] for r in failed})
        },
        "retried_calls": sum(1 for r in records if r.get("fallback_used")),
        "retries_recovered": sum(
            1 for r in records if r.get("fallback_used")
            and r["status"] in (parse.OK, parse.NO_PICK_POINT)),
    }

    # Polygon diagnostics over every parsed record.
    summary["polygons_total"] = sum(r.get("n_pred") or 0 for r in parsed)
    summary["polygons_ok"] = sum(r.get("n_polygons_ok") or 0 for r in parsed)
    summary["polygons_out_of_frame"] = sum(
        r.get("n_polygons_out_of_frame") or 0 for r in parsed)
    summary["polygons_degenerate"] = sum(
        r.get("n_polygons_degenerate") or 0 for r in parsed)
    summary["mean_vertex_count"] = _mean(
        [r.get("mean_vertex_count") for r in parsed])
    summary["polygon_bbox_iou_mean"] = _mean(
        [r.get("polygon_bbox_iou_mean") for r in parsed])
    summary["pick_to_own_polygon_px_mean"] = _mean(
        [r.get("pick_to_own_polygon_px_mean") for r in parsed])

    # Nominated-target cross-check, ground-truthed frames only.
    if scored:
        errors = [r["error_px"] for r in scored]
        summary["median_error_px"] = round(float(np.median(errors)), 2)
        summary["mean_error_px"] = _mean(errors)
        for threshold in seg_scoring.PCK_THRESHOLDS:
            summary[f"pck_{threshold}"] = _rate(
                scored, lambda r, t=threshold: r.get(f"pck_{t}") is True)
        summary["mean_bbox_iou_rough"] = _mean(
            [r.get("bbox_iou_rough") for r in scored if r.get("bbox_iou_rough") is not None])
        summary["mean_polygon_extent_iou_rough"] = _mean(
            [r.get("polygon_extent_iou_rough") for r in scored
             if r.get("polygon_extent_iou_rough") is not None])
        summary["polygon_contains_gt_rate"] = _rate(
            [r for r in parsed if r.get("has_gt")
             and r.get("polygon_contains_gt") is not None],
            lambda r: r.get("polygon_contains_gt") is True)
        summary["gt_to_target_polygon_px_mean"] = _mean(
            [r.get("gt_to_target_polygon_px") for r in parsed
             if r.get("gt_to_target_polygon_px") is not None])

    summary["total_input_tokens"] = sum(r.get("input_tokens") or 0 for r in records)
    summary["total_output_tokens"] = sum(r.get("output_tokens") or 0 for r in records)
    summary["total_tokens"] = sum(r.get("total_tokens") or 0 for r in records)
    summary["wall_s_total"] = round(sum(r.get("wall_s") or 0 for r in records), 1)
    summary["cost_usd_total"] = round(sum(r.get("cost_usd") or 0 for r in records), 4) \
        if any(r.get("cost_usd") is not None for r in records) else None
    return summary


# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------

CSV_DROP = ("prompt", "response_text", "attempts_detail", "inventory")


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


def _render_all(run_dir: Path, records: list[dict]) -> None:
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
                 f"{record.get('episode')} | {record['status']}")
        rel = Path("overlays") / f"{record['run_id']}.jpg"
        try:
            raw = imaging.load_rgb(VLM / record["image"])
            gt = None
            if record.get("has_gt") and record.get("gt_uv") is not None:
                gt = {"uv": record["gt_uv"], "rough_box": record.get("gt_rough_box"),
                      "error_px": record.get("error_px")}
            overlay = render.draw_segmentation_overlay(
                raw, record.get("inventory"), record.get("target_index"),
                gt, title)
            imaging.save_jpg(overlay, run_dir / rel)
            record["overlay"] = str(rel)
            record.pop("render_error", None)
        except Exception as exc:
            record["overlay"] = None
            record["render_error"] = f"{type(exc).__name__}: {exc}"
            continue
        rendered.append((record, run_dir / rel))

    # Ground-truthed and unlabelled frames get separate sheets: only the
    # former carry GT references, and mixing them would be misleading.
    for gt_flag, suffix in ((True, ""), (False, "__unlabelled")):
        subset = [(r, p) for r, p in rendered if bool(r.get("has_gt")) == gt_flag]
        for start in range(0, len(subset), CONTACT_SHEET_CHUNK):
            chunk = subset[start:start + CONTACT_SHEET_CHUNK]
            images = [imaging.load_rgb(path) for _, path in chunk]
            titles = [rec["sample_id"] for rec, _ in chunk]
            sheet = imaging.contact_sheet(images, titles, cols=3)
            imaging.save_jpg(sheet,
                             sheets / f"sheet_{start // CONTACT_SHEET_CHUNK + 1:02d}{suffix}.jpg")


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
    add(f"# VLM segmentation eval (vlm_eval scenes) — {verdict}")
    add("")
    if summary.get("scored_runs"):
        failed_note = (f"; {summary['calls'] - summary['parsed']} call(s) failed to "
                       f"parse and are excluded from every metric"
                       if summary["calls"] != summary["parsed"] else "")
        add(f"Headline: nominated-target picking-point median error "
            f"**{summary.get('median_error_px')} px** over {summary['scored_runs']} "
            f"scored frame(s) (the full_detection cross-check); polygons are "
            f"qualitative + diagnostics on all {summary['calls']} frames — no mask "
            f"ground truth exists on any vlm_eval source{failed_note}.")
    else:
        add(f"Headline: **QUALITATIVE** — this batch ran only unlabelled frames "
            f"({summary['calls']} calls), so no scored number exists; the polygons "
            f"are reported as overlays + internal-consistency diagnostics. Restore "
            f"the ground-truthed SROI frames with `--sources validation occluded "
            f"shunba`.")
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
    add("## Method")
    sources_note = ", ".join(sorted({r["source"] for r in records}))
    add(f"- Sources in this batch: `{sources_note}` "
        f"({summary['gt_frames']} ground-truthed, "
        f"{summary['unlabelled_frames']} unlabelled); the manifest is the base "
        f"pipeline's, reused as-is.")
    if args.provider == "agy":
        add(f"- Frames delivered via `agy -p`: staged as the ONLY file of a fresh "
            f"per-call temp workspace, fetched with the built-in `view_file` — the "
            f"one sanctioned tool call. `view_file` resamples frames to "
            f"{vre.AGY_DELIVERED_W}x{vre.AGY_DELIVERED_H}; the prompt spoke that "
            f"space and the harness scaled every point, bbox AND polygon vertex "
            f"back before scoring. Fine detail is softer than in the "
            f"codex/claude runs — a delivery-path handicap.")
    else:
        add(f"- Frames delivered unchanged (inside the 1280 px no-resize ceiling).")
    add("- Prompt: `inventory_segmentation` — the full_detection inventory's NINE "
        "fields (picking_point and target_index included) PLUS a tenth, "
        "`polygon`: an ordered vertex outline of the fruit's VISIBLE surface "
        "(follows occluder edges, never extrapolates the hidden shape; 8-20 "
        "vertices typical, 32 max; extent agrees with the reported bbox).")
    add("- Diagnostics everywhere: polygon validity (out-of-frame / degenerate), "
        "vertex budget, polygon-vs-bbox extent IoU (internal consistency), "
        "picking-point distance to its own polygon outline.")
    if summary.get("scored_runs"):
        add("- Cross-check on ground-truthed frames: the nominated target's "
            "picking point scored against the recomputed GT point exactly like "
            "full_detection (error / dx / dy / PCK), plus the polygon extent's "
            "IoU against the upstream rough box and the GT point's distance to "
            "the target outline.")
    add("")
    add(f"> {NO_MASK_GT_NOTE}")
    add("")
    add(f"> {POLYGON_FIDELITY_NOTE}")
    add("")
    if summary.get("scored_runs"):
        add(f"> {ROUGH_BOX_NOTE}")
        add("")

    # Headline metrics
    add("## Polygon diagnostics (all parsed frames)")
    add("| metric | value |")
    add("| --- | --- |")
    add(f"| calls | {summary['calls']} |")
    add(f"| parsed (ok / no_pick_point) | {summary['parsed']} "
        f"({summary['ok']} / {summary['no_pick_point']}) |")
    add(f"| parse rate | {summary['parse_rate']} |")
    add(f"| schema-valid rate | {summary['schema_valid_rate']} |")
    add(f"| polygons reported / valid / out-of-frame / degenerate | "
        f"{summary['polygons_total']} / {summary['polygons_ok']} / "
        f"{summary['polygons_out_of_frame']} / {summary['polygons_degenerate']} |")
    add(f"| mean vertices per polygon | {summary['mean_vertex_count']} |")
    add(f"| polygon-vs-bbox extent IoU (mean) | {summary['polygon_bbox_iou_mean']} |")
    add(f"| picking-point to own outline, px (mean) | "
        f"{summary['pick_to_own_polygon_px_mean']} |")
    add("")
    add("## Nominated-target cross-check (ground-truthed frames)")
    if summary.get("scored_runs"):
        add("| metric | value |")
        add("| --- | --- |")
        add(f"| scored frames | {summary['scored_runs']} |")
        add(f"| picking-point error, px (median / mean) | "
            f"**{summary.get('median_error_px')}** / {summary.get('mean_error_px')} |")
        add(f"| PCK@5 / @10 / @20 | {summary.get('pck_5')} / "
            f"{summary.get('pck_10')} / {summary.get('pck_20')} |")
        add(f"| bbox IoU vs rough box (mean) | {summary.get('mean_bbox_iou_rough')} |")
        add(f"| polygon extent IoU vs rough box (mean) | "
            f"{summary.get('mean_polygon_extent_iou_rough')} |")
        add(f"| GT point inside target polygon | "
            f"{summary.get('polygon_contains_gt_rate')} of scored frames |")
        add(f"| GT point to target outline, px (mean) | "
            f"{summary.get('gt_to_target_polygon_px_mean')} |")
        add("")
        add("_The last two rows are diagnostics, not errors: the GT grasp point "
        "sits on the peduncle ABOVE the calyx, so it should land just outside "
        "(a few px from) the fruit-body outline, not inside it._")
    else:
        add("")
        add("_Not applicable: this batch contains no ground-truthed frames. "
        "Scored segmentation is not computable on these sources at all (no "
        "mask GT); the picking-point cross-check needs the SROI frames "
        "(`--sources validation occluded shunba`)._")
    add("")
    if summary["failures_by_status"]:
        add(f"failures by status: {summary['failures_by_status']}")
        add("")
    if summary.get("retried_calls"):
        add(f"retried calls (recovered): {summary['retried_calls']} "
            f"({summary['retries_recovered']} parsed after retry; "
            f"empty→low-effort or schema-invalid→same-effort, all recorded)")
        add("")

    # Per-image table
    add("## Per-image results")
    add("| frame | source | status | found | ok polygons | target | err px | "
        "extent IoU | tok |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in records:
        tok = r.get("total_tokens")
        target = r.get("target_index")
        target_s = "–" if target is None else (
            f"#{target}" if isinstance(target, int) and target >= 0 else str(target))
        add(f"| {r['sample_id']} | {r['source']} | {r['status']} | "
            f"{r.get('n_strawberries')} | {r.get('n_polygons_ok')} | {target_s} | "
            f"{r.get('error_px') if r.get('error_px') is not None else '–'} | "
            f"{r.get('polygon_extent_iou_rough') if r.get('polygon_extent_iou_rough') is not None else '–'} | "
            f"{tok} |")
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
    add("Per-run overlays in `overlays/`: polygon outline + light fill coloured "
        "by that fruit's CONTINUOUS redness ramp (green at 0% → amber at 50% → "
        "red at 100%); cyan polygon = the nominated pick target; magenta ring = "
        "a reported picking point; on ground-truthed frames the cyan crosshair "
        "+ rough ellipse box are the GT reference, drawn last. Out-of-frame / "
        "degenerate polygons are counted in the tables, not drawn. Contact "
        "sheets in `contact_sheets/` (chunks of 24, `__unlabelled` for the "
        "no-GT frames). Overlays and sheets are JPG (quality 90); the model "
        "inputs are untouched PNG.")
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
        f"(`vlm_eval/seg/`): `{info['fingerprint']}` — the base harness's "
        f"glob does not see this directory and vice versa.")
    add(f"- manifest snapshot + catalog snapshot in this directory; "
        f"records in `responses.jsonl` carry the full identity per call.")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")


def write_artifacts(run_dir: Path, args, manifest, control, records: list[dict],
                    started: str, elapsed: float, run_summary: dict,
                    rebuilt_with: str | None = None) -> None:
    samples = {s["sample_id"]: s for s in manifest["samples"]}

    # Re-derive the scored + diagnostic blocks from the stored inventories so
    # a rebuild after a scoring fix re-scores every old run. Everything is
    # offline-safe: GT rides inside each record.
    for record in records:
        sample = samples.get(record["sample_id"], {})
        record.update(diag_block(record["status"], record.get("inventory"),
                                 record["frame_w"], record["frame_h"]))
        scored_fields = scored_block(record["status"], record.get("inventory"),
                                     record.get("target_index"), sample,
                                     record["frame_w"], record["frame_h"])
        error_px = scored_fields["error_px"]
        record.update(scored_fields)
        record["error_px"] = error_px
        record["error_pct_width"] = None if error_px is None \
            else round(error_px / float(record["frame_w"]) * 100.0, 4)
        record["picking_point"] = ((record.get("inventory") or [])
                                   [record["target_index"]].get("picking_point")
                                   if isinstance(record.get("target_index"), int)
                                   and 0 <= record["target_index"]
                                   < len(record.get("inventory") or []) else None)
        record["pck_5"] = None if error_px is None else bool(error_px <= 5)
        record["pck_10"] = None if error_px is None else bool(error_px <= 10)
        record["pck_20"] = None if error_px is None else bool(error_px <= 20)
    records.sort(key=lambda r: r["sample_id"])

    # Render BEFORE persisting: _render_all updates each record's overlay
    # field, and those updates must land in the written artefacts.
    _render_all(run_dir, records)
    (run_dir / "responses.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records))
    summary = seg_summary(records)
    write_seg_csvs(run_dir, records, summary)
    write_report(run_dir, args, manifest, control, records, summary, started,
                 elapsed, run_summary, rebuilt_with)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def dry_run(args, manifest, model) -> None:
    samples = select_samples(args, manifest)
    print(f"dry run : {len(samples)} frame(s), 1 prompt ({STYLE_NAME}), "
          f"1 control call; provider={args.provider} model={model}")
    print(f"plan    : {len(samples) + 1} model call(s) if run for real")
    sample = samples[0]
    h, w = sample["image_shape_hw"]
    gt = (f"GT picking point {sample['gt_uv_rounded']} + rough box"
          if sample.get("has_gt") else "no ground truth (qualitative)")
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
    old = next((r.get("harness_fingerprint") for r in records
                if r.get("harness_fingerprint")), None)
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
    agent_cwd = VLM / "agent_cwd"
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

    n_gt = sum(1 for s in samples if s.get("has_gt"))
    print(f"run dir : {run_dir}")
    print(f"model   : {model}  provider={args.provider}  "
          f"effort={effort or 'config default'}")
    print(f"plan    : {len(samples)} frames ({n_gt} with ground truth, "
          f"{len(samples) - n_gt} unlabelled) x 1 prompt ({STYLE_NAME}) "
          f"= {len(samples)} calls + 1 control")

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
    summary = seg_summary(records)
    print(f"\ndone in {elapsed:.0f}s -> {run_dir}")
    print(f"report  : {run_dir / 'report.md'}")
    print(f"metrics : polygons {summary.get('polygons_ok')} ok "
          f"({summary.get('polygons_out_of_frame')} oof, "
          f"{summary.get('polygons_degenerate')} degenerate)"
          + (f"  | nominated-target median err {summary.get('median_error_px')}px "
             f"over {summary.get('scored_runs')} scored frame(s)"
             if summary.get("scored_runs") else "  | qualitative (no GT frames)"))


def _progress(done: int, total: int, record: dict) -> None:
    err = record.get("error_px")
    err_s = f"err={err:.1f}px" if isinstance(err, (int, float)) else "–"
    print(f"[{done}/{total}] {record['run_id']:<44} {record['status']:<13} "
          f"poly={record.get('n_polygons_ok') or 0:<3} {err_s:<11} "
          f"tok={record.get('total_tokens')}", flush=True)


if __name__ == "__main__":
    main()
