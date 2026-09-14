#!/usr/bin/env python3
"""StrawDI detection eval — run the VLM inventory pipeline over StrawDI_Db1.

Executes the ONE prompt of the StrawDI pipeline (``inventory_plain``, imported
verbatim from the base harness, parameterised to the frame size) on every
manifest sample, then scores the inventories as multi-instance detection
against the mask-derived ground-truth boxes.

The base harness (``vlm_eval/``) supplies everything model-facing: the
provider stack (claude/codex CLIs, tool surfaces off, images as base64 blocks),
the reply parser and the synthetic vision-delivery control. This script adds
only what is StrawDI-specific — detection scoring, GT-box overlays, and its own
report. The base-harness invariants carry over unchanged:

1. the synthetic vision-delivery control runs FIRST and invalidates the batch
   on failure;
2. all tool surfaces stay disabled (``--tools "" --safe-mode
   --strict-mcp-config --no-session-persistence``, empty agent_cwd; the event
   stream is scanned for tool_use and any attempt fails acceptance);
3. no annotated image is ever an input (GT masks are only read after the call);
4. inputs stay inside the no-resize ceiling (the manifest builder enforces it);
5. every record and report states harness, base harness, model and effort.

Usage (repo root):
    python3 strawdi_eval/run_detection_eval.py --jobs 3 --tag full
    python3 strawdi_eval/verify_strawdi_run.py strawdi_eval/runs/<dir>
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from vlm_eval.lib import imaging, parse  # noqa: E402
from vlm_eval import prompts  # noqa: E402
import vlm_eval.run_vlm_eval as vre  # noqa: E402
from strawdi_eval.lib import render, scoring  # noqa: E402

HARNESS_NAME = "strawdi_eval"
HARNESS_VERSION = "0.1.0"

# The pipeline's single prompt and its answer schema, verbatim from the base
# harness — the output standard this pipeline follows.
STYLE = prompts.STYLES["inventory_plain"]
SCHEMA_PATH = vre.INVENTORY_SCHEMA_PATH

DEFAULT_RUNS_DIR = HERE / "runs"
DEFAULT_JOBS = 3
CONTACT_SHEET_CHUNK = 24          # per sheet, 3 columns

# Both caveats are printed in every report so a headline number can never be
# quoted without the semantics that produced it.
BBOX_SEMANTICS_NOTE = (
    "The prompt asks for the box of the WHOLE fruit (including parts hidden "
    "behind occluders), while StrawDI ground truth annotates the VISIBLE mask "
    "surface only. Predictions on partly occluded fruit are therefore expected "
    "to EXCEED the GT box and lose IoU — the bias is one-directional and hits "
    "heavily occluded fruit hardest. IoU@0.5 stays the primary metric (it is "
    "the standard, prompt-faithful read); the symmetric centre-containment "
    "metric is reported alongside as a semantics-robust presence check, and "
    "the occlusion-split diagnostic quantifies the gap."
)
CONFIDENCE_TIE_NOTE = (
    "AP ranks predictions by the model's confidence_pct. Inventory confidences "
    "tend to clump near 100, under which the ranking is arbitrary and AP "
    "collapses toward the single F1 operating point. F1@IoU-0.5 is the "
    "headline; AP is secondary and its tie-breaks (confidence, then inventory "
    "order) are deterministic."
)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def harness_info() -> dict:
    """Fingerprint of THIS pipeline's code (not the base harness's)."""
    digest = hashlib.sha256()
    files = sorted(HERE.glob("*.py")) + sorted((HERE / "lib").glob("*.py"))
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
    """``<ts>-<model>-<effort>-<cli>-strawdi_eval[-<tag>]`` (base convention).

    Reimplemented here because the base function hardcodes its own harness
    name into the tail.
    """
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
    ap.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    ap.add_argument("--out", type=Path, default=DEFAULT_RUNS_DIR,
                    help="runs root; a timestamped subdirectory is created inside it")
    ap.add_argument("--tag", default="", help="suffix for the run directory name")
    ap.add_argument("--limit", type=int, default=None,
                    help="max samples (manifest order)")
    ap.add_argument("--model", default=None, help="model slug")
    ap.add_argument("--reasoning-effort", default=None,
                    help="reasoning effort override; default keeps the provider's")
    ap.add_argument("--provider", choices=("claude", "codex"), default="claude")
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

def build_prompt(sample: dict) -> str:
    h, w = sample["image_shape_hw"]
    prompt = STYLE.build_prompt(frame_w=w, frame_h=h, query_origin=(0, 0),
                                multi_fruit=True)
    return f"{prompt}\n\n{vre.TOOL_NOTICE}"


def detection_block(record: dict) -> dict:
    """Score a parsed record's inventory against its GT boxes.

    Failure statuses (empty/refused/parse/exec/schema) carry no detection
    numbers at all — never zeros. A parsed inventory reporting ZERO fruit is a
    valid answer and scores as all-missed.
    """
    if record["status"] not in (parse.OK, parse.NO_PICK_POINT):
        return scoring.empty_detection()
    return scoring.score_image(record["inventory"], record["gt_boxes"],
                               record["gt_areas"], record["frame_w"],
                               record["frame_h"])


def run_one(args, manifest, sample, run_ctx) -> dict:
    """Execute (and if needed retry) one model call, then score it."""
    image = HERE / sample["images"]["raw"]
    h, w = sample["image_shape_hw"]
    prompt = build_prompt(sample)
    schema = json.loads(SCHEMA_PATH.read_text())

    model = run_ctx["model"]
    effort = run_ctx["effort"]
    stem = f"{STYLE.name}__{sample['sample_id']}"
    attempts: list[dict] = []
    fallback_used = False

    message_path = run_ctx["tmp"] / f"{stem}.last.txt"
    attempt = vre.call_model(args, manifest, image, prompt, message_path,
                             None, model, effort, run_ctx["agent_cwd"],
                             args.timeout)
    attempts.append(attempt)

    scored = vre.classify_inventory(attempt["last_message"], schema, w, h)
    if scored["status"] == parse.EMPTY_RESPONSE and effort != "low":
        # Same recorded fallback as the base harness: reasoning can exhaust the
        # output budget, leaving no message at all.
        fallback_used = True
        message_path = run_ctx["tmp"] / f"{stem}.retry.last.txt"
        retry = vre.call_model(args, manifest, image, prompt, message_path,
                               None, model, "low", run_ctx["agent_cwd"],
                               args.timeout)
        retry["fallback_of_attempt"] = 1
        attempts.append(retry)
        scored = vre.classify_inventory(retry["last_message"], schema, w, h)

    totals = vre.merge_usage(attempts)
    parsed_ok = scored["status"] in (parse.OK, parse.NO_PICK_POINT)
    inventory = scored["strawberries"] if parsed_ok else None

    record = {
        "run_id": stem,
        "style": STYLE.name,
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
        "used_output_schema": False,
        "output_shape": STYLE.output_shape,
        "image": sample["images"]["raw"],
        "image_kind": "raw",
        "frame_w": w,
        "frame_h": h,
        "prompt": prompt,
        "inventory": inventory,
        "target_index": scored.get("target_index"),
        "target_valid": scored.get("target_valid"),
        "n_strawberries": scored["n_strawberries"] if parsed_ok else None,
        "n_points_out_of_frame": scored.get("n_out_of_frame") if parsed_ok else None,
        **vre._inventory_summary(inventory),
        # Ground truth rides inside the record so the run is self-contained
        # and --rebuild-report can re-score without the dataset.
        "gt_boxes": sample["gt_boxes"],
        "gt_areas": sample["gt_areas"],
        **detection_block({
            "status": scored["status"], "inventory": inventory,
            "gt_boxes": sample["gt_boxes"], "gt_areas": sample["gt_areas"],
            "frame_w": w, "frame_h": h,
        }),
        **totals,
        "attempts": len(attempts),
        "fallback_used": fallback_used,
        "fallback_reason": ("empty response at configured reasoning effort"
                            if fallback_used else None),
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

    title = f"{STYLE.name} | {sample['sample_id']} | {sample['n_gt']}gt | {scored['status']}"
    rel = Path("overlays") / f"{stem}.png"
    try:
        raw = imaging.load_rgb(image)
        overlay = render.draw_detection_overlay(
            raw, inventory, scored.get("target_index"), sample["gt_boxes"],
            record["matches_50"], record["fn_gt_indices_50"], title)
        imaging.save_rgb(overlay, run_ctx["run_dir"] / rel)
        record["overlay"] = str(rel)
    except Exception as exc:  # a drawing bug must not throw away a paid-for answer
        record["overlay"] = None
        record["render_error"] = f"{type(exc).__name__}: {exc}"
    return record


def failure_record(sample: dict, exc: BaseException, run_ctx) -> dict:
    """A run that blew up in the harness itself, recorded rather than raised."""
    return {
        "run_id": f"{STYLE.name}__{sample['sample_id']}",
        "style": STYLE.name, "sample_id": sample["sample_id"],
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
        "used_output_schema": False, "output_shape": STYLE.output_shape,
        "image": sample["images"]["raw"], "image_kind": "raw",
        "frame_w": sample["image_shape_hw"][1],
        "frame_h": sample["image_shape_hw"][0],
        "prompt": None, "inventory": None, "target_index": None,
        "target_valid": None, "n_strawberries": None,
        "n_points_out_of_frame": None,
        **vre._inventory_summary(None),
        "gt_boxes": sample["gt_boxes"], "gt_areas": sample["gt_areas"],
        **scoring.empty_detection(),
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


def detection_summary(records: list[dict]) -> dict:
    """Every batch-level number the report prints, derived from records only."""
    parsed = [r for r in records
              if r["status"] in (parse.OK, parse.NO_PICK_POINT)]
    failed = [r for r in records if r["status"] in parse.FAILURE_STATUSES]

    summary = {
        "calls": len(records),
        "parsed": len(parsed),
        "parse_rate": round(len(parsed) / len(records), 4) if records else None,
        "schema_valid_rate": _rate(records, lambda r: r.get("schema_valid") is True),
        "ok": sum(1 for r in parsed if r["status"] == parse.OK),
        "no_pick_point": sum(1 for r in parsed if r["status"] == parse.NO_PICK_POINT),
        "failures_by_status": {
            status: sum(1 for r in failed if r["status"] == status)
            for status in sorted({r["status"] for r in failed})
        },
    }

    for threshold in scoring.IOU_THRESHOLDS:
        key = int(round(threshold * 100))
        tp = sum(r.get(f"tp_{key}") or 0 for r in parsed)
        fp = sum(r.get(f"fp_{key}") or 0 for r in parsed)
        fn = sum(r.get(f"fn_{key}") or 0 for r in parsed)
        p, recall, f1 = scoring.precision_recall_f1(tp, fp, fn)
        summary[f"tp_{key}"], summary[f"fp_{key}"], summary[f"fn_{key}"] = tp, fp, fn
        summary[f"precision_{key}"] = p
        summary[f"recall_{key}"] = recall
        summary[f"f1_{key}"] = f1

    tp = sum(r.get("tp_center") or 0 for r in parsed)
    fp = sum(r.get("fp_center") or 0 for r in parsed)
    fn = sum(r.get("fn_center") or 0 for r in parsed)
    p, recall, f1 = scoring.precision_recall_f1(tp, fp, fn)
    summary.update({"tp_center": tp, "fp_center": fp, "fn_center": fn,
                    "precision_center": p, "recall_center": recall,
                    "f1_center": f1})

    matched_ious = [m["iou"] for r in parsed for m in (r.get("matches_50") or [])]
    summary["matched_pairs_50"] = len(matched_ious)
    summary["mean_matched_iou_50"] = _mean(matched_ious)
    summary["median_matched_iou_50"] = (round(float(np.median(matched_ious)), 4)
                                        if matched_ious else None)

    count_errors = [r["count_error"] for r in parsed if r.get("count_error") is not None]
    summary["count_error_mean"] = _mean(count_errors)
    summary["count_error_mae"] = _mean([abs(e) for e in count_errors])
    summary["gt_total"] = sum(r.get("n_gt") or 0 for r in parsed)
    summary["pred_total"] = sum(r.get("n_pred") or 0 for r in parsed)

    per_image = [{
        "sample_id": r["sample_id"],
        "preds": scoring.prepare_predictions(r.get("inventory"),
                                             r["frame_w"], r["frame_h"]),
        "gt_boxes": r["gt_boxes"],
    } for r in parsed]
    aps = scoring.ap_metrics(per_image)
    summary["ap_50"] = aps["ap_50"]
    summary["map_50_95"] = aps["map_50_95"]
    summary["ap_per_threshold"] = aps["per_threshold"]

    summary["size_strata"] = scoring.size_stratified(parsed)
    summary["occlusion_split"] = scoring.occlusion_split(parsed)
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

CSV_DROP = ("prompt", "response_text", "attempts_detail")


def write_detection_csvs(run_dir: Path, records: list[dict], summary: dict) -> None:
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in CSV_DROP}
                       for r in records])
    for column in df.columns:
        df[column] = df[column].map(
            lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)
    df.to_csv(run_dir / "metrics.csv", index=False)

    flat = {k: v for k, v in summary.items()
            if not isinstance(v, (list, dict))}
    flat["ap_per_threshold"] = json.dumps(summary["ap_per_threshold"])
    pd.DataFrame([flat]).to_csv(run_dir / "detection_summary.csv", index=False)
    pd.DataFrame(summary["size_strata"]).to_csv(run_dir / "size_strata.csv",
                                                index=False)
    pd.DataFrame([{"iou_threshold": f"{t:.2f}", "ap": ap}
                  for t, ap in summary["ap_per_threshold"].items()]
                 ).to_csv(run_dir / "ap.csv", index=False)


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
                 f"{record.get('n_gt')}gt | {record['status']}")
        rel = Path("overlays") / f"{record['run_id']}.png"
        try:
            raw = imaging.load_rgb(HERE / record["image"])
            overlay = render.draw_detection_overlay(
                raw, record.get("inventory"), record.get("target_index"),
                record.get("gt_boxes"), record.get("matches_50"),
                record.get("fn_gt_indices_50"), title)
            imaging.save_rgb(overlay, run_dir / rel)
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
        imaging.save_rgb(sheet, sheets / f"sheet_{start // CONTACT_SHEET_CHUNK + 1:02d}.png")

    # Miss gallery: the images where the most GT area went unfound.
    misses = sorted((r for r in records if r.get("fn_gt_areas_50")),
                    key=lambda r: (sum(r["fn_gt_areas_50"]),
                                   len(r["fn_gt_areas_50"])), reverse=True)[:12]
    if misses:
        images = [imaging.load_rgb(run_dir / r["overlay"]) for r in misses
                  if r.get("overlay")]
        if images:
            imaging.save_rgb(imaging.contact_sheet(
                images, [f"{r['sample_id']} miss{len(r['fn_gt_areas_50'])}"
                         for r in misses if r.get("overlay")], cols=3),
                sheets / "miss_gallery.png")


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
    add(f"# StrawDI detection eval — {verdict}")
    add("")
    add(f"Headline: **F1@IoU0.5 = {headline}** "
        f"(P {summary.get('precision_50')}, R {summary.get('recall_50')}) "
        f"on {summary['calls']} frames, {summary['gt_total']} GT instances.")
    add("")
    add(f"- **Harness:** {info['name']} v{info['version']} "
        f"(fingerprint `{info['fingerprint']}`)")
    add(f"- **Base harness:** {base['name']} v{base['version']} "
        f"(fingerprint `{base['fingerprint']}`) — provider stack, prompt, "
        f"parser and control")
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
    add(f"- Frames delivered unchanged at "
        f"{manifest['samples'][0]['image_shape_hw'][1]}x"
        f"{manifest['samples'][0]['image_shape_hw'][0]} "
        f"(inside the 1280 px no-resize ceiling; GT maps 1:1).")
    add("- Prompt: `inventory_plain` verbatim from the base harness — the "
        "nine-field unbiased inventory + `target_index`; every fruit whatever "
        "its colour. Nomination/picking-point fields are recorded but never "
        "scored (StrawDI has no picking-point GT).")
    add("- Matching: greedy, confidence-descending, one GT per prediction, "
        "recomputed per threshold at IoU 0.25 / 0.5 / 0.75; centre-containment "
        "reported alongside.")
    add(f"- GT boxes: {manifest['coordinate_convention']['bbox']}.")
    add("")
    add(f"> {BBOX_SEMANTICS_NOTE}")
    add("")
    add(f"> {CONFIDENCE_TIE_NOTE}")
    add("")

    # Headline metrics
    add("## Headline metrics")
    add("| metric | value |")
    add("| --- | --- |")
    add(f"| calls | {summary['calls']} |")
    add(f"| parsed (ok / no_pick_point) | {summary['parsed']} "
        f"({summary['ok']} / {summary['no_pick_point']}) |")
    add(f"| parse rate | {summary['parse_rate']} |")
    add(f"| schema-valid rate | {summary['schema_valid_rate']} |")
    for label, extra in (("IoU 0.25", "25"), ("IoU 0.50", "50"), ("IoU 0.75", "75")):
        add(f"| P / R / F1 @{label} | {summary.get(f'precision_{extra}')} / "
            f"{summary.get(f'recall_{extra}')} / **{summary.get(f'f1_{extra}')}** "
            f"(TP {summary[f'tp_{extra}']} FP {summary[f'fp_{extra}']} "
            f"FN {summary[f'fn_{extra}']}) |")
    add(f"| P / R / F1 @centre | {summary['precision_center']} / "
        f"{summary['recall_center']} / {summary['f1_center']} "
        f"(TP {summary['tp_center']} FP {summary['fp_center']} "
        f"FN {summary['fn_center']}) |")
    add(f"| mean / median matched IoU@0.5 | {summary['mean_matched_iou_50']} / "
        f"{summary['median_matched_iou_50']} over {summary['matched_pairs_50']} pairs |")
    add(f"| AP@0.50 / mAP@[.50:.95] | {summary['ap_50']} / {summary['map_50_95']} |")
    add(f"| count error (pred − gt) | mean {summary['count_error_mean']}, "
        f"MAE {summary['count_error_mae']} |")
    add(f"| totals | {summary['gt_total']} GT, {summary['pred_total']} predicted |")
    if summary["failures_by_status"]:
        add(f"| failures by status | {summary['failures_by_status']} |")
    add("")

    # Size strata
    add("## Recall by GT size (COCO bands)")
    add("| stratum | area px² | n GT | matched | recall | mean IoU@0.5 |")
    add("| --- | --- | --- | --- | --- | --- |")
    for row in summary["size_strata"]:
        add(f"| {row['stratum']} | {row['area_band_px']} | {row['n_gt']} | "
            f"{row['n_matched']} | {row['recall']} | {row['mean_matched_iou_50']} |")
    add("")

    # Occlusion diagnostic
    occ = summary["occlusion_split"]
    add("## Occlusion diagnostic (bbox-semantics gap)")
    add("| quantity | value |")
    add("| --- | --- |")
    add(f"| matched IoU@0.5, pred occlusion < 25% | {occ['matched_iou_occl_lt25_mean']} "
        f"(n={occ['n_matched_occl_lt25']}) |")
    add(f"| matched IoU@0.5, pred occlusion ≥ 25% | {occ['matched_iou_occl_ge25_mean']} "
        f"(n={occ['n_matched_occl_ge25']}) |")
    add(f"| mean reported occlusion, matched vs unmatched preds | "
        f"{occ['mean_reported_occlusion_matched']} vs "
        f"{occ['mean_reported_occlusion_unmatched']} |")
    add("")

    # Per-image table
    add("## Per-image results")
    add("| sample | status | gt | pred | TP | FP | FN | IoU@0.5 | Δcount | tok |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in records:
        tok = r.get("total_tokens")
        add(f"| {r['sample_id']} | {r['status']} | {r.get('n_gt')} | "
            f"{r.get('n_pred')} | {r.get('tp_50')} | {r.get('fp_50')} | "
            f"{r.get('fn_50')} | {r.get('mean_matched_iou_50')} | "
            f"{r.get('count_error')} | {tok} |")
    add("")

    # Error analysis
    parsed = [r for r in records if r["status"] in (parse.OK, parse.NO_PICK_POINT)]
    worst_miss = sorted(
        (r for r in parsed if r.get("fn_gt_areas_50")),
        key=lambda r: (sum(r["fn_gt_areas_50"]), len(r["fn_gt_areas_50"])),
        reverse=True)[:10]
    high_conf_fp = []
    for r in parsed:
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
    add("Per-run overlays in `overlays/` (green = GT box, red inner = missed "
        "GT, redness-ramp = predictions with white TP corner ticks, cyan = "
        "nominated target); contact sheets in `contact_sheets/` "
        "(chunks of 24) and `contact_sheets/miss_gallery.png`.")
    add("")

    # Exact prompt
    add("## Exact prompt (verbatim, first frame)")
    add("")
    add("```text")
    add(records[0]["prompt"] if records and records[0].get("prompt")
        else build_prompt(manifest["samples"][0]))
    add("```")
    add("")

    add("## Provenance")
    add(f"- run directory name encodes model/effort/CLI/harness: "
        f"`{run_dir.name}`")
    add(f"- base-harness fingerprint covers the provider stack, prompt text, "
        f"parser and control the numbers depend on: `{base['fingerprint']}`")
    add(f"- manifest snapshot + catalog snapshot in this directory; "
        f"records in `responses.jsonl` carry the full identity per call.")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")


def write_artifacts(run_dir: Path, args, manifest, control, records: list[dict],
                    started: str, elapsed: float, run_summary: dict,
                    rebuilt_with: str | None = None) -> None:
    # Re-derive the detection fields from the stored inventories so a rebuild
    # after a scoring fix re-scores every old run (the records carry their GT).
    for record in records:
        if record.get("inventory") is not None or record["status"] in (
                parse.OK, parse.NO_PICK_POINT):
            record.update(detection_block(record))
        else:
            record.update(scoring.empty_detection())
    records.sort(key=lambda r: r["sample_id"])

    (run_dir / "responses.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records))
    summary = detection_summary(records)
    write_detection_csvs(run_dir, records, summary)
    _render_all(run_dir, records)
    write_report(run_dir, args, manifest, control, records, summary, started,
                 elapsed, run_summary, rebuilt_with)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def dry_run(args, manifest, model) -> None:
    samples = manifest["samples"][: args.limit] if args.limit else manifest["samples"]
    print(f"dry run : {len(samples)} frame(s), 1 prompt ({STYLE.name}), "
          f"1 control call; provider={args.provider} model={model}")
    print(f"plan    : {len(samples) + 1} model call(s) if run for real")
    sample = samples[0]
    h, w = sample["image_shape_hw"]
    print(f"first   : {sample['sample_id']} ({w}x{h}, {sample['n_gt']} GT)")
    print("\n--- prompt (verbatim) " + "-" * 40)
    print(build_prompt(sample))
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

    if args.provider == "claude":
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

    samples = manifest["samples"][: args.limit] if args.limit else manifest["samples"]
    if not samples:
        raise SystemExit("no samples selected")

    if args.dry_run:
        dry_run(args, manifest, model)
        return

    info = harness_info()
    base = vre.harness_info()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    cli_name = "claude" if args.provider == "claude" else "codex"
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
    print(f"plan    : {len(samples)} frames x 1 prompt ({STYLE.name}) "
          f"= {len(samples)} calls + 1 control")

    # The control image path must be resolved against THIS harness's directory
    # (the base run_control defaults to its own).
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
    summary = detection_summary(records)
    print(f"\ndone in {elapsed:.0f}s -> {run_dir}")
    print(f"report  : {run_dir / 'report.md'}")
    print(f"metrics : F1@0.5 {summary.get('f1_50')}  "
          f"P {summary.get('precision_50')}  R {summary.get('recall_50')}  "
          f"AP@50 {summary.get('ap_50')}  mAP {summary.get('map_50_95')}")


def _progress(done: int, total: int, record: dict) -> None:
    tp = record.get("tp_50")
    fp = record.get("fp_50")
    fn = record.get("fn_50")
    det = (f"TP{tp} FP{fp} FN{fn}" if tp is not None else "–")
    print(f"[{done}/{total}] {record['run_id']:<40} {record['status']:<13} "
          f"{det:<12} tok={record.get('total_tokens')}", flush=True)


if __name__ == "__main__":
    main()
