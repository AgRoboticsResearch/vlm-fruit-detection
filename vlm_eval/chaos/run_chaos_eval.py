#!/usr/bin/env python3
"""Chaos strawberry detection — detect + segment on curated chaotic scenes.

Executes the StrawDI segmentation inventory (``inventory_segmentation``:
the NINE-field standard — the eight detection fields plus ``polygon``, no
``picking_point``, no ``target_index``; see
``strawdi_eval/pipeline/strawdi_segmentation.md``) on the CURATED chaos
scenes of ``vlm_eval/chaos/manifest.json`` (dense fruit, heavy clutter,
deep occlusion; currently the shunba sb04 scene and hand-held photos
normalised to fit 1280x720, EXIF-uprighted). "Detect" = the per-fruit
``bbox``; "segment" = the visible-surface ``polygon``; both come from the
one unbiased census. The prompt and schema are FROZEN byte-identical copies
of the StrawDI pipeline's (``vlm_eval/chaos/lib/prompt.py``, diffed at
creation), so chaos answers stay comparable with the scored StrawDI
segmentation numbers; a reply volunteering a picking point or nomination is
rejected wholesale.

No ground truth exists on any chaos scene, so nothing is scored: the output
is overlays, the full per-fruit inventory, and internal-consistency
diagnostics (polygon validity, vertex budget, extent-vs-own-bbox IoU). The
synthetic vision-delivery control still gates the batch, all tool surfaces
stay off, and every record states the full provenance chain:

* ``base_harness`` = ``vlm_eval``  — provider stack, parser, control;
* ``seg_harness``  = ``vlm_seg``   — the polygon scorer this pipeline
  imports unchanged (``vlm_eval/seg/lib/seg_scoring.py``);
* ``harness``      = ``vlm_chaos`` — this pipeline's own code
  (``vlm_eval/chaos/``, prompt + schema + renderer included), outside both
  fingerprints' globs.

The base-harness invariants carry over unchanged (control first; tools off;
no annotated input; inputs inside the no-resize ceiling; full provenance).

Usage (repo root):
    python3 vlm_eval/chaos/build_chaos_manifest.py
    python3 vlm_eval/chaos/run_chaos_eval.py --dry-run
    python3 vlm_eval/chaos/run_chaos_eval.py --tag full

No acceptance gate: this pipeline deliberately runs WITHOUT a verifier step
(decision 2026-09-17) — it is a fast qualitative iteration loop.
``verify_chaos_run.py`` remains in the tree as an optional integrity
checker, not part of the workflow.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import re
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent          # vlm_eval/chaos
VLM = HERE.parent                                # vlm_eval
REPO = VLM.parent                                # repo root
sys.path.insert(0, str(REPO))

from vlm_eval.lib import imaging, parse  # noqa: E402
import vlm_eval.run_vlm_eval as vre  # noqa: E402
from vlm_eval.seg.lib import seg_scoring  # noqa: E402
import vlm_eval.seg.run_segmentation_eval as seg_runner  # noqa: E402  (harness_info)
from vlm_eval.chaos.lib import prompt as chaos_prompt  # noqa: E402
from vlm_eval.chaos.lib import render  # noqa: E402

HARNESS_NAME = "vlm_chaos"
HARNESS_VERSION = "0.2.0"

# The pipeline's single prompt and its answer schema: FROZEN byte-identical
# copies of the StrawDI seg pipeline's nine-field standard (see
# vlm_eval/chaos/lib/prompt.py for the copy rationale and the diff rule).
STYLE_NAME = chaos_prompt.STYLE_NAME
SCHEMA_PATH = HERE / "schema" / "inventory_segmentation_schema.json"

DEFAULT_RUNS_DIR = HERE / "runs"
DEFAULT_JOBS = 2

# Printed in every report: no chaos scene has any ground truth.
NO_GT_NOTE = (
    "Chaos scenes are curated photographs with NO ground truth of any kind "
    "(no picking point, no fruit box, no mask), so no error, IoU, PCK or "
    "segmentation score exists in this pipeline — overlays, the per-fruit "
    "inventory and internal-consistency diagnostics only. Scored detection "
    "and segmentation live in the StrawDI pipelines; the picking-point "
    "cross-check lives in full_detection experiment builds."
)


# ---------------------------------------------------------------------------
# Identity (three links: this pipeline -> the seg lib it imports -> the base)
# ---------------------------------------------------------------------------

def harness_info() -> dict:
    """Fingerprint of THIS pipeline's code only (``vlm_eval/chaos/``).

    Covers the runner, the frozen StrawDI prompt under ``lib/`` and the
    schema — the exact text and validation this pipeline sends and enforces.
    """
    import hashlib
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
    """``<ts>-<model>-<effort>-<cli>-vlm_chaos[-<tag>]`` (base convention)."""
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
    ap.add_argument("--manifest", type=Path, default=HERE / "manifest.json",
                    help="the chaos manifest (curated frames + control)")
    ap.add_argument("--out", type=Path, default=DEFAULT_RUNS_DIR,
                    help="runs root; a timestamped subdirectory is created inside it")
    ap.add_argument("--tag", default="", help="suffix for the run directory name")
    ap.add_argument("--limit", type=int, default=None,
                    help="max samples (manifest order)")
    ap.add_argument("--sample-id", default=None,
                    help="run a single chaos scene by id (e.g. IMG_7665)")
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
    samples = manifest["samples"]
    if args.sample_id:
        samples = [s for s in samples if s["sample_id"] == args.sample_id]
        if not samples:
            raise SystemExit(f"no chaos sample with sample_id {args.sample_id!r}")
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit("no samples selected")
    return samples


# ---------------------------------------------------------------------------
# One call
# ---------------------------------------------------------------------------

def build_prompt(args, sample: dict) -> str:
    h, w = sample["image_shape_hw"]
    pw, ph = vre.prompt_frame_size(args, w, h)
    prompt = chaos_prompt.build_inventory_segmentation(frame_w=pw, frame_h=ph)
    if args.provider == "agy":
        return prompt
    return f"{prompt}\n\n{vre.TOOL_NOTICE}"


# The per-fruit attributes kept from the inventory besides bbox/polygon.
SEG_ATTRS = ("redness_pct", "occlusion_pct", "calyx_visible",
             "peduncle_visible", "graspable", "confidence_pct", "description")


def classify_segmentation(last_message: str, schema: dict) -> dict:
    """Parse the nine-field StrawDI-standard inventory (no picking fields).

    Mirrors the StrawDI seg pipeline's classifier: a bare top-level list is
    accepted and wrapped; there is no nomination and no picking point, so
    ``no_pick_point`` cannot occur — an answer either parses (``ok``) or is
    an explicit failure.
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
        berry.update({key: entry.get(key) for key in ("polygon", *SEG_ATTRS)})
        berries.append(berry)

    result["strawberries"] = berries
    result["n_strawberries"] = len(berries)
    result["status"] = parse.OK
    return result


def run_one(args, manifest, sample, run_ctx) -> dict:
    """Execute (and if needed retry) one model call, then derive diagnostics."""
    image = VLM / sample["images"]["raw"]
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

    # Same retry policy as the seg pipelines: one low-effort retry for an
    # empty reply, one same-effort resample for a schema-invalid reply
    # (dense chaos scenes produce the longest inventories — the documented
    # extra-note-field decoration lands here and resampling recovers it).
    scored = classify_segmentation(attempt["last_message"], schema)
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
        scored = classify_segmentation(retry["last_message"], schema)

    totals = vre.merge_usage(attempts)
    parsed_ok = scored["status"] == parse.OK
    inventory = scored["strawberries"] if parsed_ok else None
    if inventory and args.provider == "agy":
        # The model answered in the delivered 800x600 frame; scale every bbox
        # and polygon vertex back to the original coordinate space.
        def scale_point(pt):
            return [int(round(v)) for v in vre.agy_scale_point(pt, w, h)]
        inventory = [dict(item,
                          bbox=vre.agy_scale_box(item["bbox"], w, h)
                          if item.get("bbox") else None,
                          polygon=[scale_point(pt) for pt in item["polygon"]]
                          if item.get("polygon") else None)
                     for item in inventory]

    record = {
        "run_id": stem,
        "style": STYLE_NAME,
        "sample_id": sample["sample_id"],
        "source": sample["source"],
        "session": sample["session"],
        "episode": sample["episode"],
        "scene_note": sample.get("scene_note"),
        "has_gt": False,
        "task": sample.get("task", "full_inventory"),
        "status": scored["status"],
        "json_method": scored["json_method"],
        "schema_valid": scored["schema_valid"],
        "schema_error": scored["schema_error"],
        "model": model,
        "harness": HARNESS_NAME,
        "harness_version": HARNESS_VERSION,
        "harness_fingerprint": run_ctx["harness"]["fingerprint"],
        "seg_harness": run_ctx["seg_harness"]["name"],
        "seg_harness_version": run_ctx["seg_harness"]["version"],
        "seg_harness_fingerprint": run_ctx["seg_harness"]["fingerprint"],
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
        **(seg_scoring.polygon_diag_block(inventory, w, h) if parsed_ok
           else seg_scoring.empty_diag()),
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

    title = f"{STYLE_NAME} | {sample['sample_id']} | chaos | {scored['status']}"
    rel = Path("overlays") / f"{stem}.jpg"
    try:
        raw = imaging.load_rgb(image)
        overlay = render.draw_segmentation_overlay(raw, inventory, title)
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
        "source": sample["source"], "session": sample["session"],
        "episode": sample["episode"], "scene_note": sample.get("scene_note"),
        "has_gt": False, "task": sample.get("task", "full_inventory"),
        "status": parse.EXEC_ERROR, "json_method": None,
        "schema_valid": None, "schema_error": None,
        "model": run_ctx["model"],
        "harness": HARNESS_NAME, "harness_version": HARNESS_VERSION,
        "harness_fingerprint": run_ctx["harness"]["fingerprint"],
        "seg_harness": run_ctx["seg_harness"]["name"],
        "seg_harness_version": run_ctx["seg_harness"]["version"],
        "seg_harness_fingerprint": run_ctx["seg_harness"]["fingerprint"],
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
# Batch aggregation + artefacts
# ---------------------------------------------------------------------------

def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(float(np.mean(values)), 4) if values else None


def _rate(records: list[dict], predicate) -> float | None:
    if not records:
        return None
    return round(sum(1 for r in records if predicate(r)) / len(records), 4)


def chaos_summary(records: list[dict]) -> dict:
    parsed = [r for r in records if r["status"] == parse.OK]
    failed = [r for r in records if r["status"] in parse.FAILURE_STATUSES]
    summary = {
        "calls": len(records),
        "parsed": len(parsed),
        "parse_rate": round(len(parsed) / len(records), 4) if records else None,
        "schema_valid_rate": _rate(records, lambda r: r.get("schema_valid") is True),
        "ok": len(parsed),
        "failures_by_status": {
            status: sum(1 for r in failed if r["status"] == status)
            for status in sorted({r["status"] for r in failed})
        },
        "retried_calls": sum(1 for r in records if r.get("fallback_used")),
        "retries_recovered": sum(
            1 for r in records if r.get("fallback_used")
            and r["status"] == parse.OK),
        "fruit_total": sum(r.get("n_inventory") or 0 for r in parsed),
        "polygons_ok": sum(r.get("n_polygons_ok") or 0 for r in parsed),
        "polygons_out_of_frame": sum(
            r.get("n_polygons_out_of_frame") or 0 for r in parsed),
        "polygons_degenerate": sum(
            r.get("n_polygons_degenerate") or 0 for r in parsed),
        "mean_vertex_count": _mean([r.get("mean_vertex_count") for r in parsed]),
        "polygon_bbox_iou_mean": _mean(
            [r.get("polygon_bbox_iou_mean") for r in parsed]),
        "total_input_tokens": sum(r.get("input_tokens") or 0 for r in records),
        "total_output_tokens": sum(r.get("output_tokens") or 0 for r in records),
        "total_tokens": sum(r.get("total_tokens") or 0 for r in records),
        "wall_s_total": round(sum(r.get("wall_s") or 0 for r in records), 1),
        "cost_usd_total": round(sum(r.get("cost_usd") or 0 for r in records), 4) \
            if any(r.get("cost_usd") is not None for r in records) else None,
    }
    return summary


CSV_DROP = ("prompt", "response_text", "attempts_detail", "inventory")


def write_csvs(run_dir: Path, records: list[dict], summary: dict) -> None:
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in CSV_DROP}
                       for r in records])
    for column in df.columns:
        df[column] = df[column].map(
            lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)
    df.to_csv(run_dir / "metrics.csv", index=False)
    flat = {k: v for k, v in summary.items() if not isinstance(v, (list, dict))}
    pd.DataFrame([flat]).to_csv(run_dir / "chaos_summary.csv", index=False)


def _render_all(run_dir: Path, records: list[dict]) -> None:
    """(Re)render every overlay + one contact sheet from the records."""
    overlays = run_dir / "overlays"
    shutil.rmtree(overlays, ignore_errors=True)
    overlays.mkdir(exist_ok=True)
    sheets = run_dir / "contact_sheets"
    shutil.rmtree(sheets, ignore_errors=True)
    sheets.mkdir(exist_ok=True)

    rendered: list[tuple[dict, Path]] = []
    for record in records:
        title = (f"{record['style']} | {record['sample_id']} | chaos | "
                 f"{record['status']}")
        rel = Path("overlays") / f"{record['run_id']}.jpg"
        try:
            raw = imaging.load_rgb(VLM / record["image"])
            overlay = render.draw_segmentation_overlay(
                raw, record.get("inventory"), title)
            imaging.save_jpg(overlay, run_dir / rel)
            record["overlay"] = str(rel)
            record.pop("render_error", None)
        except Exception as exc:
            record["overlay"] = None
            record["render_error"] = f"{type(exc).__name__}: {exc}"
            continue
        rendered.append((record, run_dir / rel))

    if rendered:
        # Chaos scenes legitimately mix aspect ratios (720p rule preserves
        # aspect), so pad every overlay onto one common canvas before tiling —
        # the base contact_sheet assumes uniform sizes.
        images = [imaging.load_rgb(path) for _, path in rendered]
        max_h = max(img.shape[0] for img in images)
        max_w = max(img.shape[1] for img in images)
        padded = [np.pad(img, ((0, max_h - img.shape[0]),
                               (0, max_w - img.shape[1]), (0, 0)),
                         constant_values=imaging.CANVAS_GREY[0])
                  for img in images]
        titles = [rec["sample_id"] for rec, _ in rendered]
        sheet = imaging.contact_sheet(padded, titles, cols=min(2, len(images)))
        imaging.save_jpg(sheet, sheets / "chaos_sheet.jpg")


def write_report(run_dir: Path, args, manifest, control, records: list[dict],
                 summary: dict, started: str, elapsed: float,
                 run_summary: dict, rebuilt_with: str | None = None) -> None:
    info = run_summary["harness"]
    seg_info = run_summary["seg_harness"]
    base = run_summary["base_harness"]
    model = run_summary["model"]
    effort = run_summary["effort"]
    lines: list[str] = []
    add = lines.append

    delivered = None if control is None else bool(control["images_delivered"])
    verdict = ("BATCH VALID" if delivered else
               "BATCH INVALID (vision delivery failed)" if delivered is False
               else "BATCH UNVERIFIED (control skipped)")
    add(f"# Chaos strawberry detection (detect + segment) — {verdict}")
    add("")
    failed_note = (f"; {summary['calls'] - summary['parsed']} call(s) failed to "
                   f"parse and are excluded from every table"
                   if summary["calls"] != summary["parsed"] else "")
    add(f"Headline: **QUALITATIVE** — {summary['fruit_total']} fruit reported "
        f"across {summary['calls']} curated chaos scene(s), with "
        f"{summary['polygons_ok']} valid visible-surface polygons. No ground "
        f"truth exists on any chaos scene, so nothing is scored{failed_note}.")
    add("")
    add(f"- **Harness:** {info['name']} v{info['version']} "
        f"(fingerprint `{info['fingerprint']}`)")
    add(f"- **Seg harness (polygon scorer, imported):** "
        f"{seg_info['name']} v{seg_info['version']} "
        f"(fingerprint `{seg_info['fingerprint']}`)")
    add(f"- **Base harness (provider stack, parser, control):** "
        f"{base['name']} v{base['version']} "
        f"(fingerprint `{base['fingerprint']}`)")
    add(f"- **Model:** `{model}` via {args.provider} CLI "
        f"({run_summary.get('cli_version', '')})")
    add(f"- **Reasoning effort:** {effort or 'provider default'}")
    if rebuilt_with:
        add(f"- ⚠ artefacts rebuilt with harness fingerprint `{rebuilt_with}`; "
            f"records were produced by `{records[0].get('harness_fingerprint')}`.")
    add("")

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
            add("Every result below is INVALID: the model did not receive the image.")
    add("")

    add("## Method")
    add("- Scenes: the curated chaos list (dense fruit, heavy clutter, deep "
        "occlusion) from `vlm_eval/chaos/manifest.json`; originals and delivery "
        "normalisation are sha256-recorded there (the 720P rule: fit inside "
        "1280x720 preserving aspect; model inputs stay PNG).")
    for s in manifest["samples"]:
        add(f"  - `{s['sample_id']}` — {s['source_image']} "
            f"({s['source_image_shape_hw'][1]}x{s['source_image_shape_hw'][0]}) "
            f"→ `{s['image']}` ({s['image_shape_hw'][1]}x"
            f"{s['image_shape_hw'][0]}); {s['normalisation']}")
    add("- Prompt: `inventory_segmentation`, a FROZEN byte-identical copy of the "
        "StrawDI seg pipeline's prompt — the StrawDI nine-field standard: the "
        "eight detection fields (bbox = **detection**) plus `polygon` (the "
        "fruit's **visible surface** = segmentation). No `picking_point`, no "
        "`target_index`: a reply volunteering either is rejected wholesale by "
        "the schema. Printed verbatim at the end of this report.")
    add("- Diagnostics (answer-internal only): polygon validity (out-of-frame / "
        "degenerate), vertex budget, polygon-vs-bbox extent IoU.")
    add("")
    add(f"> {NO_GT_NOTE}")
    add("")

    add("## Polygon diagnostics (all parsed scenes)")
    add("| metric | value |")
    add("| --- | --- |")
    add(f"| calls | {summary['calls']} |")
    add(f"| parsed (ok) | {summary['parsed']} ({summary['ok']}) |")
    add(f"| parse rate | {summary['parse_rate']} |")
    add(f"| schema-valid rate | {summary['schema_valid_rate']} |")
    add(f"| fruit reported | {summary['fruit_total']} |")
    add(f"| polygons valid / out-of-frame / degenerate | "
        f"{summary['polygons_ok']} / {summary['polygons_out_of_frame']} / "
        f"{summary['polygons_degenerate']} |")
    add(f"| mean vertices per polygon | {summary['mean_vertex_count']} |")
    add(f"| polygon-vs-bbox extent IoU (mean) | {summary['polygon_bbox_iou_mean']} |")
    add("")
    if summary["failures_by_status"]:
        add(f"failures by status: {summary['failures_by_status']}")
        add("")
    if summary.get("retried_calls"):
        add(f"retried calls (recovered): {summary['retried_calls']} "
            f"({summary['retries_recovered']} parsed after retry; "
            f"empty→low-effort or schema-invalid→same-effort, all recorded)")
        add("")

    add("## Per-scene results")
    add("| scene | original | delivered | status | found | ok polygons | tok |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    samples = {s["sample_id"]: s for s in manifest["samples"]}
    for r in records:
        tok = r.get("total_tokens")
        sample = samples.get(r["sample_id"], {})
        original = (f"{r.get('episode')} "
                    f"({sample.get('source_image_shape_hw', [0, 0])[1]}x"
                    f"{sample.get('source_image_shape_hw', [0, 0])[0]})")
        add(f"| {r['sample_id']} | {original} | "
            f"{r['frame_w']}x{r['frame_h']} | {r['status']} | "
            f"{r.get('n_strawberries')} | {r.get('n_polygons_ok')} | "
            f"{tok} |")
    add("")

    # Per-fruit detail: the point of a curated 2-scene pipeline — every fruit,
    # eyeballable against the overlay.
    add("## Per-fruit detail")
    for r in records:
        entries = r.get("inventory") or []
        add("")
        add(f"### `{r['sample_id']}` — {len(entries)} fruit "
            f"([overlay]({r.get('overlay') or 'overlays/'}))")
        if r["status"] != parse.OK:
            add(f"_{r['status']}: no inventory to list._")
            continue
        for i, berry in enumerate(entries):
            poly = berry.get("polygon")
            poly_s = f"{len(poly)} verts" if poly else "none"
            add(f"- #{i}: bbox `{berry.get('bbox')}`, polygon {poly_s}, redness "
                f"{berry.get('redness_pct')}%, occluded {berry.get('occlusion_pct')}%, "
                f"calyx {'visible' if berry.get('calyx_visible') else 'hidden'}, "
                f"peduncle {'visible' if berry.get('peduncle_visible') else 'hidden'}, "
                f"graspable {'yes' if berry.get('graspable') else 'no'}, "
                f"conf {berry.get('confidence_pct')}%"
                + (f"\n  _{berry['description'].strip()}_"
                   if isinstance(berry.get("description"), str)
                   and berry["description"].strip() else ""))
    add("")

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

    add("## Overlays")
    add("Per-scene overlays in `overlays/` and one `contact_sheets/chaos_sheet.jpg`: "
        "polygon outline + light fill coloured by that fruit's CONTINUOUS "
        "redness ramp (green at 0% → amber at 50% → red at 100%). "
        "Out-of-frame / degenerate polygons are counted in the tables, not "
        "drawn. JPG quality 90; the model inputs are untouched PNG.")
    add("")

    add("## Exact prompt (verbatim, first scene)")
    add("")
    add("```text")
    add(records[0]["prompt"] if records and records[0].get("prompt")
        else build_prompt(args, manifest["samples"][0]))
    add("```")
    add("")

    add("## Provenance")
    add(f"- run directory name encodes model/effort/CLI/harness: "
        f"`{run_dir.name}`")
    add(f"- provenance chain: base `{base['fingerprint']}` (provider stack, "
        f"parser, control) → seg `{seg_info['fingerprint']}` (the polygon "
        f"scorer this pipeline imports) → chaos `{info['fingerprint']}` "
        f"(this pipeline: runner + frozen StrawDI prompt/schema + renderer).")
    add(f"- manifest snapshot + catalog snapshot in this directory; records in "
        f"`responses.jsonl` carry the full identity per call.")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")


def write_artifacts(run_dir: Path, args, manifest, control, records: list[dict],
                    started: str, elapsed: float, run_summary: dict,
                    rebuilt_with: str | None = None) -> None:
    # Re-derive the diagnostic blocks from the stored inventories so a rebuild
    # after a scoring fix re-scores every old run.
    for record in records:
        record.update(
            seg_scoring.polygon_diag_block(record.get("inventory"),
                                           record["frame_w"], record["frame_h"])
            if record["status"] == parse.OK
            else seg_scoring.empty_diag())
    records.sort(key=lambda r: r["sample_id"])

    _render_all(run_dir, records)
    (run_dir / "responses.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records))
    summary = chaos_summary(records)
    write_csvs(run_dir, records, summary)
    write_report(run_dir, args, manifest, control, records, summary, started,
                 elapsed, run_summary, rebuilt_with)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def dry_run(args, manifest, model) -> None:
    samples = select_samples(args, manifest)
    print(f"dry run : {len(samples)} chaos scene(s), 1 prompt ({STYLE_NAME}), "
          f"1 control call; provider={args.provider} model={model}")
    print(f"plan    : {len(samples) + 1} model call(s) if run for real")
    sample = samples[0]
    h, w = sample["image_shape_hw"]
    print(f"first   : {sample['sample_id']} ({w}x{h}, no ground truth)")
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
        "seg_harness": {
            "name": records[0].get("seg_harness"),
            "version": records[0].get("seg_harness_version"),
            "fingerprint": records[0].get("seg_harness_fingerprint"),
        },
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
    seg_info = seg_runner.harness_info()
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
        "seg_harness": seg_info,
        "base_harness": base,
    }

    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()

    vre.check_input_sizes(samples, None)

    print(f"run dir : {run_dir}")
    print(f"model   : {model}  provider={args.provider}  "
          f"effort={effort or 'config default'}")
    print(f"plan    : {len(samples)} chaos scene(s) x 1 prompt ({STYLE_NAME}) "
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
                   "harness": info, "seg_harness": seg_info, "base_harness": base}
    write_artifacts(run_dir, args, manifest, control, records, started, elapsed,
                    run_summary)

    partial_path.unlink(missing_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)
    summary = chaos_summary(records)
    print(f"\ndone in {elapsed:.0f}s -> {run_dir}")
    print(f"report  : {run_dir / 'report.md'}")
    print(f"metrics : {summary['fruit_total']} fruit, "
          f"{summary['polygons_ok']} valid polygons "
          f"({summary['polygons_out_of_frame']} oof, "
          f"{summary['polygons_degenerate']} degenerate)")


def _progress(done: int, total: int, record: dict) -> None:
    print(f"[{done}/{total}] {record['run_id']:<44} {record['status']:<13} "
          f"poly={record.get('n_polygons_ok') or 0:<3} "
          f"tok={record.get('total_tokens')}", flush=True)


if __name__ == "__main__":
    main()
