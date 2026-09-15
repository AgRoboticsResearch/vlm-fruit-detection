#!/usr/bin/env python3
"""Acceptance gate for a StrawDI detection-eval run directory.

Everything asserted here is a property the harness promises, so this is the
thing to run after a batch (and after any change to the harness):

* the control reported image delivery, or the report says the batch is invalid
* every model call has a parsed-or-explicitly-failed status, an overlay PNG, a
  JSONL record and a token count
* no record leaks SROI picking-point fields (there is no such ground truth in
  StrawDI, so a stray value would be nonsense)
* ground truth reproduces: every manifest box is re-derived from the dataset
  mask PNGs by the same function that built the manifest, and the frame/label
  files are byte-identical (sha256) to what the run snapshotted
* the detection scorer reproduces: re-scoring stored inventories reproduces
  every recorded TP/FP/FN number
* the report states both harness fingerprints, the model and the effort, and
  they agree with the records

Usage:  python3 strawdi_eval/verify_strawdi_run.py strawdi_eval/runs/<dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from vlm_eval.lib import parse  # noqa: E402
from strawdi_eval import build_manifest  # noqa: E402
from strawdi_eval.lib import scoring  # noqa: E402

# Statuses that count as "the answer parsed"; everything else must be an
# explicit failure. NOTE the difference from the base harness: `no_pick_point`
# is a *parse success* here and is scored for detection — the nomination has
# no ground truth in StrawDI, so it must never gate the boxes.
PARSE_SUCCESS = {parse.OK, parse.NO_PICK_POINT}

# Fields from the SROI picking-point eval that have no meaning here; any of
# these on a record means a wrong template leaked in.
FORBIDDEN_KEYS = (
    "gt_uv", "gt_uv_rounded", "gt_z_m", "gt_status", "gt_rough_box",
    "session", "episode", "episode_dir", "sensitivity",
    "error_px", "dx_px", "dy_px", "error_pct_width",
    "bbox_iou_rough", "bbox_contains_gt", "pck_5", "pck_10", "pck_20",
    "target_scored", "panel_origin",
)


class Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def check(self, condition: bool, message: str) -> bool:
        self.checks += 1
        if not condition:
            self.failures.append(message)
            print(f"  FAIL  {message}")
        else:
            print(f"  ok    {message}")
        return condition


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"not a directory: {run_dir}")

    c = Checker()
    print(f"verifying {run_dir}")

    # ---- required artefacts ----------------------------------------------
    print("\n[artefacts]")
    for name in ("report.md", "responses.jsonl", "metrics.csv",
                 "manifest.json", "catalog.json"):
        c.check((run_dir / name).exists(), f"{name} exists")
    records = [json.loads(line) for line in
               (run_dir / "responses.jsonl").read_text().splitlines() if line.strip()]
    c.check(len(records) > 0, f"responses.jsonl has {len(records)} record(s)")

    # ---- per-record completeness -----------------------------------------
    print("\n[per-record completeness]")
    known = PARSE_SUCCESS | {parse.OUT_OF_FRAME, *parse.FAILURE_STATUSES}
    bad_status = [r["run_id"] for r in records if r.get("status") not in known]
    c.check(not bad_status, f"every run has a known status (bad: {bad_status})")

    missing_overlay = [r["run_id"] for r in records
                       if not r.get("overlay") or not (run_dir / r["overlay"]).exists()]
    c.check(not missing_overlay,
            f"every run has an overlay PNG (missing: {missing_overlay})")

    broken = []
    for record in records:
        if not record.get("overlay"):
            continue  # already flagged by the missing-overlay check above
        path = run_dir / record["overlay"]
        if path.exists():
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception as exc:
                broken.append(f"{record['run_id']}: {exc}")
    c.check(not broken, f"every overlay decodes as an image ({broken})")

    no_tokens = [r["run_id"] for r in records
                 if r["status"] in PARSE_SUCCESS and not r.get("total_tokens")]
    c.check(not no_tokens,
            f"every parsed run has a token count (missing: {no_tokens})")

    no_gt = [r["run_id"] for r in records if not isinstance(r.get("n_gt"), int)]
    c.check(not no_gt, f"every run carries its GT instance count (missing: {no_gt})")

    # ---- detection-field coherence ---------------------------------------
    print("\n[detection fields]")
    # n_gt is sample metadata (kept on failures); every other detection key
    # must be None on a failed call — never zeros.
    failure_keys = set(scoring.DETECTION_KEYS) - {"n_gt"}
    incoherent = []
    for record in records:
        if record["status"] not in PARSE_SUCCESS:
            if any(record.get(k) is not None for k in failure_keys):
                incoherent.append(f"{record['run_id']}: failure carries numbers")
            continue
        if record["tp_50"] + record["fp_50"] != record["n_pred"]:
            incoherent.append(f"{record['run_id']}: tp+fp != n_pred")
        if record["tp_50"] + record["fn_50"] != record["n_gt"]:
            incoherent.append(f"{record['run_id']}: tp+fn != n_gt")
    c.check(not incoherent, f"tp/fp/fn coherent with n_pred/n_gt ({incoherent})")

    leaked = [f"{r['run_id']}:{k}" for r in records for k in FORBIDDEN_KEYS
              if r.get(k) is not None]
    c.check(not leaked, f"no SROI picking-point fields leaked ({leaked})")

    tool_use = sorted({t for r in records for t in (r.get("tool_attempts") or [])})
    c.check(not tool_use, f"no run attempted to use a tool (attempts: {tool_use})")

    # ---- metrics.csv ------------------------------------------------------
    print("\n[metrics]")
    import pandas as pd

    metrics = pd.read_csv(run_dir / "metrics.csv")
    c.check(len(metrics) == len(records),
            f"metrics.csv has {len(metrics)} rows for {len(records)} records")

    # ---- report identity + control verdict -------------------------------
    print("\n[report identity]")
    report = (run_dir / "report.md").read_text()
    for label, needle in (("harness name", "**Harness:**"),
                          ("base harness", "**Base harness:**"),
                          ("model name", "**Model:**"),
                          ("reasoning effort", "**Reasoning effort:**")):
        c.check(needle in report, f"report states the {label}")
    models = {r.get("model") for r in records if r.get("model")}
    c.check(len(models) <= 1 and all(m and m != "unknown" for m in models),
            f"every record carries one model name ({sorted(models)})")
    if models:
        model = next(iter(models))
        c.check(f"`{model}`" in report, f"report names the model actually used ({model})")
    for label, key in (("harness fingerprint", "harness_fingerprint"),
                       ("base-harness fingerprint", "base_harness_fingerprint")):
        prints = {r.get(key) for r in records if r.get(key)}
        c.check(len(prints) <= 1,
                f"every record carries one {label} ({sorted(prints)})")
        if prints:
            c.check(next(iter(prints)) in report,
                    f"report states the {label} ({next(iter(prints))})")

    print("\n[vision delivery]")
    control_path = run_dir / "control.json"
    if control_path.exists():
        control = json.loads(control_path.read_text())
        delivered = bool(control["images_delivered"])
        c.check("Vision delivery: YES" in report or "Vision delivery: NO" in report,
                "report states the vision-delivery verdict explicitly")
        c.check(("Vision delivery: YES" in report) == delivered,
                f"report verdict matches control.json (delivered={delivered})")
        c.check(delivered, "the control image was actually read by the model")
        c.check(not control.get("tool_attempts"),
                "control made no tool attempts")
    else:
        c.check("Vision delivery: UNVERIFIED" in report,
                "control was skipped and the report says so")

    # ---- ground-truth reproducibility (I5 analogue) -----------------------
    print("\n[ground truth]")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    problems = []
    for sample in manifest["samples"]:
        label_path = Path(sample["label_image"])
        if not label_path.exists():
            problems.append(f"{sample['sample_id']}: label mask missing ({label_path})")
            continue
        if sha256(label_path) != sample["label_sha256"]:
            problems.append(f"{sample['sample_id']}: label sha256 drifted")
        mask = np.array(Image.open(label_path))
        instances = build_manifest.derive_instances(mask)
        boxes = [inst["bbox_xyxy"] for inst in instances]
        areas = [inst["area_px"] for inst in instances]
        if boxes != sample["gt_boxes"] or areas != sample["gt_areas"] \
                or len(boxes) != sample["n_gt"]:
            problems.append(f"{sample['sample_id']}: re-derived GT differs")
        frame_path = HERE / sample["images"]["raw"]
        if not frame_path.exists():
            problems.append(f"{sample['sample_id']}: frame snapshot missing")
        elif sha256(frame_path) != sample["frame_sha256"]:
            problems.append(f"{sample['sample_id']}: frame sha256 drifted")
    c.check(not problems,
            f"GT boxes reproduce from the dataset masks ({len(manifest['samples'])} "
            f"samples checked; problems: {problems})")

    # ---- scorer reproducibility -------------------------------------------
    print("\n[scorer]")
    by_id = {s["sample_id"]: s for s in manifest["samples"]}
    mismatches = []
    checked = 0
    for record in sorted(records, key=lambda r: r["sample_id"]):
        if record["status"] not in PARSE_SUCCESS or checked >= 10:
            continue
        checked += 1
        sample = by_id[record["sample_id"]]
        recomputed = scoring.score_image(record["inventory"],
                                         sample["gt_boxes"], sample["gt_areas"],
                                         record["frame_w"], record["frame_h"])
        for key in ("tp_50", "fp_50", "fn_50", "mean_matched_iou_50",
                    "count_error", "tp_75", "fn_75", "tp_center"):
            if recomputed.get(key) != record.get(key):
                mismatches.append(f"{record['run_id']}.{key}: "
                                  f"{recomputed.get(key)} != {record.get(key)}")
    c.check(checked > 0, f"scored {checked} record(s) re-scored")
    c.check(not mismatches, f"scorer reproduces every recorded number ({mismatches})")

    # ---- summary ----------------------------------------------------------
    print(f"\n{len(c.failures)} failure(s) out of {c.checks} check(s)")
    if c.failures:
        print("FAILED:")
        for failure in c.failures:
            print(f"  - {failure}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
