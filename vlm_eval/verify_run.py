#!/usr/bin/env python3
"""Check a run directory against the eval's acceptance criteria.

Everything asserted here is a property the harness promises, so this is the
thing to run after a batch (and after any change to the harness):

* the control reported image delivery, or the report says the numbers are invalid
* every model call has a parsed-or-explicitly-failed status, an overlay PNG, a
  JSONL record and a token count
* ground truth in the manifest still matches what ``target_ref`` computes now,
  and the rendered marker's top edge sits on the picking point
* the report exists and states the vision-delivery verdict explicitly

Usage:  python3 vlm_eval/verify_run.py vlm_eval/runs/<timestamp>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from vlm_eval.lib import gtbridge, imaging, parse  # noqa: E402

# Ground truth is a float projection; re-deriving it must reproduce the manifest
# bit-for-bit up to float formatting in JSON.
GT_TOLERANCE_PX = 1e-6
MARKER_TOLERANCE_PX = 1.0


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
    known_statuses = {parse.OK, parse.OUT_OF_FRAME, parse.NO_PICK_POINT,
                      *parse.FAILURE_STATUSES}
    bad_status = [r["run_id"] for r in records if r.get("status") not in known_statuses]
    c.check(not bad_status, f"every run has a known status (bad: {bad_status})")

    missing_overlay = [r["run_id"] for r in records
                       if not r.get("overlay") or not (run_dir / r["overlay"]).exists()]
    c.check(not missing_overlay, f"every run has an overlay PNG (missing: {missing_overlay})")

    broken_overlay = []
    for record in records:
        path = run_dir / record["overlay"]
        if path.exists():
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception as exc:  # pragma: no cover
                broken_overlay.append(f"{record['run_id']}: {exc}")
    c.check(not broken_overlay, f"every overlay decodes as an image ({broken_overlay})")

    no_tokens = [r["run_id"] for r in records if not r.get("total_tokens")]
    c.check(not no_tokens, f"every run has a token count (missing: {no_tokens})")

    scored = [r for r in records if r.get("error_px") is not None]
    if any(r.get("has_gt") for r in records):
        c.check(len(scored) > 0, f"{len(scored)} run(s) scored against ground truth")

    # Unlabelled frames must never carry a score: there is nothing to score them
    # against, and a stray number here would silently enter the aggregates.
    leaked = [r["run_id"] for r in records
              if r.get("has_gt", True) is False
              and any(r.get(k) is not None
                      for k in ("error_px", "dx_px", "dy_px", "bbox_iou_rough",
                                "bbox_contains_gt", "pck_5", "sensitivity"))]
    c.check(not leaked,
            f"unlabelled runs carry no error/IoU/PCK/sensitivity values (leaked: {leaked})")
    unlabelled = [r for r in records if r.get("has_gt", True) is False]
    if unlabelled:
        c.check(all(r.get("gt_uv") is None for r in unlabelled),
                "unlabelled runs have no ground-truth coordinates attached")
        empty = [r["run_id"] for r in unlabelled
                 if r.get("status") == parse.OK and r.get("n_strawberries") is None]
        c.check(not empty, f"every parsed unlabelled run reports a fruit count ({empty})")
    unaccounted = [
        r["run_id"] for r in records
        if r.get("has_gt")
        and r.get("error_px") is None
        and r.get("status") not in parse.FAILURE_STATUSES
        and r.get("status") not in (parse.OUT_OF_FRAME, parse.NO_PICK_POINT)
    ]
    c.check(not unaccounted,
            f"unscored runs are explicitly out-of-frame or failed (unaccounted: {unaccounted})")

    tool_use = sorted({t for r in records for t in (r.get("tool_attempts") or [])})
    c.check(not tool_use, f"no run attempted to use a tool (attempts: {tool_use})")

    # ---- metrics.csv ------------------------------------------------------
    print("\n[metrics]")
    import pandas as pd

    metrics = pd.read_csv(run_dir / "metrics.csv")
    c.check(len(metrics) == len(records),
            f"metrics.csv has {len(metrics)} rows for {len(records)} records")

    # ---- vision-delivery verdict -----------------------------------------
    print("\n[vision delivery]")
    report = (run_dir / "report.md").read_text()

    # ---- report identity --------------------------------------------------
    # A number is only meaningful if you know which harness produced it, which
    # model answered, and at what reasoning effort.
    print("\n[report identity]")
    for label, needle in (("harness name", "**Harness:**"),
                          ("model name", "**Model:**"),
                          ("reasoning effort", "**Reasoning effort:**")):
        c.check(needle in report, f"report states the {label}")
    models = {r.get("model") for r in records if r.get("model")}
    c.check(len(models) <= 1 and all(m and m != "unknown" for m in models),
            f"every record carries one model name ({sorted(models)})")
    if models:
        model = next(iter(models))
        c.check(f"`{model}`" in report, f"report names the model actually used ({model})")
    fingerprint = {r.get("harness_fingerprint") for r in records
                   if r.get("harness_fingerprint")}
    c.check(len(fingerprint) <= 1,
            f"every record carries one harness fingerprint ({sorted(fingerprint)})")
    if fingerprint:
        c.check(next(iter(fingerprint)) in report,
                f"report states the harness fingerprint ({next(iter(fingerprint))})")

    control_path = run_dir / "control.json"
    if control_path.exists():
        control = json.loads(control_path.read_text())
        delivered = control["images_delivered"]
        c.check("Vision delivery: YES" in report or "Vision delivery: NO" in report,
                "report states the vision-delivery verdict explicitly")
        c.check(("Vision delivery: YES" in report) == bool(delivered),
                f"report verdict matches control.json (delivered={delivered})")
        c.check(bool(delivered), "the control image was actually read by the model")
    else:
        c.check("Vision delivery: UNVERIFIED" in report,
                "control was skipped and the report says so")

    # ---- ground truth reproducibility ------------------------------------
    print("\n[ground truth]")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    target_ref = gtbridge.load_target_ref()
    tip_kin = target_ref.load_tip_kin(target_ref.DEFAULT_EXTRINSICS_CONFIG)

    mismatches = []
    for sample in manifest["samples"]:
        if not sample.get("has_gt", True):
            continue
        gt = target_ref.compute_episode_target(sample["episode_dir"], tip_kin)
        if gt["status"] != sample["gt_status"]:
            mismatches.append(f"{sample['sample_id']}: status {gt['status']} != {sample['gt_status']}")
            continue
        if gt["status"] != "ok":
            continue
        du = abs(gt["u"] - sample["gt_uv"][0])
        dv = abs(gt["v"] - sample["gt_uv"][1])
        if max(du, dv) > GT_TOLERANCE_PX:
            mismatches.append(f"{sample['sample_id']}: recomputed GT off by ({du:.2e}, {dv:.2e})px")
    c.check(not mismatches, f"manifest ground truth reproduces from target_ref ({mismatches})")

    # ---- marker geometry: the rendered box's top edge touches the point ---
    print("\n[marker geometry]")
    marker_problems = []
    for sample in manifest["samples"]:
        if not sample.get("has_gt", True):
            continue
        image_path = HERE / sample["images"]["raw"]
        raw = imaging.load_rgb(image_path)
        u, v = sample["gt_uv"]
        marked = imaging.draw_gt_marker(raw, (u, v))
        column = int(round(u))
        changed = np.where((marked[:, column] != raw[:, column]).any(axis=1))[0]
        if not len(changed):
            marker_problems.append(f"{sample['sample_id']}: marker not visible in its column")
            continue
        top = int(changed.min())
        expected = int(round(v))
        if abs(top - expected) > MARKER_TOLERANCE_PX:
            marker_problems.append(
                f"{sample['sample_id']}: marker top {top} != picking point {expected}")
    c.check(not marker_problems,
            f"marker top edge matches the picking point within {MARKER_TOLERANCE_PX:g}px "
            f"({marker_problems})")

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
