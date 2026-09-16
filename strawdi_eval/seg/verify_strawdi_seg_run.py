#!/usr/bin/env python3
"""Acceptance gate for a strawdi_seg run directory.

Re-derives everything that must reproduce from the artefacts alone (or from
the artefacts plus the sha256-verified label PNGs) and fails on any drift:

* artefacts present; per-record statuses known; overlays exist and decode;
* detection coherence: tp+fp == n_pred, tp+fn == n_gt (mask block AND box
  cross-check); failure statuses carry None, never zeros;
* no picking-point leakage (``picking_point`` / ``target_index`` volunteered
  into an inventory entry), no tool attempts, control verdict consistent
  with the report headline;
* GT reproducibility: every record's label PNG still hashes to the manifest
  snapshot and ``build_manifest.derive_instances`` still reproduces the
  recorded GT boxes/areas;
* scorer reproducibility: re-rasterising the stored polygons and re-running
  both scorers reproduces every recorded number.

Usage: python3 strawdi_eval/seg/verify_strawdi_seg_run.py <run_dir>
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent          # strawdi_eval/seg
STRAWDI = HERE.parent                            # strawdi_eval
REPO = STRAWDI.parent
sys.path.insert(0, str(REPO))

from vlm_eval.lib import parse  # noqa: E402
from strawdi_eval import build_manifest  # noqa: E402
from strawdi_eval.lib import scoring as det_scoring  # noqa: E402
from strawdi_eval.seg.lib import seg_scoring  # noqa: E402
from strawdi_eval.seg.run_segmentation_eval import (  # noqa: E402
    HARNESS_NAME, mask_block, box_metrics_block)

FORBIDDEN_KEYS = ("picking_point", "target_index")
PARSED_STATUSES = (parse.OK, parse.NO_PICK_POINT)

FAILURES: list[str] = []
CHECKS = 0


def check(condition: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    status = "ok  " if condition else "FAIL"
    print(f"  {status}  {label}")
    if not condition:
        FAILURES.append(label)


def section(title: str) -> None:
    print(f"\n{title}")


def load_label(sample: dict) -> np.ndarray:
    """The label id-map PNG, sha256-verified against the manifest snapshot."""
    label_path = Path(sample["label_image"])
    digest = hashlib.sha256(label_path.read_bytes()).hexdigest()
    if digest != sample["label_sha256"]:
        raise RuntimeError(f"{label_path.name}: sha256 drift vs manifest")
    with Image.open(label_path) as label_img:
        return np.array(label_img)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    run_dir = Path(sys.argv[1]).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"not a directory: {run_dir}")

    section("artefacts")
    for name in ("responses.jsonl", "manifest.json", "catalog.json", "report.md",
                 "metrics.csv", "seg_summary.csv", "size_strata.csv", "ap.csv"):
        check((run_dir / name).is_file(), f"{name} exists")
    records = [json.loads(line) for line in
               (run_dir / "responses.jsonl").read_text().splitlines() if line.strip()]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    samples = {s["sample_id"]: s for s in manifest["samples"]}
    control = (json.loads((run_dir / "control.json").read_text())
               if (run_dir / "control.json").exists() else None)
    report = (run_dir / "report.md").read_text()
    check(bool(records), "responses.jsonl has records")

    section("per-record status + overlays")
    known = set(PARSED_STATUSES) | set(parse.FAILURE_STATUSES)
    for record in records:
        sample_id = record["sample_id"]
        check(record["status"] in known, f"{sample_id}: known status {record['status']!r}")
        check(record.get("harness") == HARNESS_NAME,
              f"{sample_id}: harness is {HARNESS_NAME}")
        sample = samples.get(sample_id)
        check(sample is not None, f"{sample_id}: in manifest snapshot")
        if sample:
            check(record.get("n_gt") == sample["n_gt"],
                  f"{sample_id}: n_gt {record.get('n_gt')} == manifest {sample['n_gt']}")
            check(record.get("label_sha256") == sample["label_sha256"],
                  f"{sample_id}: label sha256 matches manifest snapshot")
        if record["status"] in PARSED_STATUSES:
            overlay = run_dir / (record.get("overlay") or "")
            ok = overlay.is_file()
            if ok:
                try:
                    with Image.open(overlay) as img:
                        img.verify()
                except Exception:
                    ok = False
            check(ok, f"{sample_id}: overlay exists and decodes")
        check(isinstance(record.get("total_tokens"), int),
              f"{sample_id}: token count recorded")

    section("detection coherence (mask block + box cross-check)")
    for record in records:
        sample_id = record["sample_id"]
        if record["status"] in PARSED_STATUSES and record.get("tp_50") is not None:
            check(record["tp_50"] + record["fp_50"] == record["n_pred"],
                  f"{sample_id}: mask tp+fp == n_pred")
            check(record["tp_50"] + record["fn_50"] == record["n_gt"],
                  f"{sample_id}: mask tp+fn == n_gt")
            box = record.get("box_metrics") or {}
            check(box.get("tp_50") is not None
                  and box["tp_50"] + box["fp_50"] == record["n_pred"],
                  f"{sample_id}: box tp+fp == n_pred")
            check(box.get("tp_50", 0) + box.get("fn_50", 0) == record["n_gt"],
                  f"{sample_id}: box tp+fn == n_gt")
        else:
            flat_none = all(record.get(k) is None
                            for k in seg_scoring.MASK_KEYS if k != "n_gt")
            check(flat_none, f"{sample_id}: failure carries None mask numbers")
            box = record.get("box_metrics") or {}
            check(all(box.get(k) is None for k in det_scoring.DETECTION_KEYS
                      if k != "n_gt"),
                  f"{sample_id}: failure carries None box numbers")

    section("contract hygiene")
    leaked = tools = 0
    for record in records:
        for entry in record.get("inventory") or []:
            if isinstance(entry, dict) and any(k in entry for k in FORBIDDEN_KEYS):
                leaked += 1
        if record.get("tool_attempts"):
            tools += 1
    check(leaked == 0, "no picking_point/target_index volunteered anywhere")
    check(tools == 0, "no tool attempts in any record")
    if control is not None:
        check(not control.get("tool_attempts"), "no tool attempts in the control")
        delivered = bool(control["images_delivered"])
        check(("BATCH VALID" in report) == delivered,
              "report verdict consistent with control.json")
    else:
        check("BATCH UNVERIFIED" in report, "skipped control reported as unverified")

    section("provenance")
    first = records[0]
    check(f"{first['harness']} v{first['harness_version']}" in report,
          "report states harness + version")
    check(f"`{first['harness_fingerprint']}`" in report,
          "report states the harness fingerprint")
    check(f"`{first['base_harness_fingerprint']}`" in report,
          "report states the base-harness fingerprint")
    check(f"`{first['model']}`" in report, "report states the model")
    fingerprints = {r["harness_fingerprint"] for r in records}
    check(len(fingerprints) == 1, "one harness fingerprint across records")

    section("GT reproducibility (label PNGs, sha256-guarded)")
    masks_by_sample: dict[str, list[np.ndarray]] = {}
    for record in records:
        sample_id = record["sample_id"]
        sample = samples.get(sample_id)
        if sample is None:
            continue
        try:
            label = load_label(sample)
        except Exception as exc:
            check(False, f"{sample_id}: label readable ({exc})")
            continue
        instances = build_manifest.derive_instances(label)
        boxes = [i["bbox_xyxy"] for i in instances]
        areas = [i["area_px"] for i in instances]
        check(boxes == record.get("gt_boxes") and areas == record.get("gt_areas"),
              f"{sample_id}: derive_instances reproduces gt_boxes/gt_areas")
        masks_by_sample[sample_id] = [label == i["instance_id"]
                                      for i in instances]

    section("scorer reproducibility (re-rasterise + re-score)")
    for record in records:
        sample_id = record["sample_id"]
        sample = samples.get(sample_id)
        masks = masks_by_sample.get(sample_id)
        if sample is None or masks is None:
            check(record.get("tp_50") is None,
                  f"{sample_id}: unscored without GT masks")
            continue
        recomputed = mask_block(record["status"], record.get("inventory"), masks,
                                sample["gt_areas"], record["frame_w"],
                                record["frame_h"], sample["n_gt"])
        stored = {k: record.get(k) for k in seg_scoring.MASK_KEYS + seg_scoring.MASK_DETAIL_KEYS}
        check(all(recomputed.get(k) == stored.get(k)
                  for k in seg_scoring.MASK_KEYS + seg_scoring.MASK_DETAIL_KEYS),
              f"{sample_id}: mask block reproduces from stored polygons")
        box = box_metrics_block(record["status"], record.get("inventory"),
                                record.get("gt_boxes"), record.get("gt_areas"),
                                record["frame_w"], record["frame_h"])
        check(all(box.get(k) == (record.get("box_metrics") or {}).get(k)
                  for k in det_scoring.DETECTION_KEYS + det_scoring.DETECTION_DETAIL_KEYS),
              f"{sample_id}: box cross-check reproduces from stored bboxes")

    print()
    if FAILURES:
        print(f"FAIL: {len(FAILURES)} of {CHECKS} checks failed")
        for label in FAILURES:
            print(f"  - {label}")
        sys.exit(1)
    print(f"PASS: all {CHECKS} checks passed")


if __name__ == "__main__":
    main()
