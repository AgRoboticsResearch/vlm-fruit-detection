#!/usr/bin/env python3
"""Acceptance gate for a vlm_seg run directory.

Re-derives everything that must reproduce from the artefacts alone (or from
the artefacts plus the upstream GT provenance) and fails on any drift:

* artefacts present; per-record statuses known; overlays exist and decode;
* I7 (unlabelled purity): records without ground truth carry NO scored
  value (error, dx/dy, PCK, IoUs, GT fields all None), and no record
  anywhere carries mask-matching numbers — no mask ground truth exists on
  any vlm_eval source, so a tp/fp/fn or mask-IoU key would be an invented
  number;
* detection coherence on ground-truthed records: scored values reproduce
  from the stored inventory + the manifest snapshot's GT;
* diagnostics reproduce from the stored inventories;
* GT reproducibility: the manifest snapshot's GT points still reproduce from
  upstream ``target_ref`` for every ground-truthed sample the run covered;
* contract hygiene: normalised inventory entries carry exactly the ten
  contract fields; no tool attempts; control verdict consistent with the
  report headline; provenance complete and single-fingerprinted.

Usage: python3 vlm_eval/seg/verify_vlm_seg_run.py <run_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent          # vlm_eval/seg
VLM = HERE.parent                                # vlm_eval
REPO = VLM.parent
sys.path.insert(0, str(REPO))

from vlm_eval.lib import gtbridge, parse  # noqa: E402
from vlm_eval.seg.lib import seg_scoring  # noqa: E402
from vlm_eval.seg.run_segmentation_eval import (  # noqa: E402
    HARNESS_NAME, diag_block, scored_block)

PARSED_STATUSES = (parse.OK, parse.NO_PICK_POINT)

# Mask-matching keys that must NEVER appear in a vlm_seg record: no mask
# ground truth exists on any vlm_eval source, so any such number would be
# invented, not measured.
FORBIDDEN_MASK_KEYS = (
    "tp_25", "tp_50", "tp_75", "fp_25", "fp_50", "fp_75",
    "fn_25", "fn_50", "fn_75", "f1_50", "precision_50", "recall_50",
    "mean_matched_iou_50", "matches_50", "fp_pred_indices_50",
    "fn_gt_indices_50", "mask_ap_50", "mask_map_50_95",
)

# Exactly the fields a normalised inventory entry may carry (the ten-field
# contract, picking_point included).
ALLOWED_ENTRY_KEYS = {
    "bbox", "polygon", "redness_pct", "occlusion_pct", "calyx_visible",
    "peduncle_visible", "graspable", "picking_point", "confidence_pct",
    "description",
}

# Scored keys that must stay None on unlabelled frames (I7).
I7_SCORED_KEYS = (
    "error_px", "dx_px", "dy_px", "error_pct_width", "bbox_iou_rough",
    "polygon_extent_iou_rough", "polygon_contains_gt",
    "gt_to_target_polygon_px", "pck_5", "pck_10", "pck_20",
    "gt_uv", "gt_uv_rounded", "gt_z_m", "gt_rough_box", "gt_status",
)

GT_TOLERANCE_PX = 1e-3

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


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    run_dir = Path(sys.argv[1]).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"not a directory: {run_dir}")

    section("artefacts")
    for name in ("responses.jsonl", "manifest.json", "catalog.json", "report.md",
                 "metrics.csv", "seg_summary.csv"):
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
        check(record.get("base_harness") == "vlm_eval",
              f"{sample_id}: base harness is vlm_eval")
        sample = samples.get(sample_id)
        check(sample is not None, f"{sample_id}: in manifest snapshot")
        if sample:
            check(record.get("has_gt") == bool(sample.get("has_gt")),
                  f"{sample_id}: has_gt matches manifest")
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

    section("no invented mask metrics + unlabelled purity (I7)")
    mask_keys_present = sorted(
        {key for record in records for key in record if key in FORBIDDEN_MASK_KEYS})
    check(not mask_keys_present,
          f"no mask-matching keys anywhere (found: {mask_keys_present})")
    for record in records:
        sample_id = record["sample_id"]
        if not record.get("has_gt"):
            leaked = [key for key in I7_SCORED_KEYS
                      if record.get(key) is not None]
            check(not leaked and not record.get("target_scored"),
                  f"{sample_id}: unlabelled record carries no scored "
                  f"value (leaked: {leaked})")
        # The normalised inventory must carry exactly the ten contract fields.
        extra = sorted({key for entry in record.get("inventory") or []
                        if isinstance(entry, dict)
                        for key in entry if key not in ALLOWED_ENTRY_KEYS})
        check(not extra, f"{sample_id}: inventory entries carry exactly the "
                         f"ten contract fields (extra: {extra})")

    section("contract hygiene")
    tools = 0
    for record in records:
        if record.get("tool_attempts"):
            tools += 1
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

    section("GT reproducibility (target_ref recompute)")
    gt_records = [r for r in records if r.get("has_gt")]
    if not gt_records:
        print("  –    no ground-truthed frames in this batch")
    else:
        target_ref = gtbridge.load_target_ref()
        tip_kin = target_ref.load_tip_kin(target_ref.DEFAULT_EXTRINSICS_CONFIG)
        mismatches = []
        for record in gt_records:
            sample = samples[record["sample_id"]]
            gt = target_ref.compute_episode_target(sample["episode_dir"], tip_kin)
            if gt["status"] != sample["gt_status"]:
                mismatches.append(f"{sample['sample_id']}: status drift "
                                  f"{gt['status']} != {sample['gt_status']}")
                continue
            if gt["status"] != "ok":
                continue
            du = abs(gt["u"] - sample["gt_uv"][0])
            dv = abs(gt["v"] - sample["gt_uv"][1])
            if max(du, dv) > GT_TOLERANCE_PX:
                mismatches.append(f"{sample['sample_id']}: recomputed GT off by "
                                  f"({du:.2e}, {dv:.2e})px")
        check(not mismatches,
              f"manifest GT reproduces from target_ref ({mismatches})")
        # The record's own GT block must equal the manifest snapshot's.
        drift = [r["sample_id"] for r in gt_records
                 if r.get("gt_uv") != samples[r["sample_id"]]["gt_uv"]
                 or r.get("gt_rough_box") != samples[r["sample_id"]]["gt_rough_box"]]
        check(not drift, f"records carry the manifest's GT unchanged ({drift})")

    section("scorer reproducibility (re-derive from stored inventories)")
    for record in records:
        sample_id = record["sample_id"]
        sample = samples.get(sample_id)
        if sample is None:
            continue
        recomputed_diag = diag_block(record["status"], record.get("inventory"),
                                     record["frame_w"], record["frame_h"])
        check(all(recomputed_diag.get(k) == record.get(k)
                  for k in seg_scoring.DIAG_KEYS),
              f"{sample_id}: polygon diagnostics reproduce from stored inventory")
        recomputed_scored = scored_block(record["status"], record.get("inventory"),
                                         record.get("target_index"), sample,
                                         record["frame_w"], record["frame_h"])
        mismatched = [k for k in seg_scoring.SCORED_KEYS
                      if recomputed_scored.get(k) != record.get(k)]
        pcks = {f"pck_{t}": (None if recomputed_scored["error_px"] is None
                             else bool(recomputed_scored["error_px"] <= t))
                for t in seg_scoring.PCK_THRESHOLDS}
        mismatched += [k for k, v in pcks.items() if record.get(k) != v]
        check(not mismatched,
              f"{sample_id}: scored block reproduces from stored inventory "
              f"({mismatched})")

    print()
    if FAILURES:
        print(f"FAIL: {len(FAILURES)} of {CHECKS} checks failed")
        for label in FAILURES:
            print(f"  - {label}")
        sys.exit(1)
    print(f"PASS: all {CHECKS} checks passed")


if __name__ == "__main__":
    main()
