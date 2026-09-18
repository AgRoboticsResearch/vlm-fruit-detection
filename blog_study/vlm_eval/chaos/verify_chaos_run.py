#!/usr/bin/env python3
"""Acceptance gate for a vlm_chaos run directory.

Re-derives everything that must reproduce from the artefacts alone and fails
on any drift:

* artefacts present; per-record statuses known; overlays exist and decode;
* frame integrity: every delivered chaos frame still hashes to the manifest
  snapshot (a chaotically-named PNG in the shared frames dir must never
  silently change meaning);
* purity: chaos scenes have NO ground truth, so no scored value of any kind
  (error/IoU/PCK/mask numbers) may appear anywhere — the verifier fails on
  any such key, the exact analogue of the base harness's I7;
* contract: the StrawDI nine-field standard — no picking fields may be
  volunteered anywhere (``picking_point`` / ``target_index``), and the
  normalised inventory entries carry exactly the nine contract fields;
* diagnostics reproduce from the stored inventories;
* no tool attempts; control verdict consistent with the report headline;
* provenance complete: all THREE fingerprints (base → seg → chaos) stated
  in the report, one chaos fingerprint across records.

Usage: python3 vlm_eval/chaos/verify_chaos_run.py <run_dir>
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent          # vlm_eval/chaos
VLM = HERE.parent                                # vlm_eval
REPO = VLM.parent
sys.path.insert(0, str(REPO))

from vlm_eval.lib import parse  # noqa: E402
from vlm_eval.seg.lib import seg_scoring  # noqa: E402
from vlm_eval.chaos.run_chaos_eval import HARNESS_NAME  # noqa: E402

PARSED_STATUSES = (parse.OK,)

# Scored-looking keys that must NEVER appear in a vlm_chaos record: no chaos
# scene has ground truth, so any such number would be invented, not measured.
FORBIDDEN_SCORED_KEYS = (
    "error_px", "dx_px", "dy_px", "error_pct_width", "bbox_iou_rough",
    "polygon_extent_iou_rough", "polygon_contains_gt",
    "gt_to_target_polygon_px", "pck_5", "pck_10", "pck_20",
    "gt_uv", "gt_uv_rounded", "gt_z_m", "gt_rough_box", "gt_status",
    "target_scored",
    # mask-matching keys (no mask GT anywhere in the vlm_eval scenes):
    "tp_25", "tp_50", "tp_75", "fp_25", "fp_50", "fp_75",
    "fn_25", "fn_50", "fn_75", "f1_50", "precision_50", "recall_50",
    "mean_matched_iou_50", "matches_50", "mask_ap_50", "mask_map_50_95",
)

# Exactly the fields a normalised inventory entry may carry: the StrawDI
# nine-field contract — picking_point and target_index are NOT part of it.
ALLOWED_ENTRY_KEYS = {
    "bbox", "polygon", "redness_pct", "occlusion_pct", "calyx_visible",
    "peduncle_visible", "graspable", "confidence_pct", "description",
}

# Picking-oriented keys must not be volunteered anywhere either (the schema
# rejects them wholesale; the normalised records must never carry them).
FORBIDDEN_PICKING_KEYS = ("picking_point", "target_index", "target_valid")

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
                 "metrics.csv", "chaos_summary.csv"):
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
        check(record.get("seg_harness") == "vlm_seg",
              f"{sample_id}: seg harness is vlm_seg")
        check(samples.get(sample_id) is not None,
              f"{sample_id}: in chaos manifest snapshot")
        check(record.get("has_gt") is False, f"{sample_id}: recorded as unlabelled")
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

    section("frame integrity (delivered PNG sha256 vs manifest)")
    for record in records:
        sample = samples.get(record["sample_id"])
        if sample is None:
            continue
        frame_path = VLM / sample["image"]
        ok = frame_path.is_file()
        if ok:
            digest = hashlib.sha256(frame_path.read_bytes()).hexdigest()
            ok = digest == sample["delivered_sha256"]
        check(ok, f"{record['sample_id']}: delivered frame unchanged "
                  f"({Path(sample['image']).name})")

    section("purity (no scored values — no GT exists on chaos scenes)")
    present = sorted({key for record in records for key in record
                      if key in FORBIDDEN_SCORED_KEYS})
    check(not present, f"no scored/mask keys anywhere (found: {present})")
    for record in records:
        leaked = sorted(key for key in FORBIDDEN_PICKING_KEYS
                        if key in record)
        extra = sorted({key for entry in record.get("inventory") or []
                        if isinstance(entry, dict)
                        for key in entry if key not in ALLOWED_ENTRY_KEYS})
        check(not leaked and not extra,
              f"{record['sample_id']}: no picking fields, inventory entries "
              f"carry exactly the nine contract fields "
              f"(leaked: {leaked}, extra: {extra})")

    section("contract hygiene")
    tools = sum(1 for r in records if r.get("tool_attempts"))
    check(tools == 0, "no tool attempts in any record")
    if control is not None:
        check(not control.get("tool_attempts"), "no tool attempts in the control")
        delivered = bool(control["images_delivered"])
        check(("BATCH VALID" in report) == delivered,
              "report verdict consistent with control.json")
    else:
        check("BATCH UNVERIFIED" in report, "skipped control reported as unverified")

    section("provenance (base -> seg -> chaos)")
    first = records[0]
    check(f"{first['harness']} v{first['harness_version']}" in report,
          "report states harness + version")
    for field, label in (("harness_fingerprint", "chaos"),
                         ("seg_harness_fingerprint", "seg"),
                         ("base_harness_fingerprint", "base")):
        check(f"`{first[field]}`" in report,
              f"report states the {label} fingerprint")
    check(f"`{first['model']}`" in report, "report states the model")
    fingerprints = {r["harness_fingerprint"] for r in records}
    check(len(fingerprints) == 1, "one chaos fingerprint across records")
    seg_fingerprints = {r["seg_harness_fingerprint"] for r in records}
    check(len(seg_fingerprints) == 1, "one seg fingerprint across records")

    section("scorer reproducibility (re-derive from stored inventories)")
    for record in records:
        sample_id = record["sample_id"]
        if record["status"] in PARSED_STATUSES:
            recomputed = seg_scoring.polygon_diag_block(
                record.get("inventory"), record["frame_w"], record["frame_h"])
        else:
            recomputed = seg_scoring.empty_diag()
        check(all(recomputed.get(k) == record.get(k)
                  for k in seg_scoring.DIAG_KEYS),
              f"{sample_id}: polygon diagnostics reproduce from stored inventory")

    print()
    if FAILURES:
        print(f"FAIL: {len(FAILURES)} of {CHECKS} checks failed")
        for label in FAILURES:
            print(f"  - {label}")
        sys.exit(1)
    print(f"PASS: all {CHECKS} checks passed")


if __name__ == "__main__":
    main()
