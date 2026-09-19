#!/usr/bin/env python3
"""Acceptance gate for a paper_fruit_seg run. Must print PASS.

Checks (mirrors the strawdi seg verifier's contract, leaner):

* artefacts exist (report, responses.jsonl, metrics.csv, manifest + catalog
  snapshots, control.json, per-record overlays);
* the vision-delivery control ran and DELIVERED;
* every record has a known status; parsed records are schema-valid, carry no
  tool attempts, and carry both harness fingerprints;
* the delivered frames still hash to the manifest snapshot and none exceeds
  the 1280 px no-resize ceiling;
* the BOX scorer reproduces every recorded number from the stored inventory +
  GT (deterministic); the MASK scorer reproduces for every mask-scored record
  (needs the dataset mount — GT re-read and sha256-guarded);
* summary_by_source.csv matches recomputation from the records.

Usage: python3 paper_study/verify_run.py paper_study/runs/<run dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "blog_study"))
sys.path.insert(0, str(REPO))

from vlm_eval.lib import parse  # noqa: E402
from strawdi_eval.lib import scoring as det_scoring  # noqa: E402
from strawdi_eval.seg.lib import seg_scoring  # noqa: E402
from paper_study.run_fruit_eval import (  # noqa: E402
    HARNESS_NAME, load_gt_masks, mask_gt_areas, harness_info)

MAX_SAFE_EDGE = 1280


class Checker:
    def __init__(self):
        self.failures = 0
        self.checks = 0

    def check(self, ok: bool, label: str) -> None:
        self.checks += 1
        mark = "ok  " if ok else "FAIL"
        print(f"  {mark}  {label}")
        if not ok:
            self.failures += 1


def main(argv=None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise SystemExit("usage: verify_run.py <run dir>")
    run_dir = Path(args[0]).resolve()
    c = Checker()

    print(f"verifying {run_dir}")

    # --- artefacts ---------------------------------------------------------
    for name in ("report.md", "responses.jsonl", "metrics.csv",
                 "manifest.json", "catalog.json"):
        c.check((run_dir / name).exists(), f"{name} exists")

    control_path = run_dir / "control.json"
    control = json.loads(control_path.read_text()) if control_path.exists() else None
    c.check(control is not None, "control.json present (control ran)")
    if control is not None:
        c.check(bool(control.get("images_delivered")),
                "vision control DELIVERED "
                f"(code {control.get('prediction', {}).get('code')!r}, "
                f"circle err {control.get('circle_error_px')}px)")
        c.check(not control.get("tool_attempts"),
                "control made no tool attempts")

    records = [json.loads(line) for line in
               (run_dir / "responses.jsonl").read_text().splitlines() if line.strip()]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest_ids = [s["sample_id"] for s in manifest["samples"]]
    record_ids = [r["sample_id"] for r in records]
    c.check(all(rid in manifest_ids for rid in record_ids)
            and len(set(record_ids)) == len(record_ids),
            "records are distinct manifest samples "
            f"({len(records)}/{len(manifest_ids)} frames — partial runs from "
            "--limit verify with a subset, full runs must cover all)")

    info = harness_info()

    # Fingerprint contract: all records must share ONE harness fingerprint
    # (mixed provenance inside a run is a hard failure) and carry the base
    # harness fingerprint. Whether that shared fingerprint still matches the
    # CURRENT code is identity information, not acceptance — a completed run
    # stays valid after later code edits; the drift is reported as a note.
    fingerprints = {r.get("harness_fingerprint") for r in records}
    shared = fingerprints.pop() if len(fingerprints) == 1 else None
    c.check(shared is not None and all(r.get("harness") == HARNESS_NAME
                                       for r in records),
            f"all records share one harness fingerprint ({shared})")
    c.check(all(r.get("base_harness_fingerprint") for r in records),
            "base harness fingerprint present on every record")
    if shared and shared != info["fingerprint"]:
        print(f"  note  run fingerprint {shared} predates current code "
              f"{info['fingerprint']} (post-run code edit) — identity only, "
              f"not an acceptance failure")

    # --- per-record ---------------------------------------------------------
    for rec in records:
        sid = rec.get("sample_id")
        c.check(rec.get("status") in (parse.OK, parse.OUT_OF_FRAME,
                                     parse.NO_PICK_POINT, parse.EMPTY_RESPONSE,
                                     parse.REFUSED, parse.PARSE_ERROR,
                                     parse.SCHEMA_INVALID, parse.EXEC_ERROR),
                f"[{sid}] known status ({rec.get('status')})")
        c.check(not rec.get("tool_attempts"),
                f"[{sid}] no tool attempts")
        c.check(bool(rec.get("base_harness_fingerprint")),
                f"[{sid}] base harness fingerprint present")
        if rec.get("status") == parse.OK:
            c.check(rec.get("schema_valid") is True, f"[{sid}] schema valid")
            c.check(bool(rec.get("prompt")), f"[{sid}] prompt recorded")
        overlay = rec.get("overlay")
        c.check(bool(overlay) and (run_dir / overlay).exists(),
                f"[{sid}] overlay rendered" if overlay else f"[{sid}] overlay rendered")

    # --- frames + GT ---------------------------------------------------------
    by_sample = {s["sample_id"]: s for s in manifest["samples"]}
    for sid, sample in by_sample.items():
        import hashlib
        frame = HERE / sample["image"]
        c.check(frame.exists(), f"[{sid}] delivered frame exists")
        if frame.exists():
            digest = hashlib.sha256(frame.read_bytes()).hexdigest()
            c.check(digest == sample["frame_sha256"],
                    f"[{sid}] frame sha256 matches manifest snapshot")
            h, w = sample["image_shape_hw"]
            c.check(max(h, w) <= MAX_SAFE_EDGE,
                    f"[{sid}] within 1280 px ceiling ({w}x{h})")

    # --- scorer reproducibility ----------------------------------------------
    for rec in records:
        sid = rec["sample_id"]
        if rec.get("status") != parse.OK:
            continue
        bm = det_scoring.score_image(rec["inventory"], rec["gt_boxes"],
                                     rec["gt_areas"], rec["frame_w"],
                                     rec["frame_h"])
        same = all(bm.get(k) == rec["box_metrics"].get(k) for k in
                   ("tp_50", "fp_50", "fn_50", "tp_25", "fp_25", "fn_25",
                    "tp_75", "fp_75", "fn_75", "n_pred", "count_error",
                    "mean_matched_iou_50"))
        c.check(same, f"[{sid}] box scorer reproduces recorded numbers")
        if rec.get("mask_scored"):
            sample = by_sample[sid]
            try:
                masks = load_gt_masks(sample)
                mm = seg_scoring.score_masks_image(rec["inventory"], masks,
                                                   mask_gt_areas(masks),
                                                   rec["frame_w"], rec["frame_h"])
                same = all(mm.get(k) == rec.get(k) for k in
                           ("tp_50", "fp_50", "fn_50", "tp_25", "fp_25",
                            "fn_25", "tp_75", "fp_75", "fn_75", "n_pred",
                            "count_error", "mean_matched_iou_50"))
                c.check(same, f"[{sid}] mask scorer reproduces recorded numbers")
            except Exception as exc:
                c.check(False, f"[{sid}] mask GT re-load failed: {exc}")
        else:
            sample = by_sample[sid]
            if sample["gt_masks"]["kind"] != "none":
                # Two legitimate ways to be mask-unscored on a mask source:
                # the format asks no polygons (box2), or GT failed to load.
                asks_polygons = rec.get("format", "full9") != "box2"
                c.check((not asks_polygons) or rec.get("gt_mask_error") is not None
                        or not rec.get("gt_masks_loaded"),
                        f"[{sid}] unscored mask source recorded as such "
                        f"(format={rec.get('format', 'full9')})")

    # --- summary consistency ---------------------------------------------------
    import csv
    with (run_dir / "summary_by_source.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    c.check(len(rows) == len({r["source"] for r in records}),
            f"summary_by_source covers every source ({len(rows)} rows)")
    for row in rows:
        src = row["source"]
        ok = [r for r in records if r["source"] == src
              and r["status"] == parse.OK]
        tp = sum(r["box_metrics"]["tp_50"] for r in ok)
        fp = sum(r["box_metrics"]["fp_50"] for r in ok)
        fn = sum(r["box_metrics"]["fn_50"] for r in ok)
        _, _, f1 = det_scoring.precision_recall_f1(tp, fp, fn)
        got, want = row["box_f1_50"], f1
        same = ((got in ("", None) and want is None)
                or (got not in ("", None) and want is not None
                    and abs(float(got) - want) < 5e-5))
        c.check(same, f"[{src}] summary box F1@0.5 matches records "
                      f"({got} vs {want})")

    print(f"\n{c.failures} failure(s) out of {c.checks} check(s)")
    if c.failures:
        print("FAIL")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
