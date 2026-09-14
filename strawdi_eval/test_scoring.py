#!/usr/bin/env python3
"""Self-contained correctness checks for the StrawDI scorer.

Run directly (``python3 strawdi_eval/test_scoring.py``); exits non-zero on any
failure. Deliberately dependency-free beyond the repo itself.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

import vlm_eval.run_vlm_eval as vre  # noqa: E402
from strawdi_eval import build_manifest  # noqa: E402
from strawdi_eval.lib import scoring  # noqa: E402

FAILURES: list[str] = []


def check(condition: bool, label: str) -> None:
    status = "ok  " if condition else "FAIL"
    print(f"  {status}  {label}")
    if not condition:
        FAILURES.append(label)


def test_iou_matches_base() -> None:
    rng = random.Random(0)
    agree = True
    for _ in range(2000):
        boxes = []
        for _ in range(2):
            x1, y1 = rng.uniform(0, 500), rng.uniform(0, 500)
            w, h = rng.uniform(-10, 200), rng.uniform(-10, 200)
            boxes.append([x1, y1, x1 + w, y1 + h])
        if scoring.iou(*boxes) != vre._iou(*boxes):
            agree = False
            break
    check(agree, "scoring.iou == run_vlm_eval._iou on 2000 random box pairs")


def test_center_hit() -> None:
    gt = [10, 10, 50, 50]
    check(scoring.center_hit([20, 20, 30, 30], gt), "gt contains pred centre")
    check(scoring.center_hit([0, 0, 100, 100], gt), "pred contains gt centre")
    check(not scoring.center_hit([60, 60, 90, 90], gt), "disjoint boxes miss")
    check(scoring.center_hit([10, 10, 50, 50], gt), "identical boxes hit (edges)")


def test_greedy_matching() -> None:
    gts = [{"index": 0, "bbox": [10, 10, 60, 60]},
           {"index": 1, "bbox": [200, 200, 260, 260]}]
    preds = [
        {"index": 0, "bbox": [12, 12, 58, 58], "confidence": 50},   # overlaps gt0 well
        {"index": 1, "bbox": [11, 11, 59, 59], "confidence": 99},   # claims gt0 first
        {"index": 2, "bbox": [202, 202, 258, 258], "confidence": 10},
    ]
    matched = scoring.greedy_match(preds, gts, "iou", 0.5)
    pairs = {(m["pred_index"], m["gt_index"]) for m in matched["matches"]}
    check(pairs == {(1, 0), (2, 1)},
          "higher-confidence pred claims its best GT first")
    check(matched["fp"] == 1 and matched["fp_indices"] == [0],
          "lower-confidence duplicate becomes the FP")
    check(matched["tp"] == 2 and matched["fn"] == 0, "counts consistent")
    check(matched["mean_matched_iou"] is not None, "matched IoU recorded")

    # tie on confidence -> list order decides; tie on IoU -> lowest gt index
    preds_tie = [
        {"index": 0, "bbox": [10, 10, 60, 60], "confidence": 90},
        {"index": 1, "bbox": [10, 10, 60, 60], "confidence": 90},
    ]
    tie = scoring.greedy_match(preds_tie, gts, "iou", 0.5)
    check({(m["pred_index"], m["gt_index"]) for m in tie["matches"]} == {(0, 0)},
          "confidence tie broken by list order (first pred wins gt0)")

    # invalid predictions never match
    preds_bad = [{"index": 0, "bbox": [-5, 10, 60, 60], "confidence": 99,
                  "valid": False}]
    bad = scoring.greedy_match(preds_bad, gts, "iou", 0.5)
    check(bad["tp"] == 0 and bad["fp"] == 1, "out-of-frame box is an FP, never a match")


def test_score_image() -> None:
    gt_boxes = [[10, 10, 60, 60], [200, 200, 260, 260]]
    gt_areas = [2500, 3600]
    inventory = [
        {"bbox": [12, 12, 58, 58], "confidence_pct": 90},
        {"bbox": [202, 202, 258, 258], "confidence_pct": 80},
        {"bbox": [700, 700, 900, 900], "confidence_pct": 70},   # out of frame
    ]
    result = scoring.score_image(inventory, gt_boxes, gt_areas, 1008, 756)
    check(result["tp_50"] == 2 and result["fp_50"] == 1 and result["fn_50"] == 0,
          "score_image: 2 TP, 1 FP (out-of-frame), 0 FN")
    check(result["tp_50"] + result["fp_50"] == result["n_pred"], "n_pred accounting")
    check(result["n_boxes_out_of_frame"] == 1, "out-of-frame box counted")
    check(result["count_error"] == 1, "count_error = pred - gt")
    check(result["fn_gt_areas_50"] == [], "no misses -> no missed areas")

    empty = scoring.score_image([], gt_boxes, gt_areas, 1008, 756)
    check(empty["tp_50"] == 0 and empty["fp_50"] == 0 and empty["fn_50"] == 2,
          "zero-fruit inventory is valid and scores as all-missed")
    check(empty["fn_gt_areas_50"] == [2500, 3600], "missed areas recorded")

    # a partly-overlapping box (IoU = 2500/6000 = 0.4167): matched at 0.25,
    # not at 0.5 — matching is recomputed independently per threshold
    half = scoring.score_image([{"bbox": [10, 10, 60, 130], "confidence_pct": 50}],
                               [[10, 10, 60, 60]], [2500], 1008, 756)
    check(half["tp_25"] == 1 and half["tp_50"] == 0,
          "per-threshold matching recomputed independently")
    # an exactly-0.5 overlap must match at 0.5 (>= threshold)
    edge = scoring.score_image([{"bbox": [10, 10, 60, 110], "confidence_pct": 50}],
                               [[10, 10, 60, 60]], [2500], 1008, 756)
    check(edge["tp_50"] == 1, "IoU exactly at the threshold still matches")

    # semantics gap: whole-fruit pred around a small visible-surface GT
    gap = scoring.score_image([{"bbox": [8, 8, 70, 70], "confidence_pct": 90}],
                              [[20, 20, 50, 50]], [900], 1008, 756)
    check(gap["tp_center"] == 1 and gap["tp_50"] == 0,
          "centre criterion catches what IoU punishes (bbox-semantics gap)")


def test_precision_recall_f1() -> None:
    p, r, f = scoring.precision_recall_f1(3, 1, 2)
    check((p, r, f) == (0.75, 0.6, round(2 * 0.75 * 0.6 / 1.35, 4)),
          "P/R/F1 arithmetic")
    check(scoring.precision_recall_f1(0, 0, 5) == (None, 0.0, None),
          "no predictions -> precision undefined, recall 0")


def test_average_precision() -> None:
    perfect = scoring.average_precision([(0.9, True), (0.8, True)], 2)
    check(perfect == 1.0, "perfect predictor -> AP 1.0")
    all_wrong = scoring.average_precision([(0.9, False), (0.8, False)], 2)
    check(all_wrong == 0.0, "all false positives -> AP 0.0")
    none = scoring.average_precision([], 2)
    check(none == 0.0, "no predictions with GT present -> AP 0.0")
    check(scoring.average_precision([(1.0, True)], 0) is None,
          "no GT -> AP undefined (None)")
    # classic textbook curve: 4 preds, ranks [T, F, T, T] over 3 GT.
    # precision@k = 1, 1/2, 2/3, 3/4; recall@k = 1/3, 1/3, 2/3, 1.
    # All-points interpolation takes, at each recall step, the best precision
    # at that recall or beyond (suffix max): 1, -, 3/4, 3/4.
    # AP = (1/3)*1 + (1/3)*(3/4) + (1/3)*(3/4) = 0.8333
    ap = scoring.average_precision([(0.9, True), (0.8, False),
                                    (0.7, True), (0.6, True)], 3)
    check(ap == round(1 / 3 + (1 / 3) * 0.75 + (1 / 3) * 0.75, 4),
          "all-points interpolation matches hand computation")


def test_ap_metrics() -> None:
    per_image = [
        {"sample_id": "a",
         "preds": scoring.prepare_predictions(
             [{"bbox": [10, 10, 60, 60], "confidence_pct": 90}], 1008, 756),
         "gt_boxes": [[10, 10, 60, 60]]},
        {"sample_id": "b",
         "preds": scoring.prepare_predictions(
             [{"bbox": [20, 20, 40, 40], "confidence_pct": 80}], 1008, 756),
         "gt_boxes": [[20, 20, 40, 40]]},
    ]
    aps = scoring.ap_metrics(per_image)
    check(aps["ap_50"] == 1.0 and aps["map_50_95"] == 1.0,
          "perfect batch -> AP@50 and mAP@[.50:.95] = 1.0")
    check(len(aps["per_threshold"]) == 10, "ten AP thresholds computed")


def test_derive_instances() -> None:
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[10:22, 30:45] = 1     # 12x15
    mask[50:53, 50:52] = 2     # 3x2
    instances = build_manifest.derive_instances(mask)
    check([i["instance_id"] for i in instances] == [1, 2], "instance ids sorted")
    check(instances[0]["bbox_xyxy"] == [30, 10, 45, 22],
          "half-open box: extent == pixel spread (x2 = xmax+1)")
    check(instances[0]["area_px"] == 12 * 15, "area = pixel count")
    check(instances[1]["area_px"] == 6, "small instance kept (no area filter)")
    check(build_manifest.derive_instances(np.zeros((10, 10), np.uint8)) == [],
          "empty mask -> no instances")


def main() -> int:
    print("scoring self-checks")
    for test in (test_iou_matches_base, test_center_hit, test_greedy_matching,
                 test_score_image, test_precision_recall_f1,
                 test_average_precision, test_ap_metrics, test_derive_instances):
        print(f"\n[{test.__name__}]")
        test()
    print(f"\n{len(FAILURES)} failure(s)")
    if FAILURES:
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
