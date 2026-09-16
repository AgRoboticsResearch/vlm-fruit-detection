#!/usr/bin/env python3
"""Self-contained correctness checks for the StrawDI seg scorer.

Run directly (``python3 strawdi_eval/seg/test_seg_scoring.py``); exits non-zero
on any failure. Deliberately dependency-free beyond the repo itself; the mask
analogues of ``strawdi_eval/test_scoring.py``'s matching checks live here with
the same hand-computable style.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # strawdi_eval/seg
REPO = HERE.parent.parent                       # repo root
sys.path.insert(0, str(REPO))

from strawdi_eval.seg.lib import seg_scoring  # noqa: E402

FAILURES: list[str] = []


def check(condition: bool, label: str) -> None:
    status = "ok  " if condition else "FAIL"
    print(f"  {status}  {label}")
    if not condition:
        FAILURES.append(label)


def rect_mask(w: int, h: int, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Boolean HxW mask, inclusive pixel ranges (what a rect polygon rasterises to)."""
    m = np.zeros((h, w), dtype=bool)
    m[y1:y2 + 1, x1:x2 + 1] = True
    return m


RECT = [[10, 10], [49, 10], [49, 29], [10, 29]]   # 40 x 20 px inclusive
W, H = 100, 80


def test_normalise_polygon() -> None:
    check(seg_scoring.normalise_polygon([[1.2, 3.7], ["4", 5], (6, 7), [8, 9]])
          == [[1, 4], [4, 5], [6, 7], [8, 9]],
          "normalise_polygon rounds floats and accepts numeric strings")
    check(seg_scoring.normalise_polygon([[1, 2], [3, 4]]) is None,
          "fewer than 3 vertices -> None")
    check(seg_scoring.normalise_polygon([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) is None,
          "non-pair points -> None")
    check(seg_scoring.normalise_polygon([[1, "x"], [2, 3], [4, 5]]) is None,
          "non-numeric coordinate -> None")
    check(seg_scoring.normalise_polygon("nope") is None, "non-list -> None")


def test_polygon_bbox() -> None:
    check(seg_scoring.polygon_bbox(RECT) == [10, 10, 50, 30],
          "polygon_bbox is half-open (max + 1), derive_instances convention")


def test_rasterise() -> None:
    mask = seg_scoring.rasterise_polygon(RECT, W, H)
    check(mask.dtype == bool and mask.shape == (H, W), "raster is bool HxW")
    check(int(mask.sum()) == 40 * 20,
          f"rectangle rasterises to its pixel count ({int(mask.sum())} == 800)")


def test_classify() -> None:
    flag, mask = seg_scoring.classify_and_rasterise(RECT, W, H)
    check(flag == "ok" and mask is not None and mask.any(), "in-frame polygon ok")
    flag, mask = seg_scoring.classify_and_rasterise(
        [[0, 0], [99, 0], [99, 79]], W, H)
    check(flag == "ok", "vertex at the far corner (W-1, H-1) is in-frame")
    flag, mask = seg_scoring.classify_and_rasterise(
        [[0, 0], [W, 0], [50, 50]], W, H)
    check(flag == "out_of_frame" and mask is None,
          "vertex at x == W is out_of_frame, never clamped")
    flag, mask = seg_scoring.classify_and_rasterise([[0, 0], [50, 50]], W, H)
    check(flag == "degenerate", "<3 vertices is degenerate")
    flag, mask = seg_scoring.classify_and_rasterise(None, W, H)
    check(flag == "degenerate", "None polygon is degenerate")


def test_mask_iou() -> None:
    a = rect_mask(W, H, 0, 0, 99, 99)
    check(seg_scoring.mask_iou(a, a) == 1.0, "identical masks -> IoU 1")
    b = rect_mask(W, H, 0, 0, 9, 9)
    c = rect_mask(W, H, 50, 50, 99, 99)
    check(seg_scoring.mask_iou(b, c) == 0.0, "disjoint masks -> IoU 0")
    # b = 10x10 at origin, d = 10x10 shifted by 5: inter 5x10=50, union 150
    d = rect_mask(W, H, 5, 0, 14, 9)
    check(abs(seg_scoring.mask_iou(b, d) - 50 / 150) < 1e-9,
          "half-overlap masks -> IoU 1/3")
    empty = np.zeros((H, W), dtype=bool)
    check(seg_scoring.mask_iou(empty, empty) == 0.0, "two empty masks -> IoU 0")


def inventory(polygon, bbox=None, confidence: int | None = 90, **extra) -> list[dict]:
    entry = {"polygon": polygon, "bbox": bbox or seg_scoring.polygon_bbox(polygon),
             "confidence_pct": confidence}
    entry.update(extra)
    return [entry]


def test_greedy_matching() -> None:
    gts = [rect_mask(W, H, 10, 10, 69, 69), rect_mask(W, H, 80, 10, 99, 69)]
    preds = seg_scoring.prepare_predictions_masks(
        inventory([[10, 10], [69, 10], [69, 69], [10, 69]], confidence=50)
        + inventory([[80, 10], [99, 10], [99, 69], [80, 69]], confidence=10)
        + inventory([[12, 12], [67, 12], [67, 67], [12, 67]], confidence=99),
        W, H)
    matched = seg_scoring.greedy_match_masks(preds, gts, 0.5)
    pairs = {(m["pred_index"], m["gt_index"]) for m in matched["matches"]}
    check(pairs == {(2, 0), (1, 1)},
          "higher-confidence pred claims its best GT first")
    check(matched["fp"] == 1 and matched["fp_indices"] == [0],
          "lower-confidence duplicate becomes the FP")
    check(matched["tp"] == 2 and matched["fn"] == 0, "counts consistent")

    # tie on confidence -> list order decides; tie on IoU -> lowest gt index
    preds_tie = seg_scoring.prepare_predictions_masks(
        inventory([[10, 10], [69, 10], [69, 69], [10, 69]], confidence=90)
        + inventory([[10, 10], [69, 10], [69, 69], [10, 69]], confidence=90),
        W, H)
    tie = seg_scoring.greedy_match_masks(preds_tie, gts, 0.5)
    check({(m["pred_index"], m["gt_index"]) for m in tie["matches"]} == {(0, 0)},
          "confidence tie -> earlier inventory index wins the GT")
    check(tie["fp_indices"] == [1], "the twin prediction is the FP")

    # missing confidence ranks last, flagged
    preds_missing = seg_scoring.prepare_predictions_masks(
        inventory([[10, 10], [69, 10], [69, 69], [10, 69]], confidence=None)
        + inventory([[80, 10], [99, 10], [99, 69], [80, 69]], confidence=5),
        W, H)
    check(preds_missing[0]["confidence_missing"] is True
          and preds_missing[0]["confidence"] is None,
          "non-numeric confidence flagged and treated as missing")
    order = [p["index"] for p in sorted(preds_missing, key=seg_scoring._rank_key)]
    check(order == [1, 0], "missing confidence ranks below a numeric one")

    # out-of-frame polygon never matches and is an FP
    preds_oof = seg_scoring.prepare_predictions_masks(
        inventory([[0, 0], [W, 0], [50, 50]]), W, H)
    check(preds_oof[0]["flag"] == "out_of_frame"
          and preds_oof[0]["valid"] is False, "out-of-frame polygon marked invalid")
    oof = seg_scoring.greedy_match_masks(preds_oof, gts[:1], 0.25)
    check(oof["tp"] == 0 and oof["fp"] == 1 and oof["fn"] == 1,
          "invalid prediction is an FP, its would-be GT an FN")


def test_score_masks_image() -> None:
    gt_masks = [rect_mask(W, H, 10, 10, 69, 69), rect_mask(W, H, 80, 10, 99, 69)]
    areas = [m.sum() for m in gt_masks]

    # exact polygons -> perfect detection
    exact = (inventory([[10, 10], [69, 10], [69, 69], [10, 69]], bbox=[10, 10, 70, 70])
             + inventory([[80, 10], [99, 10], [99, 69], [80, 69]],
                         bbox=[80, 10, 100, 70]))
    result = seg_scoring.score_masks_image(exact, gt_masks, areas, W, H)
    check(result["tp_50"] == 2 and result["fp_50"] == 0 and result["fn_50"] == 0,
          "exact polygons match every GT at 0.5")
    check(result["mean_matched_iou_50"] == 1.0, "exact match IoU is 1.0")
    check(result["polygon_bbox_iou_mean"] == 1.0,
          "reported bbox agrees with polygon extent")
    check(result["count_error"] == 0, "count error zero")

    # threshold ladder: pred1 covers gt1 only partially (20x36 of 20x60 px ->
    # IoU 0.6), so matching must flip between thresholds
    part = seg_scoring.score_masks_image(
        inventory([[10, 10], [69, 10], [69, 69], [10, 69]])
        + inventory([[80, 10], [99, 10], [99, 45], [80, 45]]),
        gt_masks, areas, W, H)
    check(part["tp_25"] == 2 and part["tp_50"] == 2 and part["tp_75"] == 1,
          "threshold ladder recomputed independently (0.6-IoU pred: TP@.25/.50, not .75)")
    check(part["fn_75"] == 1, "the sub-threshold GT is an FN at 0.75 only")

    # empty inventory is a valid all-missed answer, never zeros-from-failure
    none_left = seg_scoring.score_masks_image([], gt_masks, areas, W, H)
    check(none_left["tp_50"] == 0 and none_left["fn_50"] == 2
          and none_left["count_error"] == -2,
          "zero-fruit answer scores as all-missed with a negative count error")


def test_empty_block() -> None:
    block = seg_scoring.empty_masks()
    check(all(block[key] is None for key in seg_scoring.MASK_KEYS
              + seg_scoring.MASK_DETAIL_KEYS),
          "empty_masks carries None everywhere, never zeros")


def test_ap_metrics_masks() -> None:
    gt_masks = [rect_mask(W, H, 10, 10, 69, 69)]
    perfect = [{
        "preds": seg_scoring.prepare_predictions_masks(
            inventory([[10, 10], [69, 10], [69, 69], [10, 69]], confidence=100), W, H),
        "gt_masks": gt_masks,
    }]
    aps = seg_scoring.ap_metrics_masks(perfect)
    check(aps["ap_50"] == 1.0 and aps["map_50_95"] == 1.0,
          "perfect single prediction -> AP 1.0 at every threshold")
    weak = [{
        "preds": seg_scoring.prepare_predictions_masks(
            inventory([[10, 10], [69, 10], [69, 50], [10, 50]], confidence=100), W, H),
        "gt_masks": gt_masks,
    }]
    # pred covers 60x41 of the 60x60 GT -> IoU = 2460/3600 ~ 0.683: matched at
    # thresholds 0.50-0.65, missed at 0.70+ -> AP@.50 1.0, mAP 4/10
    aps = seg_scoring.ap_metrics_masks(weak)
    check(aps["ap_50"] == 1.0 and aps["map_50_95"] == 0.4,
          "AP tracks the threshold ladder (0.683-IoU pred: AP@.50 1, mAP 0.4)")


def main() -> None:
    tests = [
        test_normalise_polygon,
        test_polygon_bbox,
        test_rasterise,
        test_classify,
        test_mask_iou,
        test_greedy_matching,
        test_score_masks_image,
        test_empty_block,
        test_ap_metrics_masks,
    ]
    for test in tests:
        print(f"{test.__name__}:")
        test()
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)")
        for label in FAILURES:
            print(f"  - {label}")
        sys.exit(1)
    print("all seg scoring checks passed")


if __name__ == "__main__":
    main()
