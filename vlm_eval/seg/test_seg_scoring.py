#!/usr/bin/env python3
"""Self-contained correctness checks for the vlm_seg scorer.

Run directly (``python3 vlm_eval/seg/test_seg_scoring.py``); exits non-zero
on any failure. Deliberately dependency-free beyond the repo itself; the
polygon plumbing checks mirror ``strawdi_eval/seg/test_seg_scoring.py`` (the
semantics are shared verbatim), and the point-distance / nomination checks
are specific to this pipeline — it has no GT masks, so its scoring is the
picking-point cross-check plus polygon diagnostics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # vlm_eval/seg
REPO = HERE.parent.parent                       # repo root
sys.path.insert(0, str(REPO))

from vlm_eval.seg.lib import seg_scoring  # noqa: E402

FAILURES: list[str] = []


def check(condition: bool, label: str) -> None:
    status = "ok  " if condition else "FAIL"
    print(f"  {status}  {label}")
    if not condition:
        FAILURES.append(label)


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
          "polygon_bbox is half-open (max + 1)")


def test_rasterise() -> None:
    mask = seg_scoring.rasterise_polygon(RECT, W, H)
    check(mask.dtype == bool and mask.shape == (H, W), "raster is bool HxW")
    check(int(mask.sum()) == 40 * 20,
          f"rectangle rasterises to its pixel count ({int(mask.sum())} == 800)")


def test_classify() -> None:
    flag, mask = seg_scoring.classify_and_rasterise(RECT, W, H)
    check(flag == "ok" and mask is not None and mask.any(), "in-frame polygon ok")
    flag, _ = seg_scoring.classify_and_rasterise([[0, 0], [99, 0], [99, 79]], W, H)
    check(flag == "ok", "vertex at the far corner (W-1, H-1) is in-frame")
    flag, _ = seg_scoring.classify_and_rasterise([[0, 0], [W, 0], [50, 50]], W, H)
    check(flag == "out_of_frame", "vertex at x == W is out_of_frame, never clamped")
    flag, _ = seg_scoring.classify_and_rasterise([[0, 0], [50, 50]], W, H)
    check(flag == "degenerate", "<3 vertices is degenerate")
    flag, _ = seg_scoring.classify_and_rasterise(None, W, H)
    check(flag == "degenerate", "None polygon is degenerate")


def test_point_polygon_distance() -> None:
    # RECT spans x 10..49, y 10..29 (edges on those lines).
    check(seg_scoring.point_polygon_distance((20, 15), RECT) == 5.0,
          "point inside: distance to the nearest edge (y=10 -> 5)")
    check(seg_scoring.point_polygon_distance((55, 15), RECT) == 6.0,
          "point outside beside an edge: x=49 -> 6")
    check(seg_scoring.point_polygon_distance((10, 10), RECT) == 0.0,
          "point on a vertex -> 0")
    # (60, 40): nearest feature is the corner vertex (49, 29).
    expected = round(float(np.hypot(60 - 49, 40 - 29)), 4)
    check(seg_scoring.point_polygon_distance((60, 40), RECT) == expected,
          "point beyond the corner: distance to the vertex")
    check(seg_scoring.point_polygon_distance((20, 15), None) is None,
          "no polygon -> None")
    check(seg_scoring.point_polygon_distance((20, 15), [[1, 2]]) is None,
          "degenerate polygon -> None")


def test_iou() -> None:
    check(seg_scoring.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0,
          "identical boxes -> IoU 1")
    check(seg_scoring.iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0,
          "disjoint boxes -> IoU 0")
    # half overlap: inter 50, union 150
    check(abs(seg_scoring.iou([0, 0, 10, 10], [5, 0, 15, 10]) - 50 / 150) < 1e-9,
          "half-overlap boxes -> IoU 1/3")


def inventory(polygon, bbox=None, confidence=90, pick=None, **extra) -> list[dict]:
    entry = {"polygon": polygon, "bbox": bbox or seg_scoring.polygon_bbox(polygon),
             "confidence_pct": confidence, "picking_point": pick}
    entry.update(extra)
    return [entry]


def test_polygon_diag_block() -> None:
    diag = seg_scoring.polygon_diag_block(
        inventory(RECT, pick=(20, 12))
        + inventory([[0, 0], [W, 0], [50, 50]]),        # out-of-frame
        W, H)
    check(diag["n_pred"] == 2 and diag["n_polygons_ok"] == 1
          and diag["n_polygons_out_of_frame"] == 1
          and diag["n_polygons_degenerate"] == 0,
          "validity counts separate ok / out-of-frame / degenerate")
    check(diag["mean_vertex_count"] == 3.5, "mean vertex count over both polygons")
    check(diag["polygon_bbox_iou_mean"] == 1.0,
          "bbox agreeing with the polygon extent -> IoU 1")
    # picking point (20, 12): 2 px above the top edge (y=10) -> distance 2.
    check(diag["pick_to_own_polygon_px_mean"] == 2.0,
          "picking point to own outline distance (2 px above the top edge)")

    empty = seg_scoring.polygon_diag_block([], W, H)
    check(empty["n_pred"] == 0 and empty["n_polygons_ok"] == 0
          and empty["mean_vertex_count"] is None,
          "zero-fruit answer: counts 0, averages None (never zeros)")

    coarse = seg_scoring.polygon_diag_block(
        inventory(RECT, bbox=[0, 0, 100, 100]), W, H)
    expected = round(800 / 10000, 4)
    check(abs(coarse["polygon_bbox_iou_mean"] - expected) < 1e-9,
          "extent IoU shrinks when the bbox disagrees with the polygon")


def test_score_gt_frame() -> None:
    gt_uv = (30.0, 10.0)                    # on the top edge of RECT's extent
    rough = [0, 0, 60, 40]
    # Nominee outlines RECT and puts its grasp point 6 px above the GT point.
    inv = inventory(RECT, bbox=[10, 10, 50, 30], pick=(30, 4))
    scored = seg_scoring.score_gt_frame(inv, 0, gt_uv, rough, W, H)
    check(scored["error_px"] == 6.0 and scored["dx_px"] == 0.0
          and scored["dy_px"] == -6.0,
          "nominated picking point scores against GT (6 px straight up)")
    check(scored["polygon_extent_iou_rough"] == round(800 / 2400, 4),
          "polygon extent IoU vs rough box (800 of 2400 px2)")
    check(scored["bbox_iou_rough"] == round(800 / 2400, 4),
          "same extent reported as bbox -> identical rough-box IoU")
    check(scored["gt_to_target_polygon_px"] == 0.0,
          "GT point ON the outline -> distance 0")
    check(scored["polygon_contains_gt"] is True,
          "point on the boundary rasterises inside (PIL includes the edge)")

    inside = seg_scoring.score_gt_frame(
        inventory(RECT, pick=(30, 4)), 0, (30.0, 20.0), rough, W, H)
    check(inside["polygon_contains_gt"] is True,
          "GT point inside the outline raster -> contained")
    check(inside["gt_to_target_polygon_px"] == 9.0,
          "interior point's outline distance is to the nearest edge (y=29 -> 9)")

    # No usable nomination: the answer is kept, the numbers are None.
    none_target = seg_scoring.score_gt_frame(inv, -1, gt_uv, rough, W, H)
    check(all(none_target[k] is None for k in seg_scoring.SCORED_KEYS),
          "no nomination -> every scored field None, never zeros")
    null_pick = inventory(RECT, pick=None)
    no_point = seg_scoring.score_gt_frame(null_pick, 0, gt_uv, rough, W, H)
    check(no_point["error_px"] is None and no_point["dy_px"] is None
          and no_point["polygon_extent_iou_rough"] is not None,
          "nominee with hidden peduncle: no point number, extent still scored")

    # Unlabelled frame semantics are the caller's (scored_block); the raw
    # scorer must still work with rough_box=None.
    no_rough = seg_scoring.score_gt_frame(inv, 0, gt_uv, None, W, H)
    check(no_rough["error_px"] == 6.0 and no_rough["polygon_extent_iou_rough"] is None,
          "no rough box: point scoring unaffected, rough-box IoUs None")


def test_empty_blocks() -> None:
    diag = seg_scoring.empty_diag()
    check(all(diag[key] is None for key in seg_scoring.DIAG_KEYS),
          "empty_diag carries None everywhere, never zeros")
    scored = seg_scoring.empty_scored()
    check(all(scored[key] is None for key in seg_scoring.SCORED_KEYS),
          "empty_scored carries None everywhere, never zeros")


def main() -> None:
    tests = [
        test_normalise_polygon,
        test_polygon_bbox,
        test_rasterise,
        test_classify,
        test_point_polygon_distance,
        test_iou,
        test_polygon_diag_block,
        test_score_gt_frame,
        test_empty_blocks,
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
    print("all vlm_seg scoring checks passed")


if __name__ == "__main__":
    main()
