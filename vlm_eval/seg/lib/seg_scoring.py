#!/usr/bin/env python3
"""Polygon diagnostics + nominated-target scoring for the vlm_eval seg pipeline.

No mask ground truth exists on any vlm_eval source — the SROI frames carry
ONE recomputed picking point each (plus the upstream rough 60x60 box derived
from it), and the shunba scenes are unlabelled — so this module deliberately
contains **no mask-vs-mask matching**. The StrawDI pipeline
(``strawdi_eval/seg/``) is where polygons are scored as segmentation against
annotated instance masks. What happens here instead:

* polygon plumbing with semantics shared verbatim with that pipeline
  (``normalise_polygon`` / ``polygon_bbox`` / ``classify_and_rasterise`` /
  ``rasterise_polygon`` — same rules, so answers stay comparable);
* per-image diagnostics computable from the answer ALONE, valid on every
  frame including the unlabelled ones: polygon validity counts, vertex
  budget, polygon-vs-bbox extent IoU (internal consistency — the model that
  boxes the WHOLE fruit should outline roughly that extent's visible part),
  and each fruit's picking_point distance to its own polygon outline (the
  grasp point sits on the peduncle just above the fruit, so it should land
  at or just outside the outline, a few px away at most);
* on ground-truthed frames ONLY: the nominated target scored exactly like
  ``full_detection`` — picking-point error / dx / dy against the recomputed
  GT point, plus the polygon extent's IoU against the same upstream rough
  box behind the box pipeline's approximate bbox-IoU, plus the GT point's
  distance to the target polygon's outline (the peduncle grasp point should
  sit a few px ABOVE the fruit body's outline, not inside it — a diagnostic,
  never a headline).

Failure statuses carry None everywhere, never zeros. Determinism: no
randomness, no floating-point tie-breaking — re-running over the same stored
inventory reproduces every number (the verifier asserts this).
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

# Same ladder as the base harness's PCK thresholds.
PCK_THRESHOLDS = (5, 10, 20)

# Keys every parsed record's polygon-diagnostic block carries; failure
# records carry None for all of them.
DIAG_KEYS = (
    "n_pred",
    "n_polygons_ok", "n_polygons_out_of_frame", "n_polygons_degenerate",
    "mean_vertex_count",
    "polygon_bbox_iou_mean",
    "pick_to_own_polygon_px_mean",
)

# Keys a GROUND-TRUTHED record's scored block carries (None on unlabelled
# frames and on every failure status — I7: unlabelled runs carry no scored
# value at all).
SCORED_KEYS = (
    "error_px", "dx_px", "dy_px",
    "bbox_iou_rough", "polygon_extent_iou_rough",
    "polygon_contains_gt", "gt_to_target_polygon_px",
)


def empty_diag() -> dict:
    """Diagnostic block for a run whose answer never parsed (never zeros)."""
    return {key: None for key in DIAG_KEYS}


def empty_scored() -> dict:
    """Scored block for an unlabelled frame or an unparsed answer."""
    return {key: None for key in SCORED_KEYS}


# ---------------------------------------------------------------------------
# Polygons (semantics shared verbatim with strawdi_eval/seg/lib/seg_scoring.py)
# ---------------------------------------------------------------------------

def normalise_polygon(value) -> list[list[int]] | None:
    """Coerce a parsed polygon into ``[[x, y], ...]`` int pairs, or None.

    Same leniency as the base harness's ``parse.normalise_box``: numeric
    strings and nested lists/tuples are accepted; anything non-numeric or not
    point-shaped yields None (the scorer then treats the prediction as
    degenerate).
    """
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    points = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            return None
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        points.append([int(round(x)), int(round(y))])
    return points


def polygon_bbox(polygon) -> list[int] | None:
    """Half-open tight box of a polygon's vertices."""
    if not polygon:
        return None
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return [min(xs), min(ys), max(xs) + 1, max(ys) + 1]


def rasterise_polygon(polygon: list[list[int]], frame_w: int,
                      frame_h: int) -> np.ndarray:
    """Fill a polygon into a bool HxW mask (PIL polygon semantics)."""
    canvas = Image.new("L", (frame_w, frame_h), 0)
    ImageDraw.Draw(canvas).polygon([tuple(p) for p in polygon], fill=1)
    return np.array(canvas, dtype=bool)


def classify_and_rasterise(polygon: list[list[int]] | None, frame_w: int,
                           frame_h: int) -> tuple[str, np.ndarray | None]:
    """Per-polygon validity + rasterisation: 'ok' | 'out_of_frame' | 'degenerate'.

    Coordinate rule mirrors the box scorers: any vertex outside
    ``[0, frame_w-1] x [0, frame_h-1]`` is out_of_frame (never clamped, never
    matched). 'ok' polygons are rasterised to a bool mask.
    """
    if polygon is None or len(polygon) < 3:
        return "degenerate", None
    for x, y in polygon:
        if x < 0 or y < 0 or x > frame_w - 1 or y > frame_h - 1:
            return "out_of_frame", None
    mask = rasterise_polygon(polygon, frame_w, frame_h)
    if not mask.any():
        return "degenerate", None
    return "ok", mask


def point_polygon_distance(point, polygon) -> float | None:
    """Euclidean distance from a point to a polygon's OUTLINE (edges).

    Well-defined for points inside and outside alike: it is the distance to
    the nearest boundary segment, not "0 if inside". A picking point that
    sits on the peduncle just above the fruit should land a few px from its
    own fruit's outline; the GT grasp point likewise a few px above the
    target's outline.
    """
    if polygon is None or len(polygon) < 2:
        return None
    px, py = float(point[0]), float(point[1])
    best: float | None = None
    for (x1, y1), (x2, y2) in zip(polygon, list(polygon[1:]) + [polygon[0]]):
        # Standard point-to-segment distance, clamped to the segment.
        dx, dy = x2 - x1, y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            distance = float(np.hypot(px - x1, py - y1))
        else:
            t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))
            distance = float(np.hypot(px - (x1 + t * dx), py - (y1 + t * dy)))
        if best is None or distance < best:
            best = distance
    return round(best, 4) if best is not None else None


def iou(a, b) -> float:
    """IoU of two [x1, y1, x2, y2] boxes (same math as the base harness's)."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _point(value) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    return (x, y) if (np.isfinite(x) and np.isfinite(y)) else None


def _box4(value) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


# ---------------------------------------------------------------------------
# Per-image blocks
# ---------------------------------------------------------------------------

def polygon_diag_block(inventory, frame_w: int, frame_h: int) -> dict:
    """Polygon diagnostics from the answer ALONE (valid on unlabelled frames).

    A parsed inventory reporting zero fruit is a valid answer: n_pred 0 and
    None means (not zeros) for the averages.
    """
    entries = []
    for entry in inventory or []:
        if not isinstance(entry, dict):
            entries.append({"polygon": None, "bbox": None, "pick": None})
            continue
        polygon = normalise_polygon(entry.get("polygon"))
        flag, _ = classify_and_rasterise(polygon, frame_w, frame_h)
        entries.append({"polygon": polygon, "flag": flag,
                        "bbox": _box4(entry.get("bbox")),
                        "pick": _point(entry.get("picking_point"))})

    block: dict = {
        "n_pred": len(entries),
        "n_polygons_ok": sum(1 for e in entries if e["flag"] == "ok"),
        "n_polygons_out_of_frame": sum(1 for e in entries
                                       if e["flag"] == "out_of_frame"),
        "n_polygons_degenerate": sum(1 for e in entries
                                     if e["flag"] == "degenerate"),
    }
    vertex_counts = [len(e["polygon"]) for e in entries
                     if e["polygon"] is not None]
    block["mean_vertex_count"] = (round(sum(vertex_counts) / len(vertex_counts), 2)
                                  if vertex_counts else None)

    # Internal consistency: does the reported bbox agree with the polygon's
    # extent? Pure diagnostic (never a gate): a model that boxes the WHOLE
    # fruit but outlines only the visible surface SHOULD disagree somewhat on
    # occluded fruit.
    extent_ious = []
    for entry in entries:
        derived = polygon_bbox(entry["polygon"])
        if derived is not None and entry["bbox"] is not None:
            extent_ious.append(iou(entry["bbox"], [float(v) for v in derived]))
    block["polygon_bbox_iou_mean"] = (round(sum(extent_ious) / len(extent_ious), 4)
                                      if extent_ious else None)

    # Each fruit's picking point vs its own outline: the grasp point sits on
    # the peduncle just above the calyx, i.e. at or a few px outside the
    # fruit-body outline — large values mean the answer is geometrically
    # incoherent (point filed against another fruit's polygon).
    pick_distances = []
    for entry in entries:
        if entry["flag"] == "ok" and entry["pick"] is not None:
            distance = point_polygon_distance(entry["pick"], entry["polygon"])
            if distance is not None:
                pick_distances.append(distance)
    block["pick_to_own_polygon_px_mean"] = (
        round(sum(pick_distances) / len(pick_distances), 4)
        if pick_distances else None)
    return block


def score_gt_frame(inventory, target_index, gt_uv, rough_box,
                   frame_w: int, frame_h: int) -> dict:
    """Score the nominated target on a ground-truthed frame.

    ``full_detection``'s nomination cross-check, plus the polygon extent
    against the same upstream rough box behind the approximate bbox-IoU, plus
    the GT-point-to-outline diagnostic. A parsed inventory that nominates no
    usable target keeps its answer but carries no number (None, never zero).
    """
    block = empty_scored()
    entries = list(inventory or [])
    target = None
    if isinstance(target_index, int) and 0 <= target_index < len(entries) \
            and isinstance(entries[target_index], dict):
        target = entries[target_index]
    if target is None:
        return block

    if rough_box is not None:
        rough = [float(v) for v in rough_box]
        box = _box4(target.get("bbox"))
        if box is not None:
            block["bbox_iou_rough"] = round(iou(box, rough), 4)
        polygon = normalise_polygon(target.get("polygon"))
        flag, mask = classify_and_rasterise(polygon, frame_w, frame_h)
        extent = polygon_bbox(polygon) if flag == "ok" else None
        if flag == "ok" and mask is not None and extent is not None:
            extent = [float(v) for v in extent]
            block["polygon_extent_iou_rough"] = round(iou(extent, rough), 4)
            if gt_uv is not None:
                block["gt_to_target_polygon_px"] = point_polygon_distance(
                    gt_uv, polygon)
                gx, gy = int(round(gt_uv[0])), int(round(gt_uv[1]))
                if 0 <= gx < frame_w and 0 <= gy < frame_h:
                    block["polygon_contains_gt"] = bool(mask[gy, gx])

    point = _point(target.get("picking_point"))
    if point is not None and gt_uv is not None:
        dx, dy = point[0] - gt_uv[0], point[1] - gt_uv[1]
        block["dx_px"], block["dy_px"] = round(dx, 4), round(dy, 4)
        block["error_px"] = round(float(np.hypot(dx, dy)), 4)
    return block
