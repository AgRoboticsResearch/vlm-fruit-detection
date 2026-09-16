#!/usr/bin/env python3
"""Multi-instance segmentation scoring for the StrawDI seg pipeline.

Pure functions plus polygon rasterisation (no I/O, no model calls). The
matching contract is the detection scorer's, transplanted onto masks:

* predictions are ranked ``confidence_pct`` descending (missing/non-numeric
  confidence ranks last), then by their position in the inventory;
* each prediction greedily claims the unmatched ground-truth MASK with the
  highest mask IoU (ties -> lowest ground-truth index) and matches when the
  IoU >= the threshold; matching is recomputed independently at every
  threshold;
* a polygon with a vertex outside ``[0, W) x [0, H)`` or with an empty
  rasterisation is excluded from matching and counted as a false positive
  (it still counts toward ``n_pred``) — the exact analogue of the box
  scorer's out-of-frame/degenerate rules;
* the polygon asks for the VISIBLE surface and so does the StrawDI ground
  truth, so mask IoU carries no semantics gap: it is the pipeline's primary
  metric.

A polygon with <3 vertices cannot close and is degenerate by construction.
Collinear "sliver" polygons (a line walked back and forth) DO rasterise to a
thin line of pixels; they are left valid and simply lose every IoU contest —
visible in the overlay, not hidden by a flag.

Determinism: ties in confidence are broken by list index, ties in IoU by
ground-truth index, so a re-run of the scorer over the same records and the
same label PNG reproduces every number — which the verifier actively asserts.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from strawdi_eval.lib import scoring as det_scoring

IOU_THRESHOLDS = det_scoring.IOU_THRESHOLDS   # (0.25, 0.5, 0.75) — same ladder
AP_THRESHOLDS = det_scoring.AP_THRESHOLDS

# Keys every scored record carries (mask semantics); failure records carry
# None for all of them. Deliberately named like DETECTION_KEYS so aggregate
# code and CSVs keep their shape.
MASK_KEYS = (
    "n_gt", "n_pred", "count_error",
    "tp_25", "fp_25", "fn_25",
    "tp_50", "fp_50", "fn_50", "mean_matched_iou_50",
    "tp_75", "fp_75", "fn_75",
    "n_polygons_out_of_frame", "n_polygons_degenerate",
    "polygon_bbox_iou_mean",
)

MASK_DETAIL_KEYS = ("matches_50", "fp_pred_indices_50", "fn_gt_indices_50",
                    "fn_gt_areas_50")


def empty_masks() -> dict:
    """Mask block for a run whose answer never parsed (never zeros)."""
    return {key: None for key in MASK_KEYS + MASK_DETAIL_KEYS}


# ---------------------------------------------------------------------------
# Polygons
# ---------------------------------------------------------------------------

def normalise_polygon(value) -> list[list[int]] | None:
    """Coerce a parsed polygon into ``[[x, y], ...]`` int pairs, or None.

    Same leniency as ``parse.normalise_box``: numeric strings and nested
    lists/tuples are accepted; anything non-numeric or not point-shaped
    yields None (the scorer then treats the prediction as degenerate).
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
    """Half-open tight box of a polygon's vertices (derive_instances style)."""
    if not polygon:
        return None
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return [min(xs), min(ys), max(xs) + 1, max(ys) + 1]


def classify_and_rasterise(polygon: list[list[int]] | None, frame_w: int,
                           frame_h: int) -> tuple[str, np.ndarray | None]:
    """Per-polygon validity + rasterisation: 'ok' | 'out_of_frame' | 'degenerate'.

    Coordinate rule mirrors ``scoring.classify_boxes`` exactly: any vertex
    outside ``[0, frame_w-1] x [0, frame_h-1]`` is out_of_frame (never
    clamped, never matched). 'ok' polygons are rasterised to a bool mask.
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


def rasterise_polygon(polygon: list[list[int]], frame_w: int,
                      frame_h: int) -> np.ndarray:
    """Fill a polygon into a bool HxW mask (PIL polygon semantics)."""
    canvas = Image.new("L", (frame_w, frame_h), 0)
    ImageDraw.Draw(canvas).polygon([tuple(p) for p in polygon], fill=1)
    return np.array(canvas, dtype=bool)


# ---------------------------------------------------------------------------
# Mask IoU + matching (greedy, the detection contract on masks)
# ---------------------------------------------------------------------------

def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two bool masks (pixel counts)."""
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    inter = int(np.logical_and(a, b).sum())
    return float(inter / union)


def prepare_predictions_masks(inventory, frame_w: int,
                              frame_h: int) -> list[dict]:
    """Normalise a parsed inventory into the mask scorer's prediction records.

    Same shape as ``scoring.prepare_predictions`` with ``mask`` in place of
    ``bbox``; confidence is used only for ranking, missing/non-numeric ranks
    last and is flagged.
    """
    preds = []
    for index, entry in enumerate(inventory or []):
        polygon = entry.get("polygon") if isinstance(entry, dict) else None
        polygon = normalise_polygon(polygon)
        confidence = entry.get("confidence_pct") if isinstance(entry, dict) else None
        preds.append({
            "index": index,
            "polygon": polygon,
            "bbox": entry.get("bbox") if isinstance(entry, dict) else None,
            "confidence": float(confidence)
            if isinstance(confidence, (int, float)) else None,
            "confidence_missing": not isinstance(confidence, (int, float)),
        })
    for pred in preds:
        flag, mask = classify_and_rasterise(pred["polygon"], frame_w, frame_h)
        pred["flag"] = flag
        pred["valid"] = flag == "ok"
        pred["mask"] = mask
    return preds


def _rank_key(pred: dict):
    confidence = pred["confidence"] if pred["confidence"] is not None else 0.0
    return (-confidence, pred["index"])


def greedy_match_masks(preds: list[dict], gt_masks: list[np.ndarray],
                       iou_threshold: float = 0.5) -> dict:
    """Greedy one-to-one mask matching — ``scoring.greedy_match`` on masks.

    Identical tie-break contract: confidence-descending (then inventory
    index), each prediction claims the unmatched GT with the highest mask
    IoU (ties -> lowest GT index), invalid predictions never match.
    """
    order = sorted(preds, key=_rank_key)
    ious = [
        [mask_iou(pred["mask"], gt) for gt in gt_masks]
        if pred.get("valid") and pred["mask"] is not None else []
        for pred in order
    ]
    matched_gt: set[int] = set()
    matches: list[dict] = []
    fp_indices: list[int] = []
    for pred, row in zip(order, ious):
        if not row:
            fp_indices.append(pred["index"])
            continue
        best_index, best_score = None, -1.0
        for gt_index, score in enumerate(row):
            if gt_index in matched_gt:
                continue
            if score > best_score:  # strict '>' keeps the lowest index on ties
                best_index, best_score = gt_index, score
        if best_index is not None and best_score >= iou_threshold:
            matched_gt.add(best_index)
            matches.append({
                "pred_index": pred["index"],
                "gt_index": best_index,
                "iou": round(best_score, 4),
            })
        else:
            fp_indices.append(pred["index"])
    fn_gt_indices = [g for g in range(len(gt_masks)) if g not in matched_gt]
    matched_ious = [m["iou"] for m in matches]
    return {
        "matches": sorted(matches, key=lambda m: m["pred_index"]),
        "fp_indices": sorted(fp_indices),
        "fn_gt_indices": fn_gt_indices,
        "tp": len(matches),
        "fp": len(fp_indices),
        "fn": len(fn_gt_indices),
        "mean_matched_iou": (round(sum(matched_ious) / len(matched_ious), 4)
                             if matched_ious else None),
    }


def score_masks_image(inventory, gt_masks: list[np.ndarray], gt_areas,
                      frame_w: int, frame_h: int) -> dict:
    """One image's mask sub-record (all thresholds + polygon diagnostics)."""
    preds = prepare_predictions_masks(inventory, frame_w, frame_h)
    areas = list(gt_areas or [])

    result: dict = {
        "n_gt": len(gt_masks),
        "n_pred": len(preds),
        "count_error": len(preds) - len(gt_masks),
        "n_polygons_out_of_frame": sum(1 for p in preds
                                       if p.get("flag") == "out_of_frame"),
        "n_polygons_degenerate": sum(1 for p in preds
                                     if p.get("flag") == "degenerate"),
    }
    for threshold in IOU_THRESHOLDS:
        matched = greedy_match_masks(preds, gt_masks, threshold)
        key = int(round(threshold * 100))
        result[f"tp_{key}"] = matched["tp"]
        result[f"fp_{key}"] = matched["fp"]
        result[f"fn_{key}"] = matched["fn"]
        if abs(threshold - 0.5) < 1e-9:
            result["mean_matched_iou_50"] = matched["mean_matched_iou"]
            result["matches_50"] = matched["matches"]
            result["fp_pred_indices_50"] = matched["fp_indices"]
            result["fn_gt_indices_50"] = matched["fn_gt_indices"]
            result["fn_gt_areas_50"] = [areas[j] for j in matched["fn_gt_indices"]
                                        if j < len(areas)]

    # Does the reported bbox agree with the polygon's extent? Pure diagnostic
    # (never a gate): a model that boxes the WHOLE fruit but outlines only the
    # visible surface SHOULD disagree somewhat on occluded fruit.
    polybox_ious = []
    for pred in preds:
        derived = polygon_bbox(pred["polygon"])
        reported = pred.get("bbox")
        if derived is not None and isinstance(reported, (list, tuple)) \
                and len(reported) == 4:
            polybox_ious.append(det_scoring.iou([float(v) for v in reported],
                                                [float(v) for v in derived]))
    result["polygon_bbox_iou_mean"] = (round(sum(polybox_ious) / len(polybox_ious), 4)
                                       if polybox_ious else None)
    return result


def ap_metrics_masks(per_image: list[dict]) -> dict:
    """AP@[.50:.95] over the batch on mask IoU (mirror of scoring.ap_metrics).

    ``per_image``: ``{"preds": [...prepare_predictions_masks output...],
    "gt_masks": [...]}`` — the caller supplies rasterised predictions and GT
    masks (label PNGs must be readable; a rebuild without the dataset mount
    keeps the stored AP instead of recomputing it).
    """
    per_threshold = {}
    for threshold in AP_THRESHOLDS:
        pooled: list[tuple[float, bool]] = []
        n_gt = 0
        for image in per_image:
            gt_masks = image["gt_masks"]
            n_gt += len(gt_masks)
            matched = greedy_match_masks(image["preds"], gt_masks, threshold)
            hit = {m["pred_index"] for m in matched["matches"]}
            for pred in image["preds"]:
                confidence = pred["confidence"] if pred["confidence"] is not None else 0.0
                pooled.append((confidence, pred["index"] in hit))
        per_threshold[threshold] = det_scoring.average_precision(pooled, n_gt)
    valid = [value for value in per_threshold.values() if value is not None]
    return {
        "per_threshold": per_threshold,
        "ap_50": per_threshold.get(0.5),
        "map_50_95": round(sum(valid) / len(valid), 4) if valid else None,
    }
