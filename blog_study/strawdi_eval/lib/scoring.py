#!/usr/bin/env python3
"""Multi-instance detection scoring for the StrawDI pipeline.

Pure functions only (no I/O, no model calls). Everything here is deterministic:
ties in confidence are broken by list index, ties in IoU by ground-truth index,
so a re-run of the scorer over the same records reproduces every number — which
``verify_strawdi_run.py`` actively asserts.

The matching contract (also printed in every report):

* predictions are ranked ``confidence_pct`` descending (missing/non-numeric
  confidence ranks last), then by their position in the inventory;
* each prediction greedily claims the unmatched ground-truth box with the
  highest IoU (ties -> lowest ground-truth index) and matches when IoU >= the
  threshold; matching is recomputed independently at every threshold;
* a box with coordinates outside ``[0, W) x [0, H)`` or with zero/negative
  extent is excluded from matching and counted as a false positive (it still
  counts toward ``n_pred``);
* the secondary ``center`` criterion replaces the IoU test with symmetric
  centre containment — the prompt asks for whole-fruit boxes while the StrawDI
  masks annotate the visible surface, so occluded-fruit predictions
  systematically exceed the GT box and raw IoU is a pessimistic read on them.
"""

from __future__ import annotations

IOU_THRESHOLDS = (0.25, 0.5, 0.75)
AP_THRESHOLDS = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))

# COCO area bands (px^2), applied to the GT instance area.
SIZE_STRATA = (
    ("small", 0, 32 ** 2),
    ("medium", 32 ** 2, 96 ** 2),
    ("large", 96 ** 2, None),
)

# Keys every scored record carries; failure records carry None for all of them.
DETECTION_KEYS = (
    "n_gt", "n_pred", "count_error",
    "tp_25", "fp_25", "fn_25",
    "tp_50", "fp_50", "fn_50", "mean_matched_iou_50",
    "tp_75", "fp_75", "fn_75",
    "tp_center", "fp_center", "fn_center",
    "n_boxes_out_of_frame", "n_boxes_degenerate",
)

# Extra per-image detail kept beside the flat keys (lists, not scalars).
DETECTION_DETAIL_KEYS = ("matches_50", "fp_pred_indices_50", "fn_gt_indices_50",
                         "fn_gt_areas_50")


def empty_detection() -> dict:
    """Detection block for a run whose answer never parsed (never zeros)."""
    return {key: None for key in DETECTION_KEYS + DETECTION_DETAIL_KEYS}


def iou(a, b) -> float:
    """Axis-aligned IoU, identical semantics to ``run_vlm_eval._iou``.

    A unit test asserts the two agree; keeping the twin here lets the scorer
    stay importable without the provider stack.
    """
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def box_center(box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def center_hit(pred_box, gt_box) -> bool:
    """Symmetric, edge-inclusive centre containment."""
    pcx, pcy = box_center(pred_box)
    gcx, gcy = box_center(gt_box)
    pred_contains_gt = (pred_box[0] <= gcx <= pred_box[2]
                        and pred_box[1] <= gcy <= pred_box[3])
    gt_contains_pred = (gt_box[0] <= pcx <= gt_box[2]
                        and gt_box[1] <= pcy <= gt_box[3])
    return pred_contains_gt or gt_contains_pred


def classify_boxes(boxes, frame_w: int, frame_h: int) -> list[str]:
    """Per-prediction validity: 'ok' | 'out_of_frame' | 'degenerate'."""
    flags = []
    for box in boxes:
        if box is None:
            flags.append("degenerate")
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        if x2 <= x1 or y2 <= y1:
            flags.append("degenerate")
        elif x1 < 0 or y1 < 0 or x2 > frame_w or y2 > frame_h:
            flags.append("out_of_frame")
        else:
            flags.append("ok")
    return flags


def prepare_predictions(inventory, frame_w: int | None = None,
                        frame_h: int | None = None) -> list[dict]:
    """Normalise a parsed inventory into the scorer's prediction records.

    Confidence is used only for ranking; a missing or non-numeric value ranks
    last (treated as 0) and is flagged so the report can count it. When frame
    dimensions are given, each prediction also gets a ``valid``/``flag``
    validity annotation (see :func:`classify_boxes`).
    """
    preds = []
    for index, entry in enumerate(inventory or []):
        box = entry.get("bbox") if isinstance(entry, dict) else None
        box = [float(v) for v in box] if box is not None else None
        confidence = entry.get("confidence_pct") if isinstance(entry, dict) else None
        preds.append({
            "index": index,
            "bbox": box,
            "confidence": float(confidence)
            if isinstance(confidence, (int, float)) else None,
            "confidence_missing": not isinstance(confidence, (int, float)),
        })
    if frame_w is not None and frame_h is not None:
        for pred, flag in zip(preds, classify_boxes([p["bbox"] for p in preds],
                                                    frame_w, frame_h)):
            pred["valid"] = flag == "ok"
            pred["flag"] = flag
    return preds


def _rank_key(pred: dict):
    confidence = pred["confidence"] if pred["confidence"] is not None else 0.0
    return (-confidence, pred["index"])


def greedy_match(preds: list[dict], gts: list[dict], criterion: str = "iou",
                 iou_threshold: float = 0.5) -> dict:
    """Greedy one-to-one matching, highest-confidence prediction first.

    ``criterion='iou'``: match when IoU >= ``iou_threshold``.
    ``criterion='center'``: match when :func:`center_hit` holds (the threshold
    is then a formality — the score is 1.0 or 0.0).
    Invalid predictions (``valid=False``) never match and are false positives.
    """
    order = sorted(preds, key=_rank_key)
    matched_gt: set[int] = set()
    matches: list[dict] = []
    fp_indices: list[int] = []
    for pred in order:
        if not pred.get("valid", True) or pred["bbox"] is None:
            fp_indices.append(pred["index"])
            continue
        best_index, best_score = None, -1.0
        for gt_index, gt in enumerate(gts):
            if gt_index in matched_gt:
                continue
            if criterion == "center":
                score = 1.0 if center_hit(pred["bbox"], gt["bbox"]) else 0.0
            else:
                score = iou(pred["bbox"], gt["bbox"])
            if score > best_score:  # strict '>' keeps the lowest index on ties
                best_index, best_score = gt_index, score
        if best_index is not None and best_score >= iou_threshold:
            matched_gt.add(best_index)
            matches.append({
                "pred_index": pred["index"],
                "gt_index": best_index,
                "iou": round(iou(pred["bbox"], gts[best_index]["bbox"]), 4),
            })
        else:
            fp_indices.append(pred["index"])
    fn_gt_indices = [gt["index"] for gt in gts if gt["index"] not in matched_gt]
    matched_ious = [m["iou"] for m in matches]
    return {
        "matches": sorted(matches, key=lambda m: m["pred_index"]),
        "fp_indices": sorted(fp_indices),
        "fn_gt_indices": sorted(fn_gt_indices),
        "tp": len(matches),
        "fp": len(fp_indices),
        "fn": len(fn_gt_indices),
        "mean_matched_iou": (round(sum(matched_ious) / len(matched_ious), 4)
                             if matched_ious else None),
    }


def score_image(inventory, gt_boxes, gt_areas, frame_w: int, frame_h: int) -> dict:
    """One image's detection sub-record (all thresholds + center criterion)."""
    preds = prepare_predictions(inventory, frame_w, frame_h)
    gts = [{"index": j, "bbox": [float(v) for v in box]}
           for j, box in enumerate(gt_boxes)]
    areas = list(gt_areas or [])

    result: dict = {
        "n_gt": len(gts),
        "n_pred": len(preds),
        "count_error": len(preds) - len(gts),
        "n_boxes_out_of_frame": sum(1 for p in preds if p.get("flag") == "out_of_frame"),
        "n_boxes_degenerate": sum(1 for p in preds if p.get("flag") == "degenerate"),
    }
    for threshold in IOU_THRESHOLDS:
        matched = greedy_match(preds, gts, "iou", threshold)
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
    center = greedy_match(preds, gts, "center")
    result["tp_center"] = center["tp"]
    result["fp_center"] = center["fp"]
    result["fn_center"] = center["fn"]
    return result


def precision_recall_f1(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None
          and (precision + recall) > 0 else None)
    return (round(precision, 4) if precision is not None else None,
            round(recall, 4) if recall is not None else None,
            round(f1, 4) if f1 is not None else None)


def average_precision(pooled: list[tuple[float, bool]], n_gt: int) -> float | None:
    """All-points interpolated AP (the COCO convention) over pooled, ranked hits.

    ``pooled`` entries are ``(confidence, matched)``; the caller has already
    decided matched/unmatched at this IoU threshold. Sorting is stable and
    deterministic (Python's sort keeps pool order on confidence ties).
    """
    if n_gt == 0:
        return None
    if not pooled:
        return 0.0
    ordered = sorted(pooled, key=lambda item: -item[0])
    precisions, recalls = [], []
    tp_seen = 0
    for k, (_, matched) in enumerate(ordered, start=1):
        tp_seen += 1 if matched else 0
        precisions.append(tp_seen / k)
        recalls.append(tp_seen / n_gt)
    # All-points interpolation: walk the PR curve, and for every recall step
    # take the best precision achievable at that recall or beyond.
    suffix_max = precisions[:]
    for k in range(len(precisions) - 2, -1, -1):
        suffix_max[k] = max(precisions[k], suffix_max[k + 1])
    ap, previous_recall = 0.0, 0.0
    for k, recall in enumerate(recalls):
        if recall > previous_recall:
            ap += (recall - previous_recall) * suffix_max[k]
            previous_recall = recall
    return round(ap, 4)


def ap_metrics(per_image: list[dict]) -> dict:
    """AP@[.50:.95] over the whole batch.

    ``per_image``: ``{"sample_id", "preds": [...prepare_predictions output,
    with validity flags...], "gt_boxes": [...]}``. Matching is redone per
    threshold (a pair matched at 0.50 need not match at 0.75).
    """
    per_threshold = {}
    for threshold in AP_THRESHOLDS:
        pooled: list[tuple[float, bool]] = []
        n_gt = 0
        for image in per_image:
            gts = [{"index": j, "bbox": [float(v) for v in box]}
                   for j, box in enumerate(image["gt_boxes"])]
            n_gt += len(gts)
            matched = greedy_match(image["preds"], gts, "iou", threshold)
            hit = {m["pred_index"] for m in matched["matches"]}
            for pred in image["preds"]:
                confidence = pred["confidence"] if pred["confidence"] is not None else 0.0
                pooled.append((confidence, pred["index"] in hit))
        per_threshold[threshold] = average_precision(pooled, n_gt)
    valid = [value for value in per_threshold.values() if value is not None]
    return {
        "per_threshold": per_threshold,
        "ap_50": per_threshold.get(0.5),
        "map_50_95": round(sum(valid) / len(valid), 4) if valid else None,
    }


def stratify_area(area_px: float) -> str | None:
    for name, low, high in SIZE_STRATA:
        if area_px >= low and (high is None or area_px < high):
            return name
    return None


def size_stratified(records: list[dict]) -> list[dict]:
    """Recall and matched IoU@0.5 per GT-area stratum, from scored records.

    Records carry ``gt_areas`` and ``matches_50`` (with ``gt_index``/``iou``),
    so this never re-parses anything.
    """
    rows = []
    for name, low, high in SIZE_STRATA:
        n = hit = 0
        ious = []
        for record in records:
            if record.get("matches_50") is None:
                continue
            matched_iou = {m["gt_index"]: m["iou"] for m in record["matches_50"]}
            for gt_index, area in enumerate(record.get("gt_areas") or []):
                if stratify_area(area) != name:
                    continue
                n += 1
                if gt_index in matched_iou:
                    hit += 1
                    ious.append(matched_iou[gt_index])
        rows.append({
            "stratum": name,
            "area_band_px": (f"{low}-{high}" if high is not None else f"{low}+"),
            "n_gt": n,
            "n_matched": hit,
            "recall": round(hit / n, 4) if n else None,
            "mean_matched_iou_50": round(sum(ious) / len(ious), 4) if ious else None,
        })
    return rows


def occlusion_split(records: list[dict]) -> dict:
    """Diagnostics for the whole-fruit vs visible-surface bbox-semantics gap.

    Splits matched pairs at IoU@0.5 by the matched prediction's reported
    ``occlusion_pct`` (<25 vs >=25) and compares the reported occlusion of
    matched vs unmatched predictions.
    """
    low_ious, high_ious = [], []
    matched_occlusions, unmatched_occlusions = [], []

    def entry(record, pred_index):
        inventory = record.get("inventory") or []
        return inventory[pred_index] if 0 <= pred_index < len(inventory) else {}

    for record in records:
        if record.get("matches_50") is None:
            continue
        matched_preds = {m["pred_index"] for m in record["matches_50"]}
        for match in record["matches_50"]:
            occlusion = entry(record, match["pred_index"]).get("occlusion_pct")
            if isinstance(occlusion, (int, float)):
                (low_ious if occlusion < 25 else high_ious).append(match["iou"])
        for pred_index in record.get("fp_pred_indices_50") or []:
            occlusion = entry(record, pred_index).get("occlusion_pct")
            if isinstance(occlusion, (int, float)):
                unmatched_occlusions.append(float(occlusion))
        for pred_index in matched_preds:
            occlusion = entry(record, pred_index).get("occlusion_pct")
            if isinstance(occlusion, (int, float)):
                matched_occlusions.append(float(occlusion))

    def mean(values):
        return round(sum(values) / len(values), 2) if values else None

    return {
        "matched_iou_occl_lt25_mean": mean(low_ious),
        "matched_iou_occl_ge25_mean": mean(high_ious),
        "n_matched_occl_lt25": len(low_ious),
        "n_matched_occl_ge25": len(high_ious),
        "mean_reported_occlusion_matched": mean(matched_occlusions),
        "mean_reported_occlusion_unmatched": mean(unmatched_occlusions),
    }
