#!/usr/bin/env python3
"""Ground-truth loaders for the paper_study mixed testbed.

One loader per dataset family, each returning the SAME shape so the runner and
scorers stay dataset-agnostic:

    boxes  -> list[[x1, y1, x2, y2]] int, in the DELIVERED frame's pixels
    masks  -> list[HxW bool] or None, aligned 1:1 with boxes (same order)

All coordinates are returned in the delivered (possibly downscaled) frame
space; loaders take the source->delivered scale factor and apply it. Masks are
NEAREST-resampled so instances stay crisp at 1280 px.

Conventions were verified empirically against the data (2026-09-18):

* ACFR circles  ``item,c-x,c-y,radius,label``  -> centre + radius (validated
  against the apples' semantic pixel masks: circle centres land on apple
  pixels; per-image union IoU 0.5-0.8, the residual being the masks' own
  coarser annotation).
* ACFR rectangles ``item,x,y,dx,dy,label`` -> TOP-LEFT + size (the bright-fruit
  coverage test picks top-left over centre 6:1 on mangoes; same header and
  tool across mangoes/almonds).
* WGISD ``.txt`` lines ``class cx cy w h`` are YOLO-normalised against
  2048x1365; the ``.npz`` masks are HxWxN in source pixels with mask i
  matching box line i (per the dataset README).
* MinneApple (HF COCO mirror) instances may carry multiple polygon rings
  (``segmentation`` is a list of lists); all rings rasterise into ONE mask.
* StrawDI label PNGs are grayscale id-maps, 1..N instance ids in sorted order.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------
# generic helpers
# --------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scale_box(box, s: float):
    x1, y1, x2, y2 = box
    return [int(round(x1 * s)), int(round(y1 * s)),
            int(round(x2 * s)), int(round(y2 * s))]


def resize_mask(mask: np.ndarray, out_wh: tuple[int, int]) -> np.ndarray:
    """NEAREST-resample a bool mask to the delivered frame size."""
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    img = img.resize(out_wh, Image.Resampling.NEAREST)
    return np.array(img) > 0


# --------------------------------------------------------------------------
# ACFR (apples: circles / mangoes+almonds: top-left rectangles)
# --------------------------------------------------------------------------

def acfr_rows(csv_path: Path) -> list[list[float]]:
    rows = []
    for line in Path(csv_path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append([float(v) for v in line.split(",")])
    return rows


def acfr_boxes(csv_path: Path, kind: str) -> list[list[int]]:
    """`kind` is 'circles' (apples) or 'rects' (mangoes, almonds, KFuji)."""
    boxes = []
    for row in acfr_rows(csv_path):
        if kind == "circles":
            _, cx, cy, r, _label = row
            boxes.append([int(round(cx - r)), int(round(cy - r)),
                          int(round(cx + r)), int(round(cy + r))])
        else:
            _, x, y, dx, dy, _label = row
            boxes.append([int(round(x)), int(round(y)),
                          int(round(x + dx)), int(round(y + dy))])
    return boxes


# --------------------------------------------------------------------------
# KFuji RGB-DS (per-patch CSVs, same annotation lineage as ACFR: item,x,y,dx,dy
# top-left squares — validated by red-coverage 16/20, 2026-09-18; the item
# column is a GLOBAL apple id that continues across patches)
# --------------------------------------------------------------------------

def kfuji_boxes(csv_path: Path) -> list[list[int]]:
    return acfr_boxes(csv_path, "rects")


# --------------------------------------------------------------------------
# WGISD (YOLO-normalised boxes + per-cluster npz masks at source resolution)
# --------------------------------------------------------------------------

def wgisd_boxes(txt_path: Path, src_w: int, src_h: int) -> list[list[int]]:
    boxes = []
    for line in Path(txt_path).read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _cls, cx, cy, w, h = (float(v) for v in parts)
        boxes.append([int(round((cx - w / 2) * src_w)),
                      int(round((cy - h / 2) * src_h)),
                      int(round((cx + w / 2) * src_w)),
                      int(round((cy + h / 2) * src_h))])
    return boxes


def wgisd_masks(npz_path: Path, delivered_wh: tuple[int, int],
                src_wh: tuple[int, int]):
    """Per-cluster bool masks, order-aligned with the .txt lines."""
    with np.load(npz_path) as npz:
        arr = npz["arr_0"]
    masks = []
    for i in range(arr.shape[2]):
        m = arr[..., i] > 0
        if tuple(delivered_wh) != tuple(src_wh):
            m = resize_mask(m, delivered_wh)
        masks.append(m)
    return masks


# --------------------------------------------------------------------------
# MinneApple (COCO polygons; an instance may have several rings)
# --------------------------------------------------------------------------

def minneapple_by_file(coco_json: Path) -> dict:
    """file_name -> list of instances: {bbox (x1y1x2y2), polygons, area}."""
    data = json.loads(Path(coco_json).read_text())
    per_image: dict[str, list[dict]] = {}
    for ann in data["annotations"]:
        img = next(im for im in data["images"] if im["id"] == ann["image_id"])
        x, y, w, h = ann["bbox"]
        rings = [[(pts[k], pts[k + 1]) for k in range(0, len(pts), 2)]
                 for pts in ann["segmentation"] if len(pts) >= 6]
        per_image.setdefault(img["file_name"], []).append({
            "bbox": [int(round(x)), int(round(y)),
                     int(round(x + w)), int(round(y + h))],
            "polygons": rings,
            "area": float(ann.get("area") or 0.0),
        })
    for instances in per_image.values():
        instances.sort(key=lambda inst: inst["bbox"])   # stable GT order
    return per_image


def minneapple_instance_mask(polygons: list[list[tuple[int, int]]],
                             delivered_wh: tuple[int, int]) -> np.ndarray:
    """Rasterise one instance's polygon rings into a single bool mask."""
    from PIL import ImageDraw
    w, h = delivered_wh
    img = Image.new("1", (w, h), 0)
    draw = ImageDraw.Draw(img)
    for ring in polygons:
        if len(ring) < 3:
            continue
        flat = [c for pt in ring for c in pt]
        draw.polygon(flat, outline=1, fill=1)
    return np.array(img, dtype=bool)
