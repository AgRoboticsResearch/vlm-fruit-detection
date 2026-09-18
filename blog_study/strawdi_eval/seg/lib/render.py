#!/usr/bin/env python3
"""Segmentation diagnostics rendering for the StrawDI seg pipeline.

Never fed to the model — after-the-fact overlays pairing the model's polygon
inventory with the ground-truth instance masks. Reuses the base conventions
of ``vlm_eval/lib/imaging.py`` (redness colour ramp, header strip) and draws
a match-coloured segmentation layer:

* white translucent fill – a ground-truth instance's visible-surface mask;
  one the model missed carries a red ``MISS`` label at its GT-box position
  (no extra outline — the fill IS the ground truth)
* green outline + fill – a prediction polygon that matched a GT mask at
  mask IoU >= 0.5 (TP)
* red outline + fill – a prediction polygon that matched nothing (FP)
* label text – same colour as its polygon, except the ``red xx%`` segment,
  which is coloured by that fruit's reported redness
* small legend – bottom-left swatches: GT fill / TP / FP

Ground truth is drawn as FILLS rather than outlines on purpose: extracting
contours from the id-map masks would need cv2/skimage, and a translucent
region reads just as clearly against foliage.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from vlm_eval.lib import imaging
from . import seg_scoring

GT_COLOR = (255, 255, 255)      # white – ground-truth mask fill
TP_COLOR = (60, 230, 90)        # green – polygon that matched a GT mask
FP_COLOR = (255, 60, 60)        # red   – unmatched polygon; also MISS text

GT_FILL_ALPHA = 0.30
PRED_FILL_ALPHA = 0.28


def _blend(region: np.ndarray, mask: np.ndarray, colour, alpha: float) -> None:
    """In-place flat-colour blend of ``mask`` pixels inside ``region`` (RGB uint8)."""
    if not mask.any():
        return
    px = region[mask].astype(np.float32)
    region[mask] = (px * (1.0 - alpha)
                    + np.array(colour, dtype=np.float32) * alpha).round().astype(np.uint8)


def _text(draw: ImageDraw.ImageDraw, xy, text, size=14, fill=(255, 255, 255),
          anchor="la"):
    """Same haloed text as imaging._text (kept local: that one is private)."""
    fnt = imaging.font(size, bold=True)
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2)):
        draw.text((xy[0] + dx, xy[1] + dy), text, font=fnt, fill=(0, 0, 0),
                  anchor=anchor)
    draw.text(xy, text, font=fnt, fill=fill, anchor=anchor)


def _text_segments(draw: ImageDraw.ImageDraw, xy, segments, size: int = 14):
    """Left-to-right haloed text where each segment keeps its own colour."""
    fnt = imaging.font(size, bold=True)
    widths = [draw.textlength(text, font=fnt) for text, _ in segments]
    x = xy[0]
    for (text, _), width in zip(segments, widths):
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2)):
            draw.text((x + dx, xy[1] + dy), text, font=fnt, fill=(0, 0, 0),
                      anchor="la")
        x += width
    x = xy[0]
    for (text, colour), width in zip(segments, widths):
        draw.text((x, xy[1]), text, font=fnt, fill=colour, anchor="la")
        x += width


def _clamped_label_xy(draw: ImageDraw.ImageDraw, xy, text, size: int,
                      canvas_wh, top_limit: int):
    """Keep a label fully inside the canvas and below the header strip."""
    canvas_w, canvas_h = canvas_wh
    fnt = imaging.font(size, bold=True)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=fnt)
    width, height = right - left, bottom - top
    x = min(max(2, xy[0]), max(2, canvas_w - width - 2))
    y = xy[1]
    if y < top_limit:
        y = top_limit + 2
    y = min(y, canvas_h - height - 2)
    return x, y


def _legend(draw: ImageDraw.ImageDraw, canvas_wh, scale: float) -> None:
    """Small bottom-left legend: filled GT swatch, TP/FP outline swatches."""
    size = max(11, int(round(12 * scale)))
    swatch = max(8, int(round(9 * scale)))
    pad = 4
    entries: list[tuple[tuple[int, int, int], str, bool]] = [
        (GT_COLOR, "GT fill", True), (TP_COLOR, "TP", False),
        (FP_COLOR, "FP", False)]
    widths = [swatch + 4 + draw.textlength(label, font=imaging.font(size, bold=True))
              + 14 for _, label, _ in entries]
    total = sum(widths) + 2 * pad
    height = swatch + 2 * pad
    top = canvas_wh[1] - height - 6
    draw.rectangle([4, top - 2, 6 + total, top + height + 2], fill=(0, 0, 0))
    x = 4 + pad
    for (colour, label, filled), width in zip(entries, widths):
        if filled:
            tint = tuple(int(round(c * 0.55 + 255 * 0.45 * GT_FILL_ALPHA)) for c in colour)
            draw.rectangle([x, top + pad, x + swatch, top + pad + swatch],
                           fill=tint, outline=colour, width=1)
        else:
            draw.rectangle([x, top + pad, x + swatch, top + pad + swatch],
                           outline=colour, width=2)
        _text(draw, (x + swatch + 4, top + height // 2), label, size=size,
              anchor="lm")
        x += width


def draw_segmentation_overlay(image_rgb: np.ndarray,
                              inventory: list[dict] | None,
                              gt_masks: list[np.ndarray] | None,
                              gt_boxes: list | None,
                              matches_50: list[dict] | None,
                              fn_gt_indices: list[int] | None,
                              title: str,
                              header_px: int = 34) -> np.ndarray:
    """Per-run diagnostic: white GT mask fills under match-coloured polygons."""
    inventory = list(inventory or [])
    gt_boxes = list(gt_boxes or [])
    fn_set = set(fn_gt_indices or [])
    h, w = image_rgb.shape[:2]

    frame = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).copy()
    canvas = Image.new("RGB", (w, h + header_px), (0, 0, 0))
    canvas.paste(frame, (0, header_px))

    # --- fills (numpy blend, frame region only) --------------------------------
    arr = np.asarray(canvas).copy()
    region = arr[header_px:header_px + h]
    for gt_mask in list(gt_masks or []):
        if isinstance(gt_mask, np.ndarray) and gt_mask.shape == (h, w):
            _blend(region, gt_mask.astype(bool), GT_COLOR, GT_FILL_ALPHA)
    tp_pred_set = {m["pred_index"] for m in (matches_50 or [])}
    for index, berry in enumerate(inventory):
        polygon = seg_scoring.normalise_polygon(berry.get("polygon")
                                                if isinstance(berry, dict) else None)
        if polygon is None:
            continue
        flag, mask = seg_scoring.classify_and_rasterise(polygon, w, h)
        if flag != "ok":
            continue  # out-of-frame / degenerate polygons are counted, not drawn
        colour = TP_COLOR if index in tp_pred_set else FP_COLOR
        _blend(region, mask, colour, PRED_FILL_ALPHA)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    # --- outlines + labels (PIL, on top of the fills) --------------------------
    scale = max(1.0, min(h, w) / 720.0)
    text_size = int(14 * scale)
    outline_width = max(2, int(round(3 * scale)))

    for index, berry in enumerate(inventory):
        polygon = seg_scoring.normalise_polygon(berry.get("polygon")
                                                if isinstance(berry, dict) else None)
        if polygon is None:
            continue
        flag, _ = seg_scoring.classify_and_rasterise(polygon, w, h)
        if flag != "ok":
            continue
        is_tp = index in tp_pred_set
        colour = TP_COLOR if is_tp else FP_COLOR
        extent = seg_scoring.polygon_bbox(polygon)
        # Halo pass under the outline so it survives against the fills.
        pts = [(p[0], p[1] + header_px) for p in polygon]
        draw.line(pts + [pts[0]], fill=(0, 0, 0), width=outline_width + 2,
                  joint="curve")
        draw.line(pts + [pts[0]], fill=colour, width=outline_width, joint="curve")
        segments: list[tuple[str, tuple]] = [(f"#{index} {'TP' if is_tp else 'FP'}",
                                              colour)]
        if berry.get("redness_pct") is not None:
            segments.append((f" red {berry['redness_pct']}%",
                             imaging.redness_colour(berry["redness_pct"])))
        if berry.get("occlusion_pct") is not None:
            segments.append((f" occ {berry['occlusion_pct']}%", colour))
        label = "".join(text for text, _ in segments)
        xy = _clamped_label_xy(draw, (extent[0], extent[1] - int(17 * scale)
                                      + header_px),
                               label, text_size, img.size, header_px)
        _text_segments(draw, xy, segments, size=text_size)

    for gt_index in sorted(fn_set):
        if gt_index >= len(gt_boxes):
            continue
        x1, y1, x2, y2 = [float(v) for v in gt_boxes[gt_index]]
        xy = _clamped_label_xy(draw, (x1, y2 + header_px + int(2 * scale)),
                               "MISS", text_size, img.size, header_px)
        _text(draw, xy, "MISS", size=text_size, fill=FP_COLOR)

    n_pred = len(inventory)
    tp = len(matches_50 or [])
    fp = n_pred - tp
    fn = len(fn_set)
    _text(draw, (6, header_px // 2), title, size=17, fill=(255, 255, 255),
          anchor="lm")
    summary = f"pred {n_pred} / gt {len(gt_masks or [])}  TP {tp} FP {fp} FN {fn}"
    _text(draw, (w - 6, header_px // 2), summary, size=15,
          fill=(255, 120, 120) if fn else (255, 255, 255), anchor="rm")
    _legend(draw, img.size, scale)
    return np.array(img)
