#!/usr/bin/env python3
"""Detection diagnostics rendering for the StrawDI pipeline.

Never fed to the model — these are the after-the-fact overlays that pair the
model's inventory with the mask-derived ground truth. Reuses the base
conventions of ``vlm_eval/lib/imaging.py`` (redness-ramp prediction boxes,
header strip) and adds the GT layer:

* green thin box   – a ground-truth strawberry (visible-surface mask extent);
  one the model missed carries a red ``MISS`` label under it (no extra box)
* white corner ticks – a prediction that matched a GT box at IoU >= 0.5 (TP)
* ``FP`` label     – a prediction that matched nothing
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from vlm_eval.lib import imaging

GT_COLOR = (60, 230, 90)        # green – ground-truth box (missed or not)
MISS_COLOR = (255, 60, 60)      # red   – colour of the MISS text label
TICK_COLOR = (255, 255, 255)    # white – TP corner ticks


def _text(draw: ImageDraw.ImageDraw, xy, text, size=14, fill=(255, 255, 255),
          anchor="la"):
    """Same haloed text as imaging._text (kept local: that one is private)."""
    fnt = imaging.font(size, bold=True)
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2)):
        draw.text((xy[0] + dx, xy[1] + dy), text, font=fnt, fill=(0, 0, 0),
                  anchor=anchor)
    draw.text(xy, text, font=fnt, fill=fill, anchor=anchor)


def _corner_ticks(draw: ImageDraw.ImageDraw, box, length: int, width: int,
                  colour=TICK_COLOR):
    x1, y1, x2, y2 = box
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                           (x1, y2, 1, -1), (x2, y2, -1, -1)):
        draw.line([(cx, cy), (cx + dx * length, cy)], fill=colour, width=width)
        draw.line([(cx, cy), (cx, cy + dy * length)], fill=colour, width=width)


def _clamped_label_xy(draw: ImageDraw.ImageDraw, xy, text, size: int,
                      canvas_wh, top_limit: int):
    """Keep a box label fully inside the canvas and below the header strip.

    Boxes hugging the right edge would otherwise clip their label; boxes at
    the very top of the frame would otherwise draw it over the header.
    """
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


def draw_detection_overlay(image_rgb: np.ndarray,
                           inventory: list[dict] | None,
                           gt_boxes: list | None,
                           matches_50: list[dict] | None,
                           fn_gt_indices: list[int] | None,
                           title: str,
                           header_px: int = 34) -> np.ndarray:
    """Per-run diagnostic: predictions (redness ramp) over GT boxes (green)."""
    inventory = list(inventory or [])
    gt_boxes = list(gt_boxes or [])
    fn_set = set(fn_gt_indices or [])

    frame = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).copy()
    h, w = frame.height, frame.width
    img = Image.new("RGB", (w, h + header_px), (0, 0, 0))
    img.paste(frame, (0, header_px))
    draw = ImageDraw.Draw(img)
    oy = header_px

    scale = max(1.0, min(h, w) / 720.0)
    text_size = int(14 * scale)
    gt_width = max(2, int(round(2 * scale)))
    pred_width_base = max(2, int(round(3 * scale)))
    tick_length = max(6, int(round(9 * scale)))

    # GT first, so the (thicker) prediction strokes land on top. A missed GT
    # is marked only by its red MISS label below the box — never by a second
    # rectangle, which reads as a prediction box.
    for box in gt_boxes:
        x1, y1, x2, y2 = [float(v) for v in box]
        target = [x1, y1 + oy, x2, y2 + oy]
        draw.rectangle(target, outline=GT_COLOR, width=gt_width)

    tp_pred_set = {m["pred_index"] for m in (matches_50 or [])}
    for index, berry in enumerate(inventory):
        bbox = berry.get("bbox")
        if bbox is None:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        box = [min(x1, x2), min(y1, y2) + oy, max(x1, x2), max(y1, y2) + oy]
        colour = imaging.redness_colour(berry.get("redness_pct"))
        occlusion = berry.get("occlusion_pct")
        width = max(1, int(round((pred_width_base
                                  if occlusion is None or occlusion < 60 else 1)
                                 * scale)))
        draw.rectangle(box, outline=(0, 0, 0), width=width + 2)
        draw.rectangle(box, outline=colour, width=width)
        if index in tp_pred_set:
            _corner_ticks(draw, box, tick_length, max(2, gt_width))
            tag = "TP"
        else:
            tag = "FP"
        label = (f"#{index} {tag}"
                 + (f" red {berry['redness_pct']}%"
                    if berry.get("redness_pct") is not None else "")
                 + (f" occ {berry['occlusion_pct']}%"
                    if berry.get("occlusion_pct") is not None else ""))
        xy = _clamped_label_xy(draw, (box[0], box[1] - int(17 * scale)),
                               label, text_size, img.size, oy)
        _text(draw, xy, label, size=text_size, fill=colour)

    for gt_index in sorted(fn_set):
        if gt_index >= len(gt_boxes):
            continue
        x1, y1, x2, y2 = [float(v) for v in gt_boxes[gt_index]]
        xy = _clamped_label_xy(draw, (x1, y2 + oy + int(2 * scale)), "MISS",
                               text_size, img.size, oy)
        _text(draw, xy, "MISS", size=text_size, fill=MISS_COLOR)

    n_pred = len(inventory)
    tp = len(matches_50 or [])
    fp = n_pred - tp
    fn = len(fn_set)
    _text(draw, (6, header_px // 2), title, size=17, fill=(255, 255, 255),
          anchor="lm")
    summary = f"pred {n_pred} / gt {len(gt_boxes)}  TP {tp} FP {fp} FN {fn}"
    _text(draw, (w - 6, header_px // 2), summary, size=15,
          fill=(255, 120, 120) if fn else (255, 255, 255), anchor="rm")
    return np.array(img)


def draw_control_overlay(image_rgb: np.ndarray, truth: dict,
                         prediction: dict | None) -> np.ndarray:
    """Small diagnostic for the synthetic vision-delivery control."""
    img = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).copy()
    draw = ImageDraw.Draw(img)
    for key, colour in (("red_circle", (255, 0, 255)),
                        ("green_square", (0, 255, 255))):
        point = (prediction or {}).get(key)
        if isinstance(point, (list, tuple)) and len(point) == 2:
            px, py = float(point[0]), float(point[1])
            draw.ellipse([px - 12, py - 12, px + 12, py + 12], outline=colour,
                         width=3)
            tx, ty = truth[key]
            draw.line([px, py, tx, ty], fill=colour, width=2)
    return np.array(img)
