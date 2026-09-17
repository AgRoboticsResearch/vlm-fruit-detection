#!/usr/bin/env python3
"""Segmentation overlay rendering for the vlm_eval seg pipeline.

Never fed to the model — after-the-fact diagnostics pairing the model's
polygon inventory with what ground truth exists. There are no GT masks on
any vlm_eval source, so (unlike the StrawDI seg overlays) there is no
TP/FP colouring and no GT mask fill; the conventions follow the base
harness's inventory overlay instead:

* polygon outline + light fill – coloured by the fruit's CONTINUOUS redness
  ramp (green at 0% -> amber at 50% -> red at 100%), the same encoding the
  box pipeline uses for its boxes;
* cyan polygon – the nominated pick target (`target_index`), matching the
  base harness's cyan-target convention;
* magenta ring – each fruit's reported picking point (drawn whether or not
  the peduncle was reported visible — the point is null in that case and no
  ring appears);
* on ground-truthed frames only – the cyan GT crosshair and the upstream
  rough 60x60 ellipse box, drawn LAST so the reference is never obscured
  (the base harness's rule), plus the nominated point's error in px;
* label text – `#i [TARGET] red xx% occ yy%`, label colour follows the
  polygon except the redness-coloured `red xx%` segment;
* small legend – bottom-left swatches: polygon / target / picking point (+GT
  on scored frames).

Out-of-frame and degenerate polygons are counted by the scorer, not drawn.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from vlm_eval.lib import imaging
from . import seg_scoring

TARGET_COLOR = (0, 255, 255)      # cyan – nominated pick target (base convention)
PRED_COLOR = (255, 0, 255)        # magenta – reported picking points
GT_COLOR = (0, 220, 255)          # cyan – ground-truth point (base convention)
PRED_FILL_ALPHA = 0.18


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


def _clamped_label_xy(draw: ImageDraw.ImageDraw, xy, text: str, size: int,
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


def _legend(draw: ImageDraw.ImageDraw, canvas_wh, scale: float,
            with_gt: bool) -> None:
    """Small bottom-left legend: polygon / target / picking point (+ GT)."""
    size = max(11, int(round(12 * scale)))
    swatch = max(8, int(round(9 * scale)))
    pad = 4
    entries: list[tuple[tuple[int, int, int], str]] = [
        ((60, 200, 60), "polygon = visible surface"),
        (TARGET_COLOR, "target"),
        (PRED_COLOR, "picking point"),
    ]
    if with_gt:
        entries.append((GT_COLOR, "GT point"))
    widths = [swatch + 4 + draw.textlength(label, font=imaging.font(size, bold=True))
              + 14 for _, label in entries]
    total = sum(widths) + 2 * pad
    height = swatch + 2 * pad
    top = canvas_wh[1] - height - 6
    draw.rectangle([4, top - 2, 6 + total, top + height + 2], fill=(0, 0, 0))
    x = 4 + pad
    for (colour, label), width in zip(entries, widths):
        draw.rectangle([x, top + pad, x + swatch, top + pad + swatch],
                       outline=colour, width=2)
        _text(draw, (x + swatch + 4, top + height // 2), label, size=size,
              anchor="lm")
        x += width


def draw_segmentation_overlay(image_rgb: np.ndarray,
                              inventory: list[dict] | None,
                              target_index: int | None,
                              gt: dict | None,
                              title: str,
                              header_px: int = 34) -> np.ndarray:
    """Per-run diagnostic: redness-ramped polygons, cyan target, GT reference.

    ``gt`` is None on unlabelled frames, else
    ``{"uv": (u, v), "rough_box": [x1, y1, x2, y2], "error_px": float | None}``.
    """
    inventory = list(inventory or [])
    h, w = image_rgb.shape[:2]

    frame = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).copy()
    canvas = Image.new("RGB", (w, h + header_px), (0, 0, 0))
    canvas.paste(frame, (0, header_px))

    # --- polygon fills (numpy blend, frame region only) ------------------------
    arr = np.asarray(canvas).copy()
    region = arr[header_px:header_px + h]
    for index, berry in enumerate(inventory):
        polygon = seg_scoring.normalise_polygon(berry.get("polygon")
                                                if isinstance(berry, dict) else None)
        if polygon is None:
            continue
        flag, mask = seg_scoring.classify_and_rasterise(polygon, w, h)
        if flag != "ok":
            continue  # out-of-frame / degenerate polygons are counted, not drawn
        is_target = target_index is not None and index == target_index
        colour = TARGET_COLOR if is_target \
            else imaging.redness_colour(berry.get("redness_pct"))
        _blend(region, mask, colour, PRED_FILL_ALPHA)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    # --- outlines + labels + picking rings (PIL, on top of the fills) ----------
    scale = max(1.0, min(h, w) / 720.0)
    text_size = int(14 * scale)
    outline_width = max(2, int(round(3 * scale)))
    ring = max(3, int(round(6 * scale)))

    for index, berry in enumerate(inventory):
        polygon = seg_scoring.normalise_polygon(berry.get("polygon")
                                                if isinstance(berry, dict) else None)
        if polygon is None:
            continue
        flag, _ = seg_scoring.classify_and_rasterise(polygon, w, h)
        if flag != "ok":
            continue
        is_target = target_index is not None and index == target_index
        colour = TARGET_COLOR if is_target \
            else imaging.redness_colour(berry.get("redness_pct"))
        extent = seg_scoring.polygon_bbox(polygon)
        # Halo pass under the outline so it survives against the fills.
        pts = [(p[0], p[1] + header_px) for p in polygon]
        draw.line(pts + [pts[0]], fill=(0, 0, 0), width=outline_width + 2,
                  joint="curve")
        draw.line(pts + [pts[0]], fill=colour, width=outline_width, joint="curve")

        segments: list[tuple[str, tuple]] = [
            (f"#{index}{' TARGET' if is_target else ''}", colour)]
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

        point = berry.get("picking_point")
        if point is not None:
            px, py = float(point[0]), float(point[1]) + header_px
            draw.ellipse([px - ring, py - ring, px + ring, py + ring],
                         outline=(0, 0, 0), width=outline_width + 1)
            draw.ellipse([px - ring, py - ring, px + ring, py + ring],
                         outline=PRED_COLOR, width=max(2, outline_width - 1))

    # --- ground truth LAST (base-harness rule: the reference is never obscured)
    if gt is not None and gt.get("uv") is not None:
        gx, gy = float(gt["uv"][0]), float(gt["uv"][1]) + header_px
        box = gt.get("rough_box")
        if box is not None:
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0 + header_px
            half = max((box[2] - box[0]), (box[3] - box[1])) / 2.0
            draw.ellipse([cx - half, cy - half, cx + half, cy + half],
                         outline=(0, 0, 0), width=5)
            draw.ellipse([cx - half, cy - half, cx + half, cy + half],
                         outline=GT_COLOR, width=3)
        arm = 16
        draw.line([(gx - arm, gy), (gx + arm, gy)], fill=GT_COLOR, width=3)
        draw.line([(gx, gy - arm), (gx, gy + arm)], fill=GT_COLOR, width=3)
        if gt.get("error_px") is not None:
            _text(draw, (gx + arm + 4, gy - int(20 * scale)),
                  f"{gt['error_px']:.1f}px", size=text_size, fill=GT_COLOR)

    _text(draw, (6, header_px // 2), title, size=17, fill=(255, 255, 255),
          anchor="lm")
    n_target = (0 if target_index is None or target_index < 0 else 1)
    summary = f"{len(inventory)} found, {n_target} target" if inventory else "0 found"
    _text(draw, (w - 6, header_px // 2), summary, size=15,
          fill=(255, 255, 255) if inventory else (255, 120, 120), anchor="rm")
    _legend(draw, img.size, scale, with_gt=gt is not None)
    return np.array(img)
