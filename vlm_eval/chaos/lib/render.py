#!/usr/bin/env python3
"""Segmentation overlay rendering for the chaos pipeline.

Never fed to the model — after-the-fact diagnostics of the nine-field
StrawDI-standard answers (no picking point, no nomination, no ground truth
on chaos scenes), so this render has none of those elements. Adapted from
``vlm_eval/seg/lib/render.py`` with the target / picking-point / GT layers
removed and a redness-ramp legend instead:

* polygon outline + light fill – coloured by the fruit's CONTINUOUS redness
  ramp (green at 0% -> amber at 50% -> red at 100%), the base harness's
  inventory-overlay encoding;
* label text – `#i red xx% occ yy%` (the `red xx%` segment redness-coloured);
* legend – bottom-left: three ramp swatches + "polygon = visible surface".

Out-of-frame and degenerate polygons are counted by the scorer, not drawn.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from vlm_eval.lib import imaging
from vlm_eval.seg.lib import seg_scoring

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


def _legend(draw: ImageDraw.ImageDraw, canvas_wh, scale: float) -> None:
    """Small bottom-left legend: redness-ramp swatches + the polygon meaning."""
    size = max(11, int(round(12 * scale)))
    swatch = max(8, int(round(9 * scale)))
    pad = 4
    label = "polygon = visible surface"
    fnt = imaging.font(size, bold=True)
    label_w = draw.textlength(label, font=fnt)
    ramps = [imaging.redness_colour(0), imaging.redness_colour(50),
             imaging.redness_colour(100)]
    total = 3 * swatch + 3 + swatch + 4 + label_w + 2 * pad + 10
    height = swatch + 2 * pad
    top = canvas_wh[1] - height - 6
    draw.rectangle([4, top - 2, 6 + total, top + height + 2], fill=(0, 0, 0))
    x = 4 + pad
    for colour in ramps:
        draw.rectangle([x, top + pad, x + swatch, top + pad + swatch],
                       fill=colour, outline=(0, 0, 0), width=1)
        x += swatch + 3
    x += 10
    draw.rectangle([x, top + pad, x + swatch, top + pad + swatch],
                   outline=(255, 255, 255), width=2)
    _text(draw, (x + swatch + 4, top + height // 2), label, size=size,
          anchor="lm")


def draw_segmentation_overlay(image_rgb: np.ndarray,
                              inventory: list[dict] | None,
                              title: str,
                              header_px: int = 34) -> np.ndarray:
    """Per-scene diagnostic: redness-ramped visible-surface polygons."""
    inventory = list(inventory or [])
    h, w = image_rgb.shape[:2]

    frame = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).copy()
    canvas = Image.new("RGB", (w, h + header_px), (0, 0, 0))
    canvas.paste(frame, (0, header_px))

    # --- polygon fills (numpy blend, frame region only) ------------------------
    arr = np.asarray(canvas).copy()
    region = arr[header_px:header_px + h]
    for berry in inventory:
        polygon = seg_scoring.normalise_polygon(berry.get("polygon")
                                                if isinstance(berry, dict) else None)
        if polygon is None:
            continue
        flag, mask = seg_scoring.classify_and_rasterise(polygon, w, h)
        if flag != "ok":
            continue  # out-of-frame / degenerate polygons are counted, not drawn
        _blend(region, mask, imaging.redness_colour(berry.get("redness_pct")),
               PRED_FILL_ALPHA)
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
        colour = imaging.redness_colour(berry.get("redness_pct"))
        extent = seg_scoring.polygon_bbox(polygon)
        # Halo pass under the outline so it survives against the fills.
        pts = [(p[0], p[1] + header_px) for p in polygon]
        draw.line(pts + [pts[0]], fill=(0, 0, 0), width=outline_width + 2,
                  joint="curve")
        draw.line(pts + [pts[0]], fill=colour, width=outline_width, joint="curve")

        segments: list[tuple[str, tuple]] = [(f"#{index}", colour)]
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

    _text(draw, (6, header_px // 2), title, size=17, fill=(255, 255, 255),
          anchor="lm")
    summary = f"{len(inventory)} found"
    _text(draw, (w - 6, header_px // 2), summary, size=15,
          fill=(255, 255, 255) if inventory else (255, 120, 120), anchor="rm")
    _legend(draw, img.size, scale)
    return np.array(img)
