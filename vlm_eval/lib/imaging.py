#!/usr/bin/env python3
"""Image preparation and rendering for the VLM raw-capability eval.

Everything here is deterministic and uses only Pillow + numpy.  Colour
convention is RGB throughout (matching ``target_ref``), because that is what
Pillow, the PNG on disk, and the images we hand to the model all agree on.

Two families of helpers live here:

* ``build_*`` – construct the images we *feed* the model (synthetic vision
  control, grid overlay, few-shot exemplar composite).
* ``draw_*`` – render diagnostics we never feed the model (ground-truth sanity
  check, per-run prediction overlay, contact sheet, error histogram).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Marker colours (all RGB).  Chosen to be distinguishable on strawberry frames.
GT_COLOR = (0, 220, 255)      # cyan  – ground-truth picking point
PRED_COLOR = (255, 0, 255)    # magenta – model prediction
BOX_COLOR = (255, 200, 0)     # amber – bounding boxes
OK_COLOR = (0, 200, 0)        # green – example annotation / good hit

CANVAS_GREY = (128, 128, 128)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    try:
        return ImageFont.truetype(path, size)
    except OSError:  # pragma: no cover - font is present on this image
        return ImageFont.load_default()


def load_rgb(path: Path | str) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def save_rgb(image_rgb: np.ndarray, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).save(path)


def _text(draw: ImageDraw.ImageDraw, xy, text, size=14, fill=(255, 255, 255),
          anchor="la", bold=True, halo=(0, 0, 0)):
    """Draw text with a 1-2 px dark halo so it stays readable on any background."""
    fnt = font(size, bold=bold)
    if halo is not None:
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2)):
            draw.text((xy[0] + dx, xy[1] + dy), text, font=fnt, fill=halo, anchor=anchor)
    draw.text(xy, text, font=fnt, fill=fill, anchor=anchor)
    return fnt


# ---------------------------------------------------------------------------
# Synthetic vision control
# ---------------------------------------------------------------------------

CONTROL_CODE = "6305"
CONTROL_CIRCLE_UV = (520, 80)     # centre of the red circle
CONTROL_SQUARE_UV = (200, 400)    # centre of the green square


def build_control_image() -> tuple[np.ndarray, dict]:
    """Deterministic 640x480 synthetic frame used to prove images reach the model.

    Returns (image_rgb, truth) where truth carries the code and the two shape
    centres in pixels.
    """
    h, w = 480, 640
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Four large digits in the upper-left quadrant.
    draw.text((30, 30), CONTROL_CODE, font=font(150, bold=True), fill=(0, 0, 0))

    # Red circle, upper-right.
    cx, cy = CONTROL_CIRCLE_UV
    r = 28
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 20, 20), outline=(0, 0, 0), width=2)

    # Green square, lower-left.
    sx, sy = CONTROL_SQUARE_UV
    half = 40
    draw.rectangle([sx - half, sy - half, sx + half, sy + half],
                   fill=(20, 150, 20), outline=(0, 0, 0), width=2)

    truth = {
        "code": CONTROL_CODE,
        "red_circle": list(CONTROL_CIRCLE_UV),
        "green_square": list(CONTROL_SQUARE_UV),
    }
    return np.array(img), truth


# ---------------------------------------------------------------------------
# Input style: grid overlay
# ---------------------------------------------------------------------------

def draw_grid_overlay(image_rgb: np.ndarray, step: int = 64) -> np.ndarray:
    """Overlay a labelled coordinate grid.  No fruit annotation of any kind."""
    img = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).copy()
    draw = ImageDraw.Draw(img, "RGBA")
    h, w = img.height, img.width

    for x in range(0, w, step):
        draw.line([(x, 0), (x, h)], fill=(0, 0, 0, 110), width=2)
        draw.line([(x, 0), (x, h)], fill=(255, 255, 255, 150), width=1)
    for y in range(0, h, step):
        draw.line([(0, y), (w, y)], fill=(0, 0, 0, 110), width=2)
        draw.line([(0, y), (w, y)], fill=(255, 255, 255, 150), width=1)

    for x in range(0, w, step):
        _text(draw, (x + 3, 3), str(x), size=13, fill=(255, 255, 0))
    for y in range(step, h, step):
        _text(draw, (3, y + 2), str(y), size=13, fill=(255, 255, 0))
    return np.array(img)


# ---------------------------------------------------------------------------
# Input style: few-shot exemplar composite
# ---------------------------------------------------------------------------

# Geometry of the composite, in global composite pixels.  The right (query)
# panel is a full 640x480 frame placed at this offset; the harness subtracts the
# offset to convert the model's global answer back into frame coordinates.
EXEMPLAR_LEFT_ORIGIN = (48, 56)
EXEMPLAR_RIGHT_ORIGIN = (808, 56)
EXEMPLAR_PANEL = (640, 480)
EXEMPLAR_GAP = (688, 808)  # x-range of the grey gutter between panels


def build_exemplar_composite(
    exemplar_rgb: np.ndarray,
    exemplar_uv: tuple[float, float],
    target_rgb: np.ndarray,
    box_offset_px: int = 30,
    box_half_px: int = 30,
) -> np.ndarray:
    """Two-panel image: labelled example (left) + unannotated query (right).

    The composite carries global-coordinate tick labels along its bottom and left
    edges so the model can read positions off the axes instead of doing offset
    arithmetic; the prompt states the right panel's global rectangle explicitly.
    """
    lx, ly = EXEMPLAR_LEFT_ORIGIN
    rx, ry = EXEMPLAR_RIGHT_ORIGIN
    pw, ph = EXEMPLAR_PANEL
    width = rx + pw + 22
    height = ry + ph + 64

    canvas = Image.new("RGB", (width, height), CANVAS_GREY)
    canvas.paste(Image.fromarray(exemplar_rgb), (lx, ly))
    canvas.paste(Image.fromarray(target_rgb), (rx, ry))
    draw = ImageDraw.Draw(canvas)

    # ---- example annotation (green, matches the "correct answer" convention) --
    u, v = exemplar_uv
    cx, cy = lx + u, ly + v
    bx0, by0 = lx + u - box_half_px, ly + v + box_offset_px - box_half_px
    bx1, by1 = lx + u + box_half_px, ly + v + box_offset_px + box_half_px
    draw.rectangle([bx0, by0, bx1, by1], outline=OK_COLOR, width=3)
    _text(draw, (bx0, by0 - 20), "fruit bbox", size=15, fill=OK_COLOR)

    arm = 22
    draw.line([(cx - arm, cy), (cx + arm, cy)], fill=OK_COLOR, width=3)
    draw.line([(cx, cy - arm), (cx, cy + arm)], fill=OK_COLOR, width=3)
    draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], outline=OK_COLOR, width=3)
    _text(draw, (cx + arm + 6, cy - 26),
          f"picking_point\n[{int(round(u))}, {int(round(v))}]",
          size=15, fill=OK_COLOR)

    # ---- panel titles -------------------------------------------------------
    _text(draw, (lx + pw // 2, 26), "EXAMPLE (annotated)", size=20,
          fill=(255, 255, 255), anchor="mm")
    _text(draw, (rx + pw // 2, 26), "YOUR FRAME (unannotated)", size=20,
          fill=(255, 255, 255), anchor="mm")
    _text(draw, ((EXEMPLAR_GAP[0] + EXEMPLAR_GAP[1]) // 2, ly + ph // 2),
          "→", size=44, fill=(255, 255, 255), anchor="mm")

    # ---- global coordinate axes --------------------------------------------
    for gx in range(0, width - 20, 128):
        draw.line([(gx, height - 24), (gx, height - 16)], fill=(255, 255, 255), width=2)
        _text(draw, (gx, height - 14), str(gx), size=13, fill=(255, 255, 255),
              anchor="ma")
    for gy in range(64, ry + ph, 64):
        draw.line([(30, gy), (38, gy)], fill=(255, 255, 255), width=2)
        _text(draw, (28, gy), str(gy), size=13, fill=(255, 255, 255), anchor="rm")
    _text(draw, (width - 20, height - 14), "x →", size=14, fill=(255, 255, 255),
          anchor="ra")

    return np.array(canvas)


def exemplar_query_offset() -> tuple[int, int]:
    """Global offset of the query panel inside the composite."""
    return EXEMPLAR_RIGHT_ORIGIN


# ---------------------------------------------------------------------------
# Diagnostics (never fed to the model)
# ---------------------------------------------------------------------------

def draw_gt_marker(image_rgb: np.ndarray, uv: tuple[float, float],
                   box_offset_px: int = 30, box_half_px: int = 30) -> np.ndarray:
    """Upstream-consistent GT rendering used only for visual sanity checks.

    Mirrors ``target_ref.draw_target_marker``'s convention: a rough box centred
    at ``(u, v + offset)`` whose top edge touches the picking point.
    """
    img = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).copy()
    draw = ImageDraw.Draw(img)
    u, v = uv
    cx, cy = int(round(u)), int(round(v + box_offset_px))
    draw.ellipse([cx - box_half_px, cy - box_half_px, cx + box_half_px, cy + box_half_px],
                 outline=(0, 0, 0), width=5)
    draw.ellipse([cx - box_half_px, cy - box_half_px, cx + box_half_px, cy + box_half_px],
                 outline=GT_COLOR, width=3)
    return np.array(img)


def draw_prediction_overlay(
    image_rgb: np.ndarray,
    gt_uv: tuple[float, float],
    pred_uv: tuple[float, float] | None,
    pred_bbox: tuple[float, float, float, float] | None,
    title: str,
    box_offset_px: int = 30,
    box_half_px: int = 30,
    header_px: int = 34,
) -> np.ndarray:
    """Per-run diagnostic: GT marker, prediction, bounding box and error vector.

    The title strip is added *above* the frame rather than drawn on top of it, so
    no part of the frame is ever hidden by the annotation.
    """
    frame = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).copy()
    h, w = frame.height, frame.width
    img = Image.new("RGB", (w, h + header_px), (0, 0, 0))
    img.paste(frame, (0, header_px))
    draw = ImageDraw.Draw(img)
    oy = header_px  # frame rows are offset by the strip

    # Reference box derived from the upstream label convention.
    cx, cy = int(round(gt_uv[0])), int(round(gt_uv[1] + box_offset_px)) + oy
    draw.ellipse([cx - box_half_px, cy - box_half_px, cx + box_half_px, cy + box_half_px],
                 outline=(0, 0, 0), width=5)
    draw.ellipse([cx - box_half_px, cy - box_half_px, cx + box_half_px, cy + box_half_px],
                 outline=GT_COLOR, width=3)

    if pred_bbox is not None:
        x1, y1, x2, y2 = [float(v) for v in pred_bbox]
        draw.rectangle([min(x1, x2), min(y1, y2) + oy, max(x1, x2), max(y1, y2) + oy],
                       outline=BOX_COLOR, width=3)

    gx, gy = int(round(gt_uv[0])), int(round(gt_uv[1])) + oy
    arm_gt = 16

    # Draw order matters: error vector, then the prediction, then the ground
    # truth LAST so the reference is never obscured by a close prediction.
    # A near-perfect prediction hiding partly under the reference is fine; not
    # being able to see the reference is not.
    if pred_uv is not None:
        px, py = float(pred_uv[0]), float(pred_uv[1]) + oy
        draw.line([(gx, gy), (px, py)], fill=(255, 255, 255), width=4)
        draw.line([(gx, gy), (px, py)], fill=(0, 0, 0), width=2)

        # The prediction is a HOLLOW ring plus a dot rather than a crosshair: a
        # ring leaves the reference visible through its middle.
        ring = 11
        draw.ellipse([px - ring, py - ring, px + ring, py + ring],
                     outline=(0, 0, 0), width=4)
        draw.ellipse([px - ring, py - ring, px + ring, py + ring],
                     outline=PRED_COLOR, width=2)
        draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=PRED_COLOR)
        # The error is a frame-space distance; the strip cancels out of it.
        error = float(np.hypot(px - gx, py - gy))
        _text(draw, (px + ring + 4, py + 6), f"{error:.1f}px", size=15, fill=PRED_COLOR)

    draw.line([(gx - arm_gt, gy), (gx + arm_gt, gy)], fill=GT_COLOR, width=3)
    draw.line([(gx, gy - arm_gt), (gx, gy + arm_gt)], fill=GT_COLOR, width=3)

    # Title strip so each overlay is self-describing.
    _text(draw, (6, header_px // 2), title, size=17, fill=(255, 255, 255), anchor="lm")
    _text(draw, (w - 6, header_px // 2), f"GT [{gx}, {gy - oy}]", size=15, fill=GT_COLOR,
          anchor="rm")
    return np.array(img)


def contact_sheet(images: list[np.ndarray], titles: list[str], cols: int = 3) -> np.ndarray:
    """Tile per-run overlays into a single sheet (each already has a title bar)."""
    if not images:
        raise ValueError("contact_sheet needs at least one image")
    n = len(images)
    cols = max(1, min(cols, n))
    rows = int(np.ceil(n / cols))
    ih, iw = images[0].shape[:2]
    sheet = np.full((rows * ih, cols * iw, 3), CANVAS_GREY, dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        patch = np.asarray(img, dtype=np.uint8)
        sheet[r * ih:r * ih + patch.shape[0], c * iw:c * iw + patch.shape[1]] = patch
    return sheet


def draw_multi_overlay(
    image_rgb: np.ndarray,
    strawberries: list[dict] | None,
    title: str,
    header_px: int = 34,
) -> np.ndarray:
    """Per-run diagnostic for an *unlabelled* multi-strawberry scene.

    There is no ground truth here, so the overlay shows only what the model
    reported: a box and a picking-point ring per strawberry, numbered so the
    report table can refer to them. A missing or empty prediction is still
    rendered (header only) so every run has an artefact.
    """
    frame = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).copy()
    h, w = frame.height, frame.width
    img = Image.new("RGB", (w, h + header_px), (0, 0, 0))
    img.paste(frame, (0, header_px))
    draw = ImageDraw.Draw(img)
    oy = header_px

    # Scale the annotation to the image so it stays legible on large frames.
    scale = max(1.0, min(h, w) / 720.0)
    width = max(2, int(round(3 * scale)))
    ring = int(round(11 * scale))

    for index, berry in enumerate(strawberries or [], start=1):
        bbox = berry.get("bbox")
        point = berry.get("picking_point")
        if bbox is not None:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            draw.rectangle([min(x1, x2), min(y1, y2) + oy, max(x1, x2), max(y1, y2) + oy],
                           outline=BOX_COLOR, width=width)
            _text(draw, (min(x1, x2), min(y1, y2) + oy - int(18 * scale)),
                  f"#{index}", size=int(15 * scale), fill=BOX_COLOR)
        if point is not None:
            px, py = float(point[0]), float(point[1]) + oy
            draw.ellipse([px - ring, py - ring, px + ring, py + ring],
                         outline=(0, 0, 0), width=width + 1)
            draw.ellipse([px - ring, py - ring, px + ring, py + ring],
                         outline=PRED_COLOR, width=width)
            draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=PRED_COLOR)

    count = len(strawberries or [])
    _text(draw, (6, header_px // 2), title, size=17, fill=(255, 255, 255), anchor="lm")
    _text(draw, (w - 6, header_px // 2), f"{count} reported", size=15,
          fill=PRED_COLOR if count else (255, 120, 120), anchor="rm")
    return np.array(img)


TARGET_COLOR = (0, 255, 255)      # cyan = the nominated pick target


def redness_colour(redness_pct: float | None) -> tuple[int, int, int]:
    """Continuous green -> amber -> red ramp for a reported redness value.

    The inventory asks for a number rather than a category, so the overlay encodes
    the number directly instead of snapping it back into buckets.
    """
    if redness_pct is None:
        return (200, 200, 200)
    t = max(0.0, min(100.0, float(redness_pct))) / 100.0
    if t < 0.5:                      # green -> amber
        k = t / 0.5
        return (int(60 + (255 - 60) * k), int(200 + (215 - 200) * k), int(60 * (1 - k)))
    k = (t - 0.5) / 0.5              # amber -> red
    return (255, int(215 * (1 - k)), int(0))


def draw_inventory_overlay(
    image_rgb: np.ndarray,
    strawberries: list[dict] | None,
    target_index: int | None,
    title: str,
    header_px: int = 34,
) -> np.ndarray:
    """Per-run diagnostic for a full-scene inventory.

    Boxes are coloured by reported ripeness, labelled with the fruit's index and
    its redness/occlusion, and drawn thinner the more occluded the model judged
    the fruit. The nominated pick target gets a cyan box and a target marker.
    Nothing here is ground truth.
    """
    frame = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).copy()
    h, w = frame.height, frame.width
    img = Image.new("RGB", (w, h + header_px), (0, 0, 0))
    img.paste(frame, (0, header_px))
    draw = ImageDraw.Draw(img)
    oy = header_px

    scale = max(1.0, min(h, w) / 720.0)
    text_size = int(14 * scale)
    berries = strawberries or []

    for index, berry in enumerate(berries):
        bbox = berry.get("bbox")
        if bbox is None:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        box = [min(x1, x2), min(y1, y2) + oy, max(x1, x2), max(y1, y2) + oy]
        colour = redness_colour(berry.get("redness_pct"))
        is_target = target_index is not None and index == target_index

        # Heavily occluded fruit is drawn thinner so the box stays readable
        # against the leaves that are hiding it.
        occlusion = berry.get("occlusion_pct")
        width = max(1, int(round((3 if occlusion is None or occlusion < 60 else 1) * scale)))
        draw.rectangle(box, outline=(0, 0, 0), width=width + 2)
        draw.rectangle(box, outline=TARGET_COLOR if is_target else colour, width=width)

        label = (f"#{index}"
                 + (f" TARGET" if is_target else "")
                 + (f" red {berry['redness_pct']}%" if berry.get("redness_pct") is not None else "")
                 + (f" occ {berry['occlusion_pct']}%" if berry.get("occlusion_pct") is not None else ""))
        _text(draw, (box[0], box[1] - int(17 * scale)), label, size=text_size,
              fill=TARGET_COLOR if is_target else colour)

        point = berry.get("picking_point")
        if point is not None:
            px, py = float(point[0]), float(point[1]) + oy
            ring = max(3, int(round(6 * scale)))
            draw.ellipse([px - ring, py - ring, px + ring, py + ring],
                         outline=(0, 0, 0), width=2)
            draw.ellipse([px - ring, py - ring, px + ring, py + ring],
                         outline=PRED_COLOR, width=2)

    _text(draw, (6, header_px // 2), title, size=17, fill=(255, 255, 255), anchor="lm")
    reds = [b.get("redness_pct") for b in berries
            if isinstance(b.get("redness_pct"), (int, float))]
    summary = (f"{len(berries)} found"
               + (f", redness {min(reds)}-{max(reds)}%" if reds else "")
               + (f", target #{target_index}" if target_index is not None and target_index >= 0
                  else ", no target"))
    _text(draw, (w - 6, header_px // 2), summary, size=15,
          fill=TARGET_COLOR if berries else (255, 120, 120), anchor="rm")
    return np.array(img)
