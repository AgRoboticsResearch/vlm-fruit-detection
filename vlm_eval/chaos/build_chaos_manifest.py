#!/usr/bin/env python3
"""Build ``vlm_eval/chaos/manifest.json`` — the curated chaos-scene frame list.

The chaos_strawberry_detection pipeline runs the segmentation inventory
(``inventory_segmentation``: the ten-field census incl. ``polygon``) on a
CURATED set of deliberately chaotic strawberry scenes — dense fruit, heavy
clutter, deep occlusion — rather than on a dataset selection. There is no
ground truth on any chaos scene: the pipeline is qualitative + diagnostics,
same as the shunba scope of its siblings.

Each entry names the ORIGINAL image and a delivery rule; the builder
materialises the delivered frame under ``vlm_eval/data/frames/`` as
``chaos__<sample_id>.png`` (model inputs stay PNG) and records the sha256 of
both the original and the delivered file, so a chaos run is reproducible
even if the source store changes. Frames added via ``--frame name=path``
are normalised to fit inside 1280x720 (the pipeline's "720P" rule: preserve
aspect, never exceed 1280x720 — a 16:9 source lands exactly 1280x720, a 4:3
source lands 960x720).

Deterministic: no RNG, no timestamps beyond ``generated_at``; the curated
list lives in CHAOS_FRAMES below (adding a scene = one entry + rebuild).

Usage (repo root):
    python3 vlm_eval/chaos/build_chaos_manifest.py
    python3 vlm_eval/chaos/build_chaos_manifest.py --frame IMG_7701=vlm_eval/data/frames/IMG_7701.jpeg
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import sys
from pathlib import Path

from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent          # vlm_eval/chaos
VLM = HERE.parent                                # vlm_eval
REPO = VLM.parent
sys.path.insert(0, str(REPO))

from vlm_eval.lib import imaging  # noqa: E402

# The pipeline's delivery ceiling: fit inside 1280x720 preserving aspect.
FIT_W, FIT_H = 1280, 720

# The curated chaos scenes (sample_id, original image, delivery rule).
#   rule None   -> deliver the original bytes unchanged (already 720p);
#   rule (w, h) -> fit inside (w, h) preserving aspect, LANCZOS.
# Both are sha256-recorded; the delivered frame is a PNG copy named
# chaos__<sample_id>.png so later base-pipeline rebuilds can never silently
# change what a chaos sample_id means.
CHAOS_FRAMES: list[tuple[str, Path, tuple[int, int] | None, str]] = [
    ("sb04", VLM / "data" / "frames" / "sb04.png", None,
     "shunba scene 0000093.jpg, already normalised to 1280x720 (= 720p) by "
     "the base pipeline's manifest builder; the densest shunba frame "
     "(19 reported fruits in the 2026-09-16 vlm_seg run)"),
    ("IMG_7665", VLM / "data" / "frames" / "IMG_7665.jpeg", (FIT_W, FIT_H),
     "hand-held photo, stored 4032x3024 with EXIF orientation 6 (portrait "
     "3024x4032 once transposed); fitted inside 1280x720 preserving aspect "
     "-> 540x720 for this pipeline"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_manifest_path(path: Path) -> str:
    """Store a path relative to vlm_eval/ when it lives inside it, else absolute."""
    try:
        return str(path.relative_to(VLM))
    except ValueError:
        return str(path)


def build_sample(sample_id: str, original: Path, rule: tuple[int, int] | None,
                 note: str) -> dict:
    if not original.is_file():
        raise SystemExit(f"chaos source image not found: {original}")
    with Image.open(original) as img:
        # Upright orientation first: hand-held photos carry an EXIF orientation
        # tag that PIL does NOT apply on open — without this, a portrait photo
        # arrives at the model sideways.
        exif = img.getexif()
        orientation = exif.get(274) if exif else None
        stored_hw = [img.height, img.width]
        transposed = ImageOps.exif_transpose(img)
        if transposed is not None:
            img = transposed
        rgb = img.convert("RGB")
        original_hw = [rgb.height, rgb.width]
        delivered_hw = original_hw
        if rule is not None:
            fit_w, fit_h = rule
            scale = min(fit_w / rgb.width, fit_h / rgb.height)
            if scale < 1.0:
                rgb = rgb.resize((max(1, round(rgb.width * scale)),
                                  max(1, round(rgb.height * scale))), Image.LANCZOS)
                delivered_hw = [rgb.height, rgb.width]

    delivered = VLM / "data" / "frames" / f"chaos__{sample_id}.png"
    delivered.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(delivered)
    return {
        "sample_id": sample_id,
        "source": "chaos",
        "has_gt": False,
        "task": "full_inventory",
        "session": "chaos",
        "episode": original.stem,
        "scene_note": note,
        "source_image": str(original),
        "source_sha256": sha256(original),
        "image": as_manifest_path(delivered),
        "delivered_sha256": sha256(delivered),
        "image_shape_hw": delivered_hw,
        "source_image_shape_hw": original_hw,
        "stored_image_shape_hw": stored_hw,
        "exif_orientation": orientation,
        "downscaled": original_hw != delivered_hw,
        "normalisation": ("none (delivered unchanged)" if rule is None else
                          f"fitted inside {rule[0]}x{rule[1]} preserving aspect, "
                          "LANCZOS")
        + ("" if orientation in (None, 1) else
           f"; EXIF orientation {orientation} applied first (upright)"),
        "gt_status": "none",
        "gt_uv": None,
        "gt_uv_rounded": None,
        "gt_z_m": None,
        "gt_rough_box": None,
        "label_provenance": (
            "none: curated chaos scene, no demonstrations, no trajectory and "
            "no ground-truth point or mask. Reported qualitatively only; "
            "excluded from every scored metric."
        ),
        "images": {"raw": as_manifest_path(delivered)},
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", action="append", default=[], metavar="NAME=PATH",
                    help="add (or replace) a curated chaos scene; the original "
                         "image is normalised to fit inside 1280x720 preserving "
                         "aspect. Repeatable.")
    ap.add_argument("--out", type=Path, default=HERE / "manifest.json")
    args = ap.parse_args()
    args.out = args.out.resolve()
    return args


def main() -> None:
    args = parse_args()
    frames = list(CHAOS_FRAMES)
    for spec in args.frame:
        name, _, path = spec.partition("=")
        if not name or not path:
            raise SystemExit(f"--frame expects NAME=PATH, got {spec!r}")
        entry = (name, Path(path), (FIT_W, FIT_H), "CLI-added scene; fitted "
                 "inside 1280x720 preserving aspect")
        frames = [f for f in frames if f[0] != name] + [entry]

    samples = [build_sample(*frame) for frame in frames]

    # The synthetic vision control is the base harness's (deterministic image;
    # regenerated here so this manifest is self-sufficient).
    control_img, control_truth = imaging.build_control_image()
    control_path = VLM / "data" / "control" / "control_6305.png"
    imaging.save_rgb(control_img, control_path)

    manifest = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "generator": "vlm_eval/chaos/build_chaos_manifest.py",
        "python": platform.python_version(),
        "repo": str(REPO),
        "harness_dir": str(HERE),
        "pipeline": "chaos_strawberry_detection",
        "purpose": (
            "detection + visible-surface segmentation on curated CHAOTIC "
            "strawberry scenes; no ground truth exists on any chaos scene, so "
            "every result is qualitative + internal-consistency diagnostics"
        ),
        "delivery_rule": (
            f"model inputs fit inside {FIT_W}x{FIT_H} preserving aspect "
            "(the '720P' rule); model inputs stay PNG"
        ),
        "control": {
            "image": as_manifest_path(control_path),
            "truth": control_truth,
            "purpose": "proves the CLI actually delivered an image to the model "
                       "before any result is reported",
        },
        "samples": samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {args.out}")
    for s in samples:
        print(f"  {s['sample_id']:<10} {s['source_image_shape_hw'][1]}x"
              f"{s['source_image_shape_hw'][0]} -> {s['image_shape_hw'][1]}x"
              f"{s['image_shape_hw'][0]}  {s['image']}")


if __name__ == "__main__":
    main()
