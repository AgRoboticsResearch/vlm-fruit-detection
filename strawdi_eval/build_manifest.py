#!/usr/bin/env python3
"""Build the StrawDI detection-eval manifest from the StrawDI_Db1 dataset.

The dataset ships RGB frames plus per-instance segmentation masks (grayscale
PNGs: 0 = background, 1..N = instance id, one class "strawberry"). This
script derives the ground truth the eval scores against — one tight,
axis-aligned bounding box per instance — copies the frames into the repo as an
immutable snapshot, and writes ``manifest.json`` with full provenance.

Invariants inherited from the base harness (``vlm_eval/AGENTS.md``):

* ground truth is *derived*, never hand-edited: the verifier re-runs
  :func:`derive_instances` over the same mask PNGs and must reproduce every
  box in the manifest bit-for-bit;
* inputs stay inside the CLI's no-resize ceiling: StrawDI frames are
  1008x756 (< 1280), so they are copied byte-for-byte and never resampled —
  a larger future split is a hard error, not a silent downscale;
* no annotated image is ever produced here: the masks are read, the frames
  are copied untouched, and nothing drawn ever reaches the model.

Usage (repo root):
    python3 strawdi_eval/build_manifest.py [--limit 30] [--split val]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from vlm_eval.lib import gtbridge, imaging  # noqa: E402
from vlm_eval import prompts  # noqa: E402

DEFAULT_DATASET_ROOT = Path("/media/zfei/GLOWAY/strawberry_detection/StrawDI_Db1")
SPLITS = ("train", "val", "test")

# The largest max edge verified to pass through the CLI untouched is 1920; the
# harness policy normalises to 1280. StrawDI is 1008, comfortably inside, so
# frames are delivered unchanged and GT coordinates map 1:1.
MAX_EDGE = 1280

BOX_CONVENTION = ("tight axis-aligned bbox of the instance's mask pixels, "
                  "half-open [x1, y1, x2, y2): x2 = mask_xmax + 1, y2 = mask_ymax + 1, "
                  "so pixel extent and box extent agree")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--split", choices=SPLITS, default="val")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first k of an evenly-spaced selection "
                         "(deterministic; indices recorded in the manifest)")
    ap.add_argument("--out", type=Path, default=HERE / "manifest.json")
    ap.add_argument("--data-dir", type=Path, default=HERE / "data")
    args = ap.parse_args(argv)
    args.dataset_root = args.dataset_root.resolve()
    args.out = args.out.resolve()
    args.data_dir = args.data_dir.resolve()
    return args


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_instances(mask: np.ndarray) -> list[dict]:
    """One tight box per instance id in a StrawDI label mask.

    Returns dicts sorted by instance id:
    ``{"instance_id", "bbox_xyxy": [x1, y1, x2, y2], "area_px"}`` with the
    half-open convention documented in ``BOX_CONVENTION``. This is the single
    source of truth for GT — the verifier imports this same function.
    """
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {mask.shape}")
    instances = []
    ids = sorted(int(v) for v in np.unique(mask) if v != 0)
    for instance_id in ids:
        ys, xs = np.where(mask == instance_id)
        instances.append({
            "instance_id": instance_id,
            "bbox_xyxy": [int(xs.min()), int(ys.min()),
                          int(xs.max()) + 1, int(ys.max()) + 1],
            "area_px": int(len(xs)),
        })
    return instances


def list_split(root: Path, split: str) -> list[dict]:
    """Pair <root>/<split>/img/*.png with <root>/<split>/label/<stem>.png."""
    img_dir, label_dir = root / split / "img", root / split / "label"
    for directory in (img_dir, label_dir):
        if not directory.is_dir():
            raise SystemExit(f"missing dataset directory: {directory}")
    images = sorted(img_dir.glob("*.png"))
    if not images:
        raise SystemExit(f"no PNG frames in {img_dir}")
    pairs = []
    for image in images:
        label = label_dir / image.name
        if not label.exists():
            raise SystemExit(f"label mask missing for frame: {image}")
        pairs.append({"stem": image.stem, "image": image, "label": label})
    return pairs


def verify_frame(image: Image.Image, path: Path) -> tuple[int, int]:
    """Enforce the delivery ceiling. StrawDI never trips this; a future,
    larger split must fail loudly rather than be silently resized."""
    w, h = image.size
    if max(w, h) > MAX_EDGE:
        raise SystemExit(
            f"{path.name} is {w}x{h}, over the {MAX_EDGE}px delivery ceiling. "
            "Normalise the frames (LANCZOS, record both shapes) before running "
            "the eval — never let the CLI resize them silently.")
    return w, h


def build_control(data_dir: Path) -> dict:
    control_dir = data_dir / "control"
    control_dir.mkdir(parents=True, exist_ok=True)
    rgb, truth = imaging.build_control_image()
    path = control_dir / "control_6305.png"
    imaging.save_rgb(rgb, path)
    return {
        "image": str(path.relative_to(HERE)),
        "truth": truth,
        "prompt": prompts.CONTROL_PROMPT,
        "purpose": ("synthetic vision-delivery gate: if the model cannot read "
                    "the code and shape centres off this image, every number "
                    "in the batch is invalid"),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    pairs = list_split(args.dataset_root, args.split)

    indices = list(range(len(pairs)))
    selection = "all"
    if args.limit and args.limit < len(pairs):
        indices = gtbridge.even_spacing_indices(len(pairs), args.limit)
        selection = "even_spacing"

    frames_dir = args.data_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    all_areas = []
    for index in indices:
        pair = pairs[index]
        with Image.open(pair["image"]) as img:
            img = img.convert("RGB")
            w, h = verify_frame(img, pair["image"])
        dst = frames_dir / f"{pair['stem']}.png"
        shutil.copyfile(pair["image"], dst)

        with Image.open(pair["label"]) as mask_img:
            mask = np.array(mask_img)
        if mask.dtype != np.uint8:
            raise SystemExit(f"{pair['label'].name}: unexpected mask dtype "
                             f"{mask.dtype} (uint8 instance ids expected)")
        if mask.shape != (h, w):
            raise SystemExit(f"{pair['label'].name}: mask {mask.shape} does not "
                             f"match frame {(h, w)}")
        instances = derive_instances(mask)
        areas = [inst["area_px"] for inst in instances]
        all_areas.extend(areas)

        samples.append({
            "sample_id": pair["stem"],
            "source": f"strawdi_{args.split}",
            "split": args.split,
            "has_gt": True,
            "task": "full_inventory",
            "image": str(dst.relative_to(HERE)),
            "images": {"raw": str(dst.relative_to(HERE))},
            "image_shape_hw": [h, w],
            "source_image": str(pair["image"]),
            "source_image_shape_hw": [h, w],
            "downscaled": False,
            "label_image": str(pair["label"]),
            "label_sha256": sha256(pair["label"]),
            "frame_sha256": sha256(dst),
            "n_gt": len(instances),
            "gt_boxes": [inst["bbox_xyxy"] for inst in instances],
            "gt_areas": areas,
            "gt_instance_ids": [inst["instance_id"] for inst in instances],
            "label_provenance": {
                "method": BOX_CONVENTION,
                "min_instance_area_px": None,
                "note": "every instance in the mask is kept, however small; "
                        "areas are recorded so metrics can stratify by size",
            },
        })

    manifest = {
        "schema_version": "strawdi_eval/1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "generator": {
            "script": str(Path(__file__).relative_to(HERE)),
            "argv": sys.argv[1:],
            "sha256": sha256(Path(__file__)),
        },
        "python": sys.version.split()[0],
        "dataset": {
            "name": "StrawDI_Db1",
            "url": "https://strawdi.github.io/",
            "root": str(args.dataset_root),
            "split": args.split,
            "n_total": len(pairs),
            "n_selected": len(samples),
            "selection": selection,
            "selection_indices": indices,
            "frame_policy": {
                "max_edge": MAX_EDGE,
                "delivered": "unchanged (1008x756 is inside the ceiling)",
                "note": "frames copied byte-for-byte; no resampling, so GT "
                        "coordinates map 1:1 onto the delivered image",
            },
        },
        "coordinate_convention": {
            "origin": "top-left, x right, y down (matches the prompt)",
            "bbox": BOX_CONVENTION,
            "semantics_gap": ("the prompt asks for the WHOLE-fruit box "
                              "(including parts hidden behind occluders) while "
                              "GT boxes cover the VISIBLE mask surface only — "
                              "occluded-fruit predictions are expected to "
                              "exceed the GT box and lose IoU"),
        },
        "control": build_control(args.data_dir),
        "statistics": {
            "n_instances_total": len(all_areas),
            "instances_per_image": {
                "min": min(s["n_gt"] for s in samples),
                "max": max(s["n_gt"] for s in samples),
                "mean": round(len(all_areas) / len(samples), 2),
            },
            "instance_area_px": {
                "min": int(min(all_areas)),
                "p25": int(np.percentile(all_areas, 25)),
                "median": int(np.percentile(all_areas, 50)),
                "p75": int(np.percentile(all_areas, 75)),
                "max": int(max(all_areas)),
            },
        },
        "samples": samples,
    }

    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    areas = manifest["statistics"]["instance_area_px"]
    print(f"manifest : {args.out}")
    print(f"frames   : {len(samples)} of {len(pairs)} ({args.split}) "
          f"copied to {frames_dir}")
    print(f"instances: {len(all_areas)} total, "
          f"{manifest['statistics']['instances_per_image']['mean']}/image, "
          f"areas {areas['min']}-{areas['max']} px "
          f"(median {areas['median']})")


if __name__ == "__main__":
    main()
