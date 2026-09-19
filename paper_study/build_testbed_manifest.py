#!/usr/bin/env python3
"""Build the paper_study mixed testbed manifest.

Curates THREE frames per dataset (all sources except the unfinished KFuji)
from each dataset's TEST split, reads their ground truth, normalises delivery
frames to the 1280 px no-resize ceiling, and writes ``manifest.json`` +
``data/frames/``. The datasets themselves stay on the removable mount and are
read only here and AFTER the model calls at score time (sha256-guarded) —
label files are never model inputs.

Selection rule (deterministic, recorded in the manifest): from each source's
test split, keep frames with at least MIN_GT ground-truth instances (WGISD:
also require the cluster-mask ``.npz``), sort by file name, pick three evenly
spaced indices.

GT conventions verified empirically 2026-09-18 (see lib/gtload.py docstring):
ACFR circles are centre+radius, ACFR rectangles top-left+size, WGISD txt
YOLO-normalised with npz mask i ↔ box line i, MinneApple COCO polygons may
carry several rings per instance, StrawDI labels are id-maps.

Usage (from the repo root):

    python3 paper_study/build_testbed_manifest.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent          # paper_study
REPO = HERE.parent                              # repo root
sys.path.insert(0, str(REPO / "blog_study"))
sys.path.insert(0, str(REPO))

from paper_study.lib import gtload  # noqa: E402  (repo-root import)

GLOWAY = Path("/media/zfei/GLOWAY/strawberry_detection")
EXTRA = GLOWAY / "extra_fruits"
ACFR = EXTRA / "acfr_multifruit" / "extracted" / "acfr-fruit-dataset"
MINNEAPPLE = EXTRA / "minneapple" / "hf_coco_mirror"
WGISD = EXTRA / "wgisd"
STRAWDI = GLOWAY / "StrawDI_Db1"
KFUJI = EXTRA / "kfuji_rgb_ds" / "extracted" / "preprocessed data"

MAX_EDGE = 1280        # the base harness's no-resize ceiling (invariant I6)
PER_SOURCE = 3          # default; --per-source overrides (testbed20 etc.)
FRAMES_DIR = HERE / "data" / "frames"   # default; main() retargets per --per-source
MIN_GT = 3

SCHEMA_VERSION = "paper_testbed/1"
HARNESS_NAME = "paper_fruit_seg"

# Protocol framing per source (what the dataset's own literature scores).
# Metric names follow each dataset's convention; the run report maps our
# measured numbers onto these.
PROTOCOL = {
    "strawdi": {
        "task": "strawberry instance segmentation (visible-surface masks); "
                "detection as mask-derived boxes",
        "gt_unit": "visible-surface mask per strawberry",
        "reference_metrics": "mask F1@IoU-0.5 (+ mask AP@50, mAP; box F1@0.5)",
    },
    "minneapple": {
        "task": "apple detection (COCO boxes) + instance segmentation "
                "(COCO polygons)",
        "gt_unit": "apple instance (polygon; occluded apples included)",
        "reference_metrics": "COCO-style AP@[.50:.95] + AP@50 (detection and "
                             "segmentation); F1@0.5 reported for the "
                             "single-model VLM setting",
    },
    "wgisd": {
        "task": "grape-cluster detection (boxes); instance segmentation on the "
                "masked subset (cluster masks)",
        "gt_unit": "grape cluster (bunch), not berry",
        "reference_metrics": "mAP@0.5 (bunch detection); F1@0.5 for the VLM "
                             "setting",
    },
    "acfr_apples": {
        "task": "apple detection (circle annotations read as boxes)",
        "gt_unit": "apple (circle -> square box)",
        "reference_metrics": "object-wise F1 (IoU 0.5) per Bargoti & Underwood "
                             "2017 convention",
    },
    "acfr_mangoes": {
        "task": "mango detection (rectangles)",
        "gt_unit": "mango (box)",
        "reference_metrics": "object-wise F1 (IoU 0.5)",
    },
    "acfr_almonds": {
        "task": "almond detection (rectangles)",
        "gt_unit": "almond fruit/hull (box)",
        "reference_metrics": "object-wise F1 (IoU 0.5)",
    },
    "kfuji": {
        "task": "Fuji apple detection (RGB patches of Kinect v2 frames; "
                "RGB-D variants exist but this study is RGB-only)",
        "gt_unit": "apple (box, top-left squares)",
        "reference_metrics": "P/R/F1 (frame-level, Gené-Mola et al. 2019 "
                             "convention) + AP; F1@0.5 here",
    },
}


def pick_evenly(names: list[str]) -> list[str]:
    n = len(names)
    return [names[round(i * (n - 1) / (PER_SOURCE - 1))] for i in range(PER_SOURCE)]


def normalise_frame(src: Path, dst: Path) -> tuple[tuple[int, int], bool]:
    """Copy the frame, downscaling only if it exceeds the ceiling.

    Returns ((w, h) delivered, downscaled?). The GT is rescaled by the SAME
    factor by the caller (scale = delivered_edge / source_edge).
    """
    with Image.open(src) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        scale = min(1.0, MAX_EDGE / max(w, h))
        if scale < 1.0:
            rgb = rgb.resize((round(w * scale), round(h * scale)),
                             Image.Resampling.LANCZOS)
        rgb.save(dst)
        return rgb.size, scale < 1.0


# --------------------------------------------------------------------------
# per-source sampling
# --------------------------------------------------------------------------

def source_strawdi() -> list[dict]:
    test_dir = STRAWDI / "test"
    names = sorted(p.name for p in (test_dir / "img").glob("*.png"))
    counts = {p.stem: len([v for v in np.unique(np.array(Image.open(p))) if v != 0])
              for p in (test_dir / "label").glob("*.png")}
    names = [n for n in names if counts.get(Path(n).stem, 0) >= MIN_GT]
    picked = pick_evenly(names)
    samples = []
    for name in picked:
        stem = Path(name).stem
        src_img = test_dir / "img" / name
        src_lab = test_dir / "label" / name
        dst = FRAMES_DIR / f"strawdi-{stem}.png"
        (wh, downscaled) = normalise_frame(src_img, dst)
        s = wh[0] / Image.open(src_img).size[0]
        mask = np.array(Image.open(src_lab))
        ids = sorted(int(v) for v in np.unique(mask) if v != 0)
        boxes, areas = [], []
        for i in ids:
            ys, xs = np.where(mask == i)
            x1, y1 = int(round(xs.min() * s)), int(round(ys.min() * s))
            x2, y2 = int(round(xs.max() * s)), int(round(ys.max() * s))
            boxes.append([min(x1, wh[0] - 1), min(y1, wh[1] - 1),
                          min(x2, wh[0] - 1), min(y2, wh[1] - 1)])
            areas.append(float((mask == i).sum()) * s * s)   # mask-pixel area
        samples.append({
            "sample_id": f"strawdi-{stem}",
            "source": "strawdi", "split": "test", "has_gt": True,
            "image": f"{FRAMES_DIR.relative_to(HERE).as_posix()}/strawdi-{stem}.png",
            "images": {"raw": f"{FRAMES_DIR.relative_to(HERE).as_posix()}/strawdi-{stem}.png"},
            "image_shape_hw": [wh[1], wh[0]],
            "source_image": str(src_img),
            "source_image_shape_hw": list(reversed(Image.open(src_img).size)),
            "downscaled": downscaled,
            "frame_sha256": gtload.sha256_file(dst),
            "n_gt": len(ids),
            "gt_boxes": boxes,
            "gt_areas": areas,
            "gt_masks": {
                "kind": "strawdi_idmap",
                "label_image": str(src_lab),
                "label_sha256": gtload.sha256_file(src_lab),
                "scale": s,
            },
        })
    return samples


def source_minneapple() -> list[dict]:
    coco = MINNEAPPLE / "test" / "test.json"
    by_file = gtload.minneapple_by_file(coco)
    names = sorted(n for n, inst in by_file.items() if len(inst) >= MIN_GT)
    picked = pick_evenly(names)
    coco_sha = gtload.sha256_file(coco)
    samples = []
    for name in picked:
        src_img = MINNEAPPLE / "test" / "images" / name
        stem = Path(name).stem
        dst = FRAMES_DIR / f"minneapple-{stem}.png"
        (wh, _downscaled) = normalise_frame(src_img, dst)
        instances = by_file[name]
        samples.append({
            "sample_id": f"minneapple-{stem}",
            "source": "minneapple", "split": "test", "has_gt": True,
            "image": f"{FRAMES_DIR.relative_to(HERE).as_posix()}/minneapple-{stem}.png",
            "images": {"raw": f"{FRAMES_DIR.relative_to(HERE).as_posix()}/minneapple-{stem}.png"},
            "image_shape_hw": [wh[1], wh[0]],
            "source_image": str(src_img),
            "source_image_shape_hw": list(reversed(Image.open(src_img).size)),
            "downscaled": False,
            "frame_sha256": gtload.sha256_file(dst),
            "n_gt": len(instances),
            "gt_boxes": [inst["bbox"] for inst in instances],
            "gt_areas": [inst["area"] for inst in instances],
            "gt_masks": {
                "kind": "coco_polygons",
                "coco_json": str(coco),
                "coco_json_sha256": coco_sha,
                "file_name": name,
                "polygons": [inst["polygons"] for inst in instances],
            },
        })
    return samples


def source_wgisd() -> list[dict]:
    test = json.loads((WGISD / "coco_annotations" / "test_bbox_instances.json").read_text())
    test_names = sorted(Path(im["file_name"]).name for im in test["images"])
    with_npz = [n for n in test_names if (WGISD / "data" / f"{Path(n).stem}.npz").exists()]
    # instance count from the .txt lines
    with_npz = [n for n in with_npz
                if len([l for l in (WGISD / "data" / f"{Path(n).stem}.txt").read_text().splitlines()
                        if l.strip()]) >= MIN_GT]
    picked = pick_evenly(with_npz)
    samples = []
    for name in picked:
        stem = Path(name).stem
        src_img = WGISD / "data" / name
        src_txt = WGISD / "data" / f"{stem}.txt"
        src_npz = WGISD / "data" / f"{stem}.npz"
        dst = FRAMES_DIR / f"wgisd-{stem}.png"
        (wh, downscaled) = normalise_frame(src_img, dst)
        src_wh = Image.open(src_img).size
        boxes = gtload.wgisd_boxes(src_txt, src_wh[0], src_wh[1])
        boxes = [gtload.scale_box(b, wh[0] / src_wh[0]) for b in boxes]
        with np.load(src_npz) as npz:
            n_clusters = npz["arr_0"].shape[2]
        if n_clusters != len(boxes):
            raise RuntimeError(f"{stem}: npz has {n_clusters} masks but txt has "
                               f"{len(boxes)} boxes")
        # mask areas in delivered space, recomputed at load time; approximate
        # here from the box areas for the manifest's gt_areas
        samples.append({
            "sample_id": f"wgisd-{stem}",
            "source": "wgisd", "split": "test", "has_gt": True,
            "image": f"{FRAMES_DIR.relative_to(HERE).as_posix()}/wgisd-{stem}.png",
            "images": {"raw": f"{FRAMES_DIR.relative_to(HERE).as_posix()}/wgisd-{stem}.png"},
            "image_shape_hw": [wh[1], wh[0]],
            "source_image": str(src_img),
            "source_image_shape_hw": list(reversed(src_wh)),
            "downscaled": downscaled,
            "frame_sha256": gtload.sha256_file(dst),
            "n_gt": len(boxes),
            "gt_boxes": boxes,
            "gt_areas": [float((b[2] - b[0]) * (b[3] - b[1])) for b in boxes],
            "gt_masks": {
                "kind": "wgisd_npz",
                "npz_image": str(src_npz),
                "npz_sha256": gtload.sha256_file(src_npz),
                "scale": wh[0] / src_wh[0],
            },
        })
    return samples


def source_kfuji() -> list[dict]:
    """KFuji RGB-DS: preprocessed 548x373 patches, RGBhr variant, box GT.

    The dataset's own split files (sets/test.txt, 193 patches) define the
    protocol split; annotations are the same ACFR-lineage top-left squares.
    """
    names = [n.strip() for n in (KFUJI / "sets" / "test.txt").read_text().splitlines()
             if n.strip()]
    names = [n for n in names if len(gtload.kfuji_boxes(KFUJI / "annotations" / f"{n}.csv")) >= MIN_GT]
    picked = pick_evenly(names)
    samples = []
    for entry in picked:
        src_img = KFUJI / "images" / f"{entry.replace('_RGB', '_RGBhr')}.jpg"
        dst = FRAMES_DIR / f"kfuji-{entry}.png"
        (wh, downscaled) = normalise_frame(src_img, dst)
        boxes = gtload.kfuji_boxes(KFUJI / "annotations" / f"{entry}.csv")
        samples.append({
            "sample_id": f"kfuji-{entry}",
            "source": "kfuji", "split": "test", "has_gt": True,
            "image": f"{FRAMES_DIR.relative_to(HERE).as_posix()}/kfuji-{entry}.png",
            "images": {"raw": f"{FRAMES_DIR.relative_to(HERE).as_posix()}/kfuji-{entry}.png"},
            "image_shape_hw": [wh[1], wh[0]],
            "source_image": str(src_img),
            "source_image_shape_hw": list(reversed(Image.open(src_img).size)),
            "downscaled": downscaled,
            "frame_sha256": gtload.sha256_file(dst),
            "n_gt": len(boxes),
            "gt_boxes": boxes,
            "gt_areas": [float((b[2] - b[0]) * (b[3] - b[1])) for b in boxes],
            "gt_masks": {"kind": "none",
                         "note": "KFuji ships box annotations only (plus depth "
                                 "point clouds) — polygons asked, unscored."},
        })
    return samples


def source_acfr(fruit: str, kind: str) -> list[dict]:
    names = [n.strip() for n in (ACFR / fruit / "sets" / "test.txt").read_text().splitlines()
             if n.strip()]
    names = [f"{n}.csv" for n in names]
    names = [n for n in names if len(gtload.acfr_rows(ACFR / fruit / "annotations" / n)) >= MIN_GT]
    picked = pick_evenly(names)
    samples = []
    for csv_name in picked:
        stem = csv_name[:-4]
        src_img = ACFR / fruit / "images" / f"{stem}.png"
        dst = FRAMES_DIR / f"acfr_{fruit}-{stem}.png"
        (wh, downscaled) = normalise_frame(src_img, dst)
        boxes = gtload.acfr_boxes(ACFR / fruit / "annotations" / csv_name, kind)
        samples.append({
            "sample_id": f"acfr_{fruit}-{stem}",
            "source": f"acfr_{fruit}", "split": "test", "has_gt": True,
            "image": f"{FRAMES_DIR.relative_to(HERE).as_posix()}/acfr_{fruit}-{stem}.png",
            "images": {"raw": f"{FRAMES_DIR.relative_to(HERE).as_posix()}/acfr_{fruit}-{stem}.png"},
            "image_shape_hw": [wh[1], wh[0]],
            "source_image": str(src_img),
            "source_image_shape_hw": list(reversed(Image.open(src_img).size)),
            "downscaled": downscaled,
            "frame_sha256": gtload.sha256_file(dst),
            "n_gt": len(boxes),
            "gt_boxes": boxes,
            "gt_areas": [float((b[2] - b[0]) * (b[3] - b[1])) for b in boxes],
            "gt_masks": {"kind": "none",
                         "note": "ACFR apples ship a SEMANTIC (non-instance) "
                                 "pixel mask; mangoes/almonds none. Instance "
                                 "GT is boxes only — polygons are asked but "
                                 "unscored, as in the base vlm_seg pipeline."},
        })
    return samples


def main() -> None:
    import argparse
    global PER_SOURCE, FRAMES_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-source", type=int, default=3,
                    help="frames per dataset (default 3; N != 3 writes "
                         "manifest{N}.json + data/frames{N}/ so the default "
                         "testbed stays intact)")
    args = ap.parse_args()
    PER_SOURCE = args.per_source
    suffix = "" if PER_SOURCE == 3 else str(PER_SOURCE)
    FRAMES_DIR = HERE / "data" / f"frames{suffix}"
    out_path = HERE / f"manifest{suffix}.json"

    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    FRAMES_DIR.mkdir(parents=True)
    control_src = REPO / "blog_study" / "vlm_eval" / "data" / "control" / "control_6305.png"
    (HERE / "data" / "control").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(control_src, HERE / "data" / "control" / "control_6305.png")

    samples = (source_strawdi()
               + source_minneapple()
               + source_wgisd()
               + source_acfr("apples", "circles")
               + source_acfr("mangoes", "rects")
               + source_acfr("almonds", "rects")
               + source_kfuji())

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "harness": HARNESS_NAME,
        "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "generator": "paper_study/build_testbed_manifest.py",
        "dataset": {
            "description": f"mixed cross-fruit testbed: {PER_SOURCE} frames per "
                           "dataset, each dataset's own TEST split, GT per its "
                           "protocol",
            "roots": {"gloway": str(GLOWAY)},
            "per_source": PER_SOURCE,
            "min_gt": MIN_GT,
            "selection_rule": f"per source: test split, keep frames with >= {MIN_GT} "
                              "GT instances (wgisd: also require cluster-mask .npz), "
                              "sort by file name, pick {PER_SOURCE} evenly spaced "
                              "indices",
            "protocols": PROTOCOL,
            "skipped": {},
        },
        "coordinate_convention": {
            "bbox": "[x1, y1, x2, y2] whole pixels, top-left origin, "
                    "delivered-frame space",
            "gt_conventions": {
                "acfr_apples": "circles (c-x, c-y, radius) -> square box",
                "acfr_mangoes": "rectangles (x, y = TOP-LEFT, dx, dy = size)",
                "acfr_almonds": "rectangles (x, y = TOP-LEFT, dx, dy = size)",
                "wgisd": "YOLO-normalised txt boxes; npz mask i == txt line i",
                "minneapple": "COCO bbox + polygon rings (instance may have >1 ring)",
                "strawdi": "grayscale id-map label PNG (1..N instances)",
            },
        },
        "control": {
            "image": "data/control/control_6305.png",
            "truth": {"code": "6305", "red_circle": [520, 80], "green_square": [200, 400]},
        },
        "statistics": {
            "n_samples": len(samples),
            "by_source": {},
        },
        "samples": samples,
    }
    for s in samples:
        manifest["statistics"]["by_source"].setdefault(s["source"], 0)
        manifest["statistics"]["by_source"][s["source"]] += 1

    out = out_path
    out.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {out} — {len(samples)} samples:")
    for src in sorted({s['source'] for s in samples}):
        rows = [s for s in samples if s["source"] == src]
        print(f"  {src:14s} {len(rows)} frames, "
              f"n_gt {[s['n_gt'] for s in rows]}, "
              f"masks {rows[0]['gt_masks']['kind']}")


if __name__ == "__main__":
    main()
