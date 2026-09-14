#!/usr/bin/env python3
"""Build ``vlm_eval/manifest.json`` and every image the eval feeds the model.

Ground truth is recomputed here from the upstream ``target_ref`` module rather
than being baked into this file, so a change in the geometric derivation flows
through automatically.  The selection is deterministic: for each source we take
the GT-valid episodes in sorted order and pick evenly spaced indices, recording
the indices, the seed and the exact episode list in the manifest.

The query image of an episode is always its FIRST colour frame
(``color_000000.png``).  The pre-rendered ``target_ref`` marker image is never an
input - it would hand the model the answer - and is used only for sanity checks.

Usage:
    python3 vlm_eval/build_manifest.py
    python3 vlm_eval/build_manifest.py --validation-k 8 --occluded-k 2
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import shutil
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from vlm_eval.lib import gtbridge, imaging  # noqa: E402
from vlm_eval import prompts  # noqa: E402

DEFAULT_VALIDATION_ROOT = Path(
    "/mnt/data1/sroi/sroi_v2/sroiv2_strawberry_picking_lab/"
    "validation_pngs/validation_20260714_160922-png"
)
DEFAULT_OCCLUDED_ROOT = Path(
    "/mnt/data1/sroi/sroi_v2/sroiv2_strawberry_picking_lab/20260803-occluded-cases-pngs"
)

# Unlabelled multi-strawberry scenes: no ground truth exists, so these are only
# ever reported qualitatively. They are also far larger than the wrist-camera
# frames, and codex downsizes images above a size threshold before sending them
# (measured: <=1920x1080 passes through byte-identical, >=2048x1536 is rescaled).
# We therefore normalise them ourselves to a verified-safe size and record the
# normalised dimensions as the frame the model actually sees.
DEFAULT_SHUNBA_ROOT = Path("/mnt/data1/strawberry_robot/shunba_sb_data/images")
DEFAULT_SHUNBA_MAX_EDGE = 1280

FRAME_NAME = "color_000000.png"   # first frame of every episode, by policy


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION_ROOT)
    ap.add_argument("--occluded-root", type=Path, default=DEFAULT_OCCLUDED_ROOT)
    ap.add_argument("--validation-k", type=int, default=4)
    ap.add_argument("--occluded-k", type=int, default=1)
    ap.add_argument("--shunba-root", type=Path, default=DEFAULT_SHUNBA_ROOT,
                    help="unlabelled multi-strawberry scenes (no ground truth)")
    ap.add_argument("--shunba-k", type=int, default=10)
    ap.add_argument("--shunba-max-edge", type=int, default=DEFAULT_SHUNBA_MAX_EDGE,
                    help="max edge to normalise unlabelled images to; must stay at or "
                         "below the CLI's no-resize ceiling (verified safe: 1280)")
    ap.add_argument("--skip-sroi", action="store_true",
                    help="omit the ground-truthed SROI sources (validation, occluded) "
                         "from the manifest entirely, so those frames are not even "
                         "copied or ground-truthed. They are still built by default; "
                         "the run step ignores them unless --sources asks for them.")
    ap.add_argument("--seed", type=int, default=20260714,
                    help="recorded for provenance; selection itself is deterministic spacing")
    ap.add_argument("--target-ref-root", type=Path, default=gtbridge.DEFAULT_TARGET_REF_ROOT)
    ap.add_argument("--sensitivity", type=Path, default=gtbridge.DEFAULT_SENSITIVITY_JSON)
    ap.add_argument("--out", type=Path, default=HERE / "manifest.json")
    ap.add_argument("--data-dir", type=Path, default=HERE / "data")
    args = ap.parse_args()
    args.out = args.out.resolve()
    args.data_dir = args.data_dir.resolve()
    return args


def scan_source(root: Path, target_ref, tip_kin) -> list[dict]:
    """Every episode under a source, with GT and a validity flag."""
    rows = []
    for entry in gtbridge.discover_episodes(root):
        gt = gtbridge.compute_gt(entry["path"], target_ref, tip_kin)
        rows.append({**entry, "gt": gt})
    return rows


def select(rows: list[dict], k: int, tag: str) -> tuple[list[dict], dict]:
    """Evenly spaced pick over the GT-valid rows; returns (selected, provenance)."""
    valid = sorted(
        (r for r in rows if r["gt"]["status"] == "ok"),
        key=lambda r: (r["session"], r["episode"]),
    )
    indices = gtbridge.even_spacing_indices(len(valid), k)
    selected = []
    for i, idx in enumerate(indices, start=1):
        row = valid[idx]
        selected.append({**row, "sample_id": f"{tag}{i:02d}", "gt_index": idx})
    provenance = {
        "n_episodes": len(rows),
        "n_gt_valid": len(valid),
        "k": k,
        "indices": indices,
        "episodes": [f"{r['session']}/{r['episode']}" for r in selected],
        "excluded_gt_status": {
            status: sum(1 for r in rows if r["gt"]["status"] == status)
            for status in sorted({r["gt"]["status"] for r in rows})
        },
    }
    return selected, provenance


def write_frame(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def as_manifest_path(path: Path) -> str:
    """Store a path relative to the harness when it lives inside it, else absolute.

    Downstream code joins these onto HERE, and ``Path("/abs") / "/abs/x"`` is just
    ``/abs/x``, so an absolute path keeps working for an external --data-dir.
    """
    try:
        return str(Path(path).relative_to(HERE))
    except ValueError:
        return str(Path(path))


def build_unlabelled_samples(root: Path, k: int, max_edge: int, data_dir: Path) -> tuple[list[dict], dict]:
    """Evenly spaced, size-normalised samples from a directory with no ground truth."""
    files = sorted(p for p in root.iterdir()
                   if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        raise SystemExit(f"no images found under {root}")
    indices = gtbridge.even_spacing_indices(len(files), min(k, len(files)))

    samples = []
    for ordinal, index in enumerate(indices, start=1):
        source = files[index]
        with Image.open(source) as img:
            rgb = img.convert("RGB")
            original_hw = [rgb.height, rgb.width]
            longest = max(rgb.size)
            if longest > max_edge:
                scale = max_edge / longest
                rgb = rgb.resize((max(1, round(rgb.width * scale)),
                                  max(1, round(rgb.height * scale))), Image.LANCZOS)
            normalised_hw = [rgb.height, rgb.width]

        sample_id = f"sb{ordinal:02d}"
        dst = data_dir / "frames" / f"{sample_id}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(dst)

        samples.append({
            "sample_id": sample_id,
            "source": "shunba",
            "has_gt": False,
            "task": "enumerate",
            "session": root.name,
            "episode": source.stem,
            "source_image": str(source),
            "source_index_in_list": index,
            "image": as_manifest_path(dst),
            "image_shape_hw": normalised_hw,
            "source_image_shape_hw": original_hw,
            "downscaled": original_hw != normalised_hw,
            "gt_status": "none",
            "gt_uv": None,
            "gt_uv_rounded": None,
            "gt_z_m": None,
            "gt_rough_box": None,
            "label_provenance": (
                "none: this directory has no demonstrations, no trajectory and no "
                "ground-truth point. Reported qualitatively only; excluded from every "
                "scored metric."
            ),
        })

    provenance = {
        "root": str(root),
        "n_images": len(files),
        "n_gt_valid": 0,
        "k": len(samples),
        "indices": indices,
        "episodes": [f"{root.name}/{s['episode']}" for s in samples],
        "has_ground_truth": False,
        "normalisation": {
            "max_edge": max_edge,
            "reason": (
                "codex downscales images above a size threshold before sending them; "
                "normalising here keeps the model's coordinate frame identical to the "
                "frame the harness draws on"
            ),
            "downscaled_count": sum(1 for s in samples if s["downscaled"]),
        },
    }
    return samples, provenance


def main() -> None:
    args = parse_args()
    target_ref = gtbridge.load_target_ref(args.target_ref_root)
    tip_kin = target_ref.load_tip_kin(target_ref.DEFAULT_EXTRINSICS_CONFIG)

    # The SROI wrist-camera sources hold the only ground truth, so they are what
    # turns the scored metrics on. They are still built by default (the run step
    # simply does not select them unless asked); --skip-sroi drops them entirely
    # so those frames are not even copied or ground-truthed.
    sources = [] if args.skip_sroi else [
        ("validation", args.validation_root, args.validation_k, "val"),
        ("occluded", args.occluded_root, args.occluded_k, "occ"),
    ]

    samples: list[dict] = []
    selection: dict[str, dict] = {}
    all_gt_valid_by_source: dict[str, list[dict]] = {}

    for name, root, k, tag in sources:
        rows = scan_source(root, target_ref, tip_kin)
        selected, provenance = select(rows, k, tag)
        provenance["root"] = str(root)
        selection[name] = provenance
        all_gt_valid_by_source[name] = sorted(
            (r for r in rows if r["gt"]["status"] == "ok"),
            key=lambda r: (r["session"], r["episode"]),
        )

        for row in selected:
            ep_dir = Path(row["path"])
            frame_src = ep_dir / FRAME_NAME
            if not frame_src.exists():
                raise SystemExit(f"missing first frame: {frame_src}")
            raw_dst = args.data_dir / "frames" / f"{row['sample_id']}.png"
            write_frame(frame_src, raw_dst)

            h, w = row["gt"]["image_shape"]
            samples.append({
                "sample_id": row["sample_id"],
                "source": name,
                "has_gt": True,
                "task": "single_target",
                "session": row["session"],
                "episode": row["episode"],
                "episode_dir": str(ep_dir),
                "source_image": str(frame_src),
                "gt_index_in_valid_list": row["gt_index"],
                "image": as_manifest_path(raw_dst),
                "frame_name": FRAME_NAME,
                "image_shape_hw": [int(h), int(w)],
                "source_image_shape_hw": [int(h), int(w)],
                "downscaled": False,
                "gt_status": row["gt"]["status"],
                "gt_uv": [row["gt"]["u"], row["gt"]["v"]],
                "gt_uv_rounded": [int(round(row["gt"]["u"])), int(round(row["gt"]["v"]))],
                "gt_z_m": row["gt"]["z"],
                "gt_traj_index": row["gt"]["t_g"],
                "gt_rough_box": list(gtbridge.rough_box((row["gt"]["u"], row["gt"]["v"]))),
                "label_provenance": (
                    "target_ref.compute_episode_target: gripper-tip pose at the episode's "
                    "last ORB-SLAM trajectory frame, projected through the initial-frame "
                    "camera pose and intrinsics into the initial image"
                ),
            })

    # --- unlabelled multi-strawberry scenes (qualitative only) -------------
    shunba_samples, shunba_provenance = build_unlabelled_samples(
        args.shunba_root, args.shunba_k, args.shunba_max_edge, args.data_dir)
    selection["shunba"] = shunba_provenance
    for sample in shunba_samples:
        sample["images"] = {"raw": sample["image"]}
    samples.extend(shunba_samples)

    # --- vision control ----------------------------------------------------
    control_img, control_truth = imaging.build_control_image()
    control_path = args.data_dir / "control" / "control_6305.png"
    imaging.save_rgb(control_img, control_path)

    # --- few-shot exemplar episode (must sit outside the eval set) ----------
    # Only the SROI sources can supply it: the exemplar is a ground-truth
    # annotation. Without them the exemplar prompts simply have no frames to run
    # on, which is expected rather than an error.
    exemplar_row = None
    exemplar_annotated = exemplar_uv = None
    if not args.skip_sroi:
        chosen_dirs = {s["episode_dir"] for s in samples if s.get("episode_dir")}
        exemplar_row = next(
            (r for r in all_gt_valid_by_source["validation"]
             if str(r["path"]) not in chosen_dirs),
            None,
        )
        if exemplar_row is None:
            raise SystemExit("no validation episode left for the few-shot exemplar")
        exemplar_gt = exemplar_row["gt"]
        exemplar_uv = (exemplar_gt["u"], exemplar_gt["v"])
        exemplar_src = Path(exemplar_row["path"]) / FRAME_NAME
        exemplar_rgb = imaging.load_rgb(exemplar_src)
        exemplar_annotated = imaging.draw_gt_marker(exemplar_rgb, exemplar_uv)
        imaging.save_rgb(exemplar_annotated,
                         args.data_dir / "exemplar" / "exemplar_annotated.png")

    # --- derived inputs: grid overlay + exemplar composite, per sample -----
    # Only the scored single-target styles need these, so unlabelled scenes skip
    # them (they run only the raw-frame enumeration style).
    for sample in samples:
        if not sample["has_gt"]:
            continue
        raw = imaging.load_rgb(HERE / sample["image"])
        grid = imaging.draw_grid_overlay(raw, step=64)
        grid_rel = Path("data") / "derived" / f"grid__{sample['sample_id']}.png"
        imaging.save_rgb(grid, HERE / grid_rel)

        composite = imaging.build_exemplar_composite(exemplar_annotated, exemplar_uv, raw)
        comp_rel = Path("data") / "derived" / f"exemplar__{sample['sample_id']}.png"
        imaging.save_rgb(composite, HERE / comp_rel)

        sample["images"] = {
            "raw": sample["image"],
            "grid_overlay": str(grid_rel),
            "few_shot_exemplar": str(comp_rel),
        }
        sample["exemplar_query_origin"] = list(imaging.exemplar_query_offset())
        sample["derived_image_shape_hw"] = {
            "grid_overlay": list(grid.shape[:2]),
            "few_shot_exemplar": list(composite.shape[:2]),
            "raw": list(raw.shape[:2]),
        }

    sensitivity = gtbridge.load_sensitivity(args.sensitivity)

    manifest = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "generator": "vlm_eval/build_manifest.py",
        "python": platform.python_version(),
        "repo": str(REPO),
        "harness_dir": str(HERE),
        "prompt_styles": list(prompts.DEFAULT_STYLE_ORDER),
        "frame_policy": (
            f"first colour frame of each episode ({FRAME_NAME}); the pre-rendered "
            "target_ref marker image is never used as model input"
        ),
        "coordinate_convention": (
            "pixels, origin at the frame's top-left corner, x right, y down; the "
            "few_shot_exemplar style is the only exception and states its own global "
            "composite convention explicitly in the prompt"
        ),
        "ground_truth": {
            "module": str(args.target_ref_root / "target_ref.py"),
            "extrinsics_config": str(target_ref.DEFAULT_EXTRINSICS_CONFIG),
            "definition": (
                "gripper tip at the last trajectory frame of the episode projected into "
                "the initial frame (the demonstrator ends above the picked strawberry)"
            ),
            "rough_box_convention": {
                "half_px": gtbridge.ROUGH_BOX_HALF,
                "offset_px": gtbridge.ROUGH_BOX_OFFSET,
                "note": "no hand-annotated fruit boxes exist; bbox scoring is approximate",
            },
        },
        "selection": {
            "method": "deterministic even spacing over the sorted GT-valid episode list",
            "seed": args.seed,
            "note": "seed is recorded for provenance; spacing is deterministic and needs no RNG",
            "per_source": selection,
        },
        "control": {
            "image": str((Path("data") / "control" / "control_6305.png")),
            "truth": control_truth,
            "prompt": prompts.CONTROL_PROMPT,
            "purpose": "proves the CLI actually delivered an image to the model before any "
                       "capability number is reported",
        },
        "exemplar": ({
            "session": exemplar_row["session"],
            "episode": exemplar_row["episode"],
            "episode_dir": str(exemplar_row["path"]),
            "image": str(Path("data") / "exemplar" / "exemplar_annotated.png"),
            "gt_uv": [exemplar_uv[0], exemplar_uv[1]],
            "gt_uv_rounded": [int(round(exemplar_uv[0])), int(round(exemplar_uv[1]))],
            "in_eval_set": False,
        } if exemplar_row else None),
        "sensitivity_bridge": sensitivity,
        "samples": samples,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {args.out}")
    print(f"samples: {len(samples)} "
          f"({', '.join(s['sample_id'] + '=' + s['episode'] for s in samples)})")
    for name, prov in selection.items():
        if prov.get("has_ground_truth", True):
            print(f"  {name:10s} {prov['n_episodes']} episodes, "
                  f"{prov['n_gt_valid']} GT-valid, picked indices {prov['indices']}")
        else:
            print(f"  {name:10s} {prov['n_images']} images, NO ground truth "
                  f"(qualitative only), picked indices {prov['indices']}, "
                  f"{prov['normalisation']['downscaled_count']} normalised to "
                  f"max edge {prov['normalisation']['max_edge']}")
    if exemplar_row:
        print(f"  exemplar  {exemplar_row['session']}/{exemplar_row['episode']} "
              f"@ {manifest['exemplar']['gt_uv_rounded']}")
    else:
        print("  exemplar  none (SROI sources skipped; the exemplar prompts have no "
              "frames to run on)")
    print(f"control image -> {HERE / manifest['control']['image']}")


if __name__ == "__main__":
    main()
