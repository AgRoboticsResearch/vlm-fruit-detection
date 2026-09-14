#!/usr/bin/env python3
"""Ground truth, sample selection and the pixel -> gripper sensitivity bridge.

Ground truth is *not* hard-coded anywhere.  It is recomputed at manifest-build
time by importing the upstream ``target_ref`` module, which derives the picking
location geometrically: the gripper-tip pose at the LAST trajectory frame of an
episode, projected into the INITIAL image.  No detector, tracker or
segmentation model participates.

The sensitivity bridge turns a pixel error into picking-relevant units (mm of
gripper-endpoint displacement, and gripper-closure change) using a measured
dose-response sweep that was produced elsewhere by replaying the policy with
perturbed target points.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

# Upstream module lives outside this repo; keep the path configurable.
DEFAULT_TARGET_REF_ROOT = Path("/mnt/data0/code/sroi/sroi_rosbag_utilities")
DEFAULT_SENSITIVITY_JSON = Path(
    "/mnt/data1/projects/target_condition_sb_picking/viz/target_move/target_move_ep0_frame20.json"
)

# The upstream "rough fruit box" convention: a 60x60 box centred 30 px below the
# picking point, i.e. its top edge touches the picking point.  There are no
# hand-annotated fruit boxes in this data, so bbox scoring is approximate.
ROUGH_BOX_HALF = 30
ROUGH_BOX_OFFSET = 30


def load_target_ref(root: Path | str = DEFAULT_TARGET_REF_ROOT):
    """Import ``target_ref.py`` from its source tree and return the module."""
    root = Path(root)
    module_path = root / "target_ref.py"
    if not module_path.exists():
        raise SystemExit(f"target_ref.py not found under {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("vlm_eval_target_ref", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_gt(episode_dir: Path | str, target_ref_module, tip_kin) -> dict:
    """Ground truth for one episode: status, picking point (u, v) and depth z."""
    result = dict(target_ref_module.compute_episode_target(Path(episode_dir), tip_kin))
    result["episode_dir"] = str(episode_dir)
    for key in ("u", "v", "z"):
        value = result.get(key)
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            result[key] = None
    return result


def rough_box(uv, half: int = ROUGH_BOX_HALF, offset: int = ROUGH_BOX_OFFSET):
    """Upstream-convention rough fruit box around a picking point."""
    u, v = float(uv[0]), float(uv[1])
    cx, cy = u, v + offset
    return (cx - half, cy - half, cx + half, cy + half)


# ---------------------------------------------------------------------------
# Sample discovery / selection
# ---------------------------------------------------------------------------

def discover_episodes(source_root: Path | str) -> list[dict]:
    """List episode directories under a source, handling both layouts.

    Flat layout:  <root>/episode_XXX/
    Nested layout: <root>/<session>*/episode_XXX/
    """
    root = Path(source_root)
    if not root.is_dir():
        raise SystemExit(f"source root not found: {root}")

    episodes: list[dict] = []
    direct = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("episode_"))
    if direct:
        for path in direct:
            episodes.append({"session": root.name, "episode": path.name, "path": path})
        return episodes

    for session in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(p for p in session.iterdir() if p.is_dir() and p.name.startswith("episode_")):
            episodes.append({"session": session.name, "episode": path.name, "path": path})
    if not episodes:
        raise SystemExit(f"no episode directories found under {root}")
    return episodes


def even_spacing_indices(n: int, k: int) -> list[int]:
    """k deterministic, evenly spaced indices over ``range(n)``.

    k == 1 picks the median element (a representative middle sample rather than
    an arbitrary first one); k >= 2 spans both endpoints inclusive.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    if k > n:
        raise ValueError(f"cannot pick {k} distinct samples from {n}")
    if k == 1:
        return [n // 2]
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


# ---------------------------------------------------------------------------
# Sensitivity bridge: pixel offset -> gripper endpoint / closure change
# ---------------------------------------------------------------------------

def load_sensitivity(path: Path | str = DEFAULT_SENSITIVITY_JSON) -> dict:
    """Fit a local linear pixel->gripper model from a dose-response sweep.

    The sweep reports, for a handful of target points near one reference point,
    the mean gripper endpoint (mm) and mean final gripper value that replaying
    the policy produced.  We take central differences along each axis to get a
    per-pixel rate, which lets a pixel error be quoted in picking-relevant
    units.  The reference point of the sweep happens to be validation episode 1.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    origin = np.asarray(data["original_uv"], dtype=float)

    points: dict[tuple[int, int], dict] = {}
    for key, value in data["points"].items():
        if key == "none":
            continue
        x, y = (int(part) for part in key.split(","))
        points[(x, y)] = value
    if not points:
        raise SystemExit(f"{path}: no perturbed points to fit")

    base_key = min(points, key=lambda k: (k[0] - origin[0]) ** 2 + (k[1] - origin[1]) ** 2)
    base = points[base_key]
    base_e = np.asarray(base["endpoint_mean_mm"], dtype=float)
    base_g = float(base["grip_end_mean"])

    def axis_rate(varying_axis: str) -> dict:
        """Central difference along one axis with the *other* axis held fixed."""
        moving = 0 if varying_axis == "x" else 1   # index of the coordinate that varies
        fixed = 1 - moving                          # index of the coordinate that stays put
        subset = {k: v for k, v in points.items() if k[fixed] == base_key[fixed]}
        lower = [k for k in subset if k[moving] < base_key[moving]]
        upper = [k for k in subset if k[moving] > base_key[moving]]
        if lower and upper:
            lo = min(lower, key=lambda k: base_key[moving] - k[moving])
            hi = min(upper, key=lambda k: k[moving] - base_key[moving])
            span = hi[moving] - lo[moving]
            de = (np.asarray(subset[hi]["endpoint_mean_mm"], dtype=float)
                  - np.asarray(subset[lo]["endpoint_mean_mm"], dtype=float)) / span
            dg = (float(subset[hi]["grip_end_mean"]) - float(subset[lo]["grip_end_mean"])) / span
        else:
            side = (lower or upper)
            if not side:
                raise SystemExit(f"{path}: axis {varying_axis} has no neighbours to difference")
            k = min(side, key=lambda k: abs(k[moving] - base_key[moving]))
            span = k[moving] - base_key[moving]
            de = (np.asarray(subset[k]["endpoint_mean_mm"], dtype=float) - base_e) / span
            dg = (float(subset[k]["grip_end_mean"]) - base_g) / span
        return {"endpoint_mm_per_px": de.tolist(), "grip_per_px": float(dg),
                "span_px": int(abs(span))}

    rate_x = axis_rate("x")
    rate_y = axis_rate("y")
    return {
        "source": str(path),
        "reference_original_uv": origin.tolist(),
        "reference_key": f"{base_key[0]},{base_key[1]}",
        "reference_endpoint_mean_mm": base_e.tolist(),
        "reference_grip_end_mean": base_g,
        "d_endpoint_dx_mm_per_px": rate_x["endpoint_mm_per_px"],
        "d_endpoint_dy_mm_per_px": rate_y["endpoint_mm_per_px"],
        "d_grip_dx_per_px": rate_x["grip_per_px"],
        "d_grip_dy_per_px": rate_y["grip_per_px"],
        "note": (
            "Local central-difference rates from a single measured dose-response sweep "
            "around one reference point; magnitudes are indicative, not a calibrated "
            "per-episode Jacobian."
        ),
    }


def sensitivity_effect(model: dict, dx: float, dy: float) -> dict:
    """Apply the fitted rates to a pixel offset (predicted - ground truth)."""
    ex = np.asarray(model["d_endpoint_dx_mm_per_px"], dtype=float)
    ey = np.asarray(model["d_endpoint_dy_mm_per_px"], dtype=float)
    shift = ex * float(dx) + ey * float(dy)
    return {
        "endpoint_shift_mm": shift.tolist(),
        "endpoint_shift_magnitude_mm": float(np.linalg.norm(shift)),
        "grip_delta": float(model["d_grip_dx_per_px"] * dx + model["d_grip_dy_per_px"] * dy),
    }
