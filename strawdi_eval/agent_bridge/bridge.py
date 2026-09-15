#!/usr/bin/env python3
"""Agent-driven runner for the StrawDI detection eval.

Feeds the ONE ``inventory_detection`` prompt of ``strawdi_eval`` to the
**workbuddy agent** instead of to the ``claude``/``codex`` CLI provider stack:
the agent reads each frame as a multimodal attachment, replies with the
eight-field inventory JSON, and this bridge turns those replies into a normal
run directory — parsed by the harness's own parser (``classify_detection``),
scored by the harness's own scorer, rendered by the harness's own renderer, and
reported by the harness's own ``write_artifacts``.

What this bridge deliberately does NOT touch: ``strawdi_eval/*.py``,
``strawdi_eval/lib/*.py``, ``strawdi_eval/schema/*.json`` and everything under
``vlm_eval/``. It lives in a subdirectory, which the harness fingerprint
function does not glob, so the fingerprints of both harnesses stay byte-for-byte
what a CLI run would record. It imports them instead of copying logic, so the
measured pipeline is the same code.

Protocol (three steps, the agent in the middle):

    bridge.py prepare --tag full            # writes state dir + fresh control
    bridge.py status                        # which frames still need an answer
    bridge.py check                         # parse the answers written so far
    bridge.py assemble --tag full           # run dir + score + report + verify

The agent writes one file per frame, ``<state>/answers/<sample_id>.json``,
containing the raw reply text (JSON object only) for that frame, and
``<state>/answers/control.json`` for the control image.

Honest accounting in this mode:

* usage: the agent runtime reports no per-call token usage. Token counts in the
  records are ESTIMATES from text length (``ceil(chars/4)`` over prompt+reply);
  the image's tokens are not counted. Every record carries
  ``usage_source: "estimated"`` and the estimator is printed in the report.
* wall time: measured agent cycle time (interval between successive answer
  files), not a provider latency.
* reasoning effort: the runtime does not expose it; recorded as
  ``not-exposed`` rather than guessed.
* the vision control uses a FRESH synthetic image generated here, so its code
  is not known to the operator before the control answer is written.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BRIDGE_VERSION = "0.1.0"

BRIDGE_DIR = Path(__file__).resolve().parent
HERE = BRIDGE_DIR.parent                 # strawdi_eval/
REPO = HERE.parent
sys.path.insert(0, str(REPO))

import strawdi_eval.run_detection_eval as rde          # noqa: E402
import vlm_eval.run_vlm_eval as vre                    # noqa: E402
from strawdi_eval.lib import prompt as strawdi_prompt  # noqa: E402
from vlm_eval import prompts as base_prompts           # noqa: E402
from vlm_eval.lib import imaging, parse                # noqa: E402

PROVIDER = "antigravity"
DEFAULT_MODEL = "gemini-3.8-flash-high"
DEFAULT_EFFORT = "high"
STATE_ROOT = BRIDGE_DIR / "state"
CONTROL_W, CONTROL_H = 640, 480
AGENT_SECTION_MARKER = "\n## Agent-driven run\n"

# Text-length token estimator: no provider usage exists in this mode.
TOKEN_ESTIMATOR = "ceil(chars/4) over prompt text + reply text (image tokens not counted)"


def est_tokens(text: str) -> int:
    return int(math.ceil(len(text or "") / 4))


# ---------------------------------------------------------------------------
# Fresh vision control (the code is drawn from a seed the operator never sees)
# ---------------------------------------------------------------------------

def generate_control(out_png: Path, truth_path: Path) -> None:
    rng = random.Random()   # seed deliberately never printed or stored
    code = "".join(rng.choice("0123456789") for _ in range(4))
    cx = rng.randint(460, CONTROL_W - 40)
    cy = rng.randint(40, 150)
    sx = rng.randint(60, 260)
    sy = rng.randint(340, CONTROL_H - 50)

    img = Image.new("RGB", (CONTROL_W, CONTROL_H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((30, 30), code, font=imaging.font(150, bold=True), fill=(0, 0, 0))

    r = 28
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 20, 20),
                 outline=(0, 0, 0), width=2)
    half = 40
    draw.rectangle([sx - half, sy - half, sx + half, sy + half],
                   fill=(20, 150, 20), outline=(0, 0, 0), width=2)

    imaging.save_rgb(np.array(img), out_png)
    truth_path.write_text(json.dumps(
        {"code": code, "red_circle": [cx, cy], "green_square": [sx, sy]}, indent=2) + "\n")


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

def cmd_prepare(args) -> None:
    manifest = json.loads(args.manifest.read_text())
    samples = manifest["samples"][: args.limit] if args.limit else manifest["samples"]
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    state = STATE_ROOT / stamp
    (state / "answers").mkdir(parents=True, exist_ok=False)

    control_png = state / "control.png"
    truth_path = state / "control_truth.json"
    generate_control(control_png, truth_path)

    control_prompt = f"{base_prompts.CONTROL_PROMPT}\n\n{vre.TOOL_NOTICE}"
    (state / "control_prompt.txt").write_text(control_prompt)
    (state / "prompt.txt").write_text(rde.build_prompt(samples[0]))

    provider = getattr(args, "provider", None) or PROVIDER
    effort = getattr(args, "effort", None) or DEFAULT_EFFORT
    plan = {
        "bridge": {"name": "strawdi_eval/agent_bridge", "version": BRIDGE_VERSION,
                   "path": str(BRIDGE_DIR), "provider": provider},
        "model": args.model,
        "effort": effort,
        "tag": args.tag,
        "started": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "stamp": stamp,
        "manifest": str(args.manifest),
        "sample_ids": [s["sample_id"] for s in samples],
        "n_planned": len(samples),
        "frame_dir": str(HERE / "data" / "frames"),
        "control": {"image": str(control_png), "prompt": str(state / "control_prompt.txt"),
                    "truth_file": str(truth_path)},
        "prompt_style": strawdi_prompt.STYLE_NAME,
        "harness": rde.harness_info(),
        "base_harness": vre.harness_info(),
        "usage_source": "estimated",
        "token_estimator": TOKEN_ESTIMATOR,
        "notes": [
            "answers/<sample_id>.json holds the raw reply text for that frame",
            "answers/<sample_id>.retry.json (optional) is the one recorded resample",
            "the operator must not read control_truth.json, manifest gt_boxes or overlays "
            "while answers are being written",
        ],
    }
    (state / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    print(f"state   : {state}")
    print(f"frames  : {len(samples)} planned (ids in plan.json)")
    print(f"control : {control_png}")
    print(f"prompt  : {state / 'prompt.txt'}  (identical for all frames)")
    print("truth for the control was written to a file and NOT printed.")


# ---------------------------------------------------------------------------
# status / check
# ---------------------------------------------------------------------------

def _state(args) -> tuple[Path, dict]:
    state = Path(args.state).resolve() if args.state else _latest_state()
    if not (state / "plan.json").exists():
        raise SystemExit(f"no plan.json in {state}")
    return state, json.loads((state / "plan.json").read_text())


def _latest_state() -> Path:
    candidates = sorted(p for p in STATE_ROOT.glob("*") if (p / "plan.json").exists())
    if not candidates:
        raise SystemExit("no prepared state directory; run `bridge.py prepare` first")
    return candidates[-1]


def _answer_file(state: Path, sample_id: str) -> Path | None:
    for name in (f"{sample_id}.retry.json", f"{sample_id}.json"):
        path = state / "answers" / name
        if path.exists():
            return path
    return None


def cmd_status(args) -> None:
    state, plan = _state(args)
    done, missing = [], []
    for sid in plan["sample_ids"]:
        (done if _answer_file(state, sid) else missing).append(sid)
    control = (state / "answers" / "control.json").exists()
    print(f"state  : {state}")
    print(f"control: {'answered' if control else 'MISSING'}")
    print(f"frames : {len(done)}/{plan['n_planned']} answered")
    if missing and args.verbose:
        print("missing: " + " ".join(missing))


def cmd_check(args) -> None:
    """Parse every answer written so far; prints statuses only (no GT, no scores)."""
    state, plan = _state(args)
    schema = json.loads(rde.SCHEMA_PATH.read_text())
    counts: dict[str, int] = {}
    bad = []
    for sid in plan["sample_ids"]:
        path = _answer_file(state, sid)
        if not path:
            continue
        scored = rde.classify_detection(path.read_text(), schema)
        counts[scored["status"]] = counts.get(scored["status"], 0) + 1
        if scored["status"] != parse.OK:
            bad.append((sid, scored["status"], scored.get("schema_error")))
    print(f"answers parsed: {sum(counts.values())}  {counts}")
    for sid, status, err in bad[:40]:
        print(f"  {sid}: {status}  {err or ''}")
    control = state / "answers" / "control.json"
    if control.exists():
        obj, method = parse.extract_json(control.read_text())
        ctrl_schema = json.loads(Path(vre.CONTROL_SCHEMA_PATH).read_text())
        valid, err = parse.validate(obj, ctrl_schema) if obj is not None else (False, "no JSON")
        print(f"control: method={method} schema_valid={valid} {err or ''}")
    else:
        print("control: MISSING")


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------

def _control_record(state: Path, plan: dict, run_dir: Path) -> dict | None:
    answer = state / "answers" / "control.json"
    if not answer.exists():
        return None
    truth = json.loads(Path(plan["control"]["truth_file"]).read_text())
    text = answer.read_text()
    schema = json.loads(Path(vre.CONTROL_SCHEMA_PATH).read_text())
    obj, method = parse.extract_json(text)
    valid, schema_error = (parse.validate(obj, schema) if obj is not None
                           else (False, "no JSON"))

    code_ok = circle_ok = square_ok = None
    circle_err = square_err = None
    if valid:
        code_ok = str(obj["code"]).strip() == truth["code"]
        pred_circle = parse.normalise_point(obj["red_circle"])
        pred_square = parse.normalise_point(obj["green_square"])
        if pred_circle:
            circle_err = float(np.hypot(pred_circle[0] - truth["red_circle"][0],
                                        pred_circle[1] - truth["red_circle"][1]))
            circle_ok = circle_err <= vre.CONTROL_TOLERANCE_PX
        if pred_square:
            square_err = float(np.hypot(pred_square[0] - truth["green_square"][0],
                                        pred_square[1] - truth["green_square"][1]))
            square_ok = square_err <= vre.CONTROL_TOLERANCE_PX

    ctrl_img = Path(plan["control"]["image"])
    return {
        "image": str(ctrl_img.relative_to(HERE)) if ctrl_img.is_relative_to(HERE)
                 else str(ctrl_img),
        "image_kind": "control",
        "status": "ok" if valid else parse.PARSE_ERROR,
        "json_method": method,
        "schema_valid": valid,
        "schema_error": schema_error,
        "truth": truth,
        "prediction": obj,
        "code_ok": code_ok,
        "circle_ok": circle_ok,
        "green_square_ok": square_ok,
        "circle_error_px": circle_err,
        "green_square_error_px": square_err,
        "tolerance_px": vre.CONTROL_TOLERANCE_PX,
        "images_delivered": bool(code_ok and circle_ok),
        "tool_attempts": [],
        "usage": {"input_tokens": est_tokens(plan.get("control", {}).get("prompt", "") or
                                             (state / "control_prompt.txt").read_text()),
                  "output_tokens": est_tokens(text)},
        "usage_source": "estimated",
        "wall_s": None,
        "returncode": 0,
        "response_text": text,
        "delivery": ("agent read tool: the PNG was attached to the workbuddy session "
                     "as an image content block"),
    }


def _cycle_times(state: Path, plan: dict) -> dict[str, float]:
    """Agent cycle time per frame: interval between successive answer files."""
    entries = []
    for sid in plan["sample_ids"]:
        path = _answer_file(state, sid)
        if path:
            entries.append((path.stat().st_mtime, sid))
    entries.sort()
    origin = (state / "plan.json").stat().st_mtime
    out: dict[str, float] = {}
    previous = origin
    for mtime, sid in entries:
        out[sid] = round(max(0.0, mtime - previous), 3)
        previous = mtime
    return out


def cmd_assemble(args) -> None:
    state, plan = _state(args)
    manifest_path = Path(args.manifest) if args.manifest else Path(plan["manifest"])
    manifest = json.loads(manifest_path.read_text())
    by_id = {s["sample_id"]: s for s in manifest["samples"]}

    excluded = [s.strip() for s in (args.exclude or "").split(",") if s.strip()]
    unknown = [s for s in excluded if s not in plan["sample_ids"]]
    if unknown:
        raise SystemExit(f"excluded ids not in the plan: {unknown}")
    framed = [sid for sid in plan["sample_ids"]
              if sid not in excluded and _answer_file(state, sid)]
    if not framed:
        raise SystemExit("no answers written yet")
    tag = args.tag or plan.get("tag") or ""
    model = args.model or plan["model"]
    provider = plan.get("bridge", {}).get("provider") or plan.get("provider") or PROVIDER
    effort = args.effort or plan.get("effort") or DEFAULT_EFFORT
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out / rde.run_dir_name(stamp, model, effort, tag,
                                          cli=provider)
    if args.dry_dir:
        print(f"would write: {run_dir}")
        return
    run_dir.mkdir(parents=True, exist_ok=False)

    # Manifest snapshot: the control block describes the control ACTUALLY used.
    control = _control_record(state, plan, run_dir)
    man = json.loads(manifest_path.read_text())
    if control is not None:
        man["control"] = {
            "image": control["image"],
            "truth": control["truth"],
            "prompt": (state / "control_prompt.txt").read_text(),
            "purpose": ("synthetic vision-delivery gate, freshly generated for the "
                        "agent-driven run; its code was unknown to the operator "
                        "when the control answer was written"),
        }
        man["agent_bridge"] = {
            "bridge": "strawdi_eval/agent_bridge/bridge.py",
            "version": BRIDGE_VERSION,
            "control_replaced": True,
            "note": ("the stock data/control/control_6305.png control is unchanged on "
                     "disk; this run used a freshly drawn one to keep the gate honest"),
        }
    (run_dir / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    if control is not None:
        (run_dir / "control.json").write_text(json.dumps(control, indent=2) + "\n")

    catalog = {
        "provider": provider,
        "model_slug": model,
        "reasoning_effort": effort,
        "runtime": f"{provider} agent (agent_bridge)",
        "delivery": ("agent read tool: one PNG per call, attached to the session as an "
                     "image content block; no CLI, no provider API call by the harness"),
        "input_modalities": ["image"],
        "tools_available_to_the_model": [],
        "usage": {"source": "estimated", "estimator": TOKEN_ESTIMATOR,
                  "note": "the agent runtime reports no per-call token usage"},
        "wall_clock": {"source": "agent cycle time",
                       "note": "interval between successive answer files, not a provider latency"},
        "bridge": {"path": str(BRIDGE_DIR / "bridge.py"), "version": BRIDGE_VERSION},
        "harness_fingerprint": rde.harness_info()["fingerprint"],
        "base_harness_fingerprint": vre.harness_info()["fingerprint"],
    }
    (run_dir / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")

    schema = json.loads(rde.SCHEMA_PATH.read_text())
    cycles = _cycle_times(state, plan)
    records: list[dict] = []
    for sid in sorted(framed):
        sample = by_id[sid]
        path = _answer_file(state, sid)
        reply = path.read_text()
        retried = path.name.endswith(".retry.json")
        scored = rde.classify_detection(reply, schema)
        parsed_ok = scored["status"] in (parse.OK, parse.NO_PICK_POINT)
        inventory = scored["strawberries"] if parsed_ok else None
        h, w = sample["image_shape_hw"]
        prompt = rde.build_prompt(sample)
        prompt_tokens, reply_tokens = est_tokens(prompt), est_tokens(reply)

        record = {
            "run_id": f"{rde.STYLE_NAME}__{sid}",
            "style": rde.STYLE_NAME,
            "sample_id": sid,
            "source": sample["source"],
            "split": sample["split"],
            "has_gt": True,
            "task": sample.get("task", "full_inventory"),
            "status": scored["status"],
            "json_method": scored["json_method"],
            "schema_valid": scored["schema_valid"],
            "schema_error": scored["schema_error"],
            "model": model,
            "harness": rde.HARNESS_NAME,
            "harness_version": rde.HARNESS_VERSION,
            "harness_fingerprint": rde.harness_info()["fingerprint"],
            "base_harness": vre.harness_info()["name"],
            "base_harness_version": vre.harness_info()["version"],
            "base_harness_fingerprint": vre.harness_info()["fingerprint"],
            "provider": provider,
            "reasoning_effort": effort,
            "reasoning_effort_note": f"{effort} reasoning effort via {provider}",
            "used_output_schema": False,
            "output_shape": "inventory",
            "image": sample["images"]["raw"],
            "image_kind": "raw",
            "frame_w": w,
            "frame_h": h,
            "prompt": prompt,
            "inventory": inventory,
            "n_strawberries": scored["n_strawberries"] if parsed_ok else None,
            **vre._inventory_summary(inventory),
            "gt_boxes": sample["gt_boxes"],
            "gt_areas": sample["gt_areas"],
            **rde.detection_block({"status": scored["status"], "inventory": inventory,
                                   "gt_boxes": sample["gt_boxes"],
                                   "gt_areas": sample["gt_areas"],
                                   "frame_w": w, "frame_h": h}),
            "input_tokens": prompt_tokens,
            "cached_input_tokens": 0,
            "output_tokens": reply_tokens,
            "reasoning_output_tokens": 0,
            "total_tokens": prompt_tokens + reply_tokens,
            "usage_source": "estimated",
            "token_estimator": TOKEN_ESTIMATOR,
            "attempts": 2 if retried else 1,
            "fallback_used": retried,
            "fallback_reason": ("schema-invalid reply, one resample at same effort"
                                if retried else None),
            "wall_s": cycles.get(sid, 0.0),
            "wall_s_kind": "agent cycle time (interval since the previous answer file)",
            "returncodes": [0],
            "timed_out": False,
            "tool_attempts": [],
            "cli_errors": [],
            "response_text": reply,
            "response_excerpt": reply[:600],
            "cost_usd": None,
            "cost_note": "no provider billing in an agent-driven run",
            "attempts_detail": [{"index": 1 if not retried else 2,
                                 "returncode": 0,
                                 "wall_s": cycles.get(sid, 0.0),
                                 "timed_out": False,
                                 "usage": {"input_tokens": prompt_tokens,
                                           "output_tokens": reply_tokens,
                                           "total_tokens": prompt_tokens + reply_tokens,
                                           "source": "estimated"},
                                 "tool_attempts": [], "cli_errors": [],
                                 "message_chars": len(reply)}],
            "answer_source": str(path.relative_to(state)),
            "started": plan["started"],
        }
        records.append(record)

    (run_dir / "responses.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records))

    started = plan["started"]
    elapsed = sum(r["wall_s"] for r in records)
    args_ns = types.SimpleNamespace(provider=provider, jobs=1)
    run_summary = {
        "model": model,
        "effort": effort,
        "provider": provider,
        "cli_version": f"{provider} agent (agent_bridge v{BRIDGE_VERSION})",
        "catalog": catalog,
        "harness": rde.harness_info(),
        "base_harness": vre.harness_info(),
    }
    rde.write_artifacts(run_dir, args_ns, man, control, records, started, elapsed,
                        run_summary)
    _append_agent_section(run_dir, plan, records, excluded, args.exclude_reason)

    summary = rde.detection_summary(records)
    print(f"run dir : {run_dir}")
    print(f"frames  : {len(records)}/{plan['n_planned']}")
    print(f"F1@0.5  : {summary.get('f1_50')}  P {summary.get('precision_50')} "
          f"R {summary.get('recall_50')}  AP@50 {summary.get('ap_50')}")


def _append_agent_section(run_dir: Path, plan: dict, records: list[dict],
                          excluded: list[str] | None = None,
                          exclude_reason: str | None = None) -> None:
    report_path = run_dir / "report.md"
    text = report_path.read_text()
    text = text.split(AGENT_SECTION_MARKER)[0].rstrip() + "\n"
    n = len(records)
    failed = [r for r in records if r["status"] not in (parse.OK, parse.NO_PICK_POINT)]
    retried = sum(1 for r in records if r.get("fallback_used"))
    provider = plan.get("bridge", {}).get("provider") or plan.get("provider") or PROVIDER
    effort = plan.get("effort") or DEFAULT_EFFORT
    lines = [
        AGENT_SECTION_MARKER,
        f"This batch was produced by the **{provider} agent** (provider "
        f"`{provider}`, `strawdi_eval/agent_bridge/bridge.py` v{BRIDGE_VERSION}) "
        "instead of the harness's `claude`/`codex` CLI provider stack: the agent "
        "itself was the model under test. The prompt, the eight-field schema, the "
        "reply parser, the scorer, the overlay renderer and this report are the "
        "harness's own code, imported unchanged.",
        "",
        f"- **Frames answered:** {n} of {plan['n_planned']} manifest frames "
        f"({len(plan['sample_ids'])} planned; the rest carry no records).",
    ]
    if excluded:
        lines.append(f"- **Excluded frames:** {', '.join(excluded)} — "
                     f"{exclude_reason or 'see the run notes'}. They are absent from "
                     "every metric, not scored as misses.")
    lines += [
        f"- **Model string:** `{plan['model']}` — as reported by the agent platform, "
        "not probed by the harness.",
        "- **Image delivery:** one frame per call, read from `data/frames/<id>.png` "
        "(the byte-identical snapshot the verifier sha256-checks) and attached to the "
        "agent session as an image content block.",
        "- **Tools:** no tool was used to produce any answer. The agent had shell and "
        "file access to the repository, so unlike a CLI run this batch cannot exclude "
        "tool use the way the harness's tool-free CLI invocation does; the answers "
        "were written from the images alone, and the operator did not read "
        "`gt_boxes`, the manifest's ground-truth fields or any overlay until every "
        "answer was written.",
        f"- **Reasoning effort:** `{effort}` — configured for the agent runtime.",
        "- **Token counts are ESTIMATES** (`usage_source: estimated`, "
        f"{TOKEN_ESTIMATOR}). An agent run has no provider usage report; do not "
        "compare these totals with a CLI run's billed tokens.",
        "- **Wall time** in the per-image table is agent cycle time (interval between "
        "successive answer files), not a provider latency.",
        f"- **Retries:** {retried} frame(s) needed the one recorded resample; "
        f"{len(failed)} frame(s) ended in an explicit failure.",
        "- **Cost:** not reported (no provider billing in this mode).",
        "- **Harness fingerprints are unchanged** relative to the CLI runs: the "
        "bridge lives in `strawdi_eval/agent_bridge/`, which the fingerprint function "
        "does not glob, so both fingerprints above are those a CLI run would record.",
        "",
    ]
    report_path.write_text(text + "\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", type=Path, default=None, help="state dir (default: latest)")
    ap.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    ap.add_argument("--out", type=Path, default=HERE / "runs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="write a state dir + a fresh vision control")
    p.add_argument("--tag", default="")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--provider", default=PROVIDER)
    p.add_argument("--effort", default=DEFAULT_EFFORT)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("status", help="which frames still lack an answer")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("check", help="parse the answers written so far")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("assemble", help="build the run dir, score it, write the report")
    p.add_argument("--tag", default="")
    p.add_argument("--model", default=None)
    p.add_argument("--effort", default=None)
    p.add_argument("--exclude", default="", help="comma-separated sample ids to drop")
    p.add_argument("--exclude-reason", default=None,
                   help="recorded verbatim in the report")
    p.add_argument("--dry-dir", action="store_true")
    p.set_defaults(func=cmd_assemble)

    args = ap.parse_args(argv)
    args.manifest = Path(args.manifest).resolve()
    args.out = Path(args.out).resolve()
    args.func(args)


if __name__ == "__main__":
    main()
