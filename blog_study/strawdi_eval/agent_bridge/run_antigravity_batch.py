#!/usr/bin/env python3
"""Automated batch runner for strawdi_eval using Antigravity CLI and Gemini.

Executes the StrawDI detection eval pipeline over the full dataset (or a limit)
by invoking the Antigravity CLI (`agy`) with `gemini-3.8-flash-high` under the
`agent_bridge` protocol:
1. Prepares a fresh state directory with vision-delivery control.
2. Runs the synthetic control call and verifies delivery.
3. Distributes frame calls across parallel workers (--jobs).
4. Handles the one recorded resample on schema-invalid replies.
5. Runs `bridge.py assemble` to score, render overlays, and write report.md.
6. Runs `verify_strawdi_run.py` to assert the 32 acceptance gates.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

BRIDGE_DIR = Path(__file__).resolve().parent
HERE = BRIDGE_DIR.parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from strawdi_eval.agent_bridge import bridge
from strawdi_eval import run_detection_eval as rde
from strawdi_eval.lib import prompt as strawdi_prompt
from vlm_eval import prompts as base_prompts
from vlm_eval.lib import parse


def call_agy(prompt: str, model: str, timeout: float = 240.0) -> str:
    cmd = [
        "agy", "--model", model,
        "--dangerously-skip-permissions",
        "--output-format", "json",
        "--print", prompt,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"agy exited with code {p.returncode}: {p.stderr.strip()[:300]}")
    try:
        data = json.loads(p.stdout)
        resp = data.get("response", "")
        if not resp and data.get("error"):
            raise RuntimeError(f"agy error: {data['error']}")
        return resp
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"failed to parse agy JSON: {exc}; raw output: {p.stdout[:300]}")


def run_batch(args) -> None:
    manifest_path = (HERE / "manifest.json").resolve()
    manifest = json.loads(manifest_path.read_text())
    by_id = {s["sample_id"]: s for s in manifest["samples"]}

    # Prepare or load state
    if args.state:
        state, plan = bridge._state(args)
        print(f"Resuming existing state: {state}")
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        state = bridge.STATE_ROOT / stamp
        (state / "answers").mkdir(parents=True, exist_ok=False)

        control_png = state / "control.png"
        truth_path = state / "control_truth.json"
        bridge.generate_control(control_png, truth_path)

        control_prompt = f"{base_prompts.CONTROL_PROMPT}\n\n{rde.vre.TOOL_NOTICE}"
        (state / "control_prompt.txt").write_text(control_prompt)

        samples = manifest["samples"][: args.limit] if args.limit else manifest["samples"]
        (state / "prompt.txt").write_text(rde.build_prompt(samples[0]))

        plan = {
            "bridge": {"name": "strawdi_eval/agent_bridge", "version": bridge.BRIDGE_VERSION,
                       "path": str(BRIDGE_DIR), "provider": args.provider},
            "model": args.model,
            "effort": args.effort,
            "tag": args.tag,
            "started": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "stamp": stamp,
            "manifest": str(manifest_path),
            "sample_ids": [s["sample_id"] for s in samples],
            "n_planned": len(samples),
            "frame_dir": str(HERE / "data" / "frames"),
            "control": {"image": str(control_png), "prompt": str(state / "control_prompt.txt"),
                        "truth_file": str(truth_path)},
            "prompt_style": strawdi_prompt.STYLE_NAME,
            "harness": rde.harness_info(),
            "base_harness": rde.vre.harness_info(),
            "usage_source": "estimated",
            "token_estimator": bridge.TOKEN_ESTIMATOR,
            "notes": [
                "answers/<sample_id>.json holds the raw reply text for that frame",
                "answers/<sample_id>.retry.json (optional) is the one recorded resample",
                "the operator must not read control_truth.json, manifest gt_boxes or overlays while answers are being written",
            ],
        }
        (state / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        print(f"Prepared state: {state} ({len(samples)} samples planned)")

    # 1. Vision Delivery Control
    control_answer_path = state / "answers" / "control.json"
    if not control_answer_path.exists():
        ctrl_img = Path(plan["control"]["image"]).resolve()
        ctrl_prompt = (
            f"First view the image file at {ctrl_img}.\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. Read the code and shape coordinates directly from the image.\n"
            "2. Do NOT run any commands or execute scripts.\n"
            "3. Reply with ONLY this JSON object and nothing else - no prose, no code fences:\n"
            '{"code": "0000", "red_circle": [u, v], "green_square": [u, v]}\n\n'
            f"{base_prompts.CONTROL_PROMPT}"
        )
        print(f"Running vision delivery control on {ctrl_img.name}...", flush=True)
        t0 = time.time()
        ctrl_reply = call_agy(ctrl_prompt, args.model, args.timeout)
        control_answer_path.write_text(ctrl_reply)
        print(f"Control answer recorded in {time.time() - t0:.1f}s", flush=True)

    # 2. Scored Frame Calls
    schema = json.loads(rde.SCHEMA_PATH.read_text())
    sample_ids = plan["sample_ids"]
    lock = threading.Lock()
    done_count = sum(1 for sid in sample_ids if bridge._answer_file(state, sid))
    print(f"Starting frame evaluation: {done_count}/{len(sample_ids)} already answered. Concurrency: {args.jobs}", flush=True)

    def process_sample(sid: str) -> None:
        nonlocal done_count
        answer_path = state / "answers" / f"{sid}.json"
        retry_path = state / "answers" / f"{sid}.retry.json"
        if bridge._answer_file(state, sid):
            return

        sample = by_id[sid]
        h, w = sample["image_shape_hw"]
        frame_path = (HERE / sample["images"]["raw"]).resolve()

        prompt_body = strawdi_prompt.build_inventory_detection(frame_w=w, frame_h=h)
        full_prompt = (
            f"First inspect the image file at {frame_path}.\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. Answer strictly from visual perception of this image alone.\n"
            "2. Do NOT run any shell commands, python scripts, or other tools.\n"
            "3. Reply with ONLY a JSON object and nothing else - no prose, no code fences, no extra fields.\n\n"
            f"{prompt_body}"
        )

        t0 = time.time()
        try:
            reply = call_agy(full_prompt, args.model, args.timeout)
            answer_path.write_text(reply)
            scored = rde.classify_detection(reply, schema)
            status = scored["status"]
            n_berries = scored["n_strawberries"]
            elapsed = time.time() - t0

            # One retry on schema invalid reply (harness v0.2.1 policy)
            if status == parse.SCHEMA_INVALID:
                with lock:
                    print(f"[{sid}] schema_invalid on first attempt ({scored.get('schema_error')}); retrying once...", flush=True)
                t_retry = time.time()
                try:
                    retry_reply = call_agy(full_prompt, args.model, args.timeout)
                    retry_path.write_text(retry_reply)
                    scored_retry = rde.classify_detection(retry_reply, schema)
                    status = scored_retry["status"]
                    n_berries = scored_retry["n_strawberries"]
                    elapsed += time.time() - t_retry
                except Exception as exc:
                    with lock:
                        print(f"[{sid}] retry call error: {exc}", flush=True)

            with lock:
                done_count += 1
                pct = (done_count / len(sample_ids)) * 100
                print(f"[{done_count:3d}/{len(sample_ids)}] frame {sid:5s} -> {status:14s} ({n_berries} berries) in {elapsed:.1f}s [{pct:5.1f}%]", flush=True)
        except Exception as exc:
            with lock:
                print(f"[{sid}] ERROR: {exc}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(process_sample, sid) for sid in sample_ids]
        concurrent.futures.wait(futures)

    # 3. Assemble and Verify
    print("\nAll planned calls finished. Checking answers...", flush=True)
    assemble_args = argparse.Namespace(
        state=state,
        manifest=manifest_path,
        out=HERE / "runs",
        tag=args.tag,
        model=args.model,
        effort=args.effort,
        exclude="",
        exclude_reason=None,
        dry_dir=False,
    )
    bridge.cmd_check(assemble_args)
    print("\nAssembling run directory...", flush=True)
    bridge.cmd_assemble(assemble_args)

    # Find the newly assembled run directory
    tag_suffix = f"-{args.tag}" if args.tag else ""
    runs = sorted(p for p in (HERE / "runs").glob(f"*-{rde.vre.slugify(args.model)}-{rde.vre.slugify(args.effort)}-{rde.vre.slugify(args.provider)}-strawdi_eval{tag_suffix}"))
    if not runs:
        # Fallback to latest
        runs = sorted((HERE / "runs").iterdir())
    latest_run = runs[-1]

    print(f"\nRunning acceptance verification on {latest_run}...", flush=True)
    verify_script = HERE / "verify_strawdi_run.py"
    res = subprocess.run([sys.executable, str(verify_script), str(latest_run)], text=True)
    if res.returncode == 0:
        print("\n=== EVALUATION COMPLETE: PASS ===", flush=True)
    else:
        print("\n=== EVALUATION FAILED ACCEPTANCE GATE ===", flush=True)
        sys.exit(res.returncode)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", type=Path, default=None, help="resume from existing state directory")
    ap.add_argument("--jobs", type=int, default=3, help="parallel agy calls")
    ap.add_argument("--tag", default="full", help="tag for the run (default: full)")
    ap.add_argument("--model", default=bridge.DEFAULT_MODEL, help="model slug")
    ap.add_argument("--provider", default=bridge.PROVIDER, help="provider name")
    ap.add_argument("--effort", default=bridge.DEFAULT_EFFORT, help="reasoning effort")
    ap.add_argument("--limit", type=int, default=None, help="limit frames for testing")
    ap.add_argument("--timeout", type=float, default=240.0, help="seconds per call timeout")
    args = ap.parse_args()
    run_batch(args)


if __name__ == "__main__":
    main()
