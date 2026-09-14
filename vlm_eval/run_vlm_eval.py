#!/usr/bin/env python3
"""Raw-capability VLM eval: strawberry detection + picking point, via ``codex exec``.

The model under test is whatever ``codex`` is configured with (the user's
``model`` setting; ``deepseek-flash`` at the time of writing).  Each run hands the
model one prepared image plus one prompt and nothing else, then scores the answer
against geometric ground truth.  No detector, tracker, segmenter or other model
participates anywhere in the loop.

Two things make the numbers trustworthy, and both are checked rather than
assumed:

1. **Images actually arrive.**  ``codex exec -i`` only accepts images when the
   model catalog says the slug is multimodal, so a local catalog override is
   passed via ``-c``.  A synthetic control image is run first; if the model
   cannot read its code and shapes, the report says the run is invalid instead
   of quietly reporting blind numbers.
2. **The model cannot peek.**  ``codex exec`` normally gives the model a shell,
   with which it will happily open the frame (or the episode's trajectory files
   and intrinsics) straight off disk.  Every tool surface is therefore disabled
   for these runs, and any tool attempt is recorded per run.

Usage:
    python3 vlm_eval/run_vlm_eval.py --dry-run
    python3 vlm_eval/run_vlm_eval.py --limit 1 --styles baseline_json
    python3 vlm_eval/run_vlm_eval.py --jobs 3
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from vlm_eval.lib import gtbridge, imaging, parse  # noqa: E402
from vlm_eval import prompts  # noqa: E402

# Identity of the harness itself. Every report states which harness produced it,
# which model answered, and at what reasoning effort, so a number can never be
# detached from the code and configuration that generated it.
HARNESS_NAME = "vlm_eval"
HARNESS_VERSION = "1.2.0"

# Default evaluation scope. The SROI wrist-camera sources (validation / occluded)
# are still built by build_manifest.py and still supported, but they are not
# evaluated unless explicitly requested:
#     --sources validation occluded
# They are the only sources with ground truth, so including them is what turns on
# every scored metric (error, PCK, IoU, sensitivity).
DEFAULT_SOURCES = ("shunba",)

DEFAULT_RUNS_DIR = HERE / "runs"
SCHEMA_PATH = HERE / "schema" / "response_schema.json"
CONTROL_SCHEMA_PATH = HERE / "schema" / "control_schema.json"
ENUMERATE_SCHEMA_PATH = HERE / "schema" / "enumerate_schema.json"
INVENTORY_SCHEMA_PATH = HERE / "schema" / "inventory_schema.json"

# Empirically measured (via a local capture server): codex delivers an image
# byte-for-byte when it is at or below this size, and silently downscales it
# above. A downscaled image would put the model's coordinates in a different
# frame from the one the harness draws on, so inputs must stay inside this box.
#   640x480, 1280x720, 1280x960, 1470x600, 1920x1080 -> unchanged
#   2048x1536, 2560x1440, 4032x3024                   -> downscaled
MAX_SAFE_EDGE = 1920
MAX_SAFE_PIXELS = 1920 * 1080

# Tool surfaces disabled for every run.  ``shell_tool``/``unified_exec`` are the
# ones that matter; the rest are belt-and-braces so no other read path exists.
DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "view_image",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "image_generation",
    "tool_suggest",
)

TOOL_NOTICE = (
    "Answer only from the image you were given. You have no tools, no shell and no "
    "file access for this task, so do not attempt to read any file, and do not emit "
    "tool calls."
)

# Heuristic: the CLI logs blocked tool attempts like
#   ERROR codex_core::tools::router: error=unsupported call: exec
TOOL_ATTEMPT_RE = re.compile(r"unsupported call:\s*(\w+)", re.IGNORECASE)

PCK_THRESHOLDS = (5, 10, 20)


# ---------------------------------------------------------------------------
# Configuration plumbing
# ---------------------------------------------------------------------------

def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex"))


def harness_info() -> dict:
    """Name, version and a content fingerprint of the code that computes the numbers.

    The fingerprint covers the harness sources and schemas, so two reports can be
    told apart even when both claim the same version.
    """
    digest = hashlib.sha256()
    files = sorted(HERE.glob("*.py")) + sorted((HERE / "lib").glob("*.py")) \
        + sorted((HERE / "schema").glob("*.json"))
    for path in files:
        digest.update(path.relative_to(HERE).as_posix().encode())
        digest.update(path.read_bytes())
    return {
        "name": HARNESS_NAME,
        "version": HARNESS_VERSION,
        "fingerprint": digest.hexdigest()[:12],
        "path": str(HERE),
        "files_hashed": len(files),
    }


def slugify(text, fallback: str = "unknown") -> str:
    """Filesystem-safe path component (no separators, no surprises)."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text or "")).strip("-._")
    return cleaned or fallback


def run_directory_name(stamp: str, model: str, effort: str | None,
                       tag: str = "", cli: str = "") -> str:
    """Name a run directory so the identity is readable without opening anything.

    Layout: <timestamp>-<model>-<effort>-<cli>-<harness>[-<tag>]
    e.g.    20260914-192928-glm-5.3-flash-default-claude-vlm_eval-full
            20260914-171146-deepseek-flash-high-codex-vlm_eval-full

    ``cli`` is the agent harness that talked to the model (codex, claude, kimi,
    ...) — the same number through a different CLI is a different measurement.
    """
    parts = [stamp, slugify(model), slugify(effort or "default")]
    if cli:
        parts.append(slugify(cli))
    parts.append(slugify(HARNESS_NAME))
    if tag:
        parts.append(slugify(tag))
    return "-".join(parts)


def read_user_config() -> dict:
    path = codex_home() / "config.toml"
    if not path.exists():
        return {}
    import tomllib

    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except Exception:
        return {}


def catalog_entry(args, model: str) -> dict:
    """Look the model up in the vision-enabled catalog override.

    This is a hard error rather than a warning: if the slug is missing, codex
    falls back to generic metadata, and whether the fallback happens to support
    images is not something a capability measurement should depend on.
    """
    path = args.catalog
    if not path.exists():
        raise SystemExit(
            f"catalog override not found: {path}\n"
            "Run: python3 vlm_eval/make_catalog.py"
        )
    catalog = json.loads(path.read_text())
    entries = {m.get("slug") or m.get("id"): m for m in catalog.get("models", [])}
    entry = entries.get(model)
    if entry is None:
        raise SystemExit(
            f"model {model!r} is not in {path.name} (present: {', '.join(sorted(map(str, entries)))}).\n"
            "The user-level catalog probably changed. Regenerate the override with:\n"
            "    python3 vlm_eval/make_catalog.py"
        )
    if "image" not in (entry.get("input_modalities") or []):
        raise SystemExit(
            f"model {model!r} in {path.name} does not declare the 'image' input modality, "
            "which silently disables image delivery. Regenerate with make_catalog.py."
        )
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_slug": model,
        "input_modalities": entry.get("input_modalities"),
        "source_catalog": str(args.catalog_source),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    ap.add_argument("--styles", nargs="*", default=None,
                    help="prompts to run, by name or title (e.g. inventory_plain, "
                         "'Full inventory + target (scan first)'). Omit to run all. "
                         "Pre-rename names (inventory_json, baseline_json, ...) still "
                         "resolve.")
    ap.add_argument("--list-styles", action="store_true",
                    help="print the available prompts (name, what it runs on, title) "
                         "and exit")
    ap.add_argument("--out", type=Path, default=DEFAULT_RUNS_DIR,
                    help="runs root; a timestamped subdirectory is created inside it")
    ap.add_argument("--tag", default="", help="suffix for the run directory name")
    ap.add_argument("--limit", type=int, default=None, help="max samples (manifest order)")
    ap.add_argument("--sources", nargs="*", default=None,
                    help="only these manifest sources. Default: "
                         f"{' '.join(DEFAULT_SOURCES)} (unlabelled, qualitative). "
                         "Pass 'validation occluded shunba' to restore the "
                         "ground-truthed SROI frames and the scored metrics.")
    ap.add_argument("--model", default=None,
                    help="model slug; defaults to the user's configured model")
    ap.add_argument("--reasoning-effort", default=None,
                    help="model_reasoning_effort override; default keeps the user's setting")
    ap.add_argument("--provider", choices=("codex", "glm", "claude"), default="codex")
    ap.add_argument("--glm-model", default=None,
                    help="model slug to use with --provider glm")
    ap.add_argument("--claude-model", default=None,
                    help="model slug to use with --provider claude. Default: glm-5.3-flash, "
                         "the multimodal GLM-5.3 — the flagship glm-5.3 slug rejects image "
                         "content on this key, so the vision control would fail it")
    ap.add_argument("--catalog", type=Path, default=HERE / "model_catalog_vision.json",
                    help="vision-enabled model catalog override passed with -c")
    ap.add_argument("--catalog-source", type=Path,
                    default=codex_home() / "cc-switch-model-catalog.json",
                    help="source used by make_catalog.py, recorded for provenance")
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds per model call")
    ap.add_argument("--jobs", type=int, default=1, help="parallel model calls")
    ap.add_argument("--price-in-per-mtok", type=float, default=None,
                    help="optional input price for the cost column")
    ap.add_argument("--price-out-per-mtok", type=float, default=None,
                    help="optional output price for the cost column")
    ap.add_argument("--control-image", type=Path, default=None,
                    help="override the prepared control image path")
    ap.add_argument("--skip-control", action="store_true",
                    help="skip the vision-delivery control (results marked unverified)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and assembled prompts without calling the model")
    ap.add_argument("--rebuild-report", action="store_true",
                    help="refresh a finished run's artefacts from its responses.jsonl "
                         "instead of calling the model; pass the run directory via --out")
    args = ap.parse_args(argv)

    args.manifest = args.manifest.resolve()
    args.out = args.out.resolve()
    args.catalog = args.catalog.resolve()
    return args


# ---------------------------------------------------------------------------
# Running the model
# ---------------------------------------------------------------------------

def build_command(args, manifest, image: Path, prompt: str, out_message: Path,
                  schema: Path | None, model: str, effort: str | None,
                  working_dir: Path, extra: list[str] | None = None) -> list[str]:
    cmd = ["codex", "exec"]
    for feature in DISABLED_FEATURES:
        cmd += ["--disable", feature]
    cmd += [
        "-c", f'model_catalog_json="{args.catalog}"',
        "--skip-git-repo-check",
        "--ephemeral",
        "--sandbox", "read-only",
        "-C", str(working_dir),
        "-m", model,
    ]
    if effort:
        cmd += ["-c", f'model_reasoning_effort="{effort}"']
    cmd += provider_overrides(args)
    cmd += ["-i", str(image)]
    if schema is not None:
        cmd += ["--output-schema", str(schema)]
    cmd += [
        "-o", str(out_message),
        "--json",
        "--",
        prompt,
    ]
    if extra:
        cmd += extra
    return cmd


def provider_overrides(args) -> list[str]:
    """Provider switch.  ``codex`` = whatever the user configured; ``glm`` reroutes."""
    if args.provider == "codex":
        return []
    key_env = "ZHIPU_API_KEY"
    if not os.environ.get(key_env):
        raise SystemExit(
            f"--provider glm needs {key_env} in the environment (the GLM API key). "
            "No GLM comparison runs were part of this round."
        )
    return [
        "-c", 'model_provider="glm"',
        "-c", 'model_providers.glm.name="glm"',
        "-c", 'model_providers.glm.base_url="https://open.bigmodel.cn/api/v1"',
        "-c", 'model_providers.glm.wire_api="responses"',
        "-c", f'model_providers.glm.env_key="{key_env}"',
    ]


# --- provider: claude -------------------------------------------------------
#
# Runs the model through the ``claude`` CLI in headless print mode instead of
# ``codex exec``.  The mapping of the eight invariants:
#
#   I1 images   the frame is a base64 image content block in the one stream-json
#               user message written to stdin (``--input-format stream-json``);
#               there is no codex catalog on this path, so the synthetic control
#               is the delivery gate.  NOTE: whether the image survives depends
#               on the *endpoint*: e.g. the flagship glm-5.3 slug rejects image
#               content server-side, while glm-5.3-flash delivers it byte-for-byte.
#   I3 tools    ``--tools ""`` removes every tool from the API request itself, so
#               a tool call cannot even be generated; ``--safe-mode`` additionally
#               drops hooks/plugins/skills/CLAUDE.md, and ``--strict-mcp-config``
#               loads no MCP servers.  The event stream is still scanned for
#               tool_use blocks and permission denials, and any hit fails verify.
#   schema      ``--json-schema <file>`` is the CLI's enforced structured output
#               (an internal ``StructuredOutput`` tool call, excluded from the
#               tool-attempt scan because it is harness machinery, not the model
#               reaching for a tool).

# Tool calls the CLI itself generates to implement --json-schema; not the model
# reaching for a tool surface, so they are excluded from tool_attempts.
CLAUDE_INTERNAL_TOOLS = {"StructuredOutput"}


def build_claude_command(schema: Path | None, model: str,
                         effort: str | None) -> list[str]:
    cmd = [
        "claude", "-p",
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--verbose",
        "--model", model,
        "--tools", "",
        "--safe-mode",
        "--strict-mcp-config",
        "--no-session-persistence",
    ]
    if effort:
        cmd += ["--effort", effort]
    if schema is not None:
        cmd += ["--json-schema", schema.read_text()]
    return cmd


def claude_child_env(model: str) -> dict:
    """Environment for the headless CLI call.

    The ANTHROPIC_* routing (base URL + auth token) is inherited so the call goes
    through exactly the endpoint this Claude Code installation is configured
    with; this session's CLAUDE_* markers are dropped so the child is a clean,
    independent invocation.
    """
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("CLAUDE")}
    env["ANTHROPIC_MODEL"] = model
    return env


def claude_stdin_payload(image: Path, prompt: str) -> bytes:
    """One stream-json user message: the frame as pixels, plus the prompt."""
    media = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
    payload = {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": media,
                    "data": base64.b64encode(image.read_bytes()).decode()}},
                {"type": "text", "text": prompt},
            ],
        },
    }
    return (json.dumps(payload) + "\n").encode()


def parse_claude_events(stdout: str) -> dict:
    """Pull the answer, usage and tool attempts out of the stream-json output."""
    messages: list[str] = []
    errors: list[str] = []
    usage = None
    last_message = ""
    structured_output = None
    api_cost_usd = None
    tool_names: set[str] = set()
    permission_denials = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = event.get("type")
        if etype == "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    tool_names.add(str(block.get("name")))
                elif block.get("type") == "text" and str(block.get("text", "")).strip():
                    messages.append(block["text"])
        elif etype == "result":
            structured_output = event.get("structured_output")
            result_text = event.get("result")
            if isinstance(result_text, str) and result_text.strip():
                last_message = result_text
            if event.get("is_error"):
                errors.append(str(result_text or event.get("subtype") or "error"))
            permission_denials = event.get("permission_denials") or []
            for entry in (event.get("modelUsage") or {}).values():
                usage = {
                    "input_tokens": entry.get("inputTokens"),
                    "cached_input_tokens": (entry.get("cacheReadInputTokens") or 0)
                                           + (entry.get("cacheCreationInputTokens") or 0),
                    "output_tokens": entry.get("outputTokens"),
                }
                api_cost_usd = entry.get("costUSD")
    if not last_message and messages:
        last_message = messages[-1]
    tool_attempts = {name for name in tool_names if name not in CLAUDE_INTERNAL_TOOLS}
    for denial in permission_denials:
        name = denial.get("tool") if isinstance(denial, dict) else str(denial)
        if name and name not in CLAUDE_INTERNAL_TOOLS:
            tool_attempts.add(f"denied:{name}")
    return {
        "usage": usage,
        "messages": messages,
        "errors": errors,
        "last_message": last_message,
        "structured_output": structured_output,
        "api_cost_usd": api_cost_usd,
        "tool_attempts": sorted(tool_attempts),
    }


def call_model_claude(args, image: Path, prompt: str, schema: Path | None,
                      model: str, effort: str | None, working_dir: Path,
                      timeout: float) -> dict:
    cmd = build_claude_command(schema, model, effort)
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, input=claude_stdin_payload(image, prompt),
                              capture_output=True, timeout=timeout,
                              env=claude_child_env(model), cwd=str(working_dir))
        returncode, stdout, stderr = proc.returncode, \
            proc.stdout.decode(errors="replace"), proc.stderr.decode(errors="replace")
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        returncode = None
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) \
            else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) \
            else (exc.stderr or "")
        timed_out = True
    wall = time.perf_counter() - started

    parsed = parse_claude_events(stdout)
    if returncode not in (0, None):
        parsed["errors"].append(stderr.strip()[:400])
    return {
        "command": cmd,
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_s": round(wall, 3),
        "stdout": stdout,
        "stderr": stderr,
        "usage": parsed["usage"],
        "events": [],
        "errors": parsed["errors"],
        "tool_attempts": parsed["tool_attempts"],
        "last_message": parsed["last_message"],
        "structured_output": parsed["structured_output"],
        "api_cost_usd": parsed["api_cost_usd"],
    }


def claude_catalog_info(args, model: str) -> dict:
    """Provenance snapshot for the claude-code provider.

    There is no codex model-catalog override on this path; what matters instead
    is which endpoint the CLI routed to and with which slug, because image
    delivery is decided server-side (the glm-5.3 flagship silently rejects
    images; glm-5.3-flash delivers them).  The synthetic control remains the
    hard gate either way.
    """
    return {
        "provider": "claude",
        "model_slug": model,
        "routing_base_url": os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        "delivery": "claude -p --input-format stream-json: base64 image content block on stdin",
        "input_modalities": ["image"],
        "cli_version": claude_version(),
        "note": "No codex catalog override applies. Image delivery is gated by the "
                "synthetic vision control, which runs first and invalidates the "
                "batch if the endpoint drops the image.",
    }


def claude_version() -> str:
    try:
        version = subprocess.run(["claude", "--version"], capture_output=True, text=True,
                                 timeout=30)
        return (version.stdout.strip() or version.stderr.strip()) or "unknown"
    except Exception:
        return "unknown"


def cli_identity(provider: str) -> dict:
    if provider == "claude":
        return {"cli_name": "claude", "cli_version": claude_version()}
    return {"cli_name": "codex", "cli_version": codex_version()}


def parse_events(stdout: str) -> dict:
    """Pull usage, message items and tool attempts out of the ``--json`` stream."""
    events, usage, messages, errors = [], None, [], []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        events.append(event)
        etype = event.get("type")
        if etype == "turn.completed" and isinstance(event.get("usage"), dict):
            usage = event["usage"]
        item = event.get("item") or {}
        if etype == "item.completed":
            if item.get("type") == "agent_message" and item.get("text"):
                messages.append(item["text"])
            elif item.get("type") == "error":
                errors.append(str(item.get("message", "")))
    return {"events": events, "usage": usage, "messages": messages, "errors": errors}


def call_model(args, manifest, image: Path, prompt: str, out_message: Path,
               schema: Path | None, model: str, effort: str | None,
               working_dir: Path, timeout: float) -> dict:
    if args.provider == "claude":
        return call_model_claude(args, image, prompt, schema, model, effort,
                                 working_dir, timeout)
    cmd = build_command(args, manifest, image, prompt, out_message, schema, model,
                        effort, working_dir)
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, timeout=timeout)
        returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        returncode = None
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        timed_out = True
    wall = time.perf_counter() - started

    parsed = parse_events(stdout)
    tool_attempts = sorted(set(TOOL_ATTEMPT_RE.findall(stderr + "\n" + stdout)))
    last_message = out_message.read_text() if out_message.exists() else ""
    if not last_message and parsed["messages"]:
        last_message = parsed["messages"][-1]
    return {
        "command": cmd,
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_s": round(wall, 3),
        "stdout": stdout,
        "stderr": stderr,
        "usage": parsed["usage"],
        "events": parsed["events"],
        "errors": parsed["errors"],
        "tool_attempts": tool_attempts,
        "last_message": last_message,
    }


def token_totals(usage: dict | None) -> dict:
    usage = usage or {}
    inp = int(usage.get("input_tokens") or 0)
    cached = int(usage.get("cached_input_tokens") or 0)
    out = int(usage.get("output_tokens") or 0)
    reasoning = int(usage.get("reasoning_output_tokens") or 0)
    return {
        "input_tokens": inp,
        "cached_input_tokens": cached,
        "output_tokens": out,
        "reasoning_output_tokens": reasoning,
        "total_tokens": inp + out,
    }


def merge_usage(attempts: list[dict]) -> dict:
    totals = {k: 0 for k in ("input_tokens", "cached_input_tokens", "output_tokens",
                             "reasoning_output_tokens", "total_tokens")}
    for attempt in attempts:
        for key, value in token_totals(attempt.get("usage")).items():
            totals[key] += value
    return totals


# ---------------------------------------------------------------------------
# One scored run
# ---------------------------------------------------------------------------

def prepare_image(style, sample: dict) -> tuple[Path, tuple[int, int]]:
    """Return (image path, panel origin) for a style/sample pair."""
    if style.image_kind == "grid":
        return HERE / sample["images"]["grid_overlay"], (0, 0)
    if style.image_kind == "exemplar":
        return HERE / sample["images"]["few_shot_exemplar"], tuple(sample["exemplar_query_origin"])
    return HERE / sample["images"]["raw"], (0, 0)


def style_applies(style, sample: dict) -> bool:
    """Whether a prompt style is meaningful for a sample (scored vs unlabelled)."""
    if style.applies_to == "any":
        return True
    return style.applies_to == ("gt" if sample.get("has_gt") else "no_gt")


def select_samples(args, manifest) -> list[dict]:
    """Apply --sources then --limit, in manifest order.

    With no --sources, only DEFAULT_SOURCES are evaluated; other sources stay in
    the manifest and can be switched back on explicitly.
    """
    samples = manifest["samples"]
    requested = set(args.sources) if args.sources else set(DEFAULT_SOURCES)
    known = {s["source"] for s in samples}
    unknown = sorted(requested - known)
    if unknown:
        raise SystemExit(f"unknown source(s): {unknown}; known: {sorted(known)}")
    samples = [s for s in samples if s["source"] in requested]
    return samples[: args.limit] if args.limit else samples


def check_input_sizes(samples: list[dict], styles) -> None:
    """Refuse to run if any input image is large enough for codex to downscale it.

    A silently rescaled image would shift every coordinate the model reports
    relative to the frame the harness annotates, which is undetectable from the
    scores when a sample has no ground truth.
    """
    offenders = []
    for sample in samples:
        shapes = {"raw": sample.get("image_shape_hw")}
        shapes.update(sample.get("derived_image_shape_hw") or {})
        for kind, shape in shapes.items():
            if not shape:
                continue
            h, w = shape
            if max(h, w) > MAX_SAFE_EDGE or h * w > MAX_SAFE_PIXELS:
                offenders.append(f"{sample['sample_id']}:{kind} {w}x{h}")
    if offenders:
        raise SystemExit(
            "input image(s) too large for the CLI's no-resize ceiling "
            f"({MAX_SAFE_EDGE}px max edge / {MAX_SAFE_PIXELS} px): "
            + ", ".join(offenders)
            + "\nRebuild the manifest with a smaller normalisation, e.g. "
              "`python3 vlm_eval/build_manifest.py --shunba-max-edge 1280`."
        )


def classify(last_message: str, schema: dict, origin: tuple[int, int],
             frame_w: int, frame_h: int) -> dict:
    """Turn a raw reply into a status plus normalised frame coordinates."""
    result = {
        "status": None, "json_method": None, "schema_valid": None,
        "schema_error": None, "parse_error": None,
        "picking_point_global": None, "picking_point": None, "bbox": None,
        "bbox_global": None, "ripe": None,
    }
    text = (last_message or "").strip()
    if not text:
        result["status"] = parse.EMPTY_RESPONSE
        return result

    obj, method = parse.extract_json(text)
    result["json_method"] = method
    if obj is None:
        result["status"] = parse.REFUSED if parse.looks_like_refusal(text) else parse.PARSE_ERROR
        return result

    valid, error = parse.validate(obj, schema)
    result["schema_valid"] = valid
    result["schema_error"] = error
    if not valid:
        result["status"] = parse.SCHEMA_INVALID
        return result

    point = parse.normalise_point(obj["picking_point"])
    box = parse.normalise_box(obj["bbox"])
    result["ripe"] = obj.get("ripe")
    result["picking_point_global"] = list(point) if point else None
    result["bbox_global"] = list(box) if box else None

    ox, oy = origin
    if point is None:
        result["status"] = parse.SCHEMA_INVALID
        result["schema_error"] = "picking_point not a 2-element numeric array"
        return result
    local = (point[0] - ox, point[1] - oy)
    result["picking_point"] = [local[0], local[1]]
    if box is not None:
        result["bbox"] = [box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy]

    in_frame = 0 <= local[0] < frame_w and 0 <= local[1] < frame_h
    result["status"] = parse.OK if in_frame else parse.OUT_OF_FRAME
    return result


def classify_enumerate(last_message: str, schema: dict, frame_w: int, frame_h: int) -> dict:
    """Turn a raw reply into a status plus a normalised list of strawberries."""
    result = {
        "status": None, "json_method": None, "schema_valid": None,
        "schema_error": None, "strawberries": [], "n_strawberries": 0,
        "n_out_of_frame": 0, "ripe_true": 0,
    }
    text = (last_message or "").strip()
    if not text:
        result["status"] = parse.EMPTY_RESPONSE
        return result

    obj, method = parse.extract_json(text)
    result["json_method"] = method
    if obj is None:
        result["status"] = parse.REFUSED if parse.looks_like_refusal(text) else parse.PARSE_ERROR
        return result

    # Accept both {"strawberries": [...]} and a bare top-level list.
    payload = {"strawberries": obj} if isinstance(obj, list) else obj
    valid, error = parse.validate(payload, schema)
    result["schema_valid"] = valid
    result["schema_error"] = error
    if not valid:
        result["status"] = parse.SCHEMA_INVALID
        return result

    berries = []
    for entry in payload["strawberries"]:
        point = parse.normalise_point(entry.get("picking_point"))
        box = parse.normalise_box(entry.get("bbox"))
        berries.append({
            "ripe": entry.get("ripe"),
            "bbox": list(box) if box else None,
            "picking_point": list(point) if point else None,
        })
    result["strawberries"] = berries
    result["n_strawberries"] = len(berries)
    result["ripe_true"] = sum(1 for b in berries if b["ripe"] is True)
    result["n_out_of_frame"] = sum(
        1 for b in berries
        if b["picking_point"] is not None
        and not (0 <= b["picking_point"][0] < frame_w and 0 <= b["picking_point"][1] < frame_h)
    )
    result["status"] = parse.OK
    return result


# Continuous numbers first, then booleans, then the free-text description. There is
# deliberately no ripeness category: redness is reported as a 0-100 value.
INVENTORY_ATTRS = ("redness_pct", "occlusion_pct", "calyx_visible",
                   "peduncle_visible", "graspable", "confidence_pct", "description")


def _reported_count(scored: dict, berries, is_list: bool, is_inventory: bool):
    """How many fruit a run reported, across all three answer shapes.

    Returns 0 for "reported none", a positive count otherwise, and None when the
    answer could not be parsed at all - the three cases must stay distinguishable.
    """
    if isinstance(berries, list):
        return len(berries)
    if is_list or is_inventory:
        return None
    # single-target styles: one fruit, unless the answer offers no grasp point.
    if scored.get("status") == parse.OK:
        return 1
    if scored.get("status") == parse.NO_PICK_POINT:
        return 0
    return None


def _inventory_summary(berries: list[dict] | None) -> dict:
    """Aggregate the per-fruit attributes so the report can summarise a scene."""
    if berries is None:
        return {key: None for key in (
            "n_inventory", "n_occluded", "n_mostly_hidden", "n_graspable",
            "n_calyx_visible", "n_peduncle_visible", "n_described",
            "mean_redness_pct", "max_redness_pct", "mean_occlusion_pct",
            "mean_confidence_pct")}

    def values(key):
        return [b.get(key) for b in berries if isinstance(b.get(key), (int, float))]

    def count(key, value=True):
        return sum(1 for b in berries if b.get(key) is value)

    def mean(key):
        v = values(key)
        return round(sum(v) / len(v), 1) if v else None

    return {
        "n_inventory": len(berries),
        "n_occluded": sum(1 for b in berries
                          if isinstance(b.get("occlusion_pct"), (int, float))
                          and b["occlusion_pct"] > 0),
        "n_mostly_hidden": sum(1 for b in berries
                               if isinstance(b.get("occlusion_pct"), (int, float))
                               and b["occlusion_pct"] >= 50),
        "n_graspable": count("graspable"),
        "n_calyx_visible": count("calyx_visible"),
        "n_peduncle_visible": count("peduncle_visible"),
        "n_described": sum(1 for b in berries
                           if isinstance(b.get("description"), str) and b["description"].strip()),
        "mean_redness_pct": mean("redness_pct"),
        "max_redness_pct": max(values("redness_pct")) if values("redness_pct") else None,
        "mean_occlusion_pct": mean("occlusion_pct"),
        "mean_confidence_pct": mean("confidence_pct"),
    }


def classify_inventory(last_message: str, schema: dict, frame_w: int,
                       frame_h: int) -> dict:
    """Parse an unbiased full-scene inventory plus its nominated pick target."""
    result = {
        "status": None, "json_method": None, "schema_valid": None,
        "schema_error": None, "strawberries": [], "n_strawberries": 0,
        "target_index": None, "target": None, "target_valid": None,
        "n_out_of_frame": 0,
    }
    text = (last_message or "").strip()
    if not text:
        result["status"] = parse.EMPTY_RESPONSE
        return result

    obj, method = parse.extract_json(text)
    result["json_method"] = method
    if obj is None:
        result["status"] = parse.REFUSED if parse.looks_like_refusal(text) else parse.PARSE_ERROR
        return result

    # Accept a bare list too: treat it as an inventory with no nomination.
    payload = {"strawberries": obj, "target_index": -1} if isinstance(obj, list) else obj
    valid, error = parse.validate(payload, schema)
    result["schema_valid"] = valid
    result["schema_error"] = error
    if not valid:
        result["status"] = parse.SCHEMA_INVALID
        return result

    berries = []
    for entry in payload["strawberries"]:
        point = parse.normalise_point(entry.get("picking_point"))
        box = parse.normalise_box(entry.get("bbox"))
        berry = {"bbox": list(box) if box else None,
                 "picking_point": list(point) if point else None}
        berry.update({key: entry.get(key) for key in INVENTORY_ATTRS})
        berries.append(berry)

    result["strawberries"] = berries
    result["n_strawberries"] = len(berries)
    index = payload.get("target_index")
    if isinstance(index, int) and 0 <= index < len(berries):
        result["target_index"] = index
        result["target"] = berries[index]
        result["target_valid"] = True
    else:
        result["target_index"] = -1 if index in (-1, None) else index
        result["target_valid"] = index in (-1, None)
    result["n_out_of_frame"] = sum(
        1 for b in berries
        if b["picking_point"] is not None
        and not (0 <= b["picking_point"][0] < frame_w and 0 <= b["picking_point"][1] < frame_h)
    )
    # A parsed inventory that offers no usable grasp point - the model found the
    # fruit but reported the peduncle as hidden - is a valid answer that yields no
    # number, so it gets its own status rather than looking like an unscored run.
    result["status"] = (parse.OK if (result["target"] or {}).get("picking_point")
                        else parse.NO_PICK_POINT)
    return result


def run_one(args, manifest, style, sample, run_ctx) -> dict:
    """Execute (and if needed retry) one model call, then score it."""
    image, origin = prepare_image(style, sample)
    h, w = sample["image_shape_hw"]
    has_gt = bool(sample.get("has_gt"))
    prompt = style.build_prompt(frame_w=w, frame_h=h,
                                query_origin=sample.get("exemplar_query_origin") or (0, 0),
                                multi_fruit=not has_gt)
    prompt = f"{prompt}\n\n{TOOL_NOTICE}"
    # Parse the reply according to the shape this style ASKED for, not according to
    # whether the frame happens to have ground truth: a single-target style can
    # legitimately run on an unlabelled multi-fruit frame.
    is_list = style.output_shape == "list"
    is_inventory = style.output_shape == "inventory"
    schema_path = (INVENTORY_SCHEMA_PATH if is_inventory
                   else ENUMERATE_SCHEMA_PATH if is_list else SCHEMA_PATH)
    schema = json.loads(schema_path.read_text())

    def run_scorer(text):
        if is_inventory:
            return classify_inventory(text, schema, w, h)
        if is_list:
            return classify_enumerate(text, schema, w, h)
        return classify(text, schema, origin, w, h)

    model = run_ctx["model"]
    effort = run_ctx["effort"]
    stem = f"{style.name}__{sample['sample_id']}"
    attempts: list[dict] = []
    fallback_used = False

    message_path = run_ctx["tmp"] / f"{stem}.last.txt"
    attempt = call_model(args, manifest, image, prompt, message_path,
                         SCHEMA_PATH if style.use_output_schema else None,
                         model, effort, run_ctx["agent_cwd"], args.timeout)
    attempts.append(attempt)

    scored = run_scorer(attempt["last_message"])
    if scored["status"] == parse.EMPTY_RESPONSE and effort != "low":
        # Recorded fallback: reasoning can exhaust the output budget, leaving no
        # message at all.  Retry once at reduced effort and keep both attempts.
        fallback_used = True
        message_path = run_ctx["tmp"] / f"{stem}.retry.last.txt"
        retry = call_model(args, manifest, image, prompt, message_path,
                           SCHEMA_PATH if style.use_output_schema else None,
                           model, "low", run_ctx["agent_cwd"], args.timeout)
        retry["fallback_of_attempt"] = 1
        attempts.append(retry)
        scored = run_scorer(retry["last_message"])

    totals = merge_usage(attempts)
    gt_uv = sample["gt_uv"] if has_gt else None
    pred = scored.get("picking_point")
    # An inventory nominates its own pick target; that nomination is what gets
    # scored against ground truth, so a single call yields both the unbiased
    # inventory and a comparable picking point.
    target = scored.get("target") if is_inventory else None
    if is_inventory:
        pred = (target or {}).get("picking_point")
    # For a single-target style on an unlabelled frame we still want the point
    # recorded (it is what the overlay draws), just never scored.
    berries = scored.get("strawberries")
    if berries is None and pred is not None:
        berries = [{"ripe": scored.get("ripe"), "bbox": scored.get("bbox"),
                    "picking_point": pred}]

    error_px = dx = dy = None
    iou = contains = sensitivity = None
    if has_gt:
        rough = sample["gt_rough_box"]
        if pred is not None and sample["gt_status"] == "ok":
            dx = pred[0] - gt_uv[0]
            dy = pred[1] - gt_uv[1]
            error_px = float(np.hypot(dx, dy))
        scored_box = (target or {}).get("bbox") if is_inventory else scored.get("bbox")
        if scored_box is not None and gt_uv is not None:
            box = scored_box
            iou = _iou(box, rough)
            contains = bool(box[0] <= gt_uv[0] <= box[2] and box[1] <= gt_uv[1] <= box[3])
        if error_px is not None:
            sensitivity = gtbridge.sensitivity_effect(manifest["sensitivity_bridge"], dx, dy)

    record = {
        "run_id": stem,
        "style": style.name,
        "sample_id": sample["sample_id"],
        "has_gt": has_gt,
        "source": sample["source"],
        "session": sample["session"],
        "episode": sample["episode"],
        "attempts": len(attempts),
        "fallback_used": fallback_used,
        "fallback_reason": "empty response at configured reasoning effort" if fallback_used else None,
        "status": scored["status"],
        "json_method": scored["json_method"],
        "schema_valid": scored["schema_valid"],
        "schema_error": scored["schema_error"],
        "ripe": scored.get("ripe"),
        "model": model,
        "harness": HARNESS_NAME,
        "harness_version": HARNESS_VERSION,
        "harness_fingerprint": harness_info()["fingerprint"],
        "provider": args.provider,
        "reasoning_effort": effort,
        "used_output_schema": bool(style.use_output_schema),
        "output_shape": style.output_shape,
        "image": str(image.relative_to(HERE)),
        "image_kind": style.image_kind,
        "panel_origin": list(origin),
        "prompt": prompt,
        "gt_uv": gt_uv,
        "gt_uv_rounded": sample["gt_uv_rounded"] if has_gt else None,
        "gt_z_m": sample["gt_z_m"] if has_gt else None,
        "picking_point_global": scored.get("picking_point_global"),
        "picking_point": pred,
        "bbox": (target or {}).get("bbox") if is_inventory else scored.get("bbox"),
        "inventory": berries if is_inventory else None,
        "target_index": scored.get("target_index") if is_inventory else None,
        "target_valid": scored.get("target_valid") if is_inventory else None,
        "target_scored": bool(is_inventory and has_gt and pred is not None),
        **_inventory_summary(berries if is_inventory else None),
        # `strawberries` is the flat list answer; an inventory keeps its richer
        # per-fruit records under `inventory` instead.
        "strawberries": berries if (not has_gt and is_list) else None,
        # Distinguish "reported none" (0) from "could not parse" (None).
        # "How many fruit did this run report" - covers all three answer shapes.
        # "How many fruit did this run report" - meaningful for every shape,
        # including an inventory run on a ground-truthed frame.
        "n_strawberries": _reported_count(scored, berries, is_list, is_inventory),
        "n_ripe_reported": (
            scored.get("ripe_true") if is_list
            else (1 if scored.get("ripe") is True else (0 if has_gt else None))
        ) if (not has_gt and is_list) else None,
        "n_points_out_of_frame": (
            scored.get("n_out_of_frame") if (is_list or is_inventory) else
            (0 if pred is not None and 0 <= pred[0] < w and 0 <= pred[1] < h
             else (1 if pred is not None else None))
        ) if (not has_gt or is_inventory) else None,
        "error_px": error_px,
        "dx_px": dx,
        "dy_px": dy,
        "error_pct_width": None if error_px is None else error_px / float(w) * 100.0,
        "bbox_iou_rough": iou,
        "bbox_contains_gt": contains,
        "sensitivity": sensitivity,
        **{f"pck_{t}": None if error_px is None else bool(error_px <= t)
           for t in PCK_THRESHOLDS},
        **totals,
        "wall_s": round(sum(a["wall_s"] for a in attempts), 3),
        "returncodes": [a["returncode"] for a in attempts],
        "timed_out": any(a["timed_out"] for a in attempts),
        "tool_attempts": sorted({t for a in attempts for t in a["tool_attempts"]}),
        "cli_errors": [e for a in attempts for e in a["errors"]],
        "response_text": attempts[-1]["last_message"],
        "response_excerpt": (attempts[-1]["last_message"] or "")[:600],
        # Prefer explicit --price flags; otherwise report what the endpoint charged
        # (only the claude provider gets a per-call cost back from the API).
        "cost_usd": _cost(totals, args) if (args.price_in_per_mtok is not None
                                            and args.price_out_per_mtok is not None)
        else (round(sum(a.get("api_cost_usd") or 0.0 for a in attempts), 6) or None),
        "attempts_detail": [
            {
                "index": i,
                "returncode": a["returncode"],
                "wall_s": a["wall_s"],
                "timed_out": a["timed_out"],
                "usage": token_totals(a["usage"]),
                "tool_attempts": a["tool_attempts"],
                "cli_errors": a["errors"],
                "message_chars": len(a["last_message"] or ""),
            }
            for i, a in enumerate(attempts, start=1)
        ],
    }

    title = (f"{style.name} | {sample['sample_id']} | {sample['episode']} | "
             f"{scored['status']}")
    rel = Path("overlays") / f"{stem}.png"
    try:
        raw = imaging.load_rgb(HERE / sample["images"]["raw"])
        if has_gt:
            overlay = imaging.draw_prediction_overlay(
                raw, tuple(gt_uv), pred,
                (target or {}).get("bbox") if is_inventory else scored.get("bbox"),
                title)
        elif is_inventory:
            overlay = imaging.draw_inventory_overlay(
                raw, berries, scored.get("target_index"), title)
        else:
            overlay = imaging.draw_multi_overlay(raw, berries, title)
        imaging.save_rgb(overlay, run_ctx["run_dir"] / rel)
        record["overlay"] = str(rel)
    except Exception as exc:  # a drawing bug must not throw away a paid-for answer
        record["overlay"] = None
        record["render_error"] = f"{type(exc).__name__}: {exc}"
    return record


def _iou(a, b) -> float | None:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _overlay(raw_path: Path, gt_uv, pred, bbox, title):
    raw = imaging.load_rgb(raw_path)
    return imaging.draw_prediction_overlay(raw, tuple(gt_uv), pred, bbox, title)


def _cost(totals: dict, args) -> float | None:
    if args.price_in_per_mtok is None or args.price_out_per_mtok is None:
        return None
    return round(
        totals["input_tokens"] / 1e6 * args.price_in_per_mtok
        + totals["output_tokens"] / 1e6 * args.price_out_per_mtok,
        6,
    )


# ---------------------------------------------------------------------------
# Vision-delivery control
# ---------------------------------------------------------------------------

def run_control(args, manifest, run_ctx) -> dict:
    control_path = (args.control_image.resolve() if args.control_image
                    else HERE / manifest["control"]["image"])
    truth = manifest["control"]["truth"]
    message_path = run_ctx["tmp"] / "control.last.txt"
    attempt = call_model(args, manifest, control_path,
                         f"{prompts.CONTROL_PROMPT}\n\n{TOOL_NOTICE}",
                         message_path, CONTROL_SCHEMA_PATH,
                         run_ctx["model"], run_ctx["effort"], run_ctx["agent_cwd"],
                         args.timeout)
    schema = json.loads(CONTROL_SCHEMA_PATH.read_text())
    obj, method = parse.extract_json(attempt["last_message"])
    valid, schema_error = (parse.validate(obj, schema) if obj is not None else (False, "no JSON"))

    code_ok = circle_ok = square_ok = None
    circle_err = square_err = None
    if valid:
        code_ok = str(obj["code"]).strip() == truth["code"]
        pred_circle = parse.normalise_point(obj["red_circle"])
        pred_square = parse.normalise_point(obj["green_square"])
        if pred_circle:
            circle_err = float(np.hypot(pred_circle[0] - truth["red_circle"][0],
                                        pred_circle[1] - truth["red_circle"][1]))
            circle_ok = circle_err <= CONTROL_TOLERANCE_PX
        if pred_square:
            square_err = float(np.hypot(pred_square[0] - truth["green_square"][0],
                                        pred_square[1] - truth["green_square"][1]))
            square_ok = square_err <= CONTROL_TOLERANCE_PX

    delivered = bool(code_ok and circle_ok)
    return {
        "image": str(control_path.relative_to(HERE)) if control_path.is_relative_to(HERE)
                 else str(control_path),
        "image_kind": "control",
        "status": "ok" if valid else "parse_error",
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
        "tolerance_px": CONTROL_TOLERANCE_PX,
        "images_delivered": delivered,
        "tool_attempts": attempt["tool_attempts"],
        "usage": token_totals(attempt["usage"]),
        "wall_s": attempt["wall_s"],
        "returncode": attempt["returncode"],
        "response_text": attempt["last_message"],
    }


CONTROL_TOLERANCE_PX = 25.0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def summarise(df: pd.DataFrame) -> pd.DataFrame:
    scored = df[df["error_px"].notna()]
    rows = []
    # Cover the styles actually present, not just the active ones: a rebuilt
    # report of a pre-archiving run still carries archived style names.
    present = list(dict.fromkeys(df["style"]))
    ordered = ([s for s in prompts.DEFAULT_STYLE_ORDER if s in present]
               + [s for s in present if s not in prompts.DEFAULT_STYLE_ORDER])
    for style in ordered:
        sub = df[df["style"] == style]
        if sub.empty:
            continue
        hit = scored[scored["style"] == style]
        errors = hit["error_px"].astype(float)
        rows.append({
            "style": style,
            "runs": len(sub),
            "scored": len(hit),
            "parse_rate": _rate(sub, lambda s: s in (parse.OK, parse.OUT_OF_FRAME,
                                                     parse.NO_PICK_POINT)),
            "schema_valid_rate": _rate(sub, lambda s: s is True, column="schema_valid"),
            "out_of_frame_rate": _rate(sub, lambda s: s == parse.OUT_OF_FRAME),
            "empty_or_error_rate": _rate(sub, lambda s: s in parse.FAILURE_STATUSES),
            "mean_error_px": float(errors.mean()) if len(errors) else None,
            "median_error_px": float(errors.median()) if len(errors) else None,
            "p90_error_px": float(errors.quantile(0.9)) if len(errors) else None,
            "mean_error_pct_width": float(hit["error_pct_width"].mean()) if len(hit) else None,
            "pck@5": _rate(hit, lambda v: v is True, column="pck_5"),
            "pck@10": _rate(hit, lambda v: v is True, column="pck_10"),
            "pck@20": _rate(hit, lambda v: v is True, column="pck_20"),
            "mean_bbox_iou_rough": float(hit["bbox_iou_rough"].dropna().mean())
                                     if hit["bbox_iou_rough"].notna().any() else None,
            "bbox_contains_gt_rate": _rate(hit, lambda v: v is True, column="bbox_contains_gt"),
            "mean_total_tokens": float(sub["total_tokens"].mean()),
            "mean_wall_s": float(sub["wall_s"].mean()),
        })
    return pd.DataFrame(rows)


def _rate(df: pd.DataFrame, predicate, column: str | None = None) -> float | None:
    if df.empty:
        return None
    values = df[column] if column else df["status"]
    if column and values.isna().all():
        return None
    return float(sum(1 for v in values if predicate(v)) / len(values))


def error_histogram(df: pd.DataFrame, path: Path) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scored = df[df["error_px"].notna()]
    if scored.empty:
        return False
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=140)
    bins = np.arange(0, max(40, float(scored["error_px"].max()) + 10), 10)
    styles = list(dict.fromkeys(scored["style"]))
    ax.hist([scored[scored["style"] == s]["error_px"].astype(float) for s in styles],
            bins=bins, label=styles, stacked=True)
    for threshold, colour in zip(PCK_THRESHOLDS, ("#2ca02c", "#ff7f0e", "#d62728")):
        ax.axvline(threshold, color=colour, linestyle="--", linewidth=1)
        ax.text(threshold, ax.get_ylim()[1] * 0.94, f"{threshold}px", color=colour,
                ha="center", fontsize=8)
    ax.set_xlabel("picking-point error (px)")
    ax.set_ylabel("runs")
    ax.set_title("Picking-point error distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def fmt(value, digits=2, dash="–"):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return dash
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def pct(value) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"


def write_report(run_dir: Path, args, manifest, control, df, summary, histogram_ok,
                 prompts_used: dict, started_at: str, elapsed: float,
                 run_summary: dict) -> None:
    lines: list[str] = []
    add = lines.append
    delivered = control.get("images_delivered") if control else None
    # Scored runs and unlabelled runs are reported separately and never mixed:
    # an unlabelled frame has no ground truth, so it cannot contribute to any
    # error, PCK, IoU or sensitivity number.
    gt = df[df["has_gt"]] if "has_gt" in df.columns else df
    ng = df[~df["has_gt"]] if "has_gt" in df.columns else df.iloc[0:0]
    n_runs = len(gt)
    scored = gt[gt["error_px"].notna()]
    overall_median = float(scored["error_px"].median()) if len(scored) else None
    overall_pck10 = _rate(scored, lambda v: v is True, column="pck_10")

    add("# VLM raw-capability eval — strawberry detection + picking point")
    add("")
    add(f"- Started: `{started_at}`  ·  elapsed `{elapsed:.0f}s`")
    harness = run_summary.get("harness") or harness_info()
    add(f"- **Harness:** `{harness['name']}` v{harness['version']}  ·  "
        f"fingerprint `{harness['fingerprint']}`  ·  `{harness['path']}`")
    if harness.get("rebuilt_with"):
        add(f"  - ⚠ artefacts rebuilt with harness fingerprint "
            f"`{harness['rebuilt_with']}`; the runs themselves were produced by "
            f"`{harness['fingerprint']}`")
    add(f"- **Model:** `{run_summary['model'] or 'unknown'}` (provider "
        f"`{args.provider}`)")
    add(f"- **Reasoning effort:** `{run_summary['effort'] or 'config default'}`")
    add(f"- Scored runs: **{n_runs}** across {gt['sample_id'].nunique()} ground-truthed "
        f"frames (5 styles each)")
    if len(ng):
        add(f"- Unlabelled runs: **{len(ng)}** across {ng['sample_id'].nunique()} "
            f"multi-strawberry frames that have **no ground truth** — these are shown "
            f"qualitatively only and contribute to no metric below")
    add(f"- Ground truth: `target_ref.compute_episode_target` "
        f"(gripper tip at last trajectory frame → initial image), recomputed at "
        f"manifest build time")
    add(f"- No detector, tracker or segmenter is used anywhere in the loop.")
    add("")

    # ---- verdict banner ---------------------------------------------------
    add("## Verdict")
    add("")
    if delivered is True:
        add(f"- **Vision delivery: YES.** Control image read correctly — code "
            f"`{control['prediction'].get('code')}` (expected "
            f"`{control['truth']['code']}`), red-circle error "
            f"{control['circle_error_px']:.1f}px (tolerance "
            f"{control['tolerance_px']:.0f}px). "
            + ("The capability numbers below are valid." if len(scored)
               else "These qualitative results are valid."))
        if len(scored):
            add(f"- **Picking point:** median error "
                f"**{fmt(overall_median)} px** across {len(scored)} scored runs "
                f"({fmt(100 * overall_median / manifest['samples'][0]['image_shape_hw'][1] if overall_median else None, 1)}% of frame "
                f"width); PCK@10 = **{pct(overall_pck10)}**.")
        else:
            add("- **No scored runs.** This batch contains no ground-truthed frames, so "
                "there is no picking-point error to report. Every result below is "
                "**qualitative**: what the model found and what it said about it, "
                "checked by eye, not by a number. Switch the scored sources back on "
                "with `--sources validation occluded shunba`.")
    elif delivered is False:
        add("- **Vision delivery: NO.** The synthetic control could not be read, so the "
            "CLI is not delivering images to this model and the numbers below are "
            "**INVALID as a measure of visual capability** — they only reflect a blind "
            "model guessing.")
        add(f"  Control response: `{control.get('response_text','')[:200]}`")
    else:
        add("- **Vision delivery: UNVERIFIED** (control skipped or errored). Treat every "
            "number below as unconfirmed until the control passes.")
    tool_attempts = sorted({t for a in df["tool_attempts"] for t in (a or [])})
    add(f"- Tool use: {'none attempted' if not tool_attempts else 'ATTEMPTS: ' + ', '.join(tool_attempts)} "
        f"(all tool surfaces are disabled for these runs, so a peek at the frame on disk "
        f"is not possible).")
    add("")

    if control:
        add("### Vision-delivery control")
        add("")
        add("| field | expected | predicted | ok |")
        add("| --- | --- | --- | --- |")
        pred = control.get("prediction") or {}
        add(f"| code | `{control['truth']['code']}` | `{pred.get('code')}` | "
            f"{fmt(control['code_ok'])} |")
        add(f"| red circle | {control['truth']['red_circle']} | {pred.get('red_circle')} | "
            f"{fmt(control['circle_ok'])} (err {fmt(control['circle_error_px'],1)}px) |")
        add(f"| green square | {control['truth']['green_square']} | "
            f"{pred.get('green_square')} | {fmt(control['green_square_ok'])} "
            f"(err {fmt(control['green_square_error_px'],1)}px) |")
        add("")

    # ---- style table ------------------------------------------------------
    if len(gt) == 0:
        add("## Scored metrics")
        add("")
        add("_Not applicable: this batch has no ground-truthed frames. The scored "
            "metrics (picking-point error, PCK, bbox IoU, sensitivity) need a frame "
            "with a known picking point, which the unlabelled multi-strawberry source "
            "does not have. See the qualitative sections below._")
        add("")
    if len(gt):
        add("## Metrics by prompt style")
        add("")
        add("| style | runs | scored | parse | schema-valid | out-of-frame | "
            "median err px | mean err px | P90 px | err % width | PCK@5 | PCK@10 | PCK@20 | "
            "bbox IoU (rough) | mean tok | mean s |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for _, row in summary.iterrows():
            add(f"| `{row['style']}` | {row['runs']} | {row['scored']} | {pct(row['parse_rate'])} | "
                f"{pct(row['schema_valid_rate'])} | {pct(row['out_of_frame_rate'])} | "
                f"{fmt(row['median_error_px'],1)} | {fmt(row['mean_error_px'],1)} | "
                f"{fmt(row['p90_error_px'],1)} | {fmt(row['mean_error_pct_width'],1)} | "
                f"{pct(row['pck@5'])} | {pct(row['pck@10'])} | {pct(row['pck@20'])} | "
                f"{fmt(row['mean_bbox_iou_rough'])} | {fmt(row['mean_total_tokens'],0)} | "
                f"{fmt(row['mean_wall_s'],1)} |")
        add("")
        add("PCK@k = share of runs whose picking point lands within k pixels of ground truth. "
            "bbox IoU is **approximate**: it compares the prediction against the upstream "
            "rough 60×60 box centred 30 px below the picking point, because this dataset has "
            "no hand-annotated fruit boxes.")
        add("")

    # ---- per-image table --------------------------------------------------
    if len(gt):
        add("## Metrics by frame")
        add("")
        add("| frame | episode | GT (u, v) | GT z (m) | mean err px | best style | worst style | "
            "PCK@10 |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for sample_id, group in gt.groupby("sample_id", sort=False):
            sub = group[group["error_px"].notna()]
            first = group.iloc[0]
            if len(sub):
                by_style = sub.groupby("style")["error_px"].mean().sort_values()
                best, worst = by_style.index[0], by_style.index[-1]
                mean_err = fmt(sub["error_px"].mean(), 1)
                best_s, worst_s = f"`{best}` ({fmt(by_style.iloc[0],1)}px)", f"`{worst}` ({fmt(by_style.iloc[-1],1)}px)"
            else:
                mean_err, best_s, worst_s = "n/a", "–", "–"
            add(f"| `{sample_id}` | {first['episode']} | "
                f"({fmt(first['gt_uv'][0],1)}, {fmt(first['gt_uv'][1],1)}) | "
                f"{fmt(first['gt_z_m'],3)} | {mean_err} | {best_s} | {worst_s} | "
                f"{pct(_rate(sub, lambda v: v is True, column='pck_10'))} |")
        add("")

    # ---- per-run table ----------------------------------------------------
    if len(gt):
        add("## Every scored run")
        add("")
        add("| run | style | frame | status | pred (u, v) | GT (u, v) | err px | dx | dy | "
            "attempts | tok | s | overlay |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for _, row in gt.iterrows():
            pred = row["picking_point"]
            pred_s = f"({pred[0]:.0f}, {pred[1]:.0f})" if pred is not None else "–"
            gtuv = row["gt_uv"]
            attempts = f"{row['attempts']}" + (" (fallback)" if row["fallback_used"] else "")
            add(f"| `{row['run_id']}` | `{row['style']}` | `{row['sample_id']}` | "
                f"{row['status']} | {pred_s} | ({gtuv[0]:.1f}, {gtuv[1]:.1f}) | "
                f"{fmt(row['error_px'],1)} | {fmt(row['dx_px'],1)} | {fmt(row['dy_px'],1)} | "
                f"{attempts} | {int(row['total_tokens'])} | {fmt(row['wall_s'],1)} | "
                f"[png]({row['overlay']}) |")
        add("")

    # ---- tokens -----------------------------------------------------------
    add("## Tokens and time")
    add("")
    add("| style | runs | input tok | cached | output tok | reasoning tok | total tok | "
        "mean tok/run | mean wall s | cost |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for style in prompts_used:
        sub = df[df["style"] == style]
        if sub.empty:
            continue
        costs = sub["cost_usd"].dropna()
        cost = f"${costs.sum():.4f}" if len(costs) else "n/a"
        add(f"| `{style}` | {len(sub)} | {int(sub['input_tokens'].sum())} | "
            f"{int(sub['cached_input_tokens'].sum())} | {int(sub['output_tokens'].sum())} | "
            f"{int(sub['reasoning_output_tokens'].sum())} | "
            f"{int(sub['total_tokens'].sum())} | {fmt(sub['total_tokens'].mean(),0)} | "
            f"{fmt(sub['wall_s'].mean(),1)} | {cost} |")
    add("")
    if args.price_in_per_mtok is None or args.price_out_per_mtok is None:
        add("_Cost is reported only when `--price-in-per-mtok` and `--price-out-per-mtok` "
            "are supplied; no pricing is assumed for this model._")
        add("")

    # ---- histogram --------------------------------------------------------
    if histogram_ok:
        add("## Error distribution")
        add("")
        add("![picking-point error histogram](error_histogram.png)")
        add("")

    # ---- overlays ---------------------------------------------------------
    # The scored overlay legend only makes sense when there are ground-truthed
    # frames; an unlabelled-only batch draws inventory/list overlays instead.
    if len(gt):
        add("## Overlays (scored frames)")
        add("")
        add("How to read one overlay:")
        add("")
        add("- **Cyan circle** — *not* the answer and *not* a measurement. It is the upstream "
            "rough 60×60 fruit box, centred 30 px below the picking point so that its top "
            "edge passes exactly through it. It exists because the dataset has no "
            "hand-annotated fruit boxes; it is the reference the approximate bbox-IoU "
            "metric scores against, and it is never shown to the model.")
        add("- **Cyan crosshair** — the ground truth itself: the gripper-tip pose at the "
            "episode's last trajectory frame, projected into this frame. This point is what "
            "every error number in this report is measured from.")
        add("- **Magenta ring + dot** — the model's predicted picking point, with the error "
            "in pixels written beside it.")
        add("- **Amber rectangle** — the model's predicted bounding box.")
        add("- **White line** — the error vector between the two points.")
        add("")
        add("Ground truth is drawn last, so a prediction that lands almost on the truth can "
            "never hide the reference.")
        add("")
        for style in prompts_used:
            sheet = Path("contact_sheets") / f"{style}.png"
            if (run_dir / sheet).exists():
                add(f"### `{style}`")
                add("")
                add(f"[full-size contact sheet]({sheet})")
                add("")
                add(f"![{style} contact sheet]({sheet})")
                add("")

    # ---- full inventory (unbiased) ---------------------------------------
    inv = df[df["output_shape"] == "inventory"] if "output_shape" in df.columns \
        else df.iloc[0:0]
    if len(inv):
        add("## Full strawberry inventory (unbiased)")
        add("")
        add("These prompts are told to report **every strawberry they can see — "
            "whatever its colour — including partly occluded ones**, and are "
            "explicitly forbidden from filtering by ripeness. There are **no ripeness "
            "categories**: each fruit carries a continuous 0–100 redness value and a "
            "continuous 0–100 occlusion value, plus calyx/peduncle visibility, "
            "graspability and a free-text description in the model's own words. The "
            "model separately nominates one pick target, which keeps that decision "
            "distinct and scoreable.")
        add("")
        add("| prompt | frame | scored? | found | occluded >0% (≥50%) | redness mean / max "
            "| occlusion mean | peduncle visible | graspable | mean conf | target "
            "(redness) | target err px |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for _, row in inv.sort_values(["style", "sample_id"]).iterrows():
            target = row["target_index"]
            if target is None or pd.isna(target) or target < 0:
                target_s = "none"
            elif row["target_valid"] is False:
                target_s = f"invalid (#{int(target)})"
            else:
                target_s = f"#{int(target)}"
            target_red = None
            if isinstance(target, (int, float)) and not pd.isna(target) and int(target) >= 0:
                entries = row["inventory"] or []
                if int(target) < len(entries):
                    target_red = entries[int(target)].get("redness_pct")
            add(f"| `{row['style']}` | `{row['sample_id']}` | "
                f"{'yes' if row['target_scored'] else 'no'} | "
                f"{fmt(row['n_inventory'], 0)} | "
                f"{fmt(row['n_occluded'], 0)} ({fmt(row['n_mostly_hidden'], 0)}) | "
                f"{fmt(row['mean_redness_pct'], 0)}% / {fmt(row['max_redness_pct'], 0)}% | "
                f"{fmt(row['mean_occlusion_pct'], 0)}% | "
                f"{fmt(row['n_peduncle_visible'], 0)}/{fmt(row['n_inventory'], 0)} | "
                f"{fmt(row['n_graspable'], 0)}/{fmt(row['n_inventory'], 0)} | "
                f"{fmt(row['mean_confidence_pct'], 0)}% | "
                f"{target_s}" + (f" ({fmt(target_red, 0)}%)" if target_red is not None else "")
                + " | "
                f"{fmt(row['error_px'], 1)} |")
        add("")
        scored_inv = inv[inv["target_scored"]]
        if len(scored_inv):
            add(f"On the {len(scored_inv)} inventoried frame(s) that have ground truth, "
                f"the nominated target's picking point has median error "
                f"**{fmt(scored_inv['error_px'].median(), 1)} px** "
                f"(mean {fmt(scored_inv['error_px'].mean(), 1)} px). So the unbiased "
                f"prompt is directly comparable with the single-target styles above, "
                f"on the same frames.")
            add("")
        add("Overlay legend: box colour is the **continuous redness ramp** (green at "
            "0% → amber at 50% → red at 100%), box thickness decreases with reported "
            "occlusion, and the nominated target is drawn in cyan. Fruit the model did "
            "**not** find are simply absent — the boxes cover only what it reported.")
        add("")
        for style in list(dict.fromkeys(inv["style"])):
            for sheet, label in ((Path("contact_sheets") / f"{style}.png",
                                  "frames with ground truth (cyan = nominated target, "
                                  "cyan crosshair = ground truth)"),
                                 (Path("contact_sheets") / f"{style}__unlabelled.png",
                                  "unlabelled multi-strawberry frames")):
                if not (run_dir / sheet).exists():
                    continue
                add(f"### `{style}` — {label}")
                add("")
                add(f"[full-size contact sheet]({sheet})")
                add("")
                add(f"![{style} contact sheet]({sheet})")
                add("")
        add("### Per-fruit detail")
        add("")
        for _, row in inv.sort_values(["style", "sample_id"]).iterrows():
            entries = row["inventory"] or []
            if not entries:
                continue
            add(f"- `{row['run_id']}` — [overlay]({row['overlay']})")
            for i, berry in enumerate(entries):
                mark = " **← target**" if row["target_index"] == i else ""
                point = berry.get("picking_point")
                point_s = f"({int(point[0])}, {int(point[1])})" if point else "null"
                add(f"  - #{i}: redness {berry.get('redness_pct')}%, occluded "
                    f"{berry.get('occlusion_pct')}%, calyx "
                    f"{'visible' if berry.get('calyx_visible') else 'hidden'}, "
                    f"peduncle {'visible' if berry.get('peduncle_visible') else 'hidden'}, "
                    f"graspable {'yes' if berry.get('graspable') else 'no'}, "
                    f"conf {berry.get('confidence_pct')}%, pick {point_s}{mark}"
                    + (f"\n    - _{berry['description'].strip()}_"
                       if isinstance(berry.get("description"), str)
                       and berry["description"].strip() else ""))
        add("")

    # ---- qualitative (no ground truth) -----------------------------------
    # Inventory prompts get their own section above; this one covers the flat
    # list / single-target answers on the unlabelled frames.
    ng_simple = (ng[ng["output_shape"] != "inventory"]
                 if "output_shape" in ng.columns else ng)
    if len(ng_simple):
        add("## Qualitative results — unlabelled multi-strawberry scenes")
        add("")
        add("These frames come from a different source and have **no ground truth**: "
            "there is no demonstration, no trajectory and no projected picking point to "
            "compare against, so **no error, PCK, IoU or sensitivity number is computed "
            "for them**. They are here so the model's behaviour on a cluttered, "
            "multi-fruit scene can be inspected. Unlike the scored frames, each of these "
            "contains roughly 4–13 ripe fruit, and the prompter asks for the ripiest one "
            "(`single_basic`) or for every ripe fruit "
            "(`list_ripe_only`).")
        add("")
        add(f"| frame | source file | image | reported | ripe | points out of frame | "
            f"status |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        for _, row in ng_simple.iterrows():
            sample = next((s for s in manifest["samples"]
                           if s["sample_id"] == row["sample_id"]), {})
            shape = sample.get("image_shape_hw") or [None, None]
            reported = row["n_strawberries"]
            reported_s = "1 (single-target)" if row["style"] == "single_basic" \
                else (str(int(reported)) if reported is not None and not pd.isna(reported) else "–")
            add(f"| `{row['sample_id']}` | `{sample.get('episode', '?')}` | "
                f"{shape[1]}×{shape[0]} | `{row['style']}` → {reported_s} | "
                f"{fmt(row['n_ripe_reported'], 0)} | {fmt(row['n_points_out_of_frame'], 0)} | "
                f"{row['status']} |")
        add("")
        add("Per-frame overlays (magenta ring = a predicted picking point, amber box = a "
            "predicted fruit; nothing here is ground truth):")
        add("")
        for style in [s for s in prompts_used
                      if (run_dir / "contact_sheets" / f"{s}__unlabelled.png").exists()]:
            sheet = Path("contact_sheets") / f"{style}__unlabelled.png"
            add(f"### `{style}`")
            add("")
            add(f"[full-size contact sheet]({sheet})")
            add("")
            add(f"![{style} unlabelled contact sheet]({sheet})")
            add("")
        add("### Per-frame detail")
        add("")
        for _, row in ng_simple.iterrows():
            count = row["n_strawberries"]
            count_s = "unparsed" if count is None or pd.isna(count) else f"{int(count)} reported"
            add(f"- `{row['run_id']}` — [overlay]({row['overlay']}) — {count_s}: "
                + (", ".join(
                    f"({int(b['picking_point'][0])}, {int(b['picking_point'][1])})"
                    for b in (row["strawberries"] or []) if b.get("picking_point"))
                   or "none"))
        add("")

    # ---- sensitivity ------------------------------------------------------
    add("## Sensitivity bridge: what a pixel error means for picking")
    add("")
    sens = manifest["sensitivity_bridge"]
    add(f"Fitted from `{Path(sens['source']).name}` — a measured dose-response sweep "
        f"(replaying the policy with perturbed target points) around "
        f"{sens['reference_original_uv'][0]:.0f}, {sens['reference_original_uv'][1]:.0f}.")
    add("")
    add("| direction | gripper-endpoint rate | grip-closure rate |")
    add("| --- | --- | --- |")
    add(f"| x (px) | {np.linalg.norm(sens['d_endpoint_dx_mm_per_px']):.3f} mm/px | "
        f"{sens['d_grip_dx_per_px']:+.5f} /px |")
    add(f"| y (px) | {np.linalg.norm(sens['d_endpoint_dy_mm_per_px']):.3f} mm/px | "
        f"{sens['d_grip_dy_per_px']:+.5f} /px |")
    add("")
    if len(scored):
        for label, dxv, dyv in (
            ("median error magnitude", float(scored["dx_px"].abs().median()),
             float(scored["dy_px"].abs().median())),
            ("mean absolute error", float(scored["dx_px"].abs().mean()),
             float(scored["dy_px"].abs().mean())),
        ):
            effect = gtbridge.sensitivity_effect(sens, dxv, dyv)
            add(f"- {label}: |dx| {dxv:.1f}px, |dy| {dyv:.1f}px → endpoint shift "
                f"≈ **{effect['endpoint_shift_magnitude_mm']:.1f} mm**, grip change "
                f"≈ {effect['grip_delta']:+.3f}.")
    else:
        add("_Not applied: this batch has no scored picking points to translate. The "
            "per-pixel rates above are still the right conversion factor once "
            "ground-truthed frames are included again._")
    add("")
    add(f"_{sens['note']}_")
    add("")

    # ---- failures ---------------------------------------------------------
    failures = df[df["status"].isin(parse.FAILURE_STATUSES) |
                  (df["status"].isin((parse.OUT_OF_FRAME, parse.NO_PICK_POINT)))]
    add("## Non-scoring runs")
    add("")
    if len(failures) == 0:
        add("None: every run produced a parseable, in-frame answer.")
    else:
        add("| run | status | detail | response excerpt |")
        add("| --- | --- | --- | --- |")
        for _, row in failures.iterrows():
            detail = row["schema_error"] or row["fallback_reason"] or ""
            excerpt = (row["response_excerpt"] or "").replace("|", "\\|").replace("\n", " ")[:160]
            add(f"| `{row['run_id']}` | {row['status']} | {detail} | {excerpt} |")
    add("")

    # ---- prompts ----------------------------------------------------------
    add("## Exact prompts used")
    add("")
    for style, text in prompts_used.items():
        spec = prompts.STYLES[style]
        add(f"### `{style}`")
        add("")
        add(f"_{spec.summary}_ · input: `{spec.image_kind}` · "
            f"`--output-schema`: {'yes' if spec.use_output_schema else 'no'}")
        add("")
        add("```text")
        add(text)
        add("```")
        add("")
    add("### synthetic control")
    add("")
    add("```text")
    add(f"{prompts.CONTROL_PROMPT}\n\n{TOOL_NOTICE}")
    add("```")
    add("")

    # ---- method -----------------------------------------------------------
    add("## Method and provenance")
    add("")
    add(f"- Manifest: `manifest.json` (snapshot in this directory). Ground truth "
        f"recomputed from `{manifest['ground_truth']['module']}`.")
    add(f"- Selection: {manifest['selection']['method']}; per source "
        + "; ".join(f"{name}: {prov['indices']} of {prov['n_gt_valid']} GT-valid"
                    for name, prov in manifest["selection"]["per_source"].items()) + ".")
    add(f"- Query image per episode: `{manifest['frame_policy']}`.")
    add(f"- The pre-rendered target marker image is never an input; it would leak the "
        f"answer. It is used only to sanity-check ground truth.")
    provider = run_summary.get("provider") or args.provider
    if provider == "claude":
        add("- Every model call runs `claude -p` with `--tools \"\"` — no tool exists at "
            "the API level, so the model cannot open the frame or the episode's "
            "trajectory files — plus `--safe-mode` (no hooks, plugins, skills or "
            "CLAUDE.md), `--strict-mcp-config` (no MCP servers) and "
            "`--no-session-persistence`, inside the empty `vlm_eval/agent_cwd/` "
            "directory. The event stream is scanned for tool calls and permission "
            "denials; any attempt fails acceptance.")
    else:
        add(f"- Every model call runs with all tool surfaces disabled "
            f"(`{'`, `'.join(DISABLED_FEATURES)}`) and `--sandbox read-only`, so the model "
            f"cannot open the frame or the episode's trajectory files from disk.")
    catalog = run_summary.get("catalog")
    if catalog:
        if provider == "claude":
            add(f"- Image delivery: each frame is handed to `claude -p` as a base64 "
                f"image content block via `--input-format stream-json`, routed to "
                f"`{catalog.get('routing_base_url')}` as model "
                f"`{catalog.get('model_slug')}` (claude CLI "
                f"{catalog.get('cli_version', 'unknown')}). No codex catalog override "
                f"applies on this path; delivery is decided server-side by the slug, "
                f"and the synthetic control that opens this report is the gate.")
        else:
            add(f"- Image delivery: `codex exec` was given "
                f"`-c model_catalog_json={catalog['path']}` "
                f"(sha256 `{catalog['sha256'][:12]}…`), which declares "
                f"`{catalog['model_slug']}` with input modalities "
                f"`{catalog['input_modalities']}`. The user-level catalog declares that slug "
                f"text-only, and `codex exec -i` silently drops images in that case, so the "
                f"override is what makes the model see the frame at all.")
    else:
        add("- Image delivery: catalog provenance not recorded for this run.")
    cli_name = run_summary.get("cli_name") or ("claude" if provider == "claude" else "codex")
    add(f"- Environment: python {platform.python_version()}, {cli_name} CLI "
        f"`{run_summary.get('cli_version', 'unknown')}`.")
    add(f"- Raw responses, per-run records and usage: `responses.jsonl`; flat table: "
        f"`metrics.csv`.")
    add("")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def dry_run(args, manifest, styles, model: str) -> None:
    print(f"manifest : {args.manifest}")
    print(f"catalog  : {args.catalog}")
    print(f"model    : {model}")
    print(f"out      : {args.out}")
    print(f"styles   : {', '.join(s.name for s in styles)}")
    samples = select_samples(args, manifest)
    print(f"samples  : {len(samples)}")
    plan = [(s, sample) for s in styles for sample in samples if style_applies(s, sample)]
    print(f"calls    : {len(plan)} model calls "
          f"({sum(1 for s in samples if s.get('has_gt'))} scored frames, "
          f"{sum(1 for s in samples if not s.get('has_gt'))} unlabelled)")
    for style in styles:
        gt_n = sum(1 for s in samples if s.get("has_gt") and style_applies(style, s))
        ng_n = sum(1 for s in samples if not s.get("has_gt") and style_applies(style, s))
        print(f"           {style.name:<24} {gt_n} scored + {ng_n} unlabelled")
    print()
    for style in styles:
        sample = next((s for s in samples if style_applies(style, s)), None)
        if sample is None:
            continue
        image, origin = prepare_image(style, sample)
        h, w = sample["image_shape_hw"]
        prompt = style.build_prompt(frame_w=w, frame_h=h,
                                    query_origin=sample.get("exemplar_query_origin") or (0, 0),
                                    multi_fruit=not sample.get("has_gt"))
        print("=" * 78)
        print(f"style `{style.name}`  sample={sample['sample_id']}  "
              f"image={image.relative_to(HERE)}  "
              f"panel_origin={origin}  output_schema={style.use_output_schema}")
        print("-" * 78)
        print(prompt)
        print()
        print(TOOL_NOTICE)
        print()


def codex_version() -> str:
    try:
        version = subprocess.run(["codex", "--version"], capture_output=True, text=True,
                                 timeout=30)
        return (version.stdout.strip() or version.stderr.strip()) or "unknown"
    except Exception:
        return "unknown"


def write_artifacts(run_dir: Path, args, manifest, control, records: list[dict],
                    styles, samples, started: str, elapsed: float,
                    run_summary: dict) -> None:
    """Write every derived artefact for a set of run records.

    Split out of ``main`` so ``--rebuild-report`` can refresh the tables,
    overlays and report from an existing ``responses.jsonl`` without spending
    another round of model calls.
    """
    # Re-render every overlay from the records, so a rebuilt report can never
    # disagree with the numbers it prints (and so a rendering fix applies to
    # old runs without new model calls).
    # Anything already in these directories is derived, so clear it first rather
    # than leaving orphans behind when the run's shape changes.
    for stale in ("overlays", "contact_sheets"):
        directory = run_dir / stale
        if directory.is_dir():
            for existing in directory.iterdir():
                if existing.is_file():
                    existing.unlink()

    raw_by_sample = {s["sample_id"]: s["images"]["raw"] for s in manifest["samples"]}
    shape_by_sample = {s["sample_id"]: s.get("image_shape_hw") for s in manifest["samples"]}

    # Re-derive the unlabelled summary counts from the stored answers, so a report
    # can never disagree with the responses and a counting fix applies to old runs
    # without new model calls. 0 ("found none") stays distinct from None ("unparsed").
    for record in records:
        # A parsed inventory with no usable grasp point is a distinct outcome from
        # a normal success. This applies on scored frames too, so re-derive it
        # before the unlabelled-only handling below.
        if record.get("output_shape") == "inventory" and record.get("status") == parse.OK:
            entries = record.get("inventory") or []
            index = record.get("target_index")
            if not (isinstance(index, int) and 0 <= index < len(entries)
                    and entries[index].get("picking_point")):
                record["status"] = parse.NO_PICK_POINT

        if record.get("has_gt", True):
            continue

        berries = record.get("strawberries") if record.get("output_shape") == "list" \
            else record.get("inventory")
        if berries is None and record.get("output_shape") == "single":
            record["n_strawberries"] = (
                1 if record.get("status") == parse.OK
                else 0 if record.get("status") == parse.NO_PICK_POINT else None)
            continue
        if berries is None:
            continue
        record["n_strawberries"] = len(berries)
        record["n_ripe_reported"] = sum(1 for b in berries if b.get("ripe") is True)
        shape = shape_by_sample.get(record["sample_id"]) or [None, None]
        h, w = shape
        if w and h:
            record["n_points_out_of_frame"] = sum(
                1 for b in berries
                if b.get("picking_point") is not None
                and not (0 <= b["picking_point"][0] < w and 0 <= b["picking_point"][1] < h)
            )

    for record in records:
        raw_rel = raw_by_sample.get(record["sample_id"])
        if raw_rel is None:
            continue
        raw = imaging.load_rgb(HERE / raw_rel)
        title = (f"{record['style']} | {record['sample_id']} | {record['episode']} | "
                 f"{record['status']}")
        if record.get("has_gt"):
            overlay = imaging.draw_prediction_overlay(
                raw, tuple(record["gt_uv"]), record.get("picking_point"),
                record.get("bbox"), title)
        elif record.get("output_shape") == "inventory":
            overlay = imaging.draw_inventory_overlay(
                raw, record.get("inventory"), record.get("target_index"), title)
        else:
            overlay = imaging.draw_multi_overlay(
                raw, record.get("strawberries"), title)
        rel = Path("overlays") / f"{record['run_id']}.png"
        imaging.save_rgb(overlay, run_dir / rel)
        record["overlay"] = str(rel)

    with (run_dir / "responses.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    df = pd.DataFrame(records)
    flat = df.drop(columns=["prompt", "response_text", "attempts_detail", "events"],
                   errors="ignore")
    for column in ("picking_point", "gt_uv", "gt_uv_rounded", "bbox", "tool_attempts",
                   "returncodes", "cli_errors", "sensitivity", "panel_origin",
                   "picking_point_global", "bbox_global"):
        if column in flat.columns:
            flat[column] = flat[column].apply(
                lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)
    flat.to_csv(run_dir / "metrics.csv", index=False)

    # Style summary covers the scored runs only; unlabelled runs have no metric.
    summary_df = summarise(df[df["has_gt"]] if "has_gt" in df.columns else df)
    summary_df.to_csv(run_dir / "summary_by_style.csv", index=False)

    style_names = [s.name if hasattr(s, "name") else s for s in styles]
    style_objs = [s if hasattr(s, "name") else prompts.style_by_name(s) for s in styles]

    has_gt_col = df["has_gt"] if "has_gt" in df.columns else pd.Series(True, index=df.index)

    for style in style_names:
        subset = df[(df["style"] == style) & df["overlay"].notna() & has_gt_col]
        if subset.empty:
            continue
        images = [imaging.load_rgb(run_dir / rel) for rel in subset["overlay"]]
        sheet = imaging.contact_sheet(images, list(subset["sample_id"]),
                                      cols=min(3, len(images)))
        imaging.save_rgb(sheet, run_dir / "contact_sheets" / f"{style}.png")

    # Unlabelled runs get their own sheets: they are much larger frames and carry
    # no reference markers, so mixing them with the scored overlays would be
    # misleading.
    for style in style_names:
        subset = df[(df["style"] == style) & df["overlay"].notna() & ~has_gt_col]
        if subset.empty:
            continue
        images = [imaging.load_rgb(run_dir / rel) for rel in subset["overlay"]]
        sheet = imaging.contact_sheet(images, list(subset["sample_id"]),
                                      cols=min(2, len(images)))
        imaging.save_rgb(sheet, run_dir / "contact_sheets" / f"{style}__unlabelled.png")

    histogram_ok = error_histogram(df[has_gt_col] if "has_gt" in df.columns else df,
                                   run_dir / "error_histogram.png")

    prompts_used = {}
    for style in style_objs:
        # Show the prompt for a frame the style actually runs on, so the report
        # never prints the multi-fruit wording for a single-target style (or vice
        # versa).
        sample = next((s for s in samples if style_applies(style, s)), samples[0])
        frame_h, frame_w = sample["image_shape_hw"]
        prompts_used[style.name] = (
            style.build_prompt(frame_w=frame_w, frame_h=frame_h,
                               query_origin=sample.get("exemplar_query_origin") or (0, 0),
                               multi_fruit=not sample.get("has_gt"))
            + f"\n\n{TOOL_NOTICE}"
        )

    write_report(run_dir, args, manifest, control, df, summary_df, histogram_ok,
                 prompts_used, started, elapsed, run_summary)


def rebuild_report(args) -> None:
    """Refresh a finished run's artefacts from its own records."""
    run_dir = args.out
    records_path = run_dir / "responses.jsonl"
    if not records_path.exists():
        raise SystemExit(f"no responses.jsonl in {run_dir}; --rebuild-report needs a "
                         "finished run directory (pass it via --out)")
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    os.environ.setdefault("MPLCONFIGDIR", str(run_dir / ".mpl"))
    manifest = json.loads((run_dir / "manifest.json").read_text())
    control = (json.loads((run_dir / "control.json").read_text())
               if (run_dir / "control.json").exists() else None)
    style_names = list(dict.fromkeys(r["style"] for r in records))
    styles = [prompts.style_by_name(name) for name in style_names]
    # Records from before the style rename carry the old names; normalise them to
    # the current name so a rebuilt report groups with current conventions (the
    # stored records themselves are rewritten only in memory, never on disk).
    for record in records:
        record["style"] = prompts.style_by_name(record["style"]).name
    first = records[0]
    started = f"{run_dir.name} (rebuilt)"
    elapsed = sum(r.get("wall_s") or 0.0 for r in records)
    catalog = (json.loads((run_dir / "catalog.json").read_text())
               if (run_dir / "catalog.json").exists() else None)
    # Prefer the identity recorded with the runs; fall back to the current code.
    harness = ({key: first.get(field) for key, field in
                (("name", "harness"), ("version", "harness_version"),
                 ("fingerprint", "harness_fingerprint"))}
               if first.get("harness") else None)
    current = harness_info()
    if harness and harness.get("fingerprint") and harness["fingerprint"] != current["fingerprint"]:
        # The runs were produced by different code than the code doing the
        # rebuilding. Say so, rather than stamping the report with an identity
        # that did not generate its numbers.
        harness = {**harness, "path": str(HERE), "rebuilt_with": current["fingerprint"]}
        print(f"note: these runs were produced by harness fingerprint "
              f"{harness['fingerprint']}; this rebuild used {current['fingerprint']}")
    else:
        harness = harness or current
        harness.setdefault("path", str(HERE))
    write_artifacts(run_dir, args, manifest, control, records, styles,
                    manifest["samples"], started, elapsed,
                    {"model": first.get("model"), "effort": first.get("reasoning_effort"),
                     "provider": first.get("provider") or "codex",
                     **cli_identity(first.get("provider") or "codex"),
                     "catalog": catalog, "harness": harness})
    print(f"rebuilt artefacts in {run_dir}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.list_styles:
        print("Available prompts (select with --styles <name|title> ...):\n")
        print(prompts.format_style_table())
        print("\n'runs on': gt = only frames with ground truth, "
              "no_gt = only the unlabelled multi-strawberry scenes, any = both.")
        return

    if args.rebuild_report:
        rebuild_report(args)
        return

    manifest = json.loads(args.manifest.read_text())

    cfg = read_user_config()
    if args.provider == "claude":
        # The claude-code path has no codex config or catalog.  Deliberately NOT
        # defaulted from ANTHROPIC_MODEL: that env names this installation's agent
        # model (here the text-only glm-5.3 flagship, which cannot receive images);
        # glm-5.3-flash is the multimodal GLM-5.3 and the sensible default.
        model = args.claude_model or args.model or "glm-5.3-flash"
        effort = args.reasoning_effort
        if re.split(r"[\[]", model)[0].strip() == "glm-5.3":
            print("warning: the flagship glm-5.3 slug rejects image content on this key "
                  "(native API: messages.content.type restricted to ['text']), so the "
                  "vision control will fail and the batch will be reported INVALID. "
                  "glm-5.3-flash is the multimodal GLM-5.3.", file=sys.stderr)
        catalog_info = claude_catalog_info(args, model)
    else:
        model = args.model or args.glm_model or cfg.get("model") or "deepseek-v4-flash"
        effort = args.reasoning_effort or cfg.get("model_reasoning_effort")
        if args.provider == "glm" and args.glm_model:
            model = args.glm_model
        catalog_info = catalog_entry(args, model)

    # Prompts are selected by name or by title; no selection means "all of them".
    styles = prompts.resolve_styles(args.styles)
    archived = [s.name for s in styles if s.name in prompts.ARCHIVED_STYLES]
    if archived:
        print(f"note: archived prompt(s) selected explicitly: {', '.join(archived)} "
              f"(not part of the default pipeline)")

    samples = select_samples(args, manifest)
    if not samples:
        raise SystemExit("no samples selected")

    if args.dry_run:
        dry_run(args, manifest, styles, model)
        return

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    cli_name = "claude" if args.provider == "claude" else "codex"
    run_dir = args.out / run_directory_name(stamp, model, effort, args.tag, cli=cli_name)
    run_dir.mkdir(parents=True, exist_ok=False)
    tmp = run_dir / ".tmp"
    tmp.mkdir()
    agent_cwd = HERE / "agent_cwd"
    agent_cwd.mkdir(exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(tmp / "mpl"))

    shutil.copyfile(args.manifest, run_dir / "manifest.json")
    (run_dir / "catalog.json").write_text(json.dumps(catalog_info, indent=2) + "\n")

    run_ctx = {
        "run_dir": run_dir,
        "tmp": tmp,
        "agent_cwd": agent_cwd,
        "model": model,
        "effort": effort,
    }

    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()

    check_input_sizes(samples, styles)
    work = [(style, sample) for style in styles for sample in samples
            if style_applies(style, sample)]
    total = len(work)
    done = 0

    print(f"run dir : {run_dir}")
    print(f"model   : {model}  provider={args.provider}  effort={effort or 'config default'}")
    n_gt = sum(1 for s in samples if s.get("has_gt"))
    print(f"plan    : {len(styles)} styles x {len(samples)} frames "
          f"({n_gt} with ground truth, {len(samples) - n_gt} unlabelled) "
          f"= {len(work)} model calls")

    control = None
    if args.skip_control:
        print("control : SKIPPED (results will be marked unverified)")
    else:
        print("control : running synthetic vision-delivery check ...", flush=True)
        control = run_control(args, manifest, run_ctx)
        verdict = ("DELIVERED" if control["images_delivered"] else "NOT DELIVERED")
        print(f"control : {verdict}  code={control.get('prediction', {}).get('code')!r} "
              f"(expected {control['truth']['code']!r})  "
              f"circle_err={control['circle_error_px']}px  "
              f"tool_attempts={control['tool_attempts']}")
        (run_dir / "control.json").write_text(json.dumps(control, indent=2) + "\n")

    records: list[dict] = []
    write_lock = threading.Lock()
    partial_path = run_dir / "responses.partial.jsonl"

    def collect(record: dict) -> None:
        """Record a finished run and persist it immediately.

        A model call is expensive and cannot be replayed for free, so results are
        appended as they arrive: a crash later in the batch (or in report
        rendering) can never throw away completed work again.
        """
        with write_lock:
            records.append(record)
            with partial_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")

    def failure_record(style, sample, exc: BaseException) -> dict:
        """A run that blew up in the harness itself, recorded rather than raised."""
        return {
            "run_id": f"{style.name}__{sample['sample_id']}",
            "style": style.name, "sample_id": sample["sample_id"],
            "has_gt": bool(sample.get("has_gt")), "source": sample["source"],
            "session": sample["session"], "episode": sample["episode"],
            "output_shape": style.output_shape, "status": parse.EXEC_ERROR,
            "attempts": 0, "fallback_used": False,
            "error_px": None, "dx_px": None, "dy_px": None,
            "picking_point": None, "bbox": None, "overlay": None,
            "strawberries": None, "inventory": None, "target_index": None,
            "total_tokens": 0, "wall_s": 0.0, "tool_attempts": [],
            "response_text": "", "response_excerpt": "",
            "harness_error": f"{type(exc).__name__}: {exc}",
            **{f"pck_{t}": None for t in PCK_THRESHOLDS},
            **{key: None for key in ("n_strawberries", "n_inventory", "n_occluded",
                                     "n_mostly_hidden", "n_graspable", "n_calyx_visible",
                                     "n_peduncle_visible", "n_described", "n_ripe_reported",
                                     "mean_redness_pct", "max_redness_pct",
                                     "mean_occlusion_pct", "mean_confidence_pct",
                                     "n_points_out_of_frame")},
        }

    if args.jobs <= 1:
        for style, sample in work:
            try:
                record = run_one(args, manifest, style, sample, run_ctx)
            except Exception as exc:  # keep the batch alive
                record = failure_record(style, sample, exc)
            collect(record)
            done += 1
            print(f"[{done}/{total}] {record['run_id']:<44} {record['status']:<13} "
                  f"err={fmt(record['error_px'],1):>6}px tok={record['total_tokens']}",
                  flush=True)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            def guarded(style, sample):
                try:
                    return run_one(args, manifest, style, sample, run_ctx)
                except Exception as exc:
                    return failure_record(style, sample, exc)

            futures = {pool.submit(guarded, style, sample): (style, sample)
                       for style, sample in work}
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                collect(record)
                done += 1
                print(f"[{done}/{total}] {record['run_id']:<44} {record['status']:<13} "
                      f"err={fmt(record['error_px'],1):>6}px tok={record['total_tokens']}",
                      flush=True)

    records.sort(key=lambda r: (r["style"], r["sample_id"]))
    elapsed = time.perf_counter() - t0

    write_artifacts(run_dir, args, manifest, control, records, styles, samples,
                    started, elapsed,
                    {"model": model, "effort": effort, "provider": args.provider,
                     **cli_identity(args.provider), "catalog": catalog_info,
                     "harness": harness_info()})

    # The canonical records now live in responses.jsonl; drop the bookkeeping.
    partial_path.unlink(missing_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\ndone in {elapsed:.0f}s -> {run_dir}")
    print(f"report  : {run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
