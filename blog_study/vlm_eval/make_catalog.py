#!/usr/bin/env python3
"""Regenerate ``model_catalog_vision.json`` from the user-level Codex catalog.

Why this exists
---------------
The user-level catalog (``$CODEX_HOME/cc-switch-model-catalog.json``) declares the
DeepSeek slugs with ``input_modalities: ["text"]``.  ``codex exec -i`` honours that
and silently drops attached images, so a vision eval routed through the CLI would
score a blind model.  Passing ``-c model_catalog_json=<override>`` makes the CLI
accept images again.

The override is a *minimal textual edit* of the source catalog: text-only
``input_modalities`` blocks are rewritten, while already vision-enabled catalogs
are copied unchanged. The user's own config is never touched.

Usage:  python3 vlm_eval/make_catalog.py [--source PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

TEXT_ONLY = '"input_modalities": [\n        "text"\n      ]'
VISION = '"input_modalities": [\n        "text",\n        "image"\n      ]'

REPO = Path(__file__).resolve().parent.parent


def default_source() -> Path:
    import os

    codex_home = Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex"))
    return codex_home / "cc-switch-model-catalog.json"


def build(source: Path) -> tuple[str, list[str]]:
    text = source.read_text()
    patched = text.replace(TEXT_ONLY, VISION)

    catalog = json.loads(patched)
    slugs = []
    for model in catalog["models"]:
        if model.get("input_modalities") == ["text", "image"]:
            slugs.append(model.get("slug") or model.get("id") or "<unnamed>")
    if not slugs or any(
        "image" not in model.get("input_modalities", [])
        for model in catalog["models"]
    ):
        raise SystemExit(
            f"{source}: catalog must contain models and every model must declare image input"
        )
    return patched, slugs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=default_source())
    ap.add_argument("--out", type=Path, default=REPO / "vlm_eval" / "model_catalog_vision.json")
    args = ap.parse_args()

    patched, slugs = build(args.source)
    args.out.write_text(patched)
    digest = hashlib.sha256(patched.encode()).hexdigest()
    print(f"source : {args.source}")
    print(f"out    : {args.out}")
    print(f"sha256 : {digest}")
    print(f"vision-enabled slugs: {', '.join(slugs)}")


if __name__ == "__main__":
    main()
