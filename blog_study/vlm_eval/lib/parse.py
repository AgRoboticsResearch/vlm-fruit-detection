#!/usr/bin/env python3
"""Extract and validate the model's JSON answer from a raw response string.

The model is asked for bare JSON, but reasoning-first prompts append prose and
some models still wrap the object in code fences, so extraction is deliberately
forgiving.  What is *not* forgiving is treating an unparseable or missing answer
as a success: every failure mode gets an explicit status.
"""

from __future__ import annotations

import json
import re
from typing import Any

import jsonschema

# Status values recorded per run.
OK = "ok"
OUT_OF_FRAME = "out_of_frame"
# Parsed correctly, but the answer offers no grasp point to score - e.g. an
# inventory that found the fruit yet reported the peduncle as hidden. This is a
# legitimate answer, not a failure, but it produces no number.
NO_PICK_POINT = "no_pick_point"
SCHEMA_INVALID = "schema_invalid"
EMPTY_RESPONSE = "empty"
REFUSED = "refused"
PARSE_ERROR = "parse_error"
EXEC_ERROR = "exec_error"

FAILURE_STATUSES = (EMPTY_RESPONSE, REFUSED, PARSE_ERROR, EXEC_ERROR, SCHEMA_INVALID)

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
REFUSAL_RE = re.compile(
    r"\b(i (?:can(?:no|')t|am unable|will not|won't)|unable to (?:assist|help)|"
    r"as an ai|i'm sorry|i am sorry|cannot (?:assist|help|comply))\b",
    re.IGNORECASE,
)


def _balanced_spans(text: str, opener: str, closer: str):
    """Yield every balanced ``opener..closer`` substring, respecting strings."""
    depth = 0
    start = None
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            if depth == 0:
                start = i
            depth += 1
        elif ch == closer:
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start:i + 1]
                    start = None


def _balanced_objects(text: str):
    """Yield every balanced {...} substring."""
    yield from _balanced_spans(text, "{", "}")


def _balanced_arrays(text: str):
    """Yield every balanced [...] substring."""
    yield from _balanced_spans(text, "[", "]")


def extract_json(text: str) -> tuple[Any | None, str]:
    """Return (parsed_object, method).  ``method`` documents how it was found."""
    if text is None:
        return None, "none"
    stripped = (text or "").strip()
    if not stripped:
        return None, "empty"

    try:
        return json.loads(stripped), "direct"
    except json.JSONDecodeError:
        pass

    for match in FENCE_RE.finditer(stripped):
        body = match.group(1).strip()
        try:
            return json.loads(body), "fence"
        except json.JSONDecodeError:
            continue

    # Prefer the LAST balanced object: reasoning-first answers end with the JSON.
    candidates = list(_balanced_objects(stripped))
    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, "balanced"

    # Last resort: a lenient fix for trailing commas / single quotes.
    for candidate in reversed(candidates):
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate).replace("'", '"')
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, "repaired"

    # Some prompts (the multi-strawberry style) allow a bare top-level list.
    for candidate in reversed(list(_balanced_arrays(stripped))):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed, "array"
    return None, "no_json"


def looks_like_refusal(text: str) -> bool:
    return bool(REFUSAL_RE.search(text or ""))


def validate(obj: Any, schema: dict) -> tuple[bool, str | None]:
    try:
        jsonschema.validate(obj, schema)
    except jsonschema.ValidationError as exc:
        path = "/".join(str(p) for p in exc.absolute_path) or "<root>"
        return False, f"{path}: {exc.message}"
    except jsonschema.SchemaError as exc:  # pragma: no cover - schema is checked in CI
        raise SystemExit(f"invalid schema: {exc.message}") from exc
    return True, None


def normalise_point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def normalise_box(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
