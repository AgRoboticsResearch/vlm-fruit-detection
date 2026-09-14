#!/usr/bin/env python3
"""The eight prompt styles used by the eval.

A style bundles three things:

* ``image_kind`` – which prepared image the model actually receives
  (``raw`` | ``grid`` | ``exemplar``).
* ``use_output_schema`` – whether the CLI's ``--output-schema`` is applied.
* ``build_prompt`` – the exact text sent as the prompt.

The report prints these strings verbatim, so prompt text is never a hidden
variable.  ``{frame_w}``/``{frame_h}`` are filled from the manifest so nothing
assumes 640x480 beyond the defaults the data actually has.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

ImageKind = Literal["raw", "grid", "exemplar"]
AppliesTo = Literal["gt", "no_gt", "any"]
# The shape of the JSON the style asks for, which decides how the answer is parsed
# and validated: one strawberry, or a list of them.
OutputShape = Literal["single", "list", "inventory"]


@dataclass(frozen=True)
class Style:
    name: str
    title: str
    summary: str
    image_kind: ImageKind
    use_output_schema: bool
    build_prompt: Callable[..., str]
    applies_to: AppliesTo = "gt"
    output_shape: OutputShape = "single"


def _target_definition(multi_fruit: bool) -> str:
    """What counts as 'the' target strawberry in this frame."""
    if multi_fruit:
        return (
            "This frame may contain SEVERAL strawberries, and more than one may be "
            "ripe. There is no single designated target. Treat as the target the "
            "ripest fruit: the most fully and uniformly red strawberry that is "
            "clearly visible and not badly occluded. If several qualify, pick the "
            "largest, most prominent one."
        )
    return (
        "Exactly one strawberry in the frame is the picking target: the ripe (fully "
        "red) fruit that the robot is about to pick."
    )


def _header(frame_w: int, frame_h: int, multi_fruit: bool = False) -> str:
    return (
        f"You are looking at ONE image: a single {frame_w}x{frame_h} colour frame from a "
        "strawberry-harvesting robot's wrist camera (Intel RealSense D405), taken at the "
        "start of a picking attempt.\n"
        f"{_target_definition(multi_fruit)}\n"
        "\n"
        "COORDINATE SYSTEM for every coordinate you report: pixel coordinates, origin "
        "(0, 0) at the TOP-LEFT corner of the frame, x increasing to the RIGHT "
        f"(valid 0..{frame_w - 1}), y increasing DOWNWARD (valid 0..{frame_h - 1}). "
        "Report whole numbers of pixels."
    )


# ---------------------------------------------------------------------------
# 1. single_basic (was baseline_json)
# ---------------------------------------------------------------------------

def build_baseline_json(frame_w: int = 640, frame_h: int = 480,
                        multi_fruit: bool = False, **_) -> str:
    return (
        f"{_header(frame_w, frame_h, multi_fruit)}\n"
        "\n"
        "TASK\n"
        "1. Locate the target strawberry"
        + (" (see the selection rule above)" if multi_fruit else "") + ".\n"
        "2. Give the point where the robot's gripper should close to pick it: the "
        "picking point on the peduncle (stem), just above the fruit's green calyx.\n"
        "\n"
        "Reply with ONLY a JSON object and nothing else - no prose, no explanation, no "
        "code fences:\n"
        '{"ripe": true, "bbox": [x1, y1, x2, y2], "picking_point": [u, v]}\n'
        "\n"
        '- "ripe": true if the target fruit is fully ripe, otherwise false.\n'
        '- "bbox": the target strawberry\'s bounding box; (x1, y1) is the top-left '
        "corner and (x2, y2) the bottom-right corner.\n"
        '- "picking_point": [u, v], the grasp point on the peduncle just above the '
        "calyx.\n"
        "\n"
        "The JSON object must contain exactly these three fields - no additional "
        "fields, no renamed fields."
    )


# ---------------------------------------------------------------------------
# 2. single_strict (was strict_schema)
# ---------------------------------------------------------------------------

def build_strict_schema(frame_w: int = 640, frame_h: int = 480,
                        multi_fruit: bool = False, **_) -> str:
    return (
        f"{_header(frame_w, frame_h, multi_fruit)}\n"
        "\n"
        "TASK\n"
        "Return the target strawberry's bounding box and its picking point (the grasp "
        "point on the peduncle, immediately above the green calyx).\n"
        "\n"
        "OUTPUT CONTRACT - a JSON schema is enforced for this response, so any deviation "
        "is rejected outright:\n"
        '- "ripe": boolean.\n'
        f'- "bbox": exactly 4 integers [x1, y1, x2, y2] with x1 <= x2, y1 <= y2, '
        f"x1/x2 in [0, {frame_w - 1}], y1/y2 in [0, {frame_h - 1}].\n"
        '- "picking_point": exactly 2 integers [u, v] with u in '
        f"[0, {frame_w - 1}] and v in [0, {frame_h - 1}].\n"
        '- The object contains exactly these three fields - nothing more, nothing '
        "renamed.\n"
        "Emit that JSON object and nothing else."
    )


# ---------------------------------------------------------------------------
# 3. single_reasoned (was reasoning_first)
# ---------------------------------------------------------------------------

def build_reasoning_first(frame_w: int = 640, frame_h: int = 480,
                          multi_fruit: bool = False, **_) -> str:
    return (
        f"{_header(frame_w, frame_h, multi_fruit)}\n"
        "\n"
        "Work through the scene in this exact order before answering.\n"
        "1. SEPALS - find the green calyx (sepal ring) of the target fruit.\n"
        "2. FRUIT - find the fully ripe red fruit hanging below that calyx, and note "
        "how far it extends in every direction.\n"
        "3. PEDUNCLE - find the stem immediately above the calyx and trace where it "
        "leaves the fruit.\n"
        "4. GRASP POINT - choose the point on the peduncle just above the calyx where a "
        "parallel-jaw gripper would close. It sits at or slightly above the top of the "
        "fruit, never on the body of the fruit.\n"
        "\n"
        "Write your reasoning in a few short lines. Then finish with a final line that "
        "contains ONLY this JSON object, with no code fences:\n"
        '{"ripe": true, "bbox": [x1, y1, x2, y2], "picking_point": [u, v]}\n'
        "\n"
        "That object must contain exactly the three fields shown - no additional "
        "fields, no renamed fields."
    )


# ---------------------------------------------------------------------------
# 4. single_example (was few_shot_exemplar)
# ---------------------------------------------------------------------------

def build_few_shot_exemplar(
    frame_w: int = 640,
    frame_h: int = 480,
    query_origin: tuple[int, int] | None = None,
    multi_fruit: bool = False,
    **_,
) -> str:
    ox, oy = query_origin or (0, 0)
    qx0, qx1 = ox, ox + frame_w - 1
    qy0, qy1 = oy, oy + frame_h - 1
    return (
        "You are looking at ONE image that contains TWO panels placed side by side on a "
        "grey canvas.\n"
        "\n"
        "LEFT PANEL - an EXAMPLE, from the same robot. It shows a correct annotation: "
        "the target strawberry outlined by a box, and the picking point marked with a "
        "green crosshair labelled with its coordinates. The picking point is on the "
        "peduncle (stem) just above the fruit's green calyx.\n"
        "RIGHT PANEL - YOUR FRAME. It is unannotated; you must produce the annotation.\n"
        + (f"{_target_definition(multi_fruit)}\n" if multi_fruit else "")
        +
        "\n"
        "The tick labels along the bottom and left edges of the whole image give "
        "coordinates in the GLOBAL pixel coordinate system of this composite image: "
        "origin (0, 0) at the top-left of the WHOLE image, x increasing to the RIGHT, y "
        "increasing DOWNWARD.\n"
        f"The RIGHT panel is exactly a {frame_w}x{frame_h} frame occupying global x from "
        f"{qx0} to {qx1} and global y from {qy0} to {qy1}.\n"
        "\n"
        "TASK\n"
        "Produce the same kind of annotation for the RIGHT panel: a box around the target "
        "strawberry's body, plus the picking point on the peduncle just above the calyx. "
        "In the example the marked point sits at the top edge of the box.\n"
        "\n"
        "Report bbox and picking_point as GLOBAL coordinates in this composite image, "
        "read directly off the labelled axes. Reply with ONLY a JSON object and nothing "
        "else - no prose, no code fences:\n"
        '{"ripe": true, "bbox": [x1, y1, x2, y2], "picking_point": [u, v]}\n'
        "\n"
        "That object must contain exactly the three fields shown - no additional "
        "fields, no renamed fields."
    )


# ---------------------------------------------------------------------------
# 5. single_grid (was grid_overlay)
# ---------------------------------------------------------------------------

def build_grid_overlay(frame_w: int = 640, frame_h: int = 480, grid_step: int = 64, **_) -> str:
    return (
        "You are looking at ONE image: a single colour frame from a "
        "strawberry-harvesting robot's wrist camera (Intel RealSense D405), taken at the "
        "start of a picking attempt.\n"
        "\n"
        f"A coordinate grid with a spacing of {grid_step} pixels has been drawn over the "
        f"frame. Each grid line is labelled along the top edge (x values) and the left "
        f"edge (y values) with the image coordinate it marks. The frame is "
        f"{frame_w}x{frame_h}: pixel coordinates have their origin (0, 0) at the TOP-LEFT "
        f"corner, x increases to the RIGHT (valid 0..{frame_w - 1}) and y increases "
        f"DOWNWARD (valid 0..{frame_h - 1}).\n"
        "\n"
        "The grid is the only thing drawn on top of the photo. The strawberry is NOT "
        "outlined, marked or labelled in any way, so read its position off the grid "
        "yourself. Exactly one strawberry is the picking target: the ripe (fully red) "
        "fruit about to be picked.\n"
        "\n"
        "TASK\n"
        "Locate the target strawberry and the point where the gripper should close to "
        "pick it: the picking point on the peduncle (stem), just above the fruit's green "
        "calyx. Read both off the grid.\n"
        "\n"
        "Reply with ONLY a JSON object and nothing else - no prose, no code fences:\n"
        '{"ripe": true, "bbox": [x1, y1, x2, y2], "picking_point": [u, v]}\n'
        "\n"
        "That object must contain exactly the three fields shown - no additional "
        "fields, no renamed fields."
    )


def build_enumerate_strawberries(frame_w: int = 640, frame_h: int = 480,
                                 multi_fruit: bool = True, **_) -> str:
    """For scenes that are not a single-target pick: report every ripe fruit."""
    return (
        f"You are looking at ONE image: a {frame_w}x{frame_h} colour photograph of a "
        "strawberry plant. Unlike a single-target picking frame, this scene usually "
        "contains SEVERAL strawberries, and more than one may be ripe.\n"
        "\n"
        "COORDINATE SYSTEM for every coordinate you report: pixel coordinates, origin "
        "(0, 0) at the TOP-LEFT corner of the image, x increasing to the RIGHT "
        f"(valid 0..{frame_w - 1}), y increasing DOWNWARD (valid 0..{frame_h - 1}). "
        "Report whole numbers of pixels.\n"
        "\n"
        "TASK\n"
        "Find EVERY strawberry that is ripe enough to pick (fully or almost fully red, "
        "with no large green or white patch). For each one, report:\n"
        '- "bbox": its bounding box as [x1, y1, x2, y2], where (x1, y1) is the '
        "top-left corner of the fruit and (x2, y2) the bottom-right corner.\n"
        '- "picking_point": [u, v], the point where a gripper should close to pick '
        "it - on the peduncle (stem) just above the fruit's green calyx, not on the "
        "body of the fruit.\n"
        '- "ripe": true for the fruits you are reporting.\n'
        "\n"
        "Report every ripe fruit you can see, even partially occluded ones, but do not "
        "report unripe green fruit and do not invent fruit that is not there. If you "
        "can see no ripe strawberry at all, return an empty list.\n"
        "\n"
        "Reply with ONLY a JSON object and nothing else - no prose, no code fences:\n"
        '{"strawberries": [{"ripe": true, "bbox": [x1, y1, x2, y2], '
        '"picking_point": [u, v]}, ...]}\n'
        "\n"
        "Each fruit object must contain exactly the three fields shown - no "
        "additional fields, no renamed fields."
    )


_INVENTORY_FIELDS = (
    '- "bbox": [x1, y1, x2, y2] - the bounding box of the WHOLE fruit. For a partly '
    "hidden fruit, give the box the fruit would occupy if the occluder were not "
    "there, not just the visible sliver.\n"
    '- "redness_pct": 0-100, a CONTINUOUS estimate - do not round to a category. '
    "Percent of the fruit's VISIBLE surface that is red, judged by hue and "
    "saturation together: 0 = no red at all (green or white), 25 = a pale pink "
    "flush, 50 = about half the visible surface is red, 75 = mostly red with pale "
    "or green patches left, 100 = fully saturated deep red everywhere. Use the "
    "whole range; 63 is a better answer than 60.\n"
    '- "occlusion_pct": 0-100, also CONTINUOUS. How much of the fruit is hidden '
    "behind leaves, stems or other fruit: 0 = fully visible, 25 = a leaf edge "
    "clips it, 50 = about half hidden, 75 = mostly hidden, 100 = only a sliver "
    "shows.\n"
    '- "calyx_visible": true if the green calyx (sepal ring) can be seen.\n'
    '- "peduncle_visible": true if the stem is visible where it meets the fruit.\n'
    '- "graspable": true if a gripper could pick this fruit right now, judging only '
    "from what is visible.\n"
    '- "picking_point": [u, v] on the peduncle just above the calyx, or null if the '
    "peduncle is not visible.\n"
    '- "confidence_pct": 0-100. How sure you are that this really is a strawberry.\n'
    '- "description": one or two sentences of plain language describing this fruit: '
    "its colour and colour pattern, what is occluding it and where, how big it looks, "
    "where it sits on the plant, and anything else a picker would want to know. This "
    "is free text - write what you actually see, not a restatement of the numbers.\n"
)

_INVENTORY_RULES = (
    "Report EVERY strawberry you can see, whatever its colour. This is an inventory, "
    "not a picking decision: do NOT filter by ripeness and do NOT omit a fruit just "
    "because it is green, small, or partly hidden. Partly occluded fruit matters as "
    "much as fully visible fruit - include a fruit if you can see any part of it, and "
    "say how much is hidden in occlusion_pct. Order the list from largest to smallest "
    "apparent size. Do not invent fruit that is not there.\n"
    "\n"
    "Do not label fruit as 'ripe' or 'unripe' - report the continuous redness and "
    "occlusion numbers and describe the fruit in words instead.\n"
    "\n"
    "Each strawberry must carry EXACTLY the nine fields listed above - no more, no "
    "fewer, none renamed. If you have nothing for a field, still include it; null is "
    "valid only for picking_point. Put any extra remarks inside description rather "
    "than adding a field.\n"
    "\n"
    "Then nominate the single fruit you would pick NOW, as target_index: the 0-based "
    "index into your list. Judge that from the redness and how reachable the fruit "
    "is, and say why in its description. If no fruit is worth picking, use -1. The "
    "nomination is separate from the inventory - it does not change which fruit you "
    "listed.\n"
    "\n"
    "Reply with ONLY a JSON object and nothing else - no prose, no code fences:\n"
    '{"strawberries": [{"bbox": [x1, y1, x2, y2], "redness_pct": 0, '
    '"occlusion_pct": 0, "calyx_visible": true, "peduncle_visible": true, '
    '"graspable": true, "picking_point": [u, v], "confidence_pct": 0, '
    '"description": "..."}, ...], "target_index": 0}'
)


def build_inventory_json(frame_w: int = 640, frame_h: int = 480, **_) -> str:
    """Unbiased full-scene strawberry inventory, one pass, no ripeness filter."""
    return (
        f"You are looking at ONE image: a {frame_w}x{frame_h} colour photograph of a "
        "strawberry plant.\n"
        "\n"
        "COORDINATE SYSTEM for every coordinate you report: pixel coordinates, origin "
        "(0, 0) at the TOP-LEFT corner of the image, x increasing to the RIGHT "
        f"(valid 0..{frame_w - 1}), y increasing DOWNWARD (valid 0..{frame_h - 1}). "
        "Report whole numbers of pixels.\n"
        "\n"
        "TASK\n"
        "Produce a complete inventory of the strawberries in this image. For each one, "
        "describe it fully:\n"
        f"{_INVENTORY_FIELDS}"
        "\n"
        f"{_INVENTORY_RULES}"
    )


def build_inventory_reasoning(frame_w: int = 640, frame_h: int = 480, **_) -> str:
    """Same inventory task, but scan the scene systematically before answering."""
    return (
        f"You are looking at ONE image: a {frame_w}x{frame_h} colour photograph of a "
        "strawberry plant.\n"
        "\n"
        "COORDINATE SYSTEM for every coordinate you report: pixel coordinates, origin "
        "(0, 0) at the TOP-LEFT corner of the image, x increasing to the RIGHT "
        f"(valid 0..{frame_w - 1}), y increasing DOWNWARD (valid 0..{frame_h - 1}). "
        "Report whole numbers of pixels.\n"
        "\n"
        "Work through the image systematically before answering, so that partly hidden "
        "fruit is not missed:\n"
        "1. Scan the image in bands - top to bottom, left to right - and note every "
        "place where strawberry tissue (red, pink, pale or green fruit surface) is "
        "visible.\n"
        "2. For each one, look for the calyx and the peduncle, and work out how much of "
        "the fruit is hidden behind leaves, stems or other fruit.\n"
        "3. For each one, judge how red the visible surface is on a 0-100 scale, and "
        "how much of it is hidden.\n"
        "4. Only then decide which single fruit, if any, is worth picking now.\n"
        "\n"
        "Write your scan briefly. Then finish with a final line containing ONLY this "
        "JSON object, with no code fences:\n"
        "\n"
        "Each strawberry must carry these fields:\n"
        f"{_INVENTORY_FIELDS}"
        "\n"
        f"{_INVENTORY_RULES}"
    )


# The one prompt the pipeline runs going forward (2026-09-14 decision): the
# unbiased full inventory. Everything else is archived below -- still selectable
# with an explicit --styles flag and still resolvable for --rebuild-report, but
# excluded from default batches and listed separately by --list-styles.
STYLES: dict[str, Style] = {
    "inventory_plain": Style(
        name="inventory_plain",
        title="Full inventory + target (single pass)",
        summary="Unbiased inventory of EVERY strawberry with ripeness, redness, "
                "occlusion and graspability, plus a nominated pick target. No colour "
                "filter.",
        image_kind="raw",
        use_output_schema=False,
        build_prompt=build_inventory_json,
        applies_to="any",
        output_shape="inventory",
    ),
}

ARCHIVED_STYLES: dict[str, Style] = {
    "single_basic": Style(
        name="single_basic",
        title="Baseline JSON (single target)",
        summary="Minimal instruction: bbox + picking point as JSON. Raw frame.",
        image_kind="raw",
        use_output_schema=False,
        build_prompt=build_baseline_json,
        # Also runs on the multi-strawberry frames (unscored there), so that a
        # single-target answer and an enumerate-everything answer can be compared
        # on the same scene.
        applies_to="any",
    ),
    "single_strict": Style(
        name="single_strict",
        title="Strict schema (single target)",
        summary="Explicit ranges and ordering, enforced by --output-schema. Raw frame.",
        image_kind="raw",
        use_output_schema=True,
        build_prompt=build_strict_schema,
    ),
    "single_reasoned": Style(
        name="single_reasoned",
        title="Reasoning first (single target)",
        summary="Ground sepals -> fruit -> peduncle -> grasp point, then emit JSON. Raw frame.",
        image_kind="raw",
        use_output_schema=False,
        build_prompt=build_reasoning_first,
    ),
    "single_example": Style(
        name="single_example",
        title="Few-shot exemplar (single target)",
        summary="Annotated example panel beside the query panel; tests in-context visual reference.",
        image_kind="exemplar",
        use_output_schema=False,
        build_prompt=build_few_shot_exemplar,
    ),
    "single_grid": Style(
        name="single_grid",
        title="Grid overlay (single target)",
        summary="Query frame with a labelled 64 px grid; coordinates read off the grid.",
        image_kind="grid",
        use_output_schema=False,
        build_prompt=build_grid_overlay,
    ),
    "list_ripe_only": Style(
        name="list_ripe_only",
        title="Ripe-only enumeration",
        summary="List EVERY ripe strawberry with its bbox and picking point. "
                "Used for the unlabelled multi-strawberry scenes. BIASED: ripe only.",
        image_kind="raw",
        use_output_schema=False,
        build_prompt=build_enumerate_strawberries,
        applies_to="no_gt",
        output_shape="list",
    ),
    "inventory_scan_first": Style(
        name="inventory_scan_first",
        title="Full inventory + target (scan first)",
        summary="Same unbiased inventory, but the model scans the scene in bands first "
                "so partly hidden fruit is not missed.",
        image_kind="raw",
        use_output_schema=False,
        build_prompt=build_inventory_reasoning,
        applies_to="any",
        output_shape="inventory",
    ),
}

# Names from before the 2026-09-14 rename, kept working as selectors and for
# rebuilding runs whose records carry the old names.
STYLE_ALIASES = {
    "baseline_json": "single_basic",
    "strict_schema": "single_strict",
    "reasoning_first": "single_reasoned",
    "few_shot_exemplar": "single_example",
    "grid_overlay": "single_grid",
    "enumerate_strawberries": "list_ripe_only",
    "inventory_json": "inventory_plain",
    "inventory_reasoning": "inventory_scan_first",
}

DEFAULT_STYLE_ORDER = list(STYLES)

# Everything defined, active or archived -- what lookups resolve against.
ALL_STYLES: dict[str, Style] = {**STYLES, **ARCHIVED_STYLES}


def style_by_name(name: str) -> Style:
    """Look a style up by current name or pre-rename alias, active or archived."""
    if name in ALL_STYLES:
        return ALL_STYLES[name]
    canonical = STYLE_ALIASES.get(name)
    if canonical:
        return ALL_STYLES[canonical]
    raise SystemExit(
        f"unknown prompt style {name!r}. Known: {', '.join(ALL_STYLES)} "
        f"(aliases: {', '.join(STYLE_ALIASES)})"
    )


def resolve_styles(requested: list[str] | None) -> list[Style]:
    """Resolve prompt selectors to styles, accepting either the name or the title.

    Matching is case-insensitive and ignores surrounding whitespace, so a caller
    can pass `inventory_plain`, `Full inventory + target (single pass)`, or any
    unique prefix of either. Pre-rename names (`inventory_json`, ...) resolve to
    their current style via STYLE_ALIASES. With no selectors, only the ACTIVE
    styles run; archived styles must be named explicitly.
    """
    if not requested:
        return [STYLES[name] for name in DEFAULT_STYLE_ORDER]

    resolved: list[Style] = []
    for raw in requested:
        token = raw.strip().lower()
        token = STYLE_ALIASES.get(token, token)
        exact = [s for s in ALL_STYLES.values() if token in (s.name.lower(), s.title.lower())]
        if not exact:
            exact = [s for s in ALL_STYLES.values()
                     if s.name.lower().startswith(token) or s.title.lower().startswith(token)]
        if len(exact) != 1:
            raise SystemExit(
                f"prompt selector {raw!r} matched {len(exact)} prompts "
                f"({[s.name for s in exact]}).\nAvailable prompts:\n"
                + format_style_table()
            )
        resolved.append(exact[0])

    # De-duplicate, preserving first-mention order.
    seen, unique = set(), []
    for style in resolved:
        if style.name not in seen:
            seen.add(style.name)
            unique.append(style)
    return unique


def format_style_table() -> str:
    """Human/agent readable catalogue of the available prompts."""
    width = max(len(s.name) for s in ALL_STYLES.values())
    rows = [f"  {'name'.ljust(width)}  {'runs on':<9} {'answer':<10} title",
            f"  {'-' * width}  {'-' * 9} {'-' * 10} {'-' * 40}"]
    for style in STYLES.values():
        rows.append(f"  {style.name.ljust(width)}  {style.applies_to:<9} "
                    f"{style.output_shape:<10} {style.title}")
    if ARCHIVED_STYLES:
        rows.append("")
        rows.append("  archived (run only when named explicitly in --styles):")
        for style in ARCHIVED_STYLES.values():
            rows.append(f"  {style.name.ljust(width)}  {style.applies_to:<9} "
                        f"{style.output_shape:<10} {style.title}")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Synthetic vision control
# ---------------------------------------------------------------------------

CONTROL_PROMPT = (
    "This is a synthetic test image, 640x480 pixels, containing a 4-digit code and two "
    "coloured shapes on a plain white background.\n"
    "\n"
    "Report three things.\n"
    "1. The 4-digit code, read digit by digit, as a string.\n"
    "2. The centre of the red circle, as [u, v].\n"
    "3. The centre of the green square, as [u, v].\n"
    "\n"
    "Coordinates use a top-left origin: x increases to the right (0..639), y increases "
    "downward (0..479), whole pixels.\n"
    "\n"
    "Reply with ONLY this JSON object and nothing else - no prose, no code fences:\n"
    '{"code": "0000", "red_circle": [u, v], "green_square": [u, v]}'
)
