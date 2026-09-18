#!/usr/bin/env python3
"""The segmentation pipeline's prompt: the detection inventory, plus a polygon.

This is the StrawDI detection prompt (``inventory_detection``: the base
harness's unbiased inventory minus the picking-oriented parts) with ONE field
added: a ``polygon`` tracing each fruit's VISIBLE surface. The nine-field
output standard is otherwise carried over verbatim so the eight scored
detection fields stay comparable across the two pipelines, and the box the
model reports is scored against the same GT boxes as the detection eval.

The polygon is the point of this pipeline. Crucially its semantics — the
VISIBLE surface, following occluder edges rather than extrapolating the hidden
shape — match what the StrawDI ground-truth masks annotate, so the mask-vs-mask
comparison is semantics-clean (the whole-fruit vs visible-surface gap that
biases the box metric does not exist here).

Validated by ``../schema/inventory_segmentation_schema.json`` (nine fields,
``additionalProperties: false`` everywhere — a reply that volunteers a
``picking_point`` or ``target_index`` is rejected wholesale, same rule as the
base harness).
"""

from __future__ import annotations

STYLE_NAME = "inventory_segmentation"

# Per-fruit fields: the eight detection fields verbatim, with the polygon
# inserted right after the bbox it must agree with.
_FIELDS = (
    '- "bbox": [x1, y1, x2, y2] - the bounding box of the WHOLE fruit. For a partly '
    "hidden fruit, give the box the fruit would occupy if the occluder were not "
    "there, not just the visible sliver. Box ONLY the fruit body: do not extend "
    "the box to cover the calyx (the green sepal leaves) or the stem when they "
    "sit apart from the fruit - a calyx that lies flat against the fruit is of "
    "course inside the box.\n"
    '- "polygon": an ordered list of [x, y] vertices tracing the outline of the '
    "fruit's VISIBLE surface - the part you can actually see. Walk the boundary "
    "once, clockwise or counter-clockwise, starting anywhere. Where a leaf, stem "
    "or another fruit hides part of this fruit, follow the occluder's edge; do "
    "NOT extrapolate the shape you cannot see (unlike bbox, polygon covers ONLY "
    "what is visible). Same fruit-body rule as bbox: exclude the calyx and stem "
    "where they sit apart from the fruit. Use 8 to 20 vertices for a typical "
    "fruit and never more than 32; a simple closed outline (no self-crossing, "
    "no repeated closing vertex) drawn with straight segments between vertices. "
    "The polygon's extent must agree with the visible part of the fruit, and "
    "whole-pixel coordinates.\n"
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
    '- "confidence_pct": 0-100. How sure you are that this really is a strawberry.\n'
    '- "description": one or two sentences of plain language describing this fruit: '
    "its colour and colour pattern, what is occluding it and where, how big it "
    "looks, where it sits on the plant, and anything else a picker would want to "
    "know. This is free text - write what you actually see, not a restatement of "
    "the numbers.\n"
)

_RULES = (
    "Report EVERY strawberry you can see, whatever its colour. This is an inventory, "
    "not a picking decision: do NOT filter by ripeness and do NOT omit a fruit just "
    "because it is green, small, or partly hidden. Partly occluded fruit matters as "
    "much as fully visible fruit - include a fruit if you can see any part of it, "
    "and say how much is hidden in occlusion_pct. Order the list from largest to smallest "
    "apparent size. Do not invent fruit that is not there.\n"
    "\n"
    "Do not label fruit as 'ripe' or 'unripe' - report the continuous redness and "
    "occlusion numbers and describe the fruit in words instead.\n"
    "\n"
    "Each strawberry must carry EXACTLY the nine fields listed above - no more, no "
    "fewer, none renamed. If you have nothing for a field, still include it. Put any "
    "extra remarks inside description rather than adding a field.\n"
    "\n"
    "Reply with ONLY a JSON object and nothing else - no prose, no code fences:\n"
    '{"strawberries": [{"bbox": [x1, y1, x2, y2], "polygon": [[x, y], [x, y], ...], '
    '"redness_pct": 0, "occlusion_pct": 0, "calyx_visible": true, '
    '"peduncle_visible": true, "graspable": true, "confidence_pct": 0, '
    '"description": "..."}, ...]}'
)


def build_inventory_segmentation(frame_w: int, frame_h: int, **_) -> str:
    """Unbiased full-scene strawberry inventory with visible-surface polygons."""
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
        "describe it fully and trace the outline of its visible surface:\n"
        f"{_FIELDS}"
        "\n"
        f"{_RULES}"
    )
