#!/usr/bin/env python3
"""The vlm_eval segmentation pipeline's prompt: the picking inventory, plus a polygon.

This is the full_detection pipeline's ``inventory_plain`` (the base harness's
unbiased nine-field inventory + pick nomination) with ONE field added: a
``polygon`` tracing each fruit's VISIBLE surface, inserted right after the
bbox it must agree with. The ten-field output standard is otherwise carried
over VERBATIM — picking_point, target_index and every anchor sentence stay
byte-identical — so the nominated-target cross-check scores against ground
truth exactly like ``full_detection`` and the eight census fields stay
comparable across the two pipelines.

The polygon's semantics match the StrawDI seg pipeline's prompt
(``strawdi_eval/seg/lib/prompt.py``): the VISIBLE surface, following occluder
edges rather than extrapolating the hidden shape. One wording difference,
deliberate: the fruit-body rule is stated inline ("Outline ONLY the fruit
body: exclude the calyx and stem where they sit apart from the fruit")
because ``inventory_plain``'s bbox field carries no such rule to refer back
to — the StrawDI prompt could say "same fruit-body rule as bbox", this one
cannot.

No mask ground truth exists on any vlm_eval source, so the polygon is never
scored against an annotated mask here (the StrawDI pipeline is where that
comparison lives). It is rendered, checked for internal consistency against
the answer's own bbox/picking_point, and — on ground-truthed frames — its
extent is scored against the same upstream rough box the box pipeline's
approximate bbox-IoU uses.

Validated by ``../schema/inventory_segmentation_schema.json`` (ten fields +
target_index, ``additionalProperties: false`` everywhere).
"""

from __future__ import annotations

STYLE_NAME = "inventory_segmentation"

# Per-fruit fields: the nine inventory_plain fields verbatim, with the polygon
# inserted right after the bbox it must agree with.
_FIELDS = (
    '- "bbox": [x1, y1, x2, y2] - the bounding box of the WHOLE fruit. For a partly '
    "hidden fruit, give the box the fruit would occupy if the occluder were not "
    "there, not just the visible sliver.\n"
    '- "polygon": an ordered list of [x, y] vertices tracing the outline of the '
    "fruit's VISIBLE surface - the part you can actually see. Walk the boundary "
    "once, clockwise or counter-clockwise, starting anywhere. Where a leaf, stem "
    "or another fruit hides part of this fruit, follow the occluder's edge; do "
    "NOT extrapolate the shape you cannot see (unlike bbox, polygon covers ONLY "
    "what is visible). Outline ONLY the fruit body: exclude the calyx (the green "
    "sepal leaves) and the stem where they sit apart from the fruit. Use 8 to 20 "
    "vertices for a typical fruit and never more than 32; a simple closed outline "
    "(no self-crossing, no repeated closing vertex) drawn with straight segments "
    "between vertices. The polygon's extent must agree with the visible part of "
    "the fruit, and whole-pixel coordinates.\n"
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
    "its colour and colour pattern, what is occluding it and where, how big it "
    "looks, where it sits on the plant, and anything else a picker would want to "
    "know. This is free text - write what you actually see, not a restatement of "
    "the numbers.\n"
)

_RULES = (
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
    "Each strawberry must carry EXACTLY the ten fields listed above - no more, no "
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
    '{"strawberries": [{"bbox": [x1, y1, x2, y2], "polygon": [[x, y], [x, y], ...], '
    '"redness_pct": 0, "occlusion_pct": 0, "calyx_visible": true, '
    '"peduncle_visible": true, "graspable": true, "picking_point": [u, v], '
    '"confidence_pct": 0, "description": "..."}, ...], "target_index": 0}'
)


def build_inventory_segmentation(frame_w: int, frame_h: int, **_) -> str:
    """Unbiased full-scene inventory with visible-surface polygons + pick nomination."""
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
