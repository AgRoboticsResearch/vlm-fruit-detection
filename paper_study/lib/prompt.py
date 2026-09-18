#!/usr/bin/env python3
"""The paper_study prompt: the StrawDI segmentation inventory, fruit-generic.

The output standard is the strawdi_segmentation nine-field census carried over
VERBATIM (same field names, same types, same ranges, same schema shape — only
the JSON key changes ``strawberries`` → ``fruits`` so records from the two
pipelines can never be confused). What changes is the wording around it: the
prompt no longer assumes a strawberry plant. The scene phrase, the fruit noun
and the ripe-colour definition are parameterised per dataset, because the
mixed testbed spans apples, grape clusters, mangoes, almonds and strawberries.

Parameterisation is part of the task definition, not a confound: each dataset's
protocol asks for one known fruit class, so naming that class in the prompt is
the cross-dataset analogue of the class query a detector is trained with.

The polygon semantics stay the StrawDI ones (VISIBLE surface, occluder edges,
8-20 typical vertices, 32 cap) — on the datasets whose GT annotates the
visible surface (StrawDI, WGISD cluster masks, MinneApple polygons) the
mask-vs-mask comparison stays semantics-clean; on ACFR (boxes only) the
polygon is asked but unscored, exactly like the base vlm_seg pipeline treats
unmaskable GT.
"""

from __future__ import annotations

STYLE_NAME = "fruit_inventory_segmentation"

# The nine fields: strawdi_segmentation's eight detection fields verbatim with
# the polygon after the bbox it must agree with. Wording genericised from
# "strawberry" to the parameterised fruit noun; anchors kept recognisable.
_FIELDS = (
    '- "bbox": [x1, y1, x2, y2] - the bounding box of the WHOLE fruit. For a partly '
    "hidden fruit, give the box the fruit would occupy if the occluder were not "
    "there, not just the visible sliver. Box ONLY the fruit body: do not extend "
    "the box to cover the calyx (the sepal leaves) or the stem when they "
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
    "How much of the fruit's VISIBLE surface shows its RIPE colour. "
    "{ripe_colour} "
    "Anchors: 0 = none of the visible surface is ripe-coloured (fully "
    "green/immature), 25 = a pale first flush, 50 = about half the visible "
    "surface is ripe-coloured, 75 = mostly ripe-coloured with green or pale "
    "patches left, 100 = fully ripe-coloured everywhere. Use the whole range; "
    "63 is a better answer than 60.\n"
    '- "occlusion_pct": 0-100, also CONTINUOUS. How much of the fruit is hidden '
    "behind leaves, stems or other fruit: 0 = fully visible, 25 = a leaf edge "
    "clips it, 50 = about half hidden, 75 = mostly hidden, 100 = only a sliver "
    "shows.\n"
    '- "calyx_visible": true if the green calyx (sepal ring at the fruit\'s stem '
    "end) can be seen; false if this fruit type has no calyx or it is not "
    "visible.\n"
    '- "peduncle_visible": true if the stem is visible where it meets the fruit; '
    "false if this fruit type has no visible stem or it is not visible.\n"
    '- "graspable": true if a gripper could pick this fruit right now, judging only '
    "from what is visible.\n"
    '- "confidence_pct": 0-100. How sure you are that this really is '
    "a {fruit_singular}.\n"
    '- "description": one or two sentences of plain language describing this fruit: '
    "its colour and colour pattern, what is occluding it and where, how big it "
    "looks, where it sits on the plant, and anything else a picker would want to "
    "know. This is free text - write what you actually see, not a restatement of "
    "the numbers.\n"
)

_RULES = (
    "Report EVERY {fruit_singular} you can see, whatever its colour. This is an inventory, "
    "not a picking decision: do NOT filter by ripeness and do NOT omit a fruit just "
    "because it is green, small, or partly hidden. Partly occluded fruit matters as "
    "much as fully visible fruit - include a fruit if you can see any part of it, "
    "and say how much is hidden in occlusion_pct. Order the list from largest to smallest "
    "apparent size. Do not invent fruit that is not there.\n"
    "\n"
    "Do not label fruit as 'ripe' or 'unripe' - report the continuous colour and "
    "occlusion numbers and describe the fruit in words instead.\n"
    "\n"
    "Each fruit must carry EXACTLY the nine fields listed above - no more, no "
    "fewer, none renamed. If you have nothing for a field, still include it. Put any "
    "extra remarks inside description rather than adding a field.\n"
    "\n"
    "Reply with ONLY a JSON object and nothing else - no prose, no code fences:\n"
    '{{"fruits": [{{"bbox": [x1, y1, x2, y2], "polygon": [[x, y], [x, y], ...], '
    '"redness_pct": 0, "occlusion_pct": 0, "calyx_visible": true, '
    '"peduncle_visible": true, "graspable": true, "confidence_pct": 0, '
    '"description": "..."}}, ...]}}'
)

# Per-dataset wording. `instance` is what one detected object IS (for grapes the
# instance is the CLUSTER, not the berry). `ripe_colour` anchors redness_pct.
SCENE_WORDING = {
    "strawdi": {
        "scene": "a strawberry field",
        "fruit_singular": "strawberry",
        "fruit_plural": "strawberries",
        "instance": "strawberry (one whole berry, not its seeds)",
        "ripe_colour": "for a strawberry this is red, judged by hue and "
                       "saturation together - 0 = green or white, 100 = fully "
                       "saturated deep red everywhere.",
    },
    "minneapple": {
        "scene": "an apple orchard",
        "fruit_singular": "apple",
        "fruit_plural": "apples",
        "instance": "apple (one whole fruit, not a cluster or a tree)",
        "ripe_colour": "for an apple this is the red blush of a ripe fruit - "
                       "0 = fully green, 100 = deep red over the whole visible "
                       "surface.",
    },
    "wgisd": {
        "scene": "a vineyard",
        "fruit_singular": "grape cluster",
        "fruit_plural": "grape clusters",
        "instance": "grape cluster (one whole bunch of berries hanging together "
                    "- count the BUNCH, not the individual berries)",
        "ripe_colour": "for a grape cluster this is the mature colour of that "
                       "variety (deep red-purple for red varieties, "
                       "yellow-green for white varieties) - 0 = hard green "
                       "immature berries, 100 = fully coloured mature cluster.",
    },
    "acfr_apples": {
        "scene": "an apple orchard at night, lit by artificial illumination",
        "fruit_singular": "apple",
        "fruit_plural": "apples",
        "instance": "apple (one whole fruit, not a cluster or a tree)",
        "ripe_colour": "for an apple this is the red blush of a ripe fruit - "
                       "0 = fully green, 100 = deep red over the whole visible "
                       "surface.",
    },
    "acfr_mangoes": {
        "scene": "a mango orchard at night, lit by artificial illumination",
        "fruit_singular": "mango",
        "fruit_plural": "mangoes",
        "instance": "mango (one whole fruit, not a cluster or a tree)",
        "ripe_colour": "for a mango this is the yellow-red blush of a ripe "
                       "fruit - 0 = fully green, 100 = fully yellow-red.",
    },
    "kfuji": {
        "scene": "a Fuji apple orchard",
        "fruit_singular": "apple",
        "fruit_plural": "apples",
        "instance": "apple (one whole fruit, not a cluster or a tree)",
        "ripe_colour": "for a Fuji apple this is the red-striped blush of a "
                       "ripe fruit - 0 = fully green, 100 = deep red over the "
                       "whole visible surface.",
    },
    "acfr_almonds": {
        "scene": "an almond orchard; the fruit is the green-ish husked almond "
                 "hanging on the tree (the in-shell almond with its leathery "
                 "outer hull, NOT the tree's pink blossoms, leaves or a "
                 "shelled nut)",
        "fruit_singular": "almond fruit (husk on the tree)",
        "fruit_plural": "almond fruits (husks on the tree)",
        "instance": "almond fruit - the whole green/greyish hanging hull "
                    "containing the nut",
        "ripe_colour": "for an almond hull this is how far the fuzzy green hull "
                       "has turned towards its dry, splitting brown-grey ripe "
                       "state - 0 = fully green, 100 = fully brown and split.",
    },
}


def build_fruit_inventory_segmentation(frame_w: int, frame_h: int, source: str,
                                       **_) -> str:
    """Unbiased full-scene fruit inventory with visible-surface polygons."""
    wording = SCENE_WORDING[source]
    fields = _FIELDS.format(ripe_colour=wording["ripe_colour"],
                            fruit_singular=wording["fruit_singular"])
    rules = _RULES.format(fruit_singular=wording["fruit_singular"])
    return (
        f"You are looking at ONE image: a {frame_w}x{frame_h} colour photograph of "
        f"{wording['scene']}.\n"
        "\n"
        "COORDINATE SYSTEM for every coordinate you report: pixel coordinates, origin "
        "(0, 0) at the TOP-LEFT corner of the image, x increasing to the RIGHT "
        f"(valid 0..{frame_w - 1}), y increasing DOWNWARD (valid 0..{frame_h - 1}). "
        "Report whole numbers of pixels.\n"
        "\n"
        "TASK\n"
        f"Produce a complete inventory of the {wording['fruit_plural']} in this "
        f"image - each instance is one {wording['instance']}. For each one, "
        "describe it fully and trace the outline of its visible surface:\n"
        f"{fields}"
        "\n"
        f"{rules}"
    )
