# The three fruit-inventory prompts

Source of truth: [`lib/prompt.py`](lib/prompt.py) (composable field texts +
per-dataset wording); answer contracts:
[`schema/`](schema/). Every call's exact prompt is recorded verbatim in its
run's `responses.jsonl` — this document is for reading, the code is for
rendering, the record is for proof.

## Shared contract (identical across all three prompts)

Everything except the per-fruit field list is held constant, so runs across
formats are comparable (see [`FORMAT_COMPARISON.md`](FORMAT_COMPARISON.md)):

* **Delivery** — one image, described as `a {W}x{H} colour photograph of
  {scene}`.
* **Coordinate system** — top-left origin, x right / y down, valid ranges
  spelled out for the frame size, whole pixels.
* **Census rules** — "Report EVERY {fruit} you can see, whatever its colour.
  This is an inventory, not a picking decision: do NOT filter by ripeness
  and do NOT omit a fruit just because it is green, small, or partly hidden.
  Partly occluded fruit matters as much as fully visible fruit — include a
  fruit if you can see any part of it. Order the list from largest to
  smallest apparent size. Do not invent fruit that is not there."
* **Output discipline** — "Each fruit must carry EXACTLY the {N} fields
  listed above — no more, no fewer, none renamed. If you have nothing for a
  field, still include it." + "Reply with ONLY a JSON object and nothing
  else — no prose, no code fences" + a one-line JSON example with the exact
  field set.
* **Field texts** — where two prompts share a field (`bbox`, `polygon`,
  `confidence_pct`), the field text is byte-identical between them.

## The three variants

| key | style name | per-fruit fields | schema | polygons? | fits |
| --- | --- | --- | --- | --- | --- |
| `full9` | `fruit_inventory_segmentation` | bbox, polygon, redness_pct, occlusion_pct, calyx_visible, peduncle_visible, graspable, confidence_pct, description | `fruit_inventory_schema.json` | yes | the strawdi_segmentation census — qualitative picking report + geometry |
| `seg3` | `fruit_box_polygon` | bbox, polygon, confidence_pct | `fruit_box_polygon_schema.json` | yes | minimal format for a SEGMENTATION task |
| `box2` | `fruit_box_minimal` | bbox, confidence_pct | `fruit_box_minimal_schema.json` | no | minimal format for a pure DETECTION task |

Field semantics (full wording in the verbatim texts below):

* `bbox` — `[x1, y1, x2, y2]` whole-pixel box of the **WHOLE** fruit
  (extrapolated past occluders; fruit body only, calyx/stem excluded where
  they sit apart).
* `polygon` — ordered outline of the **VISIBLE** surface (follows occluder
  edges, never extrapolates; 8–20 typical vertices, 32 max; simple closed,
  whole-pixel). The bbox/polygon contrast is deliberate and stated twice.
* `redness_pct` — continuous 0–100, how much of the visible surface shows
  the fruit's RIPE colour; anchors parameterised per dataset (below).
* `occlusion_pct` — continuous 0–100, how much of the fruit is hidden.
* `calyx_visible` / `peduncle_visible` — booleans, false when the fruit type
  has none.
* `graspable` — could a gripper pick it now, from what is visible.
* `confidence_pct` — 0–100, how sure this is really the asked fruit.
* `description` — one or two sentences of the model's own words.

## Prompt 1 — `full9` (rendered for strawdi, 1008×756)

```text
You are looking at ONE image: a 1008x756 colour photograph of a strawberry field.

COORDINATE SYSTEM for every coordinate you report: pixel coordinates, origin (0, 0) at the TOP-LEFT corner of the image, x increasing to the RIGHT (valid 0..1007), y increasing DOWNWARD (valid 0..755). Report whole numbers of pixels.

TASK
Produce a complete inventory of the strawberries in this image - each instance is one strawberry (one whole berry, not its seeds). For each one, describe it fully and trace the outline of its visible surface:
- "bbox": [x1, y1, x2, y2] - the bounding box of the WHOLE fruit. For a partly hidden fruit, give the box the fruit would occupy if the occluder were not there, not just the visible sliver. Box ONLY the fruit body: do not extend the box to cover the calyx (the sepal leaves) or the stem when they sit apart from the fruit - a calyx that lies flat against the fruit is of course inside the box.
- "polygon": an ordered list of [x, y] vertices tracing the outline of the fruit's VISIBLE surface - the part you can actually see. Walk the boundary once, clockwise or counter-clockwise, starting anywhere. Where a leaf, stem or another fruit hides part of this fruit, follow the occluder's edge; do NOT extrapolate the shape you cannot see (unlike bbox, polygon covers ONLY what is visible). Same fruit-body rule as bbox: exclude the calyx and stem where they sit apart from the fruit. Use 8 to 20 vertices for a typical fruit and never more than 32; a simple closed outline (no self-crossing, no repeated closing vertex) drawn with straight segments between vertices. The polygon's extent must agree with the visible part of the fruit, and whole-pixel coordinates.
- "redness_pct": 0-100, a CONTINUOUS estimate - do not round to a category. How much of the fruit's VISIBLE surface shows its RIPE colour. for a strawberry this is red, judged by hue and saturation together - 0 = green or white, 100 = fully saturated deep red everywhere. Anchors: 0 = none of the visible surface is ripe-coloured (fully green/immature), 25 = a pale first flush, 50 = about half the visible surface is ripe-coloured, 75 = mostly ripe-coloured with green or pale patches left, 100 = fully ripe-coloured everywhere. Use the whole range; 63 is a better answer than 60.
- "occlusion_pct": 0-100, also CONTINUOUS. How much of the fruit is hidden behind leaves, stems or other fruit: 0 = fully visible, 25 = a leaf edge clips it, 50 = about half hidden, 75 = mostly hidden, 100 = only a sliver shows.
- "calyx_visible": true if the green calyx (sepal ring at the fruit's stem end) can be seen; false if this fruit type has no calyx or it is not visible.
- "peduncle_visible": true if the stem is visible where it meets the fruit; false if this fruit type has no visible stem or it is not visible.
- "graspable": true if a gripper could pick this fruit right now, judging only from what is visible.
- "confidence_pct": 0-100. How sure you are that this really is a strawberry.
- "description": one or two sentences of plain language describing this fruit: its colour and colour pattern, what is occluding it and where, how big it looks, where it sits on the plant, and anything else a picker would want to know. This is free text - write what you actually see, not a restatement of the numbers.

Report EVERY strawberry you can see, whatever its colour. This is an inventory, not a picking decision: do NOT filter by ripeness and do NOT omit a fruit just because it is green, small, or partly hidden. Partly occluded fruit matters as much as fully visible fruit - include a fruit if you can see any part of it. Order the list from largest to smallest apparent size. Do not invent fruit that is not there.

Say how much of each fruit is hidden in occlusion_pct, and do not label fruit as 'ripe' or 'unripe' - report the continuous colour and occlusion numbers and describe the fruit in words instead.


Each fruit must carry EXACTLY the nine fields listed above - no more, no fewer, none renamed. If you have nothing for a field, still include it. Put any extra remarks inside description rather than adding a field.

Reply with ONLY a JSON object and nothing else - no prose, no code fences:
{"fruits": [{"bbox": [x1, y1, x2, y2], "polygon": [[x, y], [x, y], ...], "redness_pct": 0, "occlusion_pct": 0, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 0, "description": "..."}, ...]}
```

## Prompt 2 — `seg3` (rendered for strawdi, 1008×756)

```text
You are looking at ONE image: a 1008x756 colour photograph of a strawberry field.

COORDINATE SYSTEM for every coordinate you report: pixel coordinates, origin (0, 0) at the TOP-LEFT corner of the image, x increasing to the RIGHT (valid 0..1007), y increasing DOWNWARD (valid 0..755). Report whole numbers of pixels.

TASK
Produce a complete inventory of the strawberries in this image - each instance is one strawberry (one whole berry, not its seeds). For each one, give its bounding box and trace the outline of its visible surface:
- "bbox": [x1, y1, x2, y2] - the bounding box of the WHOLE fruit. For a partly hidden fruit, give the box the fruit would occupy if the occluder were not there, not just the visible sliver. Box ONLY the fruit body: do not extend the box to cover the calyx (the sepal leaves) or the stem when they sit apart from the fruit - a calyx that lies flat against the fruit is of course inside the box.
- "polygon": an ordered list of [x, y] vertices tracing the outline of the fruit's VISIBLE surface - the part you can actually see. Walk the boundary once, clockwise or counter-clockwise, starting anywhere. Where a leaf, stem or another fruit hides part of this fruit, follow the occluder's edge; do NOT extrapolate the shape you cannot see (unlike bbox, polygon covers ONLY what is visible). Same fruit-body rule as bbox: exclude the calyx and stem where they sit apart from the fruit. Use 8 to 20 vertices for a typical fruit and never more than 32; a simple closed outline (no self-crossing, no repeated closing vertex) drawn with straight segments between vertices. The polygon's extent must agree with the visible part of the fruit, and whole-pixel coordinates.
- "confidence_pct": 0-100. How sure you are that this really is a strawberry.

Report EVERY strawberry you can see, whatever its colour. This is an inventory, not a picking decision: do NOT filter by ripeness and do NOT omit a fruit just because it is green, small, or partly hidden. Partly occluded fruit matters as much as fully visible fruit - include a fruit if you can see any part of it. Order the list from largest to smallest apparent size. Do not invent fruit that is not there.


Each fruit must carry EXACTLY the three fields listed above - no more, no fewer, none renamed. If you have nothing for a field, still include it.

Reply with ONLY a JSON object and nothing else - no prose, no code fences:
{"fruits": [{"bbox": [x1, y1, x2, y2], "polygon": [[x, y], [x, y], ...], "confidence_pct": 0}, ...]}
```

## Prompt 3 — `box2` (rendered for strawdi, 1008×756)

```text
You are looking at ONE image: a 1008x756 colour photograph of a strawberry field.

COORDINATE SYSTEM for every coordinate you report: pixel coordinates, origin (0, 0) at the TOP-LEFT corner of the image, x increasing to the RIGHT (valid 0..1007), y increasing DOWNWARD (valid 0..755). Report whole numbers of pixels.

TASK
Produce a complete inventory of the strawberries in this image - each instance is one strawberry (one whole berry, not its seeds). For each one, give its bounding box:
- "bbox": [x1, y1, x2, y2] - the bounding box of the WHOLE fruit. For a partly hidden fruit, give the box the fruit would occupy if the occluder were not there, not just the visible sliver. Box ONLY the fruit body: do not extend the box to cover the calyx (the sepal leaves) or the stem when they sit apart from the fruit - a calyx that lies flat against the fruit is of course inside the box.
- "confidence_pct": 0-100. How sure you are that this really is a strawberry.

Report EVERY strawberry you can see, whatever its colour. This is an inventory, not a picking decision: do NOT filter by ripeness and do NOT omit a fruit just because it is green, small, or partly hidden. Partly occluded fruit matters as much as fully visible fruit - include a fruit if you can see any part of it. Order the list from largest to smallest apparent size. Do not invent fruit that is not there.


Each fruit must carry EXACTLY the two fields listed above - no more, no fewer, none renamed. If you have nothing for a field, still include it.

Reply with ONLY a JSON object and nothing else - no prose, no code fences:
{"fruits": [{"bbox": [x1, y1, x2, y2], "confidence_pct": 0}, ...]}
```

## Per-dataset parameterisation

Three slots vary by source (scene line, fruit nouns/instance definition,
ripe-colour anchor); everything else is byte-identical. Naming the target
fruit is part of the task definition — each dataset's protocol detects one
known class.

| source | scene | instance | ripe-colour anchor (redness_pct) |
| --- | --- | --- | --- |
| `strawdi` | a strawberry field | strawberry (one whole berry, not its seeds) | red, judged by hue and saturation together |
| `minneapple` | an apple orchard | apple (one whole fruit, not a cluster or a tree) | the red blush of a ripe apple |
| `wgisd` | a vineyard | grape cluster — count the BUNCH, not the individual berries | the variety's mature colour (deep red-purple / yellow-green) |
| `acfr_apples` | an apple orchard at night, lit by artificial illumination | apple | the red blush of a ripe apple |
| `acfr_mangoes` | a mango orchard at night, lit by artificial illumination | mango | the yellow-red blush of a ripe mango |
| `kfuji` | a Fuji apple orchard | apple | the red-striped blush of a ripe Fuji |
| `acfr_almonds` | an almond orchard; the fruit is the green-ish husked almond hanging on the tree (NOT blossoms, leaves or a shelled nut) | almond fruit — the whole green/greyish hanging hull | how far the fuzzy green hull has turned dry brown-grey and split |

Render any variant for any source with:

```bash
python3 paper_study/run_fruit_eval.py --format <full9|seg3|box2> --dry-run
# or programmatically:
python3 -c "import sys; sys.path.insert(0,'paper_study'); \
from paper_study.lib.prompt import build_prompt; \
print(build_prompt('seg3', 1280, 853, 'wgisd'))"
```

Changing any prompt text shifts the harness fingerprint (prompt.py is in
its glob) — old runs keep their recorded prompt and stay valid; new runs
carry the new identity.
