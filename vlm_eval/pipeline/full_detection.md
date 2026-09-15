# full_detection — full-scene strawberry inventory + pick nomination

Standalone, **model-agnostic** spec of the first perception pipeline. More
pipeline docs will live in this `pipeline/` subfolder; each one defines what
the pipeline does, its input/output contract, and how to run it through the
harness — independently of which model executes it.

Implementation lives in the parent directory (`prompts.py`, `run_vlm_eval.py`,
`schema/inventory_schema.json`); [`../AGENTS.md`](../AGENTS.md) remains the
binding runbook for the harness's measurement invariants.

---

## 1. What this pipeline does

One image in → one complete, **unbiased** inventory of **every** strawberry
visible in it, plus a single nominated pick target with its grasp point. The
model works from pixels alone: **no detector, tracker, segmenter or any other
model participates** — that is the point of the pipeline.

"Unbiased" has a precise meaning here:

* every fruit is reported, whatever its colour — red, pink, pale, green;
* partly occluded fruit is included (a fruit is listed if *any* part shows);
* there are **no ripeness categories** — redness and occlusion are continuous
  0–100 numbers (63, not "ripe");
* the pick nomination (`target_index`) is made **after and separately from**
  the census, so the perception output is never filtered by the decision.

## 2. Input / output contract

**Input:** one RGB frame. Sources are normalised to max edge ≤1280 px (a
hard ceiling: silent downscaling would shift the coordinate frame). Current
default source: the unlabelled multi-strawberry `shunba` scenes (qualitative
only). With `--sources validation occluded shunba`, ground-truthed SROI frames
are included and the nomination becomes scoreable.

**Output:** per frame, one JSON object:

```json
{"strawberries": [ { ...one object per fruit... }, ... ],
 "target_index": 0}
```

Each fruit object carries **exactly nine fields, no more, no fewer, none
renamed** (extra or missing fields reject the whole reply):

| field | type | meaning |
| --- | --- | --- |
| `bbox` | 4 ints `[x1,y1,x2,y2]` | whole fruit incl. hidden parts (the box it would occupy without the occluder) |
| `redness_pct` | int 0–100 | continuous: how red the **visible** surface is |
| `occlusion_pct` | int 0–100 | continuous: how much of the fruit is hidden |
| `calyx_visible` | bool | green sepal ring visible |
| `peduncle_visible` | bool | stem visible where it meets the fruit |
| `graspable` | bool | could be picked right now, judging only from what is visible |
| `picking_point` | `[u,v]` or `null` | grasp point on the peduncle just above the calyx; `null` only when the peduncle is hidden |
| `confidence_pct` | int 0–100 | confidence this is a strawberry |
| `description` | string (optional value) | 1–2 sentences in the model's own words |

Coordinates: pixel coordinates, top-left origin, x right, y down, whole
numbers, listed largest-to-smallest apparent size. `target_index` is the
0-based index of the one fruit to pick now (`-1` if none).

**Prompt:** `inventory_plain` in `prompts.py` (verbatim, frame-size
parameterised, printed in every report — prompt text is never a hidden
variable).

**Run statuses:** `ok` (parsed, in-frame) · `no_pick_point` (valid inventory,
nominated fruit has no visible peduncle → answer kept, no point scored) ·
`schema_invalid` / `parse_error` / `out_of_frame` / `empty` / `refused` /
`exec_error` (explicit failures, never silently treated as success).

**Artefacts per run** (`vlm_eval/runs/<timestamp>-<model>-<effort>-<cli>-vlm_eval[-tag]/`):
`report.md`, `responses.jsonl` (raw + parsed answers, usage, prompt),
`metrics.csv`, per-frame overlays + contact sheets (redness-ramp boxes, cyan =
nominated target; overlays and sheets are saved as JPG quality 90 since
2026-09-15, model inputs stay PNG), `manifest.json` + `catalog.json`
provenance snapshots.

## 3. How to run it (any model, any CLI harness)

```bash
# via Claude Code (any vision model routed through it, e.g. glm-5.3-flash):
python3 vlm_eval/run_vlm_eval.py --provider claude --claude-model <slug> --jobs 3 --tag full

# via Codex (whatever codex is configured with):
python3 vlm_eval/make_catalog.py        # vision-enabled catalog override first
python3 vlm_eval/run_vlm_eval.py --jobs 3 --tag full

# acceptance gate — must print PASS:
python3 vlm_eval/verify_run.py vlm_eval/runs/<run dir>
```

Only `inventory_plain` runs by default → 10 calls on the default shunba scope
(+1 synthetic control). To score the nomination against ground truth, add
`--sources validation occluded shunba` (15 calls; 5 scored).

Non-negotiable gates regardless of provider (see AGENTS.md §1 for the why):

1. **Vision-delivery control first** — a synthetic image the model must read
   (code + shape centres). If it fails, the report says `Vision delivery: NO`
   and every number in it is invalid.
2. **All tool surfaces off** — the model must not be able to read anything from
   disk; any tool attempt fails acceptance.
3. **No annotated image is ever an input.**
4. **Inputs ≤1280 px max edge.**
5. **Provenance** — every report states harness, model, effort; the run
   directory name carries `<model>-<effort>-<cli>`.

## 4. Reference points (examples, not the contract)

| date | model (cli) | calls | control | ok / no_pick_point / schema-invalid | scored nomination (median err) |
| --- | --- | --- | --- | --- | --- |
| 2026-09-14 | glm-5.3-flash (claude) | 10 | 1.0 px | 8 / 2 / **0** | n/a (shunba-only) |
| 2026-09-14 | glm-5.3-flash (claude) | 75-run subset | 1.0 px | — | 16.2 px (`inventory_plain`, 5 GT frames) |
| 2026-09-14 | deepseek-flash (codex) | 75 | 1.4 px | 71 / 1 / 0 (all styles) | 23.8 px (`inventory_json`, same frames) |

Typical cost per frame on a flash-tier model: ~3k input + 5–10k output tokens
(the census describes every fruit). Empty or over-long scenes can exhaust the
output budget; the harness retries once at reduced effort and records
`fallback_used`.

## 5. Known behaviours

* `no_pick_point` on sparse/unripe scenes (e.g. sb07, sb10) is a **valid
  answer** — the model found fruit but reported no visible peduncle on its
  nominee — not a failure, and never averaged in as zero.
* A scene with genuinely no fruit reports `0` fruit; that is kept as 0, distinct
  from "could not parse".
* The exact-nine-fields rule is enforced by JSON Schema
  (`additionalProperties: false`); models that decorate fruit with extra note
  fields are rejected wholesale — by design, and the prompt now says so
  explicitly.
