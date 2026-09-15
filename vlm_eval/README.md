# VLM raw-capability eval — strawberry detection + picking point

Measures how well the model that Codex itself runs on (currently
`deepseek-flash`) localises a strawberry and its picking point in a raw 640×480
wrist-camera frame — from the pixels alone, with **no detector, tracker,
segmenter or other model anywhere in the loop**.

> **Current scope: shunba only.** The unlabelled multi-strawberry frames from
> `shunba_sb_data` are the only source evaluated at the moment. The SROI
> wrist-camera sources (`validation`, `occluded`) are still built and fully
> supported, but are not run unless you ask
> (`--sources validation occluded shunba`). Because shunba has no ground truth,
> the default batch is **entirely qualitative** — no error, PCK, IoU or
> sensitivity numbers — and the report says so rather than showing empty tables.

> **Agents: read [`AGENTS.md`](AGENTS.md) instead of this file.** It is the
> binding operational runbook (invariants, exact workflow, troubleshooting).
> This README is the human-facing explanation of what was built and what it found.
> `CLAUDE.md` is a symlink to `AGENTS.md`.

Each run hands the model one prepared image plus one prompt over the `codex exec`
CLI, then scores the answer against ground truth that is derived geometrically
from the demonstration, not annotated by hand.

## Quickstart

```bash
cd /mnt/data0/code/sroi/strawberry_detection

python3 vlm_eval/make_catalog.py          # (re)build the vision-enabled catalog
python3 vlm_eval/build_manifest.py        # ground truth + images + manifest.json
python3 vlm_eval/run_vlm_eval.py --dry-run          # inspect the exact prompts
python3 vlm_eval/run_vlm_eval.py                    # the full 25-call batch
python3 vlm_eval/verify_run.py vlm_eval/runs/<timestamp>   # acceptance checks
```

Useful flags: `--limit`, `--styles`, `--sources`, `--jobs`, `--tag`,
`--model`, `--reasoning-effort`, `--provider {codex,glm}`, `--timeout`,
`--price-in-per-mtok/--price-out-per-mtok`, and `--rebuild-report` (refresh a
finished run's tables and report from its own `responses.jsonl`, no model calls).
`--list-styles` prints the prompt library.

Every report states the **harness** (name, version, code fingerprint), the
**model**, and the **reasoning effort** that produced it, and each record in
`responses.jsonl` carries the same identity — so a number can never be detached
from the code and configuration behind it.

Run directories are named the same way, so a result is identifiable from the path
alone:

```
runs/<timestamp>-<model>-<effort>-<harness>[-<tag>]/
runs/20260914-171146-deepseek-flash-high-vlm_eval-full/
```

## The two things that make the numbers trustworthy

### 1. The images actually arrive

`codex exec -i` only delivers an image when the model catalog says that slug is
multimodal. The user-level catalog declares `deepseek-flash` as `["text"]`, and
the CLI then **silently drops the image** — you get a confident-looking answer
from a blind model. Measured directly:

| catalog | control image result |
| --- | --- |
| user catalog (text-only slug) | *"the image content wasn't passed to me … image content was omitted because I don't support image input"* — no answer at all |
| `model_catalog_vision.json` | reads the code `6305`, red circle `[518, 81]` (true `[520, 80]`), green square `[200, 398]` (true `[200, 400]`) |

`model_catalog_vision.json` is a minimal textual edit of the user's catalog:
exactly the `input_modalities` block is rewritten, everything else is copied
byte-for-byte, and the user's own config is never touched. Regenerate it with
`make_catalog.py` whenever the user-level catalog changes — the harness hard-errors
rather than run if the model slug is missing or text-only.

Every batch therefore starts with a synthetic control image (code `6305` plus a
red circle and a green square at known coordinates). If the model cannot read it,
the report says **Vision delivery: NO**, labels the numbers invalid, and does not
present them as capability.

### 2. The model cannot peek

`codex exec` normally gives the model a shell. In a first attempt at this eval the
model immediately ran `python3 -c "...open('.../control_6305.png')"` and computed
the answer from the file — and for the real frames it could equally well have read
`CameraTrajectory*.txt` and `camera_info_color.json` from the episode directory and
reproduced the ground truth exactly.

Every run therefore disables all tool surfaces (`shell_tool`, `unified_exec`,
`view_image`, `browser_use`, `browser_use_external`, `computer_use`,
`image_generation`, `tool_suggest`) and runs with `--sandbox read-only` in an empty
working directory. A blocked attempt surfaces as `unsupported call: exec`, which
the harness records per run; `verify_run.py` fails if any run shows tool attempts.

### Verified: images arrive unresized — up to a size ceiling

Pointing the CLI at a local capture server and decoding the outgoing request shows
an image on the wire is **byte-for-byte identical** to the source file (same
sha256) at native size with `detail=high` — but only while it stays small enough.
Above a ceiling the CLI silently downscales it, which would put the model's
coordinates in a different frame from the one the harness annotates:

| input | on the wire |
| --- | --- |
| 640×480, 1280×720, 1280×960, 1470×600, 1920×1080 | unchanged (identical sha256) |
| 2048×1536 | downscaled to 1824×1368 |
| 2560×1440 | downscaled to 2048×1152 |
| 4032×3024 | downscaled to 1824×1368 |

So the ceiling sits between 1920×1080 and 2048×1536 (max edge 2048, then an area
cap around 2.5 MP). The harness therefore **normalises every input to max edge
1280** and refuses to run if any image exceeds the verified-safe box
(`MAX_SAFE_EDGE`/`MAX_SAFE_PIXELS` in `run_vlm_eval.py`). Every frame in the
shipped manifest is verified to pass through untouched.

## Layout

| path | role |
| --- | --- |
| `AGENTS.md` | **agent runbook — the operational source of truth** (`CLAUDE.md` → symlink) |
| `manifest.json` | samples, ground truth, selection provenance, control spec |
| `build_manifest.py` | recomputes ground truth, copies frames, derives images |
| `run_vlm_eval.py` | the harness: runs, parses, scores, writes the report |
| `verify_run.py` | asserts the acceptance criteria on a finished run |
| `prompts.py` | the five prompt styles, verbatim |
| `make_catalog.py` | regenerates the vision-enabled catalog override |
| `model_catalog_vision.json` | the override itself (committed, reproducible) |
| `schema/` | JSON Schemas for the answer and for the control |
| `lib/` | ground truth/sensitivity bridge, imaging, JSON parsing |
| `data/frames/` | the query frame of each sampled episode, plus the normalised unlabelled scenes |
| `data/derived/` | the grid-overlay and exemplar-composite inputs |
| `data/control/` | the synthetic vision-delivery control |
| `runs/<timestamp>/` | report, overlays, contact sheets, JSONL, CSVs |

## The example set

**Scores** come from five wrist-camera frames with exact geometric ground truth,
25 scored calls (5 styles × 5 frames):

* **4** GT-valid episodes from `validation_pngs/validation_20260714_160922-png`
  (92 of 100 episodes yield a valid ground-truth point),
* **1** from `20260803-occluded-cases-pngs` (174 of 199 valid),

picked by deterministic even spacing over the sorted GT-valid list — indices
`[0, 30, 61, 91]` and `[87]`. The seed, indices, episode ids and the excluded
episodes' failure statuses all land in `manifest.json`.

**Qualitative results** come from ten frames of a second, unrelated source
(`/mnt/data1/strawberry_robot/shunba_sb_data/images`, 273 JPEGs at 1920×1080 and
4032×3024) which has **no ground truth at all** — no demonstrations, no
trajectories, nothing to project. Each of those frames contains roughly 4–13 ripe
fruit rather than one designated target, so they are normalised to 1280 px on the
long edge and reported in their own section: 20 further calls
(`single_basic` + `list_ripe_only` × 10 frames), **never scored**.

The query image is always the episode's **first** colour frame
(`color_000000.png`). The pre-rendered `target_ref` marker image is never used as
input — it would hand the model the answer — and serves only to sanity-check
ground truth.

## Ground truth

Recomputed at manifest-build time by importing the upstream `target_ref` module
(`/mnt/data0/code/sroi/sroi_rosbag_utilities/target_ref.py`), never hard-coded:
the gripper-tip pose at the episode's **last** ORB-SLAM trajectory frame,
projected through the initial camera pose and intrinsics into the **initial**
image. Demonstrators end every episode holding the gripper above the picked
strawberry, so the final tip pose *is* the picking location.

`verify_run.py` re-derives every manifest ground-truth point and checks the
rendered marker's top edge sits on the picking point (the upstream convention: a
rough 60×60 box centred 30 px below it).

**How to read an overlay** (`runs/<ts>/overlays/*.jpg`, PNG in older runs):

| marking | meaning |
| --- | --- |
| cyan circle | *not* the answer — the upstream rough 60×60 fruit box, centred 30 px below the picking point so its **top edge passes through it**. It is the reference for the approximate bbox-IoU metric, and is never shown to the model |
| cyan crosshair | the ground truth itself: the gripper-tip projection. Every error number is measured from this point |
| magenta ring + dot | the model's predicted picking point, with its error labelled in px |
| amber rectangle | the model's predicted bounding box |
| white line | error vector between prediction and truth |

Ground truth is drawn last so a near-perfect prediction cannot hide the reference.

## Prompt styles

Prompts are a library, selected by name or title with `--styles`
(`--list-styles` prints the catalogue). Since 2026-09-14 **only `inventory_plain`
is active** — it is the pipeline's single prompt and the only one a default batch
runs; the rest are archived (named explicitly in `--styles` they still run, e.g.
for comparisons). The active pipeline is specified standalone in
[`pipeline/full_detection.md`](pipeline/full_detection.md).

| style | status | input | runs on | what it probes |
| --- | --- | --- | --- | --- |
| `inventory_plain` | **active** | raw frame | any | **unbiased inventory** of every fruit + a nominated target |
| `single_basic` | archived | raw frame | any | minimal ask |
| `single_strict` | archived | raw frame | gt | explicit ranges/ordering, schema-enforced output |
| `single_reasoned` | archived | raw frame | gt | grounding sepals → fruit → peduncle → grasp point first |
| `single_example` | archived | 1470×600 composite | gt | in-context visual reference: annotated example panel beside the query |
| `single_grid` | archived | raw frame + 64 px grid | gt | reading coordinates off a labelled grid |
| `list_ripe_only` | archived | raw frame | unlabelled | **biased legacy prompt**: ripe fruit only |
| `inventory_scan_first` | archived | raw frame | any | same as `inventory_plain`, but scans the scene in bands first |

### The unbiased inventory prompts

The two `inventory_*` prompts exist because the earlier enumeration asked only for
ripe fruit, which filters by colour and cannot tell "found nothing" apart from
"everything looked unripe". They instead ask for **every strawberry, whatever its
colour, including partly occluded ones**, with no ripeness categories at all.
Each fruit returns:

* `redness_pct` and `occlusion_pct` as **continuous 0–100 numbers**, with anchors
  across the range, so redness is measured rather than bucketed;
* `calyx_visible`, `peduncle_visible`, `graspable`, `confidence_pct`, and a
  `picking_point` that is `null` when the stem is hidden;
* `description` — one or two sentences in the model's own words about that fruit.

and, separately, `target_index`: the one fruit it would pick right now (`-1` if
none). Because the inventory and the nomination are separate, these prompts are
**also scored on the ground-truth frames** — the nominated target's picking point
is compared against ground truth exactly like the single-target styles — while
the inventory itself stays free of any colour instruction.

The `single_example` composite is the only style whose input is not 640×480.
It carries labelled global-coordinate axes and its prompt states the query
panel's global rectangle explicitly, so the harness converts the answer back with
a known offset instead of making the model do arithmetic. The exemplar episode
(`validation …/episode_002`) is asserted to be outside the eval set.

Each style declares which frames it runs on: the five scored styles need a
single-target frame, `list_ripe_only` needs the multi-fruit scenes, and
`single_basic` runs on both (on a multi-fruit frame its prompt switches to "pick
the ripest, most prominent one"). That is what makes a 15-frame, 45-call batch
instead of an uninformative 90-call one.

## Scoring

Primary: Euclidean picking-point error in pixels, plus signed `dx`/`dy` and error
as a percentage of frame width; PCK@5/10/20 per style and per frame.

Secondary: IoU of the predicted box against the upstream rough 60×60 box, and
whether the predicted box contains the ground-truth point. This is labelled
**approximate** everywhere, because the dataset has no hand-annotated fruit boxes.

Process: parse rate, schema-valid rate, out-of-frame count, empty/refused
responses, tool attempts, tokens (input/cached/output/reasoning) and wall time per
run. An empty reply (reasoning can exhaust the output budget) triggers exactly one
recorded retry at reduced reasoning effort; both attempts are kept in the record.

## Sensitivity bridge

Pixel error is translated into picking-relevant units using a measured
dose-response sweep (`viz/target_move/target_move_ep0_frame20.json`) produced
elsewhere by replaying the policy with perturbed target points. Central
differences give per-pixel rates:

| direction | gripper-endpoint | grip closure |
| --- | --- | --- |
| x | 0.085 mm/px | −0.00009 /px |
| y | 0.139 mm/px | +0.00142 /px |

So a ~16 px picking-point error corresponds to roughly 1–2 mm of gripper-endpoint
displacement. These are local rates from one sweep around one reference point:
indicative, not a calibrated per-episode Jacobian. `--price-in-per-mtok` /
`--price-out-per-mtok` add a cost column; no pricing is assumed otherwise.

## Scaling up / swapping the model

Scaling is a manifest edit: `python3 vlm_eval/build_manifest.py --validation-k 8
--occluded-k 2`. For a different model, regenerate the catalog override and pass
`--model <slug>`.

`--provider glm` reroutes to `https://open.bigmodel.cn/api/v1` (`wire_api=responses`)
with the key read from `ZHIPU_API_KEY`; it is wired up but **not run** in this
round, and it errors out early if the key or a catalog entry for the GLM slug is
missing.

## Caveats

* One batch of 25 calls is small, and the model is not deterministic: identical
  calls on one frame have been observed to differ by ~10 px. Style gaps of that
  order should not be read as real.
* bbox scoring is approximate by construction (see above).
* Ground truth relies on ORB-SLAM trajectories; episodes where the projection
  fails QC are excluded rather than guessed at.
* The catalog override exists because the user-level catalog is currently
  text-only for this slug. If upstream starts declaring image modality, the
  override becomes redundant — the harness will keep working, and will say so
  rather than fail.
