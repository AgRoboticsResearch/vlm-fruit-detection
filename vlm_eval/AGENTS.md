# VLM raw-capability eval — agent runbook

Operational instructions for any agent (Codex, Claude Code, or otherwise) asked
to run or extend the strawberry picking-point VLM eval.

**What this measures:** how well the model that Codex itself runs on localises a
ripe strawberry and its picking point in a raw 640×480 wrist-camera frame, from
pixels alone. No detector, tracker, segmenter or other model participates.

`README.md` in this directory explains *what and why* for a human reader. This
file is the *how*, and it is binding: the invariants below are the difference
between a valid measurement and a plausible-looking invalid one.

---

## 0. Before you start

### Current evaluation scope: shunba only

**Only the unlabelled `shunba` multi-strawberry frames are evaluated.** The SROI
sources (`validation`, `occluded`) are out of scope for now and are not run by
default — `DEFAULT_SOURCES` in `run_vlm_eval.py` is `("shunba",)`.

Two consequences to keep in mind:

* **The default batch is entirely qualitative.** The shunba frames have no ground
  truth, so there is no error, PCK, IoU or sensitivity number to compute. The
  report says so explicitly in its `## Scored metrics` section rather than showing
  empty tables, and the verdict calls the results qualitative.
* **Scored metrics need the SROI frames back.** They are the only source with a
  known picking point. Restore them with
  `--sources validation occluded shunba` (the manifest still builds them, and the
  `single_example` / `single_grid` / `single_strict` / `single_reasoned`
  prompts only have frames to run on when it does). To stop touching that data
  entirely, build with `python3 vlm_eval/build_manifest.py --skip-sroi`.

Do not "fix" the empty scored sections by inventing ground truth for shunba.

### Escalation

`codex exec` will not run inside a read-only sandbox — it writes session/log
files under `~/.codex` and needs network. Expect to run it with escalated
permissions (e.g. `sandbox_permissions: require_escalated` in Codex, or an
approved Bash rule in Claude Code). Symptom if you forget:

```
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

Always redirect stdin (`< /dev/null`). Otherwise the CLI prints `Reading
additional input from stdin...` and can block waiting for a pipe.

### External paths that must exist

The harness reads data outside this repo. If any are missing, fix the path via
the relevant CLI flag rather than editing code:

| what | path | used by |
| --- | --- | --- |
| upstream ground-truth module | `/mnt/data0/code/sroi/sroi_rosbag_utilities/target_ref.py` | `build_manifest.py --target-ref-root` |
| camera↔gripper extrinsics | `/mnt/data0/code/sroi/sroi_rosbag_utilities/configs/camera_gripper_extrinsics_sroi_v2_d405.json` | resolved by `target_ref` |
| validation frames | `/mnt/data1/sroi/sroi_v2/sroiv2_strawberry_picking_lab/validation_pngs/validation_20260714_160922-png` | `--validation-root` |
| occluded frames | `/mnt/data1/sroi/sroi_v2/sroiv2_strawberry_picking_lab/20260803-occluded-cases-pngs` | `--occluded-root` |
| sensitivity sweep | `/mnt/data1/projects/target_condition_sb_picking/viz/target_move/target_move_ep0_frame20.json` | `--sensitivity` |
| unlabelled multi-strawberry scenes | `/mnt/data1/strawberry_robot/shunba_sb_data/images` (no GT) | `--shunba-root`, `--shunba-k` |
| user model catalog | `$CODEX_HOME/cc-switch-model-catalog.json` (default `~/.codex/`) | `make_catalog.py --source` |

Python 3.12 with `numpy`, `pandas`, `Pillow`, `matplotlib`, `jsonschema`.
No GPU. The repo is reachable as both `/mnt/data0/code/...` and
`/home/zfei/code/...` (`/home/zfei/code` is a symlink to `/mnt/data0/code`).

---

## 1. The eight invariants — do not break these

Every one of these exists because breaking it silently produces believable but
worthless numbers. Each was verified empirically; do not re-derive them from
first principles or "simplify" them away.

### I1. Images only arrive if the catalog override is applied

`codex exec -i` **silently drops** the image when the model catalog declares the
slug as text-only. The model then answers confidently from nothing. Measured:

| catalog passed | control-image result |
| --- | --- |
| user catalog (text-only slug) | *"the image content wasn't passed to me … image content was omitted because I don't support image input"* |
| `model_catalog_vision.json` | reads `6305`, circle `[518,81]` (true `[520,80]`) |

Therefore:

* Always let the harness pass `-c model_catalog_json=<override>`. Never drop it.
* **Regenerate the override whenever the user's catalog changes** — the model
  slug is not stable. It already changed once mid-project
  (`deepseek-v4-flash` → `deepseek-flash`), leaving the committed override
  without the slug that was actually being run. That run still *appeared* to work
  because codex fell back to generic model metadata which happened to permit
  images; do not rely on that fallback. The harness now hard-errors instead.
* `run_vlm_eval.py` hard-errors if the slug is missing or text-only. If you see
  that error, run `make_catalog.py`; do **not** bypass the check.

### I2. Every batch must start with the vision-delivery control

A synthetic image (code `6305`, red circle at `[520,80]`, green square at
`[200,400]`) is run first. It is the gate: if it fails, the report must say
`Vision delivery: NO` and label the capability numbers invalid.

Never use `--skip-control` for a reported result. It exists only for debugging
the harness itself, and it marks output as unverified.

### I3. All tool surfaces stay disabled

Given a shell, this model will read the answer off disk. Observed directly: it ran
`python3 -c "...open('.../control_6305.png')"` and derived the answer from the
file. On real frames it can read `CameraTrajectory*.txt` +
`camera_info_color.json` from the episode directory and reproduce ground truth
**exactly**.

The harness passes `--disable` for `shell_tool`, `unified_exec`, `view_image`,
`browser_use`, `browser_use_external`, `computer_use`, `image_generation`,
`tool_suggest`, plus `--sandbox read-only`, and runs in the empty
`vlm_eval/agent_cwd/`.

* Seeing `ERROR codex_core::tools::router: error=unsupported call: exec` in raw
  CLI output is *expected and good* — it means the block worked.
* A non-empty `tool_attempts` field in any record means the model tried. That is
  recorded, and `verify_run.py` fails the run if it happens.

### I4. Never feed the model an annotated image

Upstream ships a pre-rendered `target_ref` marker image. It leaks the answer and
must never be an input. The only annotated image the model ever sees is the
few-shot exemplar composite, which is drawn on an episode **outside** the eval
set (currently `validation …/episode_002`).

If you add a style that draws anything, verify the annotation cannot be read as
the answer.

### I5. Ground truth is recomputed, never hard-coded

`build_manifest.py` imports upstream `target_ref` and recomputes every point:
the gripper-tip pose at the episode's **last** trajectory frame, projected
through the initial camera pose and intrinsics into the **initial** image.

Never paste coordinates into the manifest or code. `verify_run.py` re-derives
each point and fails on any drift.

### I6. Inputs must stay inside the CLI's no-resize ceiling

`codex exec` delivers an image byte-for-byte **only while it is small enough**.
Above that it silently downscales, which moves the model's coordinate frame away
from the frame the harness draws on — and on an unlabelled frame you cannot
detect that from the scores. Measured with a local capture server:

| input | on the wire |
| --- | --- |
| 640×480, 1280×720, 1280×960, 1470×600, 1920×1080 | unchanged |
| 2048×1536 | → 1824×1368 |
| 2560×1440 | → 2048×1152 |
| 4032×3024 | → 1824×1368 |

So `build_manifest.py` normalises new sources to **max edge 1280** and records the
*normalised* size as `image_shape_hw`, and `run_vlm_eval.py` refuses to run if any
input exceeds `MAX_SAFE_EDGE`/`MAX_SAFE_PIXELS` (1920 / 1920×1080 — the largest
size verified to pass through untouched). If you add a source, normalise it and
re-verify with a capture server rather than assuming.

### I7. Labelled and unlabelled frames never mix

The batch contains two kinds of frame:

* **scored** — five wrist-camera frames with exact geometric ground truth
  (`has_gt: true`); the active prompt is `inventory_plain`, whose **nominated
  target** is what gets scored. The archived `single_*` styles also run here
  when selected explicitly.
* **unlabelled** — ten frames from `shunba_sb_data` with **no ground truth of any
  kind** (`has_gt: false`), each holding ~4–13 ripe fruit.

Unlabelled runs must carry **no** error, PCK, IoU or sensitivity value, must not
appear in the style/frame metric tables, and must have no ground-truth
coordinates attached. `verify_run.py` fails the run if any such value leaks.

A style's applicability is declared on the style (`applies_to`) and its answer
shape (`output_shape`), not inferred from the frame — that is what stops a
single-target reply on a multi-fruit frame from being validated against the list
schema.

### I8. Every report states its harness, model and reasoning effort

A number is meaningless without knowing what produced it. `report.md` must carry
three explicit lines — **Harness** (name, version, content fingerprint),
**Model**, **Reasoning effort** — and every record in `responses.jsonl` carries
`harness`, `harness_version`, `harness_fingerprint`, `model` and
`reasoning_effort`. `verify_run.py` fails a report that omits any of them.

The fingerprint is a SHA-256 over the harness sources (`*.py`, `lib/*.py`) and
`schema/*.json`, so it identifies the exact code that generated the numbers.
Docs are deliberately excluded from it. If a report is rebuilt from old records
by newer code, the header says so (`⚠ artefacts rebuilt with harness fingerprint
…`) rather than stamping the old run with the new identity.

The **run directory name** carries the same identity, so it is readable without
opening anything:

```
runs/<timestamp>-<model>-<effort>-<cli>-<harness>[-<tag>]
runs/20260914-171146-deepseek-flash-high-codex-vlm_eval-full/
runs/20260914-192928-glm-5.3-flash-default-claude-vlm_eval-full/
```

`effort` is `default` when the model's configured effort is used, and `cli` is
the agent harness that talked to the model (`codex`, `claude`, `kimi`, …) — the
same model through a different CLI is a different measurement, so the CLI is
part of the run's identity. Components are slugified so an odd model slug
cannot escape the directory. `--rebuild-report` does not depend on the name (it
takes the path via `--out`), so runs created before this convention still
verify.

---

## 2. Canonical workflow

Run from the repo root (`/mnt/data0/code/sroi/strawberry_detection`).

```bash
# 0. See what prompts exist; pick the ones you want (by name OR by title)
python3 vlm_eval/run_vlm_eval.py --list-styles

# 1. Regenerate the vision-enabled catalog (do this after ANY catalog change)
python3 vlm_eval/make_catalog.py

# 2. Recompute ground truth, copy frames, derive inputs, write manifest.json
python3 vlm_eval/build_manifest.py

# 3. Inspect the exact prompts and the plan WITHOUT calling the model
python3 vlm_eval/run_vlm_eval.py --dry-run

# 4. Cheap smoke test (1 control + 1 call) before spending the batch
python3 vlm_eval/run_vlm_eval.py --limit 1 --styles inventory_plain --tag smoke

# 5. The batch. Only inventory_plain is active, so the default scope is
#    shunba only -> 10 frames x 1 prompt = 10 calls, all qualitative.
python3 vlm_eval/run_vlm_eval.py --jobs 3 --tag full

#    ...to restore the ground-truthed SROI frames and the scored metrics
#    (the nominated target of the inventory is scored against ground truth):
python3 vlm_eval/run_vlm_eval.py --sources validation occluded shunba --jobs 3

#    ...or select exactly the prompts you want, by name or title (this is also
#    how you run an archived prompt):
python3 vlm_eval/run_vlm_eval.py --styles inventory_plain inventory_scan_first --jobs 3
python3 vlm_eval/run_vlm_eval.py --styles "Full inventory + target (scan first)" \
                                 --sources shunba --jobs 3 --tag inventory

# 6. Acceptance gate — must print PASS
python3 vlm_eval/verify_run.py vlm_eval/runs/<timestamp>-full
```

Delete smoke run directories before handing off, so only the real batch remains.

### Prompt library

Prompts are selected with `--styles` by **name or title** (case-insensitive,
unique prefix accepted); omit the flag to run all of them. `--list-styles` prints
the catalogue.

> **Renamed 2026-09-14** to grouped, self-describing names: the prefix says what
> answer you get (`single_*` = one target, `list_*` = flat list,
> `inventory_*` = full inventory). The old names (`baseline_json`,
> `strict_schema`, `reasoning_first`, `few_shot_exemplar`, `grid_overlay`,
> `enumerate_strawberries`, `inventory_json`, `inventory_reasoning`) still
> resolve via `STYLE_ALIASES`, and `--rebuild-report` normalises them — but run
> reports predating the rename keep the names they were recorded with.

**Since 2026-09-14 only `inventory_plain` is active** — it is the pipeline's
single prompt, and that pipeline is specified standalone (model-agnostic) in
[`pipeline/full_detection.md`](pipeline/full_detection.md); further pipeline
docs will live in that `pipeline/` subfolder. The other seven prompts are
**archived** (`ARCHIVED_STYLES` in
`prompts.py`): excluded from default batches, but still runnable by naming them
explicitly in `--styles`, and still resolvable by `--rebuild-report`.

| name | status | runs on | answer | what it asks |
| --- | --- | --- | --- | --- |
| `inventory_plain` | **active** | any | inventory | unbiased full inventory + nominated target |
| `single_basic` | archived | any | single | minimal bbox + picking point |
| `single_strict` | archived | gt | single | explicit ranges, `--output-schema` enforced |
| `single_reasoned` | archived | gt | single | grounds sepals → fruit → peduncle → grasp point first |
| `single_example` | archived | gt | single | annotated example panel beside the query |
| `single_grid` | archived | gt | single | reads coordinates off a labelled 64 px grid |
| `list_ripe_only` | archived | no_gt | list | **biased** legacy prompt: ripe fruit only |
| `inventory_scan_first` | archived | any | inventory | same as `inventory_plain`, but scans the scene in bands first |

The two `inventory_*` prompts are the unbiased ones. They report **every**
strawberry, whatever its colour, including partly occluded ones, and are
explicitly told not to filter by ripeness. Per fruit they return:

* `redness_pct` and `occlusion_pct` as **continuous 0–100 numbers** — there are
  deliberately **no ripeness categories** (no "ripe"/"unripe" label). The prompt
  gives anchors across the range and asks for values like 63 rather than 60.
* `calyx_visible`, `peduncle_visible`, `graspable`, `confidence_pct`,
  `picking_point` (null when the peduncle is hidden).
* `description` — one or two sentences of the model's own words about that fruit.

plus a `target_index` nominating the single fruit it would pick now (`-1` if
none). Because the nomination is separate from the inventory, these prompts are
**scoreable on the ground-truth frames too**: the nominated target's picking point
is compared against ground truth, exactly like the single-target styles, while
the inventory itself stays unbiased.

When an inventory parses but offers no usable grasp point (the model found the
fruit but reported the peduncle as hidden), the run gets status
`no_pick_point` — a valid answer that yields no number, not a failure, and not
something to be quietly averaged in as a zero.

### Expected cost and shape

**Default scope (shunba only, `inventory_plain` only): ~10 model calls** —
measured on glm-5.3-flash: ~3k input + 5–10k output tokens per call, ~4 min at
`--jobs 3`, ≈$0.05–0.15 total. The inventory prompt is the expensive one because
it describes every fruit in the scene.

With the SROI frames restored (`--sources validation occluded shunba`) it is ~15
calls (10 qualitative + 5 whose nominated target is scored). The control adds one
call. Reasoning effort follows the provider's configured default unless
`--reasoning-effort` is passed.

### What a good run looks like

```
control : DELIVERED  code='6305' (expected '6305')  circle_err=1.0px  tool_attempts=[]
[10/10] inventory_plain__sb10                      ok            err=     –px tok=5151

done in ~120s -> .../vlm_eval/runs/<timestamp>-full
```

Every row must end `ok` or `no_pick_point`; unlabelled rows show `–px` for error,
which is expected (there is nothing to score them against). Expected headline
values (five independent batches): median picking-point error **13–17 px**
(~2–2.7% of frame width) on the single-target styles, 100% parse and
schema-valid rates, zero out-of-frame, zero tool attempts. (Those headline
numbers come from batches that still ran the single-target styles; the default
pipeline now runs `inventory_plain` only, whose nominated-target errors on the
SROI frames measured 4–20 px on glm-5.3-flash.)

---

## 3. Acceptance criteria

A batch is only reportable when all of these hold. `verify_run.py` enforces them
(19–29 checks depending on whether the batch contains unlabelled runs; must print
`PASS`):

* `report.md`, `responses.jsonl`, `metrics.csv`, `manifest.json`,
  `catalog.json` exist in the run directory.
* The report states the harness (name/version/fingerprint), the model, and the
  reasoning effort, and they agree with the records (I8).
* Every run has a known status — parsed (`ok`) or an explicit failure
  (`empty`, `refused`, `parse_error`, `exec_error`, `schema_invalid`,
  `out_of_frame`). A missing or empty response is **never** silently a success.
* Every run has an overlay image (JPG, PNG in older runs) that decodes, and a
  token count.
* Every unscored run is explicitly out-of-frame or failed.
* No run has tool attempts.
* The report states the vision-delivery verdict, and it agrees with
  `control.json`.
* Manifest ground truth still reproduces from `target_ref`.
* The rendered marker's top edge sits on the picking point (±1 px).

Two robustness properties worth knowing before you change the harness:

* **A model call is never wasted.** Each finished run is appended to
  `responses.partial.jsonl` as it completes, and any exception inside a single
  run is recorded as an `exec_error` record instead of killing the batch. A crash
  late in a 75-call batch costs you the remaining calls, not the whole batch.
* **Rendering bugs cannot lose answers.** If drawing an overlay throws, the
  record keeps the model's answer and notes `render_error`.

---

## 4. Reading the report

`runs/<timestamp>/report.md` sections: verdict banner, control table, metrics by
style, metrics by frame, every run, tokens/time, error histogram, overlays,
sensitivity bridge, non-scoring runs, exact prompts, method and provenance.

**Overlay legend** — the cyan circle trips people up:

| marking | meaning |
| --- | --- |
| cyan **circle** | *not* the answer and not a measurement. The upstream rough 60×60 fruit box, centred 30 px **below** the picking point so its top edge passes through it. It backs the approximate bbox-IoU metric and is never shown to the model. |
| cyan **crosshair** | the ground truth itself. Every error number is measured from this point. |
| magenta **ring + dot** | the model's predicted picking point (error labelled in px). |
| amber rectangle | the model's predicted bounding box. |
| white line | error vector. |

Ground truth is drawn last so a near-perfect prediction cannot hide it.

Machine-readable companions: `responses.jsonl` (one record per run: raw response,
parsed JSON, usage, prompt, statuses), `metrics.csv` (flat table),
`summary_by_style.csv`, `manifest.json` and `catalog.json` snapshots.

---

## 5. Common tasks

### Change the eval set size

```bash
python3 vlm_eval/build_manifest.py --validation-k 8 --occluded-k 2
python3 vlm_eval/build_manifest.py --shunba-k 20 --shunba-max-edge 1280
```

Selection is deterministic: GT-valid episodes in sorted order, evenly spaced,
with indices/seed/episode list recorded in `manifest.json`. Scaling up costs
approximately `k_total x 5` model calls (plus `k_unlabelled x 2` for the
unlabelled source, which never scales with the scored styles).

### Evaluate a different model

```bash
python3 vlm_eval/make_catalog.py            # picks up whatever the user catalog now has
python3 vlm_eval/run_vlm_eval.py --model <slug> --jobs 3 --tag <name>
```

If the configured model is absent from the cc-switch catalog but present in
Codex's `models_cache.json`, use that catalog explicitly. Already vision-enabled
entries are preserved unchanged:

```bash
python3 vlm_eval/make_catalog.py --source /home/zfei/.codex/models_cache.json
python3 vlm_eval/run_vlm_eval.py --jobs 3 --tag full \
  --catalog-source /home/zfei/.codex/models_cache.json
```

`--catalog-source` records the generation source for provenance; the harness
still passes and validates `model_catalog_vision.json`, and the vision control
is still required.

### Rebuild a report without new model calls

```bash
python3 vlm_eval/run_vlm_eval.py --rebuild-report --out vlm_eval/runs/<run dir>
```

Use this after changing *rendering or reporting* code. It re-renders every
overlay from the stored records (and clears stale overlays/contact sheets), so a
rebuilt report can never disagree with the numbers it prints.

### Claude Code provider (validated 2026-09-14)

`--provider claude` runs the batch through the `claude` CLI in headless print
mode instead of `codex exec` — same manifest, prompts, control and acceptance
gate:

```bash
python3 vlm_eval/run_vlm_eval.py --provider claude --sources validation occluded shunba --jobs 3 --tag full
```

How the invariants map on this path:

* **Delivery (I1/I6):** the frame is a base64 image content block in the one
  stream-json user message written to stdin (`--input-format stream-json`).
  There is no codex catalog override here; the synthetic control is the
  delivery gate, and `catalog.json` in the run dir records the claude routing
  (endpoint, slug, CLI version) instead of a catalog hash.
* **The slug decides delivery server-side.** The flagship `glm-5.3` slug
  **rejects image content** on open.bigmodel.cn with this account's token
  (native API: `messages.content.type` restricted to `['text']`; the
  Anthropic-compat endpoint silently drops the image block and the model
  answers blind — measured 2026-09-14, which is also why images pasted into a
  Claude Code session on this token come back as CDN URLs). **`glm-5.3-flash`
  is the natively multimodal GLM-5.3** and is the provider's default; passing
  the flagship slug prints a warning and the control will (correctly) fail the
  batch as `Vision delivery: NO`.
* **Tools (I3):** `--tools ""` (no tool exists at the API level, so a tool call
  cannot even be generated), `--safe-mode` (no hooks/plugins/skills/CLAUDE.md),
  `--strict-mcp-config` (no MCP servers), `--no-session-persistence`, empty
  `agent_cwd`. The harness additionally scans the stream for `tool_use` blocks
  and permission denials; the CLI's internal `StructuredOutput` tool that
  implements `--json-schema` is excluded from that scan as harness machinery.
* `single_strict` and the control use `--json-schema` (CLI-enforced structured
  output), the analogue of codex `--output-schema`.
* First validated batch:
  `runs/20260914-192928-glm-5.3-flash-default-claude-vlm_eval-full` — 75 calls,
  control 1.0 px, `verify_run.py` PASS, median 13.8 px / PCK@10 17 %.

### GLM comparison via codex (wired, not yet exercised)

`--provider glm` reroutes to `https://open.bigmodel.cn/api/v1`
(`wire_api=responses`), reading the key from `ZHIPU_API_KEY`. It needs a catalog
entry for the GLM slug; without one it errors early. Treat it as unvalidated.

### Antigravity via agy (validated on StrawDI, 2026-09-16)

`--provider agy` runs Google's Antigravity CLI headless (`agy -p`,
`--output-format stream-json`, default model `gemini-3.8-flash-high`). It breaks
the usual invariant shape in one fundamental way, and every agy report carries
the caveat: agy has **no headless image attachment** and **no API-level
tool-off**. Delivery goes through the built-in `view_file` tool — the frame is
staged as the only file of a per-call temp workspace (no `.agents` up-tree), the
prompt names the exact path, and the model must call `view_file` once; the event
stream is scanned and any other tool step or denied action fails the record.
Measured: `view_file` shows the model the image **resampled to 800×600**, so
prompts speak the 800×600 space (`prompt_frame_size`) and reported coordinates
are scaled back (`agy_scale_point` / `agy_scale_box`) before control checks and
scoring. The synthetic control gates that whole chain per run. Do not compare
agy IoU numbers against codex/claude runs without quoting this handicap.

---

## 6. Troubleshooting

| symptom | cause | action |
| --- | --- | --- |
| model says the image "was omitted because I don't support image input" | catalog override missing or stale after a catalog change | re-run `make_catalog.py`; confirm the harness passed `-c model_catalog_json=` |
| `model '<slug>' is not in model_catalog_vision.json` | user catalog changed | `make_catalog.py` |
| `does not declare the 'image' input modality` | override built from a text-only catalog | `make_catalog.py`; do not bypass |
| `failed to initialize in-process app-server client: Read-only file system` | sandboxed | re-run `codex exec` escalated |
| `Reading additional input from stdin...` | stdin not redirected | harmless; append `< /dev/null` |
| `ERROR ... unsupported call: exec` in the raw CLI log | tools blocked — expected | none; but confirm `tool_attempts` is empty in the records |
| `status: empty` | reasoning exhausted the output budget | the harness auto-retries once at reduced effort and records `fallback_used`; if it still fails, report it, don't hide it |
| `Model metadata for '<slug>' not found` warning | slug absent from catalog, so generic fallback metadata is used | regenerate the override; do not rely on fallback behaviour |
| matplotlib "config dir is not writable" warning | `~/.config/matplotlib` unwritable | harmless; the harness sets `MPLCONFIGDIR` |
| `verify_run.py` fails on marker geometry | GT convention or rendering changed | check `gtbridge.ROUGH_BOX_OFFSET`/`ROUGH_BOX_HALF` and `imaging.draw_gt_marker` |
| overlay markers land in the wrong place on a new image source | the CLI downscaled that image (see I6) | normalise to max edge 1280 and re-verify with a capture server |
| `input image(s) too large for the CLI's no-resize ceiling` | a manifest input exceeds the safe box | rebuild with a smaller `--shunba-max-edge` |
| unlabelled run reports `0` fruit | the model found no ripe strawberry (see `sb07` in the shipped batch) | expected sometimes; inspect the overlay, do not force a number |

---

## 7. Deliberate non-goals

Do not add these back without an explicit request:

* **Repeat/variance runs.** The `--repeat` flag was removed at the user's
  request. The model *is* nondeterministic — identical calls on one frame were
  observed to differ by ~10 px — so treat style gaps of that order as noise, but
  do not reintroduce a variance harness by default.
* **Any detector / tracker / segmenter in the loop.** The whole point is raw
  capability. `weights/yolov11-m-best.pt` and the `infer_*.py` scripts in the
  repo root are a separate, unrelated pipeline.
* **Automatic GLM comparison.** Wired but unvalidated.

---

## 8. File map

| path | role |
| --- | --- |
| `AGENTS.md` (this file) | agent runbook; `CLAUDE.md` is a symlink to it |
| `README.md` | human-facing what/why, findings, caveats |
| `pipeline/full_detection.md` | standalone model-agnostic spec of the active pipeline (full-scene inventory + pick nomination); first of a per-pipeline doc series |
| `run_vlm_eval.py` | the harness: runs, parses, scores, writes the report |
| `build_manifest.py` | recomputes ground truth, copies frames, derives images |
| `verify_run.py` | asserts the acceptance criteria on a finished run |
| `make_catalog.py` | regenerates the vision-enabled catalog override |
| `prompts.py` | `inventory_plain` (the one active pipeline prompt) + seven archived styles, verbatim, with applicability and answer shape |
| `lib/gtbridge.py` | ground truth, sample selection, sensitivity bridge |
| `lib/imaging.py` | input preparation (grid, exemplar) and diagnostics (overlays) |
| `lib/parse.py` | JSON extraction, schema validation, status classification |
| `model_catalog_vision.json` | the override passed to `codex exec` |
| `manifest.json` | samples, ground truth, selection provenance, control spec |
| `schema/` | JSON Schemas for the single-target answer, the list answer, and the control |
| `data/frames/` | query frame per sampled episode, plus the normalised unlabelled scenes |
| `data/derived/` | grid-overlay and exemplar-composite model inputs |
| `data/control/` | the synthetic vision-delivery control |
| `agent_cwd/` | empty working directory for `codex exec` — keep it empty |
| `runs/<timestamp>/` | report, overlays, contact sheets, JSONL, CSVs |
