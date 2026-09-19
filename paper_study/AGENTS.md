# paper_study — cross-fruit detection/segmentation study (agent runbook)

The extended-paper version of the blog study: the strawdi_segmentation
nine-field fruit inventory (bbox + visible-surface polygon + seven census
fields) generalised across FRUIT DATASETS, run on a mixed testbed of public
orchard/vineyard benchmarks, glm-5.3-flash with thinking at max.

Structure mirrors the blog_study harnesses deliberately: same invariants
(control first, tools off, no annotated inputs, ≤1280 px, provenance
fingerprints), same provider stack — imported unchanged from
`../blog_study/vlm_eval/` — same scorers (detection from
`../blog_study/strawdi_eval/lib/scoring.py`, masks from
`../blog_study/strawdi_eval/seg/lib/seg_scoring.py`).

## What is measured

Per dataset, ONE number pair on its own test frames: **box F1@IoU-0.5**
(definable everywhere) and **mask F1@IoU-0.5** where the dataset has instance
masks (strawdi id-maps, wgisd npz clusters, minneapple COCO polygons). ACFR
sources are box-only (their apples' pixel masks are semantic, not instance);
polygons are asked but unscored there — same rule as the base vlm_seg
pipeline's unmaskable GT. See `DATASETS.md` for the tracking table (dataset →
task → protocol → metrics → our numbers) and GT conventions.

## Layout

| path | role |
| --- | --- |
| `AGENTS.md` (this file) | runbook |
| `DATASETS.md` | tracking table: datasets, tasks, protocols, metrics |
| `PROMPTS.md` | the three prompt variants documented (full9 / seg3 / box2, verbatim + parameterisation) |
| `FORMAT_COMPARISON.md` | the 20-frames/dataset output-format comparison |
| `build_testbed_manifest.py` | curate 3 frames/dataset from each TEST split, normalise to ≤1280, read + snapshot GT → `manifest.json` + `data/frames/` |
| `run_fruit_eval.py` | the harness: control → per-frame inventory call → parse → score → report |
| `lib/prompt.py` | `fruit_inventory_segmentation` — the strawdi seg prompt, fruit-generic (scene/fruit noun/ripe-colour anchors per dataset) |
| `lib/gtload.py` | per-dataset GT loaders (conventions empirically verified; see its docstring) |
| `schema/fruit_inventory_schema.json` | the nine-field answer contract (`fruits` key) |
| `verify_run.py` | acceptance gate — must print PASS |
| `data/` (gitignored) | delivered frames, control image, GT-check overlays |
| `runs/` (gitignored) | `<ts>-<model>-<effort>-<cli>-paper_fruit_seg[-<tag>]/` |

## Workflow (from the repo root)

```bash
python3 paper_study/build_testbed_manifest.py                 # needs the GLOWAY mount
python3 paper_study/build_testbed_manifest.py --per-source 20 # temporary bigger testbed
python3 paper_study/run_fruit_eval.py --dry-run               # plan + first prompt, 0 calls
python3 paper_study/run_fruit_eval.py --jobs 3 --tag full --reasoning-effort max
python3 paper_study/run_fruit_eval.py --format box2 --tag ...  # output-format variants
python3 paper_study/verify_run.py paper_study/runs/<run dir>   # must print PASS
```

`--per-source N` (N≠3) writes `manifest{N}.json` + `data/frames{N}/` and
leaves the committed 3-frame default untouched — bigger testbeds are
scratch unless promoted deliberately.

`--format {full9,seg3,box2}` selects the per-fruit output field set
(nine-field census / bbox+polygon+confidence / bbox+confidence); task
wording is held constant across formats so runs are comparable
(`FORMAT_COMPARISON.md`). box2 asks no polygons → mask metrics n/a.

The claude CLI child needs network + write outside a read-only sandbox
(escalate). Default model glm-5.3-flash (the multimodal GLM-5.3; the
flagship slug rejects images). Reference runs use `--reasoning-effort max`.

Delete smoke run directories before handing off. Retry-merge failures back
into the originating run per the strawdi seg spec's §7 rule (same harness +
same effort, archive replaced records, rebuild report, re-verify) — never
report a retry as a separate run.

## Invariants (inherited; do not break)

1. **Vision-delivery control first** — synthetic 6305 image gates the batch.
2. **Tools off** — `--tools ""`, safe mode, no MCP, empty `agent_cwd/`; any
   tool attempt fails acceptance.
3. **No annotated image is ever an input** — GT read only after the call,
   sha256-guarded against the manifest snapshot.
4. **Inputs ≤1280 px max edge** — WGISD (2048 px) is downscaled at manifest
   build with GT rescaled by the same factor; everything else native.
5. **Provenance** — every record carries THIS harness's fingerprint (hash of
   `paper_study/*.py + lib/ + schema/`) AND the base harness's; the run dir
   name carries `<model>-<effort>-<cli>-paper_fruit_seg`.
6. **No detector in the loop** — the VLM alone, from pixels.

Fingerprint note: the glob includes `verify_run.py`, so editing the verifier
shifts the pipeline identity (same conservatism as the strawdi seg harness).
Acceptance requires all records of a run to share ONE harness fingerprint;
whether it still matches the CURRENT code is identity information — the
verifier prints it as a note when a run predates a code edit. Any post-run
edit to a run directory (e.g. the 2026-09-18 overlay reorg) gets a
`post_run_notes.json` in the run dir recording exactly what changed.

## Testbed rules

* 3 frames per dataset, from each dataset's TEST split, deterministic
  (filter ≥3 GT instances; wgisd additionally requires cluster masks; sort by
  filename; evenly spaced). Recorded in the manifest. 21 frames total across
  seven sources (incl. KFuji RGB-DS patches since 2026-09-18).
* The datasets stay on the mount; `data/frames/` holds only the 18 delivered
  copies (gitignored, regenerable).

## Extending

* New dataset: add a loader in `lib/gtload.py`, a `source_*` sampler in
  `build_testbed_manifest.py`, wording in `lib/prompt.py` `SCENE_WORDING`, a
  protocol row in `PROTOCOL` + `DATASETS.md`. Re-verify GT conventions
  empirically before trusting boxes (top-left vs centre has already bitten
  once — ACFR rectangles).
* New metric: extend `per_source_summary` + `verify_run.py` reproducibility
  together — a number the gate cannot reproduce must not reach the tracking
  table.
