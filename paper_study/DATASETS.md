# Datasets — task, protocol, metrics (the tracking table)

The paper_study cross-fruit study, one row per dataset. This file is the
living index: task definitions, ground-truth semantics, each dataset's own
evaluation protocol, reference numbers from the literature, and — as runs
happen — our measured numbers (always glm-5.3-flash, thinking max, via the
claude CLI, 3-frame testbed; a development diagnostic, NOT a benchmark).

Datasets live on the removable mount `/media/zfei/GLOWAY/strawberry_detection/`
(`extra_fruits/` + `StrawDI_Db1/`); they are read by the manifest builder and
re-read (sha256-guarded) only AFTER the model calls at score time. Label files
are never model inputs.

## Tracking table

| dataset | task | GT unit | protocol metrics | lit. reference (model) | our testbed: box F1@0.5 (P/R) | our testbed: mask F1@0.5 |
| --- | --- | --- | --- | --- | --- | --- |
| `strawdi` | strawberry instance seg (visible surface); detection as mask-derived boxes | visible-surface mask | mask F1@IoU-0.5 headline (+AP ladder; box cross-check) | — (public 2025 dataset; our blog_study baselines: glm-5.3-flash mask F1 0.665 val / k3-256k 0.718) | **0.533** (0.667/0.444) | **0.467** (0.583/0.389) |
| `minneapple` | apple detection + (semantic) segmentation | apple instance (polygon; occluded included; background-row + ground apples deliberately unannotated) | COCO AP@[.50:.95], AP@50 (official: withheld-test CodaLab server) | Faster R-CNN AP@50 0.775 / AP 0.438 (Häni et al. 2020, official test); BFPNet 0.846 / 0.435 (train re-split) | 0.179 (0.151/0.221) — over-detects (see below) | 0.123 (0.103/0.151) |
| `wgisd` | grape-cluster detection; instance seg on masked subset | grape CLUSTER (bunch, not berry) | instance P/R/F1 + AP at IoU 0.5 | Mask R-CNN F1@0.5 0.840 det / 0.847 seg (Santos et al. 2020); GrapeUL-YOLO mAP@0.5 0.912 | 0.247 (0.370/0.185) | 0.173 (0.259/0.130) |
| `acfr_apples` | apple detection (night orchard) | circle (c-x, c-y, r) → square box | object-wise F1 — protocol TP at **IoU > 0.2** (Bargoti & Underwood 2017; ours @0.5 stricter, @0.25 the bridge) | Faster R-CNN VGG16 **0.904** (Bargoti & Underwood 2017) | 0.385 (0.500/0.313); @0.25 0.539 | n/a (semantic pixels only) |
| `acfr_mangoes` | mango detection (night orchard) | rectangle (x, y = top-left, dx, dy) | object-wise F1 (protocol IoU > 0.2) | Faster R-CNN VGG16 **0.908** (P 0.958 / R 0.863) | **0.500** (0.833/0.357); @0.25 0.600 | n/a |
| `acfr_almonds` | almond (hull) detection | rectangle (top-left + size) | object-wise F1 (protocol IoU > 0.2) | Faster R-CNN VGG16 **0.775** | 0.059 (0.111/0.040); @0.25 0.294 | n/a |
| `kfuji` | Fuji apple detection (RGB patches of Kinect v2 frames; dataset also has RGB-D variants — this study is RGB-only, `RGBhr` variant) | apple (box, top-left squares) | P/R/F1 at conf 0.85 + AP, **TP at IoU > 0.5** (same strictness as ours) | Faster R-CNN VGG16 RGBhr **F1 0.867 / AP 92.7**; best RGBhr+S+D 0.898 / 94.8 (Gené-Mola et al. 2019) | 0.300 (0.360/0.257); @0.25 0.667 | n/a (boxes only) |

## Our measured numbers (2026-09-18, run `20260918-225204-glm-5.3-flash-max-claude-paper_fruit_seg-full`)

glm-5.3-flash via claude CLI, **reasoning effort max**, 21/21 calls parsed +
schema-valid, zero tool attempts, control 2.2 px, verifier **PASS (257
checks)**; 161,388 tokens (94,850 in + 113,963 out; 16,960 cached reads
excluded from the total per base-harness convention — 208,813 incl.), $3.09
billed, 13 min at `--jobs 3`. Full artefacts:
`paper_study/runs/20260918-225204-…-full/` (report.md, responses.jsonl,
metrics.csv, summary_by_source.csv, overlays/ per dataset).

**Token accounting (2026-09-18):** usage is taken from the CLI-reported
token counts (`modelUsage` in the result event: input / cached / output) —
the bill is recorded separately as the endpoint's own `costUSD` and is not
used to derive any token number. Cross-check (secondary, one-off): the
billed cost is explained exactly by the reported counts at $5/$0.5/$25 per
MTok in/cached/out (max per-call residual < 1e-4¢ over all 21 calls) — so
on this endpoint nothing is billed outside the reported tokens; thinking
rides inside `output_tokens` (the CLI does not split it;
`reasoning_output_tokens` stays 0 on the claude path). If hidden billed
thinking is ever suspected, re-run that fit and look for residuals.

Failure shape, per source:

* **Recall is the binding constraint almost everywhere** (count bias −2 to
  −9 except MinneApple): the model under-reports dense/occluded fruit —
  worst on the night-orchard sources (almond hulls 25 GT → 7 matched@0.25;
  the almond task is effectively "find green ovals on a green canopy", the
  hardest census in the testbed).
* **MinneApple over-detects** (+13 count bias; 71 preds vs 31 GT on
  `dataset1_back_1`): the prompt demands EVERY apple while the GT
  deliberately omits background-row and ground apples — a documented
  annotation-policy gap, not model error. The same frame set is also where
  the polygon mask precision collapses (0.103). Quote this row with the
  caveat, always.
* **Localisation looseness** costs a further chunk between IoU 0.25 and 0.5
  (kfuji 0.667→0.300, wgisd 0.568→0.247): boxes land on the right fruit but
  overshoot/undershoot — the same ladder shape the blog study measured on
  StrawDI.
* **StrawDI stays the best case** (0.533/0.467), consistent with the blog
  study's val-split numbers on the same model at default effort (0.700 box /
  0.665 mask F1 on 100 frames — this 3-frame testbed is noisier and the
  frames differ).
* Vs the literature: the VLM sits far below the tuned detectors on every
  dataset (e.g. KFuji 0.300 vs Faster R-CNN 0.867 at the SAME IoU 0.5
  protocol; ACFR mango 0.500@0.5 / 0.600@0.25 vs 0.908@IoU-0.2). The gap
  is structural (no training, single-pass inventory from pixels), and the
  per-source F1@0.5 ordering (strawdi 0.53 > mango 0.50 > apple 0.38 >
  Fuji 0.30 > grape 0.25 > MinneApple 0.18 > almond 0.06) tracks
  fruit-vs-background contrast and annotation policy, not dataset size.

Our headline per dataset is **box F1@IoU-0.5** (P/R with it) — the one metric
definable on ALL six sources — plus **mask F1@IoU-0.5** where instance masks
exist (strawdi, minneapple-mirror, wgisd). COCO-style AP@50 / mAP@[.50:.95]
are recorded per run (with the confidence-clumping caveat) in each run dir's
`summary_by_source.csv`. Per-run numbers land in the table above via each
run's report; keep this file updated when a run completes.

The 20-frames/dataset output-format comparison (full9 vs seg3 vs box2,
same model/effort/testbed) lives in
[`FORMAT_COMPARISON.md`](FORMAT_COMPARISON.md) — its runs use the temporary
`manifest20.json` and are separate from the 3-frame baseline above.

## GT semantics and conventions (verified against the data 2026-09-18)

* **StrawDI_Db1** — grayscale id-map PNGs (0 bg, 1..N instance), 1008×756;
  test split = 200 frames. GT = visible-surface masks; boxes derived from
  masks.
* **MinneApple** (HF COCO mirror `lauesa1/minne-apple-segmentation`) — 331
  test images 720×1280 (portrait as delivered), 12,285 instance polygons.
  NOTE: the official test GT is withheld (CodaLab submission-only); the mirror
  annotations are community labels — fine for a diagnostic, never quote them
  as official-test numbers. Train split (670 img) annotations are PARTIAL by
  design (unannotated positives exist); the mirror's test set is fully
  labelled per instance.
* **WGISD** — 300 images 2048×1365(/1536), 5 varieties (CDY/CFR/CSV/SVB/SYH);
  `.txt` = YOLO-normalised boxes, single class 0 `uva` = the BUNCH; `.npz` =
  H×W×N binary cluster masks, slice i ↔ txt line i (137 masked images). Test
  split (COCO conversion lists) = 58 images / 850 clusters, 27 with masks.
  Delivered here downscaled to max-edge 1280 (LANCZOS frames, NEAREST masks,
  GT rescaled by the same factor).
* **ACFR Orchard Fruit 2016** — apples: 1,120 imgs 308×202, circles
  (centre + radius; validated: circle centres land on the semantic-mask apple
  pixels) + a SEMANTIC (non-instance) pixel mask per image → detection GT =
  square boxes; mangoes: 1,964 imgs 500×500 (16-bit source PNGs; read via
  PIL convert RGB), rectangles TOP-LEFT + size (validated: bright-fruit
  coverage 6:1 vs centre reading; annotator boxes carry a small up-left bias
  on fuzzy/dim fruit — dataset noise, not a parsing error); almonds: 620
  imgs 300×300, same rectangle format. Splits: sets/{train,val,test}.txt per
  fruit (apples 112 / mangoes 250 / almonds 100 test).
* **KFuji RGB-DS** (Zenodo 3715991) — 967 preprocessed 548×373 patches of
  Kinect v2 Fuji-apple orchard frames (`_RGBhr` colour variant used;
  `_RGBp` + `_DS.mat` depth/IR variants unused — RGB-only study), split via
  the dataset's own `sets/*.txt` (test = 193 patches); 12,839 apples total.
  `annotations/*.csv` = the ACFR-lineage `item,x,y,dx,dy` TOP-LEFT squares
  (validated: red-coverage 16/20 top-left over centre, 2026-09-18); the item
  column is a global apple id continuing across patches. `row data/` holds
  the 110 raw 1920×1080 frames + point clouds, unused here.

## Measurement caveats carried by every report

1. **3 frames per dataset, evenly spaced over the dataset's own test split** —
   a development diagnostic for the mixed pipeline, not a benchmark estimate.
2. **Box semantics gap** — the prompt asks WHOLE-fruit boxes; WGISD/
   MinneApple/StrawDI GT annotates the visible surface → occluded-fruit
   boxes lose IoU by construction. ACFR boxes are the dataset's own
   (circles→squares on apples).
3. **Polygon fidelity ceiling** — ≤32 whole-pixel vertices, straight
   segments, PIL raster semantics (same as the strawdi seg pipeline).
4. **AP confidence clumping** — inventory confidences cluster near 100; AP
   collapses toward the F1 operating point. F1 is the headline.
5. **MinneApple mirror labels** — not the withheld official test GT.

## Reference notes (literature, for context only)

* MinneApple: Häni, Roy, Isler, RA-L 2020 (arXiv:1909.06441). Detection:
  Faster R-CNN AP@[.5:.95] 0.438 / AP@50 0.775 on the official withheld test;
  BFPNet (2022) 0.435 / 0.846 on a 9:1 train re-split. Segmentation benchmark
  is SEMANTIC (IoU-based; UNet 0.685 overall IoU) — our instance-mask read of
  the mirror polygons is a different (instance-level) quantity.
* WGISD: Santos et al., CEAGR 2020 (arXiv:1907.11819). Mask R-CNN detection
  F1@0.5 0.840 (AP 0.719) on 58 test images / 837 clusters; instance seg F1
  0.847 / AP 0.743 on the 27 masked test images / 408 clusters. Berry-level
  counting (Khoroshevsky 2021) is a separate task we do not run.
* ACFR: Bargoti & Underwood 2017 (ICRA, arXiv:1610.03677; dataset
  data.acfr.usyd.edu.au/ag/treecrops/2016-multifruit). Object-level F1 on
  the held-out test split, TP = IoU > 0.2 one-to-one (deliberately looser
  than PASCAL 0.5, small-fruit localisation noise): Faster R-CNN VGG16 apple
  **0.904** / mango **0.908** / almond **0.775**; ZF backbone 0.892 / 0.876
  / 0.726; pixel-wise CNN + watershed (JFR 2017 baseline) apple 0.861 /
  mango 0.836. Confidence + NMS tuned on val; AP for ablations only. Paper
  splits: apple 729/112/112, mango 1154/270/270, almond 385/100/100
  (train/val/test). When quoting our ACFR numbers against these, read our
  F1@0.25 as the nearest bridge to their IoU-0.2 protocol.
* KFuji: Gené-Mola et al. 2019 (Comput. Electron. Agric. 162:689-698,
  DOI 10.1016/j.compag.2019.05.016; OA postprint
  upcommons.upc.edu/handle/2117/175186; dataset paper: Data in Brief
  25:104289; official code GRAP-UdL-AT/RGBD_fruit_detection_faster-rcnn).
  Faster R-CNN VGG-16 per modality on the 193-patch test split, TP at
  IoU > 0.5, P/R/F1 at conf 0.85: **RGBhr F1 0.867 / AP 92.7**; RGBp 0.829 /
  88.7; S 0.806 / 85.9; D 0.635 / 61.3; RGBp+S+D 0.866 / 91.2; best
  RGBhr+S+D **0.898 / 94.8**. Captured at NIGHT under artificial lighting
  (ToF degrades in sun) — same night-orchard regime as ACFR. The paper
  itself contrasts its IoU-0.5 strictness with ACFR's IoU-0.2 F1s: the two
  literature rows are not measured at the same matching strictness.
