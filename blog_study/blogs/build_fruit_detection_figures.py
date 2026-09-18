#!/usr/bin/env python3
"""Rebuild blog figures from saved segmentation responses; makes no model calls.

Data source: the StrawDI instance-segmentation runs on the 200-image TEST
split (strawdi_eval/seg/runs/). Every saved ok record is re-scored from the
raw label PNGs (sha256-guarded, read-only) and the recomputed values are
asserted against the stored ones before any figure is drawn.

GPT-6 Astra uses the authoritative 200/200 run: 183 original records plus the
17 re-requested frames (8, 787, 795-995) whose first retry responses were
silently mis-routed provider-side (see the blog's routing caveat). To swap in
a newer run for any model, edit SPECS below and rerun; set GPT_PENDING = True
to mark a model's bars/panels as a partial batch again.

Run from any directory: python3 blogs/build_fruit_detection_figures.py
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/strawberry-blog-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strawdi_eval.lib.scoring import precision_recall_f1
from strawdi_eval.seg.lib.seg_scoring import (score_masks_image, prepare_predictions_masks,
                                              ap_metrics_masks)

OUT = ROOT / 'blogs/assets/fruit_detection_is_solved_by_vlms'
OUT.mkdir(parents=True, exist_ok=True)

# --- run configuration: the only block to touch when runs are replaced ------
SEG_RUNS = ROOT / 'strawdi_eval/seg/runs'
SPECS = [
 ('Kimi K3',          [SEG_RUNS / '20260916-201059-k3-256k-default-codex-strawdi_seg-full']),
 ('GLM-5.3-Flash',    [SEG_RUNS / '20260917-230853-glm-5.3-flash-max-claude-strawdi_seg-maxthinking']),
 ('DeepSeek V4.1-Flash', [SEG_RUNS / '20260916-215556-deepseek-flash-default-codex-strawdi_seg-full']),
 ('GPT-6 Astra',      [SEG_RUNS / '20260917-201419-gpt-6-astra-low-codex-strawdi_seg-retry-requested-full200']),
]
GPT_PENDING = False         # GPT-6 Astra test batch complete (200/200 ok)
COLORS = ['#687a95', '#da9b35', '#8a73ac', '#138577']
GALLERY_SCENES = ['1251', '1838', '2085', '2532', '1669', '926']   # GPT-6 Astra panels
COMPARE_SCENE = '2532'      # same-scene four-model comparison
CHAOS_SCENES = ['sb04', 'IMG_7665']
CHAOS_RUNS = [              # (panel label, run dir); None dir -> placeholder panel
 ('GPT-6 Astra (low)', ROOT / 'vlm_eval/chaos/runs/20260917-210024-gpt-6-astra-low-codex-vlm_chaos-full'),
 ('5.6 luna · low',   ROOT / 'vlm_eval/chaos/runs/20260917-210659-gpt-5.6-luna-low-codex-vlm_chaos-full'),
 ('5.6 luna · medium', ROOT / 'vlm_eval/chaos/runs/20260917-210815-gpt-5.6-luna-medium-codex-vlm_chaos-full'),
 ('5.6 luna · high',  ROOT / 'vlm_eval/chaos/runs/20260917-211003-gpt-5.6-luna-high-codex-vlm_chaos-full'),
 ('5.6 sol · low',    ROOT / 'vlm_eval/chaos/runs/20260917-211255-gpt-5.6-sol-low-codex-vlm_chaos-full'),
 ('5.6 sol · medium', ROOT / 'vlm_eval/chaos/runs/20260917-211421-gpt-5.6-sol-medium-codex-vlm_chaos-full'),
 ('5.6 sol · high',   ROOT / 'vlm_eval/chaos/runs/20260917-211606-gpt-5.6-sol-high-codex-vlm_chaos-full'),
 ('GLM-5.3-Flash',    ROOT / 'vlm_eval/chaos/runs/20260917-230902-glm-5.3-flash-max-claude-vlm_chaos-maxthinking'),
]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.facecolor': 'white'})

def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def load_gt_masks(record):
    """Per-instance bool masks from the label id-map PNG (sha256-guarded)."""
    label_path = Path(record['label_image'])
    digest = hashlib.sha256(label_path.read_bytes()).hexdigest()
    assert digest == record['label_sha256'], f'{label_path.name}: label sha256 drifted'
    mask = np.array(Image.open(label_path))
    return [mask == int(v) for v in sorted(np.unique(mask)) if v != 0]

# --- load + merge (later dirs fill in samples the earlier ones failed on) ---
runs = {}
for name, dirs in SPECS:
    merged = {}
    for d in dirs:
        for r in read(d / 'responses.jsonl'):
            cur = merged.get(r['sample_id'])
            if cur is None or (cur['status'] != 'ok' and (r['status'] == 'ok' or True)):
                if cur is None or cur['status'] != 'ok' or r['status'] == 'ok':
                    merged[r['sample_id']] = r
    runs[name] = merged

# Consistency: same 200-image manifest, same GT, and every stored ok record
# reproduces exactly under the saved scorer + raw labels.
ref = next(r for r in runs['GLM-5.3-Flash'].values())
assert all(len(records) == 200 for records in runs.values())
_gt_cache = {}
def gt_masks_for(r):
    if r['sample_id'] not in _gt_cache:
        _gt_cache[r['sample_id']] = load_gt_masks(r)
    return _gt_cache[r['sample_id']]

for name, _ in SPECS:
    for sid, r in runs[name].items():
        g = runs['GLM-5.3-Flash'][sid]
        assert (r['gt_boxes'], r['gt_areas']) == (g['gt_boxes'], g['gt_areas'])
        if r['status'] == 'ok':
            scored = score_masks_image(r['inventory'], gt_masks_for(r), r['gt_areas'],
                                       r['frame_w'], r['frame_h'])
            for key in ('tp_25', 'fp_25', 'fn_25', 'tp_50', 'fp_50', 'fn_50',
                        'tp_75', 'fp_75', 'fn_75', 'mean_matched_iou_50'):
                assert r[key] == scored[key], (name, sid, key)
            assert r['matches_50'] == scored['matches_50'], (name, sid)

def aggregate(records):
    valid = [r for r in records.values() if r['status'] == 'ok']
    out = {'n_valid': len(valid), 'gt': sum(len(r['gt_boxes']) for r in valid)}
    for t in ('25', '50', '75'):
        tp, fp, fn = (sum(r[f'{k}_{t}'] for r in valid) for k in ('tp', 'fp', 'fn'))
        p, rec, f1 = precision_recall_f1(tp, fp, fn)
        out.update({f'tp_{t}': tp, f'fp_{t}': fp, f'fn_{t}': fn,
                    f'precision_{t}': p, f'recall_{t}': rec, f'f1_{t}': f1})
    out.update(ap_metrics_masks([{'preds': prepare_predictions_masks(r['inventory'], r['frame_w'], r['frame_h']),
                                  'gt_masks': gt_masks_for(r)} for r in valid]))
    matches = [m for r in valid for m in r['matches_50']]
    out['n_matched'] = len(matches)
    out['mean_matched_iou'] = float(np.mean([m['iou'] for m in matches]))
    out['median_matched_iou'] = float(np.median([m['iou'] for m in matches]))
    strata = []
    for lo, hi in ((0, 1024), (1024, 9216), (9216, 10**12)):
        n_gt = n_hit = 0
        for r in valid:
            hit = {m['gt_index'] for m in r['matches_50']}
            for j, area in enumerate(r['gt_areas']):
                if lo <= area < hi:
                    n_gt += 1
                    n_hit += j in hit
        strata.append({'band': (lo, hi), 'n_gt': n_gt, 'n_matched': n_hit,
                       'recall': n_hit / n_gt if n_gt else None})
    out['size'] = strata
    out['count_mae'] = float(np.mean([abs(r['count_error']) for r in valid]))
    out['box_f1_50'] = precision_recall_f1(*[sum(r['box_metrics'][k] for r in valid)
                                             for k in ('tp_50', 'fp_50', 'fn_50')])[2] \
        if valid and isinstance(valid[0].get('box_metrics'), dict) else None
    out['usage'] = {k: sum(r.get(k) or 0 for r in records.values()) / len(records)
                    for k in ('input_tokens', 'cached_input_tokens', 'output_tokens',
                              'reasoning_output_tokens', 'total_tokens', 'wall_s')}
    return out

summary = {}
for name, dirs in SPECS:
    summary[name] = {'runs': [d.name for d in dirs], 'reported': aggregate(runs[name])}
common = set.intersection(*[{sid for sid, r in records.items() if r['status'] == 'ok'}
                            for records in runs.values()])
for name, _ in SPECS:
    summary[name]['common'] = aggregate({sid: runs[name][sid] for sid in common})
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
with (OUT / 'comparison.csv').open('w') as f:
    fields = ['model', 'n_valid', 'gt', 'precision_50', 'recall_50', 'f1_50', 'f1_75',
              'ap_50', 'map_50_95', 'mean_matched_iou', 'count_mae', 'common_f1_50']
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for name, _ in SPECS:
        s = summary[name]
        writer.writerow({'model': name,
                         **{k: s['reported'][k] for k in fields[1:-1]},
                         'common_f1_50': s['common']['f1_50']})

def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT / f'{name}.svg', bbox_inches='tight')
    plt.close(fig)

def tag(name):
    return f'{name}†' if GPT_PENDING and name == 'GPT-6 Astra' else name

def hatch(name):
    return '///' if GPT_PENDING and name == 'GPT-6 Astra' else ''

# --- scientific panels: original pixels + saved polygons, GT mask as fill ---
def seg_panel(ax, r, title, gt_fill=True):
    ax.imshow(Image.open(ROOT / 'strawdi_eval' / r['image']))
    masks = gt_masks_for(r)
    if gt_fill:
        union = np.any(masks, axis=0)
        fill = np.zeros((*union.shape, 4))
        fill[union] = (1, 1, 1, 0.30)
        ax.imshow(fill, interpolation='nearest')
    matched = {m['pred_index'] for m in r['matches_50']}
    for i, fruit in enumerate(r['inventory']):
        poly = fruit.get('polygon')
        if poly:
            ax.add_patch(Polygon(poly, closed=True, fill=False, lw=1.7,
                                 ec='#46ef73' if i in matched else '#ff705b'))
    for j in r['fn_gt_indices_50']:
        ys, xs = np.nonzero(masks[j])
        ax.text(xs.mean(), ys.mean(), 'MISS', color='#ff4d4d', fontsize=8,
                fontweight='bold', ha='center', va='center')
    ax.set_title(title, fontsize=10)
    ax.axis('off')

# Figure: GPT-6 Astra gallery — six test scenes
fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), layout='constrained')
for ax, sid in zip(axes.flat, GALLERY_SCENES):
    r = runs['GPT-6 Astra'][sid]
    assert r['status'] == 'ok', f'gallery scene {sid} not ok in the GPT run yet'
    seg_panel(ax, r, f"image {sid} — TP {r['tp_50']} · FP {r['fp_50']} · FN {r['fn_50']}")
fig.suptitle('GPT-6 Astra, zero-shot, on StrawDI test scenes (visible-surface polygons)', fontsize=14)
fig.supxlabel('White fill: ground-truth instance mask     Green: matched prediction     Coral: unmatched prediction     MISS: labelled fruit not found',
              fontsize=9)
save(fig, 'gpt_gallery')

# Figure: same scene, four models
fig, axes = plt.subplots(2, 2, figsize=(11, 8.6), layout='constrained')
for ax, (name, _) in zip(axes.flat, SPECS):
    r = runs[name][COMPARE_SCENE]
    seg_panel(ax, r, f"{tag(name)} — TP {r['tp_50']} · FP {r['fp_50']} · FN {r['fn_50']}")
fig.suptitle(f'Same scene, same prompt — StrawDI test image {COMPARE_SCENE}', fontsize=14)
fig.supxlabel('White fill: ground-truth instance mask     Green: matched prediction     Coral: unmatched prediction     MISS: labelled fruit not found',
              fontsize=9)
save(fig, 'same_scene_seg')

# Figure: segmentation scores
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
x = np.arange(4)
for i, (key, label) in enumerate([('precision_50', 'Precision'), ('recall_50', 'Recall'), ('f1_50', 'F1')]):
    vals = [100 * summary[name]['reported'][key] for name, _ in SPECS]
    bars = axes[0].bar(x + (i - 1) * .24, vals, .23, label=label,
                       color=['#9fbcc5', '#6095a4', '#194e62'][i])
    for rect, (name, _) in zip(bars, SPECS):
        rect.set_hatch(hatch(name))
    axes[0].bar_label(bars, fmt='%.1f', fontsize=8, padding=3)
axes[0].set(xticks=x, xticklabels=[tag(n) for n, _ in SPECS], ylim=(0, 105),
            ylabel='Score (%) at mask IoU ≥ 0.50',
            title='Precision, recall and F1 on visible-surface masks')
axes[0].grid(axis='y', alpha=.16)
axes[0].set_axisbelow(True)
axes[0].legend(loc='upper left', ncol=3, fontsize=9)
for (name, _), color in zip(SPECS, COLORS):
    s = summary[name]['reported']
    axes[1].plot([.25, .5, .75], [100 * s[f'f1_{t}'] for t in ('25', '50', '75')],
                 '--o' if GPT_PENDING and name == 'GPT-6 Astra' else '-o',
                 color=color, label=tag(name), lw=2)
axes[1].set(xlabel='Mask IoU required for a match', ylabel='F1 (%)', xticks=[.25, .5, .75],
            ylim=(0, 105), title='F1 as the localization requirement tightens')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=.18)
fig.suptitle('Instance segmentation on the StrawDI test split (200 images)', fontsize=14)
save(fig, 'seg_scores')

# Figure: recall by visible instance size
fig, ax = plt.subplots(figsize=(8.5, 4.4), layout='constrained')
w = .2
for i, ((name, _), color) in enumerate(zip(SPECS, COLORS)):
    strata = summary[name]['reported']['size']
    vals = [100 * z['recall'] for z in strata]
    bars = ax.bar(np.arange(3) + (i - 1.5) * w, vals, w * .92, label=tag(name), color=color,
                  hatch=hatch(name))
    for rect, z in zip(bars, strata):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 2,
                f"{z['n_matched']}/{z['n_gt']}", ha='center', fontsize=7.5)
ax.set(xticks=range(3),
       xticklabels=['Small\n<1,024 px²', 'Medium\n1,024–9,215 px²', 'Large\n≥9,216 px²'],
       ylabel='Recall (%) at mask IoU ≥ 0.50', ylim=(0, 112),
       title='Recall by visible instance size — small fruit remains the hard case')
ax.grid(axis='y', alpha=.16)
ax.set_axisbelow(True)
ax.legend(fontsize=9, ncol=4)
save(fig, 'size_recall')

# Figure: tokens and latency
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout='constrained')
u = [summary[name]['reported']['usage'] for name, _ in SPECS]
for i, (field, label, color) in enumerate([('input_tokens', 'Input', '#517b99'),
                                           ('output_tokens', 'Output (includes reasoning)', '#dfab51')]):
    vals = [r[field] for r in u]
    bottom = [r['input_tokens'] if i else 0 for r in u]
    bars = axes[0].bar(x, vals, bottom=bottom, label=label, color=color)
    for rect, (name, _) in zip(bars, SPECS):
        rect.set_hatch(hatch(name))
    for j, v in enumerate(vals):
        axes[0].text(j, bottom[j] + v / 2, f'{v:,.0f}', ha='center', va='center', fontsize=8)
axes[0].set(xticks=x, xticklabels=[tag(n) for n, _ in SPECS],
            ylabel='Mean recorded tokens per scheduled image',
            title='The inventory is more than a polygon')
axes[0].legend(fontsize=9)
bars = axes[1].bar(x, [r['wall_s'] for r in u], color=COLORS)
for rect, (name, _) in zip(bars, SPECS):
    rect.set_hatch(hatch(name))
axes[1].bar_label(bars, fmt='%.1f s', padding=4)
axes[1].set(xticks=x, xticklabels=[tag(n) for n, _ in SPECS],
            ylabel='Mean recorded call time (seconds)',
            title='Seconds per image, including recorded retries')
fig.suptitle('Operational cost • all 200 scheduled images per model', fontsize=14)
save(fig, 'tokens_latency')

# Figures: private chaotic scenes, qualitative (no ground truth exists) —
# one all-model grid per scene, GPT-6 Astra first.
chaos = {}
for label, d in CHAOS_RUNS:
    chaos[label] = {}
    if d is not None and (d / 'responses.jsonl').exists():
        for r in read(d / 'responses.jsonl'):
            if r['status'] == 'ok':
                chaos[label][r['sample_id']] = r

def chaos_panel(ax, r, scene, label):
    img = Image.open(ROOT / 'vlm_eval' / r['image']) if r else \
          Image.open(ROOT / 'vlm_eval/data/frames' / f'chaos__{scene}.png')
    ax.imshow(img)
    if r is None:
        ax.text(0.5, 0.5, f'{label}\nrun in progress', transform=ax.transAxes,
                ha='center', va='center', fontsize=13, color='white',
                bbox=dict(boxstyle='round,pad=0.45', fc='black', alpha=.65, ec='none'))
        ax.set_title(label, fontsize=10)
    else:
        cmap = plt.get_cmap('RdYlGn')
        for fruit in r['inventory']:
            poly = fruit.get('polygon')
            if poly:
                ax.add_patch(Polygon(poly, closed=True, fill=False, lw=1.7,
                                     ec=cmap((fruit.get('redness_pct') or 0) / 100)))
        ax.set_title(f"{label}\n{len(r['inventory'])} fruit", fontsize=10)
    ax.axis('off')

CHAOS_FIGSIZE = {'sb04': (11.5, 15.2), 'IMG_7665': (8.1, 23.5)}   # 4 rows x 2 cols, matched to scene aspect
CHAOS_LABELS = {'sb04': 'private-1', 'IMG_7665': 'private-2'}
for scene in CHAOS_SCENES:
    fig, axes = plt.subplots(4, 2, figsize=CHAOS_FIGSIZE[scene], layout='constrained')
    for ax, (label, _) in zip(axes.flat, CHAOS_RUNS):
        chaos_panel(ax, chaos[label].get(scene), scene, label)
    head = f'Zero-shot inventories on a private chaotic scene ({CHAOS_LABELS[scene]})'
    tail = 'polygon coloured by reported redness; no ground truth'
    fig.suptitle(f'{head}\n{tail}' if CHAOS_FIGSIZE[scene][0] < 9 else f'{head} — {tail}', fontsize=13)
    save(fig, f'chaos_{CHAOS_LABELS[scene]}')

# Figure: the silent-routing episode — suspect-batch vs re-requested responses
# on the same frames, same prompt, same recorded model identifier. No numbers:
# the polygons and MISS labels carry the story.
ROUTING_FRAMES = ['995', '926']
ROUTING_SUSPECT = SEG_RUNS / '20260917-153213-gpt-6-astra-low-codex-strawdi_seg-full-merged199'
ROUTING_CORRECT = SEG_RUNS / '20260917-201419-gpt-6-astra-low-codex-strawdi_seg-retry-requested-full200'
suspect = {r['sample_id']: r for r in read(ROUTING_SUSPECT / 'responses.jsonl')}
correct = {r['sample_id']: r for r in read(ROUTING_CORRECT / 'responses.jsonl')}
for sid in ROUTING_FRAMES:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout='constrained')
    seg_panel(axes[0], suspect[sid], 'Suspect batch', gt_fill=False)
    seg_panel(axes[1], correct[sid], 'Re-requested under identical settings', gt_fill=False)
    fig.suptitle(f'Same frame, same prompt, same recorded model identifier — StrawDI test image {sid}',
                 fontsize=13)
    fig.supxlabel('Green: matched prediction     Coral: unmatched prediction     MISS: labelled fruit not found',
                  fontsize=9)
    save(fig, f'routing_{sid}')

# --- blog-ready numbers ------------------------------------------------------
print((OUT / 'comparison.csv').read_text())
print(f'common subset: {len(common)} images (the GPT-6 Astra valid set)')
for name, _ in SPECS:
    s = summary[name]['reported']
    print(f"{name}: valid {s['n_valid']}/200  F1@50 {100*s['f1_50']:.1f}  F1@75 {100*s['f1_75']:.1f}  "
          f"AP50 {100*s['ap_50']:.1f}  mAP {100*s['map_50_95']:.1f}  meanIoU {s['mean_matched_iou']:.3f}  "
          f"MAE {s['count_mae']:.2f}  boxF1@50 {100*(s['box_f1_50'] or 0):.1f}  "
          f"commonF1@50 {100*summary[name]['common']['f1_50']:.1f}")
    print(f"   size recall: " + '  '.join(f"{z['n_matched']}/{z['n_gt']}" for z in s['size']) +
          f"   tokens in/out {s['usage']['input_tokens']:.0f}/{s['usage']['output_tokens']:.0f}  "
          f"wall {s['usage']['wall_s']:.1f}s")
print('chaos fruit counts:')
for label, _ in CHAOS_RUNS:
    print(' ', label, {s: len(chaos[label][s]['inventory'])
                       for s in CHAOS_SCENES if s in chaos[label]})
