#!/usr/bin/env python3
"""Rebuild blog figures from saved predictions; makes no model calls.
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
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strawdi_eval.lib.scoring import score_image, precision_recall_f1, ap_metrics, prepare_predictions, size_stratified
OUT = ROOT / 'blogs/assets/fruit_detection_is_solved_by_vlms'
OUT.mkdir(parents=True, exist_ok=True)
SPECS = [
 ('Kimi', '20260915-110939-kimi-for-coding-default-codex-strawdi_eval-full'),
 ('GLM', '20260915-113703-glm-5.3-flash-default-claude-strawdi_eval-full'),
 ('DeepSeek', '20260915-143336-deepseek-flash-high-codex-strawdi_eval-full'),
 ('GPT-6 Astra', '20260915-162534-gpt-6-astra-low-codex-strawdi_eval-consolidated'),
]
COLORS = ['#687a95', '#da9b35', '#8a73ac', '#138577']
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
                     'axes.spines.right':False, 'axes.titleweight':'bold', 'savefig.facecolor':'white'})

def read(path):
 return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

runs = {name:read(ROOT/'strawdi_eval/runs'/folder/'responses.jsonl') for name,folder in SPECS}
# Check the comparison contract and recompute every saved detection result.
ref = runs['GPT-6 Astra']
for records in runs.values():
 assert len(records)==100 and len({r['sample_id'] for r in records})==100
 for r,g in zip(records,ref):
  assert (r['sample_id'],r['prompt'],r['gt_boxes'],r['gt_areas']) == (g['sample_id'],g['prompt'],g['gt_boxes'],g['gt_areas'])
  if r['status']=='ok':
   scored=score_image(r['inventory'],r['gt_boxes'],r['gt_areas'],r['frame_w'],r['frame_h'])
   for key,value in scored.items(): assert r[key]==value, (r['sample_id'],key)
common=set.intersection(*[{r['sample_id'] for r in records if r['status']=='ok'} for records in runs.values()])
assert len(common)==98

def summarize(records):
 valid=[r for r in records if r['status']=='ok']
 result={'frames':len(records),'parsed':len(valid),'gt':sum(len(r['gt_boxes']) for r in valid)}
 for threshold in ['25','50','75','center']:
  tp,fp,fn=(sum(r[f'{key}_{threshold}'] for r in valid) for key in ('tp','fp','fn'))
  p,rec,f1=precision_recall_f1(tp,fp,fn)
  result.update({f'{k}_{threshold}':v for k,v in zip(('tp','fp','fn','precision','recall','f1'),(tp,fp,fn,p,rec,f1))})
 result.update(ap_metrics([{'sample_id':r['sample_id'],'preds':prepare_predictions(r['inventory'],r['frame_w'],r['frame_h']),'gt_boxes':r['gt_boxes']} for r in valid]))
 result['size']=size_stratified(valid)
 result['count_mae']=float(np.mean([abs(r['count_error']) for r in valid]))
 return result

summary={}
for name,folder in SPECS:
 records=runs[name]
 summary[name]={'run':folder, 'reported':summarize(records), 'common98':summarize([r for r in records if r['sample_id'] in common]),
 'usage':{k:sum(r.get(k) or 0 for r in records)/100 for k in ['input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens','total_tokens','wall_s']},
 'provenance':{k:records[0][k] for k in ['model','provider','reasoning_effort','harness_version','harness_fingerprint','base_harness_version','base_harness_fingerprint']},
 'responses_sha256':hashlib.sha256((ROOT/'strawdi_eval/runs'/folder/'responses.jsonl').read_bytes()).hexdigest()}
# First-pass sensitivity is separate; never overwrite the harness's failure-excluding metrics.
original=read(ROOT/'strawdi_eval/runs'/SPECS[-1][1]/'history/original_responses.jsonl')
summary['GPT-6 Astra']['original']=summarize(original)
gptu=summary['GPT-6 Astra']['usage']
summary['GPT-6 Astra']['api_estimate']={'uncached_usd_per_image':(10*gptu['input_tokens']+50*gptu['output_tokens'])/1e6,
 'cached_read_usd_per_image':(10*(gptu['input_tokens']-gptu['cached_input_tokens'])+gptu['cached_input_tokens']+50*gptu['output_tokens'])/1e6,
 'source':'https://developers.openai.com/api/docs/models/gpt-6-astra','checked':'2026-09-15',
 'note':'Standard-rate estimate from recorded tokens; includes recorded retries, excludes controls and unreported usage/cache-write fees; not an invoice.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (OUT/'comparison.csv').open('w') as f:
 fields=['model','parsed','gt','precision_50','recall_50','f1_50','f1_75','ap_50','map_50_95','count_mae','common98_f1']
 writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
 for name,_ in SPECS:
  s=summary[name];writer.writerow({'model':name,**{k:s['reported'][k] for k in fields[1:-1]},'common98_f1':s['common98']['f1_50']})

def save(fig,name):
 fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight')
 fig.savefig(OUT/f'{name}.svg',bbox_inches='tight')
 plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
x=np.arange(4)
for ax,cohort,title in zip(axes,['reported','common98'],['Saved runs: 99 / 99 / 100 / 100 valid images','Same 98 images: every model returned valid JSON']):
 for i,(key,label) in enumerate([('precision_50','Precision'),('recall_50','Recall'),('f1_50','F1')]):
  vals=[100*summary[name][cohort][key] for name,_ in SPECS]
  bars=ax.bar(x+(i-1)*.24,vals,.23,label=label,color=['#9fbcc5','#6095a4','#194e62'][i])
  ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=3)
 ax.set(xticks=x,xticklabels=[n for n,_ in SPECS],ylim=(0,105),ylabel='Score (%) at box IoU ≥ 0.50',title=title)
 ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
axes[0].legend(loc='upper left',ncol=3,fontsize=9)
fig.suptitle('GPT-6 Astra leads this strawberry detection comparison',fontsize=16,fontweight='bold')
save(fig,'detection_comparison')

fig,axes=plt.subplots(1,2,figsize=(11.5,4.4),layout='constrained')
for (name,_),color in zip(SPECS,COLORS):
 s=summary[name]['reported']
 axes[0].plot([.25,.5,.75],[100*s[f'f1_{t}'] for t in ['25','50','75']],'-o',color=color,label=name,lw=2)
 axes[1].plot(range(3),[100*z['recall'] for z in s['size']],'-o',color=color,lw=2)
axes[0].set(xlabel='Box IoU required for a match',ylabel='F1 (%)',xticks=[.25,.5,.75],ylim=(0,105),title='Tighter boxes still distinguish GPT')
axes[0].legend(fontsize=9)
axes[1].set(xticks=range(3),xticklabels=['Small\n<1,024 px²','Medium\n1,024–9,215 px²','Large\n≥9,216 px²'],ylabel='Recall (%) at IoU ≥ 0.50',ylim=(0,105),title='Small fruit remains the hard case')
for ax in axes: ax.grid(alpha=.18)
fig.suptitle('Localization and object size • each run’s valid images',fontsize=15,fontweight='bold')
save(fig,'localization_and_size')

fig,axes=plt.subplots(1,2,figsize=(11.5,4.4),layout='constrained')
u=[summary[name]['usage'] for name,_ in SPECS]
for i,(field,label,color) in enumerate([('input_tokens','Input','#517b99'),('output_tokens','Output (includes reasoning)','#dfab51')]):
 vals=[r[field] for r in u];bottom=[r['input_tokens'] if i else 0 for r in u]
 axes[0].bar(x,vals,bottom=bottom,label=label,color=color)
 for j,v in enumerate(vals): axes[0].text(j,bottom[j]+v/2,f'{v:,.0f}',ha='center',va='center',fontsize=9)
axes[0].set(xticks=x,xticklabels=[n for n,_ in SPECS],ylabel='Mean recorded tokens per scheduled image',ylim=(0,19000),title='The inventory is more than four coordinates')
axes[0].legend(fontsize=9)
bars=axes[1].bar(x,[r['wall_s'] for r in u],color=COLORS)
axes[1].bar_label(bars,fmt='%.1f s',padding=4)
axes[1].set(xticks=x,xticklabels=[n for n,_ in SPECS],ylabel='Mean recorded call time (seconds)',ylim=(0,125),title='Seconds per image, including recorded retries')
fig.suptitle('Operational cost • all 100 scheduled images per model',fontsize=15,fontweight='bold')
save(fig,'tokens_and_latency')

# Scientific image panels use original pixels as a background and saved boxes as data.
def panel(ax,r,title,focus=None,show_gt=True):
 ax.imshow(Image.open(ROOT/'strawdi_eval'/r['image']))
 if show_gt:
  for box in r['gt_boxes']:
   x1,y1,x2,y2=box;ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,ec='#00e5ff',lw=1.6,ls='--'))
 matched={m['pred_index'] for m in r['matches_50']}
 for i,fruit in enumerate(r['inventory']):
  x1,y1,x2,y2=fruit['bbox'];color='#46ef73' if i in matched else '#ff705b'
  ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,ec=color,lw=1.5))
 if focus: ax.set_xlim(focus[0],focus[2]);ax.set_ylim(focus[3],focus[1])
 ax.set_title(title,fontsize=10);ax.axis('off')

fig,axes=plt.subplots(1,4,figsize=(15,4),layout='constrained')
for ax,(name,_) in zip(axes,SPECS):
 r=next(r for r in runs[name] if r['sample_id']=='108')
 panel(ax,r,f"{name}\nTP {r['tp_50']} · FP {r['fp_50']} · FN {r['fn_50']}")
fig.suptitle('Same scene, same prompt • StrawDI image 108',fontsize=16,fontweight='bold')
fig.supxlabel('Dashed cyan: ground truth     Green: matched prediction     Coral: unmatched prediction',fontsize=10)
save(fig,'same_scene')

fig,axes=plt.subplots(2,3,figsize=(12,7.5),layout='constrained')
cases=[('108',(775,155,890,340),'Seeded green fruit: no separate GT instance'),('113',(835,155,985,285),'Red fruit under a leaf: no GT instance'),('1717',(465,160,540,245),'Whole-fruit box versus visible-mask box')]
manifest=json.loads((ROOT/'strawdi_eval/runs'/SPECS[-1][1]/'manifest.json').read_text())
for col,(sid,focus,title) in enumerate(cases):
 r=next(r for r in ref if r['sample_id']==sid)
 panel(axes[0,col],r,f'Image {sid}\n{title}',focus)
 sample=next(s for s in manifest['samples'] if s['sample_id']==sid)
 mask=np.asarray(Image.open(sample['label_image']))
 mask_colors=list(plt.get_cmap('tab20').colors)
 mask_colors[0]=(0.08,0.10,0.13)
 axes[1,col].imshow(mask,cmap=ListedColormap(mask_colors),vmin=0,vmax=19,interpolation='nearest')
 axes[1,col].set(xlim=(focus[0],focus[2]),ylim=(focus[3],focus[1]),title='Dataset instance mask • identical crop')
 axes[1,col].axis('off')
fig.suptitle('Inspect the labels before interpreting every unmatched box as hallucination',fontsize=14,fontweight='bold')
fig.supxlabel('Dashed cyan: GT box     Green: match     Coral: unmatched prediction     Dark mask pixels: background',fontsize=9)
save(fig,'annotation_examples')

# Qualitative only: original private scene with outline annotations, no counts or scores.
p=ROOT/'vlm_eval/runs/20260914-191505-gpt-6-astra-medium-codex-vlm_eval-full'
r=next(r for r in read(p/'responses.jsonl') if r['style']=='inventory_plain' and r['sample_id']=='sb01')
fig,ax=plt.subplots(figsize=(10,7),layout='constrained')
ax.imshow(Image.open(ROOT/'vlm_eval'/r['image']))
for i,f in enumerate(r['inventory']):
 x1,y1,x2,y2=f['bbox'];color='#08d9e8' if i==r['target_index'] else '#ffd166'
 ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,ec=color,lw=1.7))
ax.set_title('A private greenhouse scene: inventory and a proposed picking target',fontsize=13)
ax.axis('off');fig.supxlabel('Amber: reported fruit     Cyan: proposed target     Qualitative output; no ground-truth labels',fontsize=10)
save(fig,'beyond_detection')
print((OUT/'comparison.csv').read_text())
for name,_ in SPECS: print(name,summary[name]['usage'])
print('GPT estimates:',summary['GPT-6 Astra']['api_estimate'])
print('GPT original:',summary['GPT-6 Astra']['original'])
