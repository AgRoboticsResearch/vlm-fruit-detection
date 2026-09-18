# Is Visual Fruit Detection Solved by Vision-Language Models, and What Remains? — In the GPT-6 Astra Era and Beyond

Sep 16, 2026 Zhenghao Fei [📧](mailto:fei.holly@gmail.com)

> **TL;DR** — I tested whether frontier vision-language models can perform fruit detection directly from raw images and a written prompt, without task-specific training. On all 200 images of the StrawDI test set, GPT-6 Astra achieved 60.1 segmentation mAP and 86.0 mean instance IoU — exceeding the dataset’s fully supervised Mask R-CNN benchmark in mAP (45.36) and nearly matching its IoU (87.7), despite using no StrawDI training data. The other VLMs we tested were substantially behind, suggesting that this is still a frontier-model capability rather than a general property of VLMs. Astra also showed surprisingly strong performance on small fruit and transferred qualitatively to difficult private scenes, while the same prompt could return masks together with attributes such as redness, occlusion, and graspability. Important limitations remain, especially very small fruit, latency, cost, local deployment, and dependence on a small number of frontier models. Fruit detection is not universally solved, but the starting point has changed dramatically: a strong VLM is now worth testing before collecting and labeling a new task-specific dataset.

Detecting fruit in images has been a long-standing challenge in agricultural computer vision. It is hard not because we cannot achieve high accuracy on a specific dataset, but because real-world scenarios are highly variable. No two orchards look the same, and the same type of fruit can look very different depending on the environment (weather, lighting, occlusion, etc.) and the fruit itself (variety and horticultural practices). Most deep learning-based fruit detection methods rely on large amounts of labelled data to cover the variability (distribution) in the real world. Detection in out-of-distribution (OOD) scenarios often fails. Unfortunately, due to the inherent variability of real-world conditions and data privacy issues, fruit detection always has to deal with OOD scenarios. This means that one often cannot deploy a model trained in one orchard directly to another orchard without fine-tuning it with new data.

My collaborators and I have been working on improving the generalizability of fruit detection models for a while and have tried many methods, including domain adaptation, GANs (generative adversarial networks) [1], and learning from foundation models [2,3]. These methods have shown improvements, but they are still far from solving the problem.

In the meantime, I have always paid attention to vision-language models (VLMs), which are pre-trained on large-scale image–text pairs and have shown impressive generalization ability across many vision tasks. I have been wondering whether VLMs can help with fruit detection and have kept testing them on fruit detection tasks. However, the results were not promising, especially for precise detection.

General-purpose vision-language models have shown impressive visual recognition and reasoning abilities, including the ability to identify fruits across diverse scenes without task-specific training. However, their performance on precise detection has historically been less reliable, as we can see in the experiments below. Several factors may have contributed to this gap. First, there has often been a training-objective mismatch: many VLMs were trained primarily for image–text alignment, captioning, visual question answering, and instruction following rather than exhaustive instance-level localization [4]. Second, strong semantic representations do not necessarily imply equally precise spatial representations; a model may understand what objects are present without representing where every instance is with the same accuracy [4,5]. Third, visual resolution and tokenization may limit small and dense object perception, particularly when many similar objects occupy only a small number of visual tokens [6]. These should be viewed as characteristics of earlier and current VLM designs rather than fundamental limitations, as newer multimodal models such as GPT-6 Astra may substantially change this trade-off.

What motivated me to re-evaluate VLMs for fruit detection was the release of GPT-6 Astra. I had seen surprisingly strong results from Astra on tasks that appear to require substantial visual grounding and spatial understanding—for example, reconstructing and creating complex 3D scenes using tools such as Blender [7], and directly controlling robotic manipulators without task-specific policy training [8,9]. In one recent evaluation, Astra, used directly as a robot policy, even exceeded the reported performance of a specialized VLA baseline [8]. Astra has also set new highs on several visual-spatial and 3D reconstruction evaluations [7]. These results made me wonder whether the capabilities of general-purpose VLMs had reached a point where they could also achieve state-of-the-art performance on fruit detection—or, more broadly, on conventional visual detection tasks.

In this blog, I will share my experience using GPT-6 Astra and other VLMs for fruit detection, examine what has changed compared with earlier generations of VLMs, and explain why the results surprised me so much.


I am particularly impressed by three aspects of what VLMs may offer from this point forward.

1. **Accuracy.**
   Accuracy is the most fundamental metric for visual detection, and GPT-6 Astra shows a substantial improvement over previous VLMs. In our experiments, it detected strawberries with very high precision, even in challenging scenarios involving occlusion, dense clusters, and large variations in fruit size. More interestingly, when we examined the apparent “errors” carefully, many of them turned out to be annotation disagreements rather than clear model failures. In other words, Astra appears to be approaching ground-truth-level detection accuracy in some of our test cases.

2. **Zero-shot generalization.**
   All of our detection experiments were conducted in a zero-shot setting: the model was neither trained nor fine-tuned on the evaluation dataset. Despite this, GPT-6 Astra was able to detect strawberries accurately across diverse scenes. This level of generalization could fundamentally change how fruit detection systems are developed. Instead of collecting and labeling a large dataset for every new orchard, crop variety, camera setup, or environment, users may be able to apply a general-purpose VLM directly to a new detection task with little or no task-specific training.

3. **Near-infinite flexibility.**
   GPT-6 Astra is not limited to returning a bounding box and class label. For each detected fruit, it can also provide a pixel-level outline of the fruit's visible surface and rich semantic information, such as redness, occlusion, calyx visibility, stem visibility, graspability, confidence, and a natural-language description. The output can also be changed simply by modifying the prompt. This makes the detector highly flexible: users can request new attributes, redefine what counts as a target, or adapt the output to different downstream tasks without retraining the model. In this sense, the system begins to look less like a fixed-purpose detector and more like a general visual perception interface.

## A common experiment on an strawberry detection benchmark

We chose the **Strawberry Digital Images dataset (StrawDI)** as a benchmark for our experiments. It is a well-known dataset in the fruit detection community, and it provides a challenging testbed for evaluating the performance of visual detection models. The StrawDI contains photographs collected at commercial plantations in Huelva, Spain. Its annotated subset, **StrawDI_Db1**, contains 3,100 images at 1008 × 756 pixels, split into 2,800 training, 100 validation, and 200 test images, with an instance mask for every strawberry — including unripe, occluded, distant, and partly cropped fruit. [Official dataset description][strawdi]

We evaluated **all 200 test images (1,132 annotated strawberries)** — the same split the dataset's own paper benchmarks on [10], so our numbers can be set next to its supervised specialist directly. No model was trained or fine-tuned on StrawDI: this is a zero-shot evaluation in the operational sense. It worth noting that because StrawDI is public, it does not guarantee that the images were absent from model pretraining.

### The task and the controls

Every model received the same unannotated image and the same prompt: to detect all strawberry instances in the image and return nine fields for each fruit — a bounding box, a polygon outlining the fruit’s visible surface, redness, occlusion, calyx visibility, stem visibility, graspability, confidence, and a free-text description. **The polygons were scored as instance segmentation directly against the raw ground-truth masks** — the StrawDI paper's own benchmark task — and the boxes from the same responses were scored as detection as a cross-check. The other attributes have no labels in this benchmark.

The following models with cooresponding reasoning effort (thinking) were evaluated, and we will see that the reasoning effort is not the main factor for performance.

| Model name | Reasoning effort |
| --- | --- |
| Kimi K3 | High |
| GLM-5.3-Flash | Max |
| DeepSeek V4.1-Flash | High |
| GPT-6 Astra | Low |


### The exact prompt

The following prompt was identical across all four models and all 200 StrawDI test images; we reproduce it verbatim, including the whole-fruit box definition and the visible-surface polygon definition discussed above.

```text
You are looking at ONE image: a 1008x756 colour photograph of a strawberry plant.

COORDINATE SYSTEM for every coordinate you report: pixel coordinates, origin (0, 0) at the TOP-LEFT corner of the image, x increasing to the RIGHT (valid 0..1007), y increasing DOWNWARD (valid 0..755). Report whole numbers of pixels.

TASK
Produce a complete inventory of the strawberries in this image. For each one, describe it fully and trace the outline of its visible surface:
- "bbox": [x1, y1, x2, y2] - the bounding box of the WHOLE fruit. For a partly hidden fruit, give the box the fruit would occupy if the occluder were not there, not just the visible sliver. Box ONLY the fruit body: do not extend the box to cover the calyx (the green sepal leaves) or the stem when they sit apart from the fruit - a calyx that lies flat against the fruit is of course inside the box.
- "polygon": an ordered list of [x, y] vertices tracing the outline of the fruit's VISIBLE surface - the part you can actually see. Walk the boundary once, clockwise or counter-clockwise, starting anywhere. Where a leaf, stem or another fruit hides part of this fruit, follow the occluder's edge; do NOT extrapolate the shape you cannot see (unlike bbox, polygon covers ONLY what is visible). Same fruit-body rule as bbox: exclude the calyx and stem where they sit apart from the fruit. Use 8 to 20 vertices for a typical fruit and never more than 32; a simple closed outline (no self-crossing, no repeated closing vertex) drawn with straight segments between vertices. The polygon's extent must agree with the visible part of the fruit, and whole-pixel coordinates.
- "redness_pct": 0-100, a CONTINUOUS estimate - do not round to a category. Percent of the fruit's VISIBLE surface that is red, judged by hue and saturation together: 0 = no red at all (green or white), 25 = a pale pink flush, 50 = about half the visible surface is red, 75 = mostly red with pale or green patches left, 100 = fully saturated deep red everywhere. Use the whole range; 63 is a better answer than 60.
- "occlusion_pct": 0-100, also CONTINUOUS. How much of the fruit is hidden behind leaves, stems or other fruit: 0 = fully visible, 25 = a leaf edge clips it, 50 = about half hidden, 75 = mostly hidden, 100 = only a sliver shows.
- "calyx_visible": true if the green calyx (sepal ring) can be seen.
- "peduncle_visible": true if the stem is visible where it meets the fruit.
- "graspable": true if a gripper could pick this fruit right now, judging only from what is visible.
- "confidence_pct": 0-100. How sure you are that this really is a strawberry.
- "description": one or two sentences of plain language describing this fruit: its colour and colour pattern, what is occluding it and where, how big it looks, where it sits on the plant, and anything else a picker would want to know. This is free text - write what you actually see, not a restatement of the numbers.

Report EVERY strawberry you can see, whatever its colour. This is an inventory, not a picking decision: do NOT filter by ripeness and do NOT omit a fruit just because it is green, small, or partly hidden. Partly occluded fruit matters as much as fully visible fruit - include a fruit if you can see any part of it, and say how much is hidden in occlusion_pct. Order the list from largest to smallest apparent size. Do not invent fruit that is not there.

Do not label fruit as 'ripe' or 'unripe' - report the continuous redness and occlusion numbers and describe the fruit in words instead.

Each strawberry must carry EXACTLY the nine fields listed above - no more, no fewer, none renamed. If you have nothing for a field, still include it. Put any extra remarks inside description rather than adding a field.

Reply with ONLY a JSON object and nothing else - no prose, no code fences:
{"strawberries": [{"bbox": [x1, y1, x2, y2], "polygon": [[x, y], [x, y], ...], "redness_pct": 0, "occlusion_pct": 0, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 0, "description": "..."}, ...]}
```

### Metrics

The following metrics are computed for the epxeriments:

**Intersection over Union (IoU)** measures overlap between a predicted polygon (rasterised) and a ground-truth instance mask. A prediction counts as a true positive at IoU ≥ 0.50; matching is greedy in descending confidence order, each label matched at most once. Unmatched predictions are false positives; unmatched labels are false negatives.

- **Precision** = TP / (TP + FP); **Recall** = TP / (TP + FN); **F1** = 2PR / (P + R).
- **AP50** summarizes the precision–recall curve at IoU 0.50; **mAP50–95** averages AP across IoU thresholds 0.50–0.95 in steps of 0.05. (Note that the VLM output do not have the same confidence notation as a trained deeplearning detector, the confidance values are textual output from the model)
- **Mean matched mask IoU** is the average IoU of matched prediction–label pairs — the same quantity the StrawDI paper reports as mean per-instance IoU (I²oU).
- **Count MAE** is the mean absolute per-image error in fruit count — inventory error, not localization quality.

Our scorer uses all-points interpolation, not the full COCO protocol's 101 recall thresholds [11]. F1 is our primary metric: the VLMs' reported confidences cluster near 100, so AP ranking is sensitive to ties. 

## Results: instance segmentation on the dataset's own benchmark

Before the numbers, here is what the output actually looks like. GPT-6 Astra's visible-surface polygons on six test scenes — ripe and unripe fruit, dense clusters, and heavy occlusion. **What surprised me most is how close many of these predictions look to ground-truth-quality annotations: when inspecting the apparent “errors” manually, a number of them seem to reflect differences in annotation standards or missed labels rather than clear detection failures. In these examples, the GPT-6 Astra's model output is 'ground truth level'**

![Six StrawDI test scenes with GPT-6 Astra's predicted visible-surface polygons against the ground-truth masks.](assets/fruit_detection_is_solved_by_vlms/gpt_gallery.png)

*Figure 1. GPT-6 Astra, zero-shot, on six StrawDI test scenes. White fill: ground-truth instance masks; green: predictions matched at mask IoU ≥ 0.50; coral: unmatched predictions; MISS: labelled fruit the model never found. Note how the outlines follow occluder edges rather than convex hulls. Scenes selected to show the range of behaviour — including misses (images 2085, 2532, 926) — not a random sample.*

| Model | Precision @50 | Recall @50 | F1 @50 | F1 @75 | AP50 | mAP50–95 | Mean matched mask IoU | Count MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kimi K3 | 77.7% | 64.8% | 70.7% | 39.9% | 62.0% | 30.9% | 0.75 (median 0.77) | 1.04 |
| GLM | 75.0% | 61.0% | 67.3% | 28.4% | 58.2% | 24.3% | 0.72 (median 0.73) | 1.12 |
| DeepSeek | 73.1% | 59.9% | 65.9% | 26.2% | 55.8% | 22.6% | 0.71 (median 0.72) | 1.11 |
| **GPT-6 Astra** | **91.9%** | **83.8%** | **87.6%** | **75.1%** | **83.5%** | **60.1%** | **0.86** (median 0.89) | **0.75** |

*All metrics at mask IoU; 200/200 images scored for every model.*

GPT-6 Astra matches **948 of 1,132 labelled fruits** (84 unmatched predictions, 184 missed labels), exceeding the runner-up Kimi K3's F1 by **17.0 percentage points**. All four configurations returned valid responses for every test image, so no common-subset correction is needed.

![Segmentation precision, recall, F1, and F1 across IoU thresholds for the four configurations.](assets/fruit_detection_is_solved_by_vlms/seg_scores.png)

*Figure 2. Left: precision, recall, and F1 at mask IoU ≥ 0.50. Right: F1 as the required IoU tightens — GPT's advantage grows with the localization requirement.*

The same scene makes the gap visible without any numbers — here all four models answer the identical prompt on one crowded test image:

![The four models' segmentation output on the same crowded strawberry scene, StrawDI test image 2532.](assets/fruit_detection_is_solved_by_vlms/same_scene_seg.png)

*Figure 3. StrawDI test image 2532, twenty labelled instances. GPT-6 Astra finds fifteen with one unmatched prediction; the other three models find five to eight and over-segment the cluster. Illustrative example, not a representative sample.*

### Comparison with the dataset's own benchmark

The StrawDI paper's benchmark is exactly this task on exactly this split: Mask R-CNN trained on the 2,800 StrawDI training images reaches mean segmentation AP **45.36** and mean per-instance IoU (I²oU) **87.70** on the 200-image test set. [Pérez-Borrero et al., 2020][paper]

| Method | Task-specific training | Split | Segmentation mAP | Mean instance IoU |
| --- | --- | --- | ---: | ---: |
| Mask R-CNN (original paper) | 2,800 StrawDI images | test (200 images) | 45.36 | 87.70 |
| Kimi K3, zero-shot | none | test (200 images) | 30.9 | 74.9 |
| **GPT-6 Astra, zero-shot** | **none** | **test (200 images)** | **60.1** | **86.0** |
| GPT-6 Astra, zero-shot | none | val (100 images) | 61.3 | 86.9 |

*The paper's numbers are literature reference points, not reruns: the AP protocol differs (all-points interpolation here) and our polygons are limited to 32 vertices. But the task, the output representation, the ground truth, and the test split are the same. The validation row comes from an earlier 100-image batch of the same configuration, included to show the result replicates across splits.*

**A general-purpose VLM with no task-specific training exceeds the fully supervised Mask R-CNN's segmentation mAP on its own test split (60.1 vs 45.36) and lands within two points of its mean instance IoU (86.0 vs 87.7)** — With the validation-split run (61.3 / 86.9) showing the same profile, this is the strongest evidence presented in this article. The other VLMs are still far behind (best non-GPT 6 Astra: Kimi K3 at mAP 30.9 and mean IoU 74.9), suggesting that this level of performance is, for now, a frontier-model capability rather than a general property of VLMs. At the same time, we look forward to seeing whether the next generation of VLMs from other providers, including open-source models, can approach or match this level of performance.

Small fruit remain challenging — but Astra is already much stronger

Small fruit are still the most difficult category, but GPT-6 Astra performs substantially better than the other VLMs. Its size-stratified mask recall is 338/338 for large fruit, 513/542 for medium fruit, and 97/252 for small fruit (38.5%). While the recall on small fruit is clearly lower than on medium and large fruit, it is still striking compared with the other models, which detect only 7–14 of the 252 small instances.

This suggests that GPT-6 Astra has made a significant step forward even on small-object detection, although very small and weakly visible fruit remain one of the clearest failure modes. In other words, small fruit are no longer simply a case where all VLMs fail similarly — Astra shows a substantial capability gap here as well.

![Recall by visible instance size for the four configurations.](assets/fruit_detection_is_solved_by_vlms/size_recall.png)

*Figure 4. Recall at mask IoU ≥ 0.50 by visible mask area (small < 1,024 px²; medium 1,024–9,215 px²; large ≥ 9,216 px²). Labels give matched/total instances.*

### An unexpected caveat: silent provider-side routing

One episode during this evaluation is worth reporting because it nearly contaminated the GPT-6 Astra results. Every call was issued with the same recorded model identifier (`gpt-6-astra` via the Codex CLI), and every batch passed the synthetic vision-delivery control. Yet one intermediate retry batch returned responses that were perfectly schema-valid — and visibly wrong on the image: much of the fruit missing, clusters over-segmented, outlines that no longer follow occluder edges. Two of the affected frames, exactly as they came back:

![Silent routing comparison, frame 995: suspect batch vs re-requested.](assets/fruit_detection_is_solved_by_vlms/routing_995.png)

*Figure 5. StrawDI test image 995. Left: the suspect batch's answer — same prompt, same recorded model identifier, schema-valid. Right: the same frame re-requested under identical settings.*

![Silent routing comparison, frame 926: suspect batch vs re-requested.](assets/fruit_detection_is_solved_by_vlms/routing_926.png)

*Figure 6. Test image 926 — the same pattern.*

Re-requesting the affected frames under identical settings immediately recovered the dense, clean inventories shown on the right. Our suspicion is that the provider silently routed those requests to a different, weaker model while the request-side bookkeeping still said `gpt-6-astra`. The affected records were excluded wholesale and re-run; every GPT-6 Astra number in this article comes from the corrected batch. The lesson generalizes beyond this incident: with API-routed frontier models, "the request named the model" is not evidence that the model answered — and no schema validator will catch a wrong-model response.

## Zero-shot evaluation on private data

A public benchmark cannot prove out-of-distribution performance: StrawDI has been downloadable for years, so these images may be in pretraining data. As a check beyond the public benchmark, we ran the identical nine-field census prompt — byte-for-byte the same prompt and schema — on **two private, deliberately chaotic scenes from our own collection**: **private-1**, a dense hanging truss from a greenhouse row, and **private-2**, a hand-held close-up of an overlapping cluster. Neither image has been publicly released, and neither has any ground truth, so this is qualitative: what does the model's inventory look like when the scene is messy and the data cannot have been memorized as a benchmark?

Eight configurations ran both scenes: GPT-6 Astra (low effort), the six **GPT-5.6** configurations (the `luna` and `sol` variants, each at low, medium, and high reasoning effort), and GLM (maximum effort) as a non-GPT reference.

| Configuration | `private-1` | `private-2` |
| --- | ---: | ---: |
| **GPT-6 Astra (low)** | **23** | **21** |
| GPT-5.6 luna low / medium / high | 18 / 21 / 19 | 14 / 16 / 17 |
| GPT-5.6 sol low / medium / high | 17 / 21 / 21 | 18 / 17 / 18 |
| GLM-5.3-flash | 16 | 14 |

*Fruit reported per scene; counts alone say nothing about correctness — the grids below are the evidence.*

The visual verdict is unambiguous, and it indicates two things. First, **GPT-6 Astra is uniquely good**: on both scenes its detections are essentially perfect — almost every fruit I can find by eye is outlined (there are actually some missing but are very small ones hard to tell if they are fruits or not), the outlines closely follow the visible surface, and nothing is invented. Second, **the GPT-5.6 family is fairly flat**: from `luna` medium all the way to `sol` high (`luna` low is apparently the weakest), the five runs produce very similar results — similar counts, similar outlines, and similar misses — so increasing reasoning effort or switching variants does not close the gap to Astra. Whatever changed between the 5.6 and 6 generations appears to matter more here than anything the reasoning-effort setting can provide.


![All eight configurations on private-1, the greenhouse-truss scene.](assets/fruit_detection_is_solved_by_vlms/chaos_private-1.png)

*Figure 7. Scene `private-1` (dense hanging truss). Each polygon is coloured by the model's own reported redness (green → red). GPT-6 Astra (top panel) resolves the cluster fruit-by-fruit; the six GPT-5.6 panels are nearly interchangeable, and GLM undercounts. Qualitative — no ground truth exists on this scene.*

![All eight configurations on private-2, the hand-held scene.](assets/fruit_detection_is_solved_by_vlms/chaos_private-2.png)

*Figure 8. Scene `private-2` (hand-held close-up, heavy overlap). Same pattern: GPT-6 Astra's inventory is complete and cleanly separated; the GPT-5.6 panels again look alike from luna-low to sol-high.*

Even the runner-up models' output shows the flexibility that makes this paradigm interesting. On `private-2`, GLM distinguishes a "[p]rominent bright, evenly red conical fruit hanging in the centre-right on a long bare green stem … [a]lmost fully exposed and an easy pick" (redness 96%) from the "[v]ery small green fruit with a green calyx hanging low between larger neighbours, partly screened by stems; too small and crowded for a gripper" beside it (redness 2%), flags which fruit a gripper could pick right now, and traces each visible surface around the occluding leaves — all from the same prompt that produced the StrawDI numbers, with no task-specific anything. These are observed outputs, not independently validated horticultural or grasping judgments; the private scenes establish transfer, not accuracy. But the connection between an object, its appearance, and a proposed action is the useful feature: an detection supports counting, a colour description supports user-defined ripeness rules, and an occlusion description can guide which fruit an operator inspects next.

## Current limitations and future directions

### Latency and cost are far from a local detector

Recorded mean call time per scheduled image: **76.2 s (Kimi K3), 89.4 s (GLM), 25.5 s (DeepSeek), 48.4 s (GPT)**, including CLI overhead and recorded retries. For scale, Ultralytics reports **1.5–11.3 ms** for YOLO11 on a T4 GPU with TensorRT — different hardware, resolution, and workload, so not a controlled comparison, but the gap is four orders of magnitude. [12]

GPT averaged **15,336 input and 944 output tokens per scheduled image**. At OpenAI's listed standard rates — $10 / $1 / $50 per million input / cached-input / output tokens, checked September 15, 2026 — that is **≈$0.20 per image** uncached or **≈$0.12** with recorded cache reads ($40.1 vs $23.9 per 200 images); GLM's recorded batch cost was $22.86 per 200 images at its provider's rates. These are token-based estimates, not invoices. [13] For an occasional scene audit this may be acceptable; for continuous video, the first things I would test are request frequency, output verbosity, and distillation into a local model.

![Recorded input and output token usage and mean call time for the four configurations.](assets/fruit_detection_is_solved_by_vlms/tokens_latency.png)

*Figure 9. Means across all 200 scheduled images per model. Output includes reasoning tokens; GLM's input figure reflects its provider's accounting (cached tokens are not reported as input). Token accounting and CLI overhead differ across providers — these are operational measurements of the configurations used. Control calls excluded.*

### Edge deployable VLMs need their own evaluation

We have not demonstrated VLM detection on an edge device. The natural next experiment is to run locally servable VLMs such as **Qwen3.8-27B** and smaller vision-capable Qwen variants — under the same protocol, measuring memory, power, latency, and localization after quantization on the intended hardware. A model that fits in memory still has to answer within the robot's operating budget. [14]

### Looking Forward to More Capable VLMs
Another important limitation today is that this level of performance is still concentrated in only a small number of frontier models. In our experiments, GPT-6 Astra is clearly ahead of the other VLMs we tested, which means that the current result is still strongly dependent on a single provider and model family.

I expect this to change quickly. As the next generation of VLMs arrives — including models from other commercial providers and open-source communities — it will be important to see whether Astra-level visual detection becomes a broader capability rather than an isolated result. Having multiple models reach this level would improve reproducibility, reduce provider dependence, and make deployment choices more flexible in terms of cost, latency, privacy, and local inference.

## Future questions and conclusion

This experiment answers one question, but it raises several others that I think are more interesting going forward.

1. **How much of this capability can eventually run locally?**
   GPT-6 Astra is powerful, but it is still far from the latency, cost, and deployment characteristics of a local detector. An important next step is to test future edge-capable VLMs on the same images and prompts and see how much of this detection and spatial reasoning capability can survive local deployment.

2. **Which of the richer outputs are actually useful?**
   A VLM can return much more than a class label or mask: descriptions, redness, occlusion, calyx and stem visibility, graspability, confidence, and potentially many other attributes. The next question is which of these outputs are reliable enough to matter in real applications. They should ultimately be validated against expert judgments, downstream decision making, and, for robotic harvesting, actual robot outcomes.

3. **Can a strong VLM teach a much smaller model?**
   If a frontier VLM can generate high-quality detections and polygon masks zero-shot, it may also serve as a data engine for training smaller and faster task-specific models. One possible direction is to use VLM predictions as candidate annotations, review uncertain cases, and train a compact student model that can run locally. Whether such a student can retain the generalization ability of the teacher across new farms and environments remains an open question.

So, is fruit detection solved?

The strongest result in this article is that a general-purpose VLM, given only raw pixels and a written instruction, can produce strawberry detections with both bounding boxes and pixel-level outlines that, on the StrawDI test split, exceed the segmentation mAP of a fully supervised specialist and essentially match its per-instance IoU — all without task-specific training.

That changes how I would approach a new fruit-perception problem. Instead of assuming that every new orchard, crop variety, or environment requires another cycle of data collection, annotation, and model training, I would now first test what a frontier VLM can already do zero-shot.

At the same time, important questions remain: performance on the smallest fruit, latency and cost, robustness across truly independent farms and environments, local deployment, and whether this level of capability will become common across future VLMs rather than remaining concentrated in a few frontier models.

**Fruit detection is not universally solved. But the starting point has changed dramatically: instead of asking how to train a detector from scratch, we can increasingly ask how far a general-purpose visual model can already take us — and what still needs to be built on top of it.**


---


**Dataset acknowledgement:** Kindly provided by the StrawDI Team (see [the official dataset website][strawdi]).

## References

[1] Fei, Z., Olenskyj, A. G., Bailey, B. N., & Earles, M. (2021). Enlisting 3D crop models and GANs for more data-efficient and generalizable fruit detection. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 1269–1277.

[2] Wang, Y., Fei, Z., Li, R., & Ying, Y. (2025). Learn from foundation model: Fruit detection model without manual annotation. *Pattern Recognition*, 112799.

[3] Wang, Y., Li, W., Ying, Y., & Fei, Z. (2026). GEAR-Seg: A Grounded Explainable Agent for Reasoning Segmentation and Data Engine. *arXiv preprint* arXiv:2607.00544.

[4] Ranasinghe, K., et al. (2024). Learning to Localize Objects Improves Spatial Reasoning in Visual-LLMs. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

[5] Schaumloffel, L., et al. (2026). Mechanisms of Object Localization in Vision-Language Models. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

[6] SOUBench: Benchmarking Small-Object Understanding in Multimodal Large Language Models. (2026).

[7] OpenAI. (2026). GPT-6 Astra: The Next Generation in Intelligence for Work. https://openai.com/index/gpt-6-astra-next-generation-work/

[8] Su et al. (2026). GPT-6 Astra as an Embodied Policy. https://anonymous-report-421.github.io/public-website/?lang=en&view=1

[9] Robocurve. (2026). GPT-6 Astra on Robotic Manipulation. https://openai.robocurve.org/gpt-6-astra/

[10] Pérez-Borrero, I., Marín-Santos, D., Gegúndez-Arias, M. E., & Cortés-Ancos, E. (2020). A fast and accurate deep learning method for strawberry instance segmentation. *Computers and Electronics in Agriculture*, 178, 105736. https://doi.org/10.1016/j.compag.2020.105736

[11] COCO Consortium. Common Objects in Context (COCO) evaluation API (`cocoeval.py`). https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

[12] Ultralytics. YOLO11 performance benchmarks. https://docs.ultralytics.com/models/yolo11/

[13] OpenAI. GPT-6 Astra API pricing. https://developers.openai.com/api/docs/models/gpt-6-astra

[14] Qwen Team. Qwen3.8 documentation. https://github.com/QwenLM/Qwen3.8/blob/main/README.md

[strawdi]: https://strawdi.github.io/
