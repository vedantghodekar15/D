# SurfaceGuard AI: Viva Study Guide (DL & GenAI Project 26T3)

Written for someone starting from zero. Read Part 1 first, then the milestones, then the submissions, then the viva Q&A at the end.

---

## PART 1: The competition in simple language

### 1.1 What is the problem?
A factory takes photos of metal or material surfaces as they move along a belt. Some photos show **defects** (scratches, dents, stains). We want a computer to look at a photo and **colour in exactly the pixels that are defective**.

Telling the computer "this photo has a defect" is *classification*. Colouring the exact defective pixels is **segmentation**. This competition is segmentation.

### 1.2 The data
- Each photo is **128 pixels tall and 800 pixels wide**, black and white (grayscale).
- There are **4 defect types**: `defect_a`, `defect_b`, `defect_c`, `defect_d`.
- **11,393 training photos**. Each photo has 4 rows in `train.csv`, one per defect type, so 45,572 rows in total.
- Each row has the answer (the "mask") written as text called **RLE**.
- A test set is hidden; we predict masks for it and upload a `submission.csv` to Kaggle.

### 1.3 What is RLE (run-length encoding)?
A mask is a 128×800 grid of 0s and 1s (1 = defect). That is 102,400 numbers, which is wasteful to store. RLE stores only "start position, length" pairs.

Example: `"5 3 20 2"` means "pixels 5, 6, 7 are defect; pixels 20, 21 are defect".

Two rules to remember (these are viva favourites):
1. Positions start at **1**, not 0.
2. The grid is read **column by column** (top to bottom, then the next column). This is called **column-major** or Fortran order (`order="F"` in numpy).

An empty mask is an empty string.

### 1.4 Important facts about the data (from EDA)
- **48.4% of images are clean** (no defect at all): 5,516 images.
- Images per defect type: `defect_c` 4,100, `defect_d` 1,707, `defect_a` 1,081, `defect_b` only 313. So the data is **imbalanced**: defect_b is rare.
- About 10.5% of images (1,198) have **more than one** defect type. `defect_c` and `defect_d` appear together most (705 images). So one image can have several labels: this is **multi-label**.
- Defects never overlap at pixel level between classes in this data.
- Only about **0.07% / 0.02% / 1.95% / 1.05%** of all pixels are defect for classes a/b/c/d. Defects are tiny compared with the background.

### 1.5 How is the score calculated? (the "official metric")
```
score = 0.70 × Foreground Dice  +  0.20 × False-Positive score  +  0.10 × Empty accuracy
```
- **Dice** measures overlap between predicted and true defect pixels. `Dice = 2 × overlap / (predicted size + true size)`. 1 means perfect, 0 means no overlap. "Foreground" Dice is only computed on images that really have a defect.
- **False-positive score**: on clean images, how little did you colour by mistake? Colouring nothing gives 1.
- **Empty accuracy**: what share of clean images did you leave completely empty?

**The trap (very important):** if you predict *nothing* everywhere, you get 0 + 0.2 + 0.1 = **0.30**. A "naive" Dice that gives 1 to empty-vs-empty would score such a model about **0.84**, which looks great but is useless. That is why we always judge with the official score, never the naive Dice.

Because 48% of images are clean, **avoiding false alarms matters as much as finding defects**.

### 1.6 The course rules
- Need **≥ 0.65 on Kaggle** with a trained model to qualify for the viva.
- Need at least 3 models: **one from scratch, one pretrained, one more of your choice**.
- All models tracked in **Weights & Biases (W&B)**.
- Compare ≥ 3 runs with common metrics.
- At least 2 of 5 milestones submitted.
- Viva: explain code and write a small snippet live.

### 1.7 Our results on Kaggle
| Version | What it was | Score |
|---|---|---|
| v1 | single U-Net, early try | 0.57295 |
| (M3 pretrained ResNet-34) | single model | 0.57355 |
| **v2** | ensemble of 3 models | **0.66572** |
| **v3** | ensemble of 3 models (newer tricks) | **0.66510** |

Both v2 and v3 cleared the 0.65 cutoff.

---

## PART 2: Words you must know (glossary)

| Term | Simple meaning |
|---|---|
| **Neural network** | A big function with many adjustable numbers (weights) that learns from examples. |
| **Training** | Showing the network images + correct answers and nudging weights to reduce mistakes. |
| **Epoch** | One full pass through all training images. |
| **Batch size** | How many images are processed at once (we used 16). |
| **Loss** | A number measuring how wrong the network is. Lower = better. |
| **Optimizer (AdamW)** | The rule that updates the weights using the loss. |
| **Learning rate (lr)** | How big each update step is. Too big: unstable. Too small: slow. |
| **Validation set** | Images kept aside (15–20%) to test how well the model does on unseen data. |
| **Overfitting** | Model memorises training images; training loss is low but validation is bad. |
| **Underfitting** | Model hasn't learned enough yet; both are bad. |
| **Pretrained model** | A model already trained on a huge dataset (ImageNet). We start from its knowledge instead of from zero. |
| **From scratch** | Random starting weights; learns everything from our data. |
| **Encoder / Decoder** | Encoder shrinks the image to understand it; decoder grows it back to a full-size mask. |
| **U-Net** | Encoder-decoder with **skip connections** that copy detail from encoder to decoder. Classic for segmentation. |
| **Skip connection** | A shortcut that passes fine detail (edges) straight to the decoder. |
| **Sigmoid** | Squashes a number to 0–1 (a probability). One per class → multi-label. |
| **Softmax** | Makes probabilities across classes add up to 1 (only one class allowed). Not suitable when an image has several defect types. |
| **BCE (binary cross-entropy)** | Loss comparing predicted probability with 0/1 truth, pixel by pixel. |
| **Dice loss** | Loss that directly rewards overlap; good when defects are tiny. |
| **AMP (mixed precision)** | Uses 16-bit numbers where safe: faster, less memory. |
| **EMA** | Keeps a smoothed average of weights over training; usually more stable. |
| **TTA** | Test-time augmentation: predict on flipped copies and average. |
| **Ensemble** | Several models vote (average) → fewer mistakes than one. |
| **W&B** | Website that records and plots every training run. |
| **Threshold** | Probability above which a pixel is called "defect" (default 0.5). |
| **Prior shift** | If you show defects more often during training, the model starts to expect defects more, so it makes more false alarms. |

---

## PART 3: Milestones

### Milestone 1: Understanding the data (RLE + EDA)
**Goal:** learn to read the data correctly.
What we did:
1. Wrote `rle_decode` and `rle_encode` (1-indexed, column-major) and checked a round trip.
2. Counted how many pixels/defects per image and per class.
3. Class prevalence: which defect is most / least common (`defect_c` 4,100 vs `defect_b` 313).
4. Co-occurrence matrix: how often two classes appear in the same image (`defect_c`+`defect_d` highest, 705).
5. Checked pixel-level overlap between classes (none).
6. Built a PyTorch `Dataset` that returns an image tensor and a 4-channel mask.
7. Tried a pretrained model (this is where the weight download once hung, so we added a timeout and fallback).

Key idea: the mask target is `float32`, shape `(4, 128, 800)`, because `BCEWithLogitsLoss` needs float targets.

### Milestone 2: First model from scratch
**Goal:** build a segmentation model ourselves and track it.
What we did:
1. Decoded an RLE grid by hand (small 8×8 example) to prove we understand column-major order.
2. Computed class prevalence, co-occurrence, correlation of class indicators.
3. Built the **UNet3**: a small U-Net with 3 encoder levels (32, 64, 128 channels), a bottleneck (256), and a mirrored decoder. About **1.93 million parameters**.
4. Trained 5 epochs with **BCE loss, AdamW, lr 3e-4, batch 16** and logged to W&B.
5. Result: validation BCE went from **0.1057 to 0.0291**; train 0.0279 → no sign of overfitting.

Why BCE looks tiny: ~98% of pixels are background, so loss is low even for a mediocre model. It is a weak quality indicator.

### Milestone 3: Pretrained vs scratch, architectures
**Goal:** compare models fairly (same data, same settings, 5 epochs, 1-channel input).
Results (official score on validation):
| Model | Official score |
|---|---|
| U-Net ResNet-34, scratch | 0.475 |
| U-Net ResNet-34, pretrained | 0.580 |
| U-Net EfficientNet-B0, pretrained | **0.631** |

Take-aways:
- **Pretraining gives the biggest jump** (+0.105 for the same architecture).
- EfficientNet-B0 is **smaller (18.18 M parameters) yet better** thanks to depthwise-separable convolutions (MBConv).
- **1-channel input trick:** pretrained models expect RGB. We add up the three colour filters of the first layer into one. A grey image gives the identical result as feeding it 3 times, but moves 3× less data to the GPU.
- **Sigmoid, not softmax**, because an image can have several defects at once.
- **ASPP (in DeepLabV3+):** looks at the image at several zoom levels at the same time using dilated convolutions.
- Overfitting check: compare validation loss − training loss. A small gap and falling validation loss = healthy.
- Memory measurement caveat: first-run GPU memory numbers were inflated by cuDNN auto-tuning.

### Milestone 4: Training improvements and experiments
**Goal:** learn the "tricks" that improve training and how to test them honestly. Work was done on the first 1,200 images (sorted by id), split 960 train / 240 validation.

**Q1 Imbalance counts (1,200 subset):** 578 clean images; positives per class 107 / 35 / 438 / 181; 15 different label combinations; largest group 578, smallest 1.

**Q2 Pixel imbalance for defect_b:** only **0.0160%** of pixels are defect; mean area when present 561.5 px; negatives per positive pixel ≈ 6,251 → a `pos_weight` of about that size would be needed to balance BCE.

**Q3 Bounded square-root sampling:** instead of showing every image equally, show rare combinations more often. Weight of group g = `min( sqrt(n_clean / n_g), 5 )`. The cap of 5 stops tiny groups being repeated too much. Effect: clean share of draws 48.17% → 28.81%; defect_b appears ~3× more.

**Q4 Warmup + cosine scheduler:**
- Warmup: learning rate rises linearly from 0 to 3e-4 over the first 60 steps (stops early chaos).
- Cosine: then it decays smoothly to 0 (big steps early, fine steps late).
- Formula: `m(t) = t/warmup` for `t < warmup`, else `0.5·(1 + cos(π·(t−warmup)/(total−warmup)))`; lr = 3e-4 × m(t).
- Values: lr(15) = 7.5e-5, lr(270) = 1.5e-4, lr(400) = 2.61e-5.
- **Common bug:** calling `scheduler.step()` once per epoch instead of once per optimizer step. After 8 epochs the schedule is at step 8, lr = 4e-5, never leaves warmup → underfitting.

**Q5 Early stopping:** stop when the validation score hasn't improved by more than `min_delta` for `patience` epochs, then restore the best checkpoint. With patience 3, min_delta 0.002 the example stops at epoch 10 and restores epoch 7 (0.671).

**Q6 Memory and gradient accumulation:** with AMP, weights, gradients and Adam's two buffers stay in 32-bit (4 values × 4 bytes × 25 M params = 400 MB). Only activations shrink. Gradient accumulation = run several small batches, add their gradients, update once, to mimic a bigger batch when memory is short. Divide each micro-batch loss by the number of accumulation steps before `backward()`.

**Q7 Naive Dice trap:** all-empty prediction → naive 0.8415 but official 0.3000 (see 1.5).

**Q8 Training dynamics (R1 run, warmup + cosine):** best epoch 6, official 0.447, validation−train gap −0.001: still improving, not overfitting.

**Q9 Defect-size bins:** 5 pairs of size 1–100 px, 99 of 101–500 px, 657 above 500 px; 99.24% of defect pixels are in the largest bin.

**Q10 One-factor-at-a-time (OFAT) experiments** (change one thing, keep everything else equal):
| Run | Change | Best official | False-alarm rate |
|---|---|---|---|
| R0 | baseline (seed 42) | 0.426 | 12.8% |
| R0b | same, seed 7 | 0.444 | 14.0% |
| R1 | + warmup/cosine | 0.447 | 8.3% |
| R2 | + bounded sampler | 0.474 | 19.3% |

- **Noise band** = |R0 − R0b| = 0.018. A change only counts as real if the gain is bigger than ~2× this (0.036).
- R1 (+0.021): within noise. R2 (+0.049): real.
- But R2 raised false alarms 12.8% → 19.3% (**prior shift**): the model now expects defects more. Fix by tuning thresholds, minimum areas, or a classification gate.
- Why OFAT? Changing several things at once makes it impossible to know which one helped.

---

## PART 4: Kaggle submissions v2 and v3 in detail

### 4.1 Shared pipeline (both versions)
1. **Load** all 11,393 images into memory as numbers (uint8).
2. **Split** 85% train / 15% validation, stratified by the label combination so rare classes appear in both.
3. **Build 3 pretrained models** with `segmentation_models_pytorch` and train each separately.
4. **Predict** on validation and test using all 3 models and flipped copies (TTA); average probabilities.
5. **Tune post-processing on validation**, per class, to maximise the official score:
   - *threshold* (cut-off for "defect pixel"),
   - *minimum component area* (delete tiny blobs),
   - *minimum total pixels* (drop a mask that is too small),
   - *classification gate* (only keep a mask if the image-level head says "defect present").
6. **Encode to RLE** and write `submission.csv` (empty masks as a single space).

### 4.2 The tricks and why they work
| Trick | Why it helps |
|---|---|
| Pretrained encoders | Start with edge/texture knowledge → faster learning, higher score. |
| 4 sigmoid outputs | Multi-label: an image can have several defect types. |
| Loss = BCE + Dice | BCE gives stable gradients; Dice fights the tiny-defect imbalance. |
| Auxiliary classification head | Predicts "is this class present?"; used as a gate to remove false alarms on clean images (48% of data). |
| AMP | Faster, less memory. |
| EMA of weights | Smoother, more reliable final model. |
| Lower learning rate for encoder | Pretrained part changes gently; new decoder learns faster. |
| Rare-class sampling | Rare `defect_b` is seen more often. |
| Flip TTA | Several views → averaged → less noise. |
| Post-processing tuned on the real metric | Directly trades defect recall against false alarms. |

### 4.3 Version 2 (score 0.66572)
- **Models:** U-Net + ResNet-34, FPN + EfficientNet-B3, U-Net + ResNet-50.
- **Training:** 30 epochs, AdamW, peak lr 1e-3, **OneCycle** schedule (warm up then decay), encoder lr = 1/3 of the rest.
- **Input:** grey image repeated into 3 channels.
- **Sampling:** oversample images containing rare classes.
- **TTA:** original + horizontal flip + vertical flip.
- **Tuning grids:** thresholds 0.3–0.9, min area 0–800, min total 0–2000, classification gate 0–0.85.

### 4.4 Version 3 (score 0.66510)
The notebook adds Milestone 3 and Milestone 4 ideas on top:
- **Models:** U-Net + ResNet-34, FPN + EfficientNet-B3, **DeepLabV3+ + ResNet-34** (replaces U-Net ResNet-50; DeepLabV3+ uses ASPP for multi-scale context).
- **Input:** 1 channel (summed first-layer filters).
- **Scheduler:** linear warmup (2 epochs) + cosine decay, stepped **every optimizer step**.
- **Sampler:** bounded square-root sampler (Milestone 4).
- **EMA** decay 0.999; **weight decay** 1e-2.
- **Early stopping** on the official validation score (patience 8, min_delta 0.002).
- **Gradient accumulation** option (`accum_steps`).
- **Extra monitoring:** false-alarm rate per epoch; defect-size recall report after tuning.

### 4.5 Why is v3 not better than v2?
The scores differ by 0.0006. In Milestone 4 we measured that two runs of the *same* setup can differ by 0.018, so 0.0006 is a **tie**, not a real difference. Also v2 and v3 share two of three models, so the ensembles are not very different.

---

## PART 5: Code you may be asked to write in the viva

**RLE decode**
```python
import numpy as np
def rle_decode(rle, h=128, w=800):
    m = np.zeros(h * w, dtype=np.uint8)
    if isinstance(rle, str) and rle.strip():
        v = np.array(rle.split(), dtype=np.int64).reshape(-1, 2)
        for start, length in v:
            m[start - 1 : start - 1 + length] = 1      # 1-indexed
    return m.reshape((h, w), order="F")               # column-major
```

**Dice score**
```python
def dice(pred, true):
    inter = (pred & true).sum()
    return 1.0 if pred.sum() + true.sum() == 0 else 2 * inter / (pred.sum() + true.sum())
```

**Warmup + cosine multiplier**
```python
import math
def warmup_cosine(step, warmup, total):
    if step < warmup:
        return step / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))
```

**Early stopping**
```python
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.bad = -float("inf"), 0
    def step(self, score):
        if score > self.best + self.min_delta:
            self.best, self.bad = score, 0
            return False
        self.bad += 1
        return self.bad >= self.patience
```

**Bounded sqrt weights**
```python
import numpy as np
def bounded_sqrt_weights(groups, cap=5.0):
    labels, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    n_clean = counts[list(labels).index("clean")]
    return np.minimum(np.sqrt(n_clean / counts), cap)[inv.reshape(-1)]
```

**A training step with AMP + gradient accumulation**
```python
scaler = torch.amp.GradScaler("cuda")
for j, (x, y) in enumerate(loader):
    with torch.autocast("cuda"):
        loss = loss_fn(model(x.cuda()).float(), y.cuda())
    scaler.scale(loss / accum_steps).backward()
    if (j + 1) % accum_steps == 0:
        scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
```

---

## PART 6: Likely viva questions with short answers

1. **What is the task?** Multi-label semantic segmentation of 4 defect types on 128×800 grayscale strips.
2. **Why segmentation and not classification?** The score rewards pixel-level overlap (Dice), not just detecting presence.
3. **How is RLE read?** 1-indexed, column-major pairs of (start, length).
4. **Why sigmoid, not softmax?** One image can contain several defect types; sigmoid gives an independent probability per class.
5. **Why BCE + Dice?** BCE is stable, Dice handles extreme imbalance and directly rewards overlap.
6. **Why pretrained?** Already knows edges and textures; +0.105 over scratch in our test.
7. **Why not just use accuracy?** 84–98% of pixels/pairs are empty; predicting nothing looks great but scores only 0.30 on the official metric.
8. **What is overfitting and how did you check?** Train loss much lower than validation loss; we compared the gap and the validation curve.
9. **Explain U-Net.** Encoder shrinks the image to capture context; decoder expands to a mask; skip connections restore fine detail.
10. **What do EMA, AMP, TTA, early stopping do?** Smooth weights; speed/memory via 16-bit; average flipped predictions; stop when validation stops improving.
11. **What is the classification gate?** An extra head predicting if a class is present; if its probability is low, the mask is removed, cutting false alarms on clean images.
12. **What is prior shift?** Oversampling defects makes the model expect defects more, raising false alarms.
13. **How did you tune post-processing and why on validation?** Grid search per class on the official metric using validation data; the result is slightly optimistic compared to the leaderboard.
14. **Why is v3 ≈ v2?** The difference (0.0006) is far below the noise (0.018) and the ensembles overlap.
15. **What is a noise band?** The score difference between two runs of the same setup with different seeds; used to decide if an improvement is real.
16. **What would you improve?** More diverse ensemble members, k-fold cross-validation, retrain on all data, a loss for small defects, per-image post-processing.
17. **Which three models satisfy the course rule?** Scratch: UNet3 (and a scratch ResNet-34 control). Pretrained: U-Net ResNet-34 / EfficientNet-B0. Additional: FPN-EfficientNet-B3 and DeepLabV3+ ensemble members.
18. **What is your best Kaggle score?** 0.66572 (v2), above the 0.65 cutoff.
19. **What problems did you face?** Pretrained-weight download hang (added timeout), memory from storing probabilities (compute metrics per batch), stratifying a class with one member (merge rare groups).
20. **Where is W&B used?** Project `dl-genai-project1`: every run logs losses, official score, Dice and F1.

---

## PART 7: Revision checklist for tomorrow
- [ ] Say the problem in one sentence and explain RLE out loud.
- [ ] Recite the metric formula and the "predict nothing = 0.30" trap.
- [ ] Know the numbers: 11,393 images, 48.4% clean, 4 classes, 0.65 cutoff, v2 0.66572, v3 0.66510.
- [ ] Explain scratch vs pretrained: 0.475 → 0.580 → 0.631.
- [ ] Explain each Milestone 4 trick in one line and the OFAT table.
- [ ] Be able to write `rle_decode`, `dice`, `warmup_cosine`, `EarlyStopping` from memory.
- [ ] Open the Kaggle notebooks once and trace: data → split → model → train → tune → submit.
