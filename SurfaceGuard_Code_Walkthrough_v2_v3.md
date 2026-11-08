# Code Walkthrough: Kaggle Submission v2 and v3

How to use this: v3 is explained cell by cell (it is the more complete notebook). Section 12 then lists exactly what v2 does differently, so you can explain both. Each part has **What it does**, **Why**, and a **Say in the viva** line.

Notation: `B` = batch size, `H, W = 128, 800`, `N` = number of images (11,393), `4` = number of defect classes.

---

## 0. Big picture (one paragraph to memorise)
The notebook (1) loads every image and mask into memory, (2) splits into train and validation, (3) trains three different pretrained segmentation networks one after another, saving each one's best checkpoint, (4) combines the three on validation (and flipped copies), (5) searches, per class, for the cut-offs that give the best official score, (6) predicts the test set with those cut-offs and writes `submission.csv`. After each model finishes, it writes an interim `submission.csv`, so a crash or timeout still leaves a usable file.

**Cell order in the v3 notebook**
1. `pip install` → 2. imports/helpers → 3. `Config` → 4. RLE functions → 5. data helpers → 6. W&B setup → 7. load data + split + sampling probabilities → 8. model factory → 9. loss/metric/augmentation → 10. training code (`EMA`, schedulers, `train_model`) → 11. inference/tuning/submission code → 12. the run cell.

---

## 1. Imports and helpers
```python
H, W = 128, 800
CANVAS = H * W                      # 102,400 pixels
CLASSES = ["defect_a", "defect_b", "defect_c", "defect_d"]
IMG_MEAN, IMG_STD = 0.449, 0.226    # ImageNet mean/std averaged over RGB
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```
- **Libraries:** `numpy/pandas` for data, `cv2` for reading images, `torch` for the networks, `segmentation_models_pytorch` (`smp`) for ready-made architectures, `sklearn` for the split.
- **`set_seed(seed)`** fixes Python, NumPy and PyTorch random numbers so reruns are repeatable (the GPU can still add tiny differences).
- `torch.backends.cudnn.benchmark = True` lets the GPU pick the fastest convolution algorithm (faster, slightly non-deterministic).
- **Say in the viva:** "Mean 0.449 and std 0.226 are the ImageNet RGB statistics averaged, because our pretrained encoders were trained on images normalised that way."

---

## 2. `Config` (the control panel)
A Python `dataclass` holding every setting, so a run can be reproduced from one cell.

| Field (v3) | Value | Meaning |
|---|---|---|
| `models` | U-Net/ResNet-34, FPN/EfficientNet-B3, DeepLabV3+/ResNet-34 | the 3 ensemble members `(architecture, encoder)` |
| `epochs` | 35 | maximum epochs (early stopping may end sooner) |
| `batch_size` | 16 | images per step |
| `accum_steps` | 1 | gradient accumulation (micro-batch = 16 / accum) |
| `lr` | 1e-3 | peak learning rate for decoder/head |
| `encoder_lr_scale` | 0.3 | encoder lr = 3e-4 (pretrained part moves gently) |
| `weight_decay` | 1e-2 | AdamW regularisation |
| `warmup_epochs` | 2 | linear warm-up, then cosine decay |
| `patience`, `min_delta` | 8, 0.002 | early stopping rule |
| `val_frac` | 0.15 | 15% validation |
| `seed` | 42 | randomness |
| `in_channels` | 1 | grayscale input |
| `use_aux_head` | True | extra "is the class present?" classifier |
| `cls_loss_weight` | 0.5 | weight of that extra loss |
| `bce_weight`, `dice_weight` | 1.0, 1.0 | loss mix |
| `sampler_cap` | 5.0 | cap in bounded sqrt sampler |
| `ema_decay` | 0.999 | EMA smoothing |
| `tta` | (no flip), (h-flip), (v-flip) | test-time augmentation views |
| `wandb_prefix` | "v3" | names W&B runs and checkpoints |

Post-processing search grids (global constants):
```python
THR_GRID   = [0.2, 0.3, ..., 0.9]                       # probability thresholds
AREA_GRID  = [0, 50, 100, 200, 400, 800]                # minimum connected-blob size
TOTAL_GRID = [0, 100, 300, 600, 1000, 2000, 10**9]      # minimum total pixels (10**9 = "always empty")
CLS_GRID   = [0.0, 0.3, 0.5, 0.7, 0.85]                 # classification-gate threshold
```

---

## 3. RLE and image reading
**`rle_decode(rle)`**
1. Start with a flat array of `102,400` zeros.
2. Split the string into numbers and reshape to `(-1, 2)` = (start, length) pairs.
3. For each pair, set `flat[start-1 : start-1+length] = 1` (the `-1` because RLE is 1-indexed). `max/min` clip runs that exceed the canvas.
4. `reshape((H, W), order="F")`: **column-major** reshape, which is why the mask lands correctly.

**`rle_encode(mask)`**
1. Flatten column-major (`order="F"`), pad with a 0 on both ends.
2. `np.where(padded[1:] != padded[:-1])` finds positions where the value changes (starts and ends of runs; `+1` makes it 1-indexed).
3. `changes[1::2] -= changes[0::2]` turns every (start, end) into (start, length).
4. Join as a string.

**`read_gray(path)`** reads the JPEG as grayscale (`cv2.IMREAD_GRAYSCALE`) → `uint8` array `(128, 800)`.

A round-trip assert (`decode(encode(mask)) == mask`) proves the two functions are inverses.

**Say in the viva:** "A mask is 1 where the defect is. RLE stores (start, length) over the column-major flattened grid; positions are 1-indexed."

---

## 4. Data loading and split
**`load_train_arrays(cfg)`**
- Reads `train.csv`, fills missing RLE with `""`, builds a dictionary `(image_id, defect_type) → rle`.
- Creates `imgs` of shape `(N, 128, 800)` uint8 (~1.2 GB) and `gts` of shape `(N, 4, 128, 800)` uint8 (the four masks).
- Everything stays in RAM so training never waits on disk or JPEG decoding.

**`make_split(gts, cfg)`**
- `present = gts.reshape(N, 4, -1).max(axis=2) > 0` → a boolean `(N, 4)` matrix: does image *i* contain class *k*?
- `combo` = a string per image such as `"defect_a+defect_c"` or `"clean"`.
- Combinations with fewer than 2 images are relabelled `"rare"`, because `train_test_split(stratify=...)` needs at least 2 per group (a lone leftover image is merged into the biggest group).
- `train_test_split(..., test_size=0.15, random_state=42, stratify=combo)` → train/validation index arrays.

**Why stratify?** Rare combinations (e.g. defect_b) should appear in both training and validation in the same proportion.

---

## 5. W&B setup
Try to import `wandb`; try to log in using the Kaggle secret `WANDB_API_KEY`; if that fails, set `WANDB_MODE=offline` (logs saved locally). `USE_WANDB` tells later code whether to log.

---

## 6. Sampling probabilities (v3 = Milestone 4 sampler)
```python
def combo_groups(present_rows):   # "clean", "defect_a+defect_c", ...
def bounded_sqrt_weights(groups, cap=5.0):
    labels, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    n_clean = counts[list(labels).index("clean")]
    return np.minimum(np.sqrt(n_clean / counts), cap)[inv.reshape(-1)]
def make_sampling_probs(present, tr_idx, cfg):
    w = bounded_sqrt_weights(combo_groups(present[tr_idx]), cfg.sampler_cap)
    return w / w.sum()           # probabilities that sum to 1
```
- Group weight = `min(sqrt(n_clean / n_group), 5)`. The clean group has weight 1; a group 25× smaller gets weight 5 (capped).
- `np.unique(..., return_inverse=True)` gives, for each image, the index of its group, so each image receives its group's weight.
- **Square root** softens the correction (full inverse frequency would swamp training with a handful of images); the **cap** stops tiny groups from being memorised.
- The printed summary shows the clean share of draws dropping from about 48% to about 29%.

---

## 7. Model factory
**`build_model(arch, encoder, cfg, pretrained, use_aux)`**
```python
builders = {"unet": smp.Unet, "fpn": smp.FPN, "unetpp": smp.UnetPlusPlus, "deeplabv3plus": smp.DeepLabV3Plus}
aux = dict(classes=4, dropout=0.2) if use_aux else None
return builders[arch](encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                      in_channels=cfg.in_channels, classes=4, aux_params=aux)
```
- `classes=4` → four output channels (one per defect type), with **no softmax** inside: the network returns **logits**; we apply sigmoid ourselves.
- `in_channels=1` with ImageNet weights: smp **sums** the three RGB filters of the first conv layer. A grey image then gives the same result as feeding it 3 times, with 3× less data moved.
- `aux_params` adds a second head after the encoder that outputs 4 numbers per image: "is class k present in this image?" (dropout 0.2).

**`make_pretrained(...)`** wraps the build with a **timeout** (`signal.alarm`, 300 s). If the weight download hangs or fails, it either raises (strict) or falls back to random weights and prints a warning. This was added after a Kaggle download once hung forever.

**`forward_model(model, x, use_aux)`** returns `(mask_logits, class_logits)` when the aux head is on, or `(mask_logits, None)`.

**Say in the viva:** "U-Net, FPN and DeepLabV3+ differ only in the decoder; the encoder is a pretrained ResNet or EfficientNet."

---

## 8. Loss, preprocessing, augmentation, metric
**`prep(batch_uint8, cfg)`**: `uint8 (B,H,W)` → GPU float, divide by 255, subtract mean, divide by std, add a channel dimension → `(B, 1, H, W)`. (v2 repeats it to 3 channels.)

**`augment(x, y)`** (training only): random horizontal flip and random vertical flip, applied identically to image and mask via `torch.where`; random contrast (×0.85–1.15) and brightness (±0.15).

**`CompoundLoss`** = `bce_weight · BCE(logits, target) + dice_weight · SoftDice`
- `p = sigmoid(logits)`.
- Soft Dice per class: `1 − (2·Σ(p·y) + smooth) / (Σp + Σy + smooth)`, summed over the whole batch and all pixels, then averaged over the 4 classes. `smooth = 1` avoids 0/0.
- BCE gives stable gradients; Dice fights the huge background/defect imbalance.

**`evaluate(model, cfg, d, ...)`**: loops over validation in batches (to avoid running out of memory), and accumulates:
- `dice_sum` over image-class pairs that really have a defect → **foreground Dice**.
- `fp_sum` = predicted pixels / 102,400 on empty pairs → **false-positive score** = `1 − mean`.
- `tn / n_empty` → **empty accuracy**.
- `official = 0.7·fg + 0.2·(1 − fp) + 0.1·empty_acc`.
- Also **pair F1/accuracy** (treating each image-class pair as present/absent), **false-alarm rate** `fp / (fp + tn)`, and validation loss.
- Options `min_total` (drop masks with fewer than that many pixels) and `cls_thr` (apply the classification gate).

---

## 9. Training code
**`warmup_cosine(step, warmup, total)`**: `step/warmup` while warming up; afterwards `0.5·(1 + cos(π·(step−warmup)/(total−warmup)))`.

**`EarlyStopping`**: `step(epoch, score)` returns `True` when the score has failed to beat `best + min_delta` for `patience` epochs in a row.

**`EMA`** (exponential moving average of weights)
- Keeps a *shadow copy* of the model; after each optimizer step: `shadow = d·shadow + (1−d)·live`, with `d = min(0.999, (1+steps)/(10+steps))` (a short warm-up of the decay so early steps adapt fast).
- Floating-point tensors (weights and BatchNorm statistics) are blended; integer buffers are copied.
- The shadow model is the one evaluated and saved: it is smoother and usually better than the last raw weights.

**`train_model(arch, encoder, cfg, d)`**: step by step
1. **Resume logic:** if `v3_best_<name>.pt` already exists, skip training.
2. `set_seed`, build the pretrained model, move to GPU, create the EMA.
3. **Two learning rates:** encoder parameters at `lr × 0.3`, the rest at `lr`, in one `AdamW` optimizer with `weight_decay=1e-2`.
4. `LambdaLR(opt, lambda t: warmup_cosine(t, warmup_steps, total_steps))`: warm-up then cosine, stepped **per optimizer step**.
5. `GradScaler` for AMP; W&B run named `v3-<arch>-<encoder>` with its config.
6. **Each epoch:**
   - Draw `len(tr_idx)` images **with replacement**, using the sampler probabilities (`np.random.choice(..., p=probs)`).
   - Loop over micro-batches: `prep` + `augment` → forward under `torch.autocast` (16-bit) → `loss = CompoundLoss + 0.5·BCE(class_logits, any-defect-in-image)` (the target `y.amax((2,3))` is "does this class have any pixel").
   - `scaler.scale(loss / accum_steps).backward()`; when `accum_steps` micro-batches are done: `scaler.step(opt)`, `scaler.update()`, `zero_grad`, `sched.step()`, `ema.update()`.
   - After the epoch, `evaluate(ema.shadow, ..., thr=0.5, min_total=100, cls_thr=0.5)`.
   - If the official score is a new best: save `ema.shadow.state_dict()` as the checkpoint.
   - Log train loss, val loss, `val_official`, `val_false_alarm_pct`, `val_fg_dice`, `val_pair_f1`, `val_pair_acc`, `lr` to W&B.
   - `EarlyStopping.step`: break if no improvement.
7. Clean up (delete model, `gc.collect()`, empty CUDA cache) and return the checkpoint path.

**Say in the viva:** "I select the checkpoint by the official validation score, not by loss, because the loss is dominated by background pixels."

---

## 10. Inference, tuning, submission
**`load_models`**: rebuilds each architecture **without downloading** (`pretrained=False`) and loads its best EMA checkpoint, in `eval()` mode.

**`ensemble_predict(models, batch, cfg)`**
- For each model × each TTA view (none, h-flip, v-flip): flip the input, run the model, apply sigmoid, **flip the prediction back**, add to a running sum.
- Also sum the classification probabilities (`sigmoid(cl)`).
- Divide by `n = 3 models × 3 views = 9` → averaged probabilities `(B, 4, H, W)` and class probabilities `(B, 4)`.

**`postprocess(prob, thr, min_area, min_total)`** for one class mask:
1. `mask = prob > thr`.
2. If `min_area > 0`: `cv2.connectedComponentsWithStats` finds 8-connected blobs; blobs with area `< min_area` are removed.
3. If the total remaining pixels `< min_total`, empty the whole mask.

**`tune_postprocessing(val_probs, val_cls, val_gt)`** (the cleverest part)
- The official score is a sum of per-class terms with fixed denominators (`n_pos`, `n_empty` counted over all pairs), so **each class can be tuned independently**.
- For each class, loop over thresholds. For each threshold, run the connected-component analysis **once per image**, and compute for every `min_area` option the kept pixels `ps` and overlap `inter` (vectorised in tables).
- Then loop over `min_total` and `cls_thr` (classification gate): `keep = (ps >= min_total) & (cls_prob >= cls_thr)`; compute Dice for positives and false-positive/empty terms for empty pairs; keep the combination with the highest score.
- Returns the best `(thr, min_area, min_total, cls_thr)` per class and the total validation score (`0.2` baseline from an all-empty FP score is added up-front).
- Result is saved to `tuned_postprocessing.json`.

**`build_submission(specs, ckpts, cfg, d, sample_df, tag)`**
1. Load models; predict all validation images (probabilities stored as float16 to save memory) with the ensemble.
2. Run `tune_postprocessing`; print the tuned validation score.
3. (v3) `size_bin_report`: recall of presence per defect-size bin.
4. For each test image: read, ensemble-predict, and for each class apply the gate (`cls_prob >= cls_thr`) and `postprocess`; `rle_encode` the mask (or `" "` if empty).
5. Fill the `sample_submission.csv` rows in the right order and save `submission.csv`.

---

## 11. The run cell
```python
ckpts = []
for n, (arch, encoder) in enumerate(CFG.models, start=1):
    ckpts.append(train_model(arch, encoder, CFG, D))
    if n < len(CFG.models):
        build_submission(CFG.models[:n], ckpts, CFG, D, sample, tag=f"interim {n} model(s)")
final_score = build_submission(CFG.models, ckpts, CFG, D, sample, tag="final")
```
Train model 1 → write an interim submission → train model 2 → interim → train model 3 → final. A last W&B run (`v3-ensemble-final`) logs `val_official_tuned`.

---

## 12. What v2 does differently
| Part | v2 | v3 |
|---|---|---|
| Models | U-Net/ResNet-34, FPN/EfficientNet-B3, **U-Net/ResNet-50** | U-Net/ResNet-34, FPN/EfficientNet-B3, **DeepLabV3+/ResNet-34** |
| Epochs | 30 | up to 35 with early stopping |
| Input | `prep` repeats grey into **3 channels** (`in_channels=3`) | **1 channel** (summed filters) |
| Loss | `BCE(out, y) + smp.losses.DiceLoss(mode="multilabel")` + 0.5·BCE on class head | own `CompoundLoss` (BCE + pooled soft Dice) + 0.5·BCE on class head |
| LR schedule | **OneCycleLR** (pct_start 0.15), peak lr 1e-3, encoder ×1/3 | **warmup 2 epochs + cosine**, encoder ×0.3, per optimizer step |
| Weight decay | 1e-4 | 1e-2 |
| EMA | **none** (saves raw best-epoch weights) | EMA 0.999, saves EMA weights |
| Sampling | **repeat images that contain defect_b once more** (`rare_idx` appended to the epoch pool, then shuffled) | **bounded sqrt sampler** over all label combinations, draws with replacement |
| Early stopping | none (always 30 epochs) | patience 8, min_delta 0.002 |
| Gradient accumulation | none | `accum_steps` (default 1) |
| W&B metrics | `val_official`, `val_fg_dice`, `val_cls_f1`, `val_cls_acc` | adds val loss, false-alarm %, pair F1/acc, lr |
| Evaluation | `quick_eval` (global threshold 0.5, min_total 100, gate 0.5) | `evaluate` (same idea, plus batch loss and false-alarm rate) |
| Tuning grids | thresholds 0.3–0.9 | thresholds 0.2–0.9 |
| Extras | none | `size_bin_report`, `tuned_postprocessing.json` |
| W&B names / checkpoints | `submission-v2-<model>`, `best_<model>.pt` | `v3-<model>`, `v3_best_<model>.pt` |

The ensembling, TTA, post-processing tuning, gate logic, resume logic and interim submissions are **the same in both**.

---

## 13. Shape cheat-sheet
| Object | Shape / type |
|---|---|
| `imgs` | `(N, 128, 800)` uint8 |
| `gts` | `(N, 4, 128, 800)` uint8 |
| `present` | `(N, 4)` bool |
| network input `x` | `(B, 1, 128, 800)` float (v2: `(B, 3, ...)`) |
| mask logits | `(B, 4, 128, 800)` |
| class logits | `(B, 4)` |
| ensemble probs | `(B, 4, 128, 800)` averaged over 9 views |
| submission | one row per (image, class), `mask_rle` string |

---

## 14. Code questions you may get
1. **Why `order="F"`?** The RLE counts pixels column by column.
2. **Why are the outputs logits and where does sigmoid happen?** `BCEWithLogits` and our evaluation apply sigmoid; logits are numerically more stable.
3. **Why two learning rates?** The pretrained encoder already has good features; a smaller lr avoids destroying them while the new decoder learns quickly.
4. **Why scale the loss by `1/accum_steps`?** So the summed gradients equal the gradient of the average over the large effective batch.
5. **Why evaluate the EMA model?** It averages weights over many steps → smoother and usually better.
6. **Why tune thresholds per class on validation?** Each defect has a different size and difficulty; the metric penalises false alarms, so a per-class cut-off, area filter and gate give a better trade-off.
7. **Why is the tuned validation score optimistic?** The parameters were chosen to maximise that very score.
8. **What does the aux head's target `y.amax((2,3))` mean?** Maximum over height and width of the mask = 1 if the class has any pixel.
9. **How does resume work?** If the checkpoint file exists, `train_model` skips that model.
10. **Why `float16` for validation probabilities?** To fit 1,700 × 4 × 128 × 800 values in memory.
11. **Why ensemble?** Different architectures make different errors; averaging reduces variance.
12. **What happens if the weight download fails?** `make_pretrained` times out; v2/v3 fall back with a warning (the log must be checked).

---

## 15. Tiny snippets to practise writing
```python
# 1. dice
def dice(p, t):
    s = p.sum() + t.sum()
    return 1.0 if s == 0 else 2 * (p & t).sum() / s

# 2. one TTA prediction (horizontal flip) for one model
with torch.no_grad():
    p = torch.sigmoid(model(x))
    p_flip = torch.sigmoid(model(x.flip(-1))).flip(-1)
    p_avg = (p + p_flip) / 2

# 3. EMA update
for e, m in zip(ema_tensors, live_tensors):
    e.mul_(d).add_(m.detach(), alpha=1 - d)

# 4. remove small blobs
n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
keep = stats[:, cv2.CC_STAT_AREA] >= min_area
keep[0] = False                      # label 0 is background
mask = keep[labels].astype(np.uint8)
```
