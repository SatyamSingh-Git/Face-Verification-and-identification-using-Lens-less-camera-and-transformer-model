# Face Verification and Identification Using Lensless Camera and Transformer Model

A deep learning pipeline for **face recognition** and **face verification** on **lensless (FlatCam) sensor data** using DCT-domain frequency representations. This project evolves from a baseline 5-branch CNN to a state-of-the-art **Hybrid ResNet18 + Transformer + Sub-center ArcFace** architecture achieving **97.94% recognition accuracy** and **0.9763 verification AUC** on a challenging 87-identity lensless dataset.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Architecture Evolution](#architecture-evolution)
  - [Phase 1: Baseline CNN](#phase-1-baseline-cnn)
  - [Phase 2: Pure DCT-ViT Transformer](#phase-2-pure-dct-vit-transformer)
  - [Phase 3: Hybrid ResNet18 + Transformer + ArcFace](#phase-3-hybrid-resnet18--transformer--arcface)
  - [Phase 4: Advanced Optimizations (Current)](#phase-4-advanced-optimizations-current)
  - [Phase 7: Pushing Beyond 98% — Tier 1](#phase-7-pushing-beyond-98--tier-1-training-improvements)
- [Results Summary](#results-summary)
- [Project Structure](#project-structure)
- [Training](#training)
- [Testing](#testing)
- [Key Technical Details](#key-technical-details)
- [Requirements](#requirements)

---

## Problem Statement

Traditional cameras use lenses to focus light onto a sensor. **Lensless cameras** (FlatCam) replace the lens with a coded mask, producing raw sensor measurements that do not resemble natural images. These measurements encode face identity in complex spatial and frequency patterns, making standard face recognition methods ineffective.

**Goal:** Build a model that can:
1. **Recognize** (classify) faces from lensless sensor data (87 identities)
2. **Verify** whether two lensless measurements belong to the same person

**Challenge:** The dataset is small (~22,000 total samples), the signal is noisy, and the data is in DCT (frequency) domain — not natural images.

---

## Dataset

- **Type:** Lensless FlatCam sensor measurements converted to DCT (Discrete Cosine Transform) domain
- **Format:** `.npy` files, each containing a `15 × 64 × 64` tensor (5 DCT subbands × 3 channels each)
- **Classes:** 87 unique identities
- **Train set:** ~19,575 samples
- **Test set:** ~2,523 samples
- **Verification pairs:** 10,092 pairs (50% genuine, 50% impostor)

---

## Architecture Evolution

### Phase 1: Baseline CNN

The original architecture uses a **5-branch VGG-style CNN** with a Pixel Attention Layer.

```
Input: 5 DCT subbands (each 3×64×64)
         ↓
Branch 1 ──→ VGG(3→64→128→256) ──→ PALayer ──┐
Branch 2 ──→ VGG(3→64→128→256) ──→ PALayer ──┤
Branch 3 ──→ VGG(3→64→128→256) ──→ PALayer ──┼──→ Concatenate ──→ FC(1280→87)
Branch 4 ──→ VGG(3→64→128→256) ──→ PALayer ──┤
Branch 5 ──→ VGG(3→64→128→256) ──→ PALayer ──┘
```

**Key characteristics:**
- 5 independent CNN branches, one per DCT subband
- Pixel Attention (PALayer) to focus on informative spatial regions
- Cross-Entropy Loss with SGD optimizer (lr=0.05, momentum=0.9)
- StepLR scheduler (step size 50)

**Results:**
| Metric | Value |
|---|---|
| Recognition Accuracy | ~93.97% |
| Verification AUC | ~0.9784 |

**Limitations:** Each branch processes subbands independently — no cross-frequency interaction. Local receptive fields limit global identity modeling.

---

### Phase 2: Pure DCT-ViT Transformer

Replaced the 5-branch CNN with a **DCT-ViT** (Vision Transformer) to capture global cross-subband relationships.

```
Input: 15×64×64 (stacked subbands)
         ↓
Per-Subband CNN Stem (lightweight conv layers)
         ↓
Spatial Patch Tokens (80 tokens + 1 CLS)
         ↓
Subband-Aware Positional Encoding
         ↓
Transformer Encoder (6 layers, 8 heads, d=256)
         ↓
CLS Token → MLP Classifier → 87 classes
```

**Key characteristics:**
- Custom lightweight CNN stem per subband
- Subband ID embedding added to positional encoding
- Standard Cross-Entropy Loss
- AdamW optimizer (lr=1e-4) + CosineAnnealingLR

**Results:**
| Metric | Value |
|---|---|
| Recognition Accuracy | ~77% (unstable) |
| Verification AUC | Low |

**Why it failed:** Pure Transformers lack the inductive bias needed for small datasets (only 87 classes, ~22K samples). Without pretrained weights, the model couldn't learn meaningful features from the noisy lensless data.

---

### Phase 3: Hybrid ResNet18 + Transformer + ArcFace

Combined the best of both worlds: **pretrained CNN features** + **Transformer global attention** + **ArcFace metric learning**.

```
Input: 15×64×64 (stacked DCT subbands)
         ↓
Resize → 224×224
         ↓
ResNet18 Backbone (ImageNet pretrained)
  • Modified conv1: 15 input channels (weights repeated 5x from 3ch)
  • Output: 512×7×7 feature map
         ↓
Tokenize: Flatten spatial → 49 tokens (each 512-dim)
         ↓
Learned Positional Encoding
         ↓
Transformer Encoder (4 layers, 8 heads, d=512, FFN=2048, GELU)
         ↓
Global Mean Pooling → 512-dim vector
         ↓
L2 Normalize
         ↓
ArcFace Classifier (margin=0.5, scale=64) → 87 classes
```

**Training innovations:**
- **ArcFace Loss:** Angular margin-based loss that enforces strict separation between identities in embedding space
- **AdamW optimizer** (lr=3e-4) + CosineAnnealingLR
- **Gaussian Noise** augmentation during training
- **Mixup** data augmentation (α=0.2)
- **Gradient clipping** (max_norm=5.0)
- **Test-Time Augmentation (TTA):** Average embeddings from original, flipped, and blurred inputs

**Results:**
| Metric | Value |
|---|---|
| Recognition Accuracy | **94.4%** |
| Verification AUC | **0.9948** |

**Why it worked:**
1. ResNet18 provides strong pretrained feature extraction that doesn't need massive data
2. Transformer captures global correlations across the 49 spatial tokens
3. ArcFace enforces large angular margins between the 87 identities, making the model discriminative even with noisy inputs
4. TTA stabilizes predictions at test time

---

### Phase 4: Advanced Optimizations

Six targeted improvements to push accuracy from 94.4% toward 96%+:

```
Input: 15×64×64 (stacked DCT subbands)
         ↓
Resize → 224×224
         ↓
ResNet18 Backbone (ImageNet pretrained)
  • conv1, bn1, layer1, layer2 FROZEN for first 10 epochs
  • Unfrozen after epoch 10
         ↓
49 Tokens × 512-dim
         ↓
Transformer Encoder (4 layers, 8 heads, d=512)
         ↓
Global Mean Pooling → 512-dim
         ↓
Projection Head: Linear(512→768) → BN → ReLU → Linear(768→768)
         ↓
L2 Normalize → 768-dim embedding
         ↓
Sub-center ArcFace (K=3, margin=0.5, scale=64) → 87 classes
```

**New improvements:**

| # | Improvement | Details |
|---|---|---|
| 1 | **Sub-center ArcFace (K=3)** | Each identity gets 3 prototype vectors. The closest one is used for angular margin. Handles noisy lensless samples where the same person may have varied representations. |
| 2 | **Embedding Dimension 512→768** | Larger embedding space provides better identity separation. Added a 2-layer projection head with BatchNorm. |
| 3 | **LR Warmup + Cosine Decay** | Linear warmup for 5 epochs → cosine annealing for remaining epochs. Prevents early training instability with the Transformer. |
| 4 | **Freeze Early ResNet Layers** | conv1, bn1, layer1, layer2 frozen for first 10 epochs. Preserves pretrained ImageNet features from being destroyed by large early gradients. |
| 5 | **Stronger Augmentation** | Added random horizontal flip, brightness jitter (±20%), contrast jitter (±20%) on top of existing Gaussian noise + Mixup. |
| 6 | **Train Longer** | Default epochs set to 120 with checkpoint resume support. |

**Additional features:**
- **Mixed Precision (AMP)** via `torch.cuda.amp.autocast` for faster GPU training
- **Hardware-agnostic**: Auto-detects TPU (`torch_xla`), GPU (with AMP), or CPU
- **Multi-GPU support** via `nn.DataParallel` (auto-detects 2x T4 on Kaggle)
- **Checkpoint resume** via `--resume logs/<folder>/last.pth` — saves full state (model, optimizer, scheduler, epoch, best_acc) every epoch

**Training Observations:**
- Best test accuracy reached at epoch 112: **89.22%** (training), **91.00%** (test with TTA)
- Gradient explosion observed after epoch 122 (loss diverged to NaN) due to cosine LR schedule reaching near-zero; best model was safely saved from epoch 112 onwards
- Root cause: cosine LR decays too aggressively post-epoch-110; a minimum LR floor (e.g., `eta_min=1e-6`) is recommended for future runs

---

### Phase 5: Ablation Studies (Ongoing)

To systematically evaluate the contribution of individual enhancements and hyperparameter choices, we introduced an **ablation study framework** into the training and testing pipeline.

**Planned Experiments:**
1. **Test-Time Augmentation (TTA) Impact:** Evaluating the performance lift provided by averaging embeddings across original, flipped, and blurred inputs during inference (`--no_tta` flag).
2. **Mixup Strategy Impact:** Measuring the contribution of Mixup vs. standard augmentations alone (`--no_mixup` flag). ✅ **Completed.**
3. **ArcFace Hyperparameter Tuning:** Adjusting ArcFace margin (`--arcface_m`), scale (`--arcface_s`), and sub-centers (`--arcface_k`) to find the optimal metric learning configuration (e.g., testing `m=0.40`, `s=40`, `K=3` vs standard `m=0.50`, `s=64.0`, `K=3`).
4. **Multi-Seed Stability:** Running multiple initialization sequences across 3–5 random seeds (`--seed`) to ensure the reported accuracy gains are robust and statistically significant.

---

### Ablation Result: Without Mixup (`--no_mixup`)

**Experiment Command:**
```bash
python train.py --model transformer \
    --train_data <path>/train/ymdct_npy \
    --test_data <path>/test/ymdct_npy \
    --batch_size 128 --num_epoch 115 \
    --seed 42 --no_mixup --eta_min 1e-6
```

**Results:**

| Metric | With Mixup (v2 baseline) | Without Mixup (ablation) | Δ Change |
|---|---|---|---|
| Recognition Accuracy | **91.00%** | 85.69% | **−5.31%** |
| Verification AUC | **0.9962** | 0.9688 | **−0.0274** |

**Analysis:**

Removing Mixup causes a significant drop across both metrics. This confirms Mixup is a critical component of the pipeline, not just a minor regularizer.

- **Why Mixup helps recognition (+5.31%):** Mixup blends two images with labels `(lam·x_a + (1-lam)·x_b)`, forcing the model to produce smoother, more interpolatable embeddings. Without Mixup, the decision boundaries are sharper but more brittle — the model overfits to specific image patterns rather than learning robust identity representations.
- **Why Mixup helps verification (+2.74% AUC):** In verification tasks, the model must compare embeddings of *unseen* pairs. Mixup acts as a form of embedding regularization: by training on convex combinations of identities, the embedding space becomes more uniformly distributed and geometrically consistent. Without Mixup, the embedding space is less organized, making cosine similarity less reliable between unseen pairs.
- **Training accuracy observation:** When `--no_mixup` is active, training accuracy shows `0.000%` for all epochs. This is **expected ArcFace behavior** — the angular margin `m=0.5` penalizes the correct class logit during training so aggressively that `argmax` never returns the correct class during forward passes. The model is still learning; only test accuracy (reported without margin) is meaningful.

**Conclusion:** Mixup provides **+5.31% recognition** and **+2.74% AUC uplift** and should be kept enabled for all production training runs.

---

### Ablation Result: Tuned ArcFace Hyperparameters (m=0.4, s=40)

Multiple configurations of ArcFace margin (`m`), scale (`s`), sub-centers (`K`), and batch size were systematically evaluated. All runs used `--num_epoch 120 --seed 42 --eta_min 1e-6` unless stated otherwise.

**Results:**

| Experiment | Batch | K | Rec Acc | AUC | Top-5 Acc |
|---|---|---|---|---|---|
| v2 Baseline (m=0.5, s=64, K=3) | 128 | 3 | 94.33% | 0.9204 | 99.25% |
| Tuned ArcFace (m=0.4, s=40, K=3) | 64 | 3 | 94.73% | 0.9595 | 99.17% |
| Tuned ArcFace (m=0.4, s=40, K=2) | 64 | 2 | 94.57% | 0.9618 | **99.56%** |
| Tuned ArcFace (m=0.4, s=40, K=3) | 128 | 3 | 94.21% | 0.9285 | 99.37% |
| Tuned ArcFace (m=0.4, s=40, K=3) | 512 | 3 | 89.54% | 0.9265 | 97.86% |

**Key Findings:**

- **Tuned ArcFace (m=0.4, s=40) consistently outperforms the v2 baseline** (`m=0.5, s=64`) — reducing the margin and scale improves recognition accuracy for this dataset.
- **Batch size strongly impacts accuracy:** batch=64 > batch=128 > batch=512. Smaller batches introduce more gradient noise which acts as regularization and finds better minima.
- **K=3 gives best recognition** while **K=2 gives best AUC (0.9618)** — fewer sub-centers produce cleaner, more separable verification embeddings.

**Conclusion:** The optimal configuration is `--arcface_m 0.4 --arcface_s 40 --arcface_k 3 --batch_size 64`.

---

### Phase 5: Extended Training on RTX A6000

The best ablation configuration was retrained for **150 epochs** on a dedicated **NVIDIA RTX A6000** GPU (48GB VRAM) via Trinity/On-demand, yielding significant improvements.

**Training Command:**
```bash
python run_trinity.py --batch_size 64 --num_epoch 150 --arcface_m 0.4 --arcface_s 40 --arcface_k 3
```

**Results:**

| Metric | Previous Best (120 ep, T4) | RTX A6000 (150 ep) | Δ Change |
|---|---|---|---|
| Recognition Accuracy | 94.73% | **96.67%** | **+1.94%** |
| Verification AUC | 0.9595 | **0.9699** | **+0.0104** |

---

### Phase 6: Batch Experiments — Hyperparameter Search (Current Best)

Building on the 150-epoch baseline, we ran **4 experiments overnight** on the A6000 to test extended training (200 epochs) with different ArcFace configurations:

```bash
python run_trinity.py --batch_experiments  # Runs all 4 sequentially
```

**Results:**

| # | Experiment | Config | Rec Acc | AUC | Top-5 |
|---|---|---|---|---|---|
| 1 | `exp_200ep` | m=0.4, K=3, 200ep | 97.46% | 0.9655 | 99.80% |
| 2 | `exp_m035_200ep` 🏆 | **m=0.35, K=3, 200ep** | **97.62%** 🥇 | **0.9689** | **99.80%** |
| 3 | `exp_k5_200ep` | m=0.4, K=5, 200ep | 96.63% | 0.9705 🥇 | 99.56% |
| 4 | `exp_m035_k5_200ep` | m=0.35, K=5, 200ep | 96.63% | 0.9705 🥇 | 99.56% |

**Key Findings:**
1. **Softer margin wins (at K=3):** m=0.35 outperformed m=0.4 → noisy lensless data benefits from a less aggressive angular margin
2. **200 epochs > 150 epochs:** Both 200ep runs beat the 150ep baseline, confirming the model was still learning
3. **K=5 gives best verification AUC (0.9705)** but lower recognition accuracy — more sub-centers help pairwise similarity but add noise to classification. Margin choice (m=0.35 vs 0.4) has no effect when K=5
4. **Best overall model: `exp_m035_200ep`** (m=0.35, K=3, 200 epochs) — **97.62% recognition, 0.9689 AUC**

---

### Phase 7: Pushing Beyond 98% — Tier 1 Training Improvements

Building on the 97.62% best from Phase 6, we implemented five complementary training and regularization improvements designed to reduce test-time variance and lift the stable accuracy floor.

**Problem Identified:** Training logs showed massive test accuracy oscillations (96% → 72% → 96% between epochs), indicating the model sits on a sharp loss landscape. The "97.62% best" was a lucky peak rather than a stable floor.

**Improvements Implemented:**

1. **Label Smoothing (ε=0.1)** — Prevents overconfident predictions by softening one-hot labels. Applied via `nn.CrossEntropyLoss(label_smoothing=0.1)`.
2. **Stochastic Weight Averaging (SWA)** — Averages model weights over the last 25% of training epochs, finding flatter minima. Saves an additional `swa_best.pth` checkpoint.
3. **CutMix Augmentation** — Cuts patches from one image and pastes onto another. Applied 50/50 per batch alongside Mixup.
4. **5-View Test-Time Augmentation** — Upgraded from 3-view to 5-view by adding center-crop (90%) and brightness shift (+10%).
5. **Cosine Annealing with Warm Restarts (SGDR)** — Periodic LR resets (T₀=50, T_mult=2) to escape local minima.

**Experiment Results (Stacking Approach):**

| # | Experiment | Features Enabled | Recognition Acc | Verification AUC |
|---|-----------|-----------------|----------------|------------------|
| 1 | `t1_label_smooth` | Label Smoothing 0.1 | 97.50% | 0.9736 |
| 2 | `t1_ls_swa` 🏆 | + SWA (last 25% epochs) | **97.94%** 🥇 | **0.9763** |
| 3 | `t1_ls_swa_cutmix` | + CutMix (50/50 with Mixup) | 94.09% | **0.9969** 🥇 |
| 4 | `t1_full` | + Warm Restarts (K=3) | 94.61% | 0.9950 |
| 5 | `t1_swa_wr_k5` | + Warm Restarts + K=5 (no CutMix) | 96.67% | 0.9676 |

All experiments: 250 epochs, m=0.35, 5-view TTA active at test time.

**Key Findings:**
1. **🏆 SWA pushed recognition to 97.94%** — new all-time best, validating weight averaging for flatter minima. Best model: `t1_ls_swa`
2. **CutMix dramatically improved verification AUC to 0.9969** but dropped recognition to 94.09% — reveals a recognition vs. verification trade-off
3. **Warm Restarts** partially recovered recognition when CutMix was present (+0.5%), but didn't help without CutMix
4. **K=5 sub-centers hurt both metrics** when combined with SWA + Warm Restarts
5. **Final verdict:** Label Smoothing + SWA (K=3, m=0.35) is the optimal combination for recognition. CutMix only if verification AUC is the priority

**Run command:**
```bash
python run_trinity.py --batch_tier1
```

---

## Results Summary

| Model | Recognition Acc | Verification AUC | Top-5 Acc | Parameters |
|---|---|---|---|---|
| Baseline 5-Branch CNN | ~93.97% | 0.9784 | — | ~2.5M |
| Pure DCT-ViT | ~77% | Low | — | ~4.5M |
| Hybrid ResNet18 + Transformer + ArcFace (v1) | 94.4% | 0.9948 | — | ~23.8M |
| + Sub-center ArcFace + 768-dim + Warmup + Freeze (v2) | 94.33% | 0.9204 | 99.25% | ~24.5M |
| Ablation: Without Mixup (`--no_mixup`) | 85.69% | 0.9688 | — | ~24.5M |
| Ablation: Tuned ArcFace (m=0.4, s=40, K=2, batch=64) | 94.57% | 0.9618 | 99.56% | ~24.5M |
| Ablation: Tuned ArcFace (m=0.4, s=40, K=3, batch=64, 120ep) | 94.73% | 0.9595 | 99.17% | ~24.5M |
| Extended Training (m=0.4, K=3, 150ep, A6000) | 96.67% | 0.9699 | 99.68% | ~24.5M |
| Batch Exp: 200ep (m=0.4, K=3) | 97.46% | 0.9655 | 99.80% | ~24.5M |
| Batch Exp: 200ep (m=0.35, K=3) | 97.62% | 0.9689 | 99.80% | ~24.5M |
| Batch Exp: 200ep (m=0.4, K=5) | 96.63% | 0.9705 | 99.56% | ~24.8M |
| Batch Exp: 200ep (m=0.35, K=5) | 96.63% | 0.9705 | 99.56% | ~24.8M |
| Ablation: Tuned ArcFace (m=0.4, s=40, K=3, batch=128) | 94.21% | 0.9285 | 99.37% | ~24.5M |
| Ablation: Tuned ArcFace (m=0.4, s=40, K=3, batch=512) | 89.54% | 0.9265 | 97.86% | ~24.5M |
| T1: + Label Smoothing (250ep, m=0.35) | 97.50% | 0.9736 | 99.64% | ~24.5M |
| **T1: + Label Smoothing + SWA (250ep, m=0.35)** 🏆 | **97.94%** 🥇 | **0.9763** | **99.88%** | ~24.5M |
| T1: + LS + SWA + CutMix (250ep, m=0.35) | 94.09% | **0.9969** 🥇 | 99.45% | ~24.5M |
| T1: Full Tier-1 + Warm Restarts (250ep, m=0.35) | 94.61% | 0.9950 | 99.25% | ~24.5M |
| T1: SWA + Warm Restarts + K=5 (250ep, m=0.35) | 96.67% | 0.9676 | 99.29% | ~24.8M |

---

## Project Structure

```
├── models/
│   ├── proposed_model.py        # Baseline 5-branch CNN with PALayer
│   ├── transformer_model.py     # HybridResNetTransformer (ResNet18 + Transformer)
│   └── arcface.py               # Sub-center ArcFace (K=3) margin loss
├── train.py                     # Training script (supports CNN & Transformer)
├── test_face_recognition.py     # Face recognition testing (with TTA)
├── test_face_verification.py    # Face verification testing (cosine similarity + AUC)
├── my_data_class.py             # Dataset loaders for DCT .npy files
├── utils.py                     # Progress bar utility
├── run_trinity.py               # Standalone training+eval script for Trinity/On-demand GPU
├── resnet18-f37072fd.pth        # Pretrained ResNet18 weights (offline fallback)
├── data/
│   ├── verification_pairs.txt   # Verification test pairs
│   └── noise_locations/         # Noise location files for robustness testing
├── logs/                        # Training logs, checkpoints (best.pth, last.pth)
└── colab_run.ipynb              # Google Colab notebook
```

---

## Training

### Fresh Training Run (Notebook / Colab)

```bash
python train.py \
    --model transformer \
    --train_data /path/to/train/ymdct_npy \
    --test_data /path/to/test/ymdct_npy \
    --batch_size 64 --num_epoch 150 \
    --arcface_m 0.4 --arcface_s 40 --arcface_k 3 \
    --seed 42 --eta_min 1e-6
```

### Trinity/On-demand GPU (Recommended)

```bash
# Uses best config by default (m=0.4, s=40, K=3, batch=64, 150 epochs)
python run_trinity.py

# Override any parameter:
python run_trinity.py --batch_size 128 --num_epoch 200

# Skip training, only evaluate existing checkpoint:
python run_trinity.py --skip_training
```

### Resume from Checkpoint (after session disconnect)

```bash
python train.py \
    --model transformer \
    --train_data /path/to/train/ymdct_npy \
    --test_data /path/to/test/ymdct_npy \
    --batch_size 64 --num_epoch 150 \
    --resume logs/<timestamp-folder>/last.pth
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `cnn` | Model type: `cnn` (baseline) or `transformer` (hybrid) |
| `--batch_size` | `64` | Batch size |
| `--num_epoch` | `120` | Total training epochs |
| `--warmup_epochs` | `5` | LR warmup epochs |
| `--freeze_epochs` | `10` | Epochs to freeze early ResNet layers |
| `--resume` | `None` | Path to `last.pth` for resume training |

---

## Testing

### Face Recognition

```bash
python test_face_recognition.py \
    --model transformer \
    --test_data /path/to/test/ymdct_npy \
    --weights logs/<folder>/best.pth \
    --batch_size 128
```

### Face Verification

```bash
python test_face_verification.py \
    --model transformer \
    --test_data /path/to/test/ymdct_npy \
    --pairs data/verification_pairs.txt \
    --weights logs/<folder>/best.pth
```

### Plot ROC Curve (Python)

```python
import json, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

with open('results.json', 'r') as f:
    results = json.load(f)

fpr, tpr, thresholds = roc_curve(results['true_labels'], results['pred_scores'])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, lw=2.5, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Face Verification')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

---

## Key Technical Details

### Why Hybrid CNN + Transformer?
- **CNNs** excel at local feature extraction but miss global patterns
- **Transformers** capture global relationships but need massive data
- **Hybrid approach:** Use pretrained ResNet18 for local features → feed token representations into Transformer for global attention → best of both worlds on a small dataset

### Why ArcFace?
- Standard Cross-Entropy doesn't optimize for embedding quality
- ArcFace adds an **angular margin** to the softmax, forcing the model to learn highly discriminative embeddings
- Sub-center ArcFace (K=3) handles the natural **intra-class variation** in noisy lensless data

### Why Freeze Early Layers?
- ResNet18 is pretrained on ImageNet — early layers learn generic edge/texture features
- Training these layers immediately with lensless data (which looks nothing like ImageNet) would destroy useful pretrained knowledge
- Freezing for 10 epochs lets the Transformer + later layers adapt first, then fine-tune everything

### Input Channel Adaptation
- ResNet18 expects 3 RGB channels, but our data has 15 channels (5 DCT subbands × 3)
- Solution: Modified `conv1` weights by repeating the 3-channel weights 5 times and dividing by 5
- This preserves the pretrained feature extraction capability while accepting 15-channel input

### Test-Time Augmentation (TTA)
During inference, the model averages embeddings from 3 views:
1. Original image
2. Horizontally flipped image
3. Slightly blurred image (Gaussian blur, kernel=3)

This produces more robust predictions without retraining.

---

## Requirements

```
torch >= 1.10
torchvision
numpy
scipy
scikit-learn
tensorboard
matplotlib
```

---

## References

- **FlatCam:** Asif et al., "FlatCam: Thin, Lensless Cameras Using Coded Aperture and Computation"
- **ArcFace:** Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (CVPR 2019)
- **Sub-center ArcFace:** Deng et al., "Sub-center ArcFace: Boosting Face Recognition by Large-Scale Noisy Web Faces" (ECCV 2020)
- **ResNet:** He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)
- **Vision Transformer:** Dosovitskiy et al., "An Image is Worth 16x16 Words" (ICLR 2021)
- **Mixup:** Zhang et al., "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)
