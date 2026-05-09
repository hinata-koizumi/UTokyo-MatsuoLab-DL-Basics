# University of Tokyo Deep Learning Course Competition — CIFAR-10 CNN Classifier

English | [日本語版(Japanese)](README.ja.md)

## Competition Results

- **Final Rank**: **8th / 1,365participants**
- **LB Score**: **0.9683**

## Overview

Image classification of CIFAR-10 (10 classes) using a CNN-based approach centered on WideResNet-28-10. The training recipe combines strong data augmentation (RandAugment, Random Erasing, Mixup/CutMix), EMA, and a late-stage clean fine-tuning phase. Test-time augmentation averages logits over flips, slight scale changes, and ±1px shifts for robust predictions.

Target and rules follow the assignment’s constraints; please see the script header for details.

## Rules

- Training data is provided as `x_train`, `t_train`, and test data as `x_test` (NumPy arrays).
- Prediction labels must be class indices 0–9 (not one-hot).
- Do not use any training data other than the specified `x_train` and `t_train`.
- PyTorch is allowed.
- Use a CNN-based model; do not use non-CNN architectures (e.g., Vision Transformer) or pretrained/torchvision canned models.

## Approach

- Data preprocessing and split
  - Compute per-channel mean/std from `x_train` and normalize images.
  - Validation split from training data with fixed seed (`seed=42`): 5,000 samples if `N>=50,000`, otherwise 3,000.
  - Transforms
    - Train (augment phase): RandomCrop(32, padding=4), RandomHorizontalFlip, RandAugment, ToTensor, Normalize, RandomErasing.
    - Eval/Clean-FT/Test: ToTensor, Normalize only.

- Augmentation schedule (augment phase only)
  - RandAugment magnitude decays from 10 → 6 between epochs 60–100.
  - Random Erasing probability decays from 0.25 → ~0 between epochs 80–160.
  - Mixup or CutMix per batch with Beta(α, α): α=0.4 before epoch 80, α=0.2 after epoch 80. Disabled during clean fine-tuning.

- Model
  - WideResNet-28-10 (dropout=0.3), CIFAR-style stem and stages.
  - Kaiming initialization for convolution/linear, BatchNorm with affine params.
  - Global pooling → fully-connected to logits (10 classes).
  - Channels-last memory format and AMP enabled on CUDA for speed.

- Optimization and schedules
  - Optimizer: SGD, momentum=0.9, Nesterov, base LR=0.1.
  - Weight decay=5e-4 applied to conv/linear weights only (BN/bias excluded).
  - LR schedule: 5-epoch warmup → cosine annealing (min LR = base LR × 1e-4) until epoch 144.
  - EMA: Exponential Moving Average of weights with epoch-dependent decay
    - <60: 0.999, 60–99: 0.9995, ≥100: 0.9998.
  - Optional SAM (Sharpness-Aware Minimization, ρ=0.05) is implemented and can be enabled; used only in the augment phase when enabled.

- Clean fine-tuning (late stage)
  - From epoch 145:
    - Switch to clean loader (no augmentation, no Mixup/CutMix).
    - Weight decay set to 0.
    - LR manually fixed to 1e-3 for epochs 145–154, then 1e-4 for epochs 155–159.
  - Total epochs: 160.

- Validation
  - Evaluate with EMA weights.
  - Loss uses cross-entropy with label smoothing 0.1 during evaluation.
  - Track and report best validation accuracy.

- Test-time augmentation (TTA)
  - Average logits over:
    - Horizontal flip (on/off),
    - Scales {0.97, 1.00, 1.03},
    - Shifts of ±1px including diagonals (9 offsets total).
  - Final prediction is argmax of averaged logits.

- Inference and saving
  - Save predictions for `x_test` to `data/output/submission.csv` under the resolved project/work directory (header `label`, index `id`). Override the base directory with env var `CIFAR10_WORK_DIR` if needed.
 
  
## Technologies Used

- Python 3
- PyTorch (`torch`, `torchvision`)
- NumPy (`numpy`), Pandas (`pandas`)
- Pillow (`PIL`)


