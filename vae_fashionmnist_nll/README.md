# University of Tokyo Deep Learning Course Competition

English | [日本語版(Japanese)](README_JA.md)

## Competition Results

- **Final Rank**: **3rd** / 650 participants
- **LB Score**: **201.556**

## Overview

Generative modeling on FashionMNIST using VAE to minimize Negative Log-Likelihood (NLL)

## Rules

- Keep every script consolidated into a single file.
- Do not use any training data other than `x_train`.
- `x_test` is strictly for inference only.

## Approach

- Data preparation
  - Load `x_train.npy` and `x_test.npy` from `data/input/`.
  - Normalize pixel values to [0, 1].
  - Split 10% of the training set into validation data with `random_split` (fixed seed 42).

- Model (`BaseVAE` in `submission_code.py`)
  - **Type**: Custom VAE with U-Net-style Skip Connections.
  - **Encoder Structure**:
    1. `Conv2d(1, 128, kernel=3, stride=2, padding=1)` → ReLU (28x28 → 14x14)
    2. `Conv2d(128, 256, kernel=3, stride=2, padding=1)` → ReLU (14x14 → 7x7)
    3. Flatten → `Linear(256*7*7, z_dim)` for $\mu$ and $\log\sigma^2$
  - **Latent Space**:
    - `z_dim`: 64
    - Prior: Standard Normal $p(z) = \mathcal{N}(0, I)$
  - **Decoder Structure**:
    1. `Linear(z_dim, 256*7*7)` → Unflatten to (256, 7, 7)
    2. **Skip Connection 1**: Add projected Encoder features (7x7)
    3. `ConvTranspose2d(256, 128, kernel=3, stride=2, padding=1)` → ReLU (7x7 → 14x14)
    4. **Skip Connection 2**: Add projected Encoder features (14x14)
    5. `ConvTranspose2d(128, 1, kernel=3, stride=2, padding=1)` → Sigmoid (14x14 → 28x28)
  - **Skip Connections**: 
    - 1x1 Convolutions project encoder features to match decoder dimensions before addition.
    - Preserves spatial details crucial for low NLL.

- Training regimen
  - Train for 22 epochs per seed using Adam (`lr=1e-3`).
  - Loss function: Custom ELBO with weighted reconstruction loss for background pixels (`w_bg=3.0` for `x < 0.5`).
  - KL weight (`beta`) set to 1.0.
  - Save checkpoints at epochs 12, 14, 16, 18, 20, 22.

- Ensembling and inference
  - Repeat the full training loop for five seeds `[0, 1, 2, 3, 4]`.
  - Checkpoint Ensemble: Select epochs 16, 18, 20, 22 from all 5 seeds (Total 20 models).
  - Averaging: Compute mean of **logits**, apply sigmoid, and clip probabilities to `[1e-7, 1-1e-7]`.
  - Save predictions to `submission.csv` with (N, 784) format.

## Tech Stack

- Python 3.12+
- PyTorch (VAE implementation, optimization)
- NumPy (array manipulation)
- pandas (submission export)
- SciPy (special functions for ensemble)
