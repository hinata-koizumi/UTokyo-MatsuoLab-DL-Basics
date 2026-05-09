# University of Tokyo Deep Learning Basics — Competition Portfolio

English | [日本語版(Japanese)](README.ja.md)

Course competition solutions from the University of Tokyo deep learning basics sequence. Per-task write-ups live next to the code; Japanese text may be in `README.ja.md` or `README_JA.md` where linked from each assignment README.

## Leaderboard reference

Figures below reflect **leaderboards at the time of each competition**. They are not guarantees if you rerun code today.

| Task | Directory | Rank | Metric |
|------|-----------|------|--------|
| CIFAR-10 CNN classification | [cifar10-cnn-classifier](cifar10-cnn-classifier/) | 8th / 1,365 | Accuracy 0.9683 |
| Fashion-MNIST, NumPy MLP | [fashion-mnist-mlp](fashion-mnist-mlp/) | 11th / 1,466 | Accuracy 0.9204 |
| Fashion-MNIST, constrained PyTorch MLP | [fashion-mnist-scratch-mlp](fashion-mnist-scratch-mlp/) | 3rd / 1,439 | Accuracy 0.9485 |
| Fashion-MNIST, softmax regression | [fmnist-logreg-classification](fmnist-logreg-classification/) | 15th / 1,593 | Accuracy 0.905 |
| IMDb sentiment, RNN | [imdb-sentiment-rnn](imdb-sentiment-rnn/) | 2nd / 1,263 | Macro F1 0.93082 |
| Fashion-MNIST VAE, NLL | [vae_fashionmnist_nll](vae_fashionmnist_nll/) | 3rd / 650 | NLL 201.556 |

Treat ranks as historical context; for code review, prioritize readability, soundness, and clear reproduction steps.

## Technical stack overview

| Area | Examples |
|------|----------|
| Deep learning | PyTorch: CNN, RNN, VAE, constrained MLP |
| Classical / NumPy | Hand-written MLP and softmax regression, feature pipelines |
| Preprocessing / metrics | scikit-learn splits and preprocessing, NumPy / Pandas |

Course-provided `*.npy` files must be placed locally according to copyright and distribution rules. This repository does **not** ship training data.

## Environment setup

Python 3.9+ is recommended; for the VAE assignment follow that subdirectory’s README if it asks for 3.12+. Dependencies are isolated per assignment:

```bash
cd <assignment-directory>
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data layout

- **Layout A**: `<assignment-root>/data/input/` holds `x_train.npy`, labels as `t_train.npy` or `y_train.npy`, and `x_test.npy` when required.
- **Layout B**: same filenames directly under `<assignment-root>/data/`.

The CIFAR-10 code accepts layout A or B. Optionally set `CIFAR10_WORK_DIR` to override the project root for that assignment only.

## Reproduction commands

Place unpacked data under `<assignment-root>`, then run from that directory.

### CIFAR-10 CNN — GPU recommended, ~160 epochs

```bash
cd cifar10-cnn-classifier
python src/lecture05_homework.py
```

Output: `data/output/submission.csv`

### Fashion-MNIST NumPy MLP

Multi-seed behavior close to the submission uses `--preset_best`; see that folder’s README for flags.

```bash
cd fashion-mnist-mlp
python script/model.py --preset_best
```

Outputs include `data/output/y_pred.csv`; weights may be saved under `data/train/mlp_model.npz`.

### Fashion-MNIST constrained PyTorch MLP

```bash
cd fashion-mnist-scratch-mlp
python src/lecture04_homework.py
```

Output: `data/output/submission.csv`

### Fashion-MNIST softmax regression

```bash
cd fmnist-logreg-classification
python script/main/main.py --mode train --data-dir data/input --output data/output/predictions.csv
```

Cross-validation: `--mode cv`

### IMDb sentiment — multi-seed ensemble

```bash
cd imdb-sentiment-rnn
python src/lecture07_homework.py
```

Output: `data/output/submission_seed_ensemble.csv`

### Fashion-MNIST VAE — ensemble submission

Full retrains follow README seeds and epochs with `--mode train`. Example once checkpoints exist:

```bash
cd vae_fashionmnist_nll
python submission_code.py --mode ensemble --data_dir data/input --out_csv submission.csv
```

Checkpoint expectations are documented at the top of `submission_code.py`.

## Code map

| Directory | Primary entrypoint |
|-----------|-------------------|
| cifar10-cnn-classifier | `src/lecture05_homework.py` |
| fashion-mnist-mlp | `script/model.py` |
| fashion-mnist-scratch-mlp | `src/lecture04_homework.py` |
| fmnist-logreg-classification | `script/main/main.py`, `script/main/model.py` |
| imdb-sentiment-rnn | `src/lecture07_homework.py` |
| vae_fashionmnist_nll | `submission_code.py` |
