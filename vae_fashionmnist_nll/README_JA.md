[English](README.md) | **日本語版 (Japanese)**

# 東京大学 Deep Learning 基礎講座 コンペティション

## コンペティション結果

- **最終順位**: **3位** / 650人
- **LB スコア**: **201.556**

## 概要

FashionMNISTを用いた生成モデル（VAE）の構築を行い、Negative Log-Likelihood (NLL) の最小化を目指す。

## ルール

- すべてのスクリプトを単一のファイルにまとめること。
- 学習データとして `x_train` 以外を使用しないこと。
- `x_test` は推論のみに使用すること。

## アプローチ

- データ準備
  - `data/input/` から `x_train.npy` と `x_test.npy` を読み込む。
  - ピクセル値を [0, 1] に正規化。
  - `random_split` (固定シード 42) を用いて、学習データの10%を検証用データとして分割。

- モデル (`submission_code.py` 内の `BaseVAE`)
  - **タイプ**: U-Netスタイル（Skip Connection付き）のカスタムVAE。
  - **エンコーダ構造**:
    1. `Conv2d(1, 128, kernel=3, stride=2, padding=1)` → ReLU (28x28 → 14x14)
    2. `Conv2d(128, 256, kernel=3, stride=2, padding=1)` → ReLU (14x14 → 7x7)
    3. 平坦化 (Flatten) → `Linear(256*7*7, z_dim)` で $\mu$ と $\log\sigma^2$ を出力
  - **潜在空間**:
    - `z_dim`: 64
    - 事前分布: 標準正規分布 $p(z) = \mathcal{N}(0, I)$
  - **デコーダ構造**:
    1. `Linear(z_dim, 256*7*7)` → (256, 7, 7) に変形 (Unflatten)
    2. **Skip Connection 1**: 射影したエンコーダ特徴量 (7x7) を加算
    3. `ConvTranspose2d(256, 128, kernel=3, stride=2, padding=1)` → ReLU (7x7 → 14x14)
    4. **Skip Connection 2**: 射影したエンコーダ特徴量 (14x14) を加算
    5. `ConvTranspose2d(128, 1, kernel=3, stride=2, padding=1)` → Sigmoid (14x14 → 28x28)
  - **Skip Connection**:
    - 1x1 畳み込みを使用してエンコーダ特徴量をデコーダの次元に合わせて射影し、加算。
    - 低NLL実現のために重要な空間情報を保持。

- 学習設定
  - Adam (`lr=1e-3`) を使用し、各シードで22エポック学習。
  - 損失関数: 背景ピクセル (`x < 0.5`) に対して再構成誤差の重み付け (`w_bg=3.0`) を行ったカスタムELBO。
  - KL項の重み (`beta`) は 1.0 に設定。
  - エポック 12, 14, 16, 18, 20, 22 でチェックポイントを保存。

- アンサンブルと推論
  - 5つのシード `[0, 1, 2, 3, 4]` に対して完全な学習ループを実行。
  - チェックポイント・アンサンブル: 全5シードの エポック 16, 18, 20, 22 を選択（計20モデル）。
  - 平均化: **ロジット**の平均を計算し、シグモイド関数を適用した後、確率を `[1e-7, 1-1e-7]` にクリッピング。
  - `submission.csv` に (N, 784) 形式で保存。

## 技術スタック

- Python 3.12+
- PyTorch (VAE実装、最適化)
- NumPy (配列操作)
- pandas (提出ファイルの出力)
- SciPy (アンサンブル用の特殊関数)
