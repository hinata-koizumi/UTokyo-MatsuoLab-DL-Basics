# 東京大学 Deep Learning Course Competition — CIFAR-10 CNN Classifier

[English](README.md) | 日本語版(Japanese)

## コンペティション結果

- **最終順位**: **8位 / 1365人**
- **LBスコア**: **0.9683**

## 概要

WideResNet-28-10 を中核とした CNN によって CIFAR-10（10クラス）の画像分類を行いました。大規模なデータ拡張（RandAugment、Random Erasing、Mixup/CutMix）、EMA、終盤のクリーンファインチューニングを組み合わせたレシピです。推論時は左右反転、微小スケール変更、±1px シフトを組み合わせたテスト時拡張（TTA）でロジット平均を取り、堅牢性を高めています。

詳細な制約条件はスクリプト冒頭のコメントに記載されています。

## ルール

- 訓練データ `x_train`、ラベル `t_train`、テストデータ `x_test` の NumPy 配列のみを使用する。
- 予測ラベルは 0〜9 のクラス番号で出力し、one-hot 表現は用いない。
- 指定された `x_train` と `t_train` 以外の訓練データを使用しない。
- PyTorch の利用は許可されている。
- CNN ベースのモデルを使用し、非 CNN（Vision Transformer 等）や学習済み/torchvision 既製モデルは使用しない。

## アプローチ

- データ前処理と分割
  - `x_train` からチャンネルごとの平均・標準偏差を算出し、全ての画像を正規化。
  - 訓練データからバリデーションを固定シード（`seed=42`）で分割。`N>=50,000` の場合 5,000 サンプル、そうでない場合は 3,000 サンプルを検証用に確保。
  - 変換
    - 訓練（拡張フェーズ）: RandomCrop(32, padding=4)、RandomHorizontalFlip、RandAugment、ToTensor、Normalize、RandomErasing。
    - 評価/クリーンFT/テスト: ToTensor、Normalize のみ。

- 拡張スケジュール（拡張フェーズのみ）
  - Epoch 60〜100 で RandAugment の magnitude を 10→6 に線形減少。
  - Epoch 80〜160 で Random Erasing の確率を 0.25→ほぼ0 に減少。
  - Mixup / CutMix をバッチごとに Beta(α, α) で適用。Epoch 80 以前は α=0.4、以降は α=0.2。クリーンFT中は無効。

- モデル
  - WideResNet-28-10（ドロップアウト 0.3）、CIFAR 用ステム＋ブロック構成。
  - Conv/Linear には Kaiming 初期化、BatchNorm は学習可能パラメータありで初期化。
  - Global Average Pooling → 全結合による 10 クラスロジット。
  - CUDA 使用時は channels-last と AMP で高速化。

- 最適化とスケジュール
  - Optimizer: SGD（モーメンタム 0.9、Nesterov、初期学習率 0.1）。
  - Weight Decay: 5e-4（畳み込み・全結合の重みのみ、BN/バイアス除外）。
  - 学習率スケジュール: 5 エポックのウォームアップ後、Epoch 144 までは余弦減衰（最小 LR = 基本 LR × 1e-4）。
  - EMA: エポックに応じて減衰を切り替えながら指数移動平均を更新
    - <60: 0.999、60〜99: 0.9995、≥100: 0.9998。
  - オプション: SAM（Sharpness-Aware Minimization, ρ=0.05）を実装済み。拡張フェーズでのみ有効化可能。

- クリーンファインチューニング（終盤）
  - Epoch 145 以降:
    - クリーン用データローダ（拡張なし、Mixup/CutMix なし）に切り替え。
    - Weight Decay を 0 に設定。
    - 学習率を Epoch 145〜154 で 1e-3、155〜159 で 1e-4 に固定。
  - 総エポック数: 160。

- 検証
  - EMA 重みで評価。
  - 評価時の損失はラベルスムージング 0.1 を入れた交差エントロピー。
  - バリデーション精度のベスト値を記録。

- テスト時拡張（TTA）
  - 以下の組み合わせによるロジット平均化:
    - 水平反転（あり/なし）
    - スケール {0.97, 1.00, 1.03}
    - ±1px のシフト（対角含む 9 オフセット）
  - 平均ロジットの argmax を最終予測とする。

- 推論と書き出し
  - 解決されたプロジェクトルート（または環境変数 `CIFAR10_WORK_DIR`）直下の `data/output/submission.csv` に出力。ヘッダは `label`、インデックスは `id`。


## 使用技術

- Python 3
- PyTorch (`torch`, `torchvision`)
- NumPy (`numpy`), Pandas (`pandas`)
- Pillow (`PIL`)


