# DL Basics コンペ実装ポートフォリオ

東京大学 深層学習基礎講座における各種コンペティション向けの実装をまとめたリポジトリです。**制約付きの条件下で精度を引き上げるための設計判断・実装・検証**までを一続きで示すことを目的としており、実装力および機械学習パイプラインの実務イメージに近い形でアピールできるよう整理しています。

## このリポジトリで示していること

- **実装の幅**: NumPy での線形・MLP 実装から、CNN・RNN・VAE・制約付き PyTorch MLP まで、タスクに応じた適切な学習アルゴリズムとコード構成を選択していること。
- **制約下での開発経験**: 「NumPy のみ」「特定の `torch.nn` API のみ使用可」「単一ファイル提出」など、講義ルールに沿いつつ性能を追求したこと。
- **本番寄りの工夫**: データ拡張のスケジューリング、EMA / SWA、ラベルスムージング、TTA、マルチシード学習とアンサンブル、特徴設計と校正など、**スコアと安定性の両方**を意識したパイプラインであること。
- **再現性**: 課題ごとに `requirements.txt` と実行コマンドを揃え、第三者が環境を整えれば手順どおりに走査できるようにしていること。

個別のアーキテクチャ選定やハイパーパラメータの意図は、各サブディレクトリの `README.md` に記載しています。

## 参考となるリーダーボード実績

以下は開催時点のリーダーボードに基づく、相対順位と主要指標の一覧です。

| 課題 | ディレクトリ | 順位 | 主要指標 |
|------|--------------|------|----------|
| CIFAR-10 CNN 分類 | [cifar10-cnn-classifier](cifar10-cnn-classifier/) | 8位 / 1,365名 | Accuracy 0.9683 |
| Fashion-MNIST、NumPy MLP | [fashion-mnist-mlp](fashion-mnist-mlp/) | 11位 / 1,466名 | Accuracy 0.9204 |
| Fashion-MNIST、PyTorch 制約付き MLP | [fashion-mnist-scratch-mlp](fashion-mnist-scratch-mlp/) | 3位 / 1,439名 | Accuracy 0.9485 |
| Fashion-MNIST、ソフトマックス回帰 | [fmnist-logreg-classification](fmnist-logreg-classification/) | 15位 / 1,593名 | Accuracy 0.905 |
| IMDb 感情分析、RNN | [imdb-sentiment-rnn](imdb-sentiment-rnn/) | 2位 / 1,263名 | Macro F1 0.93082 |
| Fashion-MNIST、VAE と NLL | [vae_fashionmnist_nll](vae_fashionmnist_nll/) | 3位 / 650名 | NLL 201.556 |

順位はあくまで当時のコンペティション内での結果であり、再現実行でも同一スコアが保証されるものではありません。コードレビュー時は「実装の読みやすさ・妥当性・再現手順の明確さ」を主な評価軸としてください。

## 技術スタック概要

| 領域 | 使用例 |
|------|--------|
| 深層学習 | PyTorch。CNN、RNN、VAE、制約付き MLP |
| 古典的手法・NumPy 実装 | MLP・ソフトマックス回帰の手実装、特徴抽出パイプライン |
| 前処理・評価 | scikit-learn によるデータ分割・前処理・指標計算、および NumPy / Pandas |

課題配布の `*.npy` は著作権・配布ポリシーに従い各自で配置してください。本リポジトリには学習データは含みません。

## 環境構築

Python 3.9 以上を推奨します。VAE 課題では README に沿い 3.12 以上が無難です。依存関係は課題単位で分離しているため、検証したい課題のディレクトリで仮想環境を用意してください。

```bash
cd <課題ディレクトリ>
python -m venv .venv
source .venv/bin/activate   # Windows は .venv\Scripts\activate
pip install -r requirements.txt
```

## 共通のデータ配置

- **パターン A**: `<課題ルート>/data/input/` に `x_train.npy`、ラベル用ファイル `t_train.npy` または `y_train.npy`、必要なら `x_test.npy`
- **パターン B**: `<課題ルート>/data/` 直下に同様のファイル

CIFAR-10 課題はパターン A・B のどちらにも対応しています。CIFAR-10 だけ、環境変数 `CIFAR10_WORK_DIR` でプロジェクトルートを明示できます。

## 再現用の実行コマンド

`<課題ルート>` にデータを配置したうえで、該当ディレクトリをカレントにして実行します。

### CIFAR-10 CNN — GPU 推奨、約 160 epoch

```bash
cd cifar10-cnn-classifier
python src/lecture05_homework.py
```

出力: `data/output/submission.csv`

### Fashion-MNIST、NumPy MLP

コンペ時に近い複数シード・アンサンブルは `--preset_best` を参照してください。細部は該当ディレクトリの README に記載しています。

```bash
cd fashion-mnist-mlp
python script/model.py --preset_best
```

出力: `data/output/y_pred.csv` など。学習済み重みは `data/train/mlp_model.npz` などに保存されます。

### Fashion-MNIST、PyTorch 制約付き MLP

```bash
cd fashion-mnist-scratch-mlp
python src/lecture04_homework.py
```

出力: `data/output/submission.csv`

### Fashion-MNIST、ソフトマックス回帰

```bash
cd fmnist-logreg-classification
python script/main/main.py --mode train --data-dir data/input --output data/output/predictions.csv
```

クロスバリデーションは `--mode cv` で実行します。

### IMDb 感情分析 — マルチシードとアンサンブル

```bash
cd imdb-sentiment-rnn
python src/lecture07_homework.py
```

出力: `data/output/submission_seed_ensemble.csv`

### Fashion-MNIST VAE — アンサンブル提出

学習からのフル再現は README の seed・epoch に従い `--mode train` を繰り返します。チェックポイントが揃っている場合の提出生成例は次のとおりです。

```bash
cd vae_fashionmnist_nll
python submission_code.py --mode ensemble --data_dir data/input --out_csv submission.csv
```

チェックポイント構成は `submission_code.py` 先頭の docstring を参照してください。

## コードマップ

| ディレクトリ | 主なエントリポイント |
|--------------|----------------------|
| cifar10-cnn-classifier | `src/lecture05_homework.py` |
| fashion-mnist-mlp | `script/model.py` |
| fashion-mnist-scratch-mlp | `src/lecture04_homework.py` |
| fmnist-logreg-classification | `script/main/main.py`, `script/main/model.py` |
| imdb-sentiment-rnn | `src/lecture07_homework.py` |
| vae_fashionmnist_nll | `submission_code.py` |
