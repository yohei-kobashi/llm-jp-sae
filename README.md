# LLM-jp SAE
This repository provides code for training and evaluating Sparse Autoencoders (SAEs) on the internal representations of LLM-jp.
We trained SAEs separately on six different checkpoints of [LLM-jp-3-1.8B](https://huggingface.co/llm-jp/llm-jp-3-1.8b) and compared learned features across checkpoints.

- **[Demo Page](https://llm-jp.github.io/llm-jp-sae/)**: Visualize the text samples that activate each SAE feature (100 features per checkpoint).
- **[Model Weights](https://huggingface.co/llm-jp)**: SAE weights for all six checkpoints.
- **[Paper](https://arxiv.org/)**: *"How LLMs Learn: Tracing Internal Representations with Sparse Autoencoders"*


### TODO
- コード整備
  - README.md: 説明・モデル重み等へのリンクも
  - prepare_data.py: データ用意
  - model.py: モデルクラス定義
  - dataset.py: データセットクラス定義
  - train.py: SAE 学習
  - evaluate.py: 評価
    - feature の抽出
    - feature のパターン定量分析
      - 言語
      - 意味粒度
  - visualize.py: 可視化
