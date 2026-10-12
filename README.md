# SentimentScope: Transformer-based Sentiment Analysis in PyTorch

[![CI](https://github.com/JeffersonConza/sentiment-analysis-transformer/actions/workflows/ci.yml/badge.svg)](https://github.com/JeffersonConza/sentiment-analysis-transformer/actions/workflows/ci.yml)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)

An end-to-end Natural Language Processing (NLP) system implementing and training a **custom Transformer model from scratch** in PyTorch for binary sentiment classification on the **IMDB Movie Review dataset** (50,000 reviews).

---

## 📌 Project Highlights

- **From-Scratch PyTorch Transformer**: Multi-head self-attention (`MultiHeadAttention`), causal/bidirectional attention toggle, position embeddings, GELU feed-forward blocks, and layer normalization built without pre-packaged transformer heads.
- **Sequence Classification Adaptation**: Adapts encoder/decoder blocks for classification using sequence **mean pooling** (`dim=1`) mapped to a linear classification projection head.
- **Subword Tokenization**: Integrated Hugging Face's `AutoTokenizer` (`bert-base-uncased`) with padding and truncation to standardize sequences to a 128-token context window.
- **Strong Performance**: Achieved **76.88% test accuracy** on 25,000 unseen test reviews within 3 epochs of training using `AdamW` and cross-entropy loss.
- **Modular Production Code**: Refactored from experimental notebooks into modular Python packages (`src/`), complete with CLI utilities, unit tests (`pytest`), and GitHub Actions CI.

---

## 🏗️ Architecture & Model Configuration

| Hyperparameter | Value | Description |
| :--- | :---: | :--- |
| **Vocabulary Size** | 30,522 | BERT uncased vocabulary |
| **Embedding Dimension (`d_embed`)** | 128 | Hidden state / embedding dimension |
| **Context Length (`context_size`)** | 128 | Maximum sequence token length |
| **Transformer Layers (`layers_num`)** | 4 | Number of stacked transformer blocks |
| **Attention Heads (`heads_num`)** | 4 | Multi-head attention heads |
| **Head Dimension (`head_size`)** | 32 | Dimension per attention head (4 × 32 = 128) |
| **Dropout Rate** | 0.1 | Regularization rate |
| **Output Classes** | 2 | Binary sentiment (0: Negative, 1: Positive) |

```
[Input Review Text]
        │
[BERT Subword Tokenizer (max_length=128)]
        │
[Token Embeddings + Positional Embeddings]
        │
┌───────┴──────────────────────────────────────┐
│  Transformer Block × 4                       │
│  ├── Multi-Head Self-Attention (4 heads)     │
│  ├── Residual Connection + LayerNorm         │
│  ├── Feed-Forward Network (GELU)             │
│  └── Residual Connection + LayerNorm         │
└───────┬──────────────────────────────────────┘
        │
[Mean Pooling across Sequence (dim=1)] ──> shape: (Batch, 128)
        │
[Linear Classification Head] ───────────> shape: (Batch, 2)
        │
[Softmax / Argmax Prediction] (Positive / Negative)
```

---

## 📊 Training & Evaluation Results

The model was trained for **3 epochs** using `AdamW` (learning rate `3e-4`) on an NVIDIA GPU using PyTorch `DataLoader` (`batch_size=32`).

| Stage | Loss | Accuracy |
| :--- | :---: | :---: |
| **Initial Baseline (Untrained)** | ~0.6931 | 52.00% |
| **Epoch 1** | 0.5666 | 71.44% |
| **Epoch 2** | 0.4755 | 74.36% |
| **Epoch 3** | **0.4287** | **77.52%** |
| **Final Test Set Evaluation** | — | **76.88%** |

---

## 🚀 Quickstart & Usage

### 1. Installation
Clone the repository and install required dependencies:
```bash
git clone https://github.com/JeffersonConza/sentiment-analysis-transformer.git
cd sentiment-analysis-transformer
pip install -r requirements.txt
```

### 2. Run Inference from CLI
Use the modular inference tool to predict sentiment on any review text:
```bash
python -m src.infer --text "The cinematography was breathtaking and the plot was engaging!"
```
**Output:**
```text
============================================================
Review: "The cinematography was breathtaking and the plot was engaging!"
Sentiment: Positive (98.87% confidence)
============================================================
```

### 3. Programmatic Python API
```python
from src.infer import SentimentPredictor

predictor = SentimentPredictor(checkpoint_path="sentiment_transformer.pt")
result = predictor.predict("The plot was completely predictable and dull.")
print(result)
# {'text': 'The plot was completely predictable and dull.', 'label': 'Negative', 'class_id': 0, 'confidence': 96.42}
```

### 4. Running the Unit Tests
Verify model components, layer dimensions, and masking logic with `pytest`:
```bash
pytest tests/
```

---

## 📂 Repository Structure

```text
sentiment-analysis-transformer/
├── .github/workflows/ci.yml          # Automated CI pipeline running unit tests on push
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Excludes local artifacts and datasets
├── README.md                         # Project documentation and benchmarks
├── sentiment_transformer.pt          # Trained PyTorch model checkpoint weights
├── SentimentScope_sentiment_analysis.ipynb # Complete exploratory and training notebook
├── src/
│   ├── __init__.py                   # Package exports
│   ├── config.py                     # Centralized model hyperparameters
│   ├── model.py                      # MultiHeadAttention, Block, SentimentTransformer
│   ├── dataset.py                    # IMDBDataset and DataLoader pipeline
│   ├── train.py                      # Standalone CLI training script
│   └── infer.py                      # SentimentPredictor inference engine
└── tests/
    ├── __init__.py
    └── test_model.py                 # Unit tests for transformer forward passes & shapes
```

---

## 📜 Acknowledgements & Dataset
- Dataset: [Stanford IMDB Sentiment Analysis Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) (Maas et al., 2011).
- Developed as part of the **AWS AI & ML Scholarship / Udacity AI Programming** program.