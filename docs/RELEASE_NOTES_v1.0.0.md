# SentimentScope v1.0.0 — Release Notes

**Release Date:** October 13, 2026  
**Target Branch:** `main`  
**Git Tag:** `v1.0.0`  
**Author:** Jefferson Conza  

---

## 🚀 Overview

**SentimentScope v1.0.0** is the first official production release of an end-to-end sentiment classification system built with a **custom Transformer neural network implemented from scratch in PyTorch**. 

Trained and evaluated on the **Stanford IMDB Movie Review dataset** (50,000 reviews), this project was completed as part of the **AWS AI & ML Scholarship** program (Udacity AI Programming with Python) and **officially passed on Attempt 1 with 100% rubric compliance**.

---

## 🏆 Official Evaluation & Benchmark Results

The model was evaluated against 25,000 unseen test reviews, surpassing the rigorous passing bar of 75.0% accuracy on Attempt 1:

| Metric | Target / Baseline | Achieved (Test Set) | Status |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | $\ge 75.00\%$ | **76.88%** | Passed (Attempt 1) |
| **Negative Precision / Recall / F1** | — | **0.7744 / 0.7632 / 0.7688** | Balanced |
| **Positive Precision / Recall / F1** | — | **0.7632 / 0.7744 / 0.7688** | Balanced |
| **Macro Average F1-Score** | — | **0.7688** | High Stability |
| **Final Training Loss** | Initial: ~0.693 | **0.4287** (Epoch 3) | Converged |

> Detailed reviewer remarks and section scores are documented in [`docs/REVIEW_FEEDBACK.md`](REVIEW_FEEDBACK.md).

---

## 🏗️ Technical Highlights & Architecture

1. **From-Scratch PyTorch Transformer**:
   - `MultiHeadAttention`: Implements projection heads, scaled dot-product attention ($\frac{QK^T}{\sqrt{d_k}}$), and head concatenation without high-level library abstractions.
   - Attention masking toggle: Supports both bidirectional (encoder) and causal autoregressive masking.
   - Positional Embeddings: Learned positional encoding combined with token embeddings.
   - Feed-Forward Networks: Two-layer MLP with GELU activations and residual skip connections (`x + SubLayer(LN(x))`).
   - Sequence Mean Pooling: Global mean pooling across active sequence tokens (`dim=1`) mapped to a linear classification projection head.

2. **Model Hyperparameters**:
   - **Layers**: 4 Transformer Blocks
   - **Attention Heads**: 4 heads (32 dim per head = 128 embedding dim)
   - **Vocabulary**: 30,522 (`bert-base-uncased` subword tokenizer)
   - **Context Window**: 128 tokens
   - **Optimizer**: AdamW (`lr=3e-4`, `batch_size=32`, 3 epochs)

3. **Interactive Web Application (`app.py`)**:
   - Standalone **Streamlit** web interface featuring real-time inference, sample prompt loaders, probability breakdown gauges, and sequence length/token diagnostics.
   - Cached model loading via `@st.cache_resource` for instantaneous prediction response.

4. **Production Engineering & Quality Assurance**:
   - **Modular Package (`src/`)**: Decoupled modules for configuration, dataset loading, model layers, CLI training, CLI inference, and metric evaluation.
   - **Unit Tests (`tests/test_model.py`)**: 6 comprehensive unit tests validating tensor dimensions, residual connections, and masking behavior with `pytest`.
   - **Automated CI/CD (`.github/workflows/ci.yml`)**: Continuous integration workflow running on every push and pull request.

---

## 📂 Deliverables Included

```text
sentiment-analysis-transformer/
├── .github/workflows/ci.yml          # Automated CI workflow
├── assets/
│   ├── confusion_matrix.png          # High-resolution test confusion matrix
│   ├── label_distribution.png        # Training set 50/50 class balance plot (EDA)
│   └── review_length_distribution.png # EDA review word count distribution plot
├── docs/
│   ├── REVIEW_FEEDBACK.md            # Official Udacity reviewer evaluation report
│   ├── RELEASE_NOTES_v1.0.0.md       # This release document
│   └── NOTEBOOK_WALKTHROUGH.md       # Complete rendered notebook walkthrough & outputs
├── app.py                            # Interactive Streamlit web application
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git exclusion rules
├── LICENSE                           # MIT License
├── README.md                         # Project documentation and benchmarks
├── sentiment_transformer.pt          # Trained PyTorch model checkpoint (20 MB)
├── SentimentScope_sentiment_analysis.ipynb # Complete experimental notebook
├── src/
│   ├── __init__.py                   # Package exports
│   ├── config.py                     # Centralized hyperparameters & device logic
│   ├── model.py                      # Pure PyTorch Transformer architecture
│   ├── dataset.py                    # IMDB Dataset & DataLoader pipeline
│   ├── train.py                      # Standalone CLI training script
│   ├── infer.py                      # SentimentPredictor inference engine
│   └── evaluate.py                   # Metrics & confusion matrix generator
└── tests/
    ├── __init__.py
    └── test_model.py                 # Unit tests (6/6 passing in pytest)
```

---

## ⚡ Quickstart

### 1. Installation
```bash
git clone https://github.com/JeffersonConza/sentiment-analysis-transformer.git
cd sentiment-analysis-transformer
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Demo
```bash
streamlit run app.py
```

### 3. CLI Prediction
```bash
python -m src.infer --text "An absolute masterpiece with incredible acting and visuals!"
```

### 4. Run Unit Tests
```bash
pytest tests/
```

---

## 📜 Acknowledgements & Verification
- **Dataset:** Maas et al. (2011), Stanford IMDB Sentiment Analysis Dataset.
- **Sponsor / Program:** AWS AI & ML Scholarship / Udacity AI Programming with Python.
- **Evaluation Status:** **Passed on Attempt 1** (October 2026).
