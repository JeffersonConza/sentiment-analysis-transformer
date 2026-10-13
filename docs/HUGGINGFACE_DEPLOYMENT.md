# 🚀 Hugging Face Spaces Deployment Guide

This guide walks you through deploying **SentimentScope** to **Hugging Face Spaces** for free in under 3 minutes, giving you a live interactive public web demo URL to showcase on LinkedIn and your CV.

---

## 📋 Prerequisites
1. A free account on [huggingface.co](https://huggingface.co/).
2. Your GitHub repository: [`https://github.com/JeffersonConza/sentiment-analysis-transformer`](https://github.com/JeffersonConza/sentiment-analysis-transformer).

---

## 🛠️ Step-by-Step Deployment

### Method 1: Direct GitHub Sync (Recommended)

1. **Create a New Space**:
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space).
   - **Space name**: `sentiment-analysis-transformer`
   - **License**: `MIT`
   - **Space SDK**: Select **Streamlit**.
   - **Hardware**: Select **CPU basic (free)**.
   - **Space visibility**: **Public**.
   - Click **Create Space**.

2. **Connect to GitHub or Clone**:
   - In your newly created Space, navigate to **Settings** ➔ **GitHub Sync** (or clone the Space repo locally):
     ```bash
     git remote add space https://huggingface.co/spaces/YOUR_HF_USERNAME/sentiment-analysis-transformer
     git push space main
     ```

3. **Automatic Build & Launch**:
   - Hugging Face automatically detects:
     - `app.py` as the entrypoint.
     - `requirements.txt` for Python dependencies.
     - `sentiment_transformer.pt` checkpoint for PyTorch weights.
   - Within 1–2 minutes, your Space will be **Running** at:
     ```text
     https://huggingface.co/spaces/YOUR_HF_USERNAME/sentiment-analysis-transformer
     ```

---

## 🎯 Features Available in the Live Space

- **Live Sentiment Prediction**: Real-time positive/negative classification.
- **Explainable AI (XAI)**:
  - Interactive token attention heatmaps showing which words influenced the prediction.
  - Pairwise 2D self-attention matrix across sequence tokens.
  - Top influential word ranking chart.
- **Sequence Diagnostics**: Visual indicator of context window usage (128 tokens) vs. quadratic memory efficiency.
