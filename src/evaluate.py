"""
Comprehensive evaluation module for SentimentScope Transformer.
Computes Precision, Recall, F1-Score, and generates Confusion Matrix plots.
"""
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer

from src.config import DEFAULT_CONFIG, get_device
from src.dataset import build_imdb_dataframes, get_dataloaders
from src.model import SentimentTransformer


def plot_confusion_matrix(cm, class_names=("Negative", "Positive"), save_path="assets/confusion_matrix.png"):
    """
    Renders and saves a clean, publication-ready Confusion Matrix heatmap using Matplotlib.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(cax)

    # Set axis labels and ticks
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=11)

    # Annotate values inside matrix cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=13, fontweight="bold")

    ax.set_title("IMDB Sentiment Analysis — Confusion Matrix", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Sentiment Label", fontsize=11, labelpad=10)
    ax.set_ylabel("True Sentiment Label", fontsize=11, labelpad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion Matrix plot successfully saved to: {save_path}")


def evaluate_model(
    data_dir="aclImdb",
    checkpoint_path="sentiment_transformer.pt",
    batch_size=32,
    save_plot=True,
    plot_path="assets/confusion_matrix.png"
):
    """
    Runs full evaluation on the test split, producing:
      - Overall Accuracy
      - Per-class Precision, Recall, F1-Score
      - Confusion Matrix
    """
    device = get_device()
    print(f"Evaluating model on: {device}")

    # 1. Load test data
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    _, _, test_df = build_imdb_dataframes(root_dir=data_dir)
    _, _, test_loader = get_dataloaders(
        test_df, test_df, test_df, tokenizer, batch_size=batch_size, max_length=DEFAULT_CONFIG["context_size"]
    )

    # 2. Load model
    model = SentimentTransformer(DEFAULT_CONFIG).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 3. Collect predictions and ground truth
    all_preds = []
    all_labels = []

    print(f"Running inference across {len(test_loader)} test batches...")
    with torch.no_grad():
        for input_ids, labels in test_loader:
            input_ids = input_ids.to(device)
            logits = model(input_ids)
            preds = torch.argmax(logits, dim=1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # 4. Compute Metrics
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Negative (0)", "Positive (1)"],
        digits=4,
        output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 65)
    print("           DETAILED CLASSIFICATION REPORT (IMDB TEST SET)")
    print("=" * 65)
    print(classification_report(y_true, y_pred, target_names=["Negative (0)", "Positive (1)"], digits=4))
    print("=" * 65)

    if save_plot:
        plot_confusion_matrix(cm, class_names=["Negative", "Positive"], save_path=plot_path)

    return report, cm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Transformer model with classification metrics.")
    parser.add_argument("--data-dir", type=str, default="aclImdb", help="Path to unzipped IMDB dataset folder")
    parser.add_argument("--checkpoint", type=str, default="sentiment_transformer.pt", help="Path to model weights")
    parser.add_argument("--plot-path", type=str, default="assets/confusion_matrix.png", help="Path to save confusion matrix")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Warning: {args.data_dir} directory not found locally.")
        print("To generate test set metrics, ensure the dataset is unzipped or pass --data-dir.")
    else:
        evaluate_model(
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
            plot_path=args.plot_path
        )
