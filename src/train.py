"""
Training script for SentimentScope Transformer model.
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer

from src.config import DEFAULT_CONFIG, get_device
from src.dataset import build_imdb_dataframes, get_dataloaders
from src.model import SentimentTransformer


def evaluate_accuracy(model, dataloader, device):
    """
    Computes classification accuracy across a dataloader split.
    """
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            preds = torch.argmax(logits, dim=1)

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    return (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0


def train(data_dir="aclImdb", epochs=3, batch_size=32, lr=3e-4, output_checkpoint="sentiment_transformer.pt"):
    """
    Executes full training and validation cycle over specified epochs.
    """
    device = get_device()
    print(f"Using device: {device}")

    # 1. Load data
    print(f"Loading datasets from {data_dir}...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_df, val_df, test_df = build_imdb_dataframes(root_dir=data_dir)
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, tokenizer, batch_size=batch_size, max_length=DEFAULT_CONFIG["context_size"]
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    # 2. Initialize model, optimizer, criterion
    model = SentimentTransformer(DEFAULT_CONFIG).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # 3. Training Loop
    print("\nStarting Training...")
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for step, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (step + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{step+1}/{len(train_loader)}] Loss: {running_loss/100:.4f}")
                running_loss = 0.0

        val_acc = evaluate_accuracy(model, val_loader, device)
        print(f"--> Epoch {epoch+1} Completed - Validation Accuracy: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_checkpoint)
            print(f"    Checkpoint saved to {output_checkpoint}")

    # 4. Final Test Evaluation
    print("\nEvaluating on Test Set...")
    test_acc = evaluate_accuracy(model, test_loader, device)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    return test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer for Sentiment Analysis.")
    parser.add_argument("--data-dir", type=str, default="aclImdb", help="Path to unzipped aclImdb directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="DataLoader batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--save-path", type=str, default="sentiment_transformer.pt", help="Checkpoint save path")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_checkpoint=args.save_path
    )
