"""
Inference module for predicting movie review sentiment using the trained Transformer.
"""
import argparse
import torch
from transformers import AutoTokenizer

from src.config import DEFAULT_CONFIG, get_device
from src.model import SentimentTransformer


class SentimentPredictor:
    """
    High-level predictor class that encapsulates tokenization, model loading,
    and confidence probability generation.
    """

    def __init__(self, checkpoint_path="sentiment_transformer.pt", config=None, device=None):
        self.device = device or get_device()
        self.config = config or DEFAULT_CONFIG
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Instantiate model architecture and load weights
        self.model = SentimentTransformer(self.config).to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def predict(self, review_text: str):
        """
        Predicts sentiment for a single review string.

        Returns:
            dict: {
                "text": str,
                "label": "Positive" or "Negative",
                "class_id": 1 or 0,
                "confidence": float (percentage, e.g. 98.5)
            }
        """
        inputs = self.tokenizer(
            review_text,
            truncation=True,
            padding="max_length",
            max_length=self.config["context_size"],
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(logits, dim=1).item()

        label = "Positive" if pred_class == 1 else "Negative"
        confidence = probs[0][pred_class].item() * 100.0

        return {
            "text": review_text,
            "label": label,
            "class_id": pred_class,
            "confidence": round(confidence, 2)
        }

    def predict_batch(self, review_texts: list):
        """Predicts sentiment for a list of review strings."""
        return [self.predict(t) for t in review_texts]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sentiment for a movie review.")
    parser.add_argument(
        "--text",
        type=str,
        default="This movie had fantastic directing and stellar performances!",
        help="Review text to classify"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="sentiment_transformer.pt",
        help="Path to trained model weights"
    )
    args = parser.parse_args()

    predictor = SentimentPredictor(checkpoint_path=args.checkpoint)
    result = predictor.predict(args.text)

    print("=" * 60)
    print(f"Review: \"{result['text']}\"")
    print(f"Sentiment: {result['label']} (Confidence: {result['confidence']}%)")
    print("=" * 60)
