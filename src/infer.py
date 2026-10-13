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

    def predict(self, review_text: str, return_attention: bool = False):
        """
        Predicts sentiment for a single review string, with optional token-level
        attention weight analysis for model explainability.

        Returns:
            dict: Prediction metadata, class probabilities, token counts,
                  and optional token attention weights.
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
            if return_attention:
                logits, attn_map = self.model(input_ids, return_attention=True)
            else:
                logits = self.model(input_ids)
                attn_map = None

            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(logits, dim=1).item()

        label = "Positive" if pred_class == 1 else "Negative"
        confidence = probs[0][pred_class].item() * 100.0

        # Raw token count before truncation
        raw_token_count = len(self.tokenizer.encode(review_text, truncation=False))
        valid_len = min(raw_token_count, self.config["context_size"])

        res = {
            "text": review_text,
            "label": label,
            "class_id": pred_class,
            "confidence": round(confidence, 2),
            "probabilities": {
                "Negative": round(probs[0][0].item() * 100.0, 2),
                "Positive": round(probs[0][1].item() * 100.0, 2),
            },
            "token_count": raw_token_count,
            "tokens_used": valid_len,
        }

        if return_attention and attn_map is not None:
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0][:valid_len])
            active_attn = attn_map[0, :valid_len, :valid_len].cpu().float()
            
            # Column-wise average (how much attention token j received from all tokens)
            token_scores = active_attn.mean(dim=0).numpy()
            score_min, score_max = token_scores.min(), token_scores.max()
            if score_max > score_min:
                norm_scores = (token_scores - score_min) / (score_max - score_min)
            else:
                norm_scores = [0.5] * len(token_scores)

            res["tokens"] = [
                {
                    "token": t,
                    "score": round(float(s), 4),
                    "normalized": round(float(n), 4)
                }
                for t, s, n in zip(tokens, token_scores, norm_scores)
            ]
            res["attention_matrix"] = active_attn.numpy().tolist()

        return res

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
