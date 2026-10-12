"""
SentimentScope: Transformer-based sentiment analysis package.
"""
from src.config import DEFAULT_CONFIG, get_device
from src.model import SentimentTransformer, DemoGPT

__all__ = ["DEFAULT_CONFIG", "get_device", "SentimentTransformer", "DemoGPT"]
