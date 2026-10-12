"""
Unit tests for Transformer architecture layers and forward passes.
"""
import pytest
import torch

from src.config import DEFAULT_CONFIG
from src.model import (
    AttentionHead,
    MultiHeadAttention,
    FeedForward,
    Block,
    SentimentTransformer,
)


@pytest.fixture
def dummy_config():
    return {
        "vocabulary_size": 100,
        "num_classes": 2,
        "d_embed": 64,
        "context_size": 32,
        "layers_num": 2,
        "heads_num": 2,
        "head_size": 32,
        "dropout_rate": 0.0,
        "use_bias": True,
        "causal": False,
    }


@pytest.fixture
def dummy_input(dummy_config):
    batch_size = 4
    return torch.randn(batch_size, dummy_config["context_size"], dummy_config["d_embed"])


def test_attention_head_shape(dummy_config, dummy_input):
    head = AttentionHead(dummy_config)
    out = head(dummy_input)
    b, t, d = dummy_input.shape
    assert out.shape == (b, t, dummy_config["head_size"])


def test_multi_head_attention_shape(dummy_config, dummy_input):
    mha = MultiHeadAttention(dummy_config)
    out = mha(dummy_input)
    assert out.shape == dummy_input.shape


def test_feed_forward_shape(dummy_config, dummy_input):
    ff = FeedForward(dummy_config)
    out = ff(dummy_input)
    assert out.shape == dummy_input.shape


def test_block_shape(dummy_config, dummy_input):
    block = Block(dummy_config)
    out = block(dummy_input)
    assert out.shape == dummy_input.shape


def test_sentiment_transformer_forward(dummy_config):
    model = SentimentTransformer(dummy_config)
    token_ids = torch.randint(0, dummy_config["vocabulary_size"], (4, dummy_config["context_size"]))
    logits = model(token_ids)
    assert logits.shape == (4, dummy_config["num_classes"])


def test_causal_mask_toggle(dummy_config, dummy_input):
    dummy_config["causal"] = True
    head_causal = AttentionHead(dummy_config)
    out = head_causal(dummy_input)
    b, t, d = dummy_input.shape
    assert out.shape == (b, t, dummy_config["head_size"])
