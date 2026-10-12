"""
Transformer model components and architecture for sequence sentiment classification.
"""
import math
import torch
import torch.nn as nn


class AttentionHead(nn.Module):
    """Single attention head with optional causal masking."""

    def __init__(self, config):
        super().__init__()
        self.Q_weights = nn.Linear(config["d_embed"], config["head_size"], bias=config["use_bias"])
        self.K_weights = nn.Linear(config["d_embed"], config["head_size"], bias=config["use_bias"])
        self.V_weights = nn.Linear(config["d_embed"], config["head_size"], bias=config["use_bias"])
        self.dropout = nn.Dropout(config["dropout_rate"])
        
        self.is_causal = config.get("causal", True)
        casual_attention_mask = torch.tril(torch.ones(config["context_size"], config["context_size"]))
        self.register_buffer("casual_attention_mask", casual_attention_mask)

    def forward(self, x):
        """
        Forward pass for single attention head.
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, d_embed)
        Returns:
            torch.Tensor: Output tensor of shape (B, T, head_size)
        """
        batch_size, tokens_num, d_embed = x.shape
        Q = self.Q_weights(x)  # (B, T, head_size)
        K = self.K_weights(x)  # (B, T, head_size)
        V = self.V_weights(x)  # (B, T, head_size)

        # Scaled dot-product attention
        attention_scores = (Q @ K.transpose(1, 2)) / math.sqrt(K.shape[-1])  # (B, T, T)

        # Optional causal masking (if configured)
        if self.is_causal:
            attention_scores = attention_scores.masked_fill(
                self.casual_attention_mask[:tokens_num, :tokens_num] == 0,
                float("-inf")
            )

        attention_scores = torch.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        return attention_scores @ V


class MultiHeadAttention(nn.Module):
    """Multi-head attention layer combining multiple AttentionHead modules."""

    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(config) for _ in range(config["heads_num"])])
        self.linear = nn.Linear(config["heads_num"] * config["head_size"], config["d_embed"])
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, x):
        heads_outputs = [head(x) for head in self.heads]
        out = torch.cat(heads_outputs, dim=-1)
        out = self.linear(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network with GELU activation."""

    def __init__(self, config):
        super().__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(config["d_embed"], 4 * config["d_embed"]),
            nn.GELU(),
            nn.Linear(4 * config["d_embed"], config["d_embed"]),
            nn.Dropout(config["dropout_rate"])
        )

    def forward(self, x):
        return self.linear_layers(x)


class Block(nn.Module):
    """Transformer block with residual connections and layer normalization."""

    def __init__(self, config):
        super().__init__()
        self.multi_head = MultiHeadAttention(config)
        self.layer_norm_1 = nn.LayerNorm(config["d_embed"])
        self.feed_forward = FeedForward(config)
        self.layer_norm_2 = nn.LayerNorm(config["d_embed"])

    def forward(self, x):
        x = x + self.multi_head(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class SentimentTransformer(nn.Module):
    """
    Complete Transformer model for binary sentiment classification.
    Adapts token embeddings using mean pooling across sequence dimension
    followed by a linear classification head.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_layer = nn.Embedding(config["vocabulary_size"], config["d_embed"])
        self.positional_embedding_layer = nn.Embedding(config["context_size"], config["d_embed"])
        
        self.layers = nn.Sequential(*[Block(config) for _ in range(config["layers_num"])])
        self.layer_norm = nn.LayerNorm(config["d_embed"])
        
        # Classification head: projects pooled representation to class logits
        self.classifier = nn.Linear(config["d_embed"], config["num_classes"], bias=False)

    def forward(self, token_ids):
        """
        Args:
            token_ids (torch.Tensor): Input token indices of shape (B, T)
        Returns:
            torch.Tensor: Logits of shape (B, num_classes)
        """
        batch_size, tokens_num = token_ids.shape

        # Step 1: Embeddings
        x = self.token_embedding_layer(token_ids)
        positions = torch.arange(tokens_num, device=token_ids.device)
        pos_embed = self.positional_embedding_layer(positions)
        x = x + pos_embed.unsqueeze(0)

        # Step 2: Transformer layers & normalization
        x = self.layers(x)
        x = self.layer_norm(x)

        # Step 3: Mean pooling across sequence dimension (dim=1)
        pooled = torch.mean(x, dim=1)

        # Step 4: Classification head
        logits = self.classifier(pooled)
        return logits


# Alias for compatibility with starter code
DemoGPT = SentimentTransformer
