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

    def forward(self, x, return_attention=False):
        """
        Forward pass for single attention head.
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, d_embed)
            return_attention (bool): Whether to return attention weights
        Returns:
            torch.Tensor: Output tensor of shape (B, T, head_size)
            (optional) torch.Tensor: Attention weights of shape (B, T, T)
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

        attention_weights = torch.softmax(attention_scores, dim=-1)
        dropped_weights = self.dropout(attention_weights)

        out = dropped_weights @ V
        if return_attention:
            return out, attention_weights
        return out


class MultiHeadAttention(nn.Module):
    """Multi-head attention layer combining multiple AttentionHead modules."""

    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(config) for _ in range(config["heads_num"])])
        self.linear = nn.Linear(config["heads_num"] * config["head_size"], config["d_embed"])
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, x, return_attention=False):
        if return_attention:
            heads_outputs = []
            heads_attns = []
            for head in self.heads:
                h_out, h_attn = head(x, return_attention=True)
                heads_outputs.append(h_out)
                heads_attns.append(h_attn)
            out = torch.cat(heads_outputs, dim=-1)
            out = self.linear(out)
            # Average attention weights across all heads: shape (B, T, T)
            avg_attention = torch.mean(torch.stack(heads_attns, dim=1), dim=1)
            return self.dropout(out), avg_attention
        else:
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

    def forward(self, x, return_attention=False):
        if return_attention:
            attn_out, attn_weights = self.multi_head(self.layer_norm_1(x), return_attention=True)
            x = x + attn_out
            x = x + self.feed_forward(self.layer_norm_2(x))
            return x, attn_weights
        else:
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
        
        self.layers = nn.ModuleList([Block(config) for _ in range(config["layers_num"])])
        self.layer_norm = nn.LayerNorm(config["d_embed"])
        
        # Classification head: projects pooled representation to class logits
        self.classifier = nn.Linear(config["d_embed"], config["num_classes"], bias=False)

    def forward(self, token_ids, return_attention=False):
        """
        Args:
            token_ids (torch.Tensor): Input token indices of shape (B, T)
            return_attention (bool): Whether to return attention weights from final layer
        Returns:
            torch.Tensor: Logits of shape (B, num_classes)
            (optional) torch.Tensor: Attention weights of shape (B, T, T)
        """
        batch_size, tokens_num = token_ids.shape

        # Step 1: Embeddings
        x = self.token_embedding_layer(token_ids)
        positions = torch.arange(tokens_num, device=token_ids.device)
        pos_embed = self.positional_embedding_layer(positions)
        x = x + pos_embed.unsqueeze(0)

        # Step 2: Transformer layers & normalization
        attn_map = None
        for i, layer in enumerate(self.layers):
            if return_attention and i == len(self.layers) - 1:
                x, attn_map = layer(x, return_attention=True)
            else:
                x = layer(x)

        x = self.layer_norm(x)

        # Step 3: Mean pooling across sequence dimension (dim=1)
        pooled = torch.mean(x, dim=1)

        # Step 4: Classification head
        logits = self.classifier(pooled)
        if return_attention:
            return logits, attn_map
        return logits


# Alias for compatibility with starter code
DemoGPT = SentimentTransformer
