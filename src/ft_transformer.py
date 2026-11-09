"""
FT-Transformer (Feature Tokenizer Transformer) Architecture for Tabular Data
Based on: "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., NeurIPS 2021)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NumericalFeatureTokenizer(nn.Module):
    """
    Transforms numerical features into d-dimensional tokens.
    e_j = x_j * W_j + b_j
    """
    def __init__(self, n_num_features: int, d_token: int):
        super().__init__()
        self.n_num_features = n_num_features
        self.d_token = d_token
        # Weight and bias per numerical feature
        self.weight = nn.Parameter(torch.empty(1, n_num_features, d_token))
        self.bias = nn.Parameter(torch.empty(1, n_num_features, d_token))
        self.reset_parameters()

    def reset_parameters(self):
        d = self.d_token
        nn.init.uniform_(self.weight, -1.0 / math.sqrt(d), 1.0 / math.sqrt(d))
        nn.init.uniform_(self.bias, -1.0 / math.sqrt(d), 1.0 / math.sqrt(d))

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        # x_num: (batch_size, n_num_features)
        # return: (batch_size, n_num_features, d_token)
        return x_num.unsqueeze(-1) * self.weight + self.bias


class CategoricalFeatureTokenizer(nn.Module):
    """
    Transforms categorical integer IDs into d-dimensional embeddings.
    """
    def __init__(self, cat_cardinalities: list, d_token: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=card, embedding_dim=d_token)
            for card in cat_cardinalities
        ])

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        # x_cat: (batch_size, n_cat_features)
        # return: (batch_size, n_cat_features, d_token)
        tokens = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.stack(tokens, dim=1)


class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer encoder block with Multihead Attention and FFN.
    Allows extracting attention weights for intrinsic explainability.
    """
    def __init__(self, d_token: int, n_heads: int, ffn_factor: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_token, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_token)
        d_ffn = int(d_token * ffn_factor)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_token),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        normed = self.norm1(x)
        attn_out, attn_weights = self.mha(normed, normed, normed, need_weights=return_attention)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        if return_attention:
            return x, attn_weights
        return x, None


class FTTransformer(nn.Module):
    """
    FT-Transformer for Tabular Crop Recommendation Classification.
    """
    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: list,
        n_classes: int,
        d_token: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        ffn_factor: float = 2.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_num_features = n_num_features
        self.cat_cardinalities = cat_cardinalities
        self.n_classes = n_classes
        self.d_token = d_token

        # Feature Tokenizers
        self.num_tokenizer = NumericalFeatureTokenizer(n_num_features, d_token)
        self.cat_tokenizer = CategoricalFeatureTokenizer(cat_cardinalities, d_token)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_token, n_heads, ffn_factor, dropout)
            for _ in range(n_layers)
        ])

        # Classification Head
        self.norm_head = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, n_classes)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, return_attention: bool = False):
        batch_size = x_num.size(0)
        
        # Tokenize features
        # Tokens: [CLS, Cat_1, ..., Cat_k, Num_1, ..., Num_m]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        cat_tokens = self.cat_tokenizer(x_cat)
        num_tokens = self.num_tokenizer(x_num)
        
        x = torch.cat([cls_tokens, cat_tokens, num_tokens], dim=1)

        attention_maps = []
        for layer in self.layers:
            x, attn_w = layer(x, return_attention=return_attention)
            if return_attention:
                attention_maps.append(attn_w)

        # Output from [CLS] token
        cls_out = self.norm_head(x[:, 0, :])
        logits = self.head(cls_out)

        if return_attention:
            # attention_maps: list of (batch, seq_len, seq_len)
            return logits, attention_maps
        return logits

    def predict_proba(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_num, x_cat)
            probs = F.softmax(logits, dim=-1)
        return probs
