"""Permutation-equivariant self-attention over a supplied image gallery."""
import torch
from torch import nn


class GalleryContext(nn.Module):
    def __init__(self, dim, depth=2, heads=8, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=dim * 4,
                dropout=dropout, activation="gelu", batch_first=True,
                norm_first=True,
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, features):
        x = features
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
