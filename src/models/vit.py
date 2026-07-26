"""
Vision Transformer

Author: Ayush Raj
"""

import torch.nn as nn

from src.models.vit_layers import PatchEmbedding
from src.models.transformer_encoder import TransformerEncoderBlock


class VisionTransformer(nn.Module):

    def __init__(
        self,
        image_size=96,
        patch_size=16,
        num_classes=2,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        dropout=0.1,
    ):

        super().__init__()

        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        self.encoder = nn.Sequential(

            *[
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )

                for _ in range(depth)

            ]

        )

        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(
            embed_dim,
            num_classes,
        )

    def forward(self, x):

        x = self.patch_embedding(x)

        x = self.encoder(x)

        x = self.norm(x)

        cls = x[:, 0]

        logits = self.head(cls)

        return logits