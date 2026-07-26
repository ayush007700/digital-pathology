"""
Multi-Head Self Attention

Author: Ayush Raj
"""

import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
    ):

        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):

        B, N, C = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(
            B,
            N,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        K = K.view(
            B,
            N,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        V = V.view(
            B,
            N,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1),
        )

        scores /= math.sqrt(self.head_dim)

        attention = torch.softmax(
            scores,
            dim=-1,
        )

        output = torch.matmul(
            attention,
            V,
        )

        output = output.transpose(
            1,
            2,
        ).contiguous()

        output = output.view(
            B,
            N,
            C,
        )

        output = self.out(output)

        return output