"""
Scaled Dot Product Attention

Author: Ayush Raj
"""

import math
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):

    def __init__(self, embed_dim=768):

        super().__init__()

        self.query = nn.Linear(embed_dim, embed_dim)

        self.key = nn.Linear(embed_dim, embed_dim)

        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):

        Q = self.query(x)

        K = self.key(x)

        V = self.value(x)

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1),
        )

        scores = scores / math.sqrt(Q.size(-1))

        attention = torch.softmax(
            scores,
            dim=-1,
        )

        output = torch.matmul(
            attention,
            V,
        )

        return output