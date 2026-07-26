import torch

from src.models.multihead_attention import (
    MultiHeadSelfAttention,
)

model = MultiHeadSelfAttention()

x = torch.randn(
    2,
    37,
    768,
)

y = model(x)

print(x.shape)

print(y.shape)