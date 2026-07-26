import torch

from src.models.attention import (
    ScaledDotProductAttention,
)

model = ScaledDotProductAttention()

x = torch.randn(
    2,
    37,
    768,
)

y = model(x)

print(x.shape)

print(y.shape)