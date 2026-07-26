import torch

from src.models.transformer_encoder import (
    TransformerEncoderBlock,
)

model = TransformerEncoderBlock()

x = torch.randn(
    2,
    37,
    768,
)

y = model(x)

print(x.shape)

print(y.shape)