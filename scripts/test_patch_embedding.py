import torch

from src.models.vit_layers import PatchEmbedding

model = PatchEmbedding()

x = torch.randn(
    2,
    3,
    96,
    96,
)

y = model(x)

print(y.shape)