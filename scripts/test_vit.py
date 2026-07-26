import torch

from src.models.vit import VisionTransformer

model = VisionTransformer()

x = torch.randn(
    2,
    3,
    96,
    96,
)

y = model(x)

print(model)

print()

print(y.shape)