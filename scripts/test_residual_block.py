import torch

from src.models.blocks import ResidualBlock

x = torch.randn(
    4,
    32,
    96,
    96,
)

block = ResidualBlock(32)

y = block(x)

print("Input :", x.shape)

print("Output:", y.shape)