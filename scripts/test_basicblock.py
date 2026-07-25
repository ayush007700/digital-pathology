import torch

from src.models.resnet_blocks import BasicBlock

x = torch.randn(2, 64, 96, 96)

block = BasicBlock(
    64,
    128,
    stride=2,
)

y = block(x)

print(x.shape)

print(y.shape)