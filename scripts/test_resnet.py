import torch

from src.models.resnet import MiniResNet

# model = MiniResNet()

from torchvision.models import resnet18

model = resnet18(
    weights="DEFAULT",
)

x = torch.randn(
    4,
    3,
    96,
    96,
)

y = model(x)

print(model)

print()

print(x.shape)

print(y.shape)