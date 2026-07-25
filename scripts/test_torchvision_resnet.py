import torch

from src.models.resnet18 import MedicalResNet18

model = MedicalResNet18()

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