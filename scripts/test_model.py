import torch

from src.models.simple_cnn import SimpleCNN

model = SimpleCNN()

x = torch.randn(
    4,
    3,
    96,
    96,
)

print(model)

y = model(x)

print()

print("Input Shape :", x.shape)

print("Output Shape:", y.shape)