from src.models.efficientnet import MedicalEfficientNet
import torch

model = MedicalEfficientNet()

x = torch.randn(2,3,96,96)

print(model(x).shape)