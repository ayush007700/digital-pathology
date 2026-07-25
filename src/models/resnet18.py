"""
Production ResNet18
"""

import torch.nn as nn
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)


class MedicalResNet18(nn.Module):

    def __init__(
        self,
        num_classes=2,
        pretrained=True,
    ):

        super().__init__()

        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        self.model = resnet18(
            weights=weights,
        )

        in_features = self.model.fc.in_features

        self.model.fc = nn.Linear(
            in_features,
            num_classes,
        )

    def forward(self, x):

        return self.model(x)
    
    def freeze_backbone(self):

        for param in self.model.parameters():

            param.requires_grad = False

        for param in self.model.fc.parameters():

            param.requires_grad = True

    def unfreeze(self):

        for param in self.model.parameters():

            param.requires_grad = True
    
# Why Replace FC Layer?

# Original ResNet18

# 1000 Classes - ImageNet

# Our project - 2 Classes

# Tumor & Normal

# So, self.model.fc must change.