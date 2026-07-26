"""
Mini ResNet

Author: Ayush Raj
"""

import torch.nn as nn

from src.models.blocks import ResidualBlock

class ResNet(nn.Module):

    def __init__(

        self,

        num_classes=2,

    ):

        super().__init__()

        self.in_channels = 64

        self.stem = nn.Sequential(

            nn.Conv2d(

                3,

                64,

                kernel_size=7,

                stride=2,

                padding=3,

                bias=False,

            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(

                kernel_size=3,

                stride=2,

                padding=1,

            ),

        )

        self.layer1 = self._make_layer(

            64,

            blocks=2,

            stride=1,

        )

        self.layer2 = self._make_layer(

            128,

            blocks=2,

            stride=2,

        )

        self.layer3 = self._make_layer(

            256,

            blocks=2,

            stride=2,

        )

        self.layer4 = self._make_layer(

            512,

            blocks=2,

            stride=2,

        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(

            512,

            num_classes,

        )


# class MiniResNet(nn.Module):

#     def __init__(self, num_classes=2):

#         super().__init__()

#         self.stem = nn.Sequential(

#             nn.Conv2d(
#                 3,
#                 32,
#                 kernel_size=3,
#                 padding=1,
#                 bias=False,
#             ),

#             nn.BatchNorm2d(32),

#             nn.ReLU(inplace=True),

#         )

#         self.layer1 = ResidualBlock(32)

#         self.pool = nn.MaxPool2d(2)

#         self.layer2 = nn.Sequential(

#             nn.Conv2d(
#                 32,
#                 64,
#                 kernel_size=3,
#                 padding=1,
#                 bias=False,
#             ),

#             nn.BatchNorm2d(64),

#             nn.ReLU(inplace=True),

#             nn.MaxPool2d(2),

#         )

#         self.layer3 = ResidualBlock(64)

#         self.classifier = nn.Sequential(

#             nn.AdaptiveAvgPool2d(1),

#             nn.Flatten(),

#             nn.Linear(
#                 64,
#                 num_classes,
#             ),
#         )

#     def forward(self, x):

#         x = self.stem(x)

#         x = self.layer1(x)

#         x = self.pool(x)

#         x = self.layer2(x)

#         x = self.layer3(x)

#         x = self.classifier(x)

#         return x