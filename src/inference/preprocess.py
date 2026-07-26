"""
Inference Preprocessing

Author: Ayush Raj
"""

from PIL import Image

import torch

from torchvision import transforms


class ImagePreprocessor:

    def __init__(self, image_size=96):

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __call__(self, image_path):

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        image = image.unsqueeze(0)

        return image