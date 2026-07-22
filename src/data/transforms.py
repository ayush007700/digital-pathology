"""
MONAI Transform Pipeline

Author: Ayush Raj
"""

from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    ScaleIntensity,
    RandFlip,
    RandRotate90,
    RandGaussianNoise,
    RandZoom,
    ToTensor,
)


def get_train_transforms():

    return Compose(

        [

            EnsureChannelFirst(channel_dim=-1),

            ScaleIntensity(),

            RandFlip(prob=0.5, spatial_axis=0),

            RandFlip(prob=0.5, spatial_axis=1),

            RandRotate90(prob=0.5),

            RandGaussianNoise(prob=0.15),

            RandZoom(
                prob=0.20,
                min_zoom=0.90,
                max_zoom=1.10,
            ),

            ToTensor(),

        ]

    )


def get_validation_transforms():

    return Compose(

        [

            EnsureChannelFirst(channel_dim=-1),

            ScaleIntensity(),

            ToTensor(),

        ]

    )