"""
Medical UNet
"""

from monai.networks.nets import UNet


def build_unet():

    return UNet(

        spatial_dims=2,

        in_channels=3,

        out_channels=2,

        channels=(32,64,128,256),

        strides=(2,2,2),

        num_res_units=2,

    )