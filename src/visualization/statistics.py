"""
Dataset Statistics Utilities

Author: Ayush Raj
"""

import torch
from tqdm import tqdm


def compute_mean_std(dataloader):
    """
    Compute channel-wise mean and std.
    """

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    for images, _ in tqdm(dataloader):

        batch_size = images.size(0)

        images = images.view(batch_size, 3, -1)

        mean += images.mean(dim=2).sum(dim=0)

        std += images.std(dim=2).sum(dim=0)

        total_images += batch_size

    mean /= total_images
    std /= total_images

    return mean, std