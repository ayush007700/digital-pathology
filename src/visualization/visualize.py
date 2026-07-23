"""
Visualization Utilities

Author: Ayush Raj
"""

from typing import Optional

import matplotlib.pyplot as plt
import torch


def show_image(
    image: torch.Tensor,
    label: Optional[int] = None,
) -> None:
    """
    Display a single image.

    Parameters
    ----------
    image : torch.Tensor
        Shape (C,H,W)

    label : int
        Image label
    """

    image = image.permute(1, 2, 0).numpy()

    plt.figure(figsize=(4, 4))
    plt.imshow(image)

    if label is not None:
        plt.title(f"Label : {label}")

    plt.axis("off")
    plt.show()


def show_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    nrows: int = 2,
    ncols: int = 2,
) -> None:
    """
    Display a batch of images.
    """

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(10, 10),
    )

    axes = axes.flatten()


    num_images = min(len(images), nrows * ncols)
    for idx in range(num_images):

        image = images[idx].permute(1,2,0).numpy()

        axes[idx].imshow(image)

        axes[idx].set_title(f"Label: {int(labels[idx])}")

        axes[idx].axis("off")

    for idx in range(num_images, len(axes)):
        axes[idx].axis("off")


    plt.tight_layout()
    plt.show()