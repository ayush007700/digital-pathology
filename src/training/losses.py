"""
Loss Functions

Author: Ayush Raj
"""

import torch.nn as nn


def get_loss(loss_name: str = "cross_entropy"):
    """
    Returns the requested loss function.
    """

    loss_name = loss_name.lower()

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()

    raise ValueError(f"Unsupported loss: {loss_name}")

    loss = nn.CrossEntropyLoss()

# Why not write inside trainer?

# Because tomorrow we'll add:

# CrossEntropy

# FocalLoss

# DiceLoss

# LabelSmoothing

# WeightedCrossEntropy

# The Trainer never changes.