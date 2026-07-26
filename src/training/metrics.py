"""
Evaluation Metrics
"""

import torch


def accuracy(predictions, labels):

    predicted = torch.argmax(
        predictions,
        dim=1,
    )

    correct = (
        predicted == labels
    ).sum().item()

    return correct / len(labels)

# Why argmax() ?

# Because CrossEntropy outputs logits.

# We need the class with the largest logit.