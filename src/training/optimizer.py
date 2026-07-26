"""
Optimizer Factory
"""

import torch.optim as optim


def get_optimizer(
    model,
    optimizer_name="adam",
    learning_rate=1e-3,
):

    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":

        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )

    if optimizer_name == "sgd":

        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
        )

    raise ValueError(
        f"Unsupported optimizer: {optimizer_name}"
    )