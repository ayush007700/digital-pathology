"""
Trainer

Author: Ayush Raj
"""

import torch


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        loss_fn,
        optimizer,
        device,
    ):

        self.model = model

        self.train_loader = train_loader

        self.valid_loader = valid_loader

        self.loss_fn = loss_fn

        self.optimizer = optimizer

        self.device = device

        self.model.to(device)