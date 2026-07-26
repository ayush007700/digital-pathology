"""
GradCAM

Author: Ayush Raj
"""

import cv2
import numpy as np
import torch


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(
            self.save_activation
        )

        target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, input, output):

        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):

        self.gradients = grad_output[0].detach()

    def generate(self, image):

        self.model.zero_grad()

        logits = self.model(image)

        class_idx = logits.argmax(dim=1)

        logits[:, class_idx].backward()

        gradients = self.gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            gradients * self.activations
        ).sum(dim=1)

        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()

        cam = cv2.resize(
            cam,
            (
                image.shape[-1],
                image.shape[-2],
            ),
        )

        cam -= cam.min()

        cam /= cam.max() + 1e-8

        return cam