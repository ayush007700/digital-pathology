"""
Predictor

Author: Ayush Raj
"""

import torch


class Predictor:

    def __init__(
            self,
            model,
            device,
        ):

        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def predict(self, image):

        image = image.to(self.device)

        logits = self.model(image)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1,
        )

        return {
            "prediction": prediction.cpu(),
            "confidence": confidence.cpu(),
            "probabilities": probabilities.cpu(),
        }