"""
Loads trained model once during API startup.
"""

import torch

from src.models.resnet18 import MedicalResNet18


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def load_model():

    model = MedicalResNet18(
        pretrained=False,
    )

    checkpoint = torch.load(
        "checkpoints/best_model.pth",
        map_location=DEVICE,
    )

    # model.load_state_dict(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"]
)

    model.to(DEVICE)

    model.eval()

    return model