"""
End-to-End Test
"""

import os

import torch

from src.models.resnet18 import MedicalResNet18

from src.pipeline.factory import build_pipeline


def test_complete_pipeline():

    device = torch.device("cpu")

    model = MedicalResNet18(
        pretrained=False,
    )

    checkpoint = torch.load(
        "checkpoints/best_model.pth",
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    pipeline = build_pipeline(
        model,
        device,
    )

    result = pipeline.run(

        image_path=r"C:\Users\ayush\Downloads\pcam.jpg",

        clinical_question="Explain HER2 positive breast cancer.",

    )

    assert result["prediction"] in [
        "Tumor",
        "Normal",
    ]

    assert os.path.exists(
        result["overlay"]
    )

    assert os.path.exists(
        result["report"]
    )