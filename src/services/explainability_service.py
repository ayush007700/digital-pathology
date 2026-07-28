"""
Explainability Service
"""

import cv2
import uuid
from pathlib import Path

from src.explainability.gradcam import GradCAM
from src.explainability.heatmap import generate_heatmap
from src.explainability.overlay import overlay_heatmap


class ExplainabilityService:

    def __init__(self, model):

        self.gradcam = GradCAM(
            model,
            model.model.layer4[-1],
        )

    def explain(
        self,
        image_tensor,
        image_path,
    ):
        device = next(self.gradcam.model.parameters()).device
        cam = self.gradcam.generate(image_tensor.to(device))

        heatmap = generate_heatmap(cam)

        image = cv2.imread(image_path)

        image = cv2.resize(image, (96, 96))

        overlay = overlay_heatmap(
            image / 255.0,
            heatmap,
        )

        output_dir = Path("outputs/overlays")
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = output_dir / f"{uuid.uuid4()}.png"

        cv2.imwrite(str(filename), overlay)

        return str(filename)