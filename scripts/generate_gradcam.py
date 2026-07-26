"""
Generate GradCAM Overlay Script

Author: Ayush Raj
"""

import os
import uuid
import torch

from src.explainability.gradcam import GradCAM
from src.explainability.overlay import overlay_heatmap, save_overlay


def main():
    # 1. Setup output directory
    output_dir = "outputs/overlays"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Run GradCAM (Example workflow)
    # image_tensor = ...
    # model = ...
    # gradcam = GradCAM(model, target_layer)
    # heatmap = gradcam.generate(image_tensor)
    # overlay_img = overlay_heatmap(raw_image, heatmap)

    # 3. Generate unique filename (Step 5)
    unique_filename = f"{uuid.uuid4()}.png"
    output_path = os.path.join(output_dir, unique_filename)

    # 4. Save using your overlay utility
    save_overlay(
        original="raw_image",
        overlay="overlay_img",
        output_path=output_path,
    )

    print(f"Overlay successfully saved to: {output_path}")


if __name__ == "__main__":
    main()