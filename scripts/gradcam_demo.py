import cv2
import numpy as np
import torch

from src.models.resnet18 import MedicalResNet18

from src.explainability.gradcam import GradCAM
from src.explainability.heatmap import generate_heatmap
from src.explainability.overlay import overlay_heatmap

from src.inference.preprocess import ImagePreprocessor


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = MedicalResNet18(
    pretrained=False,
).to(device)

# TODO
# model.load_state_dict(torch.load(...))

preprocess = ImagePreprocessor()

image_tensor = preprocess(
    "sample.png"
).to(device)

cam = GradCAM(
    model,
    model.model.layer4[-1],
)

cam_map = cam.generate(
    image_tensor
)

heatmap = generate_heatmap(
    cam_map
)

original = cv2.imread("sample.png")

original = cv2.resize(
    original,
    (96, 96),
)

overlay = overlay_heatmap(
    original / 255.0,
    heatmap,
)

cv2.imwrite(
    "outputs/gradcam_overlay.png",
    overlay,
)

print("Saved outputs/gradcam_overlay.png")