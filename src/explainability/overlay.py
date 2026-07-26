"""
Overlay
"""

import cv2
import numpy as np


def overlay_heatmap(image, heatmap, alpha=0.4):

    image = np.uint8(image * 255)

    overlay = cv2.addWeighted(
        image,
        1 - alpha,
        heatmap,
        alpha,
        0,
    )

    return overlay