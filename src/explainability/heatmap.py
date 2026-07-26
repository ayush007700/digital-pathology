"""
Heatmap
"""

import cv2
import numpy as np


def generate_heatmap(cam):

    heatmap = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET,
    )

    return heatmap