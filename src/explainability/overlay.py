"""
Overlay Visualization Utilities

Author: Ayush Raj
"""

import cv2
import numpy as np


def overlay_heatmap(image, heatmap, alpha=0.4):
    """Blends a 2D GradCAM heatmap with the original image.

    Args:
        image (np.ndarray): Original image array (float [0, 1] or uint8 [0, 255]).
        heatmap (np.ndarray): Normalized 2D GradCAM heatmap array with values in [0, 1].
        alpha (float): Blending weight for the heatmap overlay (0.0 to 1.0).

    Returns:
        np.ndarray: Blended BGR image array (uint8).
    """
    # Ensure original image is uint8 in range [0, 255]
    if image.max() <= 1.0:
        image = np.uint8(image * 255)
    else:
        image = np.uint8(image)

    # Convert grayscale image to 3-channel BGR if necessary
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Convert 2D float heatmap [0, 1] to uint8 [0, 255] and apply colormap
    heatmap_uint8 = np.uint8(heatmap * 255)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Blend original image and colored heatmap
    overlay = cv2.addWeighted(
        image,
        1 - alpha,
        colored_heatmap,
        alpha,
        0,
    )

    return overlay


def save_overlay(
    original,
    overlay,
    output_path,
):
    """Saves the generated overlay image to disk.

    Args:
        original (np.ndarray): Original base image (unused in simple save, but kept for interface/logging).
        overlay (np.ndarray): Blended overlay image array to save.
        output_path (str): File destination path.
    """
    cv2.imwrite(
        output_path,
        overlay,
    )