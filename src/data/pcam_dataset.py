"""
PCam Dataset

Author: Ayush Raj
"""

from pathlib import Path
from typing import Optional, Callable

from src.data.hdf5_dataset import HDF5Dataset
from src.utils.config import config


class PCamDataset(HDF5Dataset):
    """
    PatchCamelyon Dataset
    """

    def __init__(
        self,
        split: str,
        transform: Optional[Callable] = None,
        subset_size: Optional[int] = None,
    ):

        root = Path(config.get("dataset", "root"))

        image_paths = {
            "train": root / config.get("dataset", "pcam", "train_images"),
            "valid": root / config.get("dataset", "pcam", "valid_images"),
            "test": root / config.get("dataset", "pcam", "test_images"),
        }

        label_paths = {
            "train": root / config.get("dataset", "pcam", "train_labels"),
            "valid": root / config.get("dataset", "pcam", "valid_labels"),
            "test": root / config.get("dataset", "pcam", "test_labels"),
        }

        if split not in image_paths:
            raise ValueError(
                f"Invalid split '{split}'. Choose from train, valid or test."
            )

        super().__init__(
            image_path=image_paths[split],
            label_path=label_paths[split],
            transform=transform,
            subset_size=subset_size,
        )