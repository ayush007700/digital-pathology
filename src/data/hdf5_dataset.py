"""
Generic HDF5 Dataset

Author: Ayush Raj
"""

from pathlib import Path
from typing import Callable, Optional

import h5py
import torch
from torch.utils.data import Dataset


class HDF5Dataset(Dataset):
    """
    Generic dataset for loading HDF5 images and labels.
    """

    def __init__(
        self,
        image_path: str | Path,
        label_path: str | Path,
        image_key: str = "x",
        label_key: str = "y",
        transform: Optional[Callable] = None,
        subset_size: Optional[int] = None,
    ):

        self.image_path = Path(image_path)
        self.label_path = Path(label_path)

        self.image_key = image_key
        self.label_key = label_key

        self.transform = transform

        self.image_file = h5py.File(self.image_path, "r")
        self.label_file = h5py.File(self.label_path, "r")

        self.images = self.image_file[self.image_key]
        self.labels = self.label_file[self.label_key]

        self.total_samples = len(self.images)

        self.subset_size = (
            min(subset_size, self.total_samples)
            if subset_size
            else self.total_samples
        )

    def __len__(self):

        return self.subset_size

    def __getitem__(self, idx):

        image = self.images[idx]

        label = int(self.labels[idx].squeeze())

        if self.transform:
            image = self.transform(image)

        return image, label

    def close(self):

        self.image_file.close()
        self.label_file.close()

    def __del__(self):

        self.close()