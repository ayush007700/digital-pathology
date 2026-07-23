"""
PCam DataModule

Author: Ayush Raj
"""

from torch.utils.data import DataLoader

from src.data.pcam_dataset import PCamDataset
from src.data.transforms import (
    get_train_transforms,
    get_validation_transforms,
)
from src.utils.config import config
import torch

class PCamDataModule:

    def __init__(self):

        self.batch_size = config.get("dataset", "batch_size")
        self.num_workers = config.get("dataset", "num_workers")
        self.shuffle = config.get("dataset", "shuffle")

        # Auto configure based on hardware
        self.pin_memory = torch.cuda.is_available()
        self.persistent_workers = self.num_workers > 0

        # self.persistent_workers = config.get(
        #     "dataset",
        #     "persistent_workers",
        # )

        # self.shuffle = config.get("dataset", "shuffle")

        print("=" * 50)
        print("DataModule Configuration")
        print(f"Batch Size        : {self.batch_size}")
        print(f"Num Workers       : {self.num_workers}")
        print(f"Pin Memory        : {self.pin_memory}")
        print(f"Persistent Worker : {self.persistent_workers}")
        print(f"CUDA Available    : {torch.cuda.is_available()}")
        print("=" * 50)

    def train_dataloader(self):

        dataset = PCamDataset(

            split="train",

            subset_size=500,

            transform=get_train_transforms(),

        )

        return DataLoader(

            dataset,

            batch_size=self.batch_size,

            shuffle=self.shuffle,

            num_workers=self.num_workers,

            pin_memory=self.pin_memory,

            persistent_workers=self.persistent_workers,

        )

    def valid_dataloader(self):

        dataset = PCamDataset(

            split="valid",

            subset_size=100,

            transform=get_validation_transforms(),

        )

        return DataLoader(

            dataset,

            batch_size=self.batch_size,

            shuffle=False,

            num_workers=self.num_workers,

            pin_memory=self.pin_memory,

            persistent_workers=self.persistent_workers,

        )

    def test_dataloader(self):

        dataset = PCamDataset(

            split="test",

            subset_size=100,

            transform=get_validation_transforms(),

        )

        return DataLoader(

            dataset,

            batch_size=self.batch_size,

            shuffle=False,

            num_workers=self.num_workers,

            pin_memory=self.pin_memory,

            persistent_workers=self.persistent_workers,

        )