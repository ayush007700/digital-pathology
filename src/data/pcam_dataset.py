from pathlib import Path

from src.data.hdf5_dataset import HDF5Dataset


class PCamDataset(HDF5Dataset):

    def __init__(
        self,
        split: str,
        transform=None,
        subset_size=None,
    ):

        root = Path("data/raw")

        image_paths = {

            "train": root / "pcam" / "training_split.h5",

            "valid": root / "pcam" / "validation_split.h5",

            "test": root / "pcam" / "test_split.h5",

        }

        label_paths = {

            "train": root
            / "Labels"
            / "Labels"
            / "camelyonpatch_level_2_split_train_y.h5",

            "valid": root
            / "Labels"
            / "Labels"
            / "camelyonpatch_level_2_split_valid_y.h5",

            "test": root
            / "Labels"
            / "Labels"
            / "camelyonpatch_level_2_split_test_y.h5",

        }

        super().__init__(

            image_path=image_paths[split],

            label_path=label_paths[split],

            transform=transform,

            subset_size=subset_size,

        )