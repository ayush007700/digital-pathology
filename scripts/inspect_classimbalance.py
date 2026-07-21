import h5py
import numpy as np

with h5py.File("data/raw/Labels/Labels/camelyonpatch_level_2_split_train_y.h5", "r") as f:
    labels = f["y"][:].squeeze()

unique, counts = np.unique(labels, return_counts=True)

print(dict(zip(unique, counts)))