from src.data.hdf5_dataset import HDF5Dataset

dataset = HDF5Dataset(
    image_path="data/raw/pcam/training_split.h5",
    label_path="data/raw/Labels/Labels/camelyonpatch_level_2_split_train_y.h5",
    subset_size=10,
)

print("Dataset Size:", len(dataset))

image, label = dataset[0]

print(image.shape)
print(type(label))
print(label)