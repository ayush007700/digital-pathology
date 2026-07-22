from src.data.pcam_dataset import PCamDataset
from src.data.transforms import get_train_transforms

dataset = PCamDataset(

    split="train",

    subset_size=5,

    transform=get_train_transforms(),

)

image, label = dataset[0]

print(type(image))

print(image.shape)

print(label)