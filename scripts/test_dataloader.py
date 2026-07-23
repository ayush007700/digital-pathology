from src.data.datamodule import PCamDataModule

dm = PCamDataModule()

train_loader = dm.train_dataloader()

images, labels = next(iter(train_loader))

print(images.shape)

print(labels.shape)

print(images.dtype)

print(labels.dtype)