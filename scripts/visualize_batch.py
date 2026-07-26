from src.data.datamodule import PCamDataModule
from src.visualization.visualize import show_batch

dm = PCamDataModule()

loader = dm.train_dataloader()

images, labels = next(iter(loader))

show_batch(images, labels)