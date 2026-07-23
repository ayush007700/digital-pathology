from src.data.datamodule import PCamDataModule
from src.visualization.statistics import compute_mean_std

dm = PCamDataModule()

loader = dm.train_dataloader()

mean, std = compute_mean_std(loader)

print("=" * 40)
print("Dataset Statistics")
print("=" * 40)

print("Mean :", mean)
print("Std  :", std)