from pathlib import Path
import h5py

DATA_DIR = Path("data/raw/pcam")

files = [
    "training_split.h5",
    "validation_split.h5",
    "test_split.h5",
]

for file_name in files:
    print("=" * 70)
    print(file_name)

    file_path = DATA_DIR / file_name

    with h5py.File(file_path, "r") as f:

        print("\nKeys:")
        print(list(f.keys()))

        print("\nDatasets:")

        for key in f.keys():
            obj = f[key]

            print(f"{key}")

            if isinstance(obj, h5py.Dataset):
                print(f"  Shape : {obj.shape}")
                print(f"  Dtype : {obj.dtype}")