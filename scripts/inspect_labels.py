from pathlib import Path
import h5py

LABEL_DIR = Path("data/raw/Labels/Labels")

for file in LABEL_DIR.glob("*.h5"):
    print("=" * 60)
    print(file.name)

    with h5py.File(file, "r") as f:
        print("Keys:", list(f.keys()))

        for key in f.keys():
            ds = f[key]
            print(f"{key}")
            print(" Shape :", ds.shape)
            print(" Dtype :", ds.dtype)
            print(" Unique :", set(ds[:100].flatten()))