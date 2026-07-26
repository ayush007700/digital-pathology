import os
import h5py
import cv2

IMAGE_FILE = "data/raw/pcam/training_split.h5"
LABEL_FILE = "data/raw/Labels/Labels/camelyonpatch_level_2_split_train_y.h5"

OUTPUT_DIR = "sample_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with h5py.File(IMAGE_FILE, "r") as images, \
     h5py.File(LABEL_FILE, "r") as labels:

    X = images["x"]
    y = labels["y"]

    normal = 0
    tumor = 0

    for i in range(len(X)):

        img = X[i]
        label = int(y[i].squeeze())

        if label == 0 and normal < 5:

            cv2.imwrite(
                f"{OUTPUT_DIR}/normal_{normal}.png",
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )

            normal += 1

        elif label == 1 and tumor < 5:

            cv2.imwrite(
                f"{OUTPUT_DIR}/tumor_{tumor}.png",
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )

            tumor += 1

        if normal == 5 and tumor == 5:
            break

print("Done.")