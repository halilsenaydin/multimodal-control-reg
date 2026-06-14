"""Precompute resized RGB front-camera frames → dataset_img/{split}/<stem>.npy.

Mirrors bev/lidar_to_bev.py: one .npy per parquet, rows in the SAME stable
[run_id, frame] order and with one row per frame (no drops), so that row i of the
image .npy aligns with row i of the BEV .npy and with the dataset's bev_rows.

Run once before training the image-stream (phase-d) experiments:
    python images_to_npy.py
"""

import glob
import io
import os

import numpy as np
import pandas as pd
from PIL import Image

TARGET_SIZE = 224
IMG_COL = "image_front"


def _decode(value) -> np.ndarray:
    """Decode one HuggingFace Image cell → (3, H, W) uint8 RGB, resized to TARGET_SIZE."""
    # pyarrow reads a HF Image feature as {"bytes": <png/jpeg bytes>, "path": ...};
    # be defensive about raw bytes too.
    if isinstance(value, dict):
        data = value.get("bytes")
    else:
        data = value
    if data is None:
        return np.zeros((3, TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)

    img = Image.open(io.BytesIO(data)).convert("RGB").resize((TARGET_SIZE, TARGET_SIZE))
    return np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)  # (3, H, W)


def main() -> None:
    for split in ["train", "validation", "test"]:
        files = sorted(glob.glob(f"dataset/{split}/{split}-*.parquet"))
        out_dir = f"dataset_img/{split}"
        os.makedirs(out_dir, exist_ok=True)

        for filepath in files:
            df = pd.read_parquet(filepath, columns=["run_id", "frame", IMG_COL])
            df = df.sort_values(["run_id", "frame"], kind="stable").reset_index(drop=True)

            frames = [_decode(v) for v in df[IMG_COL]]  # list of (3, H, W)
            stack = np.stack(frames)  # (N, 3, 224, 224) uint8

            fname = os.path.basename(filepath).replace(".parquet", ".npy")
            np.save(f"{out_dir}/{fname}", stack)
            print(f"Saved {split}/{fname}  {stack.shape}")


if __name__ == "__main__":
    main()
