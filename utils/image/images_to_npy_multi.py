"""Precompute resized multi-camera RGB frames → dataset_img_multi/{split}/<stem>.npy.

Phase-e-2: extends images_to_npy.py to all cameras in cfg.IMG_CAMERAS. One .npy per
parquet, rows in the SAME stable [run_id, frame] order with one row per frame (no
drops), so row i aligns with row i of the BEV .npy and the dataset's bev_rows. The
camera axis (axis 1) follows cfg.IMG_CAMERAS order — the same order the dataset and
the left/right flip swap rely on.

Output shape per file: (N, n_cam, 3, 224, 224) uint8.   (~4x the front-only size.)

Run once before training phase-e-2:
    python images_to_npy_multi.py
"""

import glob
import io
import os

import numpy as np
import pandas as pd
from PIL import Image

import config as cfg

TARGET_SIZE = 224
CAMERAS = cfg.IMG_CAMERAS


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
        out_dir = f"{cfg.IMG_DIR}/{split}"
        os.makedirs(out_dir, exist_ok=True)

        for filepath in files:
            df = pd.read_parquet(filepath, columns=["run_id", "frame", *CAMERAS])
            df = df.sort_values(["run_id", "frame"], kind="stable").reset_index(drop=True)

            # Per camera: (N, 3, H, W); stack on a new camera axis → (N, n_cam, 3, H, W).
            per_cam = [np.stack([_decode(v) for v in df[cam]]) for cam in CAMERAS]
            stack = np.stack(per_cam, axis=1)  # (N, n_cam, 3, 224, 224) uint8

            fname = os.path.basename(filepath).replace(".parquet", ".npy")
            np.save(f"{out_dir}/{fname}", stack)
            print(f"Saved {split}/{fname}  {stack.shape}")


if __name__ == "__main__":
    main()
