import glob, os
import pandas as pd
import numpy as np
from utils.bev.bev_transform import LiDARToBEV

bev_fn = LiDARToBEV(target_size=224, max_range=80)

for split in ["train", "validation", "test"]:
    files = sorted(glob.glob(f"dataset/{split}/{split}-*.parquet"))
    out_dir = f"dataset_bev/{split}"
    os.makedirs(out_dir, exist_ok=True)

    for filepath in files:
        df = pd.read_parquet(filepath)
        df = df.sort_values(["run_id", "frame"], kind="stable").reset_index(drop=True)
        bevs = []
        
        for lidar in df["lidar"]:
            bev = bev_fn(lidar).numpy()  # (3, 224, 224)
            bevs.append(bev)

        fname = os.path.basename(filepath).replace(".parquet", ".npy")
        np.save(f"{out_dir}/{fname}", np.stack(bevs))  # (N, 3, 224, 224)
        
        print(f"Saved {fname}")
