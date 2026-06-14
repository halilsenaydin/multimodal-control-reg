import random
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional
import config as cfg


def extract_box_features(
    boxes, y_thresh=0.6, x_center_range=(0.4, 0.6), area_thresh=0.02
):
    if boxes is None or (isinstance(boxes, float) and np.isnan(boxes)):
        return {
            "front_object_present": 0,
            "front_object_centered": 0,
            "front_object_low": 0,
        }

    boxes = np.array(boxes)

    if boxes.size == 0:
        return {
            "front_object_present": 0,
            "front_object_centered": 0,
            "front_object_low": 0,
        }

    if boxes.dtype == object:
        try:
            boxes = np.vstack([np.asarray(b, dtype=np.float32) for b in boxes])
        except ValueError:
            return {
                "front_object_present": 0,
                "front_object_centered": 0,
                "front_object_low": 0,
            }
    else:
        boxes = boxes.astype(np.float32)

    # Thresholds assume normalized [0,1] box coordinates (YOLO format).
    # Auto-normalize if coordinates appear to be in pixel space (max > 2).
    if boxes.ndim >= 2 and boxes.shape[-1] >= 4:
        max_val = boxes[:, :4].max()
        if max_val > 2.0:
            x_max = max(boxes[:, 0].max(), boxes[:, 2].max())
            y_max = max(boxes[:, 1].max(), boxes[:, 3].max())
            if x_max > 0:
                boxes[:, [0, 2]] /= x_max
            if y_max > 0:
                boxes[:, [1, 3]] /= y_max

    front_present = centered = low = 0

    for box in boxes:
        if len(box) < 4:
            continue

        x1, y1, x2, y2 = box[:4]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        area = (x2 - x1) * (y2 - y1)

        if cy > y_thresh and area > area_thresh:
            front_present = low = 1

            if x_center_range[0] <= cx <= x_center_range[1]:
                centered = 1

    return {
        "front_object_present": front_present,
        "front_object_centered": centered,
        "front_object_low": low,
    }


def extract_lidar_front_distance(lidar, front_ratio=0.2):
    if lidar is None or (isinstance(lidar, float) and np.isnan(lidar)):
        return np.nan

    try:
        arr = np.asarray(lidar, dtype=object)
        if arr.ndim == 1 and arr.dtype == object:
            arr = np.vstack(arr).astype(np.float32)
        else:
            arr = arr.astype(np.float32).flatten()
            remainder = arr.size % 4
            if remainder:
                arr = arr[:-remainder]
            if arr.size == 0:
                return np.nan
            arr = arr.reshape(-1, 4)
    except Exception:
        return np.nan

    if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] == 0:
        return np.nan

    x, y = arr[:, 0], arr[:, 1]
    # Frontal cone: x > 0 (forward) within ±(front_ratio * 180°) of the x-axis.
    # front_ratio=0.2 → ±36° half-angle cone.
    half_angle = np.pi * front_ratio
    front_mask = (x > 0) & (np.abs(np.arctan2(y, x)) <= half_angle)

    if not front_mask.any():
        return np.nan

    distances = np.sqrt(x[front_mask] ** 2 + y[front_mask] ** 2)
    finite_mask = np.isfinite(distances)
    if not finite_mask.any():
        return np.nan

    return float(distances[finite_mask].min())


def compute_object_risk(front_present, centered, lidar_dist, dist_thresh=10.0):
    if not front_present:
        return 0.0

    risk = 0.5

    if centered:
        risk += 0.3

    if not np.isnan(lidar_dist) and lidar_dist < dist_thresh:
        risk += 0.2

    return min(risk, 1.0)


EGO_COLS = [
    "speed_kmh",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "rotation_pitch",
    "rotation_yaw",
    "nearby_vehicles_50m",
    "total_npc_vehicles",
    "total_npc_walkers",
    "weather_cloudiness",
    "weather_fog_density",
    "is_stopped",
    "accel_x",
    "accel_y",
    "accel_z",
    "accel_magnitude",
    "delta_speed",
    "object_risk_score",
    "front_object_present",
    "front_object_centered",
    "min_lidar_distance_front",
]

TARGET_COLS = ["brake", "throttle", "steer"]

LOAD_COLS = [
    "run_id",
    "frame",
    "speed_kmh",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "rotation_pitch",
    "rotation_yaw",
    "nearby_vehicles_50m",
    "total_npc_vehicles",
    "total_npc_walkers",
    "weather_cloudiness",
    "weather_fog_density",
    "throttle",
    "steer",
    "brake",
    "lidar",
    "boxes",
    "box_labels",
]


# File loading


def _load_file(parquet_path: str):
    """Load one parquet and apply feature engineering.

    Returns (ego, labels, run_ids, surviving_positions) or raises.
    """
    df = pd.read_parquet(parquet_path, columns=LOAD_COLS)
    df = df.sort_values(["run_id", "frame"], kind="stable").reset_index(drop=True)

    # box features
    box_feats = df["boxes"].apply(extract_box_features)
    df = pd.concat([df, pd.DataFrame(list(box_feats))], axis=1)

    # lidar distance
    df["min_lidar_distance_front"] = df["lidar"].apply(extract_lidar_front_distance)
    # NaN means no point in the frontal cone → treat as "no obstacle at sensor range"
    df["min_lidar_distance_front"] = df["min_lidar_distance_front"].fillna(cfg.LIDAR_MAX_RANGE)

    # object risk
    df["object_risk_score"] = df.apply(
        lambda r: compute_object_risk(
            r["front_object_present"],
            r["front_object_centered"],
            r["min_lidar_distance_front"],
        ),
        axis=1,
    )

    # drop heavy raw cols
    df.drop(
        columns=[
            "boxes",
            "box_labels",
            "lidar",
            "front_object_low",
        ],
        inplace=True,
        errors="ignore",
    )

    # is_stopped
    df["is_stopped"] = (df["speed_kmh"] == 0).astype(int)

    # drop idle rows
    idle_mask = (
        (df["speed_kmh"] == 0) & (df["throttle"] == 0) & (df["brake"] == 0)
    ).tolist()

    # accel / delta_speed — groupby run_id to avoid cross-run boundary artifacts
    df["accel_x"] = df.groupby("run_id")["velocity_x"].diff().fillna(0)
    df["accel_y"] = df.groupby("run_id")["velocity_y"].diff().fillna(0)
    df["accel_z"] = df.groupby("run_id")["velocity_z"].diff().fillna(0)
    df["accel_magnitude"] = np.sqrt(
        df["accel_x"] ** 2 + df["accel_y"] ** 2 + df["accel_z"] ** 2
    )
    df["delta_speed"] = df.groupby("run_id")["speed_kmh"].diff().fillna(0)
    df.drop(
        columns=["frame"],
        inplace=True,
        errors="ignore",
    )

    surviving_positions = df.index.tolist()
    ego = df[EGO_COLS].values.astype(np.float32)
    labels = df[TARGET_COLS].values.astype(np.float32)
    run_ids = df["run_id"].tolist() if "run_id" in df.columns else ["default"] * len(df)

    return ego, labels, run_ids, surviving_positions, idle_mask


def _load_file_safe(args):
    """Wrapper for ProcessPoolExecutor — returns (file_idx, result | exception)."""
    file_idx, parquet_path = args
    try:
        return file_idx, _load_file(parquet_path)
    except Exception as e:
        return file_idx, e


# Dataset


class CarlaSequenceDataset(Dataset):
    """Sequence dataset for CARLA driving data."""

    def __init__(
        self,
        file_pairs: List[Tuple[str, str]],
        scaler: Optional[StandardScaler],
        fit_scaler: bool = False,
        num_workers: int = 8,
    ):
        self.file_pairs = file_pairs
        self.scaler = scaler
        self.fit_scaler = fit_scaler

        n_files = len(file_pairs)
        self._ego: List[Optional[np.ndarray]] = [None] * n_files
        self._labels: List[Optional[np.ndarray]] = [None] * n_files
        self._bev_rows: List[Optional[List[int]]] = [None] * n_files
        self._bev_mmaps = [None] * n_files
        self._img_mmaps = [None] * n_files

        self.index: List[Tuple[int, int, int]] = []

        print(f"Loading {n_files} file pairs with {num_workers} workers...")

        # Parallel file loading — collect all candidate windows without sampling
        # so that the idle-downsampling step is deterministic (sort then seed).
        args = [(i, pair[0]) for i, pair in enumerate(file_pairs)]
        _pending: List[Tuple[int, int, int, bool]] = []  # (file_idx, start, end, is_idle)

        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(_load_file_safe, a): a[0] for a in args}
            done = 0

            for fut in as_completed(futures):
                file_idx, result = fut.result()
                done += 1

                if isinstance(result, Exception):
                    print(f"  [WARN] file {file_idx} failed: {result}")
                    continue

                ego, labels, run_ids, bev_rows, idle_mask = result
                self._ego[file_idx] = ego
                self._labels[file_idx] = labels
                self._bev_rows[file_idx] = bev_rows

                n = len(run_ids)
                i = 0

                while i < n:
                    run = run_ids[i]
                    j = i

                    while j < n and run_ids[j] == run:
                        j += 1

                    if j - i >= cfg.SEQ_LEN:
                        for end in range(i + cfg.SEQ_LEN - 1, j, cfg.STRIDE):
                            start_idx = end - cfg.SEQ_LEN + 1
                            is_window_idle = all(
                                idle_mask[k] for k in range(start_idx, end + 1)
                            )
                            _pending.append((file_idx, start_idx, end, is_window_idle))

                    i = j

                if done % 50 == 0 or done == n_files:
                    print(f"  {done}/{n_files} files loaded", flush=True)

        # Sort for deterministic order before sampling (as_completed is non-deterministic).
        _pending.sort()

        # Idle downsampling is applied only on the training split (fit_scaler=True).
        # Val/test sets keep all windows for reproducible evaluation.
        _rng = random.Random(cfg.SEED)
        for file_idx, start_idx, end, is_idle in _pending:
            if is_idle and fit_scaler:
                if _rng.random() < 0.30:
                    self.index.append((file_idx, start_idx, end))
            else:
                self.index.append((file_idx, start_idx, end))

        # Scaler fit & transform
        valid_ego = [e for e in self._ego if e is not None]

        if fit_scaler and self.scaler is not None and valid_ego:
            all_ego = np.vstack(valid_ego)
            self.scaler.fit(all_ego)

            print(f"  Scaler fitted on {all_ego.shape[0]} rows")

        print(f"  Total sequences: {len(self.index)}")

        bev_mean = torch.tensor(cfg.BEV_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        bev_std = torch.tensor(cfg.BEV_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self._bev_mean = bev_mean  # (1, 3, 1, 1) — for broadcast
        self._bev_std = bev_std

        # ImageNet normalization for the RGB frames: (3, 1, 1) broadcasts over any
        # leading (n_cam, T) dims.
        self._cam_flip_perm: Optional[List[int]] = None
        if getattr(cfg, "USE_IMAGE", False):
            self._img_mean = torch.tensor(cfg.IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
            self._img_std = torch.tensor(cfg.IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
            # Left-right world mirror (BEV flip) also swaps the left/right cameras.
            # Precompute the camera-axis permutation that maps each "left" view to its
            # "right" counterpart (and vice versa); identity for views without a pair.
            cams = getattr(cfg, "IMG_CAMERAS", None)
            if cams:
                name_to_idx = {c: i for i, c in enumerate(cams)}
                perm = list(range(len(cams)))
                for c, i in name_to_idx.items():
                    if "left" in c:
                        mirror = c.replace("left", "right")
                        if mirror in name_to_idx:
                            perm[i] = name_to_idx[mirror]
                            perm[name_to_idx[mirror]] = i
                if perm != list(range(len(cams))):
                    self._cam_flip_perm = perm

    @property
    def ego_dim(self) -> int:
        for e in self._ego:
            if e is not None:
                return e.shape[1]

        return 0

    def _get_bev_mmap(self, file_idx: int) -> np.ndarray:
        if self._bev_mmaps[file_idx] is None:
            npy_path = self.file_pairs[file_idx][1]
            self._bev_mmaps[file_idx] = np.load(npy_path, mmap_mode="r")

        return self._bev_mmaps[file_idx]

    def _get_img_mmap(self, file_idx: int) -> np.ndarray:
        if self._img_mmaps[file_idx] is None:
            img_path = self.file_pairs[file_idx][2]
            self._img_mmaps[file_idx] = np.load(img_path, mmap_mode="r")

        return self._img_mmaps[file_idx]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        file_idx, start, end = self.index[idx]

        bev_rows = self._bev_rows[file_idx]
        bev_indices = [bev_rows[k] for k in range(start, end + 1)]

        bev_mmap = self._get_bev_mmap(file_idx)
        bev_seq = np.stack([bev_mmap[r] for r in bev_indices])  # (T, 3, H, W)
        bev_seq = torch.from_numpy(bev_seq.copy()).float()

        # Reproduce arch_best's channel-0 (max(0, z)) — paired with arch_best
        # BEV_MEAN[0]/BEV_STD[0] in config (the proven b2 steer recovery).
        bev_seq[:, 0].clamp_(min=0.0)

        ego_seq = torch.from_numpy(self._ego[file_idx][start : end + 1].copy()).float()
        label = torch.from_numpy(self._labels[file_idx][end].copy()).float()

        use_img = getattr(cfg, "USE_IMAGE", False)
        img = None
        if use_img:
            img_mmap = self._get_img_mmap(file_idx)
            # Decision-frame multi-camera image: the end-frame row (same as the BEV
            # window end). Each row of the multi-cam npy is (n_cam, 3, H, W).
            img_np = np.asarray(img_mmap[bev_rows[end]])  # (n_cam, 3, H, W) uint8
            img = torch.from_numpy(img_np.copy()).float() / 255.0  # (n_cam, 3, H, W) [0, 1]

        if self.fit_scaler:
            bev_seq, ego_seq, label, img = self._augment_sequence(
                bev_seq, ego_seq, label, img
            )

        bev_seq = (bev_seq - self._bev_mean) / self._bev_std
        if use_img:
            img = (img - self._img_mean) / self._img_std

        if self.scaler is not None and hasattr(self.scaler, "mean_"):
            ego_seq_np = ego_seq.numpy()
            ego_seq_scaled = self.scaler.transform(ego_seq_np)
            ego_seq = torch.from_numpy(ego_seq_scaled).float()

        if use_img:
            return bev_seq, ego_seq, img, label
        return bev_seq, ego_seq, label

    def _augment_sequence(self, bev_seq, ego_seq, label, img=None):
        """
        bev_seq: (T, 3, H, W)
        ego_seq: (T, ego_dim)
        label: (3,) -> [brake, throttle, steer]
        img: (n_cam, 3, H, W) or None — decision-frame multi-camera RGB
        """
        noise = torch.zeros_like(bev_seq)
        noise[:, :2] = torch.randn_like(bev_seq[:, :2]) * 0.005
        bev_seq = bev_seq + noise

        brightness = torch.rand(1).item() * 0.4 + 0.8  # [0.8, 1.2]
        bev_seq[:, :2] = bev_seq[:, :2] * brightness

        if torch.rand(1) > 0.5:
            bev_seq = torch.flip(bev_seq, dims=[-2])
            label[2] = -label[2]
            ego_seq[:, 2] = -ego_seq[:, 2]   # velocity_y
            ego_seq[:, 5] = -ego_seq[:, 5]   # rotation_yaw
            ego_seq[:, 13] = -ego_seq[:, 13]  # accel_y
            if img is not None:
                # Same left-right world mirror: flip each view's width axis...
                img = torch.flip(img, dims=[-1])
                # ...and swap the left/right cameras (camera axis = dim 0).
                if self._cam_flip_perm is not None:
                    img = img[self._cam_flip_perm]

        return bev_seq, ego_seq, label, img


# DataLoader builders


def _get_pairs(parquet_dir: str, bev_dir: str, split: str) -> List[Tuple[str, ...]]:
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, split, "*.parquet")))
    pairs = []
    use_img = getattr(cfg, "USE_IMAGE", False)

    for p in parquet_files:
        stem = os.path.splitext(os.path.basename(p))[0]
        npy = os.path.join(bev_dir, split, stem + ".npy")

        if not os.path.exists(npy):
            continue

        if use_img:
            img_npy = os.path.join(cfg.IMG_DIR, split, stem + ".npy")
            if not os.path.exists(img_npy):
                continue
            pairs.append((p, npy, img_npy))
        else:
            pairs.append((p, npy))

    return pairs


def build_dataloaders() -> Tuple[DataLoader, DataLoader]:
    """Merge train+validation pairs, shuffle, 90/10 file-level split."""
    all_pairs = _get_pairs(cfg.PARQUET_DIR, cfg.BEV_DIR, "train") + _get_pairs(
        cfg.PARQUET_DIR, cfg.BEV_DIR, "validation"
    )

    random.seed(cfg.SEED)
    random.shuffle(all_pairs)

    split_idx = int(len(all_pairs) * (1 - cfg.VAL_RATIO))
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    print(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")

    scaler = StandardScaler()

    # Use half of NUM_WORKERS for loading (leave headroom for DataLoader workers)
    load_workers = max(1, cfg.NUM_WORKERS // 2)

    train_ds = CarlaSequenceDataset(
        train_pairs, scaler, fit_scaler=True, num_workers=load_workers
    )
    val_ds = CarlaSequenceDataset(
        val_pairs, scaler, fit_scaler=False, num_workers=load_workers
    )

    nw = cfg.NUM_WORKERS
    loader_kwargs = dict(
        batch_size=cfg.BATCH_SIZE,
        num_workers=nw,
        pin_memory=cfg.PIN_MEMORY,
        **(
            {"prefetch_factor": cfg.PREFETCH, "persistent_workers": True}
            if nw > 0
            else {}
        ),
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader


def build_test_loader(scaler: StandardScaler) -> DataLoader:
    """Test loader. Pass the scaler fitted on train data."""
    test_pairs = _get_pairs(cfg.PARQUET_DIR, cfg.BEV_DIR, "test")

    print(f"Test pairs: {len(test_pairs)}")

    load_workers = max(1, cfg.NUM_WORKERS // 2)
    test_ds = CarlaSequenceDataset(
        test_pairs, scaler, fit_scaler=False, num_workers=load_workers
    )

    nw = cfg.NUM_WORKERS

    return DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=nw,
        pin_memory=cfg.PIN_MEMORY,
        **(
            {"prefetch_factor": cfg.PREFETCH, "persistent_workers": True}
            if nw > 0
            else {}
        ),
    )
