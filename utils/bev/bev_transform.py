import ast
import numpy as np
import torch
from typing import Any, Optional


class LiDARToBEV:
    """Convert raw CARLA LiDAR point clouds [x, y, z, intensity] to a
    3-channel Bird's-Eye-View tensor of shape (3, target_size, target_size).

    Channels:
        0 - Height (Z), max pooling over overlapping points
        1 - Intensity, max pooling over overlapping points
        2 - Occupancy binary mask
    """

    def __init__(self, target_size: int = 224, max_range: float = 80.0) -> None:
        """Initialize the BEV transform.

        Args:
            target_size: Spatial dimension of the output BEV image (height = width).
            max_range: Half of the LiDAR range in meters; BEV covers [-max_range, max_range].
        """
        self.target_size: int = target_size
        self.max_range: float = max_range
        self.x_range: np.ndarray = np.linspace(-max_range, max_range, num=target_size)
        self.y_range: np.ndarray = np.linspace(-max_range, max_range, num=target_size)

    def _parse_points(self, lidar_data: Any) -> Optional[np.ndarray]:
        """Parse LiDAR data into a (N, 4) float32 numpy array.

        Supported input formats:
            - np.ndarray with shape (N, 4)
            - np.ndarray with shape (N,) and dtype=object (CARLA parquet)
            - list of lists / list of arrays
            - flat 1-D list/array whose length is a multiple of 4

        Args:
            lidar_data: Raw LiDAR data in any supported format.

        Returns:
            A (N, 4) float32 array, or None if parsing fails.
        """
        if lidar_data is None:
            return None

        if isinstance(lidar_data, str):
            try:
                lidar_data = ast.literal_eval(lidar_data)
            except Exception:
                return None

        if isinstance(lidar_data, dict) and "points" in lidar_data:
            lidar_data = lidar_data["points"]

        arr: np.ndarray = np.asarray(lidar_data)

        # shape=(N,), dtype=object, each element is a shape=(4,) float32 array
        if arr.ndim == 1 and arr.dtype == object:
            try:
                arr = np.vstack(arr).astype(np.float32)  # -> (N, 4)
            except Exception:
                return None

        # already (N, 4)
        if arr.ndim == 2 and arr.shape[1] == 4:
            return arr.astype(np.float32)

        # flat 1-D (multiple of 4)
        if arr.ndim == 1:
            arr = arr.astype(np.float32).flatten()
            remainder: int = arr.size % 4
            if remainder:
                arr = arr[:-remainder]
            if arr.size == 0:
                return None
            return arr.reshape(-1, 4)

        return None

    def __call__(self, lidar_data: Any) -> torch.Tensor:
        """Transform a single LiDAR frame into a BEV tensor.

        Args:
            lidar_data: Raw point cloud in any format accepted by _parse_points.

        Returns:
            BEV tensor of shape (3, target_size, target_size).
        """
        empty: torch.Tensor = torch.zeros(3, self.target_size, self.target_size)

        points: Optional[np.ndarray] = self._parse_points(lidar_data)
        if points is None or len(points) == 0:
            return empty

        # FOV filter
        mask: np.ndarray = (
            (points[:, 0] > -self.max_range)
            & (points[:, 0] < self.max_range)
            & (points[:, 1] > -self.max_range)
            & (points[:, 1] < self.max_range)
        )
        pts: np.ndarray = points[mask]
        if len(pts) == 0:
            return empty

        # Pixel indices
        x_min: float = self.x_range[0]
        x_max: float = self.x_range[-1]
        y_min: float = self.y_range[0]
        y_max: float = self.y_range[-1]

        col_idx: np.ndarray = (
            (pts[:, 0] - x_min) / (x_max - x_min) * (self.target_size - 1)
        ).astype(int)
        row_idx: np.ndarray = (
            (y_max - pts[:, 1]) / (y_max - y_min) * (self.target_size - 1)
        ).astype(int)

        valid: np.ndarray = (
            (col_idx >= 0)
            & (col_idx < self.target_size)
            & (row_idx >= 0)
            & (row_idx < self.target_size)
        )
        col_idx = col_idx[valid]
        row_idx = row_idx[valid]
        pts = pts[valid]

        # 3 channels: Z (height), Intensity, Occupancy
        # Initialize z_ch with -inf so that cells with negative-z points are
        # distinguishable from truly empty cells (CARLA road points have z < 0).
        z_ch: np.ndarray = np.full(
            (self.target_size, self.target_size), fill_value=-np.inf, dtype=np.float32
        )
        i_ch: np.ndarray = np.zeros(
            (self.target_size, self.target_size), dtype=np.float32
        )
        occ: np.ndarray = np.zeros(
            (self.target_size, self.target_size), dtype=np.float32
        )

        # Keep the highest value for overlapping points
        np.maximum.at(z_ch, (row_idx, col_idx), pts[:, 2])
        np.maximum.at(i_ch, (row_idx, col_idx), pts[:, 3])
        occ[row_idx, col_idx] = 1.0
        # Empty cells (never hit by a point) remain -inf → reset to 0
        z_ch[~np.isfinite(z_ch)] = 0.0

        return torch.from_numpy(np.stack([z_ch, i_ch, occ], axis=0))  # [3, H, W]
