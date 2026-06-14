import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import glob


def evaluate_multioutput(y_true, y_pred, targets):
    """
    Evaluate multi-output regression model.

    Provides:
    - Per-target R2, MAE, RMSE
    - Macro (unweighted) averages
    - Joint R2 (all outputs together)

    Args:
        y_true (pd.DataFrame | np.ndarray): Ground truth.
        y_pred (np.ndarray): Model predictions.
        targets (list): Target column names.
    """

    print("\n── Per-Target Evaluation ──")

    r2_list, mae_list, rmse_list = [], [], []

    for i, target in enumerate(targets):
        yt = y_true[target]
        yp = y_pred[:, i]

        r2 = r2_score(yt, yp)
        mae = mean_absolute_error(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))

        r2_list.append(r2)
        mae_list.append(mae)
        rmse_list.append(rmse)

        print(f"{target:10s} | R2: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    print("\n── Macro Averages ──")
    print(f"R2   : {np.mean(r2_list):.4f}")
    print(f"MAE  : {np.mean(mae_list):.4f}")
    print(f"RMSE : {np.mean(rmse_list):.4f}")

    print("\n── Joint Evaluation ──")

    joint_r2 = r2_score(y_true, y_pred)
    joint_mae = mean_absolute_error(y_true, y_pred)
    joint_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"Joint R2   : {joint_r2:.4f}")
    print(f"Joint MAE  : {joint_mae:.4f}")
    print(f"Joint RMSE : {joint_rmse:.4f}")

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
    """
    Compute minimum LiDAR distance in the frontal region.

    Args:
        lidar (list | np.ndarray): LiDAR scan as (N, 4) [x, y, z, intensity].
        front_ratio (float): Half-angle of the frontal cone as a fraction of π
            (0.2 → ±36° half-angle).

    Returns:
        float: Minimum Euclidean distance to a frontal point, or NaN.
    """
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
    """
    Compute heuristic object risk score.

    Returns:
        float: Risk score in [0, 1].
    """
    if not front_present:
        return 0.0

    risk = 0.5

    if centered:
        risk += 0.3

    if not np.isnan(lidar_dist) and lidar_dist < dist_thresh:
        risk += 0.2

    return min(risk, 1.0)


def load_and_engineer(files, columns, imputer=None, fit_imputer=False):
    """
    Load parquet files and perform feature engineering.

    Args:
        files (list): File paths.
        columns (list): Columns to load.
        imputer (SimpleImputer): Shared imputer.
        fit_imputer (bool): Fit imputer or use existing.

    Returns:
        pd.DataFrame: Processed dataframe.
        SimpleImputer: Fitted imputer.
        list: Numeric feature columns.
    """
    dfs = [pd.read_parquet(f, columns=columns) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    box_feats = df["boxes"].apply(extract_box_features)
    df = pd.concat([df, pd.DataFrame(list(box_feats))], axis=1)

    df["min_lidar_distance_front"] = df["lidar"].apply(extract_lidar_front_distance)
    df["object_risk_score"] = df.apply(
        lambda r: compute_object_risk(
            r["front_object_present"],
            r["front_object_centered"],
            r["min_lidar_distance_front"],
        ),
        axis=1,
    )

    df.drop(columns=["boxes", "box_labels", "lidar"], inplace=True)

    df["is_stopped"] = (df["speed_kmh"] == 0).astype(int)
    df["accel_x"] = df.groupby("run_id")["velocity_x"].diff().fillna(0)
    df["accel_y"] = df.groupby("run_id")["velocity_y"].diff().fillna(0)
    df["accel_z"] = df.groupby("run_id")["velocity_z"].diff().fillna(0)
    df["delta_speed"] = df.groupby("run_id")["speed_kmh"].diff().fillna(0)
    df["accel_magnitude"] = np.sqrt(
        df["accel_x"] ** 2 + df["accel_y"] ** 2 + df["accel_z"] ** 2
    )
    df["is_moving"] = (df["speed_kmh"] > 0).astype(int)

    y_cols = ["brake", "throttle", "steer"]

    num_cols = [
        c
        for c in df.select_dtypes(include=["float64", "int64"]).columns
        if c not in y_cols and df[c].notna().any()
    ]

    if fit_imputer:
        imputer = SimpleImputer(strategy="mean")
        df[num_cols] = imputer.fit_transform(df[num_cols])
    else:
        # Preserve imputer's column order to avoid silent sklearn shape mismatch.
        shared = [c for c in imputer.feature_names_in_ if c in num_cols]
        df[shared] = imputer.transform(df[shared])

    df.drop(columns=["run_id"], inplace=True, errors="ignore")

    return df, imputer, num_cols


COLUMNS = [
    "run_id",
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
    "steer",
    "throttle",
    "brake",
    "lidar",
    "boxes",
    "box_labels",
]

Y_FEATURES = ["brake", "throttle", "steer"]

train_val_files = sorted(glob.glob("dataset/train/train-*.parquet")) + sorted(
    glob.glob("dataset/validation/validation-*.parquet")
)

split_idx = int(len(train_val_files) * (1 - 0.15))
train_val_files = train_val_files[:split_idx]

test_files = sorted(glob.glob("dataset/test/test-*.parquet"))

print(f"Train files: {len(train_val_files)}")
print(f"Test files : {len(test_files)}")

train_df, imputer, num_cols = load_and_engineer(
    train_val_files, COLUMNS, fit_imputer=True
)

test_df, _, _ = load_and_engineer(
    test_files, COLUMNS, imputer=imputer, fit_imputer=False
)

X_train = train_df.drop(columns=Y_FEATURES)
y_train = train_df[Y_FEATURES]

X_test = test_df.drop(columns=Y_FEATURES)
y_test = test_df[Y_FEATURES]

for col in X_train.columns:
    if col not in X_test.columns:
        X_test[col] = 0

X_test = X_test[X_train.columns]

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "model",
            MLPRegressor(
                hidden_layer_sizes=(512, 256, 128),
                activation="logistic",
                solver="adam",
                learning_rate_init=0.005,
                max_iter=5000,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=42,
            ),
        ),
    ]
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

evaluate_multioutput(y_test, y_pred, Y_FEATURES)
