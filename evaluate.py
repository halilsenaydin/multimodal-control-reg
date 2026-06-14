import torch
import numpy as np
from typing import Any
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    average_precision_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt

import config as cfg
from models.backbone import get_backbone
from models.hybrid_model import HybridModel
from data.dataset import build_test_loader


BRAKE_ACTIVE_THRESH = 0.05
THROTTLE_ACTIVE_THRESH = 0.05
STEER_ACTIVE_THRESH = 0.10
BRAKE_DECISION_THRESH = 0.10


def plot_predictions(preds: np.ndarray, labels: np.ndarray, n: int = 500) -> None:
    """Plot predicted vs actual values for each target signal.

    Args:
        preds: Model predictions of shape (N, 3).
        labels: Ground truth values of shape (N, 3).
        n: Number of samples to plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (ax, name) in enumerate(zip(axes, cfg.TARGETS)):
        ax.scatter(labels[:n, i], preds[:n, i], alpha=0.3, s=10)
        lo = float(labels[:n, i].min())
        hi = float(labels[:n, i].max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig("predictions.png", dpi=150)
    print("predictions.png saved")


def load_model(ckpt_path: str, device: torch.device) -> tuple[HybridModel, Any, dict]:
    """Load a trained HybridModel from a checkpoint file.

    Args:
        ckpt_path: Path to the checkpoint file.
        device: Target torch device.

    Returns:
        Tuple of (model, scaler, history).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    scaler = ckpt["scaler"]
    ego_dim = ckpt["ego_dim"]
    history = ckpt.get("history", {})

    backbone = get_backbone(cfg.BACKBONE, cfg.PRETRAINED, cfg.FREEZE_BACKBONE)
    model = HybridModel(backbone, ego_dim).to(device)

    state_dict = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state_dict)

    print(f"Loaded epoch {ckpt['epoch']} | val_loss {ckpt['val_loss']:.6f}")
    model.eval()
    return model, scaler, history


def evaluate(model: HybridModel, loader: Any, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on a data loader and collect predictions and labels.

    Args:
        model: Trained HybridModel in eval mode.
        loader: DataLoader yielding batches.
        device: Target torch device.

    Returns:
        Tuple of (preds, labels), each of shape (N, 3).
    """
    all_preds = []
    all_labels = []
    use_img = getattr(cfg, "USE_IMAGE", False)

    with torch.no_grad():
        for batch in loader:
            if use_img:
                bev_seq, ego_seq, img, labels = batch
                img = img.to(device, non_blocking=True)
            else:
                bev_seq, ego_seq, labels = batch
                img = None
            bev_seq = bev_seq.to(device, non_blocking=True)
            ego_seq = ego_seq.to(device, non_blocking=True)

            preds = model(bev_seq, ego_seq, img).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    preds = np.vstack(all_preds)    # [N, 3]
    labels = np.vstack(all_labels)  # [N, 3]
    return preds, labels


def print_metrics(preds: np.ndarray, labels: np.ndarray) -> None:
    """Print standard regression metrics over the full test set.

    Args:
        preds: Model predictions of shape (N, 3).
        labels: Ground truth values of shape (N, 3).
    """
    print(f"\n{'':>12} {'R²':>8} {'MAE':>8} {'RMSE':>8}")
    print("-" * 40)
    for i, name in enumerate(cfg.TARGETS):
        r2 = r2_score(labels[:, i], preds[:, i])
        mae = mean_absolute_error(labels[:, i], preds[:, i])
        rmse = np.sqrt(np.mean((preds[:, i] - labels[:, i]) ** 2))
        print(f"{name:>12} {r2:>8.4f} {mae:>8.4f} {rmse:>8.4f}")
    r2_all = r2_score(labels, preds, multioutput="uniform_average")
    print(f"\n{'overall':>12} {r2_all:>8.4f}")


def print_class_balance(labels: np.ndarray) -> None:
    """Report the fraction of samples where each signal is active.

    A high brake R² on a rarely-active brake signal indicates the model exploits
    class imbalance by predicting near-zero most of the time.

    Args:
        labels: Ground truth values of shape (N, 3).
    """
    n = len(labels)
    brake_active = (labels[:, 0] > BRAKE_ACTIVE_THRESH).mean()
    throttle_active = (labels[:, 1] > THROTTLE_ACTIVE_THRESH).mean()
    steer_active = (np.abs(labels[:, 2]) > STEER_ACTIVE_THRESH).mean()
    print(f"\nClass balance (N={n})")
    print("-" * 40)
    print(f"  brake > {BRAKE_ACTIVE_THRESH:<4}    active: {brake_active*100:5.1f}%  (passive: {(1-brake_active)*100:4.1f}%)")
    print(f"  throttle > {THROTTLE_ACTIVE_THRESH:<2} active: {throttle_active*100:5.1f}%")
    print(f"  |steer| > {STEER_ACTIVE_THRESH:<3}  active: {steer_active*100:5.1f}%")


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")


def print_active_subset_metrics(preds: np.ndarray, labels: np.ndarray) -> None:
    """Print R²/MAE/RMSE restricted to samples where each signal is active.

    Filters out the near-zero majority to show how well the model predicts
    actual braking/throttle/steering magnitude.

    Args:
        preds: Model predictions of shape (N, 3).
        labels: Ground truth values of shape (N, 3).
    """
    specs = [
        ("brake (>thr)", 0, labels[:, 0] > BRAKE_ACTIVE_THRESH),
        ("throttle(>thr)", 1, labels[:, 1] > THROTTLE_ACTIVE_THRESH),
        ("steer (|.|>thr)", 2, np.abs(labels[:, 2]) > STEER_ACTIVE_THRESH),
    ]
    print(f"\nMetrics on ACTIVE samples only")
    print(f"{'':>16} {'n':>7} {'R²':>8} {'MAE':>8} {'RMSE':>8}")
    print("-" * 50)
    for name, i, mask in specs:
        yt, yp = labels[mask, i], preds[mask, i]
        if len(yt) == 0:
            print(f"{name:>16} {0:>7} {'-':>8} {'-':>8} {'-':>8}")
            continue
        r2 = _safe_r2(yt, yp)
        mae = mean_absolute_error(yt, yp)
        rmse = np.sqrt(np.mean((yp - yt) ** 2))
        print(f"{name:>16} {len(yt):>7} {r2:>8.4f} {mae:>8.4f} {rmse:>8.4f}")


def print_brake_detection_metrics(preds: np.ndarray, labels: np.ndarray) -> None:
    """Evaluate brake as a binary braking-event detection problem.

    Reports precision, recall, F1, average precision, and a confusion matrix.
    Also sweeps decision thresholds to aid operating-point selection.

    Recall (missed braking events) is typically more safety-critical than precision.

    Args:
        preds: Model predictions of shape (N, 3).
        labels: Ground truth values of shape (N, 3).
    """
    y_true = (labels[:, 0] > BRAKE_ACTIVE_THRESH).astype(int)
    score = preds[:, 0]
    y_pred = (score > BRAKE_DECISION_THRESH).astype(int)

    pos = int(y_true.sum())
    print(f"\nBrake event detection  (active_thresh={BRAKE_ACTIVE_THRESH}, decision_thresh={BRAKE_DECISION_THRESH})")
    print("-" * 56)
    print(f"  True braking samples: {pos}/{len(y_true)} ({pos/len(y_true)*100:.1f}%)")

    if pos == 0:
        print("  [!] No braking samples in test set — detection metrics undefined.")
        return

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        ap = average_precision_score(y_true, score)
    except ValueError:
        ap = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(f"  Precision: {precision:.4f}   Recall: {recall:.4f}   F1: {f1:.4f}   AP: {ap:.4f}")
    print(f"  Confusion matrix:")
    print(f"                 pred=no_brake   pred=brake")
    print(f"     true=no     {tn:>12}    {fp:>10}   (FP = false alarm)")
    print(f"     true=brake  {fn:>12}    {tp:>10}   (FN = missed brake)")

    print(f"\n  Threshold sweep (precision / recall / F1):")
    for thr in [0.05, 0.10, 0.20, 0.30, 0.50]:
        yp = (score > thr).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, yp, average="binary", zero_division=0
        )
        print(f"     thr={thr:<4}  P={p:.3f}  R={r:.3f}  F1={f:.3f}")


def plot_history(history: dict) -> None:
    """Plot training and validation loss curves from a checkpoint history dict.

    Args:
        history: Dict with optional keys 'train_loss' and 'val_loss'.
    """
    train_loss = history.get("train_loss")
    val_loss = history.get("val_loss")
    plt.figure(figsize=(10, 5))
    if train_loss is not None:
        plt.plot(train_loss, label="Train Loss")
    if val_loss is not None:
        plt.plot(val_loss, label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler, history = load_model(f"{cfg.CKPT_DIR}/best_model.pt", device)

    print("\n=== Test ===")
    test_loader = build_test_loader(scaler)
    preds, labels = evaluate(model, test_loader, device)

    print_metrics(preds, labels)
    print_class_balance(labels)
    print_active_subset_metrics(preds, labels)
    print_brake_detection_metrics(preds, labels)
    plot_predictions(preds, labels)
