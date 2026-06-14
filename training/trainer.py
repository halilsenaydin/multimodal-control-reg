import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
import config as cfg


class WeightedMSELoss(nn.Module):
    def __init__(
        self,
        throttle_nonzero_weight: float = 2.0,
        steer_high_weight: float = 4.0,
        steer_threshold: float = 0.1,
    ):
        super().__init__()
        self.throttle_w = throttle_nonzero_weight
        self.steer_w = steer_high_weight
        self.steer_thresh = steer_threshold

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        t = targets.clone()

        mse = (preds - t) ** 2  # [B, 3]
        weight = torch.ones_like(mse)

        throttle_mask = (t[:, 1] > 0.01).float()
        weight[:, 1] = throttle_mask * (self.throttle_w - 1) + 1

        steer_mask = (t[:, 2].abs() > self.steer_thresh).float()
        weight[:, 2] = steer_mask * (self.steer_w - 1) + 1

        return (mse * weight).mean()


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def _build_param_groups(model: nn.Module) -> list[dict]:
    """Build AdamW parameter groups with zero weight-decay for BN/bias params."""
    no_decay_types = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm
    )
    no_decay_ids: set[int] = set()
    for m in model.modules():
        if isinstance(m, no_decay_types):
            for p in m.parameters(recurse=False):
                no_decay_ids.add(id(p))
        for name, p in m.named_parameters(recurse=False):
            if "bias" in name:
                no_decay_ids.add(id(p))

    def split_params(params, lr: float) -> list[dict]:
        decay, no_decay = [], []
        seen: set[int] = set()
        for p in params:
            if id(p) in seen or not p.requires_grad:
                continue
            seen.add(id(p))
            if id(p) in no_decay_ids:
                no_decay.append(p)
            else:
                decay.append(p)
        groups: list[dict] = []
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": cfg.WEIGHT_DECAY})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
        return groups

    def split(module: nn.Module, lr: float) -> list[dict]:
        return split_params(module.parameters(), lr)

    groups: list[dict] = []
    groups.extend(split(model.backbone, cfg.LR * 0.1))
    for mod in [model.convlstm, model.ego_rnn, model.fusion_head, model.proj, model.pool]:
        groups.extend(split(mod, cfg.LR))
    # Image stream: pretrained backbone at 0.1x LR, everything else (proj, convlstm,
    # pool, ...) at full LR. "Collect the rest" so new img_encoder submodules can't be
    # silently dropped from the optimizer.
    if getattr(model, "use_image", False):
        enc = model.img_encoder
        groups.extend(split(enc.backbone, cfg.LR * 0.1))
        backbone_ids = {id(p) for p in enc.backbone.parameters()}
        rest = [p for p in enc.parameters() if id(p) not in backbone_ids]
        groups.extend(split_params(rest, cfg.LR))
    return groups


def _compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """R², MAE, RMSE per target + overall R²."""
    metrics: Dict[str, float] = {}
    for i, name in enumerate(cfg.TARGETS):
        metrics[f"r2_{name}"] = r2_score(labels[:, i], preds[:, i])
        metrics[f"mae_{name}"] = mean_absolute_error(labels[:, i], preds[:, i])
        metrics[f"rmse_{name}"] = float(
            np.sqrt(np.mean((preds[:, i] - labels[:, i]) ** 2))
        )
    metrics["r2_overall"] = r2_score(labels, preds, multioutput="uniform_average")
    return metrics


def _print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    tag = f"[{prefix}] " if prefix else ""
    header = f"{'':>12} {'R²':>8} {'MAE':>8} {'RMSE':>8}"
    print(f"\n  {tag}Per-target metrics")
    print(f"  {header}")
    print(f"  {'-' * 40}")
    for name in cfg.TARGETS:
        r2 = metrics[f"r2_{name}"]
        mae = metrics[f"mae_{name}"]
        rmse = metrics[f"rmse_{name}"]
        print(f"  {name:>12} {r2:>8.4f} {mae:>8.4f} {rmse:>8.4f}")
    print(f"  {'overall':>12} {metrics['r2_overall']:>8.4f}")


class Trainer:
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = next(model.parameters()).device
        self.accum = getattr(cfg, "GRAD_ACCUM", 1)
        self.grad_clip = getattr(cfg, "GRAD_CLIP", 1.0)

        self.optimizer = torch.optim.AdamW(
            _build_param_groups(model),
            lr=cfg.LR,
        )

        if cfg.LR_SCHEDULER == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.MAX_EPOCHS
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=cfg.PATIENCE // 2, factor=0.5
            )

        self.criterion = WeightedMSELoss()
        self.use_amp = cfg.AMP and torch.cuda.is_available()
        self.grad_scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        self.early_stopping = EarlyStopping(cfg.PATIENCE)
        Path(cfg.CKPT_DIR).mkdir(exist_ok=True)

        # Epoch geçmişi — overfitting analizi için
        self.history: Dict[str, list] = {
            "train_loss": [],
            "train_eval_loss": [],
            "val_loss": [],
            "val_r2_overall": [],
            **{f"val_r2_{n}": [] for n in cfg.TARGETS},
            **{f"val_mae_{n}": [] for n in cfg.TARGETS},
            **{f"val_rmse_{n}": [] for n in cfg.TARGETS},
        }

    def _to_device(self, *tensors):
        return [t.to(self.device, non_blocking=True) for t in tensors]

    def _split_batch(self, batch):
        """Unpack a dataloader batch → (bev, ego, img|None, labels) on device."""
        if getattr(cfg, "USE_IMAGE", False):
            bev_seq, ego_seq, img, labels = batch
            bev_seq, ego_seq, img, labels = self._to_device(bev_seq, ego_seq, img, labels)
            return bev_seq, ego_seq, img, labels
        bev_seq, ego_seq, labels = batch
        bev_seq, ego_seq, labels = self._to_device(bev_seq, ego_seq, labels)
        return bev_seq, ego_seq, None, labels

    def _optimizer_step(self):
        if self.use_amp:
            # unscale_ must come before clip so we clip the real gradient norms
            self.grad_scaler.unscale_(self.optimizer)
            
        torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip
            )
       
        if self.use_amp:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()
    
        self.optimizer.zero_grad()

    def train_epoch(self) -> float:
        self.model.train()
        total, count = 0.0, 0
        n_batches = len(self.train_loader)
        log_every = max(1, n_batches // 10)

        print(f"  train: {n_batches} batches (accum={self.accum})", flush=True)
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            bev_seq, ego_seq, img, labels = self._split_batch(batch)
            if step == 0:
                print(f"  first batch received, bev={tuple(bev_seq.shape)}", flush=True)

            if self.use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    preds = self.model(bev_seq, ego_seq, img)
                    loss = self.criterion(preds, labels) / self.accum
                self.grad_scaler.scale(loss).backward()
            else:
                preds = self.model(bev_seq, ego_seq, img)
                loss = self.criterion(preds, labels) / self.accum
                loss.backward()

            total += loss.item() * self.accum
            count += 1

            if (step + 1) % self.accum == 0:
                self._optimizer_step()

            if (step + 1) % log_every == 0:
                print(
                    f"  step {step+1}/{n_batches} | loss {total/count:.4f}", flush=True
                )

        if count % self.accum != 0:
            self._optimizer_step()

        return total / count

    def eval_train_loss(self) -> float:
        """Compute train loss in eval mode (no dropout/augmentation) for fair gap comparison."""
        self.model.eval()
        total, count = 0.0, 0

        with torch.no_grad():
            for batch in self.train_loader:
                bev_seq, ego_seq, img, labels = self._split_batch(batch)
                if self.use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        preds = self.model(bev_seq, ego_seq, img)
                        loss = self.criterion(preds, labels)
                else:
                    preds = self.model(bev_seq, ego_seq, img)
                    loss = self.criterion(preds, labels)
                total += loss.item()
                count += 1

        return total / count if count > 0 else 0.0

    def val_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Validation loss + R²/MAE/RMSE metrikleri."""
        self.model.eval()
        total, count = 0.0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                bev_seq, ego_seq, img, labels = self._split_batch(batch)
                if self.use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        preds = self.model(bev_seq, ego_seq, img)
                        loss = self.criterion(preds, labels)
                else:
                    preds = self.model(bev_seq, ego_seq, img)
                    loss = self.criterion(preds, labels)

                total += loss.item()
                count += 1
                all_preds.append(preds.cpu().float().numpy())
                all_labels.append(labels.cpu().float().numpy())

        preds_np = np.vstack(all_preds)
        labels_np = np.vstack(all_labels)
        metrics = _compute_metrics(preds_np, labels_np)

        return total / count, metrics

    def _save_overfit_plot(self) -> None:
        """Train vs val loss + per-target R² eğrilerini PNG olarak kaydeder."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [WARN] matplotlib yok, grafik kaydedilemedi.")
            return

        epochs = range(len(self.history["train_loss"]))
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        # ── Sol: Loss eğrisi ──────────────────────────────────────────────────
        ax = axes[0]
        ax.plot(epochs, self.history["train_loss"], label="train loss (w/ dropout)", linewidth=1.5, linestyle="--", alpha=0.6)
        ax.plot(epochs, self.history["train_eval_loss"], label="train eval loss", linewidth=1.5)
        ax.plot(epochs, self.history["val_loss"], label="val loss", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Train vs Val Loss")
        ax.legend()
        ax.grid(alpha=0.3)

        # Overfitting bölgesi: val_loss train_eval_loss'un %20'sinden fazla yüksekse gölgele
        train_arr = np.array(self.history["train_eval_loss"])
        val_arr = np.array(self.history["val_loss"])
        overfit_mask = val_arr > train_arr * 1.20
        if overfit_mask.any():
            for i, flag in enumerate(overfit_mask):
                if flag:
                    ax.axvspan(i - 0.5, i + 0.5, alpha=0.15, color="red")

        # ── Sağ: Val R² eğrileri ─────────────────────────────────────────────
        ax = axes[1]
        for name in cfg.TARGETS:
            ax.plot(epochs, self.history[f"val_r2_{name}"], label=name, linewidth=1.5)
        ax.plot(
            epochs,
            self.history["val_r2_overall"],
            label="overall",
            linewidth=2,
            linestyle="--",
            color="black",
        )
        ax.axhline(0.90, color="gray", linestyle=":", linewidth=1, label="R²=0.90")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("R²")
        ax.set_title("Val R² per Target")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]))

        plt.tight_layout()
        out_path = Path(cfg.CKPT_DIR) / "training_curves.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  📊 Grafik kaydedildi → {out_path}")

    def _overfitting_summary(self) -> None:
        """Eğitim bittikten sonra overfitting durumunu özetler."""
        if len(self.history["train_eval_loss"]) < 2:
            return

        train_arr = np.array(self.history["train_eval_loss"])
        val_arr = np.array(self.history["val_loss"])

        best_val_epoch = int(np.argmin(val_arr))
        final_epoch = len(val_arr) - 1
        gap = val_arr[final_epoch] - train_arr[final_epoch]
        gap_pct = gap / (train_arr[final_epoch] + 1e-9) * 100

        print("\n" + "=" * 50)
        print("  Overfitting Analizi")
        print("=" * 50)
        print(
            f"  En iyi val loss : epoch {best_val_epoch:>4}  ({val_arr[best_val_epoch]:.6f})"
        )
        print(
            f"  Son epoch       : epoch {final_epoch:>4}  (train {train_arr[final_epoch]:.6f} | val {val_arr[final_epoch]:.6f})"
        )
        print(f"  Loss gap (val - train): {gap:+.6f}  ({gap_pct:+.1f}%)")

        if gap_pct < 5:
            verdict = "✅ Overfitting yok — train/val loss dengeli"
        elif gap_pct < 15:
            verdict = "⚠️  Hafif overfitting — izlemeye devam et"
        else:
            verdict = "🔴 Belirgin overfitting — regularization artırılabilir"

        print(f"  Durum: {verdict}")

        # Early stopping tetiklendiyse not düş
        if best_val_epoch < final_epoch:
            print(
                f"  Early stopping: en iyi epoch'tan {final_epoch - best_val_epoch} epoch sonra durdu"
            )

        print("=" * 50)

    def fit(self, start_epoch: int = 0, scheduler_state: dict = None):
        if cfg.LR_SCHEDULER == "cosine" and start_epoch > 0:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.MAX_EPOCHS - start_epoch,
                last_epoch=start_epoch - 1,
            )

            if scheduler_state is not None:
                self.scheduler.load_state_dict(scheduler_state)

        best_val_loss = float("inf")
        best_epoch = 0

        for epoch in range(start_epoch, cfg.MAX_EPOCHS):
            print(f"\nEpoch {epoch:03d}", flush=True)

            train_loss = self.train_epoch()

            _every = cfg.EVAL_TRAIN_LOSS_EVERY
            _run_eval = _every > 0 and (epoch % _every == 0)
            train_eval_loss = self.eval_train_loss() if _run_eval else self.history["train_eval_loss"][-1] if self.history["train_eval_loss"] else train_loss

            val_loss, metrics = self.val_epoch()

            # Scheduler
            if cfg.LR_SCHEDULER == "cosine":
                self.scheduler.step()
            else:
                self.scheduler.step(val_loss)

            lr = self.optimizer.param_groups[0]["lr"]

            # Geçmişe kaydet
            self.history["train_loss"].append(train_loss)
            self.history["train_eval_loss"].append(train_eval_loss)
            self.history["val_loss"].append(val_loss)
            for key, val in metrics.items():
                hist_key = f"val_{key}"

                if hist_key in self.history:
                    self.history[hist_key].append(val)

            # Epoch özeti
            gap_pct = (val_loss - train_eval_loss) / (train_eval_loss + 1e-9) * 100
            eval_tag = "" if _run_eval else " (cached)"
            print(
                f"Epoch {epoch:03d} | train {train_loss:.6f} | train_eval {train_eval_loss:.6f}{eval_tag} | val {val_loss:.6f} | gap {gap_pct:+.1f}% | lr {lr:.2e}",
                flush=True,
            )
            _print_metrics(metrics, prefix="val")

            # Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                # Strip torch.compile prefix so checkpoints load on any platform.
                state_dict = {
                    k.removeprefix("_orig_mod."): v
                    for k, v in self.model.state_dict().items()
                }
                torch.save(
                    {
                        "model_state_dict": state_dict,
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_metrics": metrics,
                        "history": self.history,
                        "scaler": self.train_loader.dataset.scaler,
                        "ego_dim": self.train_loader.dataset.ego_dim,
                    },
                    f"{cfg.CKPT_DIR}/best_model.pt",
                )
                print(f"  ✓ checkpoint saved", flush=True)

            if self.early_stopping(val_loss):
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

        # Eğitim sonu
        self._overfitting_summary()
        self._save_overfit_plot()
        print(f"Done. Best val loss {best_val_loss:.6f} at epoch {best_epoch}")
