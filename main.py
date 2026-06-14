import platform
import torch
import config as cfg
from utils.seed import set_seed
from data.dataset import build_dataloaders
from models.backbone import get_backbone
from models.hybrid_model import HybridModel
from training.trainer import Trainer
import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    set_seed(cfg.SEED)

    train_loader, val_loader = build_dataloaders()
    ego_dim = train_loader.dataset.ego_dim
    print(f"ego_dim: {ego_dim}")

    backbone = get_backbone(cfg.BACKBONE, cfg.PRETRAINED, cfg.FREEZE_BACKBONE)
    model = HybridModel(backbone, ego_dim).to(device)

    if hasattr(torch, "compile") and platform.system() != "Windows":
        model = torch.compile(model)
        print("torch.compile applied")

    print("Training")

    Trainer(model, train_loader, val_loader).fit()


if __name__ == "__main__":
    main()
