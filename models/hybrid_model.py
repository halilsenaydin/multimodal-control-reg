import torch
import torch.nn as nn
from torch import Tensor
from models.backbone import BaseBackbone, get_backbone
from models.convlstm import ConvLSTM
import config as cfg


class SpatialAttentionPool(nn.Module):
    """Softmax spatial attention pooling."""

    def __init__(self, channels: int):
        super().__init__()

        self.attn_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:  # x: [B, C, H, W]
        B, _, H, W = x.shape
        logits = self.attn_conv(x)  # [B, 1, H, W]
        attn = torch.softmax(logits.flatten(2), dim=2).view(B, 1, H, W)
        
        return (x * attn).sum(dim=[2, 3])  # [B, C]


class ImageEncoder(nn.Module):
    """Single decision-frame encoder over all cameras (no temporal ConvLSTM).

    e1/e2 showed the temporal image ConvLSTM fit validation but transferred worse to
    test; the single decision frame was d1's winning inductive bias (the label is at
    the final frame; temporal context already comes from BEV+ego). Each camera's
    decision frame goes through a SHARED backbone -> avg-pool -> Linear (weight-tied
    across views — param-efficient on small data); the per-view vectors are
    concatenated. = d1's image recipe applied to all cameras. Separate backbone
    instance from the BEV stream (natural-image domain, good ImageNet match).
    """

    def __init__(self, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.backbone = get_backbone(
            cfg.IMG_BACKBONE, pretrained=cfg.PRETRAINED, freeze=False
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(self.backbone.out_channels, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, img: Tensor) -> Tensor:  # img: [B, n_cam, 3, H, W]
        B, n_cam, C, H, W = img.shape
        flat = img.reshape(B * n_cam, C, H, W)  # [B*n_cam, 3, H, W]
        feat = self.backbone(flat)  # [B*n_cam, C_b, h, w]
        feat = self.pool(feat).flatten(1)  # [B*n_cam, C_b]
        feat = self.proj(feat)  # [B*n_cam, out_dim]
        return feat.reshape(B, n_cam * feat.shape[-1])  # [B, n_cam*out_dim]


class HybridModel(nn.Module):
    """Hybrid CNN–ConvLSTM model for autonomous driving control prediction."""

    def __init__(self, backbone: BaseBackbone, ego_dim: int):
        super().__init__()
        # NOTE: backbone was pretrained on ImageNet RGB; BEV channels are
        # height/intensity/occupancy (domain mismatch). Transfer learning still
        # converges in practice, but first-layer filters are not semantically aligned.
        self.backbone = backbone

        self.proj = nn.Sequential(
            nn.Conv2d(
                backbone.out_channels, cfg.PROJ_CHANNELS, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(cfg.PROJ_CHANNELS),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.10),
        )

        self.convlstm = ConvLSTM(
            in_channels=cfg.PROJ_CHANNELS,
            hidden_channels=cfg.CONVLSTM_HIDDEN,
            kernel_size=cfg.CONVLSTM_KERNEL,
            num_layers=cfg.CONVLSTM_LAYERS,
        )

        self.pool = SpatialAttentionPool(cfg.CONVLSTM_HIDDEN)
        
        self.ego_rnn = nn.GRU(ego_dim, cfg.EGO_HIDDEN, batch_first=True)
        
        self.bev_drop = nn.Dropout(p=0.20)
        self.ego_drop = nn.Dropout(p=0.20)

        # Optional temporal multi-camera RGB stream (concatenated per-view vectors).
        self.use_image = getattr(cfg, "USE_IMAGE", False)
        self.n_cam = len(getattr(cfg, "IMG_CAMERAS", [])) or 1
        if self.use_image:
            self.img_encoder = ImageEncoder(cfg.IMG_HIDDEN)

        fusion_in = cfg.CONVLSTM_HIDDEN + cfg.EGO_HIDDEN + (
            cfg.IMG_HIDDEN * self.n_cam if self.use_image else 0
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 3),
        )

    def forward(self, bev_seq: Tensor, ego_seq: Tensor, img: Tensor = None) -> Tensor:
        """
        bev_seq : [B, T, 3, H, W]
        ego_seq : [B, T, ego_dim]
        img     : [B, n_cam, 3, H, W] decision-frame cameras (or None when USE_IMAGE is off)
        returns : [B, 3]
        """
        B, T, C, H, W = bev_seq.shape

        # Process all T frames through backbone at once: [B*T, 3, H, W]
        bev_flat = bev_seq.view(B * T, C, H, W)
        feat_flat = self.backbone(bev_flat)  # [B*T, out_channels, h, w]

        feat_flat = self.proj(feat_flat)
        _, Cf, Hf, Wf = feat_flat.shape
        feat = feat_flat.view(B, T, Cf, Hf, Wf)  # [B, T, out_channels, h, w]

        # ConvLSTM temporal processing
        h = self.convlstm(feat)  # [B, CONVLSTM_HIDDEN, h, w]
        bev_out = self.bev_drop(self.pool(h))

        # Ego branch
        _, h_n = self.ego_rnn(ego_seq)
        ego_out = self.ego_drop(h_n.squeeze(0))

        # Fusion (+ optional image stream)
        feats = [bev_out, ego_out]  # [B, 256], [B, 128]
        if self.use_image:
            feats.append(self.img_encoder(img))  # [B, IMG_HIDDEN]
        raw = self.fusion_head(torch.cat(feats, dim=1))  # [B, 3]

        brake = torch.sigmoid(raw[:, 0:1])  # [0, 1]
        throttle = torch.sigmoid(raw[:, 1:2])  # [0, 1]
        steer = torch.tanh(raw[:, 2:3])  # [-1, 1]

        return torch.cat([brake, throttle, steer], dim=1)  # [B, 3]
