from abc import ABC, abstractmethod
import torch.nn as nn
import timm


class BaseBackbone(nn.Module, ABC):
    """Input: [B, 3, H, W] → Output: [B, C, h, w]  (spatial feature maps)."""

    @abstractmethod
    def forward(self, x): ...

    @property
    @abstractmethod
    def out_channels(self) -> int: ...

    @property
    @abstractmethod
    def spatial_size(self) -> int: ...

    @abstractmethod
    def freeze(self) -> None: ...

    @abstractmethod
    def unfreeze(self) -> None: ...


class EfficientNetB0Backbone(BaseBackbone):
    """EfficientNet-B0 — returns spatial feature maps [B, 1280, 7, 7] for 224×224 input."""

    def __init__(self, pretrained: bool, freeze: bool):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        self._out_channels = 1280
        self._spatial_size = 7

        if freeze:
            self.freeze()

    def forward(self, x):
        return self.backbone(x)  # [B, 1280, 7, 7]

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def spatial_size(self) -> int:
        return self._spatial_size

    def freeze(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True


class EfficientNetB0FeaturesOnlyBackbone(BaseBackbone):
    """EfficientNet-B0 features only true — [B, 112, 14, 14] for 224×224 input."""

    def __init__(self, pretrained: bool, freeze: bool):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
            features_only=True,
            out_indices=(3,),
        )
        self._out_channels = 112
        self._spatial_size = 14

        if freeze:
            self.freeze()

    def forward(self, x):
        return self.backbone(x)[0]  # liste → tensor

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def spatial_size(self) -> int:
        return self._spatial_size

    def freeze(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True


class EfficientNetB1Backbone(BaseBackbone):
    """EfficientNet-B1 — returns spatial feature maps [B, 1280, 7, 7] for 224×224 input.

    B1's head width equals B0's (1280); it is deeper/scaled, not wider at the head.
    """

    def __init__(self, pretrained: bool, freeze: bool):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b1",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",  # keep spatial dims [B, C, h, w]
        )
        self._out_channels = 1280
        self._spatial_size = 7

        if freeze:
            self.freeze()

    def forward(self, x):
        return self.backbone(x)  # [B, 1280, 7, 7]

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def spatial_size(self) -> int:
        return self._spatial_size

    def freeze(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True


class EfficientNetB3Backbone(BaseBackbone):
    """EfficientNet-B3 — returns spatial feature maps [B, 1536, 7, 7] for 224×224 input."""

    def __init__(self, pretrained: bool, freeze: bool):
        super().__init__()
        # global_pool="" keeps spatial dims; num_classes=0 removes classifier
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",  # ← must be "" to get [B, C, h, w] not [B, C]
        )
        self._out_channels = 1536
        self._spatial_size = 7

        if freeze:
            self.freeze()

    def forward(self, x):
        return self.backbone(x)  # [B, 1536, 7, 7]

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def spatial_size(self) -> int:
        return self._spatial_size

    def freeze(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True


def get_backbone(name: str, pretrained: bool, freeze: bool) -> BaseBackbone:
    registry = {
        "efficientnet_b0": EfficientNetB0Backbone,
        "efficientnet_b0_features_only": EfficientNetB0FeaturesOnlyBackbone,
        "efficientnet_b1": EfficientNetB1Backbone,
        "efficientnet_b3": EfficientNetB3Backbone,
    }
    if name not in registry:
        raise ValueError(f"Unknown backbone: {name}. Available: {list(registry)}")
    return registry[name](pretrained=pretrained, freeze=freeze)
