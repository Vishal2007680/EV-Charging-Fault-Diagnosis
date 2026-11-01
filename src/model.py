import os
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models

class EVFaultDualHeadModel(nn.Module):
    def __init__(self, pretrained: bool = True, num_conditions: int = 2, num_components: int = 4):
        super().__init__()

        if pretrained:
            try:
                # Use standard torchvision recommended weights enum
                weights = models.MobileNet_V3_Small_Weights.DEFAULT
                base_mobilenet = models.mobilenet_v3_small(weights=weights)
                print(f"[Model] Successfully initialized MobileNetV3-Small with official torchvision weights: {weights}")
            except Exception as e:
                # Fallback check for local cached checkpoint file before failing
                cached_weight_path = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "mobilenet_v3_small-047dcff4.pth"
                if cached_weight_path.exists():
                    base_mobilenet = models.mobilenet_v3_small(weights=None)
                    state_dict = torch.load(cached_weight_path, map_location="cpu", weights_only=True)
                    base_mobilenet.load_state_dict(state_dict)
                    print(f"[Model] Pretrained weights loaded from local checkpoint file: {cached_weight_path.name}")
                else:
                    raise RuntimeError(
                        f"Failed to obtain official torchvision MobileNetV3-Small pretrained weights: {e}. "
                        "Pretrained weights are required when pretrained=True to avoid training on uninitialized random features."
                    ) from e
        else:
            base_mobilenet = models.mobilenet_v3_small(weights=None)

        self.features = base_mobilenet.features
        self.avgpool = base_mobilenet.avgpool
        in_features = 576

        # Shared representation projection
        self.shared_fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.25)
        )

        # Head 1: Fault Condition (Healthy = 0, Broken = 1)
        self.condition_head = nn.Linear(256, num_conditions)

        # Head 2: Component Type (Charger = 0, Cord = 1, Plug = 2, Screen = 3)
        self.component_head = nn.Linear(256, num_components)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feat = self.shared_fc(x)

        condition_logits = self.condition_head(feat)
        component_logits = self.component_head(feat)

        return condition_logits, component_logits

    def freeze_backbone(self, freeze: bool = True):
        for param in self.features.parameters():
            param.requires_grad = not freeze
        print(f"[Model] Backbone feature extractor frozen: {freeze}")
