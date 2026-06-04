"""ResNet18 + FPN + anchor-based detection head."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


class FPN(nn.Module):
    def __init__(self, in_channels: list, out_channels: int = 256):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])
        self.smooth = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels])

    def forward(self, features: list) -> list:
        lats = [conv(f) for conv, f in zip(self.lateral, features)]
        # Top-down merge
        for i in range(len(lats) - 2, -1, -1):
            lats[i] = lats[i] + F.interpolate(lats[i + 1], size=lats[i].shape[-2:], mode="nearest")
        return [smooth(lat) for smooth, lat in zip(self.smooth, lats)]


class DetectionHead(nn.Module):
    """Shared-weight head predicting objectness, class logits, and box deltas."""

    def __init__(self, in_channels: int = 256, num_classes: int = 5, num_anchors: int = 3):
        super().__init__()
        self.A = num_anchors
        self.C = num_classes

        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.obj_head = nn.Conv2d(in_channels, num_anchors, 1)
        self.cls_head = nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        self.box_head = nn.Conv2d(in_channels, num_anchors * 4, 1)
        self._init_weights()

    def _init_weights(self):
        prior = 0.01
        bias = -math.log((1 - prior) / prior)
        nn.init.constant_(self.obj_head.bias, bias)
        for m in [self.obj_head, self.cls_head, self.box_head]:
            nn.init.normal_(m.weight, std=0.01)
        nn.init.zeros_(self.cls_head.bias)
        nn.init.zeros_(self.box_head.bias)

    def _reshape(self, x: torch.Tensor, D: int) -> torch.Tensor:
        B, _, H, W = x.shape
        A = self.A
        # [B, A*D, H, W] → [B, H, W, A, D] → [B, H*W*A, D]
        return x.permute(0, 2, 3, 1).contiguous().reshape(B, H * W, A, D).reshape(B, H * W * A, D)

    def forward(self, features: list):
        obj_list, cls_list, box_list = [], [], []
        for feat in features:
            x = self.shared(feat)
            obj_list.append(self._reshape(self.obj_head(x), 1))
            cls_list.append(self._reshape(self.cls_head(x), self.C))
            box_list.append(self._reshape(self.box_head(x), 4))
        return (
            torch.cat(obj_list, dim=1),
            torch.cat(cls_list, dim=1),
            torch.cat(box_list, dim=1),
        )


class ResNet18FPNDetector(nn.Module):
    """ResNet34-based FPN detector (name kept for checkpoint compatibility)."""

    def __init__(self, num_classes: int = 5, num_anchors: int = 3, fpn_ch: int = 256, pretrained: bool = True):
        super().__init__()
        bb = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1   # stride 4,  64 ch
        self.layer2 = bb.layer2   # stride 8,  128 ch  → C3
        self.layer3 = bb.layer3   # stride 16, 256 ch  → C4
        self.layer4 = bb.layer4   # stride 32, 512 ch  → C5

        self.fpn = FPN([128, 256, 512], out_channels=fpn_ch)
        self.head = DetectionHead(fpn_ch, num_classes, num_anchors)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p3, p4, p5 = self.fpn([c3, c4, c5])
        return self.head([p3, p4, p5])

    def freeze_backbone(self):
        for p in list(self.stem.parameters()) + list(self.layer1.parameters()) + \
                  list(self.layer2.parameters()) + list(self.layer3.parameters()) + \
                  list(self.layer4.parameters()):
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
