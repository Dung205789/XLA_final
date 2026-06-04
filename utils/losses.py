"""Detection losses: focal BCE (objectness), CE with class weights (class), CIoU (box)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .box_ops import ciou_loss, decode_deltas, encode_deltas


class FocalBCE(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


class DetectionLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        lambda_obj: float = 1.0,
        lambda_cls: float = 1.0,
        lambda_box: float = 2.0,
        cls_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.lambda_box = lambda_box
        self.focal_bce = FocalBCE(alpha=0.25, gamma=2.0)
        # Optional per-class weights to counter class imbalance
        self.register_buffer("cls_weights", cls_weights if cls_weights is not None
                             else torch.ones(num_classes))

    def forward(self, predictions, batch_targets: list, anchors: torch.Tensor):
        """
        predictions: (obj_logits [B,A,1], cls_logits [B,A,C], box_preds [B,A,4])
        batch_targets: list of dicts with obj_targets [A], cls_targets [A], box_targets [A,4]
        anchors: [A, 4]
        Returns: (total_loss, info_dict)
        """
        obj_logits, cls_logits, box_preds = predictions
        B = obj_logits.shape[0]
        device = obj_logits.device

        l_obj = l_cls = l_box = torch.tensor(0.0, device=device)
        num_pos = 0

        for i in range(B):
            obj_t = batch_targets[i]["obj_targets"].to(device)
            cls_t = batch_targets[i]["cls_targets"].to(device)
            box_t = batch_targets[i]["box_targets"].to(device)

            # ATSS produces no ignore zone: all anchors are either 0 or 1
            # IoUAssigner might produce -1 (ignore); exclude those
            valid = obj_t >= 0
            pos = obj_t == 1
            num_pos += int(pos.sum())

            if valid.any():
                l_obj = l_obj + self.focal_bce(
                    obj_logits[i, valid, 0], obj_t[valid].float()
                )

            if pos.any():
                # Classification with class-frequency weights
                l_cls = l_cls + F.cross_entropy(
                    cls_logits[i, pos],
                    cls_t[pos],
                    weight=self.cls_weights.to(device),
                )

                # Box regression: CIoU loss
                anc_pos = anchors[pos]
                pred_boxes = decode_deltas(anc_pos, box_preds[i, pos])
                l_box = l_box + ciou_loss(pred_boxes, box_t[pos]).mean()

        l_obj = l_obj / B
        l_cls = l_cls / B
        l_box = l_box / B

        total = self.lambda_obj * l_obj + self.lambda_cls * l_cls + self.lambda_box * l_box
        return total, {
            "obj": l_obj.item(),
            "cls": l_cls.item(),
            "box": l_box.item(),
            "num_pos": num_pos / B,
        }
