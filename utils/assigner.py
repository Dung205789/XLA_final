"""IoU-based anchor-to-GT assignment."""

import torch
from .box_ops import box_iou


class IoUAssigner:
    """
    Assigns ground-truth boxes to anchors based on IoU.
    Each GT is guaranteed at least one positive anchor (the best match).
    """

    def __init__(self, pos_iou: float = 0.5, neg_iou: float = 0.4):
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou

    def assign(
        self,
        anchors: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ):
        """
        Returns:
          obj_targets  [A]    1=positive, 0=negative, -1=ignore
          cls_targets  [A]    class index for positives, -1 otherwise
          box_targets  [A,4]  GT box for positives, zeros otherwise
        """
        A = anchors.shape[0]
        device = anchors.device

        obj_t = torch.zeros(A, dtype=torch.long, device=device)
        cls_t = torch.full((A,), -1, dtype=torch.long, device=device)
        box_t = torch.zeros(A, 4, dtype=torch.float32, device=device)

        if gt_boxes.shape[0] == 0:
            return obj_t, cls_t, box_t

        iou = box_iou(anchors, gt_boxes)          # [A, G]
        max_iou, best_gt = iou.max(dim=1)          # [A]

        # Negative / ignore / positive masks by IoU
        obj_t[max_iou < self.neg_iou] = 0
        obj_t[(max_iou >= self.neg_iou) & (max_iou < self.pos_iou)] = -1
        obj_t[max_iou >= self.pos_iou] = 1

        # Guarantee at least one positive per GT (best anchor regardless of IoU)
        best_anchor_per_gt = iou.max(dim=0).indices  # [G]
        obj_t[best_anchor_per_gt] = 1

        # Build matched_gt: for each anchor the GT index it's assigned to
        matched_gt = best_gt.clone()
        for g, a in enumerate(best_anchor_per_gt):
            matched_gt[a] = g

        pos = obj_t == 1
        if pos.any():
            cls_t[pos] = gt_labels[matched_gt[pos]]
            box_t[pos] = gt_boxes[matched_gt[pos]]

        return obj_t, cls_t, box_t
