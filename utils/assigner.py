"""ATSS anchor-to-GT assignment (Adaptive Training Sample Selection)."""

import torch
from .box_ops import box_iou


class ATSSAssigner:
    """
    For each GT box, selects top_k anchor candidates per FPN level by L2 center
    distance, then uses IoU mean+std as a dynamic threshold. This is far more
    effective than a fixed IoU threshold for small objects (car, chair).

    Usage: set assigner.num_anchors_per_level after anchor generation.
    """

    def __init__(self, top_k: int = 9):
        self.top_k = top_k
        self.num_anchors_per_level = None  # list[int], set from train.py

    def assign(
        self,
        anchors: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ):
        """
        Returns:
          obj_targets [A]    1=positive, 0=negative  (no ignore zone in ATSS)
          cls_targets [A]    class index for positives, -1 otherwise
          box_targets [A,4]  GT box for positives, zeros otherwise
        """
        A = anchors.shape[0]
        G = gt_boxes.shape[0]
        device = anchors.device

        obj_t = torch.zeros(A, dtype=torch.long, device=device)
        cls_t = torch.full((A,), -1, dtype=torch.long, device=device)
        box_t = torch.zeros(A, 4, dtype=torch.float32, device=device)

        if G == 0:
            return obj_t, cls_t, box_t

        num_per_level = self.num_anchors_per_level or [A]

        anchor_cx = (anchors[:, 0] + anchors[:, 2]) / 2  # [A]
        anchor_cy = (anchors[:, 1] + anchors[:, 3]) / 2  # [A]
        gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2    # [G]
        gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2    # [G]

        iou = box_iou(anchors, gt_boxes)  # [A, G]

        # Build candidate mask: top-k anchors per FPN level per GT by L2 distance
        candidate_mask = torch.zeros(A, G, dtype=torch.bool, device=device)
        start = 0
        for n_l in num_per_level:
            end = start + n_l
            # squared distance: [n_l, G]
            dx = anchor_cx[start:end, None] - gt_cx[None, :]
            dy = anchor_cy[start:end, None] - gt_cy[None, :]
            dist2 = dx.pow(2) + dy.pow(2)

            k = min(self.top_k, n_l)
            topk_idx = dist2.topk(k, dim=0, largest=False).indices  # [k, G]

            # Scatter into global index space
            g_range = torch.arange(G, device=device).unsqueeze(0).expand(k, -1)  # [k, G]
            candidate_mask[topk_idx + start, g_range] = True
            start = end

        # Compute dynamic IoU threshold per GT: mean + std over candidates
        cand_iou = iou * candidate_mask.float()  # [A, G], 0 for non-candidates
        n_cands = candidate_mask.float().sum(dim=0).clamp(min=1)   # [G]
        mean_iou = cand_iou.sum(dim=0) / n_cands                   # [G]
        var_iou = (cand_iou.pow(2).sum(dim=0) / n_cands - mean_iou.pow(2)).clamp(min=0)
        std_iou = var_iou.sqrt()
        iou_thresh = mean_iou + std_iou  # [G]

        # Require anchor center inside GT box
        inside = (
            (anchor_cx[:, None] >= gt_boxes[None, :, 0]) &
            (anchor_cx[:, None] <= gt_boxes[None, :, 2]) &
            (anchor_cy[:, None] >= gt_boxes[None, :, 1]) &
            (anchor_cy[:, None] <= gt_boxes[None, :, 3])
        )  # [A, G]

        # Positive mask: candidate AND IoU >= dynamic threshold AND inside GT
        pos_mask = candidate_mask & (iou >= iou_thresh[None, :]) & inside  # [A, G]

        # Fallback: GTs with no positives → best candidate by IoU (no inside constraint)
        no_pos = ~pos_mask.any(dim=0)  # [G]
        for g in no_pos.nonzero(as_tuple=False).view(-1):
            cands = candidate_mask[:, g]
            if cands.any():
                best = (iou[:, g] * cands.float()).argmax()
            else:
                best = iou[:, g].argmax()
            pos_mask[best, g] = True

        # Conflict resolution: anchor positive for multiple GTs → keep best IoU
        any_pos = pos_mask.any(dim=1)  # [A]
        matched_gt = (iou * pos_mask.float()).argmax(dim=1)  # [A]

        obj_t[any_pos] = 1
        cls_t[any_pos] = gt_labels[matched_gt[any_pos]]
        box_t[any_pos] = gt_boxes[matched_gt[any_pos]]

        return obj_t, cls_t, box_t


# Keep old assigner as fallback / alternative
class IoUAssigner:
    def __init__(self, pos_iou: float = 0.5, neg_iou: float = 0.4):
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou

    def assign(self, anchors, gt_boxes, gt_labels):
        A = anchors.shape[0]
        device = anchors.device
        obj_t = torch.zeros(A, dtype=torch.long, device=device)
        cls_t = torch.full((A,), -1, dtype=torch.long, device=device)
        box_t = torch.zeros(A, 4, dtype=torch.float32, device=device)
        if gt_boxes.shape[0] == 0:
            return obj_t, cls_t, box_t
        iou = box_iou(anchors, gt_boxes)
        max_iou, best_gt = iou.max(dim=1)
        obj_t[max_iou < self.neg_iou] = 0
        obj_t[(max_iou >= self.neg_iou) & (max_iou < self.pos_iou)] = -1
        obj_t[max_iou >= self.pos_iou] = 1
        best_anchor_per_gt = iou.max(dim=0).indices
        obj_t[best_anchor_per_gt] = 1
        matched_gt = best_gt.clone()
        for g, a in enumerate(best_anchor_per_gt):
            matched_gt[a] = g
        pos = obj_t == 1
        if pos.any():
            cls_t[pos] = gt_labels[matched_gt[pos]]
            box_t[pos] = gt_boxes[matched_gt[pos]]
        return obj_t, cls_t, box_t
