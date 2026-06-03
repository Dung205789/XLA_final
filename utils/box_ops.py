"""Box geometry utilities: IoU, GIoU loss, delta encode/decode, clipping."""

import torch
import torch.nn.functional as F


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes.
    boxes1: [N, 4]  boxes2: [M, 4]  both in [x1, y1, x2, y2]
    Returns: [N, M]
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    ix1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    iy1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    ix2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    iy2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    GIoU loss element-wise.
    pred, target: [N, 4] in [x1, y1, x2, y2]
    Returns: [N]
    """
    px1, py1, px2, py2 = pred.unbind(-1)
    tx1, ty1, tx2, ty2 = target.unbind(-1)

    inter = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(min=0) * \
            (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(min=0)

    area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_t = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
    union = area_p + area_t - inter

    iou = inter / union.clamp(min=1e-6)

    enc_x1 = torch.min(px1, tx1)
    enc_y1 = torch.min(py1, ty1)
    enc_x2 = torch.max(px2, tx2)
    enc_y2 = torch.max(py2, ty2)
    enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0)

    giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)
    return 1.0 - giou


def encode_deltas(anchors: torch.Tensor, gt_boxes: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Encode GT boxes as (tx, ty, tw, th) deltas relative to anchors.
    anchors, gt_boxes: [N, 4] in [x1, y1, x2, y2]
    Returns: [N, 4]
    """
    aw = (anchors[:, 2] - anchors[:, 0]).clamp(min=eps)
    ah = (anchors[:, 3] - anchors[:, 1]).clamp(min=eps)
    acx = anchors[:, 0] + 0.5 * aw
    acy = anchors[:, 1] + 0.5 * ah

    gw = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=eps)
    gh = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=eps)
    gcx = gt_boxes[:, 0] + 0.5 * gw
    gcy = gt_boxes[:, 1] + 0.5 * gh

    tx = (gcx - acx) / aw
    ty = (gcy - acy) / ah
    tw = torch.log(gw / aw)
    th = torch.log(gh / ah)
    return torch.stack([tx, ty, tw, th], dim=-1)


def decode_deltas(anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """
    Decode (tx, ty, tw, th) deltas against anchors to get [x1, y1, x2, y2] boxes.
    anchors, deltas: [N, 4]
    Returns: [N, 4]
    """
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    acx = anchors[:, 0] + 0.5 * aw
    acy = anchors[:, 1] + 0.5 * ah

    tx, ty, tw, th = deltas.unbind(-1)
    tw = tw.clamp(-4.0, 4.0)
    th = th.clamp(-4.0, 4.0)

    cx = tx * aw + acx
    cy = ty * ah + acy
    w = torch.exp(tw) * aw
    h = torch.exp(th) * ah

    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def clip_boxes(boxes: torch.Tensor, size: int) -> torch.Tensor:
    """Clip boxes to [0, size] boundaries."""
    return boxes.clamp(min=0.0, max=float(size))
