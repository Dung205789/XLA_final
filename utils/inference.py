"""Decode model outputs → filtered boxes in original image coordinates."""

import numpy as np
import torch

from .box_ops import clip_boxes, decode_deltas
from .nms import per_class_nms
from .transforms import inverse_letterbox_boxes


def decode_single(
    obj_logits: torch.Tensor,
    cls_logits: torch.Tensor,
    box_preds: torch.Tensor,
    anchors: torch.Tensor,
    input_size: int,
    conf_thresh: float,
    nms_thresh: float,
    max_dets: int = 100,
):
    """
    Decode one image's raw model output into (boxes, scores, labels) in letterboxed space.
    obj_logits: [A, 1]  cls_logits: [A, C]  box_preds: [A, 4]  anchors: [A, 4]
    Returns: (boxes [K,4], scores [K], labels [K]) tensors – already clipped
    """
    obj_score = torch.sigmoid(obj_logits[:, 0])          # [A]
    cls_prob = torch.softmax(cls_logits, dim=-1)          # [A, C]
    cls_score, cls_label = cls_prob.max(dim=-1)           # [A]
    confidence = obj_score * cls_score                    # [A]

    keep = confidence > conf_thresh
    if not keep.any():
        empty = torch.zeros(0, dtype=torch.long, device=obj_logits.device)
        return torch.zeros(0, 4, device=obj_logits.device), \
               torch.zeros(0, device=obj_logits.device), empty

    boxes = decode_deltas(anchors[keep], box_preds[keep])
    boxes = clip_boxes(boxes, input_size)
    scores = confidence[keep]
    labels = cls_label[keep]

    keep_idx = per_class_nms(boxes, scores, labels, iou_threshold=nms_thresh)
    boxes = boxes[keep_idx]
    scores = scores[keep_idx]
    labels = labels[keep_idx]

    top = scores.argsort(descending=True)[:max_dets]
    return boxes[top], scores[top], labels[top]


@torch.no_grad()
def run_inference(
    model,
    loader,
    anchors: torch.Tensor,
    classes: list,
    device: torch.device,
    conf_thresh: float = 0.05,
    nms_thresh: float = 0.5,
    input_size: int = 512,
):
    """
    Run inference over a DataLoader, return predictions list ready for JSON export.
    Loader must yield: (images, image_ids, orig_sizes, scales, pads)
    """
    model.eval()
    anchors = anchors.to(device)
    predictions = []

    for images, image_ids, orig_sizes, scales, pads in loader:
        images = images.to(device)
        obj_l, cls_l, box_p = model(images)

        B = images.shape[0]
        for i in range(B):
            boxes, scores, labels = decode_single(
                obj_l[i], cls_l[i], box_p[i], anchors,
                input_size, conf_thresh, nms_thresh,
            )

            # Unmap to original image coords
            boxes_np = inverse_letterbox_boxes(boxes, orig_sizes[i], scales[i], pads[i])
            scores_np = scores.cpu().numpy()
            labels_np = labels.cpu().numpy()

            # Filter degenerate boxes (width or height <= 0.5 px)
            valid = (boxes_np[:, 2] > boxes_np[:, 0] + 0.5) & (boxes_np[:, 3] > boxes_np[:, 1] + 0.5)
            boxes_np = boxes_np[valid]
            scores_np = scores_np[valid]
            labels_np = labels_np[valid]

            out_boxes = []
            for box, score, lbl in zip(boxes_np, scores_np, labels_np):
                out_boxes.append({
                    "class": classes[int(lbl)],
                    "confidence": round(float(score), 6),
                    "bbox": [round(float(v), 2) for v in box],
                })

            predictions.append({"image_id": image_ids[i], "boxes": out_boxes})

    return predictions
