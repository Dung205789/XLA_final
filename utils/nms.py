"""Per-class Non-Maximum Suppression."""

import torch
from torchvision.ops import nms


def per_class_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Apply NMS independently for each class.
    boxes: [N, 4]  scores: [N]  labels: [N] (int)
    Returns: kept indices tensor
    """
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    keep_all = []
    for cls in labels.unique():
        mask = labels == cls
        orig_idx = torch.where(mask)[0]
        kept = nms(boxes[mask], scores[mask], iou_threshold)
        keep_all.append(orig_idx[kept])

    return torch.cat(keep_all) if keep_all else torch.zeros(0, dtype=torch.long, device=boxes.device)
