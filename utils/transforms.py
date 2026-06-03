"""Image transforms: letterbox resize, augmentation, normalization."""

from typing import Tuple

import numpy as np
import torch
from PIL import Image
import albumentations as A

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox_resize(
    img: Image.Image,
    boxes: np.ndarray,
    target_size: int = 512,
) -> Tuple[Image.Image, np.ndarray, float, Tuple[int, int]]:
    """
    Resize image preserving aspect ratio, pad to square.
    Returns: (padded_img, transformed_boxes, scale, (pad_w, pad_h))
    Boxes in [x1, y1, x2, y2] pixel coords.
    """
    ow, oh = img.size
    scale = min(target_size / ow, target_size / oh)
    nw, nh = int(round(ow * scale)), int(round(oh * scale))
    resized = img.resize((nw, nh), Image.BILINEAR)

    pw = (target_size - nw) // 2
    ph = (target_size - nh) // 2
    padded = Image.new("RGB", (target_size, target_size), (114, 114, 114))
    padded.paste(resized, (pw, ph))

    if len(boxes) > 0:
        b = boxes.copy().astype(np.float32)
        b[:, [0, 2]] = (b[:, [0, 2]] * scale + pw).clip(0, target_size)
        b[:, [1, 3]] = (b[:, [1, 3]] * scale + ph).clip(0, target_size)
    else:
        b = boxes.copy()

    return padded, b, scale, (pw, ph)


def inverse_letterbox_boxes(
    boxes,
    orig_size: Tuple[int, int],
    scale: float,
    pad: Tuple[int, int],
) -> np.ndarray:
    """
    Unmap boxes from letterboxed space back to original image coords.
    orig_size: (orig_h, orig_w)
    """
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    orig_h, orig_w = orig_size
    pw, ph = pad
    b = boxes.copy().astype(np.float32)
    b[:, [0, 2]] = ((b[:, [0, 2]] - pw) / scale).clip(0, orig_w)
    b[:, [1, 3]] = ((b[:, [1, 3]] - ph) / scale).clip(0, orig_h)
    return b


def _to_tensor(img_pil: Image.Image) -> torch.Tensor:
    arr = np.array(img_pil).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return torch.from_numpy(arr.transpose(2, 0, 1))


class TrainTransform:
    """Augment then letterbox-resize then normalize."""

    def __init__(self, target_size: int = 512):
        self.target_size = target_size
        self.aug = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.5),
                A.RandomScale(scale_limit=0.2, p=0.5),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
                min_area=4,
                min_visibility=0.25,
            ),
        )

    def __call__(self, img: Image.Image, boxes: np.ndarray, labels: np.ndarray):
        arr = np.array(img)
        if len(boxes) > 0:
            out = self.aug(image=arr, bboxes=boxes.tolist(), labels=labels.tolist())
            arr = out["image"]
            boxes = np.array(out["bboxes"], dtype=np.float32) if out["bboxes"] else np.zeros((0, 4), dtype=np.float32)
            labels = np.array(out["labels"], dtype=np.int64) if out["labels"] else np.zeros(0, dtype=np.int64)

        img_pil, boxes, scale, pad = letterbox_resize(Image.fromarray(arr), boxes, self.target_size)
        return _to_tensor(img_pil), boxes, labels, scale, pad


class ValTransform:
    """Letterbox-resize then normalize only (no augmentation)."""

    def __init__(self, target_size: int = 512):
        self.target_size = target_size

    def __call__(self, img: Image.Image, boxes: np.ndarray, labels: np.ndarray):
        img_pil, boxes, scale, pad = letterbox_resize(img, boxes, self.target_size)
        return _to_tensor(img_pil), boxes, labels, scale, pad
