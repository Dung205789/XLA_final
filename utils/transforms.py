"""Image transforms: letterbox resize, augmentation, normalization, mosaic."""

import random
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
    """Unmap boxes from letterboxed space back to original image coordinates."""
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


def mosaic_combine(
    samples: list,
    target_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine 4 (np_array, boxes, labels) tuples into a mosaic image.
    Each np_array is (H, W, 3) uint8. Boxes are [x1, y1, x2, y2] in image coords.
    Returns: (mosaic_np [H,W,3], all_boxes [N,4], all_labels [N])
    """
    s = target_size
    cx = int(random.uniform(0.3 * s, 0.7 * s))
    cy = int(random.uniform(0.3 * s, 0.7 * s))

    canvas = np.full((s, s, 3), 114, dtype=np.uint8)
    all_boxes = []
    all_labels = []

    # (y1, y2, x1, x2) slice within canvas for each quadrant
    placements = [
        (0, cy, 0, cx),    # top-left
        (0, cy, cx, s),    # top-right
        (cy, s, 0, cx),    # bottom-left
        (cy, s, cx, s),    # bottom-right
    ]

    for idx, (y1p, y2p, x1p, x2p) in enumerate(placements):
        img_np, boxes, labels = samples[idx]
        oh, ow = img_np.shape[:2]

        pw = x2p - x1p
        ph = y2p - y1p
        if pw <= 0 or ph <= 0 or ow <= 0 or oh <= 0:
            continue

        scale = min(pw / ow, ph / oh)
        new_w = max(int(ow * scale), 1)
        new_h = max(int(oh * scale), 1)

        resized = np.array(Image.fromarray(img_np).resize((new_w, new_h), Image.BILINEAR))

        paste_w = min(new_w, pw)
        paste_h = min(new_h, ph)
        canvas[y1p:y1p + paste_h, x1p:x1p + paste_w] = resized[:paste_h, :paste_w]

        if len(boxes) > 0:
            b = boxes.copy().astype(np.float32)
            b[:, [0, 2]] = b[:, [0, 2]] * scale + x1p
            b[:, [1, 3]] = b[:, [1, 3]] * scale + y1p
            # Clip to the pasted region
            b[:, 0] = np.clip(b[:, 0], x1p, x1p + paste_w)
            b[:, 1] = np.clip(b[:, 1], y1p, y1p + paste_h)
            b[:, 2] = np.clip(b[:, 2], x1p, x1p + paste_w)
            b[:, 3] = np.clip(b[:, 3], y1p, y1p + paste_h)
            bw = b[:, 2] - b[:, 0]
            bh = b[:, 3] - b[:, 1]
            valid = (bw > 4) & (bh > 4)
            if valid.any():
                all_boxes.append(b[valid])
                all_labels.append(labels[valid])

    if all_boxes:
        return canvas, np.concatenate(all_boxes), np.concatenate(all_labels)
    return canvas, np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)


class TrainTransform:
    """Augment then letterbox-resize then normalize."""

    def __init__(self, target_size: int = 512):
        self.target_size = target_size
        self.aug = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.7),
                A.RandomScale(scale_limit=0.4, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.3,
                    border_mode=0, value=(114, 114, 114),
                ),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ], p=0.2),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
                min_area=4,
                min_visibility=0.2,
            ),
        )

    def augment_only(self, img: Image.Image, boxes: np.ndarray, labels: np.ndarray):
        """Apply augmentation only (no letterbox). Returns (np_array, boxes, labels)."""
        arr = np.array(img)
        if len(boxes) > 0:
            out = self.aug(image=arr, bboxes=boxes.tolist(), labels=labels.tolist())
            arr = out["image"]
            boxes = (np.array(out["bboxes"], dtype=np.float32)
                     if out["bboxes"] else np.zeros((0, 4), dtype=np.float32))
            labels = (np.array(out["labels"], dtype=np.int64)
                      if out["labels"] else np.zeros(0, dtype=np.int64))
        return arr, boxes, labels

    def __call__(self, img: Image.Image, boxes: np.ndarray, labels: np.ndarray):
        arr, boxes, labels = self.augment_only(img, boxes, labels)
        img_pil, boxes, scale, pad = letterbox_resize(Image.fromarray(arr), boxes, self.target_size)
        return _to_tensor(img_pil), boxes, labels, scale, pad


class ValTransform:
    """Letterbox-resize then normalize only (no augmentation)."""

    def __init__(self, target_size: int = 512):
        self.target_size = target_size

    def __call__(self, img: Image.Image, boxes: np.ndarray, labels: np.ndarray):
        img_pil, boxes, scale, pad = letterbox_resize(img, boxes, self.target_size)
        return _to_tensor(img_pil), boxes, labels, scale, pad
