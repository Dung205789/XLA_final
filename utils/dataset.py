"""Dataset classes for training, validation, and inference."""

import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .transforms import mosaic_combine, _to_tensor


class DetectionDataset(Dataset):
    """Loads COCO-style annotations for training / validation."""

    def __init__(
        self,
        json_path: str,
        image_dir: str,
        transform=None,
        mosaic_prob: float = 0.0,
    ):
        with open(json_path) as f:
            data = json.load(f)

        self.classes: List[str] = data["classes"]
        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.mosaic_prob = mosaic_prob

        ann_by_img = {}
        for ann in data.get("annotations", []):
            iid = ann["image_id"]
            ann_by_img.setdefault(iid, []).append(ann)

        self.samples = []
        for img_info in data["images"]:
            iid = img_info["id"]
            anns = ann_by_img.get(iid, [])
            boxes = [[*a["bbox"]] for a in anns]
            labels = [self.class2idx[a["class"]] for a in anns]
            self.samples.append(
                {
                    "image_id": iid,
                    "file_name": Path(img_info["file_name"]).name,
                    "width": img_info["width"],
                    "height": img_info["height"],
                    "boxes": boxes,
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.samples)

    def load_raw(self, idx: int):
        """Return (PIL_image, boxes_np, labels_np) without any transform."""
        s = self.samples[idx]
        img = Image.open(self.image_dir / s["file_name"]).convert("RGB")
        boxes = (np.array(s["boxes"], dtype=np.float32)
                 if s["boxes"] else np.zeros((0, 4), dtype=np.float32))
        labels = (np.array(s["labels"], dtype=np.int64)
                  if s["labels"] else np.zeros(0, dtype=np.int64))
        return img, boxes, labels

    def __getitem__(self, idx):
        s = self.samples[idx]
        orig_size = (s["height"], s["width"])

        # Mosaic: combine 4 images (only when TrainTransform with augment_only is available)
        use_mosaic = (
            self.mosaic_prob > 0
            and random.random() < self.mosaic_prob
            and self.transform is not None
            and hasattr(self.transform, "augment_only")
        )

        if use_mosaic:
            indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
            aug_samples = []
            for i in indices:
                img_raw, boxes_raw, labels_raw = self.load_raw(i)
                arr, b, l = self.transform.augment_only(img_raw, boxes_raw, labels_raw)
                aug_samples.append((arr, b, l))
            arr, boxes, labels = mosaic_combine(aug_samples, self.transform.target_size)
            img_tensor = _to_tensor(Image.fromarray(arr))
            scale, pad = 1.0, (0, 0)
        else:
            img, boxes, labels = self.load_raw(idx)
            scale, pad = 1.0, (0, 0)
            if self.transform is not None:
                img_tensor, boxes, labels, scale, pad = self.transform(img, boxes, labels)
            else:
                img_tensor = _to_tensor(img)
                boxes = np.zeros((0, 4), dtype=np.float32)
                labels = np.zeros(0, dtype=np.int64)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "image_id": s["image_id"],
            "orig_size": orig_size,
            "scale": scale,
            "pad": pad,
        }
        return img_tensor, target


class InferenceDataset(Dataset):
    """Loads images from a directory for inference (no annotations required)."""

    def __init__(self, image_dir: str, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.files = sorted(f for f in self.image_dir.iterdir() if f.suffix.lower() in exts)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        img = Image.open(fpath).convert("RGB")
        orig_size = (img.height, img.width)
        image_id = fpath.name

        scale, pad = 1.0, (0, 0)
        dummy_boxes = np.zeros((0, 4), dtype=np.float32)
        dummy_labels = np.zeros(0, dtype=np.int64)
        if self.transform is not None:
            img, _, _, scale, pad = self.transform(img, dummy_boxes, dummy_labels)

        return img, image_id, orig_size, scale, pad


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


def inference_collate_fn(batch):
    images, image_ids, orig_sizes, scales, pads = zip(*batch)
    return torch.stack(images), list(image_ids), list(orig_sizes), list(scales), list(pads)


def val_collate_fn(batch):
    """Collate for validation DataLoader: returns (images, image_ids, orig_sizes, scales, pads)."""
    images = torch.stack([b[0] for b in batch])
    image_ids = [b[1]["image_id"] for b in batch]
    orig_sizes = [b[1]["orig_size"] for b in batch]
    scales = [b[1]["scale"] for b in batch]
    pads = [b[1]["pad"] for b in batch]
    return images, image_ids, orig_sizes, scales, pads
