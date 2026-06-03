"""Dataset classes for training, validation, and inference."""

import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class DetectionDataset(Dataset):
    """Loads COCO-style annotations for training / validation."""

    def __init__(self, json_path: str, image_dir: str, transform=None):
        with open(json_path) as f:
            data = json.load(f)

        self.classes: List[str] = data["classes"]
        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        self.image_dir = Path(image_dir)
        self.transform = transform

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

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(self.image_dir / s["file_name"]).convert("RGB")
        boxes = np.array(s["boxes"], dtype=np.float32) if s["boxes"] else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(s["labels"], dtype=np.int64) if s["labels"] else np.zeros(0, dtype=np.int64)
        orig_size = (s["height"], s["width"])

        scale, pad = 1.0, (0, 0)
        if self.transform is not None:
            img, boxes, labels, scale, pad = self.transform(img, boxes, labels)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "image_id": s["image_id"],
            "orig_size": orig_size,
            "scale": scale,
            "pad": pad,
        }
        return img, target


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
