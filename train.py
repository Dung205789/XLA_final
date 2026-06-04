"""Training entry point.

Usage:
    python train.py \
        --train_data  ./public/annotations/train.json \
        --val_data    ./public/annotations/val.json \
        --image_dir   ./public/train/images \
        --val_image_dir ./public/val/images \
        --checkpoint_dir ./models/
"""

import argparse
import json
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.anchors import (
    DEFAULT_ANCHOR_WHS,
    STRIDES,
    compute_num_anchors_per_level,
    generate_all_anchors,
    kmeans_anchors,
)
from utils.assigner import ATSSAssigner
from utils.dataset import DetectionDataset, detection_collate_fn, val_collate_fn
from utils.losses import DetectionLoss
from utils.model import ResNet18FPNDetector
from utils.train_engine import training_loop
from utils.transforms import TrainTransform, ValTransform


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(train_json_path: str, classes: list) -> torch.Tensor:
    """Inverse-sqrt frequency weights to counter class imbalance."""
    with open(train_json_path) as f:
        data = json.load(f)
    class2idx = {c: i for i, c in enumerate(classes)}
    counts = np.ones(len(classes))  # start at 1 to avoid division by zero
    for ann in data.get("annotations", []):
        idx = class2idx.get(ann.get("class"), -1)
        if idx >= 0:
            counts[idx] += 1
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.min()  # normalize: min weight = 1.0
    print(f"  Class weights: " + "  ".join(f"{c}={w:.2f}" for c, w in zip(classes, weights)))
    return torch.tensor(weights, dtype=torch.float32)


def build_anchors(args, n_anchors_per_level: int = 3):
    """Run k-means (or fall back to defaults) and return anchors + per-level metadata."""
    n_total = n_anchors_per_level * len(STRIDES)
    print(f"Running k-means ({n_total} clusters) on training boxes …")
    try:
        all_whs = kmeans_anchors(args.train_data, n_clusters=n_total, input_size=args.input_size)
        anchor_whs_per_level = [
            all_whs[i * n_anchors_per_level: (i + 1) * n_anchors_per_level]
            for i in range(len(STRIDES))
        ]
        print(f"  K-means anchors: {all_whs}")
    except Exception as e:
        print(f"  K-means failed ({e}), using defaults.")
        anchor_whs_per_level = [
            DEFAULT_ANCHOR_WHS[i * n_anchors_per_level: (i + 1) * n_anchors_per_level]
            for i in range(len(STRIDES))
        ]

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    anchor_path = os.path.join(args.checkpoint_dir, "anchors.json")
    with open(anchor_path, "w") as f:
        json.dump({"strides": STRIDES, "anchor_whs_per_level": anchor_whs_per_level}, f, indent=2)
    print(f"  Anchors saved to {anchor_path}")

    anchors = generate_all_anchors(args.input_size, STRIDES, anchor_whs_per_level)
    n_per_level = compute_num_anchors_per_level(args.input_size, STRIDES, anchor_whs_per_level)
    return anchors, anchor_whs_per_level, n_per_level


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_data", default="./public/annotations/train.json")
    p.add_argument("--val_data", default="./public/annotations/val.json")
    p.add_argument("--image_dir", default="./public/train/images")
    p.add_argument("--val_image_dir", default="./public/val/images")
    p.add_argument("--checkpoint_dir", default="./models/")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--input_size", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--freeze_epochs", type=int, default=3)
    p.add_argument("--conf_thresh", type=float, default=0.25)
    p.add_argument("--nms_thresh", type=float, default=0.5)
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_kmeans", action="store_true")
    p.add_argument("--mosaic_prob", type=float, default=0.5,
                   help="Probability of applying mosaic augmentation (0 to disable)")
    p.add_argument("--no_class_weights", action="store_true",
                   help="Disable class-frequency weighting in classification loss")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Anchors
    if args.no_kmeans:
        anchor_whs_per_level = [
            DEFAULT_ANCHOR_WHS[i * 3: (i + 1) * 3] for i in range(len(STRIDES))
        ]
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        with open(os.path.join(args.checkpoint_dir, "anchors.json"), "w") as f:
            json.dump({"strides": STRIDES, "anchor_whs_per_level": anchor_whs_per_level}, f)
        anchors = generate_all_anchors(args.input_size, STRIDES, anchor_whs_per_level)
        n_per_level = compute_num_anchors_per_level(args.input_size, STRIDES, anchor_whs_per_level)
    else:
        anchors, anchor_whs_per_level, n_per_level = build_anchors(args)

    n_anchors = len(anchor_whs_per_level[0])
    print(f"Anchors per level: {n_per_level}  Total: {sum(n_per_level)}")

    # Datasets
    train_transform = TrainTransform(args.input_size)
    train_ds = DetectionDataset(
        args.train_data, args.image_dir,
        transform=train_transform,
        mosaic_prob=args.mosaic_prob,
    )
    val_ds = DetectionDataset(args.val_data, args.val_image_dir, ValTransform(args.input_size))
    print(f"Train: {len(train_ds)} images  Val: {len(val_ds)} images  "
          f"Mosaic prob: {args.mosaic_prob}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=detection_collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=val_collate_fn, pin_memory=True,
    )

    # Model
    with open(args.train_data) as f:
        classes = json.load(f)["classes"]
    num_classes = len(classes)

    model = ResNet18FPNDetector(
        num_classes=num_classes, num_anchors=n_anchors, pretrained=True
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # LR schedule: warmup 2 epochs then cosine decay
    warmup_epochs = 2

    def lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep + 1) / warmup_epochs
        progress = (ep - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ATSS assigner
    assigner = ATSSAssigner(top_k=9)
    assigner.num_anchors_per_level = n_per_level

    # Loss with optional class weights
    cls_weights = None
    if not args.no_class_weights:
        cls_weights = compute_class_weights(args.train_data, classes)

    criterion = DetectionLoss(num_classes=num_classes, cls_weights=cls_weights)

    training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        anchors=anchors,
        assigner=assigner,
        criterion=criterion,
        classes=classes,
        device=device,
        gt_val_json=args.val_data,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        freeze_epochs=args.freeze_epochs,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        input_size=args.input_size,
        val_every=args.val_every,
    )


if __name__ == "__main__":
    main()
