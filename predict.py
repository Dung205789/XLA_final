"""Inference entry point.

Usage:
    python predict.py \
        --image_dir /path/to/images \
        --output predictions.json \
        [--checkpoint ./models/best.pth] \
        [--anchor_cfg ./models/anchors.json] \
        [--conf_thresh 0.05] \
        [--nms_thresh 0.5]
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from utils.anchors import DEFAULT_ANCHOR_WHS, STRIDES, generate_all_anchors
from utils.dataset import InferenceDataset, inference_collate_fn
from utils.inference import run_inference
from utils.io import load_checkpoint
from utils.model import ResNet18FPNDetector
from utils.transforms import ValTransform

CLASSES = ["person", "car", "dog", "cat", "chair"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", required=True)
    p.add_argument("--output", default="predictions.json")
    p.add_argument("--checkpoint", default="./models/best.pth")
    p.add_argument("--anchor_cfg", default="./models/anchors.json")
    p.add_argument("--conf_thresh", type=float, default=0.05)
    p.add_argument("--nms_thresh", type=float, default=0.5)
    p.add_argument("--input_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load anchors
    if os.path.exists(args.anchor_cfg):
        with open(args.anchor_cfg) as f:
            cfg = json.load(f)
        strides = cfg["strides"]
        anchor_whs_per_level = cfg["anchor_whs_per_level"]
    else:
        print(f"[warn] anchor_cfg not found at {args.anchor_cfg}, using defaults")
        strides = STRIDES
        anchor_whs_per_level = [DEFAULT_ANCHOR_WHS[i * 3: (i + 1) * 3] for i in range(3)]

    anchors = generate_all_anchors(args.input_size, strides, anchor_whs_per_level)
    n_anchors = len(anchor_whs_per_level[0])

    # Model
    model = ResNet18FPNDetector(
        num_classes=len(CLASSES), num_anchors=n_anchors, pretrained=False
    ).to(device)
    load_checkpoint(args.checkpoint, model, device=str(device))
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Dataset
    ds = InferenceDataset(args.image_dir, ValTransform(args.input_size))
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=inference_collate_fn,
    )
    print(f"Images found: {len(ds)}")

    predictions = run_inference(
        model, loader, anchors, CLASSES, device,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        input_size=args.input_size,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Predictions written: {len(predictions)} images → {args.output}")


if __name__ == "__main__":
    main()
