"""Anchor generation and k-means anchor optimization."""

import json
import math

import numpy as np
import torch

# Better defaults: include small anchors for P3 to cover car/chair
DEFAULT_ANCHOR_WHS = [
    # P3 – stride 8 – small/medium objects (16–80 px)
    (20.0, 28.0), (38.0, 52.0), (64.0, 44.0),
    # P4 – stride 16 – medium objects (60–160 px)
    (80.0, 72.0), (96.0, 160.0), (152.0, 104.0),
    # P5 – stride 32 – large objects (120–400 px)
    (160.0, 192.0), (240.0, 160.0), (300.0, 300.0),
]

STRIDES = [8, 16, 32]


def _generate_level_anchors(
    feature_h: int, feature_w: int, stride: int, anchor_whs: list
) -> torch.Tensor:
    """Return [feature_h * feature_w * A, 4] anchors for one FPN level."""
    ys = (torch.arange(feature_h, dtype=torch.float32) + 0.5) * stride
    xs = (torch.arange(feature_w, dtype=torch.float32) + 0.5) * stride
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    cx = grid_x[:, :, None]
    cy = grid_y[:, :, None]

    whs = torch.tensor(anchor_whs, dtype=torch.float32)
    aw = whs[None, None, :, 0]
    ah = whs[None, None, :, 1]

    anchors = torch.stack([cx - aw / 2, cy - ah / 2, cx + aw / 2, cy + ah / 2], dim=-1)
    return anchors.reshape(-1, 4)


def compute_num_anchors_per_level(
    input_size: int,
    strides: list,
    anchor_whs_per_level: list,
) -> list:
    """Return the number of anchors for each FPN level."""
    return [(input_size // s) ** 2 * len(whs)
            for s, whs in zip(strides, anchor_whs_per_level)]


def generate_all_anchors(
    input_size: int,
    strides: list = None,
    anchor_whs_per_level: list = None,
) -> torch.Tensor:
    """
    Generate all anchors for all FPN levels.
    Returns: [total_anchors, 4] in [x1, y1, x2, y2] pixel space.
    """
    if strides is None:
        strides = STRIDES
    if anchor_whs_per_level is None:
        anchor_whs_per_level = [
            DEFAULT_ANCHOR_WHS[i * 3: (i + 1) * 3] for i in range(len(strides))
        ]

    all_anchors = []
    for stride, whs in zip(strides, anchor_whs_per_level):
        fh = fw = input_size // stride
        all_anchors.append(_generate_level_anchors(fh, fw, stride, whs))
    return torch.cat(all_anchors, dim=0)


def kmeans_anchors(
    train_json_path: str,
    n_clusters: int = 9,
    input_size: int = 512,
    n_init: int = 10,
    max_iter: int = 300,
) -> list:
    """
    Run k-means on training boxes (scaled to input_size) to find optimal (w, h) anchors.
    Uses stratified sampling to prevent person-dominated anchors from crowding out
    smaller classes (car, chair).
    Returns list of (w, h) tuples sorted by area, length = n_clusters.
    """
    from sklearn.cluster import KMeans

    with open(train_json_path) as f:
        data = json.load(f)

    id2size = {img["id"]: (img["width"], img["height"]) for img in data["images"]}

    # Collect per-class boxes
    class_boxes: dict = {}
    for ann in data.get("annotations", []):
        iw, ih = id2size[ann["image_id"]]
        x1, y1, x2, y2 = ann["bbox"]
        bw = (x2 - x1) / iw * input_size
        bh = (y2 - y1) / ih * input_size
        if bw > 2 and bh > 2:
            c = ann.get("class", "unknown")
            class_boxes.setdefault(c, []).append([bw, bh])

    if not class_boxes:
        raise ValueError("No valid boxes found")

    # Stratified sampling: cap dominant class to 3× the rarest class count
    min_count = min(len(v) for v in class_boxes.values())
    max_count = max(min_count * 3, 200)
    whs = []
    for boxes in class_boxes.values():
        sample = boxes if len(boxes) <= max_count else [
            boxes[i] for i in np.random.choice(len(boxes), max_count, replace=False)
        ]
        whs.extend(sample)

    whs = np.array(whs, dtype=np.float32)
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=n_init, max_iter=max_iter)
    km.fit(whs)
    centers = km.cluster_centers_

    order = np.argsort(centers[:, 0] * centers[:, 1])
    centers = centers[order]
    return [(float(c[0]), float(c[1])) for c in centers]
