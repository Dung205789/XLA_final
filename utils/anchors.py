"""Anchor generation and k-means anchor optimization."""

import json
import math

import numpy as np
import torch

# Sensible defaults if k-means is not run
DEFAULT_ANCHOR_WHS = [
    # P3 – stride 8 – small objects
    (26.0, 34.0), (50.0, 38.0), (38.0, 74.0),
    # P4 – stride 16 – medium objects
    (76.0, 58.0), (62.0, 122.0), (120.0, 90.0),
    # P5 – stride 32 – large objects
    (116.0, 162.0), (180.0, 120.0), (230.0, 230.0),
]

STRIDES = [8, 16, 32]


def _generate_level_anchors(
    feature_h: int, feature_w: int, stride: int, anchor_whs: list
) -> torch.Tensor:
    """Return [feature_h * feature_w * A, 4] anchors for one FPN level."""
    ys = (torch.arange(feature_h, dtype=torch.float32) + 0.5) * stride
    xs = (torch.arange(feature_w, dtype=torch.float32) + 0.5) * stride
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [H, W]

    cx = grid_x[:, :, None]  # [H, W, 1]
    cy = grid_y[:, :, None]  # [H, W, 1]

    whs = torch.tensor(anchor_whs, dtype=torch.float32)  # [A, 2]
    aw = whs[None, None, :, 0]  # [1, 1, A]
    ah = whs[None, None, :, 1]

    x1 = cx - aw / 2
    y1 = cy - ah / 2
    x2 = cx + aw / 2
    y2 = cy + ah / 2

    anchors = torch.stack([x1, y1, x2, y2], dim=-1)  # [H, W, A, 4]
    return anchors.reshape(-1, 4)  # [H*W*A, 4]


def generate_all_anchors(
    input_size: int,
    strides: list = None,
    anchor_whs_per_level: list = None,
) -> torch.Tensor:
    """
    Generate all anchors for all FPN levels.
    Returns: [total_anchors, 4] in [x1, y1, x2, y2] in input-image pixel space.
    """
    if strides is None:
        strides = STRIDES
    if anchor_whs_per_level is None:
        anchor_whs_per_level = [
            DEFAULT_ANCHOR_WHS[i * 3 : (i + 1) * 3] for i in range(len(strides))
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
    Run k-means on training boxes (scaled to input_size) to find optimal anchor (w, h) pairs.
    Returns list of (w, h) tuples sorted by area, length = n_clusters.
    """
    from sklearn.cluster import KMeans

    with open(train_json_path) as f:
        data = json.load(f)

    id2size = {img["id"]: (img["width"], img["height"]) for img in data["images"]}

    whs = []
    for ann in data.get("annotations", []):
        iw, ih = id2size[ann["image_id"]]
        x1, y1, x2, y2 = ann["bbox"]
        bw = (x2 - x1) / iw * input_size
        bh = (y2 - y1) / ih * input_size
        if bw > 2 and bh > 2:
            whs.append([bw, bh])

    whs = np.array(whs, dtype=np.float32)
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=n_init, max_iter=max_iter)
    km.fit(whs)
    centers = km.cluster_centers_

    order = np.argsort(centers[:, 0] * centers[:, 1])
    centers = centers[order]
    return [(float(c[0]), float(c[1])) for c in centers]
