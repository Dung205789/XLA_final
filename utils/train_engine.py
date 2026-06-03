"""Training and validation loops."""

import json
import os
import tempfile
from typing import Optional

import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .inference import run_inference
from .io import save_checkpoint
from .metrics import compute_map


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler: GradScaler,
    anchors: torch.Tensor,
    assigner,
    criterion,
    device: torch.device,
    epoch: int,
):
    model.train()
    anchors = anchors.to(device)
    total_loss = 0.0
    info_accum = {"obj": 0.0, "cls": 0.0, "box": 0.0, "num_pos": 0.0}
    n = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for images, targets in pbar:
        images = images.to(device)

        with autocast("cuda", enabled=device.type == "cuda"):
            predictions = model(images)

            batch_t = []
            for tgt in targets:
                gt_boxes = tgt["boxes"].to(device)
                gt_labels = tgt["labels"].to(device)
                obj_t, cls_t, box_t = assigner.assign(anchors, gt_boxes, gt_labels)
                batch_t.append({"obj_targets": obj_t, "cls_targets": cls_t, "box_targets": box_t})

            loss, info = criterion(predictions, batch_t, anchors)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for k in info_accum:
            info_accum[k] += info[k]
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", pos=f"{info['num_pos']:.1f}")

    return total_loss / max(n, 1), {k: v / max(n, 1) for k, v in info_accum.items()}


@torch.no_grad()
def validate(
    model,
    val_loader,
    anchors: torch.Tensor,
    classes: list,
    device: torch.device,
    gt_json_path: str,
    conf_thresh: float = 0.05,
    nms_thresh: float = 0.5,
    input_size: int = 512,
):
    """Run inference on val set and compute mAP@0.5."""
    predictions = run_inference(
        model, val_loader, anchors, classes, device,
        conf_thresh=conf_thresh, nms_thresh=nms_thresh, input_size=input_size,
    )
    map50, score = compute_map(predictions, gt_json_path)
    return map50, score, predictions


def training_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    anchors: torch.Tensor,
    assigner,
    criterion,
    classes: list,
    device: torch.device,
    gt_val_json: str,
    checkpoint_dir: str,
    epochs: int = 40,
    freeze_epochs: int = 3,
    conf_thresh: float = 0.05,
    nms_thresh: float = 0.5,
    input_size: int = 512,
    val_every: int = 1,
):
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    best_map = 0.0
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        if epoch == 1:
            model.freeze_backbone()
            print("Backbone frozen for first", freeze_epochs, "epochs")
        if epoch == freeze_epochs + 1:
            model.unfreeze_all()
            print("Backbone unfrozen")

        train_loss, train_info = train_one_epoch(
            model, train_loader, optimizer, scaler,
            anchors, assigner, criterion, device, epoch,
        )
        scheduler.step()

        print(
            f"[E{epoch:02d}] loss={train_loss:.4f}  "
            f"obj={train_info['obj']:.4f}  cls={train_info['cls']:.4f}  "
            f"box={train_info['box']:.4f}  pos={train_info['num_pos']:.1f}"
        )

        if epoch % val_every == 0:
            map50, score, _ = validate(
                model, val_loader, anchors, classes, device, gt_val_json,
                conf_thresh=conf_thresh, nms_thresh=nms_thresh, input_size=input_size,
            )
            per = score.get("per_class", {})
            per_str = "  ".join(f"{c}={per.get(c,{}).get('ap',0):.3f}" for c in classes)
            print(f"  → val mAP@0.5={map50:.4f}  [{per_str}]")

            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_map": best_map,
                "map50": map50,
            }
            save_checkpoint(state, os.path.join(checkpoint_dir, "last.pth"))

            if map50 > best_map:
                best_map = map50
                save_checkpoint(state, os.path.join(checkpoint_dir, "best.pth"))
                print(f"  ✓ New best mAP: {best_map:.4f}")

    print(f"\nTraining complete. Best val mAP@0.5 = {best_map:.4f}")
    return best_map
