# Object Detection — XLA Final

Self-implemented anchor-based one-stage object detector.  
Architecture: **ResNet18 (pretrained) + FPN + per-anchor detection head**.

## Classes

`person`, `car`, `dog`, `cat`, `chair`

## Environment Setup

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0.

## Training

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data   ./public/annotations/val.json \
  --image_dir  ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/
```

Key optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 40 | Total training epochs |
| `--batch_size` | 16 | Batch size |
| `--lr` | 3e-4 | Initial learning rate |
| `--input_size` | 512 | Network input size |
| `--workers` | 4 | DataLoader workers |
| `--freeze_epochs` | 3 | Freeze backbone for first N epochs |
| `--conf_thresh` | 0.05 | Confidence threshold during validation |
| `--nms_thresh` | 0.5 | NMS IoU threshold |
| `--no_kmeans` | — | Skip k-means, use default anchors |

The best checkpoint (by val mAP\@0.5) is saved to `./models/best.pth`.  
Anchor configuration is saved to `./models/anchors.json`.

## Inference

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json
```

Key optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | `./models/best.pth` | Model weights |
| `--anchor_cfg` | `./models/anchors.json` | Anchor config (generated during training) |
| `--conf_thresh` | 0.05 | Confidence threshold |
| `--nms_thresh` | 0.5 | NMS IoU threshold |

Every image in `--image_dir` will have an entry in `predictions.json`.  
Images with no detections will have `"boxes": []`.

## Evaluation

```bash
python public/tools/evaluate_predictions.py \
  --ground_truth public/annotations/val.json \
  --predictions  predictions.json \
  --output       val_score.json
```

## Model Weights

- `./models/best.pth` — best checkpoint by validation mAP\@0.5
- `./models/last.pth` — checkpoint from the last epoch
- `./models/anchors.json` — anchor sizes computed by k-means

## Architecture Summary

```
ResNet18 (pretrained ImageNet)
  → C3 (stride 8,  128 ch)
  → C4 (stride 16, 256 ch)
  → C5 (stride 32, 512 ch)

FPN (top-down + lateral)
  → P3, P4, P5 (all 256 ch)

Shared Detection Head (3 conv layers)
  → Objectness branch : sigmoid(obj) × softmax(cls)[class]
  → Classification branch
  → Box-delta branch : GIoU + SmoothL1

Anchors: 9 total (3 per FPN level), generated with k-means on training data
Input: 512×512 letterbox with aspect-ratio preservation
NMS: per-class
```
