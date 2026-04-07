#!/usr/bin/env python3
"""Segmentation metrics helpers for training and validation loops."""

from __future__ import annotations

import torch

from src.losses import dice_score_per_class


DEFAULT_CLASS_NAMES = {
    0: "background",
    1: "necrotic_non_enhancing",
    2: "edema",
    3: "enhancing",
}


def compute_dice_summary(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    class_names: dict[int, str] | None = None,
) -> tuple[float, dict[str, float]]:
    """Return mean Dice and per-class Dice from logits and class-index target."""

    names = class_names or DEFAULT_CLASS_NAMES
    dice_vec = dice_score_per_class(logits, target, num_classes=num_classes)

    per_class: dict[str, float] = {}
    for class_idx in range(num_classes):
        class_name = names.get(class_idx, f"class_{class_idx}")
        per_class[class_name] = float(dice_vec[class_idx].item())

    mean_dice = float(dice_vec.mean().item())
    return mean_dice, per_class


def _confusion_per_class(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (tp, fp, fn, tn) per class for multi-class segmentation."""

    pred = torch.argmax(logits, dim=1)

    pred_oh = torch.nn.functional.one_hot(pred.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    tgt_oh = torch.nn.functional.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    tp = torch.sum(pred_oh * tgt_oh, dim=dims)
    fp = torch.sum(pred_oh * (1.0 - tgt_oh), dim=dims)
    fn = torch.sum((1.0 - pred_oh) * tgt_oh, dim=dims)
    tn = torch.sum((1.0 - pred_oh) * (1.0 - tgt_oh), dim=dims)

    # Keep same dtype and stabilize downstream division.
    return tp + smooth * 0.0, fp + smooth * 0.0, fn + smooth * 0.0, tn + smooth * 0.0


def iou_score_per_class(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Compute IoU/Jaccard per class from argmax predictions."""

    tp, fp, fn, _ = _confusion_per_class(logits, target, num_classes, smooth=smooth)
    return (tp + smooth) / (tp + fp + fn + smooth)


def precision_per_class(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Compute precision per class from argmax predictions."""

    tp, fp, _, _ = _confusion_per_class(logits, target, num_classes, smooth=smooth)
    return (tp + smooth) / (tp + fp + smooth)


def recall_per_class(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Compute recall per class from argmax predictions."""

    tp, _, fn, _ = _confusion_per_class(logits, target, num_classes, smooth=smooth)
    return (tp + smooth) / (tp + fn + smooth)


def compute_overlap_metrics_summary(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    class_names: dict[int, str] | None = None,
) -> dict[str, float]:
    """Compute mean/per-class Dice, IoU, precision, and recall."""

    names = class_names or DEFAULT_CLASS_NAMES

    dice_vec = dice_score_per_class(logits, target, num_classes=num_classes)
    iou_vec = iou_score_per_class(logits, target, num_classes=num_classes)
    prec_vec = precision_per_class(logits, target, num_classes=num_classes)
    rec_vec = recall_per_class(logits, target, num_classes=num_classes)

    out: dict[str, float] = {
        "dice_mean": float(dice_vec.mean().item()),
        "iou_mean": float(iou_vec.mean().item()),
        "precision_mean": float(prec_vec.mean().item()),
        "recall_mean": float(rec_vec.mean().item()),
    }

    for class_idx in range(num_classes):
        cls = names.get(class_idx, f"class_{class_idx}")
        out[f"dice_{cls}"] = float(dice_vec[class_idx].item())
        out[f"iou_{cls}"] = float(iou_vec[class_idx].item())
        out[f"precision_{cls}"] = float(prec_vec[class_idx].item())
        out[f"recall_{cls}"] = float(rec_vec[class_idx].item())

    return out
