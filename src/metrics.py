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
