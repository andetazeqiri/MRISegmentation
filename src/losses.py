#!/usr/bin/env python3
"""Loss functions for brain tumor segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot_encode(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert class-index tensor (B, H, W) to one-hot tensor (B, C, H, W)."""

    return F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()


class DiceLoss(nn.Module):
    """Multi-class Dice loss computed from softmax probabilities."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            logits: Raw logits with shape (B, C, H, W)
            target: Class index target with shape (B, H, W)
        """
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        target_one_hot = one_hot_encode(target, num_classes)

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_one_hot, dim=dims)
        denominator = torch.sum(probs + target_one_hot, dim=dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice_per_class.mean()


class DiceCrossEntropyLoss(nn.Module):
    """Composite loss: alpha * CrossEntropy + beta * DiceLoss."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, target.long())
        dice_loss = self.dice(logits, target)
        return self.alpha * ce_loss + self.beta * dice_loss


def dice_score_per_class(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Compute Dice score per class from argmax predictions."""

    pred = torch.argmax(logits, dim=1)
    pred_one_hot = one_hot_encode(pred, num_classes)
    target_one_hot = one_hot_encode(target, num_classes)

    dims = (0, 2, 3)
    intersection = torch.sum(pred_one_hot * target_one_hot, dim=dims)
    denominator = torch.sum(pred_one_hot + target_one_hot, dim=dims)

    return (2.0 * intersection + smooth) / (denominator + smooth)
