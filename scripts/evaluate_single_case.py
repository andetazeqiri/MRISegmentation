#!/usr/bin/env python3
"""Evaluate trained BraTS checkpoints on a single case folder."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import BraTSDataset
from src.losses import DiceCrossEntropyLoss
from src.metrics import DEFAULT_CLASS_NAMES, compute_overlap_metrics_summary
from src.model_architecture import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model on one case.")
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "resunet"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def remap(mask: torch.Tensor) -> torch.Tensor:
    mapped = mask.clone().long()
    mapped[mapped == 4] = 3
    return mapped


def main() -> None:
    args = parse_args()

    if not args.case_dir.is_dir():
        raise FileNotFoundError(f"Case directory not found: {args.case_dir}")

    patient_ids = [args.case_dir.name]
    dataset = BraTSDataset(
        data_dir=args.case_dir.parent,
        patient_ids=patient_ids,
        target_size=args.target_size,
        augment=False,
        cache=True,
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(
        args.model,
        in_channels=4,
        num_classes=4,
        base_channels=args.base_channels,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    criterion = DiceCrossEntropyLoss(alpha=0.5, beta=0.5)
    total_loss = 0.0
    summed: dict[str, float] = {}

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = remap(masks.squeeze(1).to(device))

            logits = model(images)
            loss = criterion(logits, masks)
            metrics = compute_overlap_metrics_summary(logits, masks, num_classes=4)

            total_loss += float(loss.item())
            for key, value in metrics.items():
                summed[key] = summed.get(key, 0.0) + float(value)

    n_batches = max(1, len(loader))
    results: dict[str, float] = {
        "case": args.case_dir.name,
        "val_loss": total_loss / n_batches,
        "val_dice_mean": summed.get("dice_mean", 0.0) / n_batches,
        "val_iou_mean": summed.get("iou_mean", 0.0) / n_batches,
        "val_precision_mean": summed.get("precision_mean", 0.0) / n_batches,
        "val_recall_mean": summed.get("recall_mean", 0.0) / n_batches,
    }

    for name in DEFAULT_CLASS_NAMES.values():
        results[f"val_dice_{name}"] = summed.get(f"dice_{name}", 0.0) / n_batches
        results[f"val_iou_{name}"] = summed.get(f"iou_{name}", 0.0) / n_batches
        results[f"val_precision_{name}"] = summed.get(f"precision_{name}", 0.0) / n_batches
        results[f"val_recall_{name}"] = summed.get(f"recall_{name}", 0.0) / n_batches

    print(json.dumps(results, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results.keys()))
            writer.writeheader()
            writer.writerow(results)


if __name__ == "__main__":
    main()
