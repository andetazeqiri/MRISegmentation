#!/usr/bin/env python3
"""Evaluate trained BraTS models on a small FeTS2021 subset."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import BraTSDataset, split_train_val
from src.losses import DiceCrossEntropyLoss
from src.metrics import DEFAULT_CLASS_NAMES, compute_overlap_metrics_summary
from src.model_architecture import build_model


def remap(mask: torch.Tensor) -> torch.Tensor:
    mapped = mask.clone().long()
    mapped[mapped == 4] = 3
    return mapped


def evaluate_model(
    model_name: str,
    checkpoint_path: Path,
    batch_size: int,
    dataset: BraTSDataset,
    val_indices: list[int],
    device: torch.device,
) -> dict[str, float]:
    val_set = Subset(dataset, val_indices)
    loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(model_name, in_channels=4, num_classes=4, base_channels=32).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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

    return results


def save_metrics(prefix: str, metrics: dict[str, float]) -> None:
    out_json = Path(f"models/{prefix}.json")
    out_csv = Path(f"models/{prefix}.csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


if __name__ == "__main__":
    root = Path("MICCAI_FeTS2021_TrainingData")
    patient_ids = sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("FeTS")])[:20]

    dataset = BraTSDataset(
        data_dir=root,
        patient_ids=patient_ids,
        target_size=128,
        augment=False,
        cache=True,
    )
    _, val_split = split_train_val(dataset, val_ratio=0.2, seed=42, split_by_patient=True)
    val_indices = list(val_split.indices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet_metrics = evaluate_model(
        model_name="unet",
        checkpoint_path=Path("models/best_unet.pt"),
        batch_size=8,
        dataset=dataset,
        val_indices=val_indices,
        device=device,
    )
    save_metrics("eval_fets2021_unet_20", unet_metrics)

    resunet_metrics = evaluate_model(
        model_name="resunet",
        checkpoint_path=Path("models/best_resunet.pt"),
        batch_size=4,
        dataset=dataset,
        val_indices=val_indices,
        device=device,
    )
    save_metrics("eval_fets2021_resunet_20", resunet_metrics)

    print("FeTS2021 subset evaluation complete.")
    print("UNet val_dice_mean:", f"{unet_metrics['val_dice_mean']:.4f}")
    print("ResUNet val_dice_mean:", f"{resunet_metrics['val_dice_mean']:.4f}")
