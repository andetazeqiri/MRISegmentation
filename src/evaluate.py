#!/usr/bin/env python3
"""Evaluate a trained segmentation model on BraTS validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset

if __package__ is None or __package__ == "":
    # Allow `python3 src/evaluate.py` execution from repo root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import BraTSDataset, split_train_val
from src.losses import DiceCrossEntropyLoss
from src.metrics import DEFAULT_CLASS_NAMES, compute_dice_summary
from src.model_architecture import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "resunet"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-json", type=Path, default=None, help="Optional path to save metrics JSON")
    return parser.parse_args()


def remap_brats_labels(mask: torch.Tensor) -> torch.Tensor:
    """Remap BraTS labels {0,1,2,4} -> {0,1,2,3}."""

    remapped = mask.clone().long()
    remapped[remapped == 4] = 3
    return remapped


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: DiceCrossEntropyLoss,
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_per_class = {name: 0.0 for name in DEFAULT_CLASS_NAMES.values()}

    for images, masks in loader:
        images = images.to(device)
        masks = remap_brats_labels(masks.squeeze(1).to(device))

        logits = model(images)
        loss = criterion(logits, masks)
        mean_dice, per_class = compute_dice_summary(logits, masks, num_classes)

        total_loss += loss.item()
        total_dice += mean_dice
        for name, value in per_class.items():
            if name in total_per_class:
                total_per_class[name] += value

    n_batches = max(1, len(loader))
    results: dict[str, float] = {
        "val_loss": total_loss / n_batches,
        "val_dice_mean": total_dice / n_batches,
    }
    for name in total_per_class:
        results[f"val_dice_{name}"] = total_per_class[name] / n_batches

    return results


def main() -> None:
    args = parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_eval = BraTSDataset(
        data_dir=args.data_dir,
        target_size=args.target_size,
        augment=False,
        cache=True,
    )

    if args.max_patients is not None:
        if args.max_patients < 2:
            raise ValueError("--max-patients must be >= 2 for train/val split.")
        available_ids = [p[0].name for p in dataset_eval.patients]
        selected_ids = available_ids[: args.max_patients]
        dataset_eval = BraTSDataset(
            data_dir=args.data_dir,
            patient_ids=selected_ids,
            target_size=args.target_size,
            augment=False,
            cache=True,
        )

    _, val_split_eval = split_train_val(
        dataset_eval,
        val_ratio=args.val_ratio,
        seed=args.seed,
        split_by_patient=True,
    )

    val_set = Subset(dataset_eval, val_split_eval.indices)
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    num_classes = 4
    model = build_model(
        args.model,
        in_channels=4,
        num_classes=num_classes,
        base_channels=args.base_channels,
    ).to(device)

    # Local checkpoints include metadata; on newer PyTorch versions this requires
    # weights_only=False to allow trusted non-tensor objects (e.g., serialized args).
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        # Allow evaluation of plain state-dict checkpoints.
        state_dict = ckpt
    else:
        raise TypeError("Unsupported checkpoint format. Expected dict-like checkpoint.")

    model.load_state_dict(state_dict)

    criterion = DiceCrossEntropyLoss(alpha=0.5, beta=0.5)
    results = evaluate(model, val_loader, criterion, device, num_classes)

    print("Validation results:")
    print(f"  val_loss: {results['val_loss']:.4f}")
    print(f"  val_dice_mean: {results['val_dice_mean']:.4f}")
    print(f"  val_dice_necrotic_non_enhancing: {results['val_dice_necrotic_non_enhancing']:.4f}")
    print(f"  val_dice_edema: {results['val_dice_edema']:.4f}")
    print(f"  val_dice_enhancing: {results['val_dice_enhancing']:.4f}")

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved metrics JSON to: {args.save_json}")


if __name__ == "__main__":
    main()
