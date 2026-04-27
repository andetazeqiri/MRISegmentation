#!/usr/bin/env python3
"""Compute per-patient Dice scores and a paired Wilcoxon test for two checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader, Subset

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import BraTSDataset, split_train_val
from src.metrics import compute_dice_summary
from src.model_architecture import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-patient Dice scores and Wilcoxon test.")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--patient-prefix", type=str, default="BraTS")
    parser.add_argument("--max-patients", type=int, default=20)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--unet-checkpoint", type=Path, default=Path("./models/best_unet.pt"))
    parser.add_argument("--resunet-checkpoint", type=Path, default=Path("./models/best_resunet.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--output-prefix", type=str, default="patient_level_wilcoxon_brats2020")
    return parser.parse_args()


def remap(mask: torch.Tensor) -> torch.Tensor:
    mapped = mask.clone().long()
    mapped[mapped == 4] = 3
    return mapped


def load_model(model_name: str, checkpoint_path: Path, base_channels: int, device: torch.device) -> torch.nn.Module:
    model = build_model(model_name, in_channels=4, num_classes=4, base_channels=base_channels).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_patient(
    model: torch.nn.Module,
    dataset: BraTSDataset,
    slice_indices: list[int],
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    subset = Subset(dataset, slice_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    logits_parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []

    with torch.no_grad():
        for images, masks in loader:
            logits = model(images.to(device))
            logits_parts.append(logits.cpu())
            mask_parts.append(remap(masks.squeeze(1)).cpu())

    logits_full = torch.cat(logits_parts, dim=0)
    masks_full = torch.cat(mask_parts, dim=0)
    mean_dice, per_class = compute_dice_summary(logits_full, masks_full, num_classes=4)

    return {
        "mean_dice": float(mean_dice),
        "dice_background": float(per_class["background"]),
        "dice_necrotic_non_enhancing": float(per_class["necrotic_non_enhancing"]),
        "dice_edema": float(per_class["edema"]),
        "dice_enhancing": float(per_class["enhancing"]),
    }


def main() -> None:
    args = parse_args()

    if args.max_patients < 2:
        raise ValueError("--max-patients must be >= 2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patient_ids = sorted(
        [p.name for p in args.data_dir.iterdir() if p.is_dir() and p.name.startswith(args.patient_prefix)]
    )[: args.max_patients]

    dataset = BraTSDataset(
        data_dir=args.data_dir,
        patient_ids=patient_ids,
        target_size=args.target_size,
        augment=False,
        cache=True,
    )
    _, val_split = split_train_val(dataset, val_ratio=args.val_ratio, seed=args.seed, split_by_patient=True)

    patient_slice_indices: dict[int, list[int]] = defaultdict(list)
    for global_idx in val_split.indices:
        patient_idx, _ = dataset.slice_mapping[global_idx]
        patient_slice_indices[patient_idx].append(global_idx)

    patient_rows: list[dict[str, object]] = []
    unet = load_model("unet", args.unet_checkpoint, args.base_channels, device)
    resunet = load_model("resunet", args.resunet_checkpoint, args.base_channels, device)

    unet_patient_dice: list[float] = []
    resunet_patient_dice: list[float] = []

    for patient_idx in sorted(patient_slice_indices):
        global_indices = patient_slice_indices[patient_idx]
        info = dataset.get_patient_info(global_indices[0])
        unet_scores = evaluate_patient(unet, dataset, global_indices, args.batch_size, device)
        resunet_scores = evaluate_patient(resunet, dataset, global_indices, args.batch_size, device)

        unet_patient_dice.append(unet_scores["mean_dice"])
        resunet_patient_dice.append(resunet_scores["mean_dice"])

        patient_rows.append(
            {
                "patient_id": info["patient_id"],
                "num_slices": len(global_indices),
                "unet_mean_dice": unet_scores["mean_dice"],
                "resunet_mean_dice": resunet_scores["mean_dice"],
                "difference": unet_scores["mean_dice"] - resunet_scores["mean_dice"],
            }
        )

    stat, p_value = wilcoxon(unet_patient_dice, resunet_patient_dice)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.output_prefix}.csv"
    json_path = args.output_dir / f"{args.output_prefix}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(patient_rows[0].keys()))
        writer.writeheader()
        writer.writerows(patient_rows)

    summary = {
        "n_patients": len(patient_rows),
        "wilcoxon_statistic": float(stat),
        "wilcoxon_p_value": float(p_value),
        "unet_patient_dice": unet_patient_dice,
        "resunet_patient_dice": resunet_patient_dice,
        "mean_unet_patient_dice": float(sum(unet_patient_dice) / len(unet_patient_dice)),
        "mean_resunet_patient_dice": float(sum(resunet_patient_dice) / len(resunet_patient_dice)),
    }

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved patient-level scores to {csv_path}")
    print(f"Wilcoxon statistic: {stat:.6f}")
    print(f"Wilcoxon p-value: {p_value:.6f}")
    print(f"Mean patient Dice - U-Net: {summary['mean_unet_patient_dice']:.6f}")
    print(f"Mean patient Dice - Residual U-Net: {summary['mean_resunet_patient_dice']:.6f}")


if __name__ == "__main__":
    main()
