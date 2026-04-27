#!/usr/bin/env python3
"""Generate representative qualitative segmentation figures for Section 5.2.

This script selects validation slices reproducibly (good, average, difficult,
and disagreement cases) and saves 4-panel qualitative figures:
FLAIR input, ground truth, U-Net prediction, and Residual U-Net prediction.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import BraTSDataset, split_train_val
from src.model_architecture import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate qualitative figures for thesis Section 5.2")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--checkpoint-unet", type=Path, default=Path("./models/best_unet.pt"))
    parser.add_argument("--checkpoint-resunet", type=Path, default=Path("./models/best_resunet.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("./figures/section5_2"))
    parser.add_argument("--max-patients", type=int, default=20)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    return parser.parse_args()


def remap_brats_labels(mask: torch.Tensor) -> torch.Tensor:
    """Remap BraTS labels {0,1,2,4} -> {0,1,2,3}."""

    remapped = mask.clone().long()
    remapped[remapped == 4] = 3
    return remapped


def per_class_dice(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    """Compute per-class Dice on integer mask tensors of shape (H, W)."""

    dices = []
    for cls in range(num_classes):
        pred_c = (pred == cls).float()
        tgt_c = (target == cls).float()
        inter = torch.sum(pred_c * tgt_c)
        denom = torch.sum(pred_c) + torch.sum(tgt_c)
        dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
        dices.append(dice)
    return torch.stack(dices)


def load_model(model_name: str, checkpoint_path: Path, device: torch.device, base_channels: int) -> torch.nn.Module:
    model = build_model(
        model_name,
        in_channels=4,
        num_classes=4,
        base_channels=base_channels,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


def save_four_panel_figure(
    output_path: Path,
    flair: np.ndarray,
    gt: np.ndarray,
    pred_unet: np.ndarray,
    pred_resunet: np.ndarray,
    title: str,
    cmap_masks: ListedColormap,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), constrained_layout=True)

    axes[0].imshow(flair, cmap="gray")
    axes[0].set_title("FLAIR Input")
    axes[1].imshow(gt, cmap=cmap_masks, vmin=0, vmax=3)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_unet, cmap=cmap_masks, vmin=0, vmax=3)
    axes[2].set_title("U-Net")
    axes[3].imshow(pred_resunet, cmap=cmap_masks, vmin=0, vmax=3)
    axes[3].set_title("Residual U-Net")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_confidence_figure(
    output_path: Path,
    flair: np.ndarray,
    prediction: np.ndarray,
    confidence: np.ndarray,
    title: str,
    cmap_masks: ListedColormap,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), constrained_layout=True)

    axes[0].imshow(flair, cmap="gray")
    axes[0].set_title("FLAIR Input")
    axes[1].imshow(prediction, cmap=cmap_masks, vmin=0, vmax=3)
    axes[1].set_title("Prediction")
    conf_im = axes[2].imshow(confidence, cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].set_title("Confidence Map")

    for ax in axes:
        ax.axis("off")

    fig.colorbar(conf_im, ax=axes[2], fraction=0.046, pad=0.04, label="Max softmax")
    fig.suptitle(title, fontsize=11)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def choose_unique(candidates: list[tuple[int, float]], used: set[int]) -> int:
    for idx, _ in candidates:
        if idx not in used:
            used.add(idx)
            return idx
    raise RuntimeError("Unable to choose a unique representative case.")


def main() -> None:
    args = parse_args()

    if args.max_patients < 2:
        raise ValueError("--max-patients must be >= 2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patient_ids = sorted([p.name for p in args.data_dir.iterdir() if p.is_dir() and p.name.startswith("BraTS")])
    patient_ids = patient_ids[: args.max_patients]

    dataset = BraTSDataset(
        data_dir=args.data_dir,
        patient_ids=patient_ids,
        target_size=args.target_size,
        augment=False,
        cache=True,
    )

    _, val_split = split_train_val(dataset, val_ratio=args.val_ratio, seed=args.seed, split_by_patient=True)
    val_indices = list(val_split.indices)

    model_unet = load_model("unet", args.checkpoint_unet, device, args.base_channels)
    model_resunet = load_model("resunet", args.checkpoint_resunet, device, args.base_channels)

    rows: list[dict[str, object]] = []
    cmap_masks = ListedColormap(["black", "red", "yellow", "green"])

    with torch.no_grad():
        for global_idx in val_indices:
            image, mask = dataset[global_idx]
            image_b = image.unsqueeze(0).to(device)
            gt = remap_brats_labels(mask.squeeze(0)).to(device)

            logits_unet = model_unet(image_b)
            logits_res = model_resunet(image_b)

            prob_unet = torch.softmax(logits_unet, dim=1).squeeze(0)
            pred_unet = torch.argmax(prob_unet, dim=0)
            pred_res = torch.argmax(logits_res, dim=1).squeeze(0)

            d_unet = per_class_dice(pred_unet.cpu(), gt.cpu())
            d_res = per_class_dice(pred_res.cpu(), gt.cpu())

            # Use tumor classes (1..3) for representative-case ranking.
            tumor_unet = float(d_unet[1:].mean().item())
            tumor_res = float(d_res[1:].mean().item())

            info = dataset.get_patient_info(global_idx)
            rows.append(
                {
                    "global_idx": global_idx,
                    "patient_id": info["patient_id"],
                    "z_index": int(info["z_index"]),
                    "dice_unet_tumor_mean": tumor_unet,
                    "dice_resunet_tumor_mean": tumor_res,
                    "dice_pair_mean": (tumor_unet + tumor_res) / 2.0,
                    "dice_gap_abs": abs(tumor_unet - tumor_res),
                    "confidence_mean_unet": float(prob_unet.max(dim=0).values.mean().item()),
                    "flair": image[3].cpu().numpy(),
                    "gt": gt.cpu().numpy(),
                    "pred_unet": pred_unet.cpu().numpy(),
                    "pred_res": pred_res.cpu().numpy(),
                    "conf_unet": prob_unet.max(dim=0).values.cpu().numpy(),
                }
            )

    if len(rows) < 4:
        raise RuntimeError("Not enough validation samples to select representative cases.")

    by_pair_desc = sorted([(i, r["dice_pair_mean"]) for i, r in enumerate(rows)], key=lambda x: x[1], reverse=True)
    by_pair_asc = sorted([(i, r["dice_pair_mean"]) for i, r in enumerate(rows)], key=lambda x: x[1])
    by_gap_desc = sorted([(i, r["dice_gap_abs"]) for i, r in enumerate(rows)], key=lambda x: x[1], reverse=True)

    pair_values = np.array([r["dice_pair_mean"] for r in rows], dtype=float)
    median_val = float(np.median(pair_values))
    by_median = sorted([(i, abs(r["dice_pair_mean"] - median_val)) for i, r in enumerate(rows)], key=lambda x: x[1])

    used: set[int] = set()
    idx_good = choose_unique(by_pair_desc, used)
    idx_avg = choose_unique(by_median, used)
    idx_diff = choose_unique(by_pair_asc, used)
    idx_gap = choose_unique(by_gap_desc, used)

    selected = [
        ("figure5_1_good_case", "Good case", idx_good),
        ("figure5_2_average_case", "Average case", idx_avg),
        ("figure5_3_difficult_case", "Difficult case", idx_diff),
        ("figure5_4_model_difference_case", "Model-difference case", idx_gap),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for stem, label, idx in selected:
        r = rows[idx]
        title = (
            f"{label} | {r['patient_id']} (z={r['z_index']}) | "
            f"Tumor Dice U-Net={r['dice_unet_tumor_mean']:.3f}, ResU-Net={r['dice_resunet_tumor_mean']:.3f}"
        )
        save_four_panel_figure(
            output_path=args.output_dir / f"{stem}.png",
            flair=r["flair"],
            gt=r["gt"],
            pred_unet=r["pred_unet"],
            pred_resunet=r["pred_res"],
            title=title,
            cmap_masks=cmap_masks,
        )

    # Optional confidence map: choose the lowest-confidence selected case.
    conf_case = min((rows[idx] for _, _, idx in selected), key=lambda x: x["confidence_mean_unet"])
    conf_title = (
        f"Confidence example | {conf_case['patient_id']} (z={conf_case['z_index']}) | "
        f"Mean confidence={conf_case['confidence_mean_unet']:.3f}"
    )
    save_confidence_figure(
        output_path=args.output_dir / "figure5_5_confidence_map.png",
        flair=conf_case["flair"],
        prediction=conf_case["pred_unet"],
        confidence=conf_case["conf_unet"],
        title=conf_title,
        cmap_masks=cmap_masks,
    )

    summary_path = args.output_dir / "qualitative_case_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "label",
            "figure",
            "patient_id",
            "z_index",
            "dice_unet_tumor_mean",
            "dice_resunet_tumor_mean",
            "dice_pair_mean",
            "dice_gap_abs",
            "confidence_mean_unet",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for stem, label, idx in selected:
            r = rows[idx]
            writer.writerow(
                {
                    "label": label,
                    "figure": f"{stem}.png",
                    "patient_id": r["patient_id"],
                    "z_index": r["z_index"],
                    "dice_unet_tumor_mean": f"{r['dice_unet_tumor_mean']:.6f}",
                    "dice_resunet_tumor_mean": f"{r['dice_resunet_tumor_mean']:.6f}",
                    "dice_pair_mean": f"{r['dice_pair_mean']:.6f}",
                    "dice_gap_abs": f"{r['dice_gap_abs']:.6f}",
                    "confidence_mean_unet": f"{r['confidence_mean_unet']:.6f}",
                }
            )

    print("Saved qualitative figures to:", args.output_dir)
    print("Saved case summary:", summary_path)


if __name__ == "__main__":
    main()
