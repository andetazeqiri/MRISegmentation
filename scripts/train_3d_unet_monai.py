#!/usr/bin/env python3
"""Train a patch-based 3D U-Net (MONAI) on BraTS volumes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.brats3d_utils import (
    PatientCase3D,
    compute_brats_region_dice,
    list_patient_dirs,
    load_patient_case,
    sample_patch,
    split_patient_dirs,
    summarize_case_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MONAI 3D U-Net with patch-based sampling.")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--save-dir", type=Path, default=Path("./models"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--checkpoint-name", type=str, default="best_unet3d_monai.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-patients", type=int, default=20)
    parser.add_argument("--crop-hw", type=int, default=128)
    parser.add_argument("--patch-size", type=str, default="64,128,128", help="D,H,W")
    parser.add_argument("--channels", type=str, default="16,32,64,128,256")
    parser.add_argument("--strides", type=str, default="2,2,2,2")
    parser.add_argument("--tumor-focus-prob", type=float, default=0.7)
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--sw-overlap", type=float, default=0.25)
    parser.add_argument("--history-csv", type=Path, default=Path("./outputs/training_history_3d_unet.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("./outputs/eval_3d_unet_brats2020.json"))
    return parser.parse_args()


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def _require_monai() -> tuple[object, object]:
    try:
        from monai.inferers import sliding_window_inference
        from monai.losses import DiceCELoss
        from monai.networks.nets import UNet
    except ImportError as exc:
        raise ImportError("MONAI is required. Install with: pip install monai") from exc
    return UNet, DiceCELoss, sliding_window_inference


class RandomPatchDataset(Dataset):
    """Random patch sampler over preloaded 3D cases."""

    def __init__(
        self,
        cases: list[PatientCase3D],
        patch_size: tuple[int, int, int],
        samples_per_epoch: int,
        tumor_focus_prob: float,
        seed: int,
    ):
        self.cases = cases
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.tumor_focus_prob = tumor_focus_prob
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        _ = idx
        case = self.cases[int(self.rng.integers(0, len(self.cases)))]
        patch_image, patch_label = sample_patch(
            case,
            patch_size=self.patch_size,
            rng=self.rng,
            tumor_focus_prob=self.tumor_focus_prob,
        )
        image_t = torch.from_numpy(patch_image).float()
        label_t = torch.from_numpy(patch_label).long()
        return image_t, label_t


@torch.no_grad()
def evaluate_3d_cases(
    model: torch.nn.Module,
    cases: list[PatientCase3D],
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
    device: torch.device,
    sliding_window_inference,
) -> tuple[list[dict[str, float | str]], dict[str, float]]:
    model.eval()
    case_metrics: list[dict[str, float | str]] = []

    for case in cases:
        image = torch.from_numpy(case.image).unsqueeze(0).float().to(device)
        logits = sliding_window_inference(
            inputs=image,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
        )
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)
        metrics = compute_brats_region_dice(pred, case.label)
        metrics["patient_id"] = case.patient_id
        case_metrics.append(metrics)

    summary = summarize_case_metrics(case_metrics)
    return case_metrics, summary


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    UNet, DiceCELoss, sliding_window_inference = _require_monai()

    patch_size = _parse_int_tuple(args.patch_size)
    channels = _parse_int_tuple(args.channels)
    strides = _parse_int_tuple(args.strides)
    if len(channels) != len(strides) + 1:
        raise ValueError("channels length must be strides length + 1")

    patient_dirs = list_patient_dirs(args.data_dir, max_patients=args.max_patients)
    train_dirs, val_dirs = split_patient_dirs(patient_dirs, val_ratio=args.val_ratio, seed=args.seed)

    print(f"Loading train cases: {len(train_dirs)}")
    train_cases = [load_patient_case(p, crop_hw=args.crop_hw) for p in train_dirs]
    print(f"Loading val cases: {len(val_dirs)}")
    val_cases = [load_patient_case(p, crop_hw=args.crop_hw) for p in val_dirs]

    train_ds = RandomPatchDataset(
        train_cases,
        patch_size=patch_size,
        samples_per_epoch=args.steps_per_epoch,
        tumor_focus_prob=args.tumor_focus_prob,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=channels,
        strides=strides,
        num_res_units=2,
    ).to(device)

    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = args.save_dir / args.checkpoint_name

    best_val = float("-inf")
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for image_t, label_t in train_loader:
            image_t = image_t.to(device)
            label_t = label_t.to(device)

            optimizer.zero_grad()
            logits = model(image_t)
            loss = criterion(logits, label_t.unsqueeze(1))
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

        train_loss = running_loss / max(1, len(train_loader))
        _, val_summary = evaluate_3d_cases(
            model=model,
            cases=val_cases,
            roi_size=patch_size,
            sw_batch_size=args.sw_batch_size,
            overlap=args.sw_overlap,
            device=device,
            sliding_window_inference=sliding_window_inference,
        )

        val_score = float(val_summary["dice_mean_regions"])
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_dice_wt": float(val_summary["dice_wt_mean"]),
                "val_dice_tc": float(val_summary["dice_tc_mean"]),
                "val_dice_et": float(val_summary["dice_et_mean"]),
                "val_dice_mean_regions": val_score,
            }
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_wt={val_summary['dice_wt_mean']:.4f} | "
            f"val_tc={val_summary['dice_tc_mean']:.4f} | "
            f"val_et={val_summary['dice_et_mean']:.4f}"
        )

        if val_score > best_val:
            best_val = val_score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_val_dice_mean_regions": best_val,
                    "channels": channels,
                    "strides": strides,
                    "patch_size": patch_size,
                    "crop_hw": args.crop_hw,
                    "train_patient_ids": [p.name for p in train_dirs],
                    "val_patient_ids": [p.name for p in val_dirs],
                },
                best_ckpt,
            )
            print(f"Saved new best 3D checkpoint: {best_ckpt}")

    args.history_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.history_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_dice_wt", "val_dice_tc", "val_dice_et", "val_dice_mean_regions"],
        )
        writer.writeheader()
        writer.writerows(history_rows)

    checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    case_metrics, summary = evaluate_3d_cases(
        model=model,
        cases=val_cases,
        roi_size=patch_size,
        sw_batch_size=args.sw_batch_size,
        overlap=args.sw_overlap,
        device=device,
        sliding_window_inference=sliding_window_inference,
    )

    payload = {
        "model": "unet3d_monai",
        "summary": summary,
        "cases": case_metrics,
        "val_patient_ids": [p.name for p in val_dirs],
        "checkpoint": str(best_ckpt),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved 3D evaluation summary to: {args.summary_json}")


if __name__ == "__main__":
    main()
