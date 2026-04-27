#!/usr/bin/env python3
"""Evaluate a trained MONAI 3D U-Net and optionally compare against 2D U-Net."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.brats3d_utils import compute_brats_region_dice, list_patient_dirs, load_patient_case, split_patient_dirs, summarize_case_metrics
from src.model_architecture import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 3D U-Net and compare with optional 2D baseline.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to 3D MONAI checkpoint")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--max-patients", type=int, default=20)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-hw", type=int, default=128)
    parser.add_argument("--patch-size", type=str, default=None, help="Optional override D,H,W")
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--sw-overlap", type=float, default=0.25)
    parser.add_argument("--baseline-2d-checkpoint", type=Path, default=None)
    parser.add_argument("--baseline-2d-model", type=str, default="unet", choices=["unet", "resunet"])
    parser.add_argument("--baseline-2d-batch-size", type=int, default=16)
    parser.add_argument("--output-json", type=Path, default=Path("./outputs/compare_2d_3d_brats_regions.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("./outputs/compare_2d_3d_brats_regions.csv"))
    return parser.parse_args()


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def _require_monai() -> tuple[object, object]:
    try:
        from monai.inferers import sliding_window_inference
        from monai.networks.nets import UNet
    except ImportError as exc:
        raise ImportError("MONAI is required. Install with: pip install monai") from exc
    return UNet, sliding_window_inference


@torch.no_grad()
def predict_3d_labels(
    model: torch.nn.Module,
    image_cdhw: np.ndarray,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
    device: torch.device,
    sliding_window_inference,
) -> np.ndarray:
    model.eval()
    image_t = torch.from_numpy(image_cdhw).unsqueeze(0).float().to(device)
    logits = sliding_window_inference(
        inputs=image_t,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap,
    )
    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)
    return pred


@torch.no_grad()
def predict_2d_labels(
    model: torch.nn.Module,
    image_cdhw: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    # (C,D,H,W) -> (D,C,H,W)
    slices = np.transpose(image_cdhw, (1, 0, 2, 3)).astype(np.float32)
    preds: list[np.ndarray] = []
    n_slices = slices.shape[0]

    for start in range(0, n_slices, batch_size):
        end = min(start + batch_size, n_slices)
        batch = torch.from_numpy(slices[start:end]).to(device)
        logits = model(batch)
        pred = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)
        preds.append(pred)

    return np.concatenate(preds, axis=0)


def main() -> None:
    args = parse_args()
    UNet, sliding_window_inference = _require_monai()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    channels = tuple(ckpt.get("channels", (16, 32, 64, 128, 256)))
    strides = tuple(ckpt.get("strides", (2, 2, 2, 2)))
    roi_size = tuple(ckpt.get("patch_size", (64, 128, 128)))
    if args.patch_size is not None:
        roi_size = _parse_int_tuple(args.patch_size)

    crop_hw = int(ckpt.get("crop_hw", args.crop_hw))
    model3d = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=channels,
        strides=strides,
        num_res_units=2,
    ).to(device)
    model3d.load_state_dict(ckpt["model_state_dict"])

    patient_dirs = list_patient_dirs(args.data_dir, max_patients=args.max_patients)
    train_dirs, val_dirs = split_patient_dirs(patient_dirs, val_ratio=args.val_ratio, seed=args.seed)
    if "val_patient_ids" in ckpt and ckpt["val_patient_ids"]:
        val_id_set = set(ckpt["val_patient_ids"])
        val_dirs = sorted([p for p in patient_dirs if p.name in val_id_set])
        train_dirs = sorted([p for p in patient_dirs if p.name not in val_id_set])

    _ = train_dirs
    cases = [load_patient_case(p, crop_hw=crop_hw) for p in val_dirs]

    summary_3d_rows: list[dict[str, float | str]] = []
    for case in cases:
        pred3d = predict_3d_labels(
            model=model3d,
            image_cdhw=case.image,
            roi_size=roi_size,
            sw_batch_size=args.sw_batch_size,
            overlap=args.sw_overlap,
            device=device,
            sliding_window_inference=sliding_window_inference,
        )
        metrics = compute_brats_region_dice(pred3d, case.label)
        metrics["patient_id"] = case.patient_id
        summary_3d_rows.append(metrics)

    summary_3d = summarize_case_metrics(summary_3d_rows)

    baseline_rows: list[dict[str, float | str]] | None = None
    baseline_summary: dict[str, float] | None = None
    if args.baseline_2d_checkpoint is not None:
        model2d = build_model(args.baseline_2d_model, in_channels=4, num_classes=4, base_channels=32).to(device)
        ckpt2d = torch.load(args.baseline_2d_checkpoint, map_location=device, weights_only=False)
        state_dict2d = ckpt2d["model_state_dict"] if isinstance(ckpt2d, dict) and "model_state_dict" in ckpt2d else ckpt2d
        model2d.load_state_dict(state_dict2d)

        baseline_rows = []
        for case in cases:
            pred2d = predict_2d_labels(
                model=model2d,
                image_cdhw=case.image,
                batch_size=args.baseline_2d_batch_size,
                device=device,
            )
            metrics = compute_brats_region_dice(pred2d, case.label)
            metrics["patient_id"] = case.patient_id
            baseline_rows.append(metrics)

        baseline_summary = summarize_case_metrics(baseline_rows)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "patient_id", "dice_wt", "dice_tc", "dice_et", "dice_mean_regions"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_3d_rows:
            writer.writerow(
                {
                    "model": "3d_unet_monai",
                    "patient_id": row["patient_id"],
                    "dice_wt": row["dice_wt"],
                    "dice_tc": row["dice_tc"],
                    "dice_et": row["dice_et"],
                    "dice_mean_regions": row["dice_mean_regions"],
                }
            )
        if baseline_rows is not None:
            for row in baseline_rows:
                writer.writerow(
                    {
                        "model": f"2d_{args.baseline_2d_model}",
                        "patient_id": row["patient_id"],
                        "dice_wt": row["dice_wt"],
                        "dice_tc": row["dice_tc"],
                        "dice_et": row["dice_et"],
                        "dice_mean_regions": row["dice_mean_regions"],
                    }
                )

    payload: dict[str, object] = {
        "val_patient_ids": [p.name for p in val_dirs],
        "3d_unet_monai": {
            "summary": summary_3d,
            "cases": summary_3d_rows,
            "checkpoint": str(args.checkpoint),
        },
    }
    if baseline_summary is not None and baseline_rows is not None:
        payload[f"2d_{args.baseline_2d_model}"] = {
            "summary": baseline_summary,
            "cases": baseline_rows,
            "checkpoint": str(args.baseline_2d_checkpoint),
        }
        payload["delta_3d_minus_2d"] = {
            "dice_wt": float(summary_3d["dice_wt_mean"] - baseline_summary["dice_wt_mean"]),
            "dice_tc": float(summary_3d["dice_tc_mean"] - baseline_summary["dice_tc_mean"]),
            "dice_et": float(summary_3d["dice_et_mean"] - baseline_summary["dice_et_mean"]),
            "dice_mean_regions": float(summary_3d["dice_mean_regions"] - baseline_summary["dice_mean_regions"]),
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("3D summary:")
    print(json.dumps(summary_3d, indent=2))
    if baseline_summary is not None:
        print("2D baseline summary:")
        print(json.dumps(baseline_summary, indent=2))
    print(f"Saved case-level comparison CSV to: {args.output_csv}")
    print(f"Saved JSON summary to: {args.output_json}")


if __name__ == "__main__":
    main()
