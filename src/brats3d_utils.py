#!/usr/bin/env python3
"""Utilities for 3D BraTS loading, splitting, patch sampling, and region Dice."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.preprocessing import collect_nii_files, load_volumes, normalize_modalities


@dataclass
class PatientCase3D:
    """Container for one patient's cropped 3D multi-modal volume and segmentation."""

    patient_id: str
    image: np.ndarray  # (C, D, H, W), float32
    label: np.ndarray  # (D, H, W), int64 with classes {0,1,2,3}


def _find_key_case_insensitive(dct: dict[str, np.ndarray], wanted_key: str) -> str:
    for key in dct.keys():
        if key.lower() == wanted_key.lower():
            return key
    raise KeyError(f"Key '{wanted_key}' not found. Available keys: {list(dct.keys())}")


def _crop_or_pad_center_hw(volume_hwd: np.ndarray, target_hw: int) -> np.ndarray:
    """Center-crop or zero-pad an HxWxD volume to target_hw x target_hw x D."""

    h, w, d = volume_hwd.shape
    out = np.zeros((target_hw, target_hw, d), dtype=volume_hwd.dtype)

    src_h_start = max(0, (h - target_hw) // 2)
    src_w_start = max(0, (w - target_hw) // 2)
    src_h_end = min(h, src_h_start + target_hw)
    src_w_end = min(w, src_w_start + target_hw)

    dst_h_start = max(0, (target_hw - h) // 2)
    dst_w_start = max(0, (target_hw - w) // 2)

    copy_h = src_h_end - src_h_start
    copy_w = src_w_end - src_w_start
    out[dst_h_start : dst_h_start + copy_h, dst_w_start : dst_w_start + copy_w, :] = volume_hwd[
        src_h_start:src_h_end,
        src_w_start:src_w_end,
        :,
    ]
    return out


def _pad_to_min_shape(
    image_cdhw: np.ndarray,
    label_dhw: np.ndarray,
    patch_size: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Pad image/label so each spatial dimension is at least patch_size (D,H,W)."""

    d, h, w = label_dhw.shape
    pd, ph, pw = patch_size

    pad_d = max(0, pd - d)
    pad_h = max(0, ph - h)
    pad_w = max(0, pw - w)

    if pad_d == 0 and pad_h == 0 and pad_w == 0:
        return image_cdhw, label_dhw

    image_padded = np.pad(
        image_cdhw,
        ((0, 0), (pad_d // 2, pad_d - pad_d // 2), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
        mode="constant",
    )
    label_padded = np.pad(
        label_dhw,
        ((pad_d // 2, pad_d - pad_d // 2), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
        mode="constant",
    )
    return image_padded, label_padded


def list_patient_dirs(data_dir: Path, max_patients: int | None = None) -> list[Path]:
    """List sorted BraTS/FeTS patient directories."""

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    patient_dirs = sorted(
        [p for p in data_dir.iterdir() if p.is_dir() and (p.name.startswith("BraTS") or p.name.startswith("FeTS"))]
    )
    if max_patients is not None:
        patient_dirs = patient_dirs[:max_patients]
    if not patient_dirs:
        raise ValueError(f"No BraTS/FeTS patient folders found in: {data_dir}")
    return patient_dirs


def split_patient_dirs(
    patient_dirs: list[Path],
    val_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """Split patient directories into train/val sets by patient."""

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1).")
    if len(patient_dirs) < 2:
        raise ValueError("Need at least 2 patient folders for train/val split.")

    shuffled = patient_dirs[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(shuffled) * val_ratio)))
    n_val = min(n_val, len(shuffled) - 1)
    val_dirs = sorted(shuffled[:n_val])
    train_dirs = sorted(shuffled[n_val:])
    return train_dirs, val_dirs


def load_patient_case(
    patient_dir: Path,
    modalities: tuple[str, str, str, str] = ("t1", "t1ce", "t2", "flair"),
    crop_hw: int = 128,
) -> PatientCase3D:
    """Load one patient into (C,D,H,W) image and (D,H,W) remapped label arrays."""

    modality_paths = collect_nii_files(patient_dir)
    volumes = load_volumes(modality_paths)
    normalize_modalities(volumes, list(modalities))

    channels_hwd: list[np.ndarray] = []
    for mod in modalities:
        key = _find_key_case_insensitive(volumes, mod)
        channels_hwd.append(_crop_or_pad_center_hw(volumes[key], crop_hw))

    seg_key = _find_key_case_insensitive(volumes, "seg")
    seg_hwd = _crop_or_pad_center_hw(volumes[seg_key], crop_hw)

    image_chwd = np.stack(channels_hwd, axis=0).astype(np.float32)
    # Convert (C,H,W,D) -> (C,D,H,W)
    image_cdhw = np.transpose(image_chwd, (0, 3, 1, 2))

    # Convert (H,W,D) -> (D,H,W)
    label_dhw = np.transpose(seg_hwd, (2, 0, 1)).astype(np.int64)
    label_dhw[label_dhw == 4] = 3

    return PatientCase3D(patient_id=patient_dir.name, image=image_cdhw, label=label_dhw)


def sample_patch(
    case: PatientCase3D,
    patch_size: tuple[int, int, int],
    rng: np.random.Generator,
    tumor_focus_prob: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one random (possibly tumor-focused) patch from a case."""

    image, label = _pad_to_min_shape(case.image, case.label, patch_size)
    d, h, w = label.shape
    pd, ph, pw = patch_size

    if rng.random() < tumor_focus_prob and np.any(label > 0):
        tumor_voxels = np.argwhere(label > 0)
        center_d, center_h, center_w = tumor_voxels[rng.integers(0, len(tumor_voxels))]
    else:
        center_d = int(rng.integers(0, d))
        center_h = int(rng.integers(0, h))
        center_w = int(rng.integers(0, w))

    start_d = int(np.clip(center_d - pd // 2, 0, d - pd))
    start_h = int(np.clip(center_h - ph // 2, 0, h - ph))
    start_w = int(np.clip(center_w - pw // 2, 0, w - pw))

    patch_image = image[:, start_d : start_d + pd, start_h : start_h + ph, start_w : start_w + pw]
    patch_label = label[start_d : start_d + pd, start_h : start_h + ph, start_w : start_w + pw]
    return patch_image, patch_label


def dice_binary(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """Dice for binary masks."""

    pred_bool = pred.astype(bool)
    tgt_bool = target.astype(bool)

    pred_sum = int(pred_bool.sum())
    tgt_sum = int(tgt_bool.sum())
    if pred_sum + tgt_sum == 0:
        return 1.0

    intersection = int(np.logical_and(pred_bool, tgt_bool).sum())
    return float((2.0 * intersection + eps) / (pred_sum + tgt_sum + eps))


def compute_brats_region_dice(pred_labels: np.ndarray, true_labels: np.ndarray) -> dict[str, float]:
    """Compute BraTS-style Dice metrics: WT, TC, ET."""

    pred = pred_labels.astype(np.int64)
    true = true_labels.astype(np.int64)

    pred_wt = pred > 0
    true_wt = true > 0

    pred_tc = np.logical_or(pred == 1, pred == 3)
    true_tc = np.logical_or(true == 1, true == 3)

    pred_et = pred == 3
    true_et = true == 3

    wt = dice_binary(pred_wt, true_wt)
    tc = dice_binary(pred_tc, true_tc)
    et = dice_binary(pred_et, true_et)
    return {
        "dice_wt": wt,
        "dice_tc": tc,
        "dice_et": et,
        "dice_mean_regions": float((wt + tc + et) / 3.0),
    }


def summarize_case_metrics(case_metrics: list[dict[str, float | str]]) -> dict[str, float]:
    """Average WT/TC/ET metrics over cases."""

    if not case_metrics:
        raise ValueError("case_metrics is empty")

    wt = [float(r["dice_wt"]) for r in case_metrics]
    tc = [float(r["dice_tc"]) for r in case_metrics]
    et = [float(r["dice_et"]) for r in case_metrics]
    mean_regions = [float(r["dice_mean_regions"]) for r in case_metrics]

    return {
        "n_cases": len(case_metrics),
        "dice_wt_mean": float(np.mean(wt)),
        "dice_tc_mean": float(np.mean(tc)),
        "dice_et_mean": float(np.mean(et)),
        "dice_mean_regions": float(np.mean(mean_regions)),
    }
