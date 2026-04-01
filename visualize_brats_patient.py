#!/usr/bin/env python3
"""Load one BraTS patient and visualize MRI modalities and segmentation."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing import (
    collect_nii_files,
    combine_modalities,
    extract_tumor_slices,
    find_patient_dir,
    load_volumes,
    normalize_modalities,
    resize_slices_to_target,
)

def print_shapes(patient_dir: Path, modality_paths: dict[str, Path], volumes: dict[str, np.ndarray]) -> None:
    print(f"Patient: {patient_dir.name}")
    print("Loaded files and shapes:")
    for modality in sorted(volumes.keys()):
        print(f"  {modality:>5}: {modality_paths[modality].name} -> {volumes[modality].shape}")


def plot_modalities(volumes: dict[str, np.ndarray], slice_idx: int, patient_name: str) -> None:
    preferred_order = ["flair", "t1", "t1ce", "t2", "seg"]
    ordered_modalities = [m for m in preferred_order if m in volumes]
    ordered_modalities.extend([m for m in sorted(volumes.keys()) if m not in ordered_modalities])

    n = len(ordered_modalities)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for i, modality in enumerate(ordered_modalities):
        ax = axes[i]
        img = volumes[modality][:, :, slice_idx]

        if modality == "seg":
            ax.imshow(np.rot90(img), cmap="nipy_spectral", interpolation="nearest")
        else:
            ax.imshow(np.rot90(img), cmap="gray")

        ax.set_title(modality.upper())
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{patient_name} | slice z={slice_idx}", fontsize=14)
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load one BraTS patient and visualize all MRI modalities and segmentation mask."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path containing BraTS patient directories.",
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help="Patient folder name (e.g., BraTS20_Training_001). If omitted, first patient is used.",
    )
    parser.add_argument(
        "--slice-idx",
        type=int,
        default=None,
        help="Axial slice index to plot. If omitted, chooses tumor-centered slice or middle slice.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patient_dir = find_patient_dir(args.data_dir, args.patient_id)
    modality_paths = collect_nii_files(patient_dir)
    volumes = load_volumes(modality_paths)
    
    # Apply z-score normalization to MRI modalities
    print("Normalizing modalities (z-score):")
    normalize_modalities(volumes, ["t1", "t1ce", "t2", "flair"])

    print_shapes(patient_dir, modality_paths, volumes)
    
    # Combine four modalities into multi-channel tensor
    print("\nCombining modalities into multi-channel tensor:")
    multi_channel = combine_modalities(volumes, ["t1", "t1ce", "t2", "flair"])
    print(f"  Combined tensor shape (H, W, D, C): {multi_channel.shape}")
    print(f"  Data type: {multi_channel.dtype}")
    
    # Extract 2D axial slices with tumor
    print("\nExtracting tumor-containing axial slices:")
    slices_2d, tumor_z_indices = extract_tumor_slices(volumes, ["t1", "t1ce", "t2", "flair"])
    print(f"  Found {len(slices_2d)} slices with tumor")
    print(f"  Z-indices: {tumor_z_indices[:5]}{'...' if len(tumor_z_indices) > 5 else ''}")
    print(f"  Each slice shape (H, W, C): {slices_2d[0].shape}")
    print(f"  Data type: {slices_2d[0].dtype}")
    
    # Resize slices to 128×128 for model input
    print("\nResizing slices to fixed resolution:")
    target_resolution = 128
    slices_resized = resize_slices_to_target(slices_2d, target_size=target_resolution, method="crop_center")
    print(f"  Resized {len(slices_resized)} slices to {target_resolution}×{target_resolution}")
    print(f"  Each slice shape (H, W, C): {slices_resized[0].shape}")
    print(f"  Data type: {slices_resized[0].dtype}")
    
    slice_idx = choose_slice_index(volumes, args.slice_idx)
    plot_modalities(volumes, slice_idx, patient_dir.name)


def choose_slice_index(volumes: dict[str, np.ndarray], slice_idx: int | None) -> int:
    any_volume = next(iter(volumes.values()))
    z_dim = any_volume.shape[2]

    if slice_idx is not None:
        if not (0 <= slice_idx < z_dim):
            raise ValueError(f"slice_idx must be in [0, {z_dim - 1}], got {slice_idx}")
        return slice_idx

    if "seg" in volumes:
        seg = volumes["seg"]
        nonzero = np.argwhere(seg > 0)
        if nonzero.size > 0:
            return int(np.median(nonzero[:, 2]))

    return z_dim // 2
if __name__ == "__main__":
    main()