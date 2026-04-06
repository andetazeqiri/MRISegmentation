#!/usr/bin/env python3
"""Load one BraTS patient and visualize MRI modalities and segmentation."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.augmentation import add_gaussian_noise, horizontal_flip, rotate_90
from src.preprocessing import (
    collect_nii_files,
    combine_modalities,
    extract_tumor_slices,
    find_patient_dir,
    load_volumes,
    normalize_modalities,
    resize_slices_to_target,
)


def plot_original_vs_normalized(
    original_volumes: dict[str, np.ndarray],
    normalized_volumes: dict[str, np.ndarray],
    slice_idx: int,
    patient_name: str,
    output_path: Path | None = None,
    show: bool = True,
) -> None:
    """Plot original and normalized MRI slices side by side for key modalities."""
    modalities = ["flair", "t1", "t1ce", "t2"]
    available = [m for m in modalities if m in original_volumes and m in normalized_volumes]
    if not available:
        print("No matching MRI modalities found for comparison plot.")
        return

    fig, axes = plt.subplots(2, len(available), figsize=(4 * len(available), 8))
    axes = np.array(axes)
    if len(available) == 1:
        axes = axes.reshape(2, 1)

    for i, modality in enumerate(available):
        orig_img = original_volumes[modality][:, :, slice_idx]
        norm_img = normalized_volumes[modality][:, :, slice_idx]

        axes[0, i].imshow(np.rot90(orig_img), cmap="gray")
        axes[0, i].set_title(f"Original {modality.upper()}")
        axes[0, i].axis("off")

        axes[1, i].imshow(np.rot90(norm_img), cmap="gray")
        axes[1, i].set_title(f"Normalized {modality.upper()}")
        axes[1, i].axis("off")

    fig.suptitle(f"{patient_name} | original vs normalized | slice z={slice_idx}", fontsize=14)
    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison figure: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_slice_extraction_example(
    volumes: dict[str, np.ndarray],
    slice_idx: int,
    patient_name: str,
    output_path: Path | None = None,
    show: bool = True,
) -> None:
    """Plot a thesis-ready example of tumor slice extraction and stacked modalities."""
    if "seg" not in volumes:
        print("Segmentation mask not found; cannot build slice extraction example.")
        return

    modalities = ["flair", "t1", "t1ce", "t2"]
    available = [m for m in modalities if m in volumes]
    if len(available) < 4:
        print("Missing one or more MRI modalities; cannot build slice extraction example.")
        return

    extracted_slice = np.stack([volumes[m][:, :, slice_idx] for m in modalities], axis=-1)
    seg_slice = volumes["seg"][:, :, slice_idx] > 0

    rgb = np.zeros((extracted_slice.shape[0], extracted_slice.shape[1], 3), dtype=np.float32)
    for src_idx, dst_idx in [(0, 0), (2, 1), (3, 2)]:
        channel = extracted_slice[:, :, src_idx]
        channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
        rgb[:, :, dst_idx] = channel

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    axes[0].imshow(np.rot90(extracted_slice[:, :, 0]), cmap="gray")
    axes[0].set_title("Extracted FLAIR slice")
    axes[0].axis("off")

    axes[1].imshow(np.rot90(rgb))
    axes[1].imshow(np.rot90(seg_slice), cmap="Reds", alpha=0.25, interpolation="nearest")
    axes[1].set_title(f"Tumor-containing slice (z={slice_idx})")
    axes[1].axis("off")

    axes[2].imshow(np.rot90(rgb))
    axes[2].set_title("Stacked modalities\n(R=FLAIR, G=T1ce, B=T2)")
    axes[2].axis("off")

    fig.suptitle(f"Slice Extraction Example - {patient_name}", fontsize=14)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved slice extraction figure: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_augmentation_example(
    volumes: dict[str, np.ndarray],
    slice_idx: int,
    patient_name: str,
    output_path: Path | None = None,
    show: bool = True,
) -> None:
    """Plot original and augmented slice examples for thesis figures."""
    modalities = ["t1", "t1ce", "t2", "flair"]
    missing = [m for m in modalities if m not in volumes]
    if missing:
        print(f"Missing modalities for augmentation example: {missing}")
        return

    slice_stack = np.stack([volumes[m][:, :, slice_idx] for m in modalities], axis=-1)
    seg = volumes.get("seg")
    if seg is None:
        print("Segmentation mask not found; cannot build augmentation example.")
        return
    mask = (seg[:, :, slice_idx] > 0).astype(np.uint8)

    original = slice_stack
    flip_img, flip_mask = horizontal_flip(slice_stack, mask)
    rot_img, rot_mask = rotate_90(slice_stack, mask, k=1)
    noise_img = add_gaussian_noise(slice_stack, noise_std=0.08)

    # Use FLAIR channel for display consistency across all panels.
    display_channel = 3
    panel_data = [
        ("Original", original[:, :, display_channel], mask),
        ("Horizontal Flip", flip_img[:, :, display_channel], flip_mask),
        ("Rotation (90deg)", rot_img[:, :, display_channel], rot_mask),
        ("Gaussian Noise", noise_img[:, :, display_channel], mask),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    axes = axes.ravel()
    for ax, (title, image, panel_mask) in zip(axes, panel_data):
        ax.imshow(np.rot90(image), cmap="gray")
        ax.imshow(np.rot90(panel_mask), cmap="Reds", alpha=0.22, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(f"Augmentation Examples (FLAIR) - {patient_name} | slice z={slice_idx}", fontsize=14)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved augmentation figure: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def print_shapes(patient_dir: Path, modality_paths: dict[str, Path], volumes: dict[str, np.ndarray]) -> None:
    print(f"Patient: {patient_dir.name}")
    print("Loaded files and shapes:")
    for modality in sorted(volumes.keys()):
        print(f"  {modality:>5}: {modality_paths[modality].name} -> {volumes[modality].shape}")


def plot_modalities(
    volumes: dict[str, np.ndarray],
    slice_idx: int,
    patient_name: str,
    output_path: Path | None = None,
    show: bool = True,
) -> None:
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
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved modality figure: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


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
    parser.add_argument(
        "--compare-normalization",
        action="store_true",
        help="Show side-by-side original and normalized MRI slices for T1, T1ce, T2, and FLAIR.",
    )
    parser.add_argument(
        "--slice-extraction-example",
        action="store_true",
        help="Show a thesis-ready figure for tumor slice extraction and stacked modalities.",
    )
    parser.add_argument(
        "--augmentation-example",
        action="store_true",
        help="Show a thesis-ready figure with original, flip, rotation, and noise augmentations.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory to save generated figures as PNG files.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open plot windows (useful when only saving images).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patient_dir = find_patient_dir(args.data_dir, args.patient_id)
    modality_paths = collect_nii_files(patient_dir)
    volumes = load_volumes(modality_paths)
    original_volumes = {k: v.copy() for k, v in volumes.items()}
    
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
    show_plots = not args.no_show
    save_dir = args.save_dir

    if args.compare_normalization:
        compare_out = None
        if save_dir is not None:
            compare_out = save_dir / f"{patient_dir.name}_slice_{slice_idx:03d}_original_vs_normalized.png"
        plot_original_vs_normalized(
            original_volumes,
            volumes,
            slice_idx,
            patient_dir.name,
            output_path=compare_out,
            show=show_plots,
        )

    if args.slice_extraction_example:
        extraction_out = None
        if save_dir is not None:
            extraction_out = save_dir / f"{patient_dir.name}_slice_{slice_idx:03d}_slice_extraction_example.png"
        plot_slice_extraction_example(
            volumes,
            slice_idx,
            patient_dir.name,
            output_path=extraction_out,
            show=show_plots,
        )

    if args.augmentation_example:
        augmentation_out = None
        if save_dir is not None:
            augmentation_out = save_dir / f"{patient_dir.name}_slice_{slice_idx:03d}_augmentation_example.png"
        plot_augmentation_example(
            volumes,
            slice_idx,
            patient_dir.name,
            output_path=augmentation_out,
            show=show_plots,
        )

    modalities_out = None
    if save_dir is not None:
        modalities_out = save_dir / f"{patient_dir.name}_slice_{slice_idx:03d}_modalities_and_seg.png"
    plot_modalities(
        volumes,
        slice_idx,
        patient_dir.name,
        output_path=modalities_out,
        show=show_plots,
    )


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