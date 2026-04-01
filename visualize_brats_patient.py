#!/usr/bin/env python3
"""Load one BraTS patient and visualize MRI modalities and segmentation."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def find_patient_dir(data_dir: Path, patient_id: str | None) -> Path:
    if patient_id:
        patient_dir = data_dir / patient_id
        if not patient_dir.is_dir():
            raise FileNotFoundError(f"Patient folder not found: {patient_dir}")
        return patient_dir

    candidates = sorted([p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("BraTS")])
    if not candidates:
        raise FileNotFoundError(f"No patient folders found in: {data_dir}")
    return candidates[0]


def collect_nii_files(patient_dir: Path) -> dict[str, Path]:
    files = sorted(list(patient_dir.glob("*.nii")) + list(patient_dir.glob("*.nii.gz")))
    if not files:
        raise FileNotFoundError(f"No .nii/.nii.gz files found in: {patient_dir}")

    modalities: dict[str, Path] = {}
    prefix = f"{patient_dir.name}_"

    for file_path in files:
        stem = file_path.name
        if stem.endswith(".nii.gz"):
            stem = stem[:-7]
        elif stem.endswith(".nii"):
            stem = stem[:-4]

        modality = stem[len(prefix) :] if stem.startswith(prefix) else stem
        modalities[modality] = file_path

    return modalities


def load_volumes(modality_paths: dict[str, Path]) -> dict[str, np.ndarray]:
    volumes: dict[str, np.ndarray] = {}
    for modality, file_path in sorted(modality_paths.items()):
        volumes[modality] = nib.load(str(file_path)).get_fdata()
    return volumes


def zscore_normalize(volume: np.ndarray, exclude_zero: bool = True) -> np.ndarray:
    """Apply z-score normalization to a volume.
    
    Args:
        volume: Input 3D MRI volume
        exclude_zero: If True, compute statistics only on non-zero voxels
    
    Returns:
        Normalized volume with mean=0 and std=1 (on non-zero voxels if exclude_zero=True)
    """
    if exclude_zero:
        mask = volume != 0
        if np.sum(mask) == 0:
            return volume
        mean = np.mean(volume[mask])
        std = np.std(volume[mask])
    else:
        mean = np.mean(volume)
        std = np.std(volume)
    
    if std < 1e-8:  # Avoid division by very small numbers
        return volume
    
    normalized = (volume - mean) / std
    return normalized


def normalize_modalities(volumes: dict[str, np.ndarray], modalities_to_normalize: list[str]) -> dict[str, np.ndarray]:
    """Apply z-score normalization to specified modalities.
    
    Args:
        volumes: Dictionary of modality names to volume arrays
        modalities_to_normalize: List of modality names to normalize (case-insensitive)
    
    Returns:
        Dictionary with normalized volumes (original dict is modified in-place)
    """
    for modality in modalities_to_normalize:
        for key in volumes.keys():
            if key.lower() == modality.lower():
                volumes[key] = zscore_normalize(volumes[key])
                print(f"  Normalized {key}")
                break
    return volumes


def combine_modalities(volumes: dict[str, np.ndarray], modalities: list[str]) -> np.ndarray:
    """Combine multiple MRI modalities into a single multi-channel tensor.
    
    Args:
        volumes: Dictionary of modality names to volume arrays (shape: H x W x D)
        modalities: List of modality names to combine, in order (case-insensitive)
    
    Returns:
        Multi-channel tensor of shape (H, W, D, C) where C=len(modalities)
    
    Raises:
        ValueError: If any modality is not found in volumes
    """
    channels = []
    for modality in modalities:
        found = False
        for key in volumes.keys():
            if key.lower() == modality.lower():
                channels.append(volumes[key])
                found = True
                break
        if not found:
            raise ValueError(f"Modality '{modality}' not found in volumes. Available: {list(volumes.keys())}")
    
    # Stack along last axis: (H, W, D, C)
    multi_channel = np.stack(channels, axis=-1)
    return multi_channel


def extract_tumor_slices(
    volumes: dict[str, np.ndarray],
    modalities: list[str],
    seg_threshold: float = 0.5,
) -> tuple[list[np.ndarray], list[int]]:
    """Extract 2D axial slices from 3D volumes, keeping only slices with tumor.
    
    Args:
        volumes: Dictionary of modality names to volume arrays (shape: H x W x D)
        modalities: List of modality names to extract (case-insensitive)
        seg_threshold: Threshold for segmentation mask; voxels > threshold count as tumor
    
    Returns:
        Tuple of:
        - List of 2D slice stacks, each shape (H, W, C) where C=len(modalities)
        - List of corresponding z-indices (axial slice positions)
    
    Raises:
        ValueError: If segmentation mask not found or no tumor slices exist
    """
    # Find segmentation mask
    seg = None
    for key in volumes.keys():
        if key.lower() == "seg":
            seg = volumes[key]
            break
    
    if seg is None:
        raise ValueError("Segmentation mask ('seg') not found in volumes")
    
    # Find all slices with tumor
    z_dim = seg.shape[2]
    tumor_slice_indices = []
    for z in range(z_dim):
        if np.sum(seg[:, :, z] > seg_threshold) > 0:
            tumor_slice_indices.append(z)
    
    if not tumor_slice_indices:
        raise ValueError("No slices with tumor found in segmentation mask")
    
    # Extract 2D slices for each modality at tumor z-indices
    slices_2d = []
    for z in tumor_slice_indices:
        channels = []
        for modality in modalities:
            found = False
            for key in volumes.keys():
                if key.lower() == modality.lower():
                    slice_2d = volumes[key][:, :, z]
                    channels.append(slice_2d)
                    found = True
                    break
            if not found:
                raise ValueError(f"Modality '{modality}' not found in volumes")
        
        # Stack channels: (H, W, C)
        slice_stack = np.stack(channels, axis=-1)
        slices_2d.append(slice_stack)
    
    return slices_2d, tumor_slice_indices


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
    
    slice_idx = choose_slice_index(volumes, args.slice_idx)
    plot_modalities(volumes, slice_idx, patient_dir.name)


if __name__ == "__main__":
    main()