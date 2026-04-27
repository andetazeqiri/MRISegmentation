#!/usr/bin/env python3
"""BraTS MRI preprocessing utilities: loading, normalization, resizing, and slicing."""

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def find_patient_dir(data_dir: Path, patient_id: str | None) -> Path:
    """Find a patient directory by ID or return the first one found.
    
    Args:
        data_dir: Path to data directory
        patient_id: Specific patient folder name (e.g., BraTS20_Training_001), or None
    
    Returns:
        Path to patient directory
    
    Raises:
        FileNotFoundError: If patient not found or no BraTS folders exist
    """
    if patient_id:
        patient_dir = data_dir / patient_id
        if not patient_dir.is_dir():
            raise FileNotFoundError(f"Patient folder not found: {patient_dir}")
        return patient_dir

    candidates = sorted(
        [
            p
            for p in data_dir.iterdir()
            if p.is_dir() and (p.name.startswith("BraTS") or p.name.startswith("FeTS"))
        ]
    )
    if not candidates:
        raise FileNotFoundError(f"No BraTS/FeTS patient folders found in: {data_dir}")
    return candidates[0]


def collect_nii_files(patient_dir: Path) -> dict[str, Path]:
    """Collect all .nii/.nii.gz files in a patient directory.
    
    Args:
        patient_dir: Path to patient folder
    
    Returns:
        Dictionary mapping modality names (e.g., 't1', 'flair', 'seg') to file paths
    
    Raises:
        FileNotFoundError: If no .nii/.nii.gz files found
    """
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
    """Load NIfTI volumes from file paths.
    
    Args:
        modality_paths: Dictionary mapping modality names to file paths
    
    Returns:
        Dictionary mapping modality names to 3D numpy arrays (H × W × D)
    """
    volumes: dict[str, np.ndarray] = {}
    for modality, file_path in sorted(modality_paths.items()):
        volumes[modality] = nib.load(str(file_path)).get_fdata()
    return volumes


def zscore_normalize(volume: np.ndarray, exclude_zero: bool = True) -> np.ndarray:
    """Apply z-score normalization to a volume.
    
    Args:
        volume: Input 3D MRI volume
        exclude_zero: If True, compute statistics only on non-zero voxels (brain tissue)
    
    Returns:
        Normalized volume with mean≈0 and std≈1 (on non-zero voxels if exclude_zero=True)
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
        volumes: Dictionary of modality names to volume arrays (modified in-place)
        modalities_to_normalize: List of modality names to normalize (case-insensitive)
    
    Returns:
        Dictionary with normalized volumes (same as input, modified in-place)
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
        volumes: Dictionary of modality names to volume arrays (shape: H × W × D)
        modalities: List of modality names to combine, in order (case-insensitive)
    
    Returns:
        Multi-channel tensor of shape (H, W, D, C) where C=len(modalities)
    
    Raises:
        ValueError: If any modality not found in volumes
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
        volumes: Dictionary of modality names to volume arrays (shape: H × W × D)
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


def resize_slices_to_target(
    slices_2d: list[np.ndarray],
    target_size: int = 128,
    method: str = "crop_center",
) -> list[np.ndarray]:
    """Resize or crop 2D slices to a fixed resolution.
    
    Args:
        slices_2d: List of 2D slice stacks, each shape (H, W, C)
        target_size: Target size (assumes square output: target_size × target_size)
        method: 'crop_center' to center-crop, 'resize' to zoom/scale
    
    Returns:
        List of resized/cropped slices, each shape (target_size, target_size, C)
    """
    resized_slices = []
    
    for slice_stack in slices_2d:
        H, W, C = slice_stack.shape
        
        if method == "crop_center":
            # Center crop to target_size x target_size
            start_h = (H - target_size) // 2
            start_w = (W - target_size) // 2
            cropped = slice_stack[
                max(0, start_h) : min(H, start_h + target_size),
                max(0, start_w) : min(W, start_w + target_size),
                :
            ]
            
            # Pad if necessary (fallback for small slices)
            if cropped.shape[0] < target_size or cropped.shape[1] < target_size:
                padded = np.zeros((target_size, target_size, C), dtype=slice_stack.dtype)
                pad_h_start = (target_size - cropped.shape[0]) // 2
                pad_w_start = (target_size - cropped.shape[1]) // 2
                padded[
                    pad_h_start : pad_h_start + cropped.shape[0],
                    pad_w_start : pad_w_start + cropped.shape[1],
                    :
                ] = cropped
                resized_slices.append(padded)
            else:
                resized_slices.append(cropped)
        
        elif method == "resize":
            # Resize by scaling each channel independently
            resize_factor = target_size / max(H, W)
            resized_stack = np.zeros((target_size, target_size, C), dtype=slice_stack.dtype)
            
            for c in range(C):
                resized_channel = ndimage.zoom(slice_stack[:, :, c], resize_factor, order=1)
                h_curr, w_curr = resized_channel.shape
                
                # Center-place resized channel in output
                if h_curr < target_size or w_curr < target_size:
                    pad_h = (target_size - h_curr) // 2
                    pad_w = (target_size - w_curr) // 2
                    resized_stack[
                        pad_h : pad_h + h_curr,
                        pad_w : pad_w + w_curr
                    ] = resized_channel
                else:
                    resized_stack[:, :, c] = resized_channel[:target_size, :target_size]
            
            resized_slices.append(resized_stack)
    
    return resized_slices
