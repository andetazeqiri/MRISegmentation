#!/usr/bin/env python3
"""Custom PyTorch Dataset for BraTS MRI slices with segmentation masks."""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import (
    collect_nii_files,
    combine_modalities,
    extract_tumor_slices,
    load_volumes,
    normalize_modalities,
    resize_slices_to_target,
)


class BraTSDataset(Dataset):
    """Custom PyTorch Dataset for BraTS MRI slices with segmentation masks.
    
    Loads preprocessed 2D axial slices from one or more BraTS patients.
    Each sample is a 4-channel (T1, T1ce, T2, FLAIR) MRI slice with its segmentation mask.
    
    Args:
        data_dir: Path to the data directory containing BraTS patient folders
        patient_ids: List of specific patient IDs to use. If None, uses all patients found.
        target_size: Target size for slice resizing (default 128×128)
        resize_method: Resizing method: 'crop_center' or 'resize' (default 'crop_center')
        modalities: List of modalities to extract (default ['t1', 't1ce', 't2', 'flair'])
        seg_threshold: Threshold for segmentation mask (default 0.5)
        cache: Whether to cache preprocessed slices in memory (default True)
    """

    def __init__(
        self,
        data_dir: str | Path,
        patient_ids: list[str] | None = None,
        target_size: int = 128,
        resize_method: str = "crop_center",
        modalities: list[str] | None = None,
        seg_threshold: float = 0.5,
        cache: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.resize_method = resize_method
        self.modalities = modalities or ["t1", "t1ce", "t2", "flair"]
        self.seg_threshold = seg_threshold
        self.cache = cache

        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Find patient directories
        if patient_ids is None:
            patient_dirs = sorted(
                [p for p in self.data_dir.iterdir() if p.is_dir() and p.name.startswith("BraTS")]
            )
        else:
            patient_dirs = [self.data_dir / pid for pid in patient_ids]
            for pd in patient_dirs:
                if not pd.is_dir():
                    raise FileNotFoundError(f"Patient folder not found: {pd}")

        if not patient_dirs:
            raise ValueError(f"No BraTS patient folders found in: {self.data_dir}")

        # Build patient-slice mapping and cache
        self.patients = []  # List of (patient_dir, num_slices)
        self.slice_mapping = []  # List of (patient_idx, slice_local_idx)
        self.slice_cache = {}  # Cache: (patient_idx, slice_idx) -> (slice_tensor, mask_tensor)

        for patient_dir in patient_dirs:
            try:
                slices_2d, z_indices = self._load_and_preprocess_patient(patient_dir)
                num_slices = len(slices_2d)
                patient_idx = len(self.patients)

                self.patients.append((patient_dir, num_slices, slices_2d, z_indices))

                for slice_idx in range(num_slices):
                    self.slice_mapping.append((patient_idx, slice_idx))

                print(f"  Loaded patient {patient_dir.name}: {num_slices} tumor slices")
            except Exception as e:
                print(f"  Warning: Failed to load patient {patient_dir.name}: {e}")

        if not self.slice_mapping:
            raise ValueError("No valid slices loaded from any patient")

        print(f"Dataset initialized: {len(self.slice_mapping)} total slices from {len(self.patients)} patients")

    def _load_and_preprocess_patient(self, patient_dir: Path) -> tuple[list[np.ndarray], list[int]]:
        """Load and preprocess a single patient's data.
        
        Returns:
            Tuple of (resized 2D slices, z-indices)
        """
        # Collect and load volumes
        modality_paths = collect_nii_files(patient_dir)
        volumes = load_volumes(modality_paths)

        # Normalize specified modalities
        normalize_modalities(volumes, self.modalities)

        # Extract tumor slices
        slices_2d, z_indices = extract_tumor_slices(
            volumes,
            self.modalities,
            seg_threshold=self.seg_threshold,
        )

        # Resize to target size
        resized_slices = resize_slices_to_target(
            slices_2d,
            target_size=self.target_size,
            method=self.resize_method,
        )

        return resized_slices, z_indices

    def _get_mask_for_slice(self, patient_idx: int, local_z_idx: int) -> np.ndarray:
        """Load the segmentation mask for a specific slice.
        
        Args:
            patient_idx: Index into self.patients
            local_z_idx: Index into the slices for this patient
        
        Returns:
            Resized segmentation mask (target_size × target_size)
        """
        patient_dir = self.patients[patient_idx][0]
        z_indices = self.patients[patient_idx][3]
        z = z_indices[local_z_idx]

        # Load segmentation mask
        modality_paths = collect_nii_files(patient_dir)
        seg_path = modality_paths.get("seg") or modality_paths.get("Seg")
        if seg_path is None:
            raise FileNotFoundError(f"Segmentation mask not found for {patient_dir.name}")

        seg_volume = nib.load(str(seg_path)).get_fdata()
        seg_slice = seg_volume[:, :, z]

        # Resize mask to target size
        if seg_slice.shape != (self.target_size, self.target_size):
            from scipy import ndimage

            if self.resize_method == "crop_center":
                H, W = seg_slice.shape
                start_h = (H - self.target_size) // 2
                start_w = (W - self.target_size) // 2
                seg_resized = seg_slice[
                    max(0, start_h) : min(H, start_h + self.target_size),
                    max(0, start_w) : min(W, start_w + self.target_size),
                ]

                if seg_resized.shape[0] < self.target_size or seg_resized.shape[1] < self.target_size:
                    padded = np.zeros((self.target_size, self.target_size), dtype=seg_slice.dtype)
                    pad_h_start = (self.target_size - seg_resized.shape[0]) // 2
                    pad_w_start = (self.target_size - seg_resized.shape[1]) // 2
                    padded[
                        pad_h_start : pad_h_start + seg_resized.shape[0],
                        pad_w_start : pad_w_start + seg_resized.shape[1],
                    ] = seg_resized
                    seg_resized = padded
            else:
                resize_factor = self.target_size / max(seg_slice.shape)
                seg_resized = ndimage.zoom(seg_slice, resize_factor, order=0)
                h_curr, w_curr = seg_resized.shape
                padded = np.zeros((self.target_size, self.target_size), dtype=seg_slice.dtype)
                pad_h = (self.target_size - h_curr) // 2
                pad_w = (self.target_size - w_curr) // 2
                padded[pad_h : pad_h + h_curr, pad_w : pad_w + w_curr] = seg_resized
                seg_resized = padded
        else:
            seg_resized = seg_slice

        return seg_resized

    def __len__(self) -> int:
        """Return total number of slices across all patients."""
        return len(self.slice_mapping)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a single 2D slice and its segmentation mask as tensors.
        
        Args:
            idx: Index into the dataset
        
        Returns:
            Tuple of (slice_tensor, mask_tensor)
            - slice_tensor: shape (C, target_size, target_size) where C=4 (T1, T1ce, T2, FLAIR)
            - mask_tensor: shape (1, target_size, target_size)
        """
        patient_idx, local_slice_idx = self.slice_mapping[idx]

        # Check cache first
        if self.cache:
            cache_key = (patient_idx, local_slice_idx)
            if cache_key in self.slice_cache:
                return self.slice_cache[cache_key]

        # Load slice and mask
        slices_2d = self.patients[patient_idx][2]
        slice_stack = slices_2d[local_slice_idx]  # shape: (H, W, C)

        mask = self._get_mask_for_slice(patient_idx, local_slice_idx)  # shape: (H, W)

        # Convert to PyTorch tensors
        # Slice: (H, W, C) -> (C, H, W)
        slice_tensor = torch.from_numpy(slice_stack.transpose(2, 0, 1)).float()

        # Mask: (H, W) -> (1, H, W)
        mask_tensor = torch.from_numpy(mask[np.newaxis, :, :]).float()

        # Cache if enabled
        if self.cache:
            cache_key = (patient_idx, local_slice_idx)
            self.slice_cache[cache_key] = (slice_tensor, mask_tensor)

        return slice_tensor, mask_tensor

    def get_patient_info(self, idx: int) -> dict:
        """Get metadata about a sample.
        
        Args:
            idx: Index into the dataset
        
        Returns:
            Dictionary with patient_id, patient_idx, local_slice_idx, z_index
        """
        patient_idx, local_slice_idx = self.slice_mapping[idx]
        patient_dir, num_slices, _, z_indices = self.patients[patient_idx]
        z_idx = z_indices[local_slice_idx]

        return {
            "patient_id": patient_dir.name,
            "patient_idx": patient_idx,
            "local_slice_idx": local_slice_idx,
            "z_index": z_idx,
            "global_idx": idx,
        }


if __name__ == "__main__":
    # Example usage
    import sys

    data_dir = Path("./data")
    if not data_dir.is_dir():
        print(f"Error: Data directory '{data_dir}' not found")
        sys.exit(1)

    print("\nInitializing BraTSDataset...")
    dataset = BraTSDataset(
        data_dir,
        target_size=128,
        resize_method="crop_center",
        cache=True,
    )

    print(f"\nDataset size: {len(dataset)}")

    # Load and inspect first sample
    print("\nLoading first sample...")
    slice_tensor, mask_tensor = dataset[0]
    info = dataset.get_patient_info(0)

    print(f"  Patient: {info['patient_id']}")
    print(f"  Z-index: {info['z_index']}")
    print(f"  Slice shape (C, H, W): {slice_tensor.shape}")
    print(f"  Slice dtype: {slice_tensor.dtype}")
    print(f"  Mask shape (1, H, W): {mask_tensor.shape}")
    print(f"  Mask dtype: {mask_tensor.dtype}")
    print(f"  Slice value range: [{slice_tensor.min():.2f}, {slice_tensor.max():.2f}]")
    print(f"  Mask unique values: {torch.unique(mask_tensor).numpy()}")
