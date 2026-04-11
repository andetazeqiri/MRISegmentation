#!/usr/bin/env python3
"""Export a model-ready 4-channel .npy slice from BraTS NIfTI files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import collect_nii_files, find_patient_dir, load_volumes, normalize_modalities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one 4-channel MRI slice to .npy")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"), help="BraTS data directory")
    parser.add_argument("--patient-id", type=str, default=None, help="Patient folder name, e.g. BraTS20_Training_001")
    parser.add_argument("--slice-index", type=int, default=None, help="Axial z-index. If omitted, auto-select")
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["t1", "t1ce", "t2", "flair"],
        help="Modalities to stack in output channels",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output .npy path")
    return parser.parse_args()


def _find_key_case_insensitive(keys: list[str], name: str) -> str | None:
    for key in keys:
        if key.lower() == name.lower():
            return key
    return None


def _auto_pick_slice(volumes: dict[str, np.ndarray]) -> int:
    """Pick a representative axial slice: tumor-middle if seg exists, else volume middle."""

    seg_key = _find_key_case_insensitive(list(volumes.keys()), "seg")
    if seg_key is not None:
        seg = volumes[seg_key]
        tumor_indices = [z for z in range(seg.shape[2]) if np.any(seg[:, :, z] > 0)]
        if tumor_indices:
            return tumor_indices[len(tumor_indices) // 2]

    # Fall back to middle slice using first volume shape.
    first_volume = next(iter(volumes.values()))
    return first_volume.shape[2] // 2


def main() -> None:
    args = parse_args()

    patient_dir = find_patient_dir(args.data_dir, args.patient_id)
    modality_paths = collect_nii_files(patient_dir)
    volumes = load_volumes(modality_paths)

    normalize_modalities(volumes, args.modalities)

    mapped_modalities: list[str] = []
    for requested in args.modalities:
        key = _find_key_case_insensitive(list(volumes.keys()), requested)
        if key is None:
            raise ValueError(f"Modality '{requested}' not found. Available: {list(volumes.keys())}")
        mapped_modalities.append(key)

    z = args.slice_index if args.slice_index is not None else _auto_pick_slice(volumes)

    # Validate z against first chosen modality shape.
    reference = volumes[mapped_modalities[0]]
    if z < 0 or z >= reference.shape[2]:
        raise ValueError(f"slice-index out of range [0, {reference.shape[2] - 1}]: {z}")

    channels = [volumes[key][:, :, z].astype(np.float32) for key in mapped_modalities]
    slice_hwc = np.stack(channels, axis=-1)

    out_path = args.output
    if out_path is None:
        out_path = Path("./outputs") / f"{patient_dir.name}_slice_{z:03d}.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(out_path, slice_hwc)

    print(f"Patient: {patient_dir.name}")
    print(f"Slice index: {z}")
    print(f"Shape saved: {slice_hwc.shape}")
    print(f"Saved .npy file: {out_path}")


if __name__ == "__main__":
    main()
