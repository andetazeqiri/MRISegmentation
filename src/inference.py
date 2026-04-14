#!/usr/bin/env python3
"""Inference pipeline utilities for segmentation model serving."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from src.model_architecture import build_model
from src.preprocessing import zscore_normalize


@dataclass
class InferenceConfig:
    """Configuration for model loading and input preprocessing."""

    model_name: str = "unet"
    checkpoint_path: Path = Path("./models/best_unet.pt")
    target_size: int = 128
    in_channels: int = 4
    num_classes: int = 4
    base_channels: int = 32


def _resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Return a valid execution device for inference."""

    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def load_model_for_inference(
    config: InferenceConfig,
    device: str | torch.device | None = None,
) -> tuple[torch.nn.Module, torch.device]:
    """Stage 1: load a trained segmentation model for inference.

    The model is moved to the selected device and switched to eval mode.
    """

    checkpoint = Path(config.checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    run_device = _resolve_device(device)
    model = build_model(
        config.model_name,
        in_channels=config.in_channels,
        num_classes=config.num_classes,
        base_channels=config.base_channels,
    ).to(run_device)

    loaded = torch.load(checkpoint, map_location=run_device, weights_only=False)
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        state_dict = loaded["model_state_dict"]
    elif isinstance(loaded, dict):
        state_dict = loaded
    else:
        raise TypeError("Unsupported checkpoint format. Expected a state dict or checkpoint dict.")

    model.load_state_dict(state_dict)
    model.eval()
    return model, run_device


def _resize_slice_stack(slice_stack: np.ndarray, target_size: int) -> np.ndarray:
    """Resize each channel in an HxWxC slice stack to target_size x target_size."""

    h, w, c = slice_stack.shape
    if h == target_size and w == target_size:
        return slice_stack

    out = np.zeros((target_size, target_size, c), dtype=np.float32)
    zoom_h = target_size / float(h)
    zoom_w = target_size / float(w)

    for channel in range(c):
        out[:, :, channel] = ndimage.zoom(slice_stack[:, :, channel], (zoom_h, zoom_w), order=1)
    return out


def volume_to_multichannel_slice(
    volume: np.ndarray,
    in_channels: int = 4,
    axis: int = 2,
    slice_index: int | None = None,
) -> tuple[np.ndarray, int, str]:
    """Convert a NIfTI volume into a single 2D multi-channel slice.

    Returns:
        tuple of (slice_hwc, used_slice_index, mode)
    """

    arr = np.asarray(volume, dtype=np.float32)

    if arr.ndim == 2:
        hwc = np.repeat(arr[:, :, None], in_channels, axis=2)
        return hwc, 0, "2d_repeated_to_4ch"

    if arr.ndim == 3:
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1, or 2 for 3D volumes. Received {axis}.")
        size = arr.shape[axis]
        used_idx = size // 2 if slice_index is None else int(np.clip(slice_index, 0, size - 1))
        slice_2d = np.take(arr, used_idx, axis=axis)
        hwc = np.repeat(slice_2d[:, :, None], in_channels, axis=2)
        return hwc, used_idx, "3d_single_modality_repeated_to_4ch"

    if arr.ndim == 4:
        # Common layout: (H, W, D, C)
        if arr.shape[-1] == in_channels:
            if axis not in (0, 1, 2):
                raise ValueError(f"axis must be 0, 1, or 2 for HxWxDxC volumes. Received {axis}.")
            size = arr.shape[axis]
            used_idx = size // 2 if slice_index is None else int(np.clip(slice_index, 0, size - 1))
            hwc = np.take(arr, used_idx, axis=axis)
            return hwc.astype(np.float32), used_idx, "4d_hwdc"

        # Common layout: (C, H, W, D)
        if arr.shape[0] == in_channels:
            if axis not in (0, 1, 2):
                raise ValueError(f"axis must be 0, 1, or 2 for CxHxWxD volumes. Received {axis}.")
            source_axis = axis + 1
            size = arr.shape[source_axis]
            used_idx = size // 2 if slice_index is None else int(np.clip(slice_index, 0, size - 1))
            chw = np.take(arr, used_idx, axis=source_axis)
            hwc = np.transpose(chw, (1, 2, 0))
            return hwc.astype(np.float32), used_idx, "4d_chwd"

        # Alternative layout: (H, W, C, D)
        if arr.shape[2] == in_channels:
            size = arr.shape[3]
            used_idx = size // 2 if slice_index is None else int(np.clip(slice_index, 0, size - 1))
            hwc = arr[:, :, :, used_idx]
            return hwc.astype(np.float32), used_idx, "4d_hwcd"

    raise ValueError(
        "Unsupported NIfTI array shape for inference. Expected 2D, 3D, or 4D data with "
        f"modalities compatible with 4 channels. Received shape {arr.shape}."
    )


def process_input_slice(
    image_slice: np.ndarray,
    target_size: int = 128,
    in_channels: int = 4,
) -> np.ndarray:
    """Stage 2: normalize and resize a multi-modal MRI slice.

    Accepts either shape (H, W, C) or (C, H, W) where C==in_channels.
    Returns a processed HxWxC array with C=in_channels.
    """

    arr = np.asarray(image_slice, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array for a single slice, got shape {arr.shape}")

    if arr.shape[-1] == in_channels:
        hwc = arr
    elif arr.shape[0] == in_channels:
        hwc = np.transpose(arr, (1, 2, 0))
    else:
        raise ValueError(
            "Input must have 4 channels in either (H, W, 4) or (4, H, W) format. "
            f"Received shape {arr.shape}."
        )

    normalized = np.zeros_like(hwc, dtype=np.float32)
    for channel in range(in_channels):
        normalized[:, :, channel] = zscore_normalize(hwc[:, :, channel], exclude_zero=True).astype(np.float32)

    resized = _resize_slice_stack(normalized, target_size=target_size)
    return resized.astype(np.float32)


def prepare_input_tensor(processed_slice: np.ndarray) -> torch.Tensor:
    """Stage 3: convert processed HxWxC slice to model tensor shape (B, C, H, W)."""

    if processed_slice.ndim != 3:
        raise ValueError(f"Expected processed_slice with 3 dims (H,W,C), got {processed_slice.shape}")
    chw = np.transpose(processed_slice, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0).float()


@torch.no_grad()
def forward_pass(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Stage 4: execute deterministic forward inference on the selected device."""

    model.eval()
    return model(input_tensor.to(device))


def postprocess_logits(logits: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Stage 5: convert logits to class mask and confidence map."""

    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1)

    seg = pred.squeeze(0).cpu().numpy().astype(np.int64)
    conf = probs.max(dim=1).values.squeeze(0).cpu().numpy().astype(np.float32)
    return seg, conf


def format_output(seg: np.ndarray, conf: np.ndarray) -> dict[str, np.ndarray | dict[str, float] | list[int]]:
    """Stage 6: create structured output payload from segmentation artifacts."""

    unique, counts = np.unique(seg, return_counts=True)
    distribution = {str(int(k)): float(v / seg.size) for k, v in zip(unique, counts)}

    return {
        "segmentation": seg,
        "confidence": conf,
        "shape": [int(seg.shape[0]), int(seg.shape[1])],
        "class_distribution": distribution,
    }


def run_inference_pipeline(
    model: torch.nn.Module,
    image_slice: np.ndarray,
    device: torch.device,
    target_size: int = 128,
    in_channels: int = 4,
) -> dict[str, np.ndarray | dict[str, float] | list[int]]:
    """Run complete 6-stage inference pipeline from input slice to structured output."""

    processed = process_input_slice(
        image_slice=image_slice,
        target_size=target_size,
        in_channels=in_channels,
    )
    input_tensor = prepare_input_tensor(processed)
    logits = forward_pass(model, input_tensor, device)
    seg, conf = postprocess_logits(logits)
    return format_output(seg, conf)


def preprocess_input_slice(
    image_slice: np.ndarray,
    target_size: int = 128,
    in_channels: int = 4,
) -> torch.Tensor:
    """Backward-compatible wrapper returning model input tensor.

    Kept for compatibility with existing call sites.
    """

    processed = process_input_slice(image_slice, target_size=target_size, in_channels=in_channels)
    return prepare_input_tensor(processed)


@torch.no_grad()
def predict_segmentation(
    model: torch.nn.Module,
    image_slice: np.ndarray,
    device: torch.device,
    target_size: int = 128,
    in_channels: int = 4,
) -> dict[str, np.ndarray | dict[str, float] | list[int]]:
    """Run segmentation inference using the staged pipeline implementation."""

    return run_inference_pipeline(
        model=model,
        image_slice=image_slice,
        device=device,
        target_size=target_size,
        in_channels=in_channels,
    )
