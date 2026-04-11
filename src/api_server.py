#!/usr/bin/env python3
"""FastAPI service exposing segmentation inference endpoint."""

from __future__ import annotations

import io
import os
from pathlib import Path
import sys
import tempfile

import nibabel as nib
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import (
    InferenceConfig,
    load_model_for_inference,
    predict_segmentation,
    volume_to_multichannel_slice,
)


app = FastAPI(title="BraTS Segmentation API", version="1.0.0")

CHECKPOINT_PATH = Path(os.getenv("CHECKPOINT_PATH", "./models/best_unet.pt"))
MODEL_NAME = os.getenv("MODEL_NAME", "unet")
TARGET_SIZE = int(os.getenv("TARGET_SIZE", "128"))

_MODEL = None
_DEVICE = None
_MODEL_ERROR = None


def _ensure_model_loaded() -> tuple[object, object]:
    global _MODEL, _DEVICE, _MODEL_ERROR

    if _MODEL is not None and _DEVICE is not None:
        return _MODEL, _DEVICE
    if _MODEL_ERROR is not None:
        raise HTTPException(status_code=503, detail=_MODEL_ERROR)

    try:
        cfg = InferenceConfig(
            model_name=MODEL_NAME,
            checkpoint_path=CHECKPOINT_PATH,
            target_size=TARGET_SIZE,
            in_channels=4,
            num_classes=4,
            base_channels=32,
        )
        _MODEL, _DEVICE = load_model_for_inference(cfg)
        return _MODEL, _DEVICE
    except Exception as exc:
        _MODEL_ERROR = f"Failed to load model checkpoint: {exc}"
        raise HTTPException(status_code=503, detail=_MODEL_ERROR) from exc


@app.get("/health")
def health() -> dict[str, str]:
    """Health endpoint for service monitoring."""

    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    slice_index: int | None = Form(None),
    axis: int = Form(2),
) -> dict[str, object]:
    """Run segmentation on one uploaded MRI input.

    Supported input formats:
    - .npy: single 4-channel slice with shape (H, W, 4) or (4, H, W)
    - .nii / .nii.gz: MRI volume; one slice is extracted via axis/slice_index
    """

    filename = (file.filename or "").lower()
    if not (filename.endswith(".npy") or filename.endswith(".nii") or filename.endswith(".nii.gz")):
        raise HTTPException(status_code=400, detail="Supported formats: .npy, .nii, .nii.gz")

    if axis not in (0, 1, 2):
        raise HTTPException(status_code=400, detail="axis must be 0, 1, or 2")

    model, device = _ensure_model_loaded()

    payload = await file.read()
    input_mode = "npy"
    used_slice = 0

    if filename.endswith(".npy"):
        try:
            arr = np.load(io.BytesIO(payload), allow_pickle=False)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid .npy file: {exc}") from exc
    else:
        suffix = ".nii.gz" if filename.endswith(".nii.gz") else ".nii"
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
                tmp.write(payload)
                tmp.flush()
                volume = nib.load(tmp.name).get_fdata(dtype=np.float32)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid NIfTI file: {exc}") from exc

        try:
            arr, used_slice, input_mode = volume_to_multichannel_slice(
                volume,
                in_channels=4,
                axis=axis,
                slice_index=slice_index,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"NIfTI preprocessing failed: {exc}") from exc

    try:
        output = predict_segmentation(
            model=model,
            image_slice=arr,
            device=device,
            target_size=TARGET_SIZE,
            in_channels=4,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Inference failed: {exc}") from exc

    return {
        "shape": output["shape"],
        "class_distribution": output["class_distribution"],
        "input_mode": input_mode,
        "slice_index_used": int(used_slice),
        "segmentation": output["segmentation"].tolist(),
        "confidence": output["confidence"].tolist(),
    }
