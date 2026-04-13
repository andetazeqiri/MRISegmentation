#!/usr/bin/env python3
"""FastAPI service exposing segmentation inference endpoint."""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
import sys
import tempfile
import time

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

TARGET_SIZE = int(os.getenv("TARGET_SIZE", "128"))
DEFAULT_MODEL_TYPE = os.getenv("MODEL_NAME", "unet").lower().strip()

MODEL_CONFIGS: dict[str, Path] = {
    "unet": Path(os.getenv("CHECKPOINT_PATH_UNET", "./models/best_unet.pt")),
    "resunet": Path(os.getenv("CHECKPOINT_PATH_RESUNET", "./models/best_resunet.pt")),
}

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("brats_api")

_MODELS: dict[str, object] = {}
_MODEL_DEVICES: dict[str, object] = {}
_MODEL_ERRORS: dict[str, str] = {}


def _load_single_model(model_type: str) -> tuple[object, object]:
    """Load and cache one model by type."""

    if model_type in _MODELS and model_type in _MODEL_DEVICES:
        return _MODELS[model_type], _MODEL_DEVICES[model_type]

    if model_type in _MODEL_ERRORS:
        raise HTTPException(status_code=503, detail=_MODEL_ERRORS[model_type])

    checkpoint = MODEL_CONFIGS.get(model_type)
    if checkpoint is None:
        raise HTTPException(status_code=400, detail=f"Unsupported model_type '{model_type}'.")
    if not checkpoint.is_file():
        msg = f"Checkpoint for model_type '{model_type}' not found: {checkpoint}"
        _MODEL_ERRORS[model_type] = msg
        raise HTTPException(status_code=503, detail=msg)

    LOGGER.info("Loading model '%s' from %s", model_type, checkpoint)

    try:
        cfg = InferenceConfig(
            model_name=model_type,
            checkpoint_path=checkpoint,
            target_size=TARGET_SIZE,
            in_channels=4,
            num_classes=4,
            base_channels=32,
        )
        model, device = load_model_for_inference(cfg)
        _MODELS[model_type] = model
        _MODEL_DEVICES[model_type] = device
        return model, device
    except Exception as exc:
        msg = f"Failed to load checkpoint for '{model_type}': {exc}"
        _MODEL_ERRORS[model_type] = msg
        raise HTTPException(status_code=503, detail=msg) from exc


@app.on_event("startup")
def _warmup_models() -> None:
    """Best-effort startup loading to avoid first-request latency."""

    for model_type in ("unet", "resunet"):
        checkpoint = MODEL_CONFIGS.get(model_type)
        if checkpoint is None or not checkpoint.is_file():
            LOGGER.warning("Model '%s' skipped at startup (checkpoint missing).", model_type)
            continue
        try:
            _load_single_model(model_type)
            LOGGER.info("Model '%s' loaded at startup.", model_type)
        except HTTPException as exc:
            LOGGER.warning("Model '%s' failed at startup: %s", model_type, exc.detail)


def _validate_npy_input(arr: np.ndarray) -> None:
    """Validate incoming numpy slice shape against training constraints."""

    if arr.ndim != 3:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input rank for .npy: expected 3D array, got shape {arr.shape}",
        )
    if not (arr.shape[-1] == 4 or arr.shape[0] == 4):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid input shape for .npy. Expected 4 channels in (H, W, 4) or (4, H, W); "
                f"received {arr.shape}."
            ),
        )


@app.get("/health")
def health() -> dict[str, object]:
    """Health endpoint for service monitoring."""

    loaded = sorted(_MODELS.keys())
    return {
        "status": "ok",
        "target_size": TARGET_SIZE,
        "default_model": DEFAULT_MODEL_TYPE,
        "loaded_models": loaded,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    slice_index: int | None = Form(None),
    axis: int = Form(2),
    model_type: str = Form("unet"),
) -> dict[str, object]:
    """Run segmentation on one uploaded MRI input.

    Supported input formats:
    - .npy: single 4-channel slice with shape (H, W, 4) or (4, H, W)
    - .nii / .nii.gz: MRI volume; one slice is extracted via axis/slice_index
    """

    started = time.perf_counter()

    filename = (file.filename or "").lower()
    if not (filename.endswith(".npy") or filename.endswith(".nii") or filename.endswith(".nii.gz")):
        raise HTTPException(status_code=400, detail="Supported formats: .npy, .nii, .nii.gz")

    if axis not in (0, 1, 2):
        raise HTTPException(status_code=400, detail="axis must be 0, 1, or 2")

    model_type = (model_type or DEFAULT_MODEL_TYPE).lower().strip()
    if model_type not in {"unet", "resunet"}:
        raise HTTPException(status_code=400, detail="model_type must be 'unet' or 'resunet'")

    model, device = _load_single_model(model_type)

    payload = await file.read()
    input_mode = "npy"
    used_slice: int | None = None

    if filename.endswith(".npy"):
        try:
            arr = np.load(io.BytesIO(payload), allow_pickle=False)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid .npy file: {exc}") from exc
        _validate_npy_input(arr)
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

    seg = output["segmentation"]
    conf = output["confidence"]
    classes = sorted(int(x) for x in np.unique(seg).tolist())
    mean_conf = float(np.mean(conf))
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    LOGGER.info(
        "Prediction executed model=%s mode=%s file=%s slice=%s mean_conf=%.4f time_ms=%.2f",
        model_type,
        input_mode,
        file.filename,
        used_slice,
        mean_conf,
        elapsed_ms,
    )

    return {
        "shape": output["shape"],
        "prediction_shape": output["shape"],
        "classes": classes,
        "mean_confidence": mean_conf,
        "class_distribution": output["class_distribution"],
        "model_type": model_type,
        "device": str(device),
        "input_mode": input_mode,
        "slice_index_used": None if used_slice is None else int(used_slice),
        "inference_time_ms": round(elapsed_ms, 3),
        "segmentation": seg.tolist(),
        "confidence": conf.tolist(),
    }
