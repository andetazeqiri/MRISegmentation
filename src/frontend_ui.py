#!/usr/bin/env python3
"""Streamlit frontend for interactive MRI slice segmentation visualization."""

from __future__ import annotations

import io
import tempfile

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import requests
import streamlit as st


st.set_page_config(page_title="Brain Tumor Segmentation UI", layout="wide")
st.title("Brain Tumor Segmentation Demo")
st.write("Upload MRI data (.nii/.nii.gz or .npy) and run model inference through the API.")

api_url = st.text_input("API endpoint", value="http://127.0.0.1:8000/predict")
uploaded = st.file_uploader("Upload MRI input (.nii, .nii.gz, or .npy)", type=["npy", "nii", "gz"])


def _normalize_for_display(image: np.ndarray) -> np.ndarray:
    """Scale image for stable visualization and avoid blank white/black previews."""

    img = np.asarray(image, dtype=np.float32)
    if img.size == 0:
        return img

    p_low = np.percentile(img, 1)
    p_high = np.percentile(img, 99)
    if p_high <= p_low:
        p_low = float(np.min(img))
        p_high = float(np.max(img))
        if p_high <= p_low:
            return np.zeros_like(img, dtype=np.float32)

    out = (img - p_low) / (p_high - p_low)
    return np.clip(out, 0.0, 1.0)


def _to_hwc(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")
    if arr.shape[-1] == 4:
        return arr
    if arr.shape[0] == 4:
        return np.transpose(arr, (1, 2, 0))
    raise ValueError(f"Expected 4 channels in last or first axis, got shape {arr.shape}")


def _load_nifti_from_bytes(raw: bytes, filename: str) -> np.ndarray:
    suffix = ".nii.gz" if filename.lower().endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(raw)
        tmp.flush()
        volume = nib.load(tmp.name).get_fdata(dtype=np.float32)
    return volume


if uploaded is not None:
    raw_bytes = uploaded.read()
    filename = uploaded.name.lower()

    preview_image = None
    channel_idx = 0
    axis = 2
    slice_idx = None

    try:
        if filename.endswith(".npy"):
            local_arr = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
            local_hwc = _to_hwc(local_arr)
            channel_idx = st.slider("Display input modality channel", min_value=0, max_value=3, value=1)
            preview_image = local_hwc[:, :, channel_idx]
            st.caption(f"Loaded .npy shape: {tuple(local_arr.shape)}")
        else:
            volume = _load_nifti_from_bytes(raw_bytes, uploaded.name)
            st.caption(f"Loaded NIfTI shape: {tuple(volume.shape)}")

            axis = st.selectbox("Slice axis", options=[0, 1, 2], index=2)

            if volume.ndim >= 3:
                axis_size = int(volume.shape[axis])
                default_idx = axis_size // 2
                slice_idx = st.slider("Slice index", min_value=0, max_value=max(0, axis_size - 1), value=default_idx)
            else:
                slice_idx = 0

            if volume.ndim == 3:
                preview_image = np.take(volume, slice_idx, axis=axis)
            elif volume.ndim == 4 and volume.shape[-1] == 4:
                channel_idx = st.slider("Display input modality channel", min_value=0, max_value=3, value=1)
                preview_image = np.take(volume, slice_idx, axis=axis)[:, :, channel_idx]
            elif volume.ndim == 4 and volume.shape[0] == 4:
                channel_idx = st.slider("Display input modality channel", min_value=0, max_value=3, value=1)
                preview_image = np.take(volume, slice_idx, axis=axis + 1)[channel_idx]
            else:
                st.info(
                    "NIfTI preview is limited for this shape, but inference is still supported by the API if compatible."
                )
    except Exception as exc:
        st.error(f"Failed to load uploaded file: {exc}")
        st.stop()

    if st.button("Run Segmentation"):
        with st.spinner("Running inference..."):
            files = {
                "file": (uploaded.name, raw_bytes, "application/octet-stream"),
            }
            form_data = {}
            if filename.endswith(".nii") or filename.endswith(".nii.gz"):
                form_data["axis"] = str(axis)
                if slice_idx is not None:
                    form_data["slice_index"] = str(slice_idx)
            try:
                resp = requests.post(api_url, files=files, data=form_data, timeout=120)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                st.error(f"API request failed: {exc}")
                st.stop()

        seg = np.array(data["segmentation"], dtype=np.int64)
        conf = np.array(data["confidence"], dtype=np.float32)

        col1, col2, col3 = st.columns(3)

        with col1:
            fig, ax = plt.subplots(figsize=(4, 4))
            if preview_image is not None:
                ax.imshow(_normalize_for_display(preview_image), cmap="gray", vmin=0.0, vmax=1.0)
                ax.set_title("Input Preview")
            else:
                ax.text(0.5, 0.5, "Preview unavailable", ha="center", va="center")
                ax.set_title("Input Preview")
            ax.axis("off")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(seg, cmap="viridis", vmin=0, vmax=3)
            ax.set_title("Predicted Segmentation")
            ax.axis("off")
            st.pyplot(fig)

        with col3:
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(conf, cmap="magma", vmin=0.0, vmax=1.0)
            ax.set_title("Prediction Confidence")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)

        st.subheader("Class Distribution")
        st.json(data.get("class_distribution", {}))
        st.caption(
            f"Input mode: {data.get('input_mode', 'npy')} | Slice used: {data.get('slice_index_used', 'n/a')}"
        )
