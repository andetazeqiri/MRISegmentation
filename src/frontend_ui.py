#!/usr/bin/env python3
"""Thesis-oriented Streamlit UI for MRI segmentation inference and visual analysis."""

from __future__ import annotations

import io
import tempfile

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import requests
from scipy import ndimage
import streamlit as st


CLASS_NAMES = {
    0: "Background",
    1: "Necrotic / Non-Enhancing",
    2: "Edema",
    3: "Enhancing Tumor",
}

CLASS_COLORS = {
    0: (0.0, 0.0, 0.0),
    1: (0.95, 0.55, 0.15),
    2: (0.17, 0.62, 0.40),
    3: (0.80, 0.22, 0.30),
}


st.set_page_config(page_title="Brain Tumor Segmentation Thesis Demo", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@700&family=Source+Sans+3:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Source Sans 3', sans-serif;
    }

    .app-shell {
        background: linear-gradient(140deg, #f6f7ef 0%, #edf3f8 45%, #f8f2ea 100%);
        border: 1px solid #d8dee4;
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 14px;
    }

    .headline {
        font-family: 'Merriweather', serif;
        font-size: 2.0rem;
        line-height: 1.2;
        margin: 0;
        color: #203247;
    }

    .subline {
        margin-top: 8px;
        color: #3d5268;
        font-size: 1.03rem;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #dde5ec;
        border-radius: 10px;
        padding: 10px 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-shell">
        <p class="headline">Brain Tumor Segmentation: Inference Demonstrator</p>
        <p class="subline">
            This interface presents model-driven segmentation for MRI inputs and supports visual validation
            through confidence analysis and optional ground-truth comparison.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


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


def _resize_nearest(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize segmentation-like map using nearest interpolation."""

    if mask.shape == target_shape:
        return mask
    zoom_factors = (target_shape[0] / float(mask.shape[0]), target_shape[1] / float(mask.shape[1]))
    return ndimage.zoom(mask, zoom_factors, order=0)


def _segmentation_to_rgb(seg: np.ndarray) -> np.ndarray:
    """Convert class-index segmentation map to RGB visualization."""

    h, w = seg.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for cls, color in CLASS_COLORS.items():
        rgb[seg == cls] = np.asarray(color, dtype=np.float32)
    return rgb


def _overlay_segmentation(base: np.ndarray, seg: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay segmentation colors on grayscale base image."""

    gray = _normalize_for_display(base)
    gray_rgb = np.stack([gray, gray, gray], axis=-1)
    seg_rgb = _segmentation_to_rgb(seg)
    return np.clip((1.0 - alpha) * gray_rgb + alpha * seg_rgb, 0.0, 1.0)


def _extract_slice_for_preview(
    volume: np.ndarray,
    axis: int,
    slice_idx: int,
    channel_idx: int,
) -> np.ndarray | None:
    """Extract a displayable 2D image from flexible MRI input layouts."""

    if volume.ndim == 2:
        return volume
    if volume.ndim == 3:
        return np.take(volume, slice_idx, axis=axis)
    if volume.ndim == 4 and volume.shape[-1] == 4:
        return np.take(volume, slice_idx, axis=axis)[:, :, channel_idx]
    if volume.ndim == 4 and volume.shape[0] == 4:
        return np.take(volume, slice_idx, axis=axis + 1)[channel_idx]
    return None


def _extract_seg_slice(volume: np.ndarray, axis: int, slice_idx: int) -> np.ndarray | None:
    """Extract segmentation slice from GT file."""

    if volume.ndim == 2:
        seg = volume
    elif volume.ndim == 3:
        seg = np.take(volume, slice_idx, axis=axis)
    else:
        return None

    seg = seg.astype(np.int64)
    seg[seg == 4] = 3
    return seg


def _dice_per_class(pred: np.ndarray, gt: np.ndarray) -> dict[int, float]:
    """Compute per-class Dice for 2D masks."""

    out: dict[int, float] = {}
    for cls in range(4):
        pred_c = pred == cls
        gt_c = gt == cls
        inter = float(np.logical_and(pred_c, gt_c).sum())
        denom = float(pred_c.sum() + gt_c.sum())
        out[cls] = (2.0 * inter + 1e-6) / (denom + 1e-6)
    return out


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


def _load_upload_bytes(uploaded_file) -> tuple[bytes, str, np.ndarray]:
    """Load uploaded file as raw bytes and numpy array."""

    raw = uploaded_file.read()
    filename = uploaded_file.name.lower()
    if filename.endswith(".npy"):
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
    else:
        arr = _load_nifti_from_bytes(raw, uploaded_file.name)
    return raw, filename, arr


with st.sidebar:
    st.header("Inference Controls")
    api_url = st.text_input("API endpoint", value="http://127.0.0.1:8000/predict")
    input_file = st.file_uploader("MRI input (.nii/.nii.gz/.npy)", type=["npy", "nii", "gz"])
    gt_file = st.file_uploader("Optional ground truth mask (.nii/.nii.gz/.npy)", type=["npy", "nii", "gz"])


if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_preview" not in st.session_state:
    st.session_state.last_preview = None

if input_file is None:
    st.info("Upload an MRI file from the sidebar to begin inference.")
    st.stop()

try:
    raw_bytes, filename, input_arr = _load_upload_bytes(input_file)
except Exception as exc:
    st.error(f"Failed to read input file: {exc}")
    st.stop()

left, mid, right = st.columns([1.1, 1.1, 1.4])

with left:
    st.markdown("### Input Configuration")
    if filename.endswith(".npy"):
        axis = 2
        axis_size = 1
        slice_idx = 0
        if input_arr.ndim == 3 and (input_arr.shape[-1] == 4 or input_arr.shape[0] == 4):
            channel_idx = st.slider("Display channel", 0, 3, 1)
        else:
            channel_idx = 0
        st.caption(f"Loaded .npy shape: {tuple(input_arr.shape)}")
    else:
        axis = st.selectbox("Slice axis", options=[0, 1, 2], index=2)
        axis_size = int(input_arr.shape[axis]) if input_arr.ndim >= 3 else 1
        default_idx = axis_size // 2
        slice_idx = st.slider("Slice index", 0, max(0, axis_size - 1), default_idx)
        channel_idx = st.slider("Display channel", 0, 3, 1)
        st.caption(f"Loaded NIfTI shape: {tuple(input_arr.shape)}")

with mid:
    st.markdown("### Service Status")
    try:
        health_url = api_url.replace("/predict", "/health")
        health = requests.get(health_url, timeout=4)
        if health.status_code == 200:
            st.success("API reachable")
        else:
            st.warning(f"API responded with status {health.status_code}")
    except Exception:
        st.warning("API not reachable. Start uvicorn service before inference.")

    run_clicked = st.button("Run Segmentation", type="primary", use_container_width=True)

with right:
    st.markdown("### Study Context")
    st.markdown(
        "- Model: U-Net / Residual U-Net checkpoint inference"
        "\n- Input support: NIfTI volumes and 4-channel NumPy slices"
        "\n- Output: class-index mask, confidence map, class distribution"
    )

preview_image = _extract_slice_for_preview(input_arr, axis=axis, slice_idx=slice_idx, channel_idx=channel_idx)

if run_clicked:
    with st.spinner("Running segmentation inference..."):
        files = {
            "file": (input_file.name, raw_bytes, "application/octet-stream"),
        }
        form_data = {}
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            form_data["axis"] = str(axis)
            form_data["slice_index"] = str(slice_idx)

        try:
            response = requests.post(api_url, files=files, data=form_data, timeout=120)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            st.error(f"API request failed: {exc}")
            st.stop()

    seg = np.array(data["segmentation"], dtype=np.int64)
    conf = np.array(data["confidence"], dtype=np.float32)

    st.session_state.last_result = {
        "response": data,
        "seg": seg,
        "conf": conf,
        "axis": axis,
        "slice_idx": slice_idx,
        "channel_idx": channel_idx,
    }
    st.session_state.last_preview = preview_image

if st.session_state.last_result is None:
    st.stop()

result = st.session_state.last_result
pred_seg = result["seg"]
pred_conf = result["conf"]
preview_image = st.session_state.last_preview

if preview_image is None:
    preview_for_overlay = np.zeros_like(pred_seg, dtype=np.float32)
else:
    preview_for_overlay = _resize_nearest(np.asarray(preview_image, dtype=np.float32), pred_seg.shape)

overlay_alpha = st.slider("Overlay opacity", min_value=0.10, max_value=0.90, value=0.45, step=0.05)
overlay = _overlay_segmentation(preview_for_overlay, pred_seg, alpha=overlay_alpha)

mean_conf = float(np.mean(pred_conf))
tumor_fraction = 1.0 - float((pred_seg == 0).mean())

mc1, mc2, mc3 = st.columns(3)
with mc1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Mean Confidence", f"{mean_conf:.3f}")
    st.markdown("</div>", unsafe_allow_html=True)
with mc2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Tumor Pixel Ratio", f"{tumor_fraction:.3%}")
    st.markdown("</div>", unsafe_allow_html=True)
with mc3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Slice Used", str(result["response"].get("slice_index_used", "n/a")))
    st.markdown("</div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Core Views", "Overlay & Validation", "Class Report"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        ax.imshow(_normalize_for_display(preview_for_overlay), cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title("Input Preview")
        ax.axis("off")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        ax.imshow(_segmentation_to_rgb(pred_seg))
        ax.set_title("Predicted Segmentation")
        ax.axis("off")
        st.pyplot(fig)
    with c3:
        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        im = ax.imshow(pred_conf, cmap="cividis", vmin=0.0, vmax=1.0)
        ax.set_title("Prediction Confidence")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)

with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        ax.imshow(overlay)
        ax.set_title("Prediction Overlay on MRI")
        ax.axis("off")
        st.pyplot(fig)

    with col_b:
        if gt_file is None:
            st.info("Upload an optional ground-truth mask in the sidebar to compute comparison metrics.")
        else:
            try:
                _, _, gt_arr = _load_upload_bytes(gt_file)
                gt_slice = _extract_seg_slice(gt_arr, axis=result["axis"], slice_idx=result["slice_idx"])
                if gt_slice is None:
                    st.warning("Ground-truth file shape is unsupported for direct slice extraction.")
                else:
                    gt_resized = _resize_nearest(gt_slice, pred_seg.shape)
                    dice_cls = _dice_per_class(pred_seg, gt_resized)
                    mean_dice = float(np.mean(list(dice_cls.values())))

                    fig, ax = plt.subplots(figsize=(5.2, 5.2))
                    gt_overlay = _overlay_segmentation(preview_for_overlay, gt_resized, alpha=overlay_alpha)
                    ax.imshow(gt_overlay)
                    ax.set_title("Ground Truth Overlay")
                    ax.axis("off")
                    st.pyplot(fig)

                    st.metric("Mean Dice (2D Slice)", f"{mean_dice:.3f}")

                    rows = []
                    for cls in range(4):
                        rows.append(
                            {
                                "Class": CLASS_NAMES.get(cls, str(cls)),
                                "Dice": round(float(dice_cls[cls]), 4),
                            }
                        )
                    st.table(rows)
            except Exception as exc:
                st.error(f"Ground-truth comparison failed: {exc}")

with tab3:
    distribution = result["response"].get("class_distribution", {})
    rows = []
    for cls in range(4):
        fraction = float(distribution.get(str(cls), 0.0))
        rows.append(
            {
                "Class": CLASS_NAMES.get(cls, str(cls)),
                "Pixel Fraction": round(fraction, 6),
                "Percent": f"{fraction * 100:.2f}%",
            }
        )
    st.table(rows)

    st.caption(
        "Input mode: "
        f"{result['response'].get('input_mode', 'npy')} | "
        f"Slice used: {result['response'].get('slice_index_used', 'n/a')}"
    )
