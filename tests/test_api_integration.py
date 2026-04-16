#!/usr/bin/env python3
"""Integration tests for the FastAPI segmentation service."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from fastapi.testclient import TestClient

from src.model_architecture import build_model
import src.api_server as api_server


class ApiIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(cls._tmpdir.name)

        unet_checkpoint = tmp_path / "best_unet.pt"
        resunet_checkpoint = tmp_path / "best_resunet.pt"

        unet_model = build_model("unet", in_channels=4, num_classes=4, base_channels=32)
        resunet_model = build_model("resunet", in_channels=4, num_classes=4, base_channels=32)

        torch.save(unet_model.state_dict(), unet_checkpoint)
        torch.save(resunet_model.state_dict(), resunet_checkpoint)

        api_server.MODEL_CONFIGS = {
            "unet": unet_checkpoint,
            "resunet": resunet_checkpoint,
        }
        api_server._MODELS.clear()
        api_server._MODEL_DEVICES.clear()
        api_server._MODEL_ERRORS.clear()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def setUp(self) -> None:
        api_server._MODELS.clear()
        api_server._MODEL_DEVICES.clear()
        api_server._MODEL_ERRORS.clear()
        self.client = TestClient(api_server.app)

    def _make_npy_bytes(self, array: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        np.save(buffer, array)
        return buffer.getvalue()

    def test_health_endpoint_reports_loaded_models(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("target_size", payload)
        self.assertIn("loaded_models", payload)

    def test_predict_accepts_valid_numpy_input(self) -> None:
        image = np.random.rand(128, 128, 4).astype(np.float32)
        npy_bytes = self._make_npy_bytes(image)

        response = self.client.post(
            "/predict",
            files={"file": ("sample.npy", npy_bytes, "application/octet-stream")},
            data={"model_type": "unet"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["input_mode"], "npy")
        self.assertEqual(payload["model_type"], "unet")
        self.assertEqual(payload["prediction_shape"], [128, 128])
        self.assertIn("mean_confidence", payload)
        self.assertIn("confidence", payload)
        self.assertIn("segmentation", payload)

    def test_predict_accepts_valid_nifti_input(self) -> None:
        volume = np.random.rand(64, 64, 16).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            nii_path = Path(tmpdir) / "sample.nii.gz"
            nib.save(nib.Nifti1Image(volume, affine=np.eye(4)), nii_path)
            raw = nii_path.read_bytes()

        response = self.client.post(
            "/predict",
            files={"file": ("sample.nii.gz", raw, "application/octet-stream")},
            data={"model_type": "resunet", "axis": "2", "slice_index": "5"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model_type"], "resunet")
        self.assertEqual(payload["input_mode"], "3d_single_modality_repeated_to_4ch")
        self.assertEqual(payload["slice_index_used"], 5)
        self.assertEqual(payload["prediction_shape"], [128, 128])
        self.assertIn("class_distribution", payload)

    def test_predict_rejects_invalid_numpy_shape(self) -> None:
        invalid = np.random.rand(10, 10).astype(np.float32)
        npy_bytes = self._make_npy_bytes(invalid)

        response = self.client.post(
            "/predict",
            files={"file": ("invalid.npy", npy_bytes, "application/octet-stream")},
            data={"model_type": "unet"},
        )

        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertIn("Invalid input", payload["detail"])

    def test_predict_rejects_unsupported_model_type(self) -> None:
        image = np.random.rand(128, 128, 4).astype(np.float32)
        npy_bytes = self._make_npy_bytes(image)

        response = self.client.post(
            "/predict",
            files={"file": ("sample.npy", npy_bytes, "application/octet-stream")},
            data={"model_type": "unsupported"},
        )

        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertIn("model_type", payload["detail"])


if __name__ == "__main__":
    unittest.main()
