#!/usr/bin/env python3
"""Unit tests for the staged inference pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from src.inference import (
    InferenceConfig,
    format_output,
    load_model_for_inference,
    postprocess_logits,
    prepare_input_tensor,
    process_input_slice,
    run_inference_pipeline,
    volume_to_multichannel_slice,
)
from src.model_architecture import build_model


class InferencePipelineTests(unittest.TestCase):
    def test_process_and_prepare_shapes(self) -> None:
        image = np.random.rand(140, 160, 4).astype(np.float32)
        processed = process_input_slice(image, target_size=128, in_channels=4)
        tensor = prepare_input_tensor(processed)

        self.assertEqual(processed.shape, (128, 128, 4))
        self.assertEqual(tensor.shape, (1, 4, 128, 128))

    def test_volume_to_multichannel_slice_for_3d_volume(self) -> None:
        volume = np.random.rand(64, 64, 20).astype(np.float32)
        slice_hwc, used_slice, mode = volume_to_multichannel_slice(volume, axis=2, slice_index=5)

        self.assertEqual(slice_hwc.shape, (64, 64, 4))
        self.assertEqual(used_slice, 5)
        self.assertEqual(mode, "3d_single_modality_repeated_to_4ch")

    def test_postprocess_and_format_output_shapes(self) -> None:
        logits = torch.randn(1, 4, 128, 128)
        seg, conf = postprocess_logits(logits)
        output = format_output(seg, conf)

        self.assertEqual(seg.shape, (128, 128))
        self.assertEqual(conf.shape, (128, 128))
        self.assertIn("segmentation", output)
        self.assertIn("confidence", output)
        self.assertIn("class_distribution", output)
        self.assertEqual(output["shape"], [128, 128])

    def test_run_inference_pipeline_returns_expected_keys(self) -> None:
        model = build_model("unet", in_channels=4, num_classes=4, base_channels=32)
        model.eval()
        image = np.random.rand(128, 128, 4).astype(np.float32)
        device = torch.device("cpu")

        output = run_inference_pipeline(model, image, device, target_size=128, in_channels=4)

        self.assertIn("segmentation", output)
        self.assertIn("confidence", output)
        self.assertIn("shape", output)
        self.assertIn("class_distribution", output)
        self.assertEqual(output["shape"], [128, 128])

    def test_load_model_for_inference_from_temporary_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "dummy_unet.pt"
            model = build_model("unet", in_channels=4, num_classes=4, base_channels=32)
            torch.save(model.state_dict(), checkpoint_path)

            cfg = InferenceConfig(
                model_name="unet",
                checkpoint_path=checkpoint_path,
                target_size=128,
                in_channels=4,
                num_classes=4,
                base_channels=32,
            )

            loaded_model, device = load_model_for_inference(cfg, device="cpu")
            self.assertEqual(device.type, "cpu")
            self.assertFalse(loaded_model.training)

    def test_nifti_slice_to_multichannel_conversion_returns_metadata(self) -> None:
        volume = np.random.rand(32, 32, 10).astype(np.float32)
        slice_hwc, used_slice, mode = volume_to_multichannel_slice(volume, axis=2, slice_index=3)

        self.assertEqual(slice_hwc.shape, (32, 32, 4))
        self.assertEqual(used_slice, 3)
        self.assertTrue(mode.startswith("3d"))


if __name__ == "__main__":
    unittest.main()
