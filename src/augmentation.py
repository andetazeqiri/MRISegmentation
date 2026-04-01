#!/usr/bin/env python3
"""Data augmentation utilities for 2D BraTS MRI slices and masks."""

import numpy as np


def horizontal_flip(slice_stack: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Apply left-right flip to image channels and mask."""
	return np.flip(slice_stack, axis=1), np.flip(mask, axis=1)


def rotate_90(slice_stack: np.ndarray, mask: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
	"""Rotate image channels and mask by k * 90 degrees in-plane."""
	return np.rot90(slice_stack, k=k, axes=(0, 1)), np.rot90(mask, k=k, axes=(0, 1))


def add_gaussian_noise(slice_stack: np.ndarray, noise_std: float) -> np.ndarray:
	"""Add zero-mean Gaussian noise to MRI channels only."""
	noise = np.random.normal(loc=0.0, scale=noise_std, size=slice_stack.shape)
	return slice_stack + noise


def apply_random_augmentation(
	slice_stack: np.ndarray,
	mask: np.ndarray,
	flip_prob: float = 0.5,
	rotation_prob: float = 0.5,
	noise_prob: float = 0.5,
	noise_std: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
	"""Apply random flip, rotation, and noise augmentation.

	Geometric transforms are applied to both image and mask to preserve alignment.
	Noise is applied only to image channels.
	"""
	aug_slice = slice_stack.copy()
	aug_mask = mask.copy()

	if np.random.rand() < flip_prob:
		aug_slice, aug_mask = horizontal_flip(aug_slice, aug_mask)

	if np.random.rand() < rotation_prob:
		k = np.random.randint(1, 4)
		aug_slice, aug_mask = rotate_90(aug_slice, aug_mask, k)

	if np.random.rand() < noise_prob:
		aug_slice = add_gaussian_noise(aug_slice, noise_std)

	return np.ascontiguousarray(aug_slice), np.ascontiguousarray(aug_mask)
