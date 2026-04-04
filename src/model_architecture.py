#!/usr/bin/env python3
"""Model architecture definitions for brain tumor segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two-layer convolutional block used throughout U-Net."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """Encoder stage with a conv block followed by max-pooling."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, dropout=dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        downsampled = self.pool(features)
        return features, downsampled


class DecoderBlock(nn.Module):
    """Decoder stage with up-convolution, skip concatenation, and conv block."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    """2D U-Net for multi-modal brain tumor segmentation.

    Input shape: (B, 4, 128, 128)
    Output shape: (B, num_classes, 128, 128)
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base_channels: int = 32,
        bottleneck_dropout: float = 0.2,
    ):
        super().__init__()

        self.enc1 = EncoderBlock(in_channels, base_channels)
        self.enc2 = EncoderBlock(base_channels, base_channels * 2)
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4)
        self.enc4 = EncoderBlock(base_channels * 4, base_channels * 8)

        self.bottleneck = ConvBlock(
            base_channels * 8,
            base_channels * 16,
            dropout=bottleneck_dropout,
        )

        self.dec4 = DecoderBlock(base_channels * 16, base_channels * 8, base_channels * 8)
        self.dec3 = DecoderBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.dec1 = DecoderBlock(base_channels * 2, base_channels, base_channels)

        self.out_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        return self.out_conv(x)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable model parameters."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = UNet2D(in_channels=4, num_classes=4, base_channels=32)
    x = torch.randn(2, 4, 128, 128)
    y = model(x)

    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    print(f"Trainable params: {count_trainable_parameters(model):,}")
