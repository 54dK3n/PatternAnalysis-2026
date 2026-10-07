"""The small grayscale CNN used for the AD/NC baseline."""

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Map one grayscale slice to an uncalibrated AD logit.

    Group normalization has no running population statistics. Evaluation still
    explicitly switches to eval mode to disable dropout. Global spatial means
    support different image sizes without learning from validation images.
    """

    def __init__(self, input_channels: int = 1, output_classes: int = 1) -> None:
        """Allow controlled grayscale repetition and binary CE without changing defaults."""
        super().__init__()
        if input_channels not in (1, 3) or output_classes not in (1, 2):
            raise ValueError("Input channels must be 1/3 and outputs 1/2.")
        self.input_channels, self.output_classes = input_channels, output_classes
        layers = []
        in_channels = input_channels
        for out_channels in (16, 32, 64, 128):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(4, out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ])
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(128, self.output_classes))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return raw BCE or two-class CE logits for the configured head."""
        if images.ndim != 4 or images.shape[1] != self.input_channels or min(images.shape[-2:]) < 16:
            raise ValueError(f"Expected [batch, {self.input_channels}, height, width] with height/width >= 16.")
        features = self.features(images).mean(dim=(2, 3))
        logits = self.classifier(features)
        return logits.squeeze(1) if self.output_classes == 1 else logits
