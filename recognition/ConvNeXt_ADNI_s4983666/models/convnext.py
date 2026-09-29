"""Native PyTorch ConvNeXt-Tiny with a grayscale stem and binary AD/NC output.

Architecture: Liu et al., "A ConvNet for the 2020s":
https://arxiv.org/abs/2201.03545 . The authors' reference implementation is
https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py .
"""

import torch
from torch import nn


class LayerNorm2d(nn.LayerNorm):
    """Normalize channels independently at each spatial location of an NCHW tensor."""

    def __init__(self, channels):
        super().__init__(channels, eps=1e-6)

    def forward(self, features):
        channels_last = features.permute(0, 2, 3, 1)
        return super().forward(channels_last).permute(0, 3, 1, 2)


class StochasticDepth(nn.Module):
    """Drop a whole residual branch per image during training, preserving its mean."""

    def __init__(self, probability):
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("Stochastic-depth probability must be in [0, 1).")
        self.probability = float(probability)

    def forward(self, residual):
        if not self.training or self.probability == 0.0:
            return residual
        survival = 1.0 - self.probability
        mask_shape = (residual.shape[0],) + (1,) * (residual.ndim - 1)
        mask = residual.new_empty(mask_shape).bernoulli_(survival)
        return residual * (mask / survival)


class ConvNeXtBlock(nn.Module):
    """A depthwise spatial convolution followed by a channel expansion and residual."""

    def __init__(self, channels, drop_probability):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.expand = nn.Linear(channels, 4 * channels)
        self.activation = nn.GELU()
        self.project = nn.Linear(4 * channels, channels)
        self.layer_scale = nn.Parameter(torch.full((channels,), 1e-6))
        self.drop_path = StochasticDepth(drop_probability)

    def forward(self, features):
        residual = self.depthwise(features).permute(0, 2, 3, 1)
        residual = self.project(self.activation(self.expand(self.norm(residual))))
        residual = (residual * self.layer_scale).permute(0, 3, 1, 2)
        return features + self.drop_path(residual)


class ConvNeXtTiny(nn.Module):
    """ConvNeXt-Tiny for grayscale slices normalized to [-1, 1].

    All weights are initialized locally. Stage depths and widths follow Tiny;
    the stochastic-depth probability increases from 0 to 0.1 across 18 blocks.
    Layer normalization uses no learned dataset statistics or running averages.
    The input keeps the dataset's 240 x 256 shape, with no RGB conversion or crop.
    """

    depths = (3, 3, 9, 3)
    channels = (96, 192, 384, 768)

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, self.channels[0], 4, stride=4),
                                  LayerNorm2d(self.channels[0]))
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(LayerNorm2d(previous), nn.Conv2d(previous, current, 2, stride=2))
            for previous, current in zip(self.channels[:-1], self.channels[1:])
        ])
        self.stages = nn.ModuleList()
        block_index = 0
        for channels, depth in zip(self.channels, self.depths):
            blocks = []
            for _ in range(depth):
                probability = 0.1 * block_index / (sum(self.depths) - 1)
                blocks.append(ConvNeXtBlock(channels, probability))
                block_index += 1
            self.stages.append(nn.Sequential(*blocks))
        self.final_norm = nn.LayerNorm(self.channels[-1], eps=1e-6)
        self.classifier = nn.Linear(self.channels[-1], 1)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, images):
        """Return a binary logit per slice; spatial dimensions may be non-square."""
        if (not isinstance(images, torch.Tensor) or images.ndim != 4
                or images.shape[0] < 1 or images.shape[1] != 1
                or min(images.shape[-2:]) < 32 or not images.is_floating_point()):
            raise ValueError(
                "Expected floating-point [batch, 1, height, width] with batch >= 1 "
                "and height/width >= 32."
            )
        features = self.stages[0](self.stem(images))
        for downsample, stage in zip(self.downsample_layers, self.stages[1:]):
            features = stage(downsample(features))
        pooled = self.final_norm(features.mean(dim=(2, 3)))
        return self.classifier(pooled).squeeze(1)
