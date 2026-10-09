"""Course-facing model interface (PyTorch only, no NumPy).

The components are implemented in ``models/cnn.py`` (baseline) and
``models/convnext.py`` (ConvNeXt, Liu et al., CVPR 2022). All models are
randomly initialized; no pretrained weights are downloaded or loaded.
"""

from models import (
    MODEL_CHOICES, MODEL_NAMES, ConvNeXtBlock, ConvNeXtLite, ConvNeXtTiny,
    LayerNorm2d, SmallCNN, StochasticDepth, count_parameters, create_model,
    model_minimum_size,
)

__all__ = [
    "MODEL_CHOICES", "MODEL_NAMES", "ConvNeXtBlock", "ConvNeXtLite", "ConvNeXtTiny",
    "LayerNorm2d", "SmallCNN", "StochasticDepth", "count_parameters", "create_model",
    "model_minimum_size",
]
