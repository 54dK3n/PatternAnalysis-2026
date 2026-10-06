"""Course-facing PyTorch model interface backed by the existing models package.

Component classes are implemented locally in models/cnn.py and
models/convnext.py; see their cited architectural sources. No weights are
downloaded. Legacy CNN/Tiny checkpoint names retain their original meaning.
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
