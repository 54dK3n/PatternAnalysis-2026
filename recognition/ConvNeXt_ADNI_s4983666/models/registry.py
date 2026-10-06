"""Resolve CLI model aliases and construct fresh versioned architectures."""

from .cnn import SmallCNN
from .convnext import (ConvNeXtLite, ConvNeXtTiny, ConvNeXtLiteOverlap,
                       ConvNeXtLiteStride2, ConvNeXtLiteNoDrop)


MODEL_NAMES = ("small_cnn_v1", "convnext_tiny_v1", "convnext_lite_v1",
               "convnext_lite_overlap_v1", "convnext_lite_stride2_v1", "convnext_lite_nodrop_v1")
MODEL_CHOICES = {
    "small_cnn": "small_cnn_v1",
    "cnn": "small_cnn_v1",
    "convnext_tiny": "convnext_tiny_v1",
    "convnext": "convnext_tiny_v1",
    "convnext_lite": "convnext_lite_v1",
}


def model_minimum_size(name):
    """Return the minimum supported height and width for a versioned model name."""
    if name == "small_cnn_v1":
        return 16
    if name in MODEL_NAMES[1:]:
        return 32
    raise ValueError(f"Unsupported model name: {name}")


def create_model(name):
    """Construct a fresh model; training folds must never reuse another fold's weights."""
    if name == "small_cnn_v1":
        return SmallCNN()
    if name == "convnext_tiny_v1":
        return ConvNeXtTiny()
    if name == "convnext_lite_v1":
        return ConvNeXtLite()
    variants = {"convnext_lite_overlap_v1": ConvNeXtLiteOverlap,
                "convnext_lite_stride2_v1": ConvNeXtLiteStride2,
                "convnext_lite_nodrop_v1": ConvNeXtLiteNoDrop}
    if name in variants:
        return variants[name]()
    raise ValueError(f"Unsupported model name: {name}")


def count_parameters(model):
    """Count trainable scalar parameters for resource reporting."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def validate_feature_architecture(config: dict) -> None:
    """Require complete geometry metadata for the new versioned architecture names."""
    variants = {"convnext_lite_overlap_v1": ConvNeXtLiteOverlap,
                "convnext_lite_stride2_v1": ConvNeXtLiteStride2,
                "convnext_lite_nodrop_v1": ConvNeXtLiteNoDrop}
    cls = variants.get(config["model_name"])
    if cls is None:
        return
    expected = {"depths": list(cls.depths), "channels": list(cls.channels),
                "stem": {"kernel": cls.stem_kernel, "stride": cls.stem_stride, "padding": cls.stem_padding},
                "max_drop_path": cls.max_drop_path,
                "stage_effective_strides": [cls.stem_stride * 2**i for i in range(4)]}
    if config.get("model_architecture") != expected:
        raise ValueError("New checkpoint architecture metadata differs from its versioned model.")
