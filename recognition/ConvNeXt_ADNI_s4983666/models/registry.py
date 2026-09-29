"""Resolve CLI model aliases and construct fresh versioned architectures."""

from .cnn import SmallCNN
from .convnext import ConvNeXtTiny


MODEL_NAMES = ("small_cnn_v1", "convnext_tiny_v1")
MODEL_CHOICES = {
    "small_cnn": "small_cnn_v1",
    "cnn": "small_cnn_v1",
    "convnext_tiny": "convnext_tiny_v1",
    "convnext": "convnext_tiny_v1",
}


def model_minimum_size(name):
    """Return the minimum supported height and width for a versioned model name."""
    if name == "small_cnn_v1":
        return 16
    if name == "convnext_tiny_v1":
        return 32
    raise ValueError(f"Unsupported model name: {name}")


def create_model(name):
    """Construct a fresh model; training folds must never reuse another fold's weights."""
    if name == "small_cnn_v1":
        return SmallCNN()
    if name == "convnext_tiny_v1":
        return ConvNeXtTiny()
    raise ValueError(f"Unsupported model name: {name}")


def count_parameters(model):
    """Count trainable scalar parameters for resource reporting."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
