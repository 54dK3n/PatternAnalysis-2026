"""Course-facing dataset compatibility interface; implementations stay packaged.

Python resolves the existing dataset/ package for ordinary imports. This file
also supplies the named coursework interface without copying split logic or
changing frozen patient assignments. Direct imports use the same loader,
manifest verification and transforms as the package.
"""

from dataset.augmentation import AugmentationConfig, make_augmentation
from dataset.loaders import make_loader
from dataset.manifests import load_fold
from dataset.slices import ADNISliceDataset

__all__ = ["ADNISliceDataset", "AugmentationConfig", "make_augmentation", "make_loader", "load_fold"]
