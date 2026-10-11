"""Course-facing data interface: patient-level splits, transforms and loaders.

The implementation lives in the ``dataset/`` package; see
``dataset/manifests.py`` for the leakage-free role definitions.
"""

from dataset.augmentation import AugmentationConfig, make_augmentation
from dataset.loaders import make_loader
from dataset.manifests import load_fold, load_holdout
from dataset.slices import ADNISliceDataset

__all__ = ["ADNISliceDataset", "AugmentationConfig", "make_augmentation", "make_loader",
           "load_fold", "load_holdout"]
