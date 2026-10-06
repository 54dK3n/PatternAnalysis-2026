"""Seeded data loaders with explicit training and evaluation transform roles."""

import os
import random
import sys

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .slices import ADNISliceDataset
from .sampling import sampling_weights


def _prevent_macos_tracker_inheritance():
    """Keep PyTorch's shared-memory manager from inheriting the tracker pipe.

    On macOS with Python 3.12+, an inherited pipe can prevent the resource
    tracker from exiting while the parent waits for it during finalization.
    Marking the worker's descriptor close-on-exec retains resource tracking
    in the worker and excludes unrelated exec'd shared-memory managers.
    See https://github.com/pytorch/pytorch/issues/153050 .
    """
    if sys.platform == "darwin" and sys.version_info >= (3, 12):
        from multiprocessing.resource_tracker import getfd
        os.set_inheritable(getfd(), False)


def seed_worker(worker_id):
    """Preserve the Python seed and prevent a macOS tracker shutdown deadlock."""
    _prevent_macos_tracker_inheritance()
    random.seed(torch.initial_seed() % 2**32)


def make_loader(rows, data_root, image_size, batch_size, workers, seed, shuffle, device,
                *, role="evaluation", augmentation=None, sampling="slice_uniform",
                preprocessing=None, scan_parameters=None):
    """Keep all samples and apply only the explicitly requested role's transforms.

    Shuffling never enables augmentation. The original loader generator and
    worker seeding are unchanged when the augmentation profile is none.
    """
    sampler = None
    if sampling != "slice_uniform":
        if role != "train" or not shuffle:
            raise ValueError("Balanced sampling is permitted only for shuffled training.")
        weights, _ = sampling_weights(rows, sampling)
        sampler = WeightedRandomSampler(weights, len(rows), replacement=True,
                                        generator=torch.Generator().manual_seed(seed + 1000003))
        shuffle = False
    return DataLoader(
        ADNISliceDataset(rows, data_root, image_size=image_size, role=role,
                         augmentation=augmentation, augmentation_seed=seed,
                         preprocessing=preprocessing, scan_parameters=scan_parameters),
        batch_size=batch_size, shuffle=shuffle, sampler=sampler, drop_last=False, num_workers=workers,
        pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
