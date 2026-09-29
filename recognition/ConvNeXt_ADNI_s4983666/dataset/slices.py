"""Load immutable source slices with explicit role-specific processing."""

import io
from pathlib import Path
import random

from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, get_worker_info

from . import splits as adni_splits
from .augmentation import AugmentationConfig, augment_image


class ADNISliceDataset(Dataset):
    """Load one slice per item with fixed normalization and optional train augmentation.

    ``image_size`` is (height, width). Pixel values are mapped from [0, 255]
    to [-1, 1] with fixed constants; no training, validation, calibration, or
    test population statistics are estimated by this transform.
    """

    def __init__(self, rows, data_root, image_size=(240, 256), *, role="evaluation",
                 augmentation=None, augmentation_seed=0):
        self.data_root = Path(data_root).resolve()
        adni_splits.require(self.data_root.is_dir(), f"Missing data directory: {self.data_root}")
        adni_splits.require(len(image_size) == 2 and all(type(n) is int and n > 0 for n in image_size),
                            "image_size must contain positive integer height and width.")
        self.image_size = tuple(image_size)
        self.rows = [dict(row) for row in rows]
        adni_splits.require(self.rows, "A slice dataset cannot be empty.")
        adni_splits.require(role in ("train", "evaluation", "early_stop", "val", "calibration", "test"),
                            f"Unsupported dataset role: {role!r}")
        self.role = role
        self.augmentation = AugmentationConfig() if augmentation is None else augmentation
        adni_splits.require(isinstance(self.augmentation, AugmentationConfig),
                            "augmentation must be an AugmentationConfig.")
        adni_splits.require(type(augmentation_seed) is int, "augmentation_seed must be an integer.")
        self.augmentation_seed = augmentation_seed
        if self.augmentation.name != "none":
            adni_splits.require(role == "train", "Augmentation is permitted only for the train role.")
            adni_splits.require(all(row.get("partition") == "development" for row in self.rows),
                                "Augmentation requires development training manifests.")
        self._augmentation_rng = None
        self._augmentation_worker = None
        self.paths = []
        seen_paths, seen_resolved_paths, seen_slices = set(), set(), set()
        scan_owners = {}
        for row in self.rows:
            relative = row["relative_path"]
            adni_splits.require(isinstance(relative, str) and bool(relative),
                                "Every slice must have a nonempty relative path.")
            path = Path(relative)
            adni_splits.require(not path.is_absolute() and ".." not in path.parts,
                                f"Slice paths must stay inside the data directory: {relative}")
            resolved = (self.data_root / path).resolve()
            adni_splits.require(resolved.is_relative_to(self.data_root) and resolved.is_file(),
                                f"Slice path is missing or outside the data directory: {relative}")
            adni_splits.require(relative not in seen_paths and resolved not in seen_resolved_paths,
                                f"Duplicate slice path: {relative}")
            patient_id, image_id = row["patient_id"], row["image_id"]
            adni_splits.require(isinstance(patient_id, str) and bool(patient_id)
                                and isinstance(image_id, str) and bool(image_id),
                                "Patient and scan identifiers must be nonempty strings.")
            adni_splits.require(str(row["label"]) in ("0", "1"),
                                f"Binary labels must be NC=0 or AD=1: {relative}")
            label = int(row["label"])
            slice_index = int(row["slice_index"])
            adni_splits.require(str(slice_index) == str(row["slice_index"]) and slice_index >= 0,
                                f"Invalid nonnegative slice index: {relative}")
            scan_key = (image_id, slice_index)
            adni_splits.require(scan_key not in seen_slices,
                                f"Duplicate scan/slice identifier: {scan_key}")
            owner = (patient_id, label)
            adni_splits.require(image_id not in scan_owners or scan_owners[image_id] == owner,
                                f"One scan has inconsistent patients or labels: {image_id}")
            scan_owners[image_id] = owner
            seen_paths.add(relative)
            seen_resolved_paths.add(resolved)
            seen_slices.add(scan_key)
            self.paths.append(resolved)

    def __len__(self):
        return len(self.rows)

    def _rng(self):
        """Maintain a private RNG stream for this process and loader worker seed."""
        worker = get_worker_info()
        identity = ("main", self.augmentation_seed) if worker is None else ("worker", worker.id, worker.seed)
        if self._augmentation_worker != identity:
            seed = self.augmentation_seed if worker is None else worker.seed
            self._augmentation_rng = random.Random(seed)
            self._augmentation_worker = identity
        return self._augmentation_rng

    def __getitem__(self, index):
        row = self.rows[index]
        content = self.paths[index].read_bytes()
        # The startup audit validates all sources. Check this slice again to
        # detect source changes after that audit without rereading other data.
        if "file_sha256" in row:
            adni_splits.require(adni_splits.digest(content) == row["file_sha256"],
                                f"Source image changed after verification: {row['relative_path']}")
        with Image.open(io.BytesIO(content)) as source:
            image = ImageOps.exif_transpose(source).convert("L")
            target_size = (self.image_size[1], self.image_size[0])
            if image.size != target_size:
                image = image.resize(target_size, resample=Image.Resampling.BILINEAR)
            if self.augmentation.name != "none":
                image = augment_image(image, self.augmentation, self._rng())
            # bytearray supplies writable storage for frombuffer, and to()
            # creates an independent float tensor before the buffer expires.
            pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            tensor = pixels.to(dtype=torch.float32).reshape(1, *self.image_size)
            tensor = tensor.div(255.0).sub(0.5).div(0.5)
        return {
            "image": tensor,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "patient_id": row["patient_id"],
            "image_id": row["image_id"],
            "slice_index": int(row["slice_index"]),
            "relative_path": row["relative_path"],
        }
