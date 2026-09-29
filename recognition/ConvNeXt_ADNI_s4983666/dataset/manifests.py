"""Load one verified development fold from frozen patient manifests."""

import argparse
from collections import Counter
from pathlib import Path

from . import splits as adni_splits


def manifest_sha256(splits_dir):
    """Fingerprint the frozen manifest set for checkpoints and run records."""
    return adni_splits.digest((Path(splits_dir) / "COMPLETED.json").read_bytes())


def load_fold(data_root, splits_dir, fold):
    """Fully audit the sources before exposing one development fold.

    Verification is mandatory on every invocation. The source folders' old
    train/test names are provenance only; role membership comes exclusively
    from the newly audited patient manifests.
    """
    data_root, splits_dir = Path(data_root).resolve(), Path(splits_dir).resolve()
    adni_splits.verify(argparse.Namespace(data_root=data_root, output=splits_dir))
    seal_bytes = (splits_dir / "COMPLETED.json").read_bytes()
    seal = adni_splits.read_json(splits_dir / "COMPLETED.json")

    def read_verified(path, reader):
        relative = path.relative_to(splits_dir).as_posix()
        expected = seal["sha256"][relative]
        adni_splits.require(adni_splits.digest(path.read_bytes()) == expected,
                            f"Manifest changed after verification: {relative}")
        value = reader(path)
        adni_splits.require(adni_splits.digest(path.read_bytes()) == expected,
                            f"Manifest changed while being read: {relative}")
        return value

    report = read_verified(splits_dir / "report.json", adni_splits.read_json)
    folds = report["config"]["folds"]
    adni_splits.require(type(fold) is int and 1 <= fold <= folds,
                        f"fold must be an integer between 1 and {folds}.")
    expected_slices = report["config"]["expected_slices"]
    result = {"report": report, "manifest_sha256": adni_splits.digest(seal_bytes)}
    for role in ("train", "early_stop", "val"):
        path = splits_dir / f"fold_{fold:02d}" / f"{role}.csv"
        rows = read_verified(path, lambda value: adni_splits.read_csv(value, adni_splits.FIELDS))
        adni_splits.require(rows and all(row["partition"] == "development" for row in rows),
                            f"Only nonempty development manifests are allowed for {role}.")
        scans = Counter(row["image_id"] for row in rows)
        adni_splits.require(all(count == expected_slices for count in scans.values()),
                            f"The {role} manifest contains an incomplete scan.")
        result[role] = rows
    adni_splits.require((splits_dir / "COMPLETED.json").read_bytes() == seal_bytes,
                        "The completion marker changed while loading the fold.")
    return result
