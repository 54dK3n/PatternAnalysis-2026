"""Experiment provenance, protected output paths, tables, and learning curves."""

import csv
import hashlib
import json
import os
from pathlib import Path

def write_json(path, value):
    """Reject NaN/Infinity instead of writing nonstandard JSON."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    """Export auditable predictions or history with explicit column names."""
    if not rows:
        raise ValueError("Cannot export an empty result table.")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def code_fingerprints():
    """Identify the exact source files used, even outside a Git checkout."""
    root = Path(__file__).resolve().parents[1]
    sources = [root / name for name in ("adni_splits.py", "train.py", "predict.py")]
    for package in ("models", "dataset", "engine", "evaluation", "utils"):
        sources.extend((root / package).rglob("*.py"))
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(sources)}


def validate_output(output, data_root, splits_dir):
    """Protect source data and frozen manifests from generated experiment artifacts."""
    output = Path(output).resolve()
    for protected in (Path(data_root).resolve(), Path(splits_dir).resolve()):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("Run output must be separate from source data and split directories.")
    if output.exists():
        raise ValueError(f"Output already exists; refusing to overwrite an experiment: {output}")
    return output


def plot_history(output, history):
    """Write a standalone plot containing training and early-stop data only."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(output) / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].plot(epochs, [r["train_slice_loss"] for r in history], marker="o", label="Train slice BCE (class-weighted)")
    axes[0].plot(epochs, [r["early_stop_scan_loss"] for r in history], marker="o", label="Early-stop scan log loss")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training and checkpoint selection")
    axes[1].plot(epochs, [r["early_stop_scan_accuracy"] for r in history], marker="o", label="Scan accuracy")
    axes[1].plot(epochs, [r["early_stop_scan_macro_f1"] for r in history], marker="o", label="Scan macro F1")
    axes[1].set(xlabel="Epoch", ylabel="Score", ylim=(0, 1), title="Early-stop patients only")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.savefig(Path(output) / "learning_curves.png", dpi=160)
    plt.close(figure)
