"""Reload a model checkpoint and reproduce its held-out development predictions.

This version intentionally has no calibration/final-test scoring
mode. Those phases require a separately implemented and frozen final pipeline.
"""

import argparse
import hashlib
from pathlib import Path
import sys

import torch

from dataset import load_fold
from modules import MODEL_NAMES, create_model, model_minimum_size
from training_utils import evaluate, make_loader, seed_everything, select_device, validate_output, write_csv, write_json


def run(args):
    """Require the checkpoint's original frozen manifests before any prediction."""
    if args.batch_size < 1 or args.workers < 0 or args.threads < 1:
        raise ValueError("Batch size and threads must be positive; workers cannot be negative.")
    output = validate_output(args.output, args.data_root, args.splits_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config["checkpoint_format_version"] != 1 or config["model_name"] not in MODEL_NAMES:
        raise ValueError("Unsupported checkpoint format or model architecture.")
    if config["aggregation"] != "mean_slice_AD_probability" or config["threshold"] != 0.5:
        raise ValueError("Unsupported aggregation or decision rule.")
    if (config["normalization"] != "(grayscale_uint8 / 255 - 0.5) / 0.5"
            or config["augmentation"] != "none" or config["calibration"] != "not_fitted"):
        raise ValueError("Unsupported checkpoint preprocessing or calibration.")
    image_size = config["image_size"]
    minimum_size = model_minimum_size(config["model_name"])
    if (not isinstance(image_size, (list, tuple)) or len(image_size) != 2
            or any(type(value) is not int or value < minimum_size for value in image_size)):
        raise ValueError(f"Checkpoint image height and width must be integers of at least {minimum_size}.")
    data = load_fold(args.data_root, args.splits_dir, int(config["fold"]))
    if data["manifest_sha256"] != config["manifest_sha256"]:
        raise ValueError("Checkpoint belongs to a different set of frozen manifests.")
    if int(data["report"]["config"]["expected_slices"]) != config["expected_slices"]:
        raise ValueError("Checkpoint slice count differs from the audited split.")
    seed_everything(config["seed"])
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    # Architecture dispatch is recorded in the checkpoint, never inferred from its filename.
    model = create_model(config["model_name"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    loader = make_loader(data["val"], args.data_root, tuple(image_size),
                         args.batch_size, args.workers, config["seed"], False, device)
    scores, slices, scans = evaluate(model, loader, device, config["expected_slices"])
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "slice_predictions.csv", slices)
    write_csv(output / "scan_predictions.csv", scans)
    write_json(output / "metrics.json", {
        "status": "complete", "evaluation_role": "development_outer_validation",
        "model_name": config["model_name"],
        "fold": config["fold"], "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "manifest_sha256": data["manifest_sha256"], "calibration": "not_fitted",
        "metrics": scores,
    })
    print(f"Validation scans: {len(scans)}; accuracy={scores['scan']['accuracy']:.4f}, "
          f"macro_F1={scores['scan']['macro_f1']:.4f}, AUROC={scores['scan']['auroc']}")
    print(f"Predictions: {output / 'scan_predictions.csv'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    try:
        run(parser.parse_args(argv))
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
