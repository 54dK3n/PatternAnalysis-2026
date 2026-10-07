"""Reload a model checkpoint and reproduce its held-out development predictions.

This version intentionally has no calibration/final-test scoring
mode. Those phases require a separately implemented and frozen final pipeline.
"""

import argparse
import hashlib
from pathlib import Path
import sys

import torch

from utils.training_controls import checkpoint_controls, attach_controls, validate_device
from models.registry import validate_feature_architecture

from dataset.augmentation import AugmentationConfig
from dataset.preprocessing import checkpoint_preprocessing
from utils.preprocessing_artifacts import export_preprocessing_audit
from dataset import make_loader, load_fold
from evaluation.inference import evaluate
from evaluation.reporting import prediction_report
from evaluation.resources import profile_inference
from utils.evaluation_artifacts import export_evaluation_artifacts
from modules import MODEL_NAMES, create_model, model_minimum_size
from utils.artifacts import validate_output, write_csv, write_json
from utils.runtime import seed_everything, select_device


def run(args):
    """Require the checkpoint's original frozen manifests before any prediction."""
    if args.batch_size < 1 or args.workers < 0 or args.threads < 1:
        raise ValueError("Batch size and threads must be positive; workers cannot be negative.")
    output = validate_output(args.output, args.data_root, args.splits_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config["checkpoint_format_version"] not in (1, 2, 3, 4) or config["model_name"] not in MODEL_NAMES:
        raise ValueError("Unsupported checkpoint format or model architecture.")
    if config["aggregation"] != "mean_slice_AD_probability" or config["threshold"] != 0.5:
        raise ValueError("Unsupported aggregation or decision rule.")
    controls = checkpoint_controls(config)
    preprocessing = checkpoint_preprocessing(config)
    if config["calibration"] != "not_fitted":
        raise ValueError("Unsupported checkpoint calibration.")
    if config["checkpoint_format_version"] == 1:
        if config["augmentation"] != "none":
            raise ValueError("Legacy checkpoints support only unaugmented training.")
    else:
        augmentation = AugmentationConfig.from_dict(config["augmentation_config"])
        if augmentation.name != config["augmentation"]:
            raise ValueError("Checkpoint augmentation name and configuration disagree.")
    validate_feature_architecture(config)
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
    if config.get("initialization", "random") != "random" or config.get("pretrained_weights") is not None:
        raise ValueError("This coursework requires scratch-trained checkpoints.")
    role = getattr(args, "role", None) or (
        "early_stop" if config.get("evaluation_mode") == "inner_only" else "val")
    if role not in ("early_stop", "val"):
        raise ValueError("Prediction supports development early_stop or val only.")
    bins = config.get("calibration_bins", 15)
    reject_threshold = config.get("reject_threshold", 0.8)
    seed_everything(config["seed"])
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    validate_device(controls, device)
    # Architecture dispatch is recorded in the checkpoint, never inferred from its filename.
    model = create_model(config["model_name"], input_channels=controls.input_channels,
                         output_classes=controls.output_classes, drop_path=controls.drop_path).to(device)
    attach_controls(model, controls)
    model.load_state_dict(checkpoint["model_state"])
    loader = make_loader(data[role], args.data_root, tuple(image_size),
                         args.batch_size, args.workers, config["seed"], False, device, role=role, preprocessing=preprocessing)
    scores, slices, scans = evaluate(model, loader, device, config["expected_slices"])
    report, patients = prediction_report(slices, scans, bins, reject_threshold,
                                         patient_aggregation=controls.patient_aggregation)
    scores["patient"] = report["patient_metrics"]
    resources = profile_inference(
        model, tuple(image_size), device, enabled=not getattr(args, "skip_inference_profile", False),
        on_cpu=getattr(args, "profile_on_cpu", False),
        warmup=getattr(args, "profile_warmup", 10), repeats=getattr(args, "profile_repeats", 100))
    output.mkdir(parents=True, exist_ok=False)
    preprocessing_audit = export_preprocessing_audit(output, loader.dataset, "prediction")
    failures = export_evaluation_artifacts(
        output, report, slices, patients, data[role], args.data_root, tuple(image_size), "prediction",
        preprocessing=preprocessing, scan_parameters=loader.dataset.scan_parameters)
    write_csv(output / "slice_predictions.csv", slices)
    write_csv(output / "scan_predictions.csv", scans)
    write_json(output / "metrics.json", {
        "status": "complete", "metrics_format_version": 3,
        "evaluation_role": "development_inner_early_stop" if role == "early_stop" else "development_outer_validation",
        "evaluation_reuses_checkpoint_selection_patients": role == "early_stop",
        "coursework_report": report, "inference_profile": resources, "failure_examples": failures,
        "model_name": config["model_name"], "execution_controls": controls.to_dict(),
        "training_recipe": config.get('training_recipe', 'custom'),
        "fold": config["fold"], "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "manifest_sha256": data["manifest_sha256"], "calibration": "not_fitted",
        "metrics": scores,
        "preprocessing_config": preprocessing.to_dict() if preprocessing.name != "none" else None,
        "preprocessing_audit": preprocessing_audit,
    })
    print(f"Development {role} (primary slice): accuracy={scores['slice']['accuracy']:.4f}, "
          f"macro_F1={scores['slice']['macro_f1']:.4f}, AUROC={scores['slice']['auroc']}")
    print(f"Predictions: {output / 'scan_predictions.csv'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=("early_stop", "val"), default=None,
                        help="Default follows checkpoint mode; neither role is final test.")
    parser.add_argument("--skip-inference-profile", action="store_true")
    parser.add_argument("--profile-on-cpu", action="store_true")
    parser.add_argument("--profile-warmup", type=int, default=10)
    parser.add_argument("--profile-repeats", type=int, default=100)
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
