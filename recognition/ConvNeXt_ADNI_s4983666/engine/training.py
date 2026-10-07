"""Train from scratch on a frozen fold; default evaluation scores outer val once.

Use --inner-only for exploratory early-stop scoring without outer validation.

The calibration and final-test sets are used only by the source integrity audit;
their images/labels never enter training, checkpoint selection, or model scoring.
"""

import argparse
from collections import Counter
import math
from pathlib import Path
import sys
import time

import torch
from torch import nn
from typing import Any
from utils.training_controls import (ExecutionControls, RecipeParser, add_execution_arguments,
    controls_from_args, attach_controls, validate_device, model_input, autocast_context,
    ad_logits, make_criterion, model_controls)

from dataset.augmentation import AUGMENTATION_NAMES, make_augmentation
from dataset.preprocessing import (SCAN_NORMALIZATION, add_preprocessing_arguments,
                                   fit_training_crop, prepare_scans, preprocessing_from_args)
from utils.preprocessing_artifacts import export_preprocessing_audit
from dataset import make_loader, load_fold
from dataset.sampling import SAMPLING_NAMES, sampling_weights
from engine.scheduling import learning_rate
from evaluation.inference import evaluate
from evaluation.metrics import binary_metrics
from evaluation.reporting import prediction_report
from evaluation.resources import cuda_peak_mib, profile_inference
from utils.evaluation_artifacts import export_evaluation_artifacts
from modules import MODEL_CHOICES, create_model, count_parameters, model_minimum_size
from utils.artifacts import code_fingerprints, plot_history, validate_output, write_csv, write_json
from utils.runtime import environment_info, seed_everything, select_device, sync_device


def train_epoch(model: nn.Module, loader: Any, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device, metrics_sink: dict | None = None) -> float:
    """Update weights using only training slices and fail on non-finite loss."""
    model.train()
    total_loss, count = 0.0, 0
    observed_labels, observed_probabilities = [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        labels = batch["label"].to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(model, device):
            logits = model(model_input(model, images))
            targets = labels.long() if model_controls(model).output_classes == 2 else labels
            loss = criterion(logits, targets)
        if not torch.isfinite(loss).item():
            raise ValueError("Training produced non-finite loss; no evaluation will be published.")
        scaler = getattr(model, '_grad_scaler', None)
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.detach().item() * labels.numel()
        count += labels.numel()
        if metrics_sink is not None:
            observed_labels.extend(int(v) for v in labels.detach().cpu().tolist())
            observed_probabilities.extend(torch.sigmoid(ad_logits(model, logits.detach())).cpu().tolist())
    if metrics_sink is not None:
        metrics_sink.update(binary_metrics(observed_labels, observed_probabilities))
    return total_loss / count


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Keep checkpoint selection and outer-fold evaluation in separate phases."""
    # Preserve the original default for CLI users and existing Python callers.
    requested_model = getattr(args, "model", "small_cnn")
    if requested_model not in MODEL_CHOICES:
        raise ValueError(f"Unsupported model: {requested_model}")
    model_name = MODEL_CHOICES[requested_model]
    controls = controls_from_args(args)
    if model_name == 'small_cnn_v1' and controls.drop_path is not None:
        raise ValueError('DropPath applies to ConvNeXt only.')
    if args.patience is None:
        args.patience = args.epochs
    minimum_size = model_minimum_size(model_name)
    augmentation = make_augmentation(
        getattr(args, "augmentation", "none"),
        rotation_degrees=getattr(args, "rotation_degrees", 5.0),
        translation_fraction=getattr(args, "translation_fraction", 0.03),
        translation_pixels=getattr(args, "translation_pixels", 4),
    )
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.workers < 0 or args.threads < 1:
        raise ValueError("Epochs, patience, batch size, and threads must be positive; workers cannot be negative.")
    if min(args.image_height, args.image_width) < minimum_size:
        raise ValueError(f"Image height and width must be at least {minimum_size} for {requested_model}.")
    if not all(math.isfinite(v) for v in (args.lr, args.weight_decay, args.min_delta)):
        raise ValueError("Optimizer and early-stopping settings must be finite.")
    if args.lr <= 0 or args.weight_decay < 0 or args.min_delta < 0:
        raise ValueError("Learning rate must be positive; weight decay and min delta cannot be negative.")

    bins = getattr(args, "calibration_bins", 15)
    reject_threshold = getattr(args, "reject_threshold", 0.8)
    inner_only = getattr(args, "inner_only", False)
    profile_warmup = getattr(args, "profile_warmup", 10)
    profile_repeats = getattr(args, "profile_repeats", 100)
    if type(bins) is not int or bins < 1 or not math.isfinite(reject_threshold) or not 0.5 <= reject_threshold <= 1:
        raise ValueError("Calibration bins must be positive and rejection confidence must be in [0.5, 1].")
    if min(profile_warmup, profile_repeats) < 1:
        raise ValueError("Profile warmup and repeats must be positive.")
    sampling_name = getattr(args, "sampling", "slice_uniform")
    if sampling_name not in SAMPLING_NAMES:
        raise ValueError("Unsupported training sampling mode.")
    schedule_name = getattr(args, "lr_schedule", "constant")
    warmup_epochs = getattr(args, "warmup_epochs", 2)
    min_lr_ratio = getattr(args, "min_lr_ratio", 0.01)
    learning_rate(1, args.epochs, args.lr, schedule_name, warmup_epochs, min_lr_ratio)
    preprocessing = preprocessing_from_args(args)
    device = select_device(args.device)
    validate_device(controls, device)
    output = validate_output(args.output, args.data_root, args.splits_dir)
    data = load_fold(args.data_root, args.splits_dir, args.fold)
    expected_slices = int(data["report"]["config"]["expected_slices"])
    seed = args.seed + args.fold
    seed_everything(seed)
    torch.set_num_threads(args.threads)
    image_size = (args.image_height, args.image_width)
    preprocessing_started = time.perf_counter()
    train_scan_parameters = None
    crop_fit = {"status": "not_required"}
    if preprocessing.name != "none":
        train_scan_parameters = prepare_scans(data["train"], args.data_root, preprocessing)
        preprocessing, crop_fit = fit_training_crop(train_scan_parameters, preprocessing,
                                                    role="train", minimum_size=minimum_size)
        if preprocessing.crops:
            image_size = (preprocessing.crop_height, preprocessing.crop_width)

    preprocessing_seconds = time.perf_counter() - preprocessing_started
    # No resume option: every run/fold creates an independent model and optimizer.
    model = create_model(model_name, input_channels=controls.input_channels,
                         output_classes=controls.output_classes, drop_path=controls.drop_path).to(device)
    attach_controls(model, controls)
    if controls.precision == 'amp_fp16':
        model._grad_scaler = torch.amp.GradScaler('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_counts = Counter(int(row["label"]) for row in data["train"])
    if class_counts[0] == 0 or class_counts[1] == 0:
        raise ValueError("Training must contain both AD and NC slices.")
    _, sampling_config = sampling_weights(data["train"], sampling_name)
    # Balancing both the sampler and the loss would apply the class correction twice.
    pos_weight = class_counts[0] / class_counts[1] if sampling_name == "slice_uniform" else 1.0
    if controls.loss not in ('weighted_bce', 'weighted_cross_entropy'):
        pos_weight = 1.0
    criterion = make_criterion(controls, pos_weight, device)
    loaders_started = time.perf_counter()
    loader_args = (args.data_root, image_size, args.batch_size, args.workers, seed)
    train_loader = make_loader(data["train"], *loader_args, shuffle=True, device=device,
                               role="train", augmentation=augmentation, sampling=sampling_name,
                               preprocessing=preprocessing, scan_parameters=train_scan_parameters)
    early_loader = make_loader(data["early_stop"], *loader_args, shuffle=False, device=device,
                               role="early_stop", preprocessing=preprocessing)

    preprocessing_seconds += time.perf_counter() - loaders_started
    config = {
        "checkpoint_format_version": 2, "metrics_format_version": 3, "model_name": model_name,
        "execution_controls": controls.to_dict(), "training_recipe": getattr(args, 'recipe', 'custom'),
        "model_architecture": {"depths": list(model.depths), "channels": list(model.channels)}
        if hasattr(model, "depths") else {"name": model_name},
        "primary_evaluation_unit": "slice", "patient_separation": "frozen_patient_manifests",
        "evaluation_mode": "inner_only" if inner_only else "outer_validation_once",
        "calibration_bins": bins, "reject_threshold": reject_threshold,
        "rejection_threshold_source": "declared_before_evaluation_not_fitted",
        "epoch_budget_source": "experiment_configuration_not_course_requirement",
        "inference_profile": {"enabled": not getattr(args, "skip_inference_profile", False),
                              "on_cpu": getattr(args, "profile_on_cpu", False),
                              "batch_sizes": [1, 64], "warmup": profile_warmup, "repeats": profile_repeats},
        "initialization": "random", "pretrained_weights": None,
        "fold": args.fold, "seed": seed, "seed_base": args.seed,
        "image_size": list(image_size), "expected_slices": expected_slices,
        "normalization": "(grayscale_uint8 / 255 - 0.5) / 0.5",
        "augmentation": augmentation.name, "augmentation_config": augmentation.to_dict(),
        "aggregation": "mean_slice_AD_probability",
        "threshold": 0.5, "calibration": "not_fitted",
        "checkpoint_selection": "minimum_early_stop_scan_log_loss",
        "manifest_sha256": data["manifest_sha256"],
        "split_config": data["report"]["config"],
        "train_slice_class_counts": {str(k): v for k, v in class_counts.items()},
        "train_pos_weight": pos_weight,
        "epochs_limit": args.epochs, "patience": args.patience, "min_delta": args.min_delta,
        "lr": args.lr, "weight_decay": args.weight_decay,
        "lr_schedule": {"name": schedule_name, "warmup_epochs": warmup_epochs if schedule_name == "warmup_cosine" else 0,
                        "min_lr_ratio": min_lr_ratio if schedule_name != "constant" else 1.0},
        "training_sampling": sampling_config,
        "batch_size": args.batch_size, "workers": args.workers, "threads": args.threads,
        "data_root": str(args.data_root.resolve()), "splits_dir": str(args.splits_dir.resolve()),
        "code_sha256": code_fingerprints(), "environment": environment_info(device),
    }
    if preprocessing.name != "none":
        config.update(checkpoint_format_version=3, normalization=SCAN_NORMALIZATION,
                      preprocessing_config=preprocessing.to_dict(), preprocessing_crop_fit=crop_fit,
                      preprocessing_preparation_seconds=preprocessing_seconds)
    if controls != ExecutionControls():
        config['checkpoint_format_version'] = 4
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    preprocessing_audits = {}
    if preprocessing.name != "none":
        preprocessing_audits = {role: export_preprocessing_audit(output, loader.dataset, role)
                                for role, loader in (("train", train_loader), ("early_stop", early_loader))}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    history, best_loss, patience_loss, stale_epochs, best_epoch = [], math.inf, math.inf, 0, 0
    epoch_metrics = []
    training_peak = evaluation_peak = 0.0 if device.type == "cuda" else None
    print(f"Training {model_name}, augmentation={augmentation.name}, fold {args.fold} on {device}; "
          f"loss={controls.loss}, precision={controls.precision}, input_channels={controls.input_channels}; "
          "selection uses early-stop scans only.", flush=True)

    for epoch in range(1, args.epochs + 1):
        epoch_lr = learning_rate(epoch, args.epochs, args.lr, schedule_name, warmup_epochs, min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = epoch_lr
        sync_device(device)
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        online_metrics = {}
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, online_metrics)
        sync_device(device)
        train_seconds = time.perf_counter() - epoch_started
        if device.type == "cuda":
            training_peak = max(training_peak, cuda_peak_mib(device))
            torch.cuda.reset_peak_memory_stats(device)
        early_scores, _, _ = evaluate(model, early_loader, device, expected_slices)
        if device.type == "cuda":
            evaluation_peak = max(evaluation_peak, cuda_peak_mib(device))
        scan_scores = early_scores["scan"]
        selection_loss = scan_scores["log_loss"]
        improved = selection_loss < best_loss
        if improved:
            best_loss, best_epoch = selection_loss, epoch
            # Checkpoints contain tensors and plain data, compatible with weights_only.
            checkpoint = {
                "config": config, "epoch": epoch, "early_stop_scan_loss": selection_loss,
                "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            }
            temporary = output / "best.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(output / "best.pt")
        # Save every strict minimum; min_delta affects patience, not which
        # checkpoint is reported as the minimum observed early-stop loss.
        if selection_loss < patience_loss - args.min_delta:
            patience_loss, stale_epochs = selection_loss, 0
        else:
            stale_epochs += 1
        history.append({
            "epoch": epoch, "learning_rate": epoch_lr, "train_slice_loss": train_loss,
            "train_slice_accuracy": online_metrics["accuracy"],
            "train_slice_macro_f1": online_metrics["macro_f1"],
            "train_slice_auroc": online_metrics["auroc"],
            "train_metrics_scope": "online_training_mode_augmented_when_configured",
            "train_seconds": train_seconds,
            "early_stop_slice_loss": early_scores["slice"]["log_loss"],
            "early_stop_slice_accuracy": early_scores["slice"]["accuracy"],
            "early_stop_slice_macro_f1": early_scores["slice"]["macro_f1"],
            "early_stop_slice_auroc": early_scores["slice"]["auroc"],
            "early_stop_scan_loss": selection_loss,
            "early_stop_scan_accuracy": scan_scores["accuracy"],
            "early_stop_scan_macro_f1": scan_scores["macro_f1"],
            "early_stop_scan_auroc": scan_scores["auroc"],
            "selected_checkpoint": improved, "epoch_seconds": time.perf_counter() - epoch_started,
        })
        epoch_metrics.append({"epoch": epoch, "learning_rate": epoch_lr, "train_online_slice": online_metrics,
                              "early_stop_slice": early_scores["slice"], "early_stop_scan": scan_scores})
        write_json(output / "epoch_metrics.json", epoch_metrics)
        write_csv(output / "history.csv", history)
        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, "
              f"early_stop_scan_loss={selection_loss:.4f}, "
              f"early_stop_slice_accuracy={early_scores['slice']['accuracy']:.4f}, "
              f"early_stop_slice_AUROC={early_scores['slice']['auroc']}", flush=True)
        if stale_epochs >= args.patience:
            print(f"Early stopping after {epoch} epochs; selected epoch {best_epoch}.", flush=True)
            break

    # Release optimizer state before measuring inference memory; retain training peaks.
    del optimizer
    model.zero_grad(set_to_none=True)
    # Outer validation is constructed/scored only after the checkpoint is fixed.
    # Exploratory inner-only runs never construct or score an outer-val loader.
    selected = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(selected["model_state"])
    evaluation_role = "early_stop" if inner_only else "val"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    validation_loader = early_loader if inner_only else make_loader(
        data["val"], *loader_args, shuffle=False, device=device, role="val", preprocessing=preprocessing)
    scores, slices, scans = evaluate(model, validation_loader, device, expected_slices)
    if device.type == "cuda":
        evaluation_peak = max(evaluation_peak, cuda_peak_mib(device))
    training_and_evaluation_seconds = time.perf_counter() - started
    report, patients = prediction_report(slices, scans, bins, reject_threshold,
                                         patient_aggregation=controls.patient_aggregation)
    scores["patient"] = report["patient_metrics"]
    inference_profile = profile_inference(
        model, image_size, device, warmup=profile_warmup, repeats=profile_repeats,
        enabled=not getattr(args, "skip_inference_profile", False),
        on_cpu=getattr(args, "profile_on_cpu", False))
    sync_device(device)
    result = {
        "status": "complete", "metrics_format_version": 3,
        "evaluation_role": "development_inner_early_stop" if inner_only else "development_outer_validation",
        "evaluation_reuses_checkpoint_selection_patients": inner_only,
        "coursework_report": report,
        "model_name": model_name, "execution_controls": controls.to_dict(),
        "training_recipe": getattr(args, 'recipe', 'custom'),
        "augmentation": augmentation.name,
        "fold": args.fold, "best_epoch": best_epoch, "epochs_completed": len(history),
        "best_early_stop_scan_loss": best_loss, "manifest_sha256": data["manifest_sha256"],
        "aggregation": config["aggregation"], "threshold": 0.5, "calibration": "not_fitted",
        "metrics": scores,
        "resources": {
            "trainable_parameters": count_parameters(model),
            "training_and_evaluation_seconds": training_and_evaluation_seconds,
            "training_seconds": math.fsum(r["train_seconds"] for r in history),
            "training_peak_cuda_allocated_mib": training_peak,
            "evaluation_peak_cuda_allocated_mib": evaluation_peak,
            "device": config["environment"]["device_name"],
            "inference_profile": inference_profile,
            "peak_cuda_allocated_mib": max(training_peak, evaluation_peak) if device.type == "cuda" else None,
            "timing_note": "Forward timing excludes data loading and transfer; mean per slice at configured batch size.",
            "memory_note": "Training and early-stop peaks include optimizer state; separate inference profile follows its release.",
        },
    }
    if preprocessing.name != "none":
        if not inner_only:
            preprocessing_audits["val"] = export_preprocessing_audit(output, validation_loader.dataset, "val")
        result.update(preprocessing_config=preprocessing.to_dict(), preprocessing_crop_fit=crop_fit,
                      preprocessing_audits=preprocessing_audits,
                      preprocessing_preparation_seconds=config["preprocessing_preparation_seconds"])
    prefix = "early_stop" if inner_only else "val"
    write_csv(output / f"{prefix}_slice_predictions.csv", slices)
    write_csv(output / f"{prefix}_scan_predictions.csv", scans)
    result["failure_examples"] = export_evaluation_artifacts(
        output, report, slices, patients, data[evaluation_role], args.data_root, image_size, prefix,
        preprocessing=preprocessing, scan_parameters=validation_loader.dataset.scan_parameters)
    plot_history(output, history)
    # This marker is written last; partial/failed runs have no completed result.
    write_json(output / "metrics.json", result)
    print(f"{result['evaluation_role']} (primary slice): accuracy={scores['slice']['accuracy']:.4f}, "
          f"macro_F1={scores['slice']['macro_f1']:.4f}, AUROC={scores['slice']['auroc']}", flush=True)
    print("Final-test target remains unassessed; these are development results.", flush=True)
    print(f"Completed. Results: {output / 'metrics.json'}", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    """Parse explicit options over recipe defaults and execute one fresh run."""
    parser = RecipeParser(description=__doc__)
    add_execution_arguments(parser)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(MODEL_CHOICES), default="small_cnn",
                        help="Fresh randomly initialized architecture (default: small_cnn).")
    add_preprocessing_arguments(parser)
    parser.add_argument("--augmentation", choices=AUGMENTATION_NAMES, default="none",
                        help="Training only: none, legacy light, isolated shifts/gamma, or integer_gamma.")
    parser.add_argument("--rotation-degrees", type=float, default=5.0,
                        help="Maximum absolute rotation for light augmentation, in degrees (default: 5).")
    parser.add_argument("--translation-fraction", type=float, default=0.03,
                        help="Maximum light translation as a fraction of each dimension (default: 0.03).")
    parser.add_argument("--translation-pixels", type=int, default=4,
                        help="Maximum absolute shift for pixel-shift profiles, integer 0-8 (default: 4).")
    parser.add_argument("--sampling", choices=SAMPLING_NAMES, default="slice_uniform",
                        help="Training only: original slice shuffle or class/patient-balanced draws.")
    parser.add_argument("--lr-schedule", choices=("constant", "warmup_cosine", "cosine"), default="constant")
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30,
                        help="Experiment maximum, not a course-mandated epoch count (default: 30).")
    parser.add_argument("--inner-only", action="store_true",
                        help="Explore on early-stop patients only; do not score outer validation.")
    parser.add_argument("--calibration-bins", type=int, default=15)
    parser.add_argument("--reject-threshold", type=float, default=0.8,
                        help="Fixed raw-confidence rejection rule; not a fitted or clinical threshold.")
    parser.add_argument("--skip-inference-profile", action="store_true",
                        help="Skip the separate resource benchmark; logs record it as unmeasured.")
    parser.add_argument("--profile-on-cpu", action="store_true",
                        help="Also benchmark CPU; by default the separate benchmark runs on CUDA only.")
    parser.add_argument("--profile-warmup", type=int, default=10)
    parser.add_argument("--profile-repeats", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
