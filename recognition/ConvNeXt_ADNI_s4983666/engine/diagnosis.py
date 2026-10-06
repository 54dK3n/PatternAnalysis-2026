"""Inspect inputs and frozen features using training and inner early-stop patients.

This diagnostic does not fit the backbone, select an outer-fold model, calibrate
probabilities, or score final-test patients. Linear probes are optional and fit
only training-patient embeddings with a fixed, declared optimization budget.
"""

import argparse
from collections import defaultdict
import hashlib
import io
import math
from pathlib import Path
import random
import sys
from typing import Any

import torch

from models.registry import validate_feature_architecture
from torch import nn

from dataset.augmentation import AugmentationConfig, make_augmentation
from dataset.loaders import make_loader
from dataset.preprocessing import checkpoint_preprocessing
from utils.preprocessing_artifacts import export_preprocessing_audit
from dataset.manifests import load_fold
from evaluation.metrics import aggregate_scans, binary_metrics
from models import MODEL_NAMES, create_model, model_minimum_size
from models.convnext import ConvNeXtBlock, ConvNeXtTiny
from utils.artifacts import code_fingerprints, validate_output, write_csv, write_json
from utils.runtime import environment_info, seed_everything, select_device


def select_complete_scans(rows: list[dict], expected_slices: int, max_patients: int,
                          max_scans_per_patient: int, seed: int) -> list[dict]:
    """Sample patients by class, then scans without consulting any predictions.

    A zero limit keeps all patients or scans. Every selected scan is complete;
    the same seed and manifests yield the same identities across architectures.
    """
    if any(type(n) is not int or n < 0 for n in (max_patients, max_scans_per_patient)):
        raise ValueError("Patient and scan limits must be nonnegative integers.")
    if type(expected_slices) is not int or expected_slices < 1 or type(seed) is not int:
        raise ValueError("Expected slice count and seed must be integers.")
    scans: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("partition") != "development" or str(row["label"]) not in ("0", "1"):
            raise ValueError("Diagnostics require development rows with binary labels.")
        scans[row["image_id"]].append(row)
    patients: dict[str, list[str]] = defaultdict(list)
    labels: dict[str, set[int]] = defaultdict(set)
    for image_id, items in scans.items():
        owners = {(r["patient_id"], int(r["label"])) for r in items}
        indices = {int(r["slice_index"]) for r in items}
        if len(owners) != 1 or len(items) != expected_slices or len(indices) != expected_slices:
            raise ValueError("Diagnostics require complete scans with consistent identities.")
        patient, label = next(iter(owners))
        labels[patient].add(label)
        patients[patient].append(image_id)
    if not patients:
        raise ValueError("Cannot diagnose an empty role.")
    rng = random.Random(seed)
    strata = ((0,), (1,), (0, 1))
    by_class = {stratum: sorted(p for p in patients if tuple(sorted(labels[p])) == stratum)
                for stratum in strata}
    for candidates in by_class.values():
        rng.shuffle(candidates)
    chosen: list[str] = []
    limit = len(patients) if max_patients == 0 else min(max_patients, len(patients))
    while len(chosen) < limit:
        for stratum in strata:
            if by_class[stratum] and len(chosen) < limit:
                chosen.append(by_class[stratum].pop())
    selected = []
    for patient in sorted(chosen):
        candidates = sorted(patients[patient])
        rng.shuffle(candidates)
        if max_scans_per_patient:
            candidates = candidates[:max_scans_per_patient]
        for image_id in sorted(candidates):
            selected.extend(sorted(scans[image_id], key=lambda r: int(r["slice_index"])))
    return selected


def _integer_shift(images: torch.Tensor, horizontal: int, vertical: int) -> torch.Tensor:
    """Move pixels exactly once on a fixed canvas, using normalized black fill.

    This intentionally probes shift sensitivity without interpolation. The
    shifted edge can be clipped; changes are not automatically anatomical loss.
    """
    shifted = images.new_full(images.shape, -1.0)
    height, width = images.shape[-2:]
    if abs(horizontal) >= width or abs(vertical) >= height:
        return shifted
    sx0, sx1 = max(0, -horizontal), min(width, width - horizontal)
    sy0, sy1 = max(0, -vertical), min(height, height - vertical)
    shifted[..., sy0 + vertical:sy1 + vertical, sx0 + horizontal:sx1 + horizontal] = (
        images[..., sy0:sy1, sx0:sx1])
    return shifted


def collect_features(model: nn.Module, loader: Any, device: torch.device,
                     shift_pixels: int = 2) -> tuple[dict, list[dict], dict, dict]:
    """Collect GAP embeddings, actual residual ratios, and integer-shift responses.

    Hooks inspect the post-LayerScale residual directly, avoiding cancellation
    from subtracting nearly identical block output/input tensors. Model weights
    remain frozen and all hooks and module training flags are restored on error.
    """
    if type(shift_pixels) is not int or not 0 <= shift_pixels <= 16:
        raise ValueError("Shift pixels must be an integer from 0 to 16.")
    if isinstance(model, ConvNeXtTiny):
        stages = {f"stage_{i + 1}": stage for i, stage in enumerate(model.stages)}
        stages["head_input"] = model.final_norm
    elif hasattr(model, "features") and isinstance(model.features, nn.Sequential):
        stages = {f"stage_{i // 4 + 1}": model.features[i] for i in (3, 7, 11, 15)}
    else:
        raise ValueError("Unsupported model for feature diagnostics.")
    chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    stats: dict[str, dict] = {}
    residual_stats: dict[str, dict] = {}
    current: dict[str, torch.Tensor] = {}
    hooks = []
    flags = [(module, module.training) for module in model.modules()]
    active = {"value": True}
    predictions = []
    shifts: dict[str, dict] = {}

    def activation_hook(name: str) -> Any:
        """Create a finite-checking stage hook active only on original inputs."""
        def record(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
            """Accumulate scalar activation moments and one embedding per slice."""
            if not active["value"]:
                return
            values = output.detach().float()
            if not torch.isfinite(values).all().item():
                raise ValueError(f"Non-finite activations in {name}.")
            pooled = values.mean(dim=(2, 3)) if values.ndim == 4 else values
            chunks[name].append(pooled.cpu())
            item = stats.setdefault(name, {"elements": 0, "sum": 0.0, "squared_sum": 0.0,
                                           "shape_per_slice": list(values.shape[1:])})
            item["elements"] += values.numel()
            item["sum"] += values.sum(dtype=torch.float64).item()
            item["squared_sum"] += values.square().sum(dtype=torch.float64).item()
        return record

    def skip_hook(name: str) -> Any:
        """Create a pre-hook storing each slice's skip-branch norm."""
        def record(module: nn.Module, inputs: tuple) -> None:
            """Capture the input norm without retaining large feature tensors."""
            if active["value"]:
                current[name] = inputs[0].detach().float().flatten(1).norm(dim=1)
        return record

    def residual_hook(name: str) -> Any:
        """Create a hook on the actual scaled residual, with eval DropPath disabled."""
        def record(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
            """Accumulate per-slice residual/skip ratios rather than output differences."""
            if not active["value"]:
                return
            skip = current[name]
            ratio = output.detach().float().flatten(1).norm(dim=1) / skip.clamp_min(1e-12)
            if not torch.isfinite(ratio).all().item():
                raise ValueError(f"Non-finite residual ratio in {name}.")
            item = residual_stats.setdefault(name, {"slices": 0, "ratio_sum": 0.0,
                                                     "ratio_max": 0.0, "zero_skip_slices": 0})
            item["slices"] += ratio.numel()
            item["ratio_sum"] += ratio.sum().item()
            item["ratio_max"] = max(item["ratio_max"], ratio.max().item())
            item["zero_skip_slices"] += int((skip == 0).sum().item())
        return record

    try:
        for name, stage in stages.items():
            hooks.append(stage.register_forward_hook(activation_hook(name)))
        for name, block in model.named_modules():
            if isinstance(block, ConvNeXtBlock):
                hooks.append(block.register_forward_pre_hook(skip_hook(name)))
                hooks.append(block.drop_path.register_forward_hook(residual_hook(name)))
        model.eval()
        with torch.no_grad():
            for batch in loader:
                active["value"] = True
                images = batch["image"].to(device)
                if not torch.isfinite(images).all().item():
                    raise ValueError("Non-finite diagnostic input.")
                logits = model(images)
                if not torch.isfinite(logits).all().item():
                    raise ValueError("Non-finite diagnostic logits.")
                probabilities = logits.sigmoid()
                for i, probability in enumerate(probabilities.cpu().tolist()):
                    predictions.append({
                        "patient_id": batch["patient_id"][i], "image_id": batch["image_id"][i],
                        "slice_index": int(batch["slice_index"][i]),
                        "relative_path": batch["relative_path"][i], "label": int(batch["label"][i]),
                        "probability": probability,
                    })
                active["value"] = False
                if shift_pixels:
                    for dx, dy in ((-shift_pixels, 0), (shift_pixels, 0),
                                   (0, -shift_pixels), (0, shift_pixels)):
                        moved_logits = model(_integer_shift(images, dx, dy))
                        if not torch.isfinite(moved_logits).all().item():
                            raise ValueError("Non-finite shifted-input logits.")
                        moved = moved_logits.sigmoid()
                        delta = (moved - probabilities).abs()
                        key = f"dx_{dx}_dy_{dy}"
                        item = shifts.setdefault(key, {"slices": 0, "absolute_delta_sum": 0.0,
                                                       "maximum_absolute_delta": 0.0,
                                                       "changed_slice_decisions": 0})
                        item["slices"] += delta.numel()
                        item["absolute_delta_sum"] += delta.sum().item()
                        item["maximum_absolute_delta"] = max(item["maximum_absolute_delta"], delta.max().item())
                        item["changed_slice_decisions"] += int(((moved >= .5) != (probabilities >= .5)).sum().item())
        if not predictions:
            raise ValueError("No diagnostic slices were loaded.")
    finally:
        for hook in hooks:
            hook.remove()
        for module, training in flags:
            module.training = training
    for item in stats.values():
        mean = item.pop("sum") / item["elements"]
        second = item.pop("squared_sum") / item["elements"]
        item.update(mean=mean, standard_deviation=math.sqrt(max(0.0, second - mean * mean)))
    for name, item in residual_stats.items():
        item["mean_residual_to_skip_norm"] = item.pop("ratio_sum") / item["slices"]
        gamma = dict(model.named_modules())[name].layer_scale.detach().float().cpu()
        item["layer_scale"] = {"minimum": gamma.min().item(), "maximum": gamma.max().item(),
                                "mean_absolute": gamma.abs().mean().item(),
                                "median_absolute": gamma.abs().median().item()}
    for item in shifts.values():
        item["mean_absolute_probability_change"] = item.pop("absolute_delta_sum") / item["slices"]
        item["fraction_changed_slice_decisions"] = item["changed_slice_decisions"] / item["slices"]
    return ({name: torch.cat(parts) for name, parts in chunks.items()}, predictions,
            {"activations": stats, "residual_branches": residual_stats}, shifts)


def _validate_checkpoint(checkpoint: dict) -> dict:
    """Require supported preprocessing and strict typed checkpoint identities."""
    config = checkpoint["config"]
    if type(config["checkpoint_format_version"]) is not int or config["checkpoint_format_version"] not in (1, 2, 3):
        raise ValueError("Unsupported checkpoint format.")
    if config["model_name"] not in MODEL_NAMES:
        raise ValueError("Unsupported checkpoint architecture.")
    legacy_cnn = (config["checkpoint_format_version"] == 1
                  and config["model_name"] == "small_cnn_v1"
                  and "initialization" not in config and "pretrained_weights" not in config)
    recorded_random = (config.get("initialization") == "random"
                       and "pretrained_weights" in config and config["pretrained_weights"] is None)
    if not (legacy_cnn or recorded_random):
        raise ValueError("Unsupported checkpoint initialization metadata: expected explicit random "
                         "initialization or the legacy format-one CNN with both fields absent.")
    for key in ("fold", "seed", "expected_slices"):
        if type(config[key]) is not int or (key != "seed" and config[key] < 1):
            raise ValueError(f"Checkpoint {key} must be an integer.")
    if config["aggregation"] != "mean_slice_AD_probability" or config["threshold"] != .5:
        raise ValueError("Unsupported aggregation or threshold.")
    checkpoint_preprocessing(config)
    if config["calibration"] != "not_fitted":
        raise ValueError("Unsupported checkpoint calibration.")
    if config["checkpoint_format_version"] == 1:
        if config["augmentation"] != "none":
            raise ValueError("Legacy checkpoints must use unaugmented training.")
    else:
        augmentation = AugmentationConfig.from_dict(config["augmentation_config"])
        if augmentation.name != config["augmentation"]:
            raise ValueError("Checkpoint augmentation configuration disagrees.")
    validate_feature_architecture(config)
    size = config["image_size"]
    minimum = model_minimum_size(config["model_name"])
    if (not isinstance(size, (tuple, list)) or len(size) != 2
            or any(type(n) is not int or n < minimum for n in size)):
        raise ValueError("Invalid checkpoint image dimensions.")
    if type(checkpoint["epoch"]) is not int or checkpoint["epoch"] < 1:
        raise ValueError("Invalid checkpoint epoch.")
    return config


def run(args: argparse.Namespace) -> dict:
    """Write a reviewable diagnostic report without fitting or saving a backbone."""
    if (any(type(n) is not int for n in (args.batch_size, args.workers, args.threads))
            or args.batch_size < 1 or args.workers < 0 or args.threads < 1):
        raise ValueError("Batch size/threads must be positive and workers nonnegative.")
    for key in ("max_train_patients", "max_early_patients", "max_scans_per_patient", "max_transform_images"):
        value = getattr(args, key)
        if type(value) is not int or value < 0 or (key == "max_transform_images" and value == 0):
            raise ValueError(f"Invalid {key}.")
    if (type(args.seed) is not int or not 0 <= args.seed < 2**63
            or type(args.shift_pixels) is not int or not 0 <= args.shift_pixels <= 16):
        raise ValueError("Invalid diagnostic seed or shift size.")
    if (type(args.probe_epochs) is not int or args.probe_epochs < 1
            or not math.isfinite(args.probe_lr) or args.probe_lr <= 0
            or not math.isfinite(args.probe_weight_decay) or args.probe_weight_decay < 0):
        raise ValueError("Invalid linear-probe settings.")
    output = validate_output(args.output, args.data_root, args.splits_dir)
    checkpoint_path = Path(args.checkpoint).resolve()
    if output == checkpoint_path.parent or checkpoint_path.parent in output.parents or output in checkpoint_path.parents:
        raise ValueError("Diagnostic output must be separate from the original checkpoint run.")
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    config = _validate_checkpoint(checkpoint)
    preprocessing = checkpoint_preprocessing(config)
    data = load_fold(args.data_root, args.splits_dir, config["fold"])
    if config["manifest_sha256"] != data["manifest_sha256"]:
        raise ValueError("Checkpoint belongs to different frozen manifests.")
    if config["expected_slices"] != data["report"]["config"]["expected_slices"]:
        raise ValueError("Checkpoint slice count disagrees with frozen manifests.")
    roles = {}
    for role, limit in (("train", args.max_train_patients), ("early_stop", args.max_early_patients)):
        roles[role] = select_complete_scans(data[role], config["expected_slices"], limit,
                                           args.max_scans_per_patient, args.seed)
    if {r["patient_id"] for r in roles["train"]} & {r["patient_id"] for r in roles["early_stop"]}:
        raise ValueError("Training and early-stop patients overlap.")
    seed_everything(config["seed"])
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    model = create_model(config["model_name"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    if any(not torch.isfinite(value).all().item() for value in model.state_dict().values()):
        raise ValueError("Checkpoint contains non-finite model state.")
    output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "checkpoint_epoch": checkpoint["epoch"], "checkpoint_config": config,
        "initialization_metadata_status": ("recorded_random" if "initialization" in config
                                           else "legacy_fields_not_recorded"),
        "manifest_sha256": data["manifest_sha256"], "sampling_seed": args.seed,
        "code_sha256": code_fingerprints(), "environment": environment_info(device),
    }
    provenance["code_sha256"]["diagnose.py"] = hashlib.sha256(
        (Path(__file__).resolve().parents[1] / "diagnose.py").read_bytes()).hexdigest()
    write_json(output / "config.json", provenance)
    from evaluation.transform_audit import run_transform_audit
    report = {"status": "incomplete", "evaluation_role": "inner_feature_diagnosis",
              "model_name": config["model_name"], "roles": {},
              "limitations": [
                  "Patient/scan subsets are diagnostics, not formal model-comparison metrics.",
                  "Global activation spread and residual ratios do not prove anatomical validity.",
                  "Integer shifts may clip edges and are not the bilinear training augmentation.",
                  "No outer validation, calibration or final-test scoring is performed.",
                  "Mandatory source-integrity verification still audits all source partitions.",
              ]}
    if preprocessing.name == "none":
        report["transform_audit"] = run_transform_audit(
            data["train"], args.data_root, tuple(config["image_size"]), output / "transforms",
            max_images=args.max_transform_images, seed=args.seed,
            augmentation=(AugmentationConfig.from_dict(config["augmentation_config"])
                          if config["augmentation"] == "light" else make_augmentation("light")))
    else:
        report["transform_audit"] = {"status": "not_run_legacy_light_audit",
                                     "reason": "Scan preprocessing QA is exported for each inspected role."}
        report["preprocessing_config"] = preprocessing.to_dict()
    feature_data = {}
    for role, rows in roles.items():
        print(f"Inspecting {role}: {len({r['patient_id'] for r in rows})} patients, "
              f"{len({r['image_id'] for r in rows})} complete scans.", flush=True)
        loader = make_loader(rows, args.data_root, tuple(config["image_size"]), args.batch_size,
                             args.workers, args.seed, False, device, role=role, preprocessing=preprocessing)
        if preprocessing.name != "none":
            report.setdefault("preprocessing_audits", {})[role] = export_preprocessing_audit(
                output, loader.dataset, role, max_images=args.max_transform_images)
        features, predictions, layers, shifts = collect_features(model, loader, device, args.shift_pixels)
        scans = aggregate_scans(predictions, config["expected_slices"])
        feature_data[role] = (features, predictions)
        write_csv(output / f"{role}_selected_slices.csv", rows)
        write_csv(output / f"{role}_slice_predictions.csv", predictions)
        write_csv(output / f"{role}_scan_predictions.csv", scans)
        report["roles"][role] = {
            "selection": {"patients": len({r["patient_id"] for r in rows}),
                          "scans": len(scans), "slices": len(rows),
                          "full_role_patients": len({r["patient_id"] for r in data[role]}),
                          "full_role_scans": len({r["image_id"] for r in data[role]})},
            "metrics": {level: binary_metrics([int(r["label"]) for r in items],
                                               [r["probability"] for r in items])
                        for level, items in (("scan", scans), ("slice", predictions))},
            "layers": layers, "shift_sensitivity": shifts,
        }
    # An exact state comparison also guards future hook changes against mutation.
    if any(not torch.equal(value.cpu(), checkpoint["model_state"][name])
           for name, value in model.state_dict().items()):
        raise ValueError("Diagnostic inference unexpectedly changed model state.")
    if args.probes:
        from evaluation.feature_probes import run_feature_probes
        probe_data = {}
        report["probe_patient_exclusions"] = {}
        for role, (features, predictions) in feature_data.items():
            # A patient can legitimately change labels across longitudinal scans.
            # Scan diagnostics retain those patients; a binary patient-mean probe
            # requires one consistent label across the patient's full role.
            patient_labels: dict[str, set[int]] = defaultdict(set)
            for row in data[role]:
                patient_labels[row["patient_id"]].add(int(row["label"]))
            excluded = sorted(p for p in {r["patient_id"] for r in predictions}
                              if len(patient_labels[p]) != 1)
            mask = torch.tensor([r["patient_id"] not in excluded for r in predictions], dtype=torch.bool)
            kept = [r for r, keep in zip(predictions, mask.tolist()) if keep]
            probe_data[role] = ({name: values[mask] for name, values in features.items()}, kept)
            report["probe_patient_exclusions"][role] = {
                "patient_ids": excluded, "count": len(excluded),
                "reason": "Mixed longitudinal labels cannot define a binary patient-mean target."}
        if (not probe_data["early_stop"][1]
                or len({r["label"] for r in probe_data["train"][1]}) != 2):
            report["feature_probes"] = {"status": "not_run",
                "reason": "Eligible probes need both training classes and nonempty early-stop patients."}
        else:
            report["feature_probes"] = run_feature_probes(
                *probe_data["train"], *probe_data["early_stop"], output / "probes",
                epochs=args.probe_epochs, lr=args.probe_lr, weight_decay=args.probe_weight_decay,
                seed=args.seed)
    report["status"] = "complete"
    report["initialization_metadata_status"] = provenance["initialization_metadata_status"]
    if provenance["initialization_metadata_status"] == "legacy_fields_not_recorded":
        report["limitations"].append(
            "Legacy CNN initialization fields were not recorded. Compatibility does not infer "
            "initialization from the weight tensors or alter the original checkpoint metadata.")
    report["checkpoint_sha256"] = provenance["checkpoint_sha256"]
    report["manifest_sha256"] = data["manifest_sha256"]
    write_json(output / "summary.json", report)
    print(f"Completed inner diagnostics: {output / 'summary.json'}", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    """Expose a diagnostic CLI whose model scoring roles are fixed to inner data."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "data-root", "splits-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=3710, help="Patient selection and probe seed.")
    parser.add_argument("--max-train-patients", type=int, default=64, help="Zero keeps all training patients.")
    parser.add_argument("--max-early-patients", type=int, default=0, help="Zero keeps all early-stop patients.")
    parser.add_argument("--max-scans-per-patient", type=int, default=2, help="Zero keeps all scans.")
    parser.add_argument("--max-transform-images", type=int, default=12)
    parser.add_argument("--shift-pixels", type=int, default=2, help="Zero disables integer-shift comparisons.")
    parser.add_argument("--probes", action="store_true", help="Fit fixed-budget linear heads on frozen patient embeddings.")
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--probe-lr", type=float, default=.01)
    parser.add_argument("--probe-weight-decay", type=float, default=.01)
    try:
        run(parser.parse_args(argv))
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0
