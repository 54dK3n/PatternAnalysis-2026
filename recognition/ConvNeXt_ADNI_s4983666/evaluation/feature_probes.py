"""Fit diagnostic linear heads to frozen, patient-mean stage embeddings.

This diagnostic has a different unit and aggregation from the original model:
each patient's slice GAP embeddings, including repeated scans, are averaged
before fitting a linear head. Labels must be consistent within a patient. Probe
scores describe linear separability of these mean embeddings, not the original
slice/scan classifier, calibrated patient diagnoses, or anatomical information
preservation. No backbone, outer validation, calibration, or final-test data is
used here; callers must supply verified train and early-stop feature records.
"""

from collections import Counter
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from evaluation.metrics import binary_metrics
from utils.artifacts import write_csv, write_json


def _identifier(value: Any, name: str) -> str:
    """Keep patient, scan, and path identities explicit and unambiguous."""
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty string without surrounding whitespace.")
    return value


def _label(value: Any) -> int:
    """Accept only the integer or manifest-text NC=0/AD=1 label encodings."""
    if type(value) is int and value in (0, 1):
        return value
    if type(value) is str and value in ("0", "1"):
        return int(value)
    raise ValueError("Probe labels must use NC=0 and AD=1.")


def _group_rows(rows: list[dict[str, Any]], role: str) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    """Validate slice provenance and group indices under consistent patient labels."""
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{role} probe records must be a nonempty list.")
    required = {"patient_id", "image_id", "slice_index", "relative_path", "label"}
    patients: dict[str, dict[str, Any]] = {}
    paths: set[str] = set()
    scan_slices: set[tuple[str, int]] = set()
    scans: dict[str, tuple[str, int]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"{role} probe record {index + 1} lacks slice identity or label fields.")
        if "partition" in row and row["partition"] != "development":
            raise ValueError("Feature probes permit development train/early_stop records only.")
        if "role" in row and row["role"] != role:
            raise ValueError(f"Expected {role} probe records; outer validation is not permitted.")
        patient_id = _identifier(row["patient_id"], "patient_id")
        image_id = _identifier(row["image_id"], "image_id")
        path = _identifier(row["relative_path"], "relative_path")
        label = _label(row["label"])
        slice_index = row["slice_index"]
        if type(slice_index) is str:
            try:
                converted = int(slice_index)
            except ValueError as exc:
                raise ValueError("slice_index must be a canonical nonnegative integer.") from exc
            if str(converted) != slice_index:
                raise ValueError("slice_index must be a canonical nonnegative integer.")
            slice_index = converted
        if type(slice_index) is not int or slice_index < 0:
            raise ValueError("slice_index must be a canonical nonnegative integer.")
        if path in paths or (image_id, slice_index) in scan_slices:
            raise ValueError(f"Repeated slice identity in {role} probe records.")
        owner = (patient_id, label)
        if image_id in scans and scans[image_id] != owner:
            raise ValueError(f"Inconsistent patient or label within scan {image_id}.")
        paths.add(path)
        scan_slices.add((image_id, slice_index))
        scans[image_id] = owner
        patient = patients.setdefault(patient_id, {
            "patient_id": patient_id, "label": label, "indices": [], "scans": set(),
        })
        if patient["label"] != label:
            raise ValueError(
                f"Mixed labels for patient {patient_id}; a patient-mean feature probe is undefined."
            )
        patient["indices"].append(index)
        patient["scans"].add(image_id)
    grouped = [{
        "patient_id": patient_id, "label": patient["label"], "indices": patient["indices"],
        "num_slices": len(patient["indices"]), "num_scans": len(patient["scans"]),
    } for patient_id, patient in sorted(patients.items())]
    return grouped, set(scans), paths


def _validate_features(features: dict[str, torch.Tensor], n_rows: int, role: str) -> None:
    """Require finite CPU feature matrices aligned to the supplied slice records."""
    if not isinstance(features, dict) or not features:
        raise ValueError(f"{role} features must be a nonempty stage-to-tensor dictionary.")
    for stage, values in features.items():
        _identifier(stage, "Stage name")
        if (not isinstance(values, torch.Tensor) or values.ndim != 2
                or values.shape[0] != n_rows or values.shape[1] < 1
                or values.device.type != "cpu" or not values.is_floating_point()):
            raise ValueError(f"{role}/{stage} must be a floating CPU tensor with shape [n_slices, features].")
        if not torch.isfinite(values).all().item():
            raise ValueError(f"Non-finite frozen features in {role}/{stage}.")


def _patient_features(values: torch.Tensor, patients: list[dict[str, Any]]) -> torch.Tensor:
    """Average detached slice embeddings so each patient contributes one fitting row."""
    detached = values.detach()
    return torch.stack([
        detached.index_select(0, torch.tensor(patient["indices"], dtype=torch.long))
        .to(dtype=torch.float64).mean(dim=0)
        for patient in patients
    ])


def _standardize(train: torch.Tensor, early: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Fit population mean/std to training patients alone; constant dimensions use 1."""
    mean = train.mean(dim=0)
    std = train.std(dim=0, correction=0)
    if not torch.isfinite(mean).all().item() or not torch.isfinite(std).all().item():
        raise ValueError("Non-finite training feature standardization statistics.")
    # Using 1 for an exactly constant training dimension avoids division by zero
    # without adapting its scale to any early-stop values.
    scale = torch.where(std > 0.0, std, torch.ones_like(std))
    train_z = ((train - mean) / scale).to(dtype=torch.float32)
    early_z = ((early - mean) / scale).to(dtype=torch.float32)
    if not torch.isfinite(train_z).all().item() or not torch.isfinite(early_z).all().item():
        raise ValueError("Non-finite standardized features; no probe scores will be published.")
    stats = {
        "fit_role": "train", "fit_unit": "patient_mean_slice_GAP_embedding",
        "std_convention": "population_correction_0", "constant_dimension_scale": 1.0,
        "mean": mean.tolist(), "population_std": std.tolist(), "scale": scale.tolist(),
        "constant_dimensions": int((std == 0.0).sum().item()),
    }
    return train_z, early_z, stats


def _fit_head(train: torch.Tensor, labels: torch.Tensor, pos_weight: float, *,
              epochs: int, lr: float, weight_decay: float, seed: int) -> tuple[nn.Linear, float]:
    """Fit a fresh CPU head for fixed full-batch steps, preserving the caller's RNG."""
    # No CUDA RNG is touched: this diagnostic is entirely CPU based. Resetting
    # the CPU generator for every stage prevents iteration order choosing seeds.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        head = nn.Linear(train.shape[1], 1)
        optimizer = torch.optim.AdamW([
            {"params": [head.weight], "weight_decay": weight_decay},
            {"params": [head.bias], "weight_decay": 0.0},
        ], lr=lr)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(train).squeeze(1), labels)
            if not torch.isfinite(loss).item():
                raise ValueError("Non-finite linear-probe training loss; no probe scores will be published.")
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            final_loss = criterion(head(train).squeeze(1), labels).item()
    if not math.isfinite(final_loss):
        raise ValueError("Non-finite final probe loss; no probe scores will be published.")
    return head, final_loss


def training_pca(train: torch.Tensor, early: torch.Tensor, components: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Fit a deterministic unwhitened PCA basis to training patient rows only.

    Inputs are already standardized using training statistics. Centering, SVD,
    signs and numerical rank never depend on early-stop values or labels. A
    rank-deficient basis is retained with explicit diagnostics rather than
    silently searching for a better dimension on held-out patients.
    """
    if (type(components) is not int or components < 1 or train.ndim != 2
            or early.ndim != 2 or train.shape[1] != early.shape[1]
            or components > min(train.shape[1], train.shape[0] - 1)):
        raise ValueError("PCA dimensions must fit the feature width and training-patient rank limit.")
    train64, early64 = train.detach().double(), early.detach().double()
    center = train64.mean(0)
    centered = train64 - center
    _, singular, vt = torch.linalg.svd(centered, full_matrices=False)
    basis = vt[:components].clone()
    # Resolve arbitrary SVD signs using training-derived components only.
    pivot = basis.abs().argmax(dim=1)
    signs = basis[torch.arange(components), pivot].sign()
    basis *= torch.where(signs == 0, torch.ones_like(signs), signs).unsqueeze(1)
    tolerance = max(centered.shape) * torch.finfo(torch.float64).eps * singular.max()
    rank = int((singular > tolerance).sum().item())
    transformed_train = (centered @ basis.T).float()
    transformed_early = ((early64 - center) @ basis.T).float()
    if not torch.isfinite(transformed_train).all() or not torch.isfinite(transformed_early).all():
        raise ValueError("Non-finite PCA features.")
    metadata = {"fit_role": "train", "fit_unit": "patient_mean_embedding", "components": components,
                "whiten": False, "numerical_rank": rank, "rank_deficient": rank < components,
                "source_feature_dim": train.shape[1], "center": center.tolist(),
                "basis_rows": basis.tolist(), "singular_values": singular.tolist(),
                "sign_rule": "largest_absolute_loading_nonnegative"}
    return transformed_train, transformed_early, metadata


def run_feature_probes(train_features: dict[str, torch.Tensor], train_rows: list[dict[str, Any]],
                       early_features: dict[str, torch.Tensor], early_rows: list[dict[str, Any]],
                       output: Path, *, epochs: int = 200, lr: float = 0.01,
                       weight_decay: float = 0.01, seed: int = 3710,
                       pca_components: int | None = None) -> dict[str, Any]:
    """Write fixed-budget, train-only linear-probe diagnostics for frozen stages.

    Each feature tensor is CPU ``[n_slices, feature_dim]`` and is aligned with
    its identity records. Patient embeddings average all supplied slices across
    all that patient's scans; patients with mixed labels are rejected. Train and
    early-stop patients, scans, and paths must be disjoint. Statistics and class
    weights are fitted only to training patient embeddings. Every stage uses a
    new linear head, the same seed, and exactly ``epochs`` full-batch AdamW steps;
    early-stop data neither update it nor choose its checkpoint or threshold.

    Existing probe artifacts are never overwritten. JSON is written last after
    all stages succeed, and prediction CSVs identify their diagnostic unit.
    """
    if type(epochs) is not int or epochs < 1:
        raise ValueError("Probe epochs must be a positive integer.")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Probe seed must be an integer in [0, 2**63).")
    if pca_components is not None and (type(pca_components) is not int or pca_components < 1):
        raise ValueError("PCA components must be a positive integer or None.")
    if (type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0
            or type(weight_decay) not in (int, float) or not math.isfinite(weight_decay)
            or weight_decay < 0):
        raise ValueError("Probe learning rate must be positive and weight decay nonnegative, both finite.")
    output = Path(output)
    json_path = output / "feature_probes.json"
    csv_path = output / "feature_probe_patient_predictions.csv"
    if output.is_file() or json_path.exists() or csv_path.exists():
        raise ValueError("Probe output artifacts already exist; refusing to overwrite diagnostics.")
    train_patients, train_scans, train_paths = _group_rows(train_rows, "train")
    early_patients, early_scans, early_paths = _group_rows(early_rows, "early_stop")
    train_ids = {row["patient_id"] for row in train_patients}
    early_ids = {row["patient_id"] for row in early_patients}
    if train_ids & early_ids:
        raise ValueError("Training and early-stop probe patient identities overlap.")
    if train_scans & early_scans or train_paths & early_paths:
        raise ValueError("Training and early-stop probe scan or slice identities overlap.")
    _validate_features(train_features, len(train_rows), "train")
    _validate_features(early_features, len(early_rows), "early_stop")
    if set(train_features) != set(early_features):
        raise ValueError("Training and early-stop probe stage names must match.")
    for stage in train_features:
        if train_features[stage].shape[1] != early_features[stage].shape[1]:
            raise ValueError(f"Training and early-stop feature dimensions differ for {stage}.")
        if pca_components is not None and pca_components > min(len(train_patients) - 1,
                                                               train_features[stage].shape[1]):
            raise ValueError("PCA dimensions exceed a training-patient rank limit or stage width.")
    class_counts = Counter(row["label"] for row in train_patients)
    if not class_counts[0] or not class_counts[1]:
        raise ValueError("Training probe patients must contain both NC and AD labels.")
    pos_weight = class_counts[0] / class_counts[1]
    labels = torch.tensor([row["label"] for row in train_patients], dtype=torch.float32)
    result: dict[str, Any] = {
        "status": "complete", "diagnostic": "frozen_stage_patient_mean_linear_separability",
        "evaluation_roles": ["train", "early_stop"],
        "aggregation": "mean_slice_GAP_embedding_across_all_scans_per_patient",
        "aggregation_scope": "all_supplied_selected_complete_scan_slices_not_full_history_by_default",
        "feature_projection": "native" if pca_components is None else "train_only_unwhitened_PCA",
        "metric_unit": "patient_mean_embedding_linear_probe",
        "mixed_patient_labels": "rejected", "backbone_trained": False,
        "interpretation_limits": [
            "These scores are not the original model's slice or scan performance.",
            "These probabilities are uncalibrated diagnostic head outputs, not patient diagnoses.",
            "Linear separability does not prove anatomical relevance or preservation of all spatial information.",
            "Stage feature dimensions differ; score differences alone do not identify a causal bottleneck.",
            "Early-stop patients are development data, not independent outer-validation or final-test estimates.",
        ],
        "fitting": {
            "optimizer": "AdamW", "epochs": epochs, "full_batch_steps": epochs,
            "lr": float(lr), "weight_decay": float(weight_decay), "bias_weight_decay": 0.0,
            "seed_per_stage": seed, "device": "cpu", "threshold": 0.5,
            "selection": "last_fixed_step_no_early_stop_selection", "class_weight_unit": "train_patient",
            "train_patient_class_counts": {str(label): class_counts[label] for label in (0, 1)},
            "train_pos_weight": pos_weight,
        },
        "populations": {
            "train": {"n_patients": len(train_patients), "n_scans": len(train_scans), "n_slices": len(train_rows)},
            "early_stop": {"n_patients": len(early_patients), "n_scans": len(early_scans), "n_slices": len(early_rows)},
        },
        "stages": {},
    }
    predictions: list[dict[str, Any]] = []
    for stage in sorted(train_features):
        train_means = _patient_features(train_features[stage], train_patients)
        early_means = _patient_features(early_features[stage], early_patients)
        train_z, early_z, stats = _standardize(train_means, early_means)
        projection = None
        if pca_components is not None:
            train_z, early_z, projection = training_pca(train_z, early_z, pca_components)
        head, final_loss = _fit_head(train_z, labels, pos_weight, epochs=epochs, lr=lr,
                                     weight_decay=weight_decay, seed=seed)
        stage_result: dict[str, Any] = {
            "feature_dim": train_z.shape[1], "train_standardization": stats,
            "final_train_weighted_patient_bce": final_loss,
        }
        if projection is not None:
            stage_result["pca"] = projection
        with torch.inference_mode():
            for role, features, patients in (("train", train_z, train_patients),
                                              ("early_stop", early_z, early_patients)):
                logits = head(features).squeeze(1)
                if not torch.isfinite(logits).all().item():
                    raise ValueError("Non-finite probe logits; no probe scores will be published.")
                probabilities = torch.sigmoid(logits).tolist()
                stage_result[role] = binary_metrics([row["label"] for row in patients], probabilities)
                predictions.extend({
                    "stage": stage, "role": role, "metric_unit": result["metric_unit"],
                    "patient_id": patient["patient_id"], "label": patient["label"],
                    "num_scans": patient["num_scans"], "num_slices": patient["num_slices"],
                    "probability": probability, "prediction": int(probability >= 0.5),
                } for patient, probability in zip(patients, probabilities))
        result["stages"][stage] = stage_result
    output.mkdir(parents=True, exist_ok=True)
    write_csv(csv_path, predictions)
    write_json(json_path, result)
    return result
