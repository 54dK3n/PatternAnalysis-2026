"""Run sampling, boundary and probe controls on unchanged inner checkpoints."""

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import torch

from dataset.loaders import make_loader
from dataset.preprocessing import checkpoint_preprocessing
from utils.preprocessing_artifacts import export_preprocessing_audit
from dataset.manifests import load_fold
from engine.diagnosis import _validate_checkpoint, collect_features, select_complete_scans
from evaluation.feature_probes import run_feature_probes
from evaluation.shift_diagnostics import run_shift_sweep
from models import create_model
from utils.artifacts import code_fingerprints, validate_output, write_csv, write_json
from utils.runtime import environment_info, seed_everything, select_device


def verified_reference_rows(path: Path, full_rows: list[dict], expected_slices: int) -> list[dict]:
    """Require exact immutable role membership and complete scans in a prior CSV."""
    with path.open(newline="") as stream:
        selected = list(csv.DictReader(stream))
    full = {row["relative_path"]: row for row in full_rows}
    if not selected or len({r.get("relative_path") for r in selected}) != len(selected):
        raise ValueError("Reference selection is empty or contains duplicate paths.")
    for row in selected:
        if row.get("relative_path") not in full or row != full[row["relative_path"]]:
            raise ValueError("Reference selection differs from the verified inner-role manifest.")
    select_complete_scans(selected, expected_slices, 0, 0, 0)
    return selected


def expanded_training_rows(full_rows: list[dict], anchors: list[dict], expected_slices: int,
                           max_scans: int, seed: int) -> list[dict]:
    """Add patients while preserving every reference patient's exact selected scans.

    A separate patient-derived RNG selects additional patients' scans. Adding a
    patient cannot reshuffle an already anchored patient's history. A zero scan
    cap selects all scans only for additional patients, not the replay anchors.
    """
    if type(max_scans) is not int or max_scans < 0 or type(seed) is not int:
        raise ValueError("Invalid expansion scan cap or seed.")
    select_complete_scans(full_rows, expected_slices, 0, 0, seed)
    full = {row["relative_path"]: row for row in full_rows}
    if any(row != full.get(row["relative_path"]) for row in anchors):
        raise ValueError("Training expansion anchors do not match verified training rows.")
    if len({row["relative_path"] for row in anchors}) != len(anchors):
        raise ValueError("Duplicate expansion anchor slices.")
    select_complete_scans(anchors, expected_slices, 0, 0, seed)
    scans: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    chosen: dict[str, set[str]] = defaultdict(set)
    for row in full_rows:
        scans[row["patient_id"]][row["image_id"]].append(row)
    for row in anchors:
        chosen[row["patient_id"]].add(row["image_id"])
    if max_scans and any(len(ids) > max_scans for ids in chosen.values()):
        raise ValueError("Reference scan count exceeds the requested expansion cap.")
    expanded = []
    for patient, history in sorted(scans.items()):
        if patient in chosen:
            ids = sorted(chosen[patient])
        else:
            ids = sorted(history)
            patient_seed = int.from_bytes(hashlib.sha256(f"{seed}/{patient}".encode()).digest()[:8], "big")
            random.Random(patient_seed).shuffle(ids)
            if max_scans:
                ids = ids[:max_scans]
        for image_id in sorted(ids):
            expanded.extend(sorted(history[image_id], key=lambda row: int(row["slice_index"])))
    return expanded


def filter_mixed(features: dict[str, torch.Tensor], records: list[dict], full_role: list[dict]) -> tuple[dict, list[dict], dict]:
    """Exclude patient probe targets using their complete longitudinal role labels."""
    labels: dict[str, set[int]] = defaultdict(set)
    for row in full_role:
        labels[row["patient_id"]].add(int(row["label"]))
    selected = {row["patient_id"] for row in records}
    if not selected.issubset(labels):
        raise ValueError("Feature records contain patients outside their verified role.")
    excluded = sorted(patient for patient in selected if len(labels[patient]) != 1)
    mask = torch.tensor([row["patient_id"] not in excluded for row in records], dtype=torch.bool)
    kept = [row for row, keep in zip(records, mask.tolist()) if keep]
    return ({stage: values[mask] for stage, values in features.items()}, kept,
            {"patient_ids": excluded, "count": len(excluded), "label_scope": "full_verified_role"})


def subset_features(features: dict[str, torch.Tensor], predictions: list[dict], selected: list[dict]) -> tuple[dict, list[dict]]:
    """Replay an exact selection in its original order using cached frozen features."""
    index = {row["relative_path"]: i for i, row in enumerate(predictions)}
    if len(index) != len(predictions):
        raise ValueError("Duplicate cached feature paths.")
    indices = torch.tensor([index[row["relative_path"]] for row in selected], dtype=torch.long)
    records = [predictions[i] for i in indices.tolist()]
    return {stage: values.index_select(0, indices) for stage, values in features.items()}, records


def _selection(rows: list[dict]) -> dict[str, int]:
    """Report actual complete-scan coverage without assuming eligible counts."""
    return {"patients": len({row["patient_id"] for row in rows}),
            "scans": len({row["image_id"] for row in rows}), "slices": len(rows)}


def run(args: argparse.Namespace) -> dict:
    """Execute the first diagnostic extension, fitting fresh heads only."""
    for name, minimum in (("batch_size", 1), ("workers", 0), ("threads", 1),
                          ("reference_train_patients", 0), ("max_early_patients", 0),
                          ("max_scans_per_patient", 0), ("bootstrap_samples", 0),
                          ("probe_epochs", 1), ("pca_components", 1)):
        value = getattr(args, name)
        if type(value) is not int or value < minimum:
            raise ValueError(f"Invalid {name}.")
    if (type(args.seed) is not int or not 0 <= args.seed < 2**63
            or type(args.max_shift) is not int or not 1 <= args.max_shift <= 8):
        raise ValueError("Invalid diagnostic seed or maximum displacement.")
    if args.routes not in ("all", "spatial", "probes"):
        raise ValueError("Routes must be all, spatial or probes.")
    if (not math.isfinite(args.probe_lr) or args.probe_lr <= 0
            or not math.isfinite(args.probe_weight_decay) or args.probe_weight_decay < 0):
        raise ValueError("Invalid fixed-budget probe settings.")
    output = validate_output(args.output, args.data_root, args.splits_dir)
    checkpoint_path = Path(args.checkpoint).resolve()
    if (output == checkpoint_path.parent or checkpoint_path.parent in output.parents
            or output in checkpoint_path.parents):
        raise ValueError("Follow-up output must be separate from the original checkpoint run.")
    reference_dir = Path(args.reference_dir).resolve() if args.reference_dir is not None else None
    if reference_dir and (output == reference_dir or reference_dir in output.parents or output in reference_dir.parents):
        raise ValueError("Follow-up output must be separate from its reference diagnosis.")
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    config = _validate_checkpoint(checkpoint)
    preprocessing = checkpoint_preprocessing(config)
    data = load_fold(args.data_root, args.splits_dir, config["fold"])
    expected = data["report"]["config"]["expected_slices"]
    if config["manifest_sha256"] != data["manifest_sha256"] or config["expected_slices"] != expected:
        raise ValueError("Checkpoint differs from the verified frozen manifests.")
    reference_hashes = {}
    if reference_dir:
        reference_hashes = {name: hashlib.sha256((reference_dir / name).read_bytes()).hexdigest()
                            for name in ("config.json", "summary.json", "train_selected_slices.csv", "early_stop_selected_slices.csv")}
        previous_config = json.loads((reference_dir / "config.json").read_text())
        previous_summary = json.loads((reference_dir / "summary.json").read_text())
        if (previous_summary.get("status") != "complete" or previous_config.get("checkpoint_sha256") != checkpoint_hash
                or previous_summary.get("checkpoint_sha256") != checkpoint_hash
                or previous_summary.get("manifest_sha256") != data["manifest_sha256"]
                or previous_config.get("manifest_sha256") != data["manifest_sha256"]
                or previous_config.get("checkpoint_epoch") != checkpoint["epoch"]
                or previous_config.get("checkpoint_config") != json.loads(json.dumps(config, allow_nan=False))):
            raise ValueError("Reference diagnosis does not match this completed checkpoint and manifest.")
        reference_train = verified_reference_rows(reference_dir / "train_selected_slices.csv", data["train"], expected)
        early_rows = verified_reference_rows(reference_dir / "early_stop_selected_slices.csv", data["early_stop"], expected)
        if any(hashlib.sha256((reference_dir / name).read_bytes()).hexdigest() != digest
               for name, digest in reference_hashes.items()):
            raise ValueError("Reference diagnosis changed while its selections were read.")
    else:
        reference_train = select_complete_scans(data["train"], expected, args.reference_train_patients,
                                                args.max_scans_per_patient, args.seed)
        early_rows = select_complete_scans(data["early_stop"], expected, args.max_early_patients,
                                          args.max_scans_per_patient, args.seed)
    expanded = expanded_training_rows(data["train"], reference_train, expected, args.max_scans_per_patient, args.seed)
    if {r["patient_id"] for r in expanded} & {r["patient_id"] for r in early_rows}:
        raise ValueError("Training and early-stop patient identities overlap.")
    seed_everything(config["seed"])
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    model = create_model(config["model_name"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Checkpoint contains nonfinite model state.")
    output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": checkpoint["epoch"], "checkpoint_config": config,
        "initialization_metadata_status": ("recorded_random" if "initialization" in config else "legacy_fields_not_recorded"),
        "manifest_sha256": data["manifest_sha256"], "sampling_seed": args.seed,
        "reference_dir": str(reference_dir) if reference_dir else None, "reference_sha256": reference_hashes,
        "code_sha256": code_fingerprints(), "environment": environment_info(device),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    provenance["code_sha256"]["diagnose_followup.py"] = hashlib.sha256(
        (Path(__file__).resolve().parents[1] / "diagnose_followup.py").read_bytes()).hexdigest()
    write_json(output / "config.json", provenance)
    for name, rows in (("reference_train", reference_train), ("expanded_train", expanded), ("early_stop", early_rows)):
        write_csv(output / f"{name}_selected_slices.csv", rows)
    report: dict[str, Any] = {
        "status": "incomplete", "evaluation_role": "inner_diagnostic_followup_ABD", "backbone_trained": False,
        "checkpoint_sha256": checkpoint_hash, "manifest_sha256": data["manifest_sha256"],
        "initialization_metadata_status": provenance["initialization_metadata_status"],
        "selection": {name: _selection(rows) for name, rows in (("reference_train", reference_train),
                      ("expanded_train", expanded), ("early_stop", early_rows))},
        "full_roles": {role: _selection(data[role]) for role in ("train", "early_stop")},
        "limitations": [
            "Early-stop patients already selected the original checkpoint; this is development evidence.",
            "Expanded fitting preserves each replay patient's selected scans; no full-history target is assumed.",
            "Features recomputed in larger batches may differ slightly from historical GPU rounding.",
            "PCA and GAP probes cannot establish anatomical relevance or preservation of all information.",
            "No outer-validation, calibration or final-test scoring; mandatory source verification audits all roles.",
        ],
    }
    loader_args = (args.data_root, tuple(config["image_size"]), args.batch_size, args.workers, args.seed, False, device)
    early_loader = make_loader(early_rows, *loader_args, role="early_stop", preprocessing=preprocessing)
    if preprocessing.name != "none":
        report["preprocessing_config"] = preprocessing.to_dict()
        report["preprocessing_audits"] = {"early_stop": export_preprocessing_audit(
            output, early_loader.dataset, "early_stop")}
    if args.routes in ("all", "spatial"):
        print(f"Inspecting exact shifts and boundaries on {_selection(early_rows)}", flush=True)
        report["spatial"] = run_shift_sweep(model, early_loader, device, output / "spatial",
            max_shift=args.max_shift, bootstrap_samples=args.bootstrap_samples, seed=args.seed,
            expected_slices=expected)
    if args.routes in ("all", "probes"):
        print(f"Collecting frozen training features on {_selection(expanded)}", flush=True)
        train_loader = make_loader(expanded, *loader_args, role="train", preprocessing=preprocessing)
        if preprocessing.name != "none":
            report["preprocessing_audits"]["train"] = export_preprocessing_audit(output, train_loader.dataset, "train")
        train_features, train_predictions, _, _ = collect_features(model, train_loader, device, shift_pixels=0)
        early_features, early_predictions, _, _ = collect_features(model, early_loader, device, shift_pixels=0)
        train_features, train_predictions, expanded_exclusions = filter_mixed(train_features, train_predictions, data["train"])
        early_features, early_predictions, early_exclusions = filter_mixed(early_features, early_predictions, data["early_stop"])
        reference_eligible = [row for row in reference_train if row["patient_id"] not in expanded_exclusions["patient_ids"]]
        replay_features, replay_predictions = subset_features(train_features, train_predictions, reference_eligible)
        report["probe_patient_exclusions"] = {"train": expanded_exclusions, "early_stop": early_exclusions}
        report["probes"] = {}
        for cohort, features, predictions in (("reference", replay_features, replay_predictions),
                                              ("expanded", train_features, train_predictions)):
            report["probes"][cohort] = {}
            for projection, components in (("native", None), (f"pca_{args.pca_components}", args.pca_components)):
                print(f"Fitting fixed probe: {cohort}/{projection}", flush=True)
                report["probes"][cohort][projection] = run_feature_probes(features, predictions, early_features,
                    early_predictions, output / "probes" / cohort / projection, epochs=args.probe_epochs,
                    lr=args.probe_lr, weight_decay=args.probe_weight_decay, seed=args.seed, pca_components=components)
    if any(not torch.equal(value.cpu(), checkpoint["model_state"][name]) for name, value in model.state_dict().items()):
        raise ValueError("Follow-up unexpectedly changed backbone state.")
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != checkpoint_hash:
        raise ValueError("Original checkpoint bytes changed during diagnosis.")
    if reference_dir and any(hashlib.sha256((reference_dir / name).read_bytes()).hexdigest() != digest
                             for name, digest in reference_hashes.items()):
        raise ValueError("Reference diagnosis changed during follow-up.")
    report["status"] = "complete"
    if provenance["initialization_metadata_status"] == "legacy_fields_not_recorded":
        report["limitations"].append("Legacy CNN initialization fields were not recorded; compatibility does not infer initialization from tensor values.")
    write_json(output / "summary.json", report)
    print(f"Completed inner follow-up: {output / 'summary.json'}", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    """Expose A/B/D diagnostics with no public outer or reserved scoring roles."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "data-root", "splits-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, help="Completed prior diagnosis for exact cohort replay.")
    parser.add_argument("--routes", choices=("all", "spatial", "probes"), default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--reference-train-patients", type=int, default=64, help="Fallback without reference-dir; zero keeps all.")
    parser.add_argument("--max-early-patients", type=int, default=0, help="Fallback without reference-dir; zero keeps all.")
    parser.add_argument("--max-scans-per-patient", type=int, default=2, help="Additional patients only; anchor scans are retained.")
    parser.add_argument("--max-shift", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=1000, help="Patient resamples; zero omits conditional intervals.")
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--probe-lr", type=float, default=.01)
    parser.add_argument("--probe-weight-decay", type=float, default=.01)
    try:
        run(parser.parse_args(argv))
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0
