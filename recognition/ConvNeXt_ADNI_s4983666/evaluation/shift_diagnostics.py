"""Inspect integer-shift sampling and boundary loss without changing a model.

Strided sampling can violate shift equivariance; antialiasing is a separate
intervention, not implemented here. Rationale: Zhang, ICML 2019,
https://proceedings.mlr.press/v97/zhang19a.html . Aligned overlap is not a claim
that its features are unaffected by convolution padding or receptive fields.
"""

from collections import defaultdict
import csv
import math
import os
from pathlib import Path
import random
from statistics import mean
from typing import Any

import torch
from torch import nn

from engine.diagnosis import _integer_shift
from models.convnext import ConvNeXtTiny
from evaluation.metrics import aggregate_scans, binary_metrics
from utils.artifacts import write_csv, write_json


def spatial_observers(model: nn.Module) -> dict[str, tuple[nn.Module, int]]:
    """Identify outputs before and after sampling, with effective input strides."""
    if isinstance(model, ConvNeXtTiny):
        stem_stride = model.stem[0].stride
        if stem_stride[0] != stem_stride[1]:
            raise ValueError("Spatial observers require equal axis strides.")
        observers = {"stem": (model.stem, stem_stride[0])}
        for index, stage in enumerate(model.stages):
            stride = stem_stride[0] * 2**index
            if index:
                observers[f"downsample_{index}"] = (model.downsample_layers[index - 1], stride)
            observers[f"stage_{index + 1}"] = (stage, stride)
        return observers
    if hasattr(model, "features") and isinstance(model.features, nn.Sequential) and len(model.features) == 16:
        return {name: pair for index in range(4) for name, pair in (
            (f"conv_{index + 1}", (model.features[4 * index], 2**index)),
            (f"stage_{index + 1}", (model.features[4 * index + 3], 2**(index + 1))),
        )}
    raise ValueError("Unsupported architecture for spatial shift diagnostics.")


def foreground_margins(images: torch.Tensor) -> dict[int, list[int]]:
    """Measure intensity-proxy margins; an empty proxy is explicitly marked -1."""
    height, width = images.shape[-2:]
    gray = images.detach().add(1).mul(127.5).round()
    result = {}
    for threshold in (8, 16, 32):
        margins = []
        for image in gray:
            coordinates = torch.nonzero(image[0] >= threshold, as_tuple=False)
            if not len(coordinates):
                margins.append(-1)
            else:
                low, high = coordinates.min(0).values, coordinates.max(0).values
                margins.append(min(int(low[0]), int(low[1]), height - 1 - int(high[0]),
                                   width - 1 - int(high[1])))
        result[threshold] = margins
    return result


def aligned_feature_error(reference: torch.Tensor, moved: torch.Tensor, dx: int, dy: int,
                          stride: int) -> tuple[list[float] | None, int, int]:
    """Compare the exact common grid only for shifts divisible by effective stride.

    Compare moved[y + dy/stride, x + dx/stride] against reference[y, x]. The
    normalization is the reference overlap L2 norm, clamped to 1e-12. Padding
    can affect the whole overlap, particularly in late stages.
    """
    if reference.shape != moved.shape or reference.ndim != 4:
        raise ValueError("Feature alignment requires identical NCHW shapes.")
    if type(stride) is not int or stride < 1:
        raise ValueError("Effective stride must be positive.")
    if dx % stride or dy % stride:
        return None, 0, 0
    x, y = dx // stride, dy // stride
    height, width = reference.shape[-2:]
    sx0, sx1 = max(0, -x), min(width, width - x)
    sy0, sy1 = max(0, -y), min(height, height - y)
    if sx1 <= sx0 or sy1 <= sy0:
        return None, 0, 0
    before = reference[..., sy0:sy1, sx0:sx1].flatten(1)
    after = moved[..., sy0 + y:sy1 + y, sx0 + x:sx1 + x].flatten(1)
    relative = (after - before).norm(dim=1) / before.norm(dim=1).clamp_min(1e-12)
    return relative.cpu().tolist(), sy1 - sy0, sx1 - sx0


def confidence_stratum(probability: float) -> str:
    """Use prespecified original probability margins, without choosing thresholds."""
    margin = abs(probability - .5)
    return "margin_0_to_0.1" if margin < .1 else "margin_0.1_to_0.3" if margin < .3 else "margin_0.3_to_0.5"


def gap_feature_changes(before: torch.Tensor, after: torch.Tensor) -> dict[str, list]:
    """Measure pooled vectors with double precision and explicit zero-norm handling."""
    first, second = before.mean((2, 3)).double(), after.mean((2, 3)).double()
    norm, other = first.norm(dim=1), second.norm(dim=1)
    absolute = (second - first).norm(dim=1)
    relative = absolute / norm.clamp_min(1e-12)
    denominator = (norm * other).clamp_min(torch.finfo(torch.float64).tiny)
    cosine = (1 - (first * second).sum(1) / denominator).clamp(0, 2).cpu().tolist()
    first_norm, second_norm = norm.cpu().tolist(), other.cpu().tolist()
    return {"gap_absolute_l2": absolute.cpu().tolist(), "gap_relative_l2": relative.cpu().tolist(),
            "gap_cosine_distance": [value if a > 0 and b > 0 else None
                                    for value, a, b in zip(cosine, first_norm, second_norm)],
            "reference_gap_l2": first_norm, "shifted_gap_l2": second_norm,
            "reference_gap_zero": [value == 0 for value in first_norm]}


def _finite(values: torch.Tensor, name: str) -> None:
    """Stop instead of exporting nonfinite feature or prediction measurements."""
    if not torch.isfinite(values).all().item():
        raise ValueError(f"Non-finite {name} in shift diagnosis.")


def _patient_summary(rows: list[dict], value_keys: list[str], bootstrap_samples: int,
                     seed: int) -> dict[str, Any]:
    """Summarize one equal-patient group, resampling patients rather than slices."""
    result: dict[str, Any] = {"patients": len(rows), "slices": sum(r["slices"] for r in rows)}
    result.update({"NC_slices": sum(r["NC_slices"] for r in rows),
                   "AD_slices": sum(r["AD_slices"] for r in rows),
                   "patients_with_NC_slices": sum(r["NC_slices"] > 0 for r in rows),
                   "patients_with_AD_slices": sum(r["AD_slices"] > 0 for r in rows)})
    for key in value_keys:
        values = [float(row[key]) for row in rows if row[key] is not None]
        if not values:
            result[key] = None
            continue
        entry: dict[str, Any] = {"mean": mean(values), "patients": len(values),
                                "valid_slices": sum(row[f"{key}_valid_slices"] for row in rows)}
        if bootstrap_samples:
            rng = random.Random(seed)
            draws = sorted(mean(values[rng.randrange(len(values))] for _ in values)
                           for _ in range(bootstrap_samples))
            entry["conditional_patient_bootstrap_95"] = [draws[int(.025 * (len(draws) - 1))],
                                                        draws[int(.975 * (len(draws) - 1))]]
        result[key] = entry
    return result


def run_shift_sweep(model: nn.Module, loader: Any, device: torch.device, output: Path, *,
                    max_shift: int = 8, bootstrap_samples: int = 1000, seed: int = 3710,
                    expected_slices: int | None = None) -> dict:
    """Stream per-slice features and predictions for exact shifts and round trips.

    Dense maps are streamed per batch instead of retained for the whole cohort. Patient
    summaries average selected slices, hence repeated scans stay in their
    patient cluster. No weights, images or dense feature tensors are saved.
    All hooks and caller module modes are restored, including on exceptions.
    """
    if type(max_shift) is not int or not 1 <= max_shift <= 8:
        raise ValueError("Maximum shift must be an integer from 1 to 8.")
    if type(bootstrap_samples) is not int or bootstrap_samples < 0 or type(seed) is not int:
        raise ValueError("Bootstrap count must be nonnegative and seed an integer.")
    if expected_slices is not None and (type(expected_slices) is not int or expected_slices < 1):
        raise ValueError("Expected slices must be a positive integer or None.")
    output = Path(output)
    if output.exists():
        raise ValueError("Shift output already exists; refusing to overwrite it.")
    observers = spatial_observers(model)
    output.mkdir(parents=True)
    prediction_keys = ["probability_change", "logit_change", "decision_flip", "roundtrip_probability_change",
                       "roundtrip_logit_change", "roundtrip_decision_flip", "roundtrip_pixel_mae",
                       "roundtrip_changed_pixel_fraction"]
    feature_keys = ["gap_absolute_l2", "gap_relative_l2", "gap_cosine_distance", "aligned_relative_l2",
                    "reference_gap_l2", "shifted_gap_l2", "reference_gap_zero"]
    accum: dict[tuple, dict] = {}
    stage_accum: dict[tuple, dict] = {}
    captures: dict[str, torch.Tensor] = {}
    enabled = {"value": True}
    hooks = []
    flags = [(module, module.training) for module in model.modules()]
    shifts = [(dx, dy) for d in range(1, max_shift + 1) for dx, dy in ((-d, 0), (d, 0), (0, -d), (0, d))]
    identity_fields = ["patient_id", "image_id", "slice_index", "relative_path", "label"]
    fields = identity_fields + ["dx", "dy", "original_probability", "shifted_probability", "original_logit",
                               "shifted_logit", "roundtrip_probability", "confidence_stratum",
                               "foreground_margin_t8", "foreground_margin_t16", "foreground_margin_t32",
                               "margin_gt8_t16"] + prediction_keys
    stage_fields = identity_fields + ["dx", "dy", "layer", "effective_stride", "height", "width",
                                      "aligned_support_height", "aligned_support_width"] + feature_keys

    def hook(name: str) -> Any:
        """Create an observer that retains only the current forward outputs."""
        def record(module: nn.Module, inputs: tuple, values: torch.Tensor) -> None:
            """Keep detached floating maps only when the current pass needs them."""
            if enabled["value"]:
                _finite(values, name)
                captures[name] = values.detach().float()
        return record

    def add(table: dict, key: tuple, values: dict, metrics: list[str]) -> None:
        """Accumulate available slice measurements under one patient group."""
        item = table.setdefault(key, {"slices": 0, "sum": defaultdict(float), "count": defaultdict(int),
                                      "NC_slices": 0, "AD_slices": 0})
        item["slices"] += 1
        item["AD_slices" if values["label"] else "NC_slices"] += 1
        for name in metrics:
            value = values[name]
            if value is not None:
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite exported {name}.")
                item["sum"][name] += value
                item["count"][name] += 1

    n_slices = 0
    original_predictions = []
    try:
        for name, (module, _) in observers.items():
            hooks.append(module.register_forward_hook(hook(name)))
        model.eval()
        with (output / "shift_slice_predictions.csv").open("w", newline="") as prediction_file, \
                (output / "stage_slice_changes.csv").open("w", newline="") as stage_file:
            writer = csv.DictWriter(prediction_file, fieldnames=fields)
            layer_writer = csv.DictWriter(stage_file, fieldnames=stage_fields)
            writer.writeheader()
            layer_writer.writeheader()
            with torch.no_grad():
                for batch_index, batch in enumerate(loader):
                    images = batch["image"].to(device)
                    _finite(images, "input")
                    enabled["value"] = True
                    captures.clear()
                    original_logits = model(images)
                    _finite(original_logits, "original logits")
                    original = dict(captures)
                    if set(original) != set(observers):
                        raise ValueError("Not all expected spatial layers were observed.")
                    probabilities = original_logits.sigmoid()
                    margins = foreground_margins(images)
                    identities = [{"patient_id": batch["patient_id"][i], "image_id": batch["image_id"][i],
                                   "slice_index": int(batch["slice_index"][i]), "relative_path": batch["relative_path"][i],
                                   "label": int(batch["label"][i])} for i in range(len(images))]
                    original_predictions.extend({**identity, "probability": probability, "logit": logit}
                        for identity, probability, logit in zip(identities, probabilities.cpu().tolist(), original_logits.cpu().tolist()))
                    n_slices += len(images)
                    for dx, dy in shifts:
                        captures.clear()
                        shifted_images = _integer_shift(images, dx, dy)
                        shifted_logits = model(shifted_images)
                        _finite(shifted_logits, "shifted logits")
                        shifted_features = dict(captures)
                        shifted_probabilities = shifted_logits.sigmoid()
                        enabled["value"] = False
                        returned_images = _integer_shift(shifted_images, -dx, -dy)
                        returned_logits = model(returned_images)
                        _finite(returned_logits, "round-trip logits")
                        returned_probabilities = returned_logits.sigmoid()
                        pixel_difference = (returned_images - images).abs().flatten(1)
                        metrics = {
                            "probability_change": (shifted_probabilities - probabilities).abs().cpu().tolist(),
                            "logit_change": (shifted_logits - original_logits).abs().cpu().tolist(),
                            "decision_flip": ((shifted_probabilities >= .5) != (probabilities >= .5)).float().cpu().tolist(),
                            "roundtrip_probability_change": (returned_probabilities - probabilities).abs().cpu().tolist(),
                            "roundtrip_logit_change": (returned_logits - original_logits).abs().cpu().tolist(),
                            "roundtrip_decision_flip": ((returned_probabilities >= .5) != (probabilities >= .5)).float().cpu().tolist(),
                            "roundtrip_pixel_mae": pixel_difference.mean(1).cpu().tolist(),
                            "roundtrip_changed_pixel_fraction": (pixel_difference != 0).float().mean(1).cpu().tolist(),
                        }
                        base_probs, moved_probs, returned_probs = (v.cpu().tolist() for v in (
                            probabilities, shifted_probabilities, returned_probabilities))
                        base_logits, moved_logits = (v.cpu().tolist() for v in (original_logits, shifted_logits))
                        for i, identity in enumerate(identities):
                            stratum = confidence_stratum(base_probs[i])
                            eligible = margins[16][i] > 8
                            record = {**identity, "dx": dx, "dy": dy, "original_probability": base_probs[i],
                                      "shifted_probability": moved_probs[i], "original_logit": base_logits[i],
                                      "shifted_logit": moved_logits[i], "roundtrip_probability": returned_probs[i],
                                      "confidence_stratum": stratum, "foreground_margin_t8": margins[8][i],
                                      "foreground_margin_t16": margins[16][i], "foreground_margin_t32": margins[32][i],
                                      "margin_gt8_t16": eligible, **{key: metrics[key][i] for key in prediction_keys}}
                            writer.writerow(record)
                            for group in ["all", stratum] + (["margin_gt8_t16"] if eligible else []):
                                add(accum, (dx, dy, group, identity["patient_id"]), record, prediction_keys)
                        for name, before in original.items():
                            after = shifted_features[name]
                            gap = gap_feature_changes(before, after)
                            aligned, support_h, support_w = aligned_feature_error(before, after, dx, dy, observers[name][1])
                            for i, identity in enumerate(identities):
                                record = {**identity, "dx": dx, "dy": dy, "layer": name,
                                          "effective_stride": observers[name][1], "height": before.shape[-2], "width": before.shape[-1],
                                          "aligned_support_height": support_h, "aligned_support_width": support_w,
                                          **{key: values[i] for key, values in gap.items()},
                                          "aligned_relative_l2": aligned[i] if aligned is not None else None}
                                layer_writer.writerow(record)
                                add(stage_accum, (dx, dy, name, identity["patient_id"]), record, feature_keys)
                        enabled["value"] = True
                    print(f"Shift/boundary sweep: batch {batch_index + 1}, {n_slices} slices completed.", flush=True)
    finally:
        for observer in hooks:
            observer.remove()
        for module, training in flags:
            module.training = training
    if not n_slices:
        raise ValueError("No slices were supplied to the shift diagnostic.")

    def records(table: dict, group_field: str, metrics: list[str]) -> list[dict]:
        """Convert accumulators into explicit per-patient mean measurements."""
        return [{"dx": key[0], "dy": key[1], group_field: key[2], "patient_id": key[3],
                 "slices": value["slices"], "NC_slices": value["NC_slices"], "AD_slices": value["AD_slices"],
                 **{f"{name}_valid_slices": value["count"][name] for name in metrics},
                 **{name: value["sum"][name] / value["count"][name]
                                               if value["count"][name] else None for name in metrics}}
                for key, value in sorted(table.items())]

    patient_rows = records(accum, "group", prediction_keys)
    stage_rows = records(stage_accum, "layer", feature_keys)
    write_csv(output / "shift_patient_means.csv", patient_rows)
    write_csv(output / "stage_patient_means.csv", stage_rows)
    write_csv(output / "original_slice_predictions.csv", original_predictions)
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in patient_rows:
        grouped[(row["dx"], row["dy"], row["group"])].append(row)
    stage_grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in stage_rows:
        stage_grouped[(row["dx"], row["dy"], row["layer"])].append(row)
    report = {
        "status": "complete", "slices": n_slices, "max_shift": max_shift,
        "forward_passes_per_slice": 1 + 8 * max_shift,
        "metric_unit": "equal_patient_mean_over_selected_slices",
        "pixel_mae_unit": "normalized_input_range_minus1_to1",
        "bootstrap": {"samples": bootstrap_samples, "seed": seed, "unit": "patient",
                      "scope": "conditional on fixed checkpoint and already-used early-stop cohort"},
        "observers": {name: {"effective_stride": stride} for name, (_, stride) in observers.items()},
        "groups": [{"dx": dx, "dy": dy, "group": group,
                    **_patient_summary(rows, prediction_keys, bootstrap_samples, seed)}
                   for (dx, dy, group), rows in sorted(grouped.items())],
        "layers": [{"dx": dx, "dy": dy, "layer": layer,
                    **_patient_summary(rows, feature_keys, 0, seed)}
                   for (dx, dy, layer), rows in sorted(stage_grouped.items())],
        "limitations": [
            "Common aligned overlap is not necessarily free of padding or receptive-field boundary effects.",
            "Non-stride-divisible shifts have no interpolated feature-alignment score.",
            "Empty foreground proxies are excluded from margin subgroups, not interpreted as a full blank margin.",
            "Confidence strata differ between model checkpoints; they are not paired populations across models.",
            "Round trips isolate input edge loss but do not exclude internal-padding effects.",
            "Near-zero reference norms can yield large relative changes; inspect zero flags and absolute outputs.",
            "Feature means ignore undefined cosine/alignment values; valid slice and patient counts are explicit.",
        ],
    }
    if expected_slices is not None:
        scans = aggregate_scans(original_predictions, expected_slices)
        write_csv(output / "original_scan_predictions.csv", scans)
        report["original_metrics"] = {level: binary_metrics([int(row["label"]) for row in values],
                                                            [row["probability"] for row in values])
                                      for level, values in (("slice", original_predictions), ("scan", scans))}
    write_json(output / "summary.json", report)
    plot_shift_curves(report, output / "shift_boundary_curves.png")
    return report


def plot_shift_curves(report: dict, path: Path) -> None:
    """Plot equal-patient changes by exact displacement, preserving direction."""
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for axis, key, title in zip(axes, ("probability_change", "logit_change", "roundtrip_probability_change"),
                               ("Shifted probability change", "Shifted logit change", "Round-trip probability change")):
        for direction in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            values = [r for r in report["groups"] if r["group"] == "all" and
                      (int(math.copysign(1, r["dx"])) if r["dx"] else 0,
                       int(math.copysign(1, r["dy"])) if r["dy"] else 0) == direction]
            values.sort(key=lambda r: abs(r["dx"]) + abs(r["dy"]))
            axis.plot([abs(r["dx"]) + abs(r["dy"]) for r in values],
                      [r[key]["mean"] for r in values], marker="o", label=str(direction))
        axis.set(xlabel="Integer displacement in input pixels", ylabel="Mean absolute change", title=title)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=.2)
        axis.legend(title="Direction", fontsize=8)
    figure.savefig(path, dpi=160)
    plt.close(figure)
