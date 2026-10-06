"""Export confidence plots and factual failure examples without clinical claims."""

import os
from pathlib import Path

from dataset.slices import ADNISliceDataset
from utils.artifacts import write_csv, write_json


def select_failure_cases(slices: list[dict], limit: int = 5) -> list[dict]:
    """Prioritize confident FP/FN, preferring distinct patients before repeats."""
    errors = []
    for row in slices:
        predicted = int(row["probability"] >= 0.5)
        if predicted != row["label"]:
            errors.append({**row, "prediction": predicted,
                           "confidence": max(row["probability"], 1 - row["probability"]),
                           "error_type": "FP" if predicted else "FN"})
    ranked = sorted(errors, key=lambda r: (-r["confidence"], r["relative_path"]))
    selected, paths, patients = [], set(), set()
    # Include both error types if observed; then fill with independent patients.
    for kind in ("FP", "FN"):
        candidate = next((r for r in ranked if r["error_type"] == kind), None)
        if candidate:
            selected.append(candidate)
            paths.add(candidate["relative_path"])
            patients.add(candidate["patient_id"])
    for distinct in (True, False):
        for row in ranked:
            if len(selected) >= limit:
                break
            if row["relative_path"] in paths or (distinct and row["patient_id"] in patients):
                continue
            selected.append(row)
            paths.add(row["relative_path"])
            patients.add(row["patient_id"])
    return selected[:limit]


def export_evaluation_artifacts(output: Path, report: dict, slices: list[dict],
                                patients: list[dict], manifest_rows: list[dict],
                                data_root: Path, image_size: tuple[int, int], prefix: str, *,
                                preprocessing=None, scan_parameters=None) -> dict:
    """Save calibration tables, paired curves and at most five verified input images."""
    output = Path(output)
    if patients:
        write_csv(output / f"{prefix}_patient_predictions.csv", patients)
    for unit, values in report["confidence"].items():
        write_csv(output / f"{prefix}_{unit}_reliability.csv", values["reliability_bins"])
        write_csv(output / f"{prefix}_{unit}_risk_coverage.csv", values["risk_coverage"])
    cases = select_failure_cases(slices)
    metadata = {"evaluation_unit": "slice", "examples_selected": len(cases),
                "selection": "confident_FP_FN_then_distinct_patients",
                "clinical_interpretation": "owner_required", "cases": cases}
    write_json(output / f"{prefix}_failure_cases.json", metadata)
    if cases:
        write_csv(output / f"{prefix}_failure_cases.csv", cases)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    values = report["confidence"]["slice"]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    nonempty = [r for r in values["reliability_bins"] if r["count"]]
    axes[0].plot([r["mean_confidence"] for r in nonempty], [r["accuracy"] for r in nonempty], "o-")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="Raw predicted-class confidence", ylabel="Accuracy", title="Slice reliability", xlim=(0, 1), ylim=(0, 1))
    for kind in ("correct", "incorrect"):
        histogram = values["confidence_histogram"]
        axes[1].stairs([r[kind] for r in histogram],
                       [i / values["bins"] for i in range(values["bins"] + 1)], label=kind)
    axes[1].set(xlabel="Raw confidence", ylabel="Slice count", title="Confidence and correctness")
    axes[1].legend()
    curve = [r for r in values["risk_coverage"] if r["risk"] is not None]
    axes[2].step([r["coverage"] for r in curve], [r["risk"] for r in curve], where="post")
    axes[2].set(xlabel="Coverage", ylabel="Accepted error rate", title="Descriptive risk--coverage", xlim=(0, 1), ylim=(0, 1))
    figure.savefig(output / f"{prefix}_confidence.png", dpi=160)
    plt.close(figure)
    if cases:
        indexed = {row["relative_path"]: row for row in manifest_rows}
        # A few exported failures must retain the evaluated full-scan statistics.
        dataset = ADNISliceDataset([indexed[r["relative_path"]] for r in cases], data_root, image_size,
                                   preprocessing=preprocessing, scan_parameters=scan_parameters)
        figure, axes = plt.subplots(1, len(cases), figsize=(3.2 * len(cases), 4), squeeze=False, layout="constrained")
        for index, case in enumerate(cases):
            axis = axes[0, index]
            axis.imshow(dataset[index]["image"][0], cmap="gray", vmin=-1, vmax=1)
            axis.set_title(f"{case['error_type']} | p(AD)={case['probability']:.3f}\n"
                           f"{case['patient_id']}\nscan {case['image_id']}, slice {case['slice_index']}", fontsize=8)
            axis.axis("off")
        figure.savefig(output / f"{prefix}_failures.png", dpi=160)
        plt.close(figure)
    return {"examples_selected": len(cases), "clinical_interpretation": "owner_required"}
