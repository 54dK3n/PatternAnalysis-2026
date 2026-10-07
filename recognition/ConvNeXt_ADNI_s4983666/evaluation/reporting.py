"""Auditable confidence, rejection and patient-level summaries of fixed predictions.

ECE uses predicted-class confidence in equal-width bins on [0, 1]. Brier score
uses the AD probability. These describe raw scores, not fitted calibration.
No operating threshold is selected using these evaluation labels.
"""

import math

from evaluation.metrics import _probability, binary_metrics


def confidence_metrics(labels: list[int], probabilities: list[float], bins: int = 15,
                       reject_threshold: float = 0.8) -> dict:
    """Group confidence ties together in an empirical risk--coverage curve."""
    classification = binary_metrics(labels, probabilities)
    if type(bins) is not int or bins < 1:
        raise ValueError("Calibration bins must be a positive integer.")
    reject_threshold = _probability(reject_threshold, "Reject threshold")
    if reject_threshold < 0.5:
        raise ValueError("Reject threshold must be between 0.5 and 1.")
    probabilities = [float(p) for p in probabilities]
    observations = [(max(p, 1 - p), int(int(p >= 0.5) == y))
                    for y, p in zip(labels, probabilities)]
    buckets = [[] for _ in range(bins)]
    for confidence, correct in observations:
        buckets[min(int(confidence * bins), bins - 1)].append((confidence, correct))
    reliability = []
    histogram = []
    ece = 0.0
    for index, bucket in enumerate(buckets):
        mean = math.fsum(p for p, _ in bucket) / len(bucket) if bucket else None
        accuracy = sum(c for _, c in bucket) / len(bucket) if bucket else None
        if bucket:
            ece += len(bucket) / len(labels) * abs(mean - accuracy)
        reliability.append({"bin": index, "lower": index / bins,
                            "upper": (index + 1) / bins, "count": len(bucket),
                            "mean_confidence": mean, "accuracy": accuracy})
        histogram.append({"bin": index, "correct": sum(c for _, c in bucket),
                          "incorrect": sum(1 - c for _, c in bucket)})
    groups = {}
    for confidence, correct in observations:
        groups.setdefault(confidence, []).append(correct)
    curve = [{"confidence_threshold": None, "accepted": 0, "coverage": 0.0,
              "accepted_accuracy": None, "risk": None}]
    accepted = correct = 0
    for confidence, group in sorted(groups.items(), reverse=True):
        accepted += len(group)
        correct += sum(group)
        curve.append({"confidence_threshold": confidence, "accepted": accepted,
                      "coverage": accepted / len(labels),
                      "accepted_accuracy": correct / accepted,
                      "risk": (accepted - correct) / accepted})
    retained = [correct for confidence, correct in observations if confidence >= reject_threshold]
    return {
        "score_status": "raw_uncalibrated", "bins": bins,
        "ece_predicted_class": ece,
        "brier_ad": math.fsum((p - y) ** 2 for y, p in zip(labels, probabilities)) / len(labels),
        "reliability_bins": reliability, "confidence_histogram": histogram,
        "risk_coverage": curve,
        "fixed_rejection": {"confidence_threshold": reject_threshold,
                            "threshold_source": "declared_before_evaluation_not_fitted",
                            "accepted": len(retained), "referred": len(labels) - len(retained),
                            "coverage": len(retained) / len(labels),
                            "accepted_accuracy": sum(retained) / len(retained) if retained else None,
                            "accepted_risk": (len(retained) - sum(retained)) / len(retained) if retained else None,
                            "full_coverage_accuracy": classification["accuracy"]},
    }


def aggregate_patients(slices: list[dict], aggregation: str = "mean_probability") -> tuple[list[dict], dict]:
    """Average all slice probabilities only when every patient has one diagnosis.

    Never silently assign a label to a longitudinal patient whose diagnoses
    differ. In that case the entire patient-level summary is unavailable.
    """
    if aggregation not in ('mean_probability', 'mean_logit'):
        raise ValueError('Unsupported patient aggregation.')
    groups = {}
    for row in slices:
        groups.setdefault(row["patient_id"], []).append(row)
    mixed = sum(len({r["label"] for r in rows}) > 1 for rows in groups.values())
    status = {"status": "unavailable_mixed_diagnoses" if mixed else "available",
              "n_patients": len(groups), "mixed_diagnosis_patients": mixed,
              "aggregation": "mean_all_slice_AD_probability" if aggregation == 'mean_probability' else 'sigmoid_mean_all_slice_AD_logit'}
    if mixed:
        return [], status
    result = []
    for patient, rows in sorted(groups.items()):
        if aggregation == 'mean_logit':
            values = [r.get('ad_logit') for r in rows]
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
                raise ValueError('Logit aggregation requires finite original AD logits.')
            margin = math.fsum(values) / len(values)
            probability = 1 / (1 + math.exp(-margin)) if margin >= 0 else math.exp(margin) / (1 + math.exp(margin))
        else:
            probability = math.fsum(r["probability"] for r in rows) / len(rows)
        result.append({"patient_id": patient, "label": rows[0]["label"],
                       "probability": probability, "prediction": int(probability >= 0.5),
                       "num_slices": len(rows), "num_scans": len({r["image_id"] for r in rows})})
    return result, status


def prediction_report(slices: list[dict], scans: list[dict], bins: int = 15,
                      reject_threshold: float = 0.8, patient_aggregation: str = "mean_probability") -> tuple[dict, list[dict]]:
    """Describe fixed predictions without tuning or changing a checkpoint."""
    patients, patient_status = aggregate_patients(slices, patient_aggregation)
    rows_by_unit = {"slice": slices, "scan": scans}
    if patients:
        rows_by_unit["patient"] = patients
    confidence = {unit: confidence_metrics([r["label"] for r in rows],
                                           [r["probability"] for r in rows], bins, reject_threshold)
                  for unit, rows in rows_by_unit.items()}
    return {"primary_evaluation_unit": "slice", "secondary_units": ["scan", "patient"],
            "confidence": confidence, "patient_aggregation": patient_status,
            "patient_metrics": binary_metrics([r["label"] for r in patients],
                                               [r["probability"] for r in patients]) if patients else None,
            "final_test_target": {"minimum_accuracy": 0.8, "unit": "slice",
                                  "status": "not_assessed_development_results_only"}}, patients
