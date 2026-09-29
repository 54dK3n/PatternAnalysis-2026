"""Score complete MRI scans without changing model state."""

import time

import torch

from evaluation.metrics import aggregate_scans, binary_metrics
from utils.runtime import sync_device


@torch.inference_mode()
def evaluate(model, loader, device, expected_slices):
    """Aggregate every slice, then compute metrics once per complete scan.

    Forward timing excludes loading and host/device transfer. It is an average
    per slice at the configured batch size, not single-request latency.
    """
    model.eval()
    predictions = []
    forward_seconds = 0.0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        sync_device(device)
        started = time.perf_counter()
        logits = model(images)
        sync_device(device)
        forward_seconds += time.perf_counter() - started
        if not torch.isfinite(logits).all().item():
            raise ValueError("Model produced non-finite logits; evaluation stopped.")
        probabilities = torch.sigmoid(logits).cpu().tolist()
        for index, probability in enumerate(probabilities):
            predictions.append({
                "patient_id": batch["patient_id"][index],
                "image_id": batch["image_id"][index],
                "slice_index": int(batch["slice_index"][index]),
                "relative_path": batch["relative_path"][index],
                "label": int(batch["label"][index]),
                "probability": probability,
            })
    scans = aggregate_scans(predictions, expected_slices=expected_slices)
    scores = {
        level: binary_metrics([r["label"] for r in rows], [r["probability"] for r in rows])
        for level, rows in (("scan", scans), ("slice", predictions))
    }
    scores["n_patients"] = len({row["patient_id"] for row in scans})
    scores["forward_seconds"] = forward_seconds
    scores["mean_forward_ms_per_slice"] = forward_seconds * 1000 / len(predictions)
    return scores, predictions, scans
