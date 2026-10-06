"""Inspect training transforms without fitting models or reading held-out images.

Thresholded intensity is a foreground proxy, never a brain segmentation. The
padded reference measures content outside the fixed output canvas after the
same single bilinear operation; it does not estimate clinical information loss.
"""

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any

import PIL
from PIL import Image, ImageDraw, ImageOps

from dataset.augmentation import AugmentationConfig, augment_image, make_augmentation


THRESHOLDS = (8, 16, 32)


class _FixedDraw:
    """Replay one angle and absolute translation through the production API."""

    def __init__(self, angle: float, dx: float, dy: float, size: tuple[int, int]) -> None:
        self.values = iter((angle, dx / size[0], dy / size[1]))

    def uniform(self, lower: float, upper: float) -> float:
        """Return a predetermined in-range draw without consuming global RNG state."""
        value = next(self.values)
        if not lower - 1e-12 <= value <= upper + 1e-12:
            raise ValueError("A replayed transform draw exceeds the light profile bounds.")
        return value


def _transform(image: Image.Image, angle: float, dx: float, dy: float,
               augmentation: AugmentationConfig) -> Image.Image:
    """Use the actual single-operation light transform with fixed sampled values."""
    return augment_image(image, augmentation, _FixedDraw(angle, dx, dy, image.size))


def _foreground(image: Image.Image, threshold: int) -> dict[str, Any]:
    """Count a thresholded intensity proxy and its distances from canvas edges."""
    mask = image.point(lambda value: 255 if value >= threshold else 0)
    box = mask.getbbox()
    count = sum(mask.tobytes()) // 255
    if box is None:
        return {"count": 0, "left": None, "top": None, "right": None,
                "bottom": None, "minimum_margin": None}
    left, top, right, bottom = box
    margins = (left, top, image.width - right, image.height - bottom)
    return {"count": count, "left": margins[0], "top": margins[1],
            "right": margins[2], "bottom": margins[3], "minimum_margin": min(margins)}


def _draw_measurements(image: Image.Image, angle: float, dx: float, dy: float,
                       augmentation: AugmentationConfig | None = None) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Compare a fixed canvas with a roomy, same-center, single-resampling reference."""
    augmentation = make_augmentation("light") if augmentation is None else augmentation
    if not isinstance(augmentation, AugmentationConfig) or augmentation.name != "light":
        raise ValueError("Transform measurements require a light augmentation profile.")
    if image.mode != "L":
        raise ValueError("Transform measurements require the production uint8 grayscale input.")
    # The diagonal bound encloses the original image at every possible angle.
    # Padding is only a diagnostic reference, never an alternative model input.
    radius = math.hypot(image.width, image.height) / 2
    pad = math.ceil(radius + max(abs(dx), abs(dy)) - min(image.size) / 2) + 4
    padded = ImageOps.expand(image, border=pad, fill=0)
    reference = _transform(padded, angle, dx, dy, augmentation)
    fixed = _transform(image, angle, dx, dy, augmentation)
    reference_roi = reference.crop((pad, pad, pad + image.width, pad + image.height))
    roi_mae = statistics.mean(abs(a - b) for a, b in zip(fixed.tobytes(), reference_roi.tobytes()))
    full_mass = sum(reference.tobytes())
    fixed_mass = sum(fixed.tobytes())
    mass_retention = fixed_mass / full_mass if full_mass else None
    records = []
    for threshold in THRESHOLDS:
        original_proxy = _foreground(image, threshold)
        fixed_proxy = _foreground(fixed, threshold)
        full_proxy = _foreground(reference, threshold)
        retained = fixed_proxy["count"] / full_proxy["count"] if full_proxy["count"] else None
        records.append({
            "threshold": threshold, "angle_degrees": angle, "dx_pixels": dx,
            "dy_pixels": dy, "original_foreground_pixels": original_proxy["count"],
            "original_minimum_margin_pixels": original_proxy["minimum_margin"],
            "fixed_foreground_pixels": fixed_proxy["count"],
            "full_reference_foreground_pixels": full_proxy["count"],
            "foreground_retention_vs_full_reference": retained,
            "intensity_mass_retention_vs_full_reference": mass_retention,
            "fixed_vs_reference_roi_mae_gray_levels": roi_mae,
            "reference_padding_pixels": pad,
        })
    return fixed, records


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write finite audit rows using a stable field order."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values: list[float], fraction: float) -> float | None:
    """Return a linearly interpolated percentile, including small audit samples."""
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _validate_rows(rows: list[dict], data_root: Path) -> dict[str, list[dict]]:
    """Require development rows and safe, unique source paths before sampling."""
    if not rows or not data_root.is_dir():
        raise ValueError("Training rows and an existing source directory are required.")
    groups: dict[str, list[dict]] = {}
    seen = set()
    for row in rows:
        if row.get("partition") != "development":
            raise ValueError("Transform auditing accepts development training rows only.")
        if row.get("role", "train") != "train":
            raise ValueError("Transform auditing accepts the train role only.")
        relative = row.get("relative_path")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Training slice paths must remain inside the source directory.")
        path = (data_root / relative).resolve()
        if not path.is_relative_to(data_root) or not path.is_file() or relative in seen:
            raise ValueError("A training source path is missing, duplicated, or outside its root.")
        patient = row.get("patient_id")
        if not isinstance(patient, str) or not patient or str(row.get("label")) not in ("0", "1"):
            raise ValueError("Training rows require patient identifiers and binary labels.")
        seen.add(relative)
        groups.setdefault(patient, []).append(row)
    return groups


def run_transform_audit(rows: list[dict], data_root: Path, image_size: tuple[int, int],
                        output: Path, max_images: int = 12, seed: int = 3710, *,
                        augmentation: AugmentationConfig | None = None) -> dict:
    """Audit a fixed patient-balanced sample from an already verified train manifest.

    The caller must supply the fold's train rows: a development partition alone
    cannot prove the training role. One slice is chosen per sampled patient so
    frequent longitudinal visits do not dominate the diagnostic sample. This
    audits eight random and four extreme draws per image using the supplied
    checkpoint light bounds, or the default light profile when omitted; held-out sets are
    never loaded. The original source files and production transforms stay intact.
    """
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    augmentation = make_augmentation("light") if augmentation is None else augmentation
    if not isinstance(augmentation, AugmentationConfig) or augmentation.name != "light":
        raise ValueError("Transform auditing requires a light augmentation profile.")
    if (len(image_size) != 2 or any(type(value) is not int or value <= 0 for value in image_size)
            or type(max_images) is not int or max_images < 1 or type(seed) is not int):
        raise ValueError("Image dimensions, max_images, and seed must have valid integer values.")
    if output == data_root or output.is_relative_to(data_root):
        raise ValueError("Audit outputs must not be written inside the source data directory.")
    groups = _validate_rows(rows, data_root)
    if output.exists():
        raise ValueError("Choose a new transform audit output directory; existing outputs are protected.")
    rng = random.Random(seed)
    patients = rng.sample(sorted(groups), min(max_images, len(groups)))
    sampled = [rng.choice(sorted(groups[patient], key=lambda row: row["relative_path"])) for patient in patients]
    image_records: list[dict[str, Any]] = []
    draw_records: list[dict[str, Any]] = []
    panels: list[list[Image.Image]] = []
    target_size = (image_size[1], image_size[0])
    rotation, translation = augmentation.rotation_degrees, augmentation.translation_fraction
    for sample_index, row in enumerate(sampled, start=1):
        path = data_root / row["relative_path"]
        content = path.read_bytes()
        if row.get("file_sha256") and hashlib.sha256(content).hexdigest() != row["file_sha256"]:
            raise ValueError("A sampled training image changed after manifest verification.")
        with Image.open(io.BytesIO(content)) as source:
            source_mode, source_size = source.mode, source.size
            orientation = source.getexif().get(274, 1)
            canonical = ImageOps.exif_transpose(source).convert("L")
            resize_applied = canonical.size != target_size
            none_image = canonical.resize(target_size, Image.Resampling.BILINEAR) if resize_applied else canonical.copy()
            actual = {"source_width": source_size[0], "source_height": source_size[1],
                      "canonical_width": canonical.width, "canonical_height": canonical.height,
                      "source_mode": source_mode, "exif_orientation": orientation,
                      "exif_transform_applied": orientation in (2, 3, 4, 5, 6, 7, 8),
                      "source_channel_extrema": str(source.getextrema()),
                      "none_uint8_min": min(none_image.tobytes()), "none_uint8_max": max(none_image.tobytes()),
                      "none_unique_gray_levels": len(set(none_image.tobytes()))}
            image_record = {"sample_index": sample_index, "relative_path": row["relative_path"],
                            "patient_id": row["patient_id"], "image_id": row.get("image_id", ""),
                            "slice_index": row.get("slice_index", ""), "label": row["label"], **actual,
                            "target_width": target_size[0], "target_height": target_size[1],
                            "resize_applied": resize_applied, "grayscale_conversion_applied": source_mode != "L",
                            "manifest_dimensions_match": (str(row.get("width", canonical.width)) == str(canonical.width)
                                                          and str(row.get("height", canonical.height)) == str(canonical.height)),
                            "manifest_mode_match": row.get("mode", source_mode) == source_mode,
                            "none_pixel_identity": not resize_applied and canonical.tobytes() == none_image.tobytes()}
            for threshold in THRESHOLDS:
                proxy = _foreground(none_image, threshold)
                for key, value in proxy.items():
                    image_record[f"foreground_t{threshold}_{key}"] = value
            image_records.append(image_record)
            draws = [("random", index + 1, rng.uniform(-rotation, rotation),
                      rng.uniform(-translation, translation) * target_size[0],
                      rng.uniform(-translation, translation) * target_size[1]) for index in range(8)]
            # Four diagonal boundary cases cover both rotation signs and both
            # direction signs. They are illustrative extremes, not all corners.
            draws += [("extreme", index + 1, angle, dx * target_size[0], dy * target_size[1])
                      for index, (angle, dx, dy) in enumerate(((rotation, translation, translation),
                                                               (rotation, -translation, -translation),
                                                               (-rotation, translation, -translation),
                                                               (-rotation, -translation, translation)))]
            selected_panels = [canonical.copy(), none_image.copy()]
            for kind, draw_index, angle, dx, dy in draws:
                fixed, measurements = _draw_measurements(none_image, angle, dx, dy, augmentation)
                if kind == "random" and draw_index == 1:
                    selected_panels.append(fixed.copy())
                if kind == "extreme" and draw_index == 1:
                    selected_panels.append(fixed.copy())
                for measurement in measurements:
                    draw_records.append({"sample_index": sample_index, "patient_id": row["patient_id"],
                                         "image_id": row.get("image_id", ""), "draw_kind": kind,
                                         "draw_index": draw_index, **measurement})
            panels.append(selected_panels)
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "transform_samples.csv", image_records)
    _write_csv(output / "transform_draws.csv", draw_records)
    cell_width, cell_height = max(target_size[0], 256), target_size[1]
    header, row_caption = 58, 28
    grid = Image.new("RGB", (cell_width * 4, header + (cell_height + row_caption) * len(panels)), "white")
    painter = ImageDraw.Draw(grid)
    painter.text((8, 5), "TRAIN ONLY; intensity foreground proxy is NOT a brain mask", fill="black")
    extreme_title = f"+{rotation:g} deg, +{100 * translation:g}% x/y"
    for column, title in enumerate(("Original / canonical L", "None model input", "Light random draw 1", extreme_title)):
        painter.text((column * cell_width + 8, 30), title, fill="black")
    for index, images in enumerate(panels):
        top = header + index * (cell_height + row_caption)
        painter.text((8, top), f"Training sample {index + 1}; label {image_records[index]['label']}", fill="black")
        for column, image in enumerate(images):
            # Keep native pixels in original panels; thumbnail only when needed
            # to fit a differing source size, and record the display distinction.
            display = image.copy()
            if display.size != target_size:
                display.thumbnail(target_size, Image.Resampling.BILINEAR)
            grid.paste(display.convert("RGB"), (column * cell_width, top + row_caption))
    grid.save(output / "transform_grid.png")
    summaries = {}
    for threshold in THRESHOLDS:
        by_kind = {}
        for kind in ("random", "extreme"):
            selected = [record for record in draw_records if record["threshold"] == threshold and record["draw_kind"] == kind]
            losses = [1 - record["foreground_retention_vs_full_reference"] for record in selected
                      if record["foreground_retention_vs_full_reference"] is not None]
            by_kind[kind] = {"n_draws": len(selected), "nonempty_proxy_draws": len(losses),
                             "mean_foreground_loss_fraction": statistics.mean(losses) if losses else None,
                             "p95_foreground_loss_fraction": _percentile(losses, .95),
                             "maximum_foreground_loss_fraction": max(losses) if losses else None}
        summaries[str(threshold)] = by_kind
    summary = {
        "status": "complete", "role": "train", "seed": seed,
        "sampling": "uniform_patients_without_replacement_then_one_uniform_training_slice",
        "n_training_rows": len(rows), "n_training_patients": len(groups), "n_sampled_images": len(sampled),
        "n_sampled_patients": len(patients), "image_size_height_width": list(image_size),
        "grid_cell_width_pixels": cell_width,
        "random_draws_per_image": 8, "extreme_draws_per_image": 4,
        "augmentation": augmentation.to_dict(), "pillow_version": PIL.__version__,
        "resize_applied_count": sum(record["resize_applied"] for record in image_records),
        "exif_transform_applied_count": sum(record["exif_transform_applied"] for record in image_records),
        "grayscale_conversion_applied_count": sum(record["grayscale_conversion_applied"] for record in image_records),
        "manifest_dimension_mismatch_count": sum(not record["manifest_dimensions_match"] for record in image_records),
        "manifest_mode_mismatch_count": sum(not record["manifest_mode_match"] for record in image_records),
        "foreground_proxy_thresholds": list(THRESHOLDS), "foreground_loss_vs_full_reference": summaries,
        "interpretation": "Thresholded intensity is a foreground proxy, not a brain mask or clinical information measure.",
        "limitations": ["A small patient-balanced training sample does not audit every source image.",
                        "Four illustrated extremes do not enumerate every combination of extreme draws.",
                        "Interpolation changes threshold membership; losses are measured against the same transformed padded reference.",
                        "Pillow handles source boundaries differently after padding; small ROI pixel differences or retention above one can arise from interpolation and are not negative anatomical losses.",
                        "Original grid cells show EXIF-corrected grayscale images and are display-scaled only if their size differs.",
                        "The diagnostic sample uses an independent fixed seed, not the exact training worker RNG stream."],
        "artifacts": {"samples_csv": "transform_samples.csv", "draws_csv": "transform_draws.csv",
                      "grid_png": "transform_grid.png", "summary_json": "summary.json"},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary
