"""Validate diagnostic geometry with synthetic patterns, not MRI anatomy."""

import csv
import hashlib
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw
import torch

from dataset.augmentation import make_augmentation
from evaluation.transform_audit import _draw_measurements, run_transform_audit


class TransformGeometryTests(unittest.TestCase):
    """Check that the audit distinguishes border clipping from interpolation."""

    def test_centered_support_is_retained_while_edge_support_is_clipped(self) -> None:
        """A known centered ellipse fits; an equally bright edge bar loses content."""
        centered = Image.new("L", (256, 240), 0)
        ImageDraw.Draw(centered).ellipse((40, 40, 215, 199), fill=200)
        edge = Image.new("L", (256, 240), 0)
        ImageDraw.Draw(edge).rectangle((242, 40, 255, 199), fill=200)
        _, central_records = _draw_measurements(centered, 5, 7.68, 7.2)
        _, edge_records = _draw_measurements(edge, 5, 7.68, 7.2)
        for central, boundary in zip(central_records, edge_records):
            self.assertAlmostEqual(central["foreground_retention_vs_full_reference"], 1, places=6)
            self.assertGreater(central["intensity_mass_retention_vs_full_reference"], .999)
            self.assertLess(boundary["foreground_retention_vs_full_reference"], .8)
            self.assertLess(boundary["intensity_mass_retention_vs_full_reference"], .8)
            self.assertGreater(boundary["full_reference_foreground_pixels"], boundary["fixed_foreground_pixels"])

    def test_zero_geometry_preserves_asymmetric_pixels_and_empty_proxy_is_undefined(self) -> None:
        """Identity does not transpose axes; dark images avoid invented retention."""
        image = Image.new("L", (64, 48), 0)
        ImageDraw.Draw(image).rectangle((10, 5, 12, 30), fill=173)
        actual, records = _draw_measurements(image, 0, 0, 0)
        self.assertEqual(actual.size, (64, 48))
        self.assertEqual(actual.tobytes(), image.tobytes())
        for record in records:
            self.assertEqual(record["foreground_retention_vs_full_reference"], 1)
            self.assertEqual(record["fixed_vs_reference_roi_mae_gray_levels"], 0)
        _, blank_records = _draw_measurements(Image.new("L", (64, 48), 0), 5, 1.92, 1.44)
        for record in blank_records:
            self.assertIsNone(record["foreground_retention_vs_full_reference"])
            self.assertIsNone(record["intensity_mass_retention_vs_full_reference"])


class TransformAuditTests(unittest.TestCase):
    """Exercise source safety, patient sampling, and real input processing roles."""

    def setUp(self) -> None:
        """Create a small immutable grayscale source fixture with repeated visits."""
        temporary = tempfile.TemporaryDirectory(prefix="synthetic_transform_audit_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data_root = self.root / "source"
        self.data_root.mkdir()
        self.rows: list[dict] = []
        for patient, count in (("synthetic_a", 5), ("synthetic_b", 1), ("synthetic_c", 2)):
            for visit in range(count):
                relative = f"{patient}_{visit}.png"
                image = Image.new("L", (64, 48), 0)
                ImageDraw.Draw(image).rectangle((9 + visit, 7, 46, 37), fill=150 + visit)
                image.save(self.data_root / relative)
                self.rows.append({"relative_path": relative, "patient_id": patient,
                                  "image_id": f"{patient}_{visit}", "slice_index": "100",
                                  "label": "1" if patient == "synthetic_b" else "0",
                                  "partition": "development", "width": "64", "height": "48", "mode": "L",
                                  "file_sha256": hashlib.sha256((self.data_root / relative).read_bytes()).hexdigest()})

    def test_patient_balanced_sampling_repeats_and_preserves_sources(self) -> None:
        """Repeated visits cannot produce multiple images from the same patient."""
        original = {path.name: path.read_bytes() for path in self.data_root.iterdir()}
        first = run_transform_audit(self.rows, self.data_root, (48, 64), self.root / "first", 12, 8)
        second = run_transform_audit(list(reversed(self.rows)), self.data_root, (48, 64), self.root / "second", 12, 8)
        self.assertEqual(first, second)
        self.assertEqual(first["n_sampled_images"], 3)
        self.assertEqual(first["n_sampled_patients"], 3)
        for name in ("transform_samples.csv", "transform_draws.csv", "transform_grid.png", "summary.json"):
            self.assertEqual((self.root / "first" / name).read_bytes(), (self.root / "second" / name).read_bytes())
        with (self.root / "first/transform_samples.csv").open() as handle:
            sampled = list(csv.DictReader(handle))
        self.assertEqual(len({row["patient_id"] for row in sampled}), 3)
        self.assertTrue(all(row["none_pixel_identity"] == "True" for row in sampled))
        self.assertTrue(all(row["resize_applied"] == "False" for row in sampled))
        with (self.root / "first/transform_draws.csv").open() as handle:
            draws = list(csv.DictReader(handle))
        self.assertEqual(len(draws), 3 * 12 * 3)
        self.assertEqual({row["threshold"] for row in draws}, {"8", "16", "32"})
        self.assertEqual(original, {path.name: path.read_bytes() for path in self.data_root.iterdir()})

    def test_none_input_identity_matches_the_actual_dataset_and_size_mismatch_is_recorded(self) -> None:
        """Compare the grid's none pixels to production tensors, including resize."""
        from dataset.slices import ADNISliceDataset
        for image_size in ((48, 64), (24, 40)):
            output = self.root / f"size_{image_size[0]}"
            summary = run_transform_audit(self.rows[:1], self.data_root, image_size, output, 1, 0)
            dataset = ADNISliceDataset(self.rows[:1], self.data_root, image_size=image_size, role="train")
            expected = dataset[0]["image"].add(1).mul(127.5).round().to(dtype=torch.uint8)
            with Image.open(output / "transform_grid.png") as grid:
                width, height = image_size[1], image_size[0]
                column_left = summary["grid_cell_width_pixels"]
                none = grid.crop((column_left, 86, column_left + width, 86 + height)).convert("L")
                self.assertEqual(none.tobytes(), bytes(expected.flatten().tolist()))
            self.assertEqual(summary["resize_applied_count"], int(image_size != (48, 64)))

    def test_exif_orientation_is_reported_and_applied_before_size_check(self) -> None:
        """An orientation tag can swap canonical dimensions without model resizing."""
        image = Image.new("L", (48, 64), 128)
        exif = Image.Exif()
        exif[274] = 6
        path = self.data_root / "orientation.jpg"
        image.save(path, exif=exif)
        rows = [{"relative_path": path.name, "patient_id": "synthetic_orientation",
                 "label": "0", "partition": "development", "width": "64", "height": "48", "mode": "L"}]
        summary = run_transform_audit(rows, self.data_root, (48, 64), self.root / "exif", 1, 0)
        self.assertEqual(summary["resize_applied_count"], 0)
        with (self.root / "exif/transform_samples.csv").open() as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual((row["source_width"], row["source_height"]), ("48", "64"))
        self.assertEqual((row["canonical_width"], row["canonical_height"]), ("64", "48"))
        self.assertEqual(row["exif_orientation"], "6")

    def test_held_out_rows_changed_sources_and_unsafe_outputs_are_rejected(self) -> None:
        """The diagnostic must not read other partitions, mutate data, or overwrite results."""
        for changes in ({"partition": "test"}, {"partition": "calibration"}, {"role": "val"},
                        {"relative_path": "../outside.png"}, {"file_sha256": "incorrect"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                run_transform_audit([dict(self.rows[0], **changes)], self.data_root, (48, 64), self.root / "invalid")
        self.assertFalse((self.root / "invalid").exists())
        with self.assertRaisesRegex(ValueError, "source data"):
            run_transform_audit(self.rows, self.data_root, (48, 64), self.data_root / "audit")
        output = self.root / "protected"
        output.mkdir()
        sentinel = output / "sentinel"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "protected"):
            run_transform_audit(self.rows, self.data_root, (48, 64), output)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_checkpoint_custom_bounds_control_draws_metadata_and_known_border_loss(self) -> None:
        """A configured 10% shift removes a narrow edge bar the default partly retains."""
        path = self.data_root / "right_edge.png"
        image = Image.new("L", (64, 48), 0)
        ImageDraw.Draw(image).rectangle((59, 10, 63, 37), fill=200)
        image.save(path)
        rows = [{"relative_path": path.name, "patient_id": "synthetic_border",
                 "label": "0", "partition": "development", "width": "64", "height": "48", "mode": "L"}]
        custom = make_augmentation("light", rotation_degrees=0, translation_fraction=.1)
        output = self.root / "custom"
        summary = run_transform_audit(rows, self.data_root, (48, 64), output, 1, 7, augmentation=custom)
        self.assertEqual(summary["augmentation"], custom.to_dict())
        with (output / "transform_draws.csv").open() as handle:
            draws = list(csv.DictReader(handle))
        self.assertTrue(all(float(row["angle_degrees"]) == 0 for row in draws))
        self.assertTrue(all(abs(float(row["dx_pixels"])) <= 6.4 + 1e-12 and abs(float(row["dy_pixels"])) <= 4.8 + 1e-12
                            for row in draws))
        extreme = next(row for row in draws if row["draw_kind"] == "extreme" and row["draw_index"] == "1"
                       and row["threshold"] == "32")
        self.assertAlmostEqual(float(extreme["dx_pixels"]), 6.4)
        self.assertAlmostEqual(float(extreme["dy_pixels"]), 4.8)
        self.assertEqual(float(extreme["foreground_retention_vs_full_reference"]), 0)
        _, default_records = _draw_measurements(image, 0, 1.92, 1.44)
        self.assertGreater(default_records[-1]["foreground_retention_vs_full_reference"], .4)
        with self.assertRaisesRegex(ValueError, "light augmentation"):
            run_transform_audit(rows, self.data_root, (48, 64), self.root / "none_profile", augmentation=make_augmentation())
        self.assertFalse((self.root / "none_profile").exists())


if __name__ == "__main__":
    unittest.main()
