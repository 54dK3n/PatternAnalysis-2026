"""Verify spatial alignment, clipping controls and frozen-state restoration."""

import csv
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from engine.diagnosis import _integer_shift
from evaluation import shift_diagnostics as spatial


class ToySpatialModel(nn.Module):
    """Expose one identity feature map and a deterministic spatially weighted logit."""

    def __init__(self) -> None:
        """Use a fixed coordinate weight so translation and clipping are distinct."""
        super().__init__()
        self.identity = nn.Identity()
        self.register_buffer("weights", torch.arange(32 * 32, dtype=torch.float32).reshape(1, 1, 32, 32) / 1024)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Reduce weighted features without random operations or learned updates."""
        return ((self.identity(images) + 1) * self.weights).sum((1, 2, 3)) / 100


class ShiftDiagnosticTests(unittest.TestCase):
    """Use synthetic black canvases with a compact nonzero interior."""

    def setUp(self) -> None:
        """Prepare two unequal slice-count patients for equal-patient summaries."""
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images = torch.full((3, 1, 32, 32), -1.0)
        self.images[0, :, 12:16, 10:14] = 0.5
        self.images[1, :, 12:16, 10:14] = 0.5
        self.images[2, :, 16:20, 16:20] = 1.0
        self.batch = {"image": self.images, "patient_id": ["a", "a", "b"], "image_id": ["a1", "a1", "b1"],
                      "slice_index": [0, 1, 0], "relative_path": ["a0.jpg", "a1.jpg", "b0.jpg"],
                      "label": torch.tensor([0, 0, 1])}
        self.model = ToySpatialModel()

    def run_sweep(self, **kwargs: object) -> dict:
        """Keep production collection while substituting one interpretable observer."""
        with mock.patch.object(spatial, "spatial_observers", return_value={"identity": (self.model.identity, 1)}):
            return spatial.run_shift_sweep(self.model, [self.batch], torch.device("cpu"), self.root / "sweep",
                                           max_shift=2, bootstrap_samples=20, **kwargs)

    def test_exact_feature_alignment_signs_and_stride_divisibility(self) -> None:
        """Known shifted arrays must align exactly for positive and negative motion."""
        reference = torch.arange(8 * 9, dtype=torch.float32).reshape(1, 1, 8, 9)
        for dx, dy in ((4, 0), (-4, 0), (0, 4), (0, -4)):
            moved = _integer_shift(reference, dx // 2, dy // 2)
            values, height, width = spatial.aligned_feature_error(reference, moved, dx, dy, 2)
            self.assertEqual(values, [0.0])
            self.assertEqual((height, width), (8 - abs(dy // 2), 9 - abs(dx // 2)))
        self.assertEqual(spatial.aligned_feature_error(reference, reference, 1, 0, 2), (None, 0, 0))
        self.assertEqual(spatial.aligned_feature_error(reference, reference, 18, 0, 2), (None, 0, 0))

    def test_roundtrip_preserves_interior_but_removes_known_edge_content(self) -> None:
        """Clipping is pixel loss rather than interpolation or model uncertainty."""
        self.assertTrue(torch.equal(_integer_shift(_integer_shift(self.images, 2, 0), -2, 0), self.images))
        edge = self.images.clone()
        edge[:, :, :, -1] = 0.0
        returned = _integer_shift(_integer_shift(edge, 2, 0), -2, 0)
        self.assertTrue(torch.all(returned[..., -2:] == -1))
        self.assertFalse(torch.equal(returned, edge))

    def test_empty_proxy_and_margin_thresholds(self) -> None:
        """A blank image has no foreground margin and cannot enter the safe subgroup."""
        margins = spatial.foreground_margins(torch.cat((self.images, self.images.new_full((1, 1, 32, 32), -1))))
        self.assertEqual(margins[16], [10, 10, 12, -1])
        self.assertEqual(set(margins), {8, 16, 32})

    def test_tiny_and_zero_gap_norms_do_not_invent_cosine_change(self) -> None:
        """Identical tiny vectors have zero distance; zero vectors are undefined."""
        tiny = torch.full((1, 2, 4, 4), 1e-20)
        measured = spatial.gap_feature_changes(tiny, tiny)
        self.assertAlmostEqual(measured["gap_cosine_distance"][0], 0.0, places=12)
        self.assertEqual(measured["gap_absolute_l2"], [0.0])
        self.assertEqual(measured["reference_gap_zero"], [False])
        undefined = spatial.gap_feature_changes(torch.zeros_like(tiny), tiny)
        self.assertEqual(undefined["gap_cosine_distance"], [None])
        self.assertEqual(undefined["reference_gap_zero"], [True])

    def test_actual_shift_outputs_and_equal_patient_summary(self) -> None:
        """Compare measurements with analytical translated weighted coordinates."""
        state = copy.deepcopy(self.model.state_dict())
        self.model.train()
        self.model.identity.eval()
        flags = [m.training for m in self.model.modules()]
        result = self.run_sweep()
        self.assertEqual(result["forward_passes_per_slice"], 17)
        with (self.root / "sweep/shift_slice_predictions.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        selected = [r for r in rows if r["dx"] == "2" and r["dy"] == "0"]
        # Each nonblack pixel's coordinate weight increases by 2/1024.
        expected_logit = [16 * 1.5 * 2 / 1024 / 100, 16 * 1.5 * 2 / 1024 / 100, 16 * 2 * 2 / 1024 / 100]
        for row, expected in zip(selected, expected_logit):
            self.assertAlmostEqual(float(row["logit_change"]), expected, places=6)
            self.assertEqual(float(row["roundtrip_pixel_mae"]), 0.0)
            self.assertEqual(float(row["roundtrip_probability_change"]), 0.0)
        summary = next(r for r in result["groups"] if r["dx"] == 2 and r["dy"] == 0 and r["group"] == "all")
        self.assertAlmostEqual(summary["logit_change"]["mean"], (expected_logit[0] + expected_logit[2]) / 2, places=6)
        self.assertEqual(summary["patients"], 2)
        self.assertEqual(summary["NC_slices"], 2)
        self.assertEqual(summary["AD_slices"], 1)
        self.assertEqual([m.training for m in self.model.modules()], flags)
        self.assertFalse(self.model.identity._forward_hooks)
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, state[key]))

    def test_exception_removes_hooks_and_restores_modes(self) -> None:
        """Interrupted inspection must not leave observers on a caller model."""
        self.model.train()
        with mock.patch.object(self.model, "forward", side_effect=RuntimeError("deliberate")), \
                self.assertRaisesRegex(RuntimeError, "deliberate"):
            self.run_sweep()
        self.assertTrue(self.model.training)
        self.assertFalse(self.model.identity._forward_hooks)

    def test_confidence_strata_and_invalid_displacements(self) -> None:
        """Fixed bins and bounded shifts cannot adapt to reported scores."""
        self.assertEqual(spatial.confidence_stratum(.55), "margin_0_to_0.1")
        self.assertEqual(spatial.confidence_stratum(.7), "margin_0.1_to_0.3")
        self.assertEqual(spatial.confidence_stratum(.95), "margin_0.3_to_0.5")
        for displacement in (0, 9, True):
            with self.assertRaises(ValueError):
                spatial.run_shift_sweep(self.model, [], torch.device("cpu"), self.root / "invalid", max_shift=displacement)
        self.assertFalse((self.root / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
