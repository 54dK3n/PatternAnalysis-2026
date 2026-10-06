"""Exercise development-only diagnostics without training a backbone or using ADNI."""

import argparse
from collections import Counter
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest import mock

import torch

from dataset.augmentation import AugmentationConfig
from dataset.loaders import make_loader
from dataset.manifests import manifest_sha256
from engine import diagnosis
from models import SmallCNN
from models.convnext import ConvNeXtBlock, ConvNeXtTiny
import test_adni_splits as split_fixture


class DiagnosisTests(unittest.TestCase):
    """Use real sealed synthetic manifests and a small random-weight checkpoint."""

    def setUp(self) -> None:
        """Prepare complete two-slice scans and an independently stored checkpoint."""
        torch.set_num_threads(1)
        self.fixture = split_fixture.ADNISplitTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.prepare()
        self.run_dir = self.fixture.base / "original_run"
        self.run_dir.mkdir()
        self.checkpoint = self.run_dir / "best.pt"
        self.output = self.fixture.base / "diagnostic_output"
        self.config = {
            "checkpoint_format_version": 2,
            "model_name": "small_cnn_v1",
            "initialization": "random",
            "pretrained_weights": None,
            "fold": 1,
            "seed": 3711,
            "seed_base": 3710,
            "image_size": [32, 32],
            "expected_slices": 2,
            "normalization": "(grayscale_uint8 / 255 - 0.5) / 0.5",
            "augmentation": "none",
            "augmentation_config": AugmentationConfig().to_dict(),
            "aggregation": "mean_slice_AD_probability",
            "threshold": 0.5,
            "calibration": "not_fitted",
            "checkpoint_selection": "minimum_early_stop_scan_log_loss",
            "manifest_sha256": manifest_sha256(self.fixture.out),
        }
        self.model = SmallCNN()
        self.state = {
            name: value.detach().clone() for name, value in self.model.state_dict().items()
        }
        self.save_checkpoint()

    def save_checkpoint(self, config: dict | None = None) -> None:
        """Write only ordinary checkpoint metadata and tensor weights."""
        torch.save({
            "config": copy.deepcopy(self.config if config is None else config),
            "model_state": self.state,
            "epoch": 1,
            "early_stop_scan_loss": 0.69,
        }, self.checkpoint)

    def rows(self, role: str) -> list[dict]:
        """Read one original fold role for independent membership assertions."""
        return split_fixture.read_csv(self.fixture.out / "fold_01" / f"{role}.csv")

    def args(self, **changes: object) -> argparse.Namespace:
        """Keep the diagnostic fixture bounded to a few whole scans on CPU."""
        values = {
            "checkpoint": self.checkpoint,
            "data_root": self.fixture.root,
            "splits_dir": self.fixture.out,
            "output": self.output,
            "batch_size": 4,
            "workers": 0,
            "threads": 1,
            "device": "cpu",
            "seed": 3710,
            "max_train_patients": 4,
            "max_early_patients": 0,
            "max_scans_per_patient": 1,
            "max_transform_images": 1,
            "shift_pixels": 2,
            "probes": False,
            "probe_epochs": 2,
            "probe_lr": 0.01,
            "probe_weight_decay": 0.01,
        }
        values.update(changes)
        return argparse.Namespace(**values)

    def loader(self) -> torch.utils.data.DataLoader:
        """Load complete training scans with fixed unaugmented preprocessing."""
        rows = diagnosis.select_complete_scans(self.rows("train"), 2, 3, 1, 3710)
        return make_loader(rows, self.fixture.root, (32, 32), 4, 0, 3710,
                           False, torch.device("cpu"), role="train")

    @staticmethod
    def hook_counts(model: torch.nn.Module) -> list[tuple[int, int]]:
        """Include both kinds of hooks so exceptions cannot leave hidden observers."""
        return [(len(module._forward_hooks), len(module._forward_pre_hooks))
                for module in model.modules()]

    def test_selection_is_reproducible_and_retains_whole_scans(self) -> None:
        """Cap patients and scans without truncating any selected scan."""
        original = self.rows("train")
        snapshot = copy.deepcopy(original)
        selected = diagnosis.select_complete_scans(original, 2, 4, 1, 3710)
        self.assertEqual(original, snapshot)
        self.assertEqual(selected, diagnosis.select_complete_scans(original, 2, 4, 1, 3710))
        self.assertEqual(len({row["patient_id"] for row in selected}), 4)
        scans = Counter(row["image_id"] for row in selected)
        self.assertEqual(set(scans.values()), {2})
        self.assertEqual(len(scans), 4)
        self.assertTrue(split_fixture.paths(selected).issubset(split_fixture.paths(original)))

    def test_unlimited_selection_preserves_valid_longitudinal_label_changes(self) -> None:
        """A patient's later diagnosis may differ without invalidating scan analysis."""
        rows = self.rows("train")
        labels = {}
        for row in rows:
            labels.setdefault(row["patient_id"], set()).add(row["label"])
        self.assertTrue(any(len(values) > 1 for values in labels.values()))
        selected = diagnosis.select_complete_scans(rows, 2, 0, 0, 3710)
        self.assertEqual(split_fixture.paths(selected), split_fixture.paths(rows))
        self.assertEqual(len(selected), len(rows))

    def test_convnext_residual_statistics_use_actual_scaled_branch(self) -> None:
        """Known LayerScale and constant residuals yield independently computed ratios."""
        with mock.patch.object(ConvNeXtTiny, "channels", (8, 16, 32, 64)), \
                mock.patch.object(ConvNeXtTiny, "depths", (1, 1, 1, 1)):
            model = ConvNeXtTiny()
        model.train()
        model.final_norm.eval()
        flags = [module.training for module in model.modules()]
        for block in model.modules():
            if isinstance(block, ConvNeXtBlock):
                with torch.no_grad():
                    block.layer_scale.zero_()
        _, _, zeros, _ = diagnosis.collect_features(model, self.loader(), torch.device("cpu"), 0)
        self.assertEqual(len(zeros["residual_branches"]), 4)
        for stats in zeros["residual_branches"].values():
            self.assertEqual(stats["mean_residual_to_skip_norm"], 0.0)
            self.assertEqual(stats["ratio_max"], 0.0)
            self.assertEqual(stats["layer_scale"]["maximum"], 0.0)
        self.assertEqual([module.training for module in model.modules()], flags)
        first = model.stages[0][0]
        with torch.no_grad():
            first.project.weight.zero_()
            first.project.bias.fill_(2.0)
            first.layer_scale.fill_(0.25)
        expected = []

        def independently_measure_skip(module: torch.nn.Module, inputs: tuple) -> None:
            """A constant 0.5 residual has norm 0.5 times the square root of its size."""
            values = inputs[0].detach().flatten(1)
            expected.extend((0.5 * values.shape[1] ** 0.5 / values.norm(dim=1)).tolist())

        observer = first.register_forward_pre_hook(independently_measure_skip)
        self.addCleanup(observer.remove)
        counts = self.hook_counts(model)
        _, _, measured, _ = diagnosis.collect_features(model, self.loader(), torch.device("cpu"), 0)
        actual = measured["residual_branches"]["stages.0.0"]
        self.assertAlmostEqual(actual["mean_residual_to_skip_norm"], sum(expected) / len(expected), places=6)
        self.assertAlmostEqual(actual["ratio_max"], max(expected), places=6)
        self.assertEqual(actual["layer_scale"]["minimum"], 0.25)
        self.assertEqual(actual["layer_scale"]["maximum"], 0.25)
        self.assertEqual(self.hook_counts(model), counts)
        self.assertEqual([module.training for module in model.modules()], flags)

    def test_selection_rejects_incomplete_and_conflicting_scans(self) -> None:
        """Malformed scan coverage must fail even when a sample cap is requested."""
        rows = self.rows("train")
        conflicting = copy.deepcopy(rows)
        conflicting[0]["patient_id"] = "different_patient"
        for corrupt in (rows[1:], rows + [rows[0]], conflicting):
            with self.subTest(kind=len(corrupt)), self.assertRaises(ValueError):
                diagnosis.select_complete_scans(corrupt, 2, 4, 1, 3710)

    def test_feature_collection_preserves_weights_mode_and_hooks(self) -> None:
        """Measure true frozen features while restoring caller-owned model state."""
        loader = self.loader()
        self.model.train()
        hooks = self.hook_counts(self.model)
        features, predictions, layers, shifts = diagnosis.collect_features(
            self.model, loader, torch.device("cpu"), shift_pixels=2)
        self.assertTrue(self.model.training)
        self.assertEqual(self.hook_counts(self.model), hooks)
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, self.state[name]), name)
        self.assertTrue(features)
        self.assertTrue(layers)
        self.assertTrue(shifts)
        for values in features.values():
            self.assertEqual(values.device.type, "cpu")
            self.assertEqual(values.ndim, 2)
            self.assertEqual(len(values), len(loader.dataset))
            self.assertFalse(values.requires_grad)
            self.assertTrue(torch.isfinite(values).all())
        self.assertEqual(len(predictions), len(loader.dataset))
        self.assertEqual(split_fixture.paths(predictions), split_fixture.paths(loader.dataset.rows))
        json.dumps({"layers": layers, "shifts": shifts}, allow_nan=False)

    def test_feature_collection_removes_hooks_on_failure(self) -> None:
        """A failed forward pass cannot contaminate a subsequent model use."""
        hooks = self.hook_counts(self.model)
        self.model.train()
        original_forward = self.model.forward

        def broken_forward(images: torch.Tensor) -> torch.Tensor:
            """Trigger every ordinary feature hook before simulating failure."""
            original_forward(images)
            raise RuntimeError("deliberate diagnostic failure")

        with mock.patch.object(self.model, "forward", side_effect=broken_forward):
            with self.assertRaisesRegex(RuntimeError, "deliberate diagnostic failure"):
                diagnosis.collect_features(self.model, self.loader(), torch.device("cpu"))
        self.assertTrue(self.model.training)
        self.assertEqual(self.hook_counts(self.model), hooks)

    def test_feature_collection_rejects_nonfinite_logits(self) -> None:
        """Invalid output probabilities must not be published as valid measurements."""
        loader = self.loader()
        with torch.no_grad():
            self.model.classifier[1].bias.fill_(float("nan"))
        hooks = self.hook_counts(self.model)
        with self.assertRaises((ValueError, RuntimeError)):
            diagnosis.collect_features(self.model, loader, torch.device("cpu"))
        self.assertEqual(self.hook_counts(self.model), hooks)

    def test_run_reads_only_inner_roles_and_keeps_checkpoint_unchanged(self) -> None:
        """The full diagnostic path must not expose outer or reserved samples."""
        expected = {role: split_fixture.paths(self.rows(role)) for role in ("train", "early_stop")}
        protected = set()
        for path in (self.fixture.out / "fold_01/val.csv", self.fixture.out / "calibration.csv",
                     self.fixture.out / "test.csv"):
            protected |= split_fixture.patients(split_fixture.read_csv(path))
        observed = []
        original_loader = diagnosis.make_loader

        def checked_loader(rows: list[dict], *args: object, **kwargs: object) -> object:
            """Inspect actual loader membership instead of trusting report labels."""
            role = kwargs.get("role")
            self.assertIn(role, expected)
            self.assertTrue(split_fixture.paths(rows).issubset(expected[role]))
            self.assertTrue(split_fixture.patients(rows).isdisjoint(protected))
            self.assertEqual(set(Counter(row["image_id"] for row in rows).values()), {2})
            observed.append(role)
            return original_loader(rows, *args, **kwargs)

        before = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        with mock.patch.object(diagnosis, "make_loader", side_effect=checked_loader), \
                contextlib.redirect_stdout(io.StringIO()):
            result = diagnosis.run(self.args())
        self.assertEqual(set(observed), {"train", "early_stop"})
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["evaluation_role"], "inner_feature_diagnosis")
        self.assertEqual(set(result["roles"]), {"train", "early_stop"})
        published = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(published["roles"].keys(), result["roles"].keys())
        self.assertEqual(hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(), before)
        self.assertFalse(list(self.output.rglob("*.pt")))
        self.assertFalse(list(self.output.rglob("*.pth")))

    def test_legacy_checkpoint_is_still_supported(self) -> None:
        """Format-one unaugmented checkpoints remain usable after diagnostics arrive."""
        self.config["checkpoint_format_version"] = 1
        self.config.pop("augmentation_config")
        self.save_checkpoint()
        with contextlib.redirect_stdout(io.StringIO()):
            result = diagnosis.run(self.args(max_train_patients=2, max_early_patients=2))
        self.assertEqual(result["status"], "complete")

    def test_legacy_cnn_without_initialization_fields_preserves_missing_provenance(self) -> None:
        """Early baseline checkpoints omitted both fields; never invent their values."""
        self.config["checkpoint_format_version"] = 1
        for key in ("initialization", "pretrained_weights", "augmentation_config"):
            self.config.pop(key)
        self.save_checkpoint()
        before = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        with contextlib.redirect_stdout(io.StringIO()):
            result = diagnosis.run(self.args(max_train_patients=2, max_early_patients=2))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["initialization_metadata_status"], "legacy_fields_not_recorded")
        recorded = json.loads((self.output / "config.json").read_text())
        self.assertEqual(recorded["initialization_metadata_status"], "legacy_fields_not_recorded")
        self.assertNotIn("initialization", recorded["checkpoint_config"])
        self.assertNotIn("pretrained_weights", recorded["checkpoint_config"])
        self.assertEqual(hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(), before)

    def test_missing_initialization_exception_is_limited_to_legacy_cnn(self) -> None:
        """Incomplete modern, ConvNeXt, or explicit pretrained records stay rejected."""
        for model, version in (("small_cnn_v1", 2), ("convnext_tiny_v1", 1),
                               ("convnext_tiny_v1", 2)):
            config = {**self.config, "model_name": model, "checkpoint_format_version": version}
            config.pop("initialization")
            config.pop("pretrained_weights")
            with self.subTest(model=model, version=version), self.assertRaises(ValueError):
                diagnosis._validate_checkpoint({"config": config, "epoch": 1})
        for key in ("initialization", "pretrained_weights"):
            config = {**self.config, "checkpoint_format_version": 1}
            config.pop(key)
            with self.subTest(missing_key=key), self.assertRaises(ValueError):
                diagnosis._validate_checkpoint({"config": config, "epoch": 1})
        for pretrained in ("external.pt", "", False, []):
            config = {**self.config, "pretrained_weights": pretrained}
            with self.subTest(pretrained=pretrained), self.assertRaises(ValueError):
                diagnosis._validate_checkpoint({"config": config, "epoch": 1})

    def test_real_probes_exclude_full_role_mixed_patients_and_freeze_backbone(self) -> None:
        """A one-scan subset cannot hide longitudinal labels from patient-probe exclusions."""
        mixed = {}
        expected = {}
        for role in ("train", "early_stop"):
            labels = {}
            for row in self.rows(role):
                labels.setdefault(row["patient_id"], set()).add(row["label"])
            mixed[role] = {patient for patient, values in labels.items() if len(values) > 1}
            expected[role] = set(labels) - mixed[role]
        self.assertTrue(mixed["train"])
        before = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        with mock.patch.object(diagnosis, "create_model", return_value=self.model), \
                contextlib.redirect_stdout(io.StringIO()):
            result = diagnosis.run(self.args(probes=True, probe_epochs=2, max_train_patients=0,
                                              max_scans_per_patient=1, shift_pixels=0))
        self.assertEqual(result["feature_probes"]["status"], "complete")
        self.assertEqual(result["checkpoint_sha256"], before)
        self.assertEqual(hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(), before)
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, self.state[name]), name)
        records = split_fixture.read_csv(self.output / "probes/feature_probe_patient_predictions.csv")
        self.assertEqual({row["role"] for row in records}, {"train", "early_stop"})
        for role in ("train", "early_stop"):
            exclusions = result["probe_patient_exclusions"][role]
            self.assertEqual(set(exclusions["patient_ids"]), mixed[role])
            selected = split_fixture.read_csv(self.output / f"{role}_selected_slices.csv")
            self.assertTrue(mixed[role].issubset(split_fixture.patients(selected)))
            actual = {row["patient_id"] for row in records if row["role"] == role}
            self.assertEqual(actual, expected[role])
            self.assertTrue(actual.isdisjoint(mixed[role]))
        saved = json.loads((self.output / "probes/feature_probes.json").read_text())
        self.assertFalse(saved["backbone_trained"])
        self.assertEqual(saved["fitting"]["full_batch_steps"], 2)
        self.assertFalse(list(self.output.rglob("*.pt")))

    def test_wrong_manifest_hash_stops_before_model_construction(self) -> None:
        """A valid checkpoint from another split must not score this cohort."""
        self.config["manifest_sha256"] = "0" * 64
        self.save_checkpoint()
        with mock.patch.object(diagnosis, "create_model") as constructor, \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            diagnosis.run(self.args())
        constructor.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_changed_manifest_cannot_bypass_mandatory_audit(self) -> None:
        """Existing split verification must precede any feature model execution."""
        path = self.fixture.out / "fold_01/train.csv"
        path.write_text(path.read_text() + "unexpected,row\n")
        with mock.patch.object(diagnosis, "create_model") as constructor, \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            diagnosis.run(self.args())
        constructor.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_unsupported_checkpoint_settings_fail_before_model_construction(self) -> None:
        """Reject incompatible formats and preprocessing instead of guessing semantics."""
        for changes in ({"checkpoint_format_version": 3}, {"augmentation": "light"},
                        {"normalization": "other"}, {"calibration": "fitted"},
                        {"fold": True}, {"fold": 1.5}, {"expected_slices": True},
                        {"image_size": [32, 0]}, {"model_name": "unknown"}):
            with self.subTest(changes=changes):
                self.save_checkpoint({**self.config, **changes})
                with mock.patch.object(diagnosis, "create_model") as constructor, \
                        contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                    diagnosis.run(self.args())
                constructor.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_output_must_not_overwrite_or_enter_protected_locations(self) -> None:
        """Keep original experiments, source images, and sealed manifests separate."""
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("original")
        for destination in (self.output, self.fixture.root / "diagnosis", self.fixture.out / "diagnosis",
                            self.run_dir / "diagnosis", self.checkpoint):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                diagnosis.run(self.args(output=destination))
        self.assertEqual(marker.read_text(), "original")
        self.assertFalse((self.run_dir / "diagnosis").exists())

    def test_cli_cannot_select_outer_validation_or_protected_roles(self) -> None:
        """Neither public command-line role selection nor a final-test switch exists."""
        base = ["--checkpoint", str(self.checkpoint), "--data-root", str(self.fixture.root),
                "--splits-dir", str(self.fixture.out), "--output", str(self.output)]
        for flags in (["--role", "val"], ["--role", "test"], ["--role", "calibration"]):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    diagnosis.main(base + flags)
                self.assertEqual(failure.exception.code, 2)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
