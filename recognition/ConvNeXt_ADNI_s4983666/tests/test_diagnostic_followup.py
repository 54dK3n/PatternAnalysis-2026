"""Test complete inner-only follow-up runs on sealed synthetic scan manifests."""

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

from engine import diagnostic_followup as followup
from engine.diagnosis import select_complete_scans
from utils.artifacts import write_csv, write_json
import test_diagnosis as fixtures
import test_adni_splits as split_fixture


class DiagnosticFollowupTests(unittest.TestCase):
    """Exercise source verification and immutable checkpoint handling end to end."""

    def setUp(self) -> None:
        """Reuse a sealed synthetic cohort, not a mocked data-verification result."""
        self.fixture = fixtures.DiagnosisTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.output = self.fixture.fixture.base / "followup"

    def args(self, **changes: object) -> argparse.Namespace:
        """Bound PCA and shifts to the synthetic cohort's actual patient count."""
        base = self.fixture.args()
        values = {key: getattr(base, key) for key in ("checkpoint", "data_root", "splits_dir", "batch_size",
                                                     "workers", "threads", "device", "seed")}
        values.update(output=self.output, reference_dir=None, routes="all", reference_train_patients=4,
                      max_early_patients=0, max_scans_per_patient=1, max_shift=1,
                      bootstrap_samples=10, probe_epochs=2, probe_lr=.01, probe_weight_decay=.01,
                      pca_components=2)
        values.update(changes)
        return argparse.Namespace(**values)

    def reference(self) -> Path:
        """Construct a completed prior selection bound to the same checkpoint bytes."""
        directory = self.fixture.fixture.base / "previous_diagnosis"
        directory.mkdir()
        checkpoint_hash = hashlib.sha256(self.fixture.checkpoint.read_bytes()).hexdigest()
        write_json(directory / "config.json", {"checkpoint_sha256": checkpoint_hash, "checkpoint_epoch": 1,
            "manifest_sha256": self.fixture.config["manifest_sha256"], "checkpoint_config": self.fixture.config})
        write_json(directory / "summary.json", {"status": "complete", "checkpoint_sha256": checkpoint_hash,
                                               "manifest_sha256": self.fixture.config["manifest_sha256"]})
        for role, limit in (("train", 4), ("early_stop", 0)):
            write_csv(directory / f"{role}_selected_slices.csv", select_complete_scans(self.fixture.rows(role), 2, limit, 1, 3710))
        return directory

    def test_expansion_preserves_exact_anchor_scans_and_adds_all_patients(self) -> None:
        """A larger cohort must not silently replace the replay patients' scans."""
        full = self.fixture.rows("train")
        before = copy.deepcopy(full)
        anchors = select_complete_scans(full, 2, 4, 1, 3710)
        expanded = followup.expanded_training_rows(full, anchors, 2, 1, 3710)
        self.assertEqual(split_fixture.patients(expanded), split_fixture.patients(full))
        for patient in split_fixture.patients(anchors):
            self.assertEqual(split_fixture.paths([r for r in anchors if r["patient_id"] == patient]),
                             split_fixture.paths([r for r in expanded if r["patient_id"] == patient]))
        self.assertEqual(set(Counter(r["image_id"] for r in expanded).values()), {2})
        self.assertEqual(full, before)
        self.assertEqual(expanded, followup.expanded_training_rows(full, anchors, 2, 1, 3710))

    def test_complete_run_replays_reference_and_never_scores_reserved_roles(self) -> None:
        """Observe real loaders while collecting spatial and both projection controls."""
        reference = self.reference()
        expected = {role: split_fixture.paths(self.fixture.rows(role)) for role in ("train", "early_stop")}
        observed = []
        original = followup.make_loader

        def checked(rows: list[dict], *args: object, **kwargs: object) -> object:
            """Require every model input to belong to its actual verified inner role."""
            role = kwargs["role"]
            self.assertIn(role, expected)
            self.assertTrue(split_fixture.paths(rows).issubset(expected[role]))
            observed.append(role)
            return original(rows, *args, **kwargs)

        digest = hashlib.sha256(self.fixture.checkpoint.read_bytes()).hexdigest()
        state = copy.deepcopy(self.fixture.model.state_dict())
        with mock.patch.object(followup, "make_loader", side_effect=checked), \
                mock.patch.object(followup, "create_model", return_value=self.fixture.model), \
                contextlib.redirect_stdout(io.StringIO()):
            result = followup.run(self.args(reference_dir=reference))
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["backbone_trained"])
        self.assertEqual(set(observed), {"train", "early_stop"})
        self.assertEqual(result["checkpoint_sha256"], digest)
        self.assertEqual(hashlib.sha256(self.fixture.checkpoint.read_bytes()).hexdigest(), digest)
        self.assertFalse(list(self.output.rglob("*.pt")))
        for key, value in self.fixture.model.state_dict().items():
            self.assertTrue(torch.equal(value, state[key]))
        for cohort in ("reference", "expanded"):
            self.assertEqual(set(result["probes"][cohort]), {"native", "pca_2"})
            self.assertEqual(result["probes"][cohort]["pca_2"]["stages"]["stage_1"]["feature_dim"], 2)
        self.assertTrue((self.output / "spatial/shift_boundary_curves.png").is_file())
        published = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(published, result)

    def test_reference_rejects_wrong_role_checkpoint_and_incomplete_scan(self) -> None:
        """Reference artifacts are untrusted until identity and scan checks pass."""
        reference = self.reference()
        write_csv(reference / "early_stop_selected_slices.csv", self.fixture.rows("val"))
        with mock.patch.object(followup, "create_model") as constructor, \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            followup.run(self.args(reference_dir=reference))

        constructor.assert_not_called()
        self.assertFalse(self.output.exists())
        write_csv(reference / "early_stop_selected_slices.csv", self.fixture.rows("early_stop")[:-1])
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            followup.run(self.args(reference_dir=reference))
        metadata = json.loads((reference / "config.json").read_text())
        metadata["checkpoint_sha256"] = "0" * 64
        write_json(reference / "config.json", metadata)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Reference"):
            followup.run(self.args(reference_dir=reference))

    def test_reference_accepts_json_equivalent_legacy_container_types(self) -> None:
        """Tuples and integer dictionary keys serialize normally in old configs."""
        self.fixture.config["image_size"] = (32, 32)
        self.fixture.config["historical_counts"] = {0: 10, 1: 10}
        self.fixture.save_checkpoint()
        reference = self.reference()
        with contextlib.redirect_stdout(io.StringIO()):
            result = followup.run(self.args(reference_dir=reference, routes="probes"))
        self.assertEqual(result["status"], "complete")

    def test_full_role_mixed_labels_excluded_even_when_anchor_has_one_scan(self) -> None:
        """A one-scan selection cannot conceal mixed longitudinal patient targets."""
        with contextlib.redirect_stdout(io.StringIO()):
            result = followup.run(self.args(routes="probes"))
        labels = {}
        for row in self.fixture.rows("train"):
            labels.setdefault(row["patient_id"], set()).add(row["label"])
        mixed = {patient for patient, values in labels.items() if len(values) > 1}
        self.assertTrue(mixed)
        self.assertEqual(set(result["probe_patient_exclusions"]["train"]["patient_ids"]), mixed)
        for cohort in ("reference", "expanded"):
            rows = split_fixture.read_csv(self.output / f"probes/{cohort}/native/feature_probe_patient_predictions.csv")
            self.assertTrue(split_fixture.patients(rows).isdisjoint(mixed))

    def test_legacy_cnn_missing_initialization_metadata_stays_compatible(self) -> None:
        """Accept exactly the historical CNN format without inventing provenance."""
        config = {**self.fixture.config, "checkpoint_format_version": 1}
        for key in ("initialization", "pretrained_weights", "augmentation_config"):
            config.pop(key)
        self.fixture.save_checkpoint(config)
        with contextlib.redirect_stdout(io.StringIO()):
            result = followup.run(self.args(routes="probes"))
        self.assertEqual(result["initialization_metadata_status"], "legacy_fields_not_recorded")
        provenance = json.loads((self.output / "config.json").read_text())
        self.assertNotIn("initialization", provenance["checkpoint_config"])

    def test_source_audit_and_manifest_binding_precede_model_construction(self) -> None:
        """Do not permit an altered source or checkpoint seal to enter diagnostics."""
        changed = {**self.fixture.config, "manifest_sha256": "0" * 64}
        self.fixture.save_checkpoint(changed)
        with mock.patch.object(followup, "create_model") as constructor, \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            followup.run(self.args())
        constructor.assert_not_called()
        self.assertFalse(self.output.exists())
        self.fixture.save_checkpoint()
        path = self.fixture.fixture.out / "fold_01/train.csv"
        path.write_text(path.read_text() + "unexpected,row\n")
        with mock.patch.object(followup, "create_model") as constructor, \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            followup.run(self.args())
        constructor.assert_not_called()

    def test_output_protection_includes_original_diagnosis(self) -> None:
        """Neither source data nor a prior experiment can be overwritten."""
        reference = self.reference()
        for output in (reference / "new", self.fixture.run_dir / "new", self.fixture.fixture.root / "new",
                       self.fixture.fixture.out / "new"):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                followup.run(self.args(reference_dir=reference, output=output))
        self.assertFalse(self.output.exists())

    def test_cli_has_no_outer_test_or_calibration_scoring_switch(self) -> None:
        """Reserved role options must be rejected before any execution."""
        args = ["--checkpoint", str(self.fixture.checkpoint), "--data-root", str(self.fixture.fixture.root),
                "--splits-dir", str(self.fixture.fixture.out), "--output", str(self.output)]
        for role in ("val", "calibration", "test"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                followup.main(args + ["--role", role])


if __name__ == "__main__":
    unittest.main()
