"""Verify fixed-head probes keep fitting statistics and updates inside train patients."""

import copy
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from evaluation.feature_probes import run_feature_probes


def _record(patient: str, scan: str, index: int, label: int, role: str) -> dict:
    """Supply canonical synthetic provenance without any MRI files."""
    return {
        "patient_id": patient, "image_id": scan, "slice_index": index,
        "relative_path": f"images/{scan}_{index}.jpeg", "label": label,
        "partition": "development", "role": role,
    }


def _fixture() -> tuple[dict, list[dict], dict, list[dict]]:
    """Make unequal slice counts with four independently labelled training patients."""
    train_rows = [
        _record("train0", "t0a", 0, 0, "train"),
        _record("train0", "t0a", 1, 0, "train"),
        _record("train0", "t0b", 0, 0, "train"),
        _record("train1", "t1", 0, 0, "train"),
        _record("train2", "t2", 0, 1, "train"),
        _record("train3", "t3a", 0, 1, "train"),
        _record("train3", "t3b", 0, 1, "train"),
    ]
    early_rows = [_record("early0", "e0", 0, 0, "early_stop"),
                  _record("early1", "e1", 0, 1, "early_stop")]
    train = torch.tensor([[-3., 7.], [-2., 7.], [-1., 7.], [-1., 7.],
                          [1., 7.], [2., 7.], [4., 7.]])
    early = torch.tensor([[-2.5, 7.], [2.5, 7.]])
    return {"stage1": train}, train_rows, {"stage1": early}, early_rows


class FeatureProbeTests(unittest.TestCase):
    """Use known feature geometry and identity errors to verify diagnostic boundaries."""

    @classmethod
    def setUpClass(cls) -> None:
        """Keep tiny CPU fixtures quick without changing the application's defaults."""
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        """Restore the test process's original CPU threading configuration."""
        torch.set_num_threads(cls.previous_threads)

    def setUp(self) -> None:
        """Give every diagnostic invocation a new disposable output directory."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def _run(self, *, name: str = "probe", epochs: int = 100, inputs: tuple | None = None,
             **kwargs: float) -> dict:
        """Fit just the small synthetic head, with fixed independent patient records."""
        values = _fixture() if inputs is None else inputs
        return run_feature_probes(*values, self.root / name, epochs=epochs, lr=0.05, **kwargs)

    def test_separable_features_grouped_counts_and_exported_unit(self) -> None:
        result = self._run()
        stage = result["stages"]["stage1"]
        self.assertEqual(stage["train"]["accuracy"], 1.0)
        self.assertEqual(stage["early_stop"]["accuracy"], 1.0)
        self.assertEqual(stage["early_stop"]["auroc"], 1.0)
        self.assertEqual(result["populations"]["train"], {"n_patients": 4, "n_scans": 6, "n_slices": 7})
        self.assertEqual(result["populations"]["early_stop"], {"n_patients": 2, "n_scans": 2, "n_slices": 2})
        self.assertEqual(result["fitting"]["train_patient_class_counts"], {"0": 2, "1": 2})
        # Slice counts are 4 NC / 3 AD, but fitting uses equally weighted patients.
        self.assertEqual(result["fitting"]["train_pos_weight"], 1.0)
        self.assertFalse(result["backbone_trained"])
        self.assertEqual(result["evaluation_roles"], ["train", "early_stop"])
        with (self.root / "probe" / "feature_probe_patient_predictions.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 6)
        patient0 = next(row for row in rows if row["patient_id"] == "train0")
        self.assertEqual(patient0["num_scans"], "2")
        self.assertEqual(patient0["num_slices"], "3")
        self.assertEqual(patient0["metric_unit"], "patient_mean_embedding_linear_probe")
        self.assertEqual(json.loads((self.root / "probe" / "feature_probes.json").read_text()), result)

    def test_patient_means_and_training_population_standardization(self) -> None:
        stats = self._run(epochs=1)["stages"]["stage1"]["train_standardization"]
        # The patient means are [-2, -1, 1, 3], independent of slice frequency.
        self.assertEqual(stats["mean"], [0.25, 7.0])
        self.assertAlmostEqual(stats["population_std"][0], math.sqrt(3.6875))
        self.assertEqual(stats["population_std"][1], 0.0)
        self.assertEqual(stats["scale"][1], 1.0)
        self.assertEqual(stats["constant_dimensions"], 1)

    def test_early_shift_and_label_changes_do_not_contaminate_fitting(self) -> None:
        values = _fixture()
        original = self._run(name="original", epochs=30, inputs=values)
        train_features, train_rows, early_features, early_rows = copy.deepcopy(values)
        early_features["stage1"] += 1000.0
        for row in early_rows:
            row["label"] = 1 - row["label"]
        shifted = self._run(name="shifted", epochs=30,
                            inputs=(train_features, train_rows, early_features, early_rows))
        before = original["stages"]["stage1"]
        after = shifted["stages"]["stage1"]
        self.assertEqual(before["train_standardization"], after["train_standardization"])
        self.assertEqual(before["train"], after["train"])
        self.assertEqual(before["final_train_weighted_patient_bce"], after["final_train_weighted_patient_bce"])
        self.assertEqual(original["fitting"], shifted["fitting"])

    def test_each_stage_resets_seed_and_preserves_inputs_and_caller_rng(self) -> None:
        values = _fixture()
        train_features, train_rows, early_features, early_rows = values
        train_features["stage2"] = train_features["stage1"].clone().requires_grad_(True)
        early_features["stage2"] = early_features["stage1"].clone().requires_grad_(True)
        untouched = train_features["stage1"].clone()
        state = torch.random.get_rng_state().clone()
        result = self._run(inputs=values, epochs=25)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))
        self.assertTrue(torch.equal(train_features["stage1"], untouched))
        self.assertIsNone(train_features["stage2"].grad)
        self.assertIsNone(early_features["stage2"].grad)
        self.assertEqual(result["stages"]["stage1"], result["stages"]["stage2"])
        repeated = self._run(name="repeated", inputs=values, epochs=25)
        self.assertEqual(result, repeated)

    def test_reject_overlap_mixed_labels_and_duplicate_identities(self) -> None:
        for problem in ("patient_overlap", "scan_overlap", "path_overlap", "mixed_labels", "duplicate_slice"):
            values = copy.deepcopy(_fixture())
            train_features, train_rows, early_features, early_rows = values
            if problem == "patient_overlap":
                early_rows[0]["patient_id"] = "train0"
            elif problem == "scan_overlap":
                early_rows[0]["image_id"] = "t0a"
            elif problem == "path_overlap":
                early_rows[0]["relative_path"] = train_rows[0]["relative_path"]
            elif problem == "mixed_labels":
                # A different scan with a changed label is forbidden for the patient mean.
                train_rows[2]["label"] = 1
            else:
                train_rows[1]["slice_index"] = train_rows[0]["slice_index"]
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                self._run(name=problem, inputs=values, epochs=1)
            self.assertFalse((self.root / problem / "feature_probes.json").exists())

    def test_reject_protected_roles_bad_shapes_nonfinite_and_single_training_class(self) -> None:
        for problem in ("outer_val", "calibration", "final_test", "bad_shape", "nan", "different_stages",
                        "dimension_mismatch", "single_class", "bad_label"):
            values = copy.deepcopy(_fixture())
            train_features, train_rows, early_features, early_rows = values
            if problem == "outer_val":
                early_rows[0]["role"] = "val"
            elif problem in ("calibration", "final_test"):
                early_rows[0]["partition"] = problem
            elif problem == "bad_shape":
                train_features["stage1"] = train_features["stage1"][:-1]
            elif problem == "nan":
                early_features["stage1"][0, 0] = math.nan
            elif problem == "different_stages":
                early_features["other"] = early_features.pop("stage1")
            elif problem == "dimension_mismatch":
                early_features["stage1"] = early_features["stage1"][:, :1]
            elif problem == "single_class":
                for row in train_rows:
                    row["label"] = 0
            else:
                train_rows[0]["label"] = True
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                self._run(name=problem, inputs=values, epochs=1)

    def test_accept_manifest_text_labels_and_indices(self) -> None:
        values = copy.deepcopy(_fixture())
        for rows in (values[1], values[3]):
            for row in rows:
                row["label"] = str(row["label"])
                row["slice_index"] = str(row["slice_index"])
        self.assertEqual(self._run(inputs=values, epochs=1)["populations"]["train"]["n_patients"], 4)

    def test_reject_invalid_fit_settings_and_existing_outputs(self) -> None:
        values = _fixture()
        for kwargs in ({"epochs": 0}, {"epochs": True}, {"lr": math.nan}, {"lr": 0.0},
                       {"weight_decay": -1.0}, {"weight_decay": math.inf}, {"seed": True}):
            with self.subTest(settings=kwargs), self.assertRaises(ValueError):
                run_feature_probes(*values, self.root / "invalid", **kwargs)
        self._run(epochs=1)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self._run(epochs=1)


if __name__ == "__main__":
    unittest.main()
