"""Run full ConvNeXt training and checkpoint inference on synthetic patients.

Small synthetic images keep CPU checks practical. These tests validate the
training protocol and serialization, not ADNI classification performance.
"""

import argparse
import contextlib
import io
import json
import unittest
from unittest import mock

import torch

from engine import prediction as predict
from engine import training as train
from models import create_model
import test_adni_splits as split_fixture


class ConvNeXtTrainingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = split_fixture.ADNISplitTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.prepare()
        self.output = self.fixture.base / "convnext_run"
        self.args = argparse.Namespace(
            data_root=self.fixture.root, splits_dir=self.fixture.out,
            output=self.output, model="convnext_tiny", fold=1, epochs=4,
            patience=1, batch_size=16, workers=0, threads=1,
            lr=1e-4, weight_decay=0.05, min_delta=0.0, seed=3710,
            image_height=32, image_width=32, device="cpu",
        )

    def test_real_training_preserves_patient_boundaries_and_reproduces_checkpoint(self):
        """Exercise full-model updates, selection, and independently reloaded inference."""
        rows = {
            name: split_fixture.read_csv(self.fixture.out / f"fold_01/{name}.csv")
            for name in ("train", "early_stop", "val")
        }
        path_sets = {name: split_fixture.paths(value) for name, value in rows.items()}
        protected = set()
        for name in ("calibration", "test"):
            protected.update(split_fixture.patients(
                split_fixture.read_csv(self.fixture.out / f"{name}.csv")
            ))
        for left, right in (("train", "early_stop"), ("train", "val"), ("early_stop", "val")):
            self.assertTrue(split_fixture.patients(rows[left]).isdisjoint(
                split_fixture.patients(rows[right])
            ))

        loader_calls, evaluation_calls = [], []
        real_make_loader, real_evaluate = train.make_loader, train.evaluate

        def observed_loader(manifest_rows, *args, **kwargs):
            actual_paths = split_fixture.paths(manifest_rows)
            names = [name for name, expected in path_sets.items() if actual_paths == expected]
            self.assertEqual(len(names), 1)
            name = names[0]
            self.assertTrue(split_fixture.patients(manifest_rows).isdisjoint(protected))
            if name == "val":
                self.assertEqual(evaluation_calls, ["early_stop", "early_stop"])
                self.assertTrue((self.output / "best.pt").is_file())
            loader_calls.append(name)
            return real_make_loader(manifest_rows, *args, **kwargs)

        def observed_evaluate(model, loader, device, expected_slices):
            actual_paths = split_fixture.paths(loader.dataset.rows)
            if actual_paths == path_sets["early_stop"]:
                self.assertNotIn("val", evaluation_calls)
                evaluation_calls.append("early_stop")
            else:
                self.assertEqual(actual_paths, path_sets["val"])
                selected = torch.load(self.output / "best.pt", map_location="cpu", weights_only=True)
                self.assertEqual(selected["epoch"], 1)
                for key, tensor in model.state_dict().items():
                    self.assertTrue(torch.equal(tensor.cpu(), selected["model_state"][key]), key)
                evaluation_calls.append("val")
            scores, slices, scans = real_evaluate(model, loader, device, expected_slices)
            if actual_paths == path_sets["early_stop"]:
                # Control selection only; all batches still execute real training
                # and inference through the complete ConvNeXt-Tiny network.
                scores["scan"]["log_loss"] = 0.6 if len(evaluation_calls) == 1 else 0.8
            return scores, slices, scans

        with mock.patch("engine.training.make_loader", side_effect=observed_loader), \
                mock.patch("engine.training.evaluate", side_effect=observed_evaluate), \
                contextlib.redirect_stdout(io.StringIO()):
            result = train.run(self.args)

        self.assertEqual(loader_calls, ["train", "early_stop", "val"])
        self.assertEqual(evaluation_calls, ["early_stop", "early_stop", "val"])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(result["epochs_completed"], 2)
        self.assertEqual(result["evaluation_role"], "development_outer_validation")
        self.assertIsNone(result["resources"]["peak_cuda_allocated_mib"])
        self.assertTrue((self.output / "learning_curves.png").is_file())

        checkpoint = torch.load(self.output / "best.pt", map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        self.assertEqual(config["model_name"], "convnext_tiny_v1")
        self.assertEqual(config["initialization"], "random")
        self.assertIsNone(config["pretrained_weights"])
        self.assertEqual(config["normalization"], "(grayscale_uint8 / 255 - 0.5) / 0.5")
        self.assertEqual(config["augmentation"], "none")
        self.assertEqual(config["calibration"], "not_fitted")
        self.assertEqual(config["image_size"], [32, 32])
        counts = {label: sum(row["label"] == label for row in rows["train"]) for label in ("0", "1")}
        self.assertEqual(config["train_slice_class_counts"], counts)
        self.assertEqual(config["train_pos_weight"], counts["0"] / counts["1"])
        self.assertTrue(all(torch.isfinite(tensor).all() for tensor in checkpoint["model_state"].values()))

        torch.manual_seed(config["seed"])
        fresh = create_model("convnext_tiny_v1")
        parameters = sum(parameter.numel() for parameter in fresh.parameters() if parameter.requires_grad)
        self.assertEqual(parameters, 27_817_825)
        self.assertEqual(result["resources"]["trainable_parameters"], parameters)
        self.assertTrue(any(
            not torch.equal(tensor, checkpoint["model_state"][name])
            for name, tensor in fresh.state_dict().items()
        ))
        del fresh, checkpoint

        scan_rows = split_fixture.read_csv(self.output / "val_scan_predictions.csv")
        slice_rows = split_fixture.read_csv(self.output / "val_slice_predictions.csv")
        self.assertEqual(len(scan_rows), len({row["image_id"] for row in rows["val"]}))
        self.assertEqual(split_fixture.patients(scan_rows), split_fixture.patients(rows["val"]))
        self.assertEqual(split_fixture.paths(slice_rows), path_sets["val"])
        self.assertTrue(all(row["num_slices"] == "2" for row in scan_rows))

        prediction_dir = self.fixture.base / "convnext_prediction"
        prediction_args = argparse.Namespace(
            checkpoint=self.output / "best.pt", data_root=self.fixture.root,
            splits_dir=self.fixture.out, output=prediction_dir,
            batch_size=16, workers=0, threads=1, device="cpu",
        )
        prediction_loaders = []

        def prediction_loader(manifest_rows, *args, **kwargs):
            self.assertEqual(split_fixture.paths(manifest_rows), path_sets["val"])
            self.assertTrue(split_fixture.patients(manifest_rows).isdisjoint(protected))
            prediction_loaders.append("val")
            return real_make_loader(manifest_rows, *args, **kwargs)

        with mock.patch("engine.prediction.make_loader", side_effect=prediction_loader), \
                contextlib.redirect_stdout(io.StringIO()):
            predict.run(prediction_args)
        reproduced = json.loads((prediction_dir / "metrics.json").read_text())
        self.assertEqual(prediction_loaders, ["val"])
        self.assertEqual(reproduced["metrics"]["scan"], result["metrics"]["scan"])
        self.assertEqual(reproduced["metrics"]["slice"], result["metrics"]["slice"])
        self.assertEqual(split_fixture.read_csv(prediction_dir / "scan_predictions.csv"), scan_rows)
        self.assertEqual(split_fixture.read_csv(prediction_dir / "slice_predictions.csv"), slice_rows)

        # Format 1 used the same versioned model and fixed preprocessing, but
        # did not contain the new, explicit training-augmentation metadata.
        legacy = torch.load(self.output / "best.pt", map_location="cpu", weights_only=True)
        legacy["config"]["checkpoint_format_version"] = 1
        del legacy["config"]["augmentation_config"]
        legacy_path = self.fixture.base / "legacy_convnext.pt"
        torch.save(legacy, legacy_path)
        del legacy
        prediction_args.checkpoint = legacy_path
        prediction_args.output = self.fixture.base / "legacy_prediction"
        with contextlib.redirect_stdout(io.StringIO()):
            predict.run(prediction_args)
        self.assertEqual(split_fixture.read_csv(prediction_args.output / "scan_predictions.csv"), scan_rows)

    def test_invalid_model_or_resolution_stops_before_data_loading(self):
        for model, height, width in (("unknown", 32, 32), ("convnext_tiny", 31, 32),
                                     ("convnext_tiny", 32, 31)):
            with self.subTest(model=model, height=height, width=width):
                self.args.model, self.args.image_height, self.args.image_width = model, height, width
                with mock.patch("engine.training.load_fold") as load_fold, self.assertRaises(ValueError):
                    train.run(self.args)
                load_fold.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_modified_manifest_stops_before_model_construction(self):
        path = self.fixture.out / "fold_01/train.csv"
        path.write_text(path.read_text() + "unexpected,row\n")
        with mock.patch("engine.training.create_model") as create, \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "modified"):
            train.run(self.args)
        create.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_unsupported_checkpoint_contract_stops_before_data_loading(self):
        valid_config = {
            "checkpoint_format_version": 1, "model_name": "convnext_tiny_v1",
            "normalization": "(grayscale_uint8 / 255 - 0.5) / 0.5",
            "aggregation": "mean_slice_AD_probability", "threshold": 0.5,
            "augmentation": "none", "calibration": "not_fitted", "image_size": [32, 32],
        }
        checkpoint = self.fixture.base / "invalid_checkpoint.pt"
        args = argparse.Namespace(
            checkpoint=checkpoint, data_root=self.fixture.root,
            splits_dir=self.fixture.out, output=self.output,
            batch_size=16, workers=0, threads=1, device="cpu",
        )
        for field, value in (("model_name", "unknown"), ("image_size", [16, 32]),
                             ("normalization", "unknown"), ("augmentation", "unknown"),
                             ("calibration", "fitted")):
            with self.subTest(field=field):
                config = dict(valid_config)
                config[field] = value
                torch.save({"config": config, "model_state": {}}, checkpoint)
                with mock.patch("engine.prediction.load_fold") as load_fold, self.assertRaises(ValueError):
                    predict.run(args)
                load_fold.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_weights_from_another_architecture_are_rejected_before_prediction(self):
        with contextlib.redirect_stdout(io.StringIO()):
            data = train.load_fold(self.fixture.root, self.fixture.out, 1)
        config = {
            "checkpoint_format_version": 1, "model_name": "convnext_tiny_v1",
            "normalization": "(grayscale_uint8 / 255 - 0.5) / 0.5",
            "aggregation": "mean_slice_AD_probability", "threshold": 0.5,
            "augmentation": "none", "calibration": "not_fitted", "image_size": [32, 32],
            "fold": 1, "seed": 3711, "manifest_sha256": data["manifest_sha256"],
            "expected_slices": data["report"]["config"]["expected_slices"],
        }
        checkpoint = self.fixture.base / "wrong_architecture.pt"
        torch.save({"config": config, "model_state": create_model("small_cnn_v1").state_dict()}, checkpoint)
        args = argparse.Namespace(
            checkpoint=checkpoint, data_root=self.fixture.root,
            splits_dir=self.fixture.out, output=self.output,
            batch_size=16, workers=0, threads=1, device="cpu",
        )
        with mock.patch("engine.prediction.make_loader") as loader, \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "state_dict"):
            predict.run(args)
        loader.assert_not_called()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
