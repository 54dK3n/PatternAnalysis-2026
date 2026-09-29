"""Exercise actual augmented training, role boundaries, and checkpoint prediction."""

import argparse
import contextlib
import io
import json
import unittest
from unittest import mock

import torch

from dataset.augmentation import make_augmentation
from engine import prediction, training
import test_adni_splits as split_fixture


class AugmentedTrainingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = split_fixture.ADNISplitTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.prepare()
        self.args = argparse.Namespace(
            data_root=self.fixture.root, splits_dir=self.fixture.out,
            output=self.fixture.base / "augmented_run", model="cnn", fold=1,
            epochs=2, patience=5, min_delta=0.0001, batch_size=16, workers=0,
            threads=1, lr=0.001, weight_decay=0.0001, seed=3710,
            image_height=32, image_width=32, device="cpu", augmentation="light",
            rotation_degrees=5.0, translation_fraction=0.03,
        )

    def test_augmented_runs_repeat_and_prediction_uses_fixed_validation_processing(self):
        manifests = {
            role: split_fixture.read_csv(self.fixture.out / f"fold_01/{role}.csv")
            for role in ("train", "early_stop", "val")
        }
        protected = set()
        for role in ("calibration", "test"):
            protected.update(split_fixture.patients(
                split_fixture.read_csv(self.fixture.out / f"{role}.csv")))
        real_loader = training.make_loader
        seen = []

        def observed_loader(rows, *args, **kwargs):
            role = kwargs["role"]
            self.assertEqual(split_fixture.paths(rows), split_fixture.paths(manifests[role]))
            self.assertTrue(split_fixture.patients(rows).isdisjoint(protected))
            loader = real_loader(rows, *args, **kwargs)
            self.assertEqual(loader.dataset.augmentation.name, "light" if role == "train" else "none")
            seen.append(role)
            return loader

        first_output = self.args.output
        with mock.patch("engine.training.make_loader", side_effect=observed_loader), \
                contextlib.redirect_stdout(io.StringIO()):
            first = training.run(self.args)
        self.assertEqual(seen, ["train", "early_stop", "val"])
        self.assertEqual(first["model_name"], "small_cnn_v1")
        self.assertEqual(first["augmentation"], "light")
        saved = torch.load(first_output / "best.pt", map_location="cpu", weights_only=True)
        self.assertEqual(saved["config"]["checkpoint_format_version"], 2)
        self.assertEqual(saved["config"]["augmentation_config"], make_augmentation("light").to_dict())

        self.args.output = self.fixture.base / "repeated_run"
        with contextlib.redirect_stdout(io.StringIO()):
            repeated = training.run(self.args)
        for level in ("scan", "slice"):
            self.assertEqual(first["metrics"][level], repeated["metrics"][level])
            self.assertEqual((first_output / f"val_{level}_predictions.csv").read_bytes(),
                             (self.args.output / f"val_{level}_predictions.csv").read_bytes())
        second = torch.load(self.args.output / "best.pt", map_location="cpu", weights_only=True)
        self.assertEqual(saved["epoch"], second["epoch"])
        self.assertTrue(all(torch.equal(value, second["model_state"][key])
                            for key, value in saved["model_state"].items()))

        prediction_args = argparse.Namespace(
            checkpoint=first_output / "best.pt", data_root=self.fixture.root,
            splits_dir=self.fixture.out, output=self.fixture.base / "prediction",
            batch_size=16, workers=0, threads=1, device="cpu",
        )
        seen.clear()
        with mock.patch("engine.prediction.make_loader", side_effect=observed_loader), \
                contextlib.redirect_stdout(io.StringIO()):
            prediction.run(prediction_args)
        self.assertEqual(seen, ["val"])
        reproduced = json.loads((prediction_args.output / "metrics.json").read_text())
        for level in ("scan", "slice"):
            self.assertEqual(first["metrics"][level], reproduced["metrics"][level])
            self.assertEqual((first_output / f"val_{level}_predictions.csv").read_bytes(),
                             (prediction_args.output / f"{level}_predictions.csv").read_bytes())

        # A checkpoint may record training transforms, but may not redefine
        # the augmentation algorithm or apply it at prediction time.
        for field, value in (("scope", "all_data"), ("algorithm", "unknown"), ("fill", 255)):
            with self.subTest(field=field):
                broken = dict(saved["config"])
                broken["augmentation_config"] = dict(broken["augmentation_config"], **{field: value})
                bad_path = self.fixture.base / "bad_augmentation.pt"
                torch.save({**saved, "config": broken}, bad_path)
                prediction_args.checkpoint = bad_path
                prediction_args.output = self.fixture.base / "rejected_prediction"
                with mock.patch("engine.prediction.load_fold") as load_fold, self.assertRaises(ValueError):
                    prediction.run(prediction_args)
                load_fold.assert_not_called()
                self.assertFalse(prediction_args.output.exists())

    def test_invalid_augmentation_stops_before_source_audit(self):
        for field, value in (("augmentation", "unknown"), ("rotation_degrees", float("nan")),
                             ("translation_fraction", 0.5)):
            with self.subTest(field=field):
                args = argparse.Namespace(**vars(self.args))
                setattr(args, field, value)
                with mock.patch("engine.training.load_fold") as load_fold, self.assertRaises(ValueError):
                    training.run(args)
                load_fold.assert_not_called()
                self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
