"""Check hand-computed confidence metrics and leakage boundaries on synthetic data."""

import argparse
import contextlib
import io
import json
import unittest
from unittest import mock

import torch

from engine import prediction, training
from evaluation.metrics import aggregate_scans
from evaluation.reporting import aggregate_patients, confidence_metrics, prediction_report
from evaluation.resources import profile_inference
from models import count_parameters, create_model
from utils.evaluation_artifacts import select_failure_cases
from test_metrics import slice_record
import test_adni_splits as split_fixture


class ConfidenceReportingTests(unittest.TestCase):
    """Independent calculations expose tie handling and empty accepted sets."""

    def test_ece_brier_rejection_and_confidence_ties(self):
        report = confidence_metrics([0, 1, 0, 1], [0.1, 0.8, 0.8, 0.4], bins=2, reject_threshold=0.8)
        self.assertAlmostEqual(report["ece_predicted_class"], 0.275)
        self.assertAlmostEqual(report["brier_ad"], 0.2625)
        self.assertEqual([p["accepted"] for p in report["risk_coverage"]], [0, 1, 3, 4])
        self.assertAlmostEqual(report["risk_coverage"][-1]["risk"], 0.5)
        rule = report["fixed_rejection"]
        self.assertEqual(rule["accepted"], 3)
        self.assertEqual(rule["referred"], 1)
        self.assertAlmostEqual(rule["accepted_accuracy"], 2 / 3)
        self.assertEqual(rule["full_coverage_accuracy"], 0.5)
        self.assertEqual(report["confidence_histogram"][1], {"bin": 1, "correct": 2, "incorrect": 2})
        json.dumps(report, allow_nan=False)

    def test_empty_acceptance_and_endpoint_probabilities(self):
        empty = confidence_metrics([0, 1], [0.5, 0.5], reject_threshold=1)["fixed_rejection"]
        self.assertEqual(empty["coverage"], 0)
        self.assertIsNone(empty["accepted_accuracy"])
        exact = confidence_metrics([0, 1], [0, 1], bins=15, reject_threshold=1)
        self.assertEqual(exact["ece_predicted_class"], 0)
        self.assertEqual(exact["brier_ad"], 0)
        self.assertEqual(exact["reliability_bins"][-1]["count"], 2)
        for kwargs in ({"bins": 0}, {"bins": True}, {"reject_threshold": 0.49}, {"reject_threshold": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                confidence_metrics([0, 1], [0.1, 0.9], **kwargs)

    def test_patient_average_and_mixed_diagnoses_are_not_silently_relabelled(self):
        rows = [slice_record("a", 1, 0.1, 0), slice_record("a", 2, 0.3, 0),
                slice_record("b", 1, 0.9, 0), slice_record("b", 2, 0.7, 0)]
        patients, status = aggregate_patients(rows)
        self.assertEqual(status["status"], "available")
        self.assertEqual(patients[0]["num_scans"], 2)
        self.assertAlmostEqual(patients[0]["probability"], 0.5)
        scans = aggregate_scans(rows, expected_slices=2)
        report, _ = prediction_report(rows, scans)
        self.assertEqual(report["primary_evaluation_unit"], "slice")
        self.assertEqual(report["final_test_target"]["status"], "not_assessed_development_results_only")
        rows[2]["label"] = rows[3]["label"] = 1
        report, patients = prediction_report(rows, aggregate_scans(rows, expected_slices=2))
        self.assertEqual(patients, [])
        self.assertIsNone(report["patient_metrics"])
        self.assertEqual(report["patient_aggregation"]["mixed_diagnosis_patients"], 1)
        self.assertNotIn("patient", report["confidence"])

    def test_failure_examples_include_both_types_and_never_invent_errors(self):
        rows = [slice_record("a", 1, 0.99, 0, "a"), slice_record("b", 1, 0.01, 1, "b"),
                slice_record("c", 1, 0.95, 1, "c")]
        failures = select_failure_cases(rows)
        self.assertEqual({r["error_type"] for r in failures}, {"FP", "FN"})
        self.assertEqual(len(failures), 2)
        self.assertEqual(select_failure_cases([rows[-1]]), [])


class ResourceProfileTests(unittest.TestCase):
    """Distinguish throughput, latency, unmeasured resources and smoke protocols."""

    def test_cpu_timing_and_training_mode_restoration(self):
        class Toy(torch.nn.Module):
            def forward(self, images):
                return images.mean(dim=(1, 2, 3))
        model = Toy().train()
        with mock.patch("evaluation.resources.time.perf_counter", side_effect=[0, 0.002, 1, 1.004]):
            report = profile_inference(model, (32, 32), torch.device("cpu"),
                                       batch_sizes=(2,), warmup=1, repeats=2, on_cpu=True)
        measured = report["batches"][0]
        self.assertAlmostEqual(measured["mean_batch_ms"], 3)
        self.assertAlmostEqual(measured["std_batch_ms"], 1)
        self.assertAlmostEqual(measured["mean_ms_per_slice"], 1.5)
        self.assertIsNone(measured["peak_cuda_allocated_mib"])
        self.assertFalse(report["publication_protocol_complete"])
        self.assertTrue(model.training)
        report = profile_inference(model, (32, 32), torch.device("cpu"))
        self.assertEqual(report["status"], "not_measured_cpu")
        self.assertEqual(report["batches"], [])

    def test_cuda_oom_is_reported_without_invented_latency(self):
        class Limited(torch.nn.Module):
            def forward(self, images):
                if images.shape[0] == 64:
                    raise torch.cuda.OutOfMemoryError("synthetic resource limit")
                return images.mean(dim=(1, 2, 3))
        original_zeros = torch.zeros
        def cpu_storage(*shape, **kwargs):
            return original_zeros(*shape, device="cpu")
        with mock.patch("evaluation.resources.torch.zeros", side_effect=cpu_storage), \
                mock.patch("evaluation.resources.sync_device") as sync, \
                mock.patch("torch.cuda.reset_peak_memory_stats"), \
                mock.patch("torch.cuda.max_memory_allocated", return_value=2**20), \
                mock.patch("torch.cuda.empty_cache"):
            report = profile_inference(Limited(), (32, 32), torch.device("cuda"), warmup=1, repeats=2)
        self.assertEqual(report["status"], "partial_resource_limit")
        self.assertEqual(report["batches"][0]["peak_cuda_allocated_mib"], 1)
        self.assertEqual(report["batches"][1], {"batch_size": 64, "status": "cuda_out_of_memory"})
        self.assertFalse(report["publication_protocol_complete"])
        self.assertGreaterEqual(sync.call_count, 5)


class CourseTrainingIntegrationTests(unittest.TestCase):
    """Exercise Lite training, inner-only inference and logged metric reproduction."""

    def test_online_metrics_do_not_change_weights_or_random_stream(self):
        from models import SmallCNN
        torch.set_num_threads(1)
        torch.manual_seed(123)
        first, second = SmallCNN(), SmallCNN()
        second.load_state_dict(first.state_dict())
        batch = {"image": torch.linspace(-1, 1, 4 * 32 * 32).reshape(4, 1, 32, 32),
                 "label": torch.tensor([0., 1., 0., 1.])}
        states, losses, rng = [], [], []
        metrics = {}
        for model, sink in ((first, None), (second, metrics)):
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            torch.manual_seed(456)
            losses.append(training.train_epoch(model, [batch], optimizer,
                                                torch.nn.BCEWithLogitsLoss(), torch.device("cpu"), sink))
            states.append(model.state_dict())
            rng.append(torch.get_rng_state())
        self.assertEqual(losses[0], losses[1])
        self.assertTrue(torch.equal(rng[0], rng[1]))
        self.assertTrue(all(torch.equal(states[0][key], states[1][key]) for key in states[0]))
        self.assertEqual(metrics["n_samples"], 4)

    def test_inner_only_run_does_not_construct_or_score_outer_validation(self):
        fixture = split_fixture.ADNISplitTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.prepare()
        args = argparse.Namespace(
            data_root=fixture.root, splits_dir=fixture.out, output=fixture.base / "inner",
            fold=1, epochs=1, patience=1, batch_size=16, workers=0, threads=1,
            lr=0.0001, weight_decay=0.05, min_delta=0, seed=3710,
            image_height=32, image_width=32, device="cpu", model="convnext_lite", inner_only=True)
        real_loader = training.make_loader
        roles = []
        def observed_loader(*values, **kwargs):
            roles.append(kwargs["role"])
            self.assertNotIn(kwargs["role"], ("val", "calibration", "test"))
            return real_loader(*values, **kwargs)
        with mock.patch("engine.training.make_loader", side_effect=observed_loader), contextlib.redirect_stdout(io.StringIO()):
            result = training.run(args)
        self.assertEqual(roles, ["train", "early_stop"])
        self.assertEqual(result["evaluation_role"], "development_inner_early_stop")
        self.assertTrue(result["evaluation_reuses_checkpoint_selection_patients"])
        self.assertFalse((args.output / "val_slice_predictions.csv").exists())
        self.assertTrue((args.output / "early_stop_confidence.png").exists())
        self.assertEqual(result["resources"]["inference_profile"]["status"], "not_measured_cpu")
        rows = split_fixture.read_csv(args.output / "early_stop_slice_predictions.csv")
        independently_computed = confidence_metrics([int(r["label"]) for r in rows],
                                                   [float(r["probability"]) for r in rows])
        self.assertEqual(result["coursework_report"]["confidence"]["slice"], independently_computed)
        history = json.loads((args.output / "epoch_metrics.json").read_text())
        self.assertEqual(history[0]["early_stop_slice"]["n_samples"], len(rows))
        config = json.loads((args.output / "config.json").read_text())
        self.assertIsNone(config["pretrained_weights"])
        self.assertEqual(config["model_name"], "convnext_lite_v1")
        self.assertEqual(config["epochs_limit"], 1)
        self.assertLess(count_parameters(create_model("convnext_lite_v1")), 27_817_825)
        pred_args = argparse.Namespace(checkpoint=args.output / "best.pt", data_root=fixture.root,
                                       splits_dir=fixture.out, output=fixture.base / "prediction",
                                       batch_size=16, workers=0, threads=1, device="cpu")
        with mock.patch("engine.prediction.make_loader", side_effect=observed_loader), contextlib.redirect_stdout(io.StringIO()):
            prediction.run(pred_args)
        reproduced = json.loads((pred_args.output / "metrics.json").read_text())
        self.assertEqual(reproduced["coursework_report"], result["coursework_report"])
        self.assertEqual(reproduced["metrics"]["slice"], result["metrics"]["slice"])
        self.assertEqual(roles, ["train", "early_stop", "early_stop"])


if __name__ == "__main__":
    unittest.main()
