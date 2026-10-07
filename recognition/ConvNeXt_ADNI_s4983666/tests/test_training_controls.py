"""Exercise selectable recipes and exact replay using synthetic patient data only."""

import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from engine import prediction, training
from evaluation.reporting import aggregate_patients
from models.registry import create_model, count_parameters
from slurm import preprocessing_train as launcher
from slurm import preprocessing_suite as suite
from utils.training_controls import (ExecutionControls, attach_controls, ad_logits,
    make_criterion, checkpoint_controls, model_input)
import test_adni_splits as split_fixture


class ControlTests(unittest.TestCase):
    """Check recipe precedence, shape contracts and distinct aggregation rules."""

    def setUp(self) -> None:
        """Keep synthetic CPU checks bounded without changing training defaults."""
        torch.set_num_threads(1)

    def test_recipe_overrides_are_order_independent_and_forwarded(self) -> None:
        """Explicit controls win over a recipe before and after allocation forwarding."""
        parser = launcher.parser()
        for tokens in (['--recipe', 'peer_tiny', '--lr=0.001', '--precision', 'fp32'],
                       ['--lr=0.001', '--precision', 'fp32', '--recipe', 'peer_tiny']):
            args = parser.parse_args(tokens)
            launcher.validate(args)
            self.assertEqual((args.model, args.input_channels, args.loss), ('convnext_tiny', 3, 'cross_entropy'))
            self.assertEqual((args.lr, args.precision, args.patience), (.001, 'fp32', 30))
            forwarded = parser.parse_args(launcher.options(args))
            self.assertEqual(vars(args), vars(forwarded))
            command, _ = launcher.commands(args, Path('/tmp/preview'))
            self.assertEqual(command.count('--loss'), 1)
            self.assertEqual(command[command.index('--precision') + 1], 'fp32')
        legacy = parser.parse_args([])
        self.assertEqual((legacy.input_channels, legacy.loss, legacy.precision), (1, 'weighted_bce', 'fp32'))
        self.assertIsNone(legacy.drop_path)

    def test_all_recipes_preview_without_submission(self) -> None:
        """Dry-run lists resolved settings without touching datasets or starting jobs."""
        for recipe in ('custom', 'lite_reference', 'lite_augmented', 'peer_tiny'):
            with self.subTest(recipe=recipe), mock.patch.object(launcher.subprocess, 'run') as process, \
                    contextlib.redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(launcher.main(['--recipe', recipe, '--dry-run']), 0)
                self.assertIn('--recipe ' + recipe, stream.getvalue())
                self.assertIn('patient_aggregation=', stream.getvalue())
                process.assert_not_called()

    def test_ce_margin_matches_unweighted_binary_loss(self) -> None:
        """Verify AD class ordering and probability conversion with nontrivial logits."""
        controls = ExecutionControls(input_channels=3, loss='cross_entropy')
        model = create_model('small_cnn_v1', input_channels=3, output_classes=2)
        attach_controls(model, controls)
        raw = torch.tensor([[2., -1.], [-2., 1.], [0., 0.]])
        labels = torch.tensor([0., 1., 1.])
        margin = ad_logits(model, raw)
        self.assertTrue(torch.allclose(torch.softmax(raw, 1)[:, 1], torch.sigmoid(margin)))
        self.assertTrue(torch.allclose(make_criterion(controls, 4., torch.device('cpu'))(raw, labels.long()),
            nn.BCEWithLogitsLoss()(margin, labels)))
        gray = torch.randn(2, 1, 32, 32)
        repeated = model_input(model, gray)
        self.assertTrue(torch.equal(repeated[:, 0], repeated[:, 2]))
        self.assertEqual(model(repeated).shape, (2, 2))

    def test_tiny_peer_shapes_and_drop_path(self) -> None:
        """Check exact scratch parameter count and explicit stochastic-depth disabling."""
        model = create_model('convnext_tiny_v1', input_channels=3, output_classes=2, drop_path=0.)
        self.assertEqual(count_parameters(model), 27_821_666)
        self.assertTrue(all(block.drop_path.probability == 0. for stage in model.stages for block in stage))
        model.eval()
        with torch.inference_mode():
            self.assertEqual(model(torch.zeros(1, 3, 32, 32)).shape, (1, 2))
        with self.assertRaises(ValueError):
            ExecutionControls(drop_path=math.nan)
        with self.assertRaises(ValueError):
            create_model('small_cnn_v1', drop_path=.1)

    def test_mean_logit_requires_raw_margins_and_can_change_decision(self) -> None:
        """Do not reconstruct saturated logits from a rounded probability export."""
        rows = [{'patient_id': 'p', 'image_id': 'scan', 'label': 1, 'probability': p,
                 'ad_logit': math.log(p / (1 - p))} for p in (.999, .2, .2, .2)]
        probability, _ = aggregate_patients(rows)
        logit, status = aggregate_patients(rows, 'mean_logit')
        self.assertLess(probability[0]['probability'], .5)
        self.assertGreater(logit[0]['probability'], .5)
        self.assertEqual(status['aggregation'], 'sigmoid_mean_all_slice_AD_logit')
        del rows[0]['ad_logit']
        with self.assertRaises(ValueError):
            aggregate_patients(rows, 'mean_logit')

    def test_old_formats_cannot_redefine_execution_controls(self) -> None:
        """Reject reinterpretation of a historical single-channel checkpoint."""
        config = {'checkpoint_format_version': 2, 'execution_controls': ExecutionControls(input_channels=3).to_dict()}
        with self.assertRaises(ValueError):
            checkpoint_controls(config)
        with self.assertRaises(ValueError):
            checkpoint_controls({'checkpoint_format_version': 4})

    def test_suite_accepts_individual_execution_overrides(self) -> None:
        """A plan can change channels, loss, precision and DropPath per case."""
        with tempfile.TemporaryDirectory() as temporary:
            plan = Path(temporary) / 'plan.json'
            plan.write_text(json.dumps({'schema_version': 1, 'name': 'synthetic-controls', 'cases': [
                {'id': 'CE', 'purpose': 'Test execution overrides.', 'overrides':
                 {'input_channels': 3, 'loss': 'cross_entropy', 'precision': 'fp32',
                  'patient_aggregation': 'mean_logit', 'drop_path': 0.}}]}))
            args = suite.parser().parse_args(['--plan', str(plan)])
            resolved = suite.build_plan(args)
            effective = launcher.parser().parse_args(resolved[0]['options'])
            self.assertEqual((effective.loss, effective.input_channels, effective.drop_path), ('cross_entropy', 3, 0.))


class ExecutionReplayTests(unittest.TestCase):
    """Update real model weights and reload them against frozen synthetic manifests."""

    def setUp(self) -> None:
        """Build a small patient-disjoint fixture with two slices per complete scan."""
        self.fixture = split_fixture.ADNISplitTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.prepare()

    def arguments(self, profile: str) -> argparse.Namespace:
        """Use a lightweight CPU model for full training and artifact verification."""
        args = launcher.parser().parse_args(['--model', 'cnn', '--data-root', str(self.fixture.root),
            '--splits-dir', str(self.fixture.out), '--runs-root', str(self.fixture.base / 'runs'),
            '--epochs', '2', '--workers', '0', '--threads', '1', '--batch-size', '16',
            '--image-height', '32', '--image-width', '32', '--preprocessing', profile,
            '--crop-margin', '2', '--input-channels', '3', '--loss', 'cross_entropy',
            '--patient-aggregation', 'mean_logit'])
        launcher.validate(args)
        return args

    def test_format_four_training_replays_channels_ce_and_patient_logits(self) -> None:
        """Validate both original geometry and fitted scan-preprocessing checkpoint paths."""
        for profile in ('none', 'scan_intensity_crop'):
            with self.subTest(profile=profile):
                args = self.arguments(profile)
                output = self.fixture.base / ('execution_' + profile)
                train_args = argparse.Namespace(**vars(args), output=output / 'train', inner_only=True,
                                                 device='cpu', skip_inference_profile=True)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = training.run(train_args)
                checkpoint = torch.load(output / 'train/best.pt', map_location='cpu', weights_only=True)
                self.assertEqual(checkpoint['config']['checkpoint_format_version'], 4)
                self.assertEqual(checkpoint['config']['train_pos_weight'], 1.)
                self.assertEqual(result['epochs_completed'], 2)
                slices = split_fixture.read_csv(output / 'train/early_stop_slice_predictions.csv')
                self.assertTrue(all('ad_logit' in row for row in slices))
                replay_args = argparse.Namespace(checkpoint=output / 'train/best.pt', data_root=self.fixture.root,
                    splits_dir=self.fixture.out, output=output / 'replay', device='cpu', workers=0,
                    threads=1, batch_size=16, role='early_stop', skip_inference_profile=True)
                with contextlib.redirect_stdout(io.StringIO()):
                    prediction.run(replay_args)
                with mock.patch.object(launcher, 'frozen_seals', return_value={'synthetic': 'sealed'}):
                    verified = launcher.verify(output, args)
                self.assertTrue(verified['prediction_replay']['slice']['identical_predictions'])
                self.assertTrue(verified['prediction_replay']['patient']['identical_predictions'])
                self.assertEqual(result['resources']['inference_profile']['input_shape'][0], 3)

    def test_amp_cpu_rejected_before_loading_or_output_creation(self) -> None:
        """An explicit unsupported precision cannot silently run a different experiment."""
        args = self.arguments('none')
        args.precision = 'amp_fp16'
        args.device = 'cpu'
        args.output = self.fixture.base / 'must_not_exist'
        with mock.patch.object(training, 'load_fold') as load, self.assertRaisesRegex(ValueError, 'require CUDA'):
            training.run(args)
        load.assert_not_called()
        self.assertFalse(args.output.exists())


if __name__ == '__main__':
    unittest.main()
