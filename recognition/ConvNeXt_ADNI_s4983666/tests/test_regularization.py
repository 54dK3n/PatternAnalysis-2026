"""Verify loss mathematics, RNG compatibility and synthetic checkpoint replay."""

import argparse
import contextlib
import io
import math
from pathlib import Path
import unittest
from unittest import mock

import torch
from torch import nn
from engine import training, prediction
from slurm import preprocessing_train as launcher
from utils.training_controls import ExecutionControls, make_criterion, mixup_batch, attach_controls
import test_adni_splits as split_fixture


class RegularizationTests(unittest.TestCase):
    """Test numerical behavior instead of mirroring flags alone."""

    def setUp(self) -> None:
        """Bound synthetic CPU computation."""
        torch.set_num_threads(1)

    def test_closed_versioning_and_invalid_controls(self) -> None:
        """Legacy dictionaries remain exact; incomplete new schemas cannot load."""
        old = ExecutionControls().to_dict()
        self.assertEqual(old['algorithm'], 'scratch_execution_v1')
        self.assertNotIn('label_smoothing', old)
        self.assertEqual(ExecutionControls.from_dict(old), ExecutionControls())
        new = ExecutionControls(label_smoothing=.05, mixup_alpha=.1).to_dict()
        self.assertEqual(new['algorithm'], 'scratch_execution_v2')
        self.assertEqual(ExecutionControls.from_dict(new).to_dict(), new)
        del new['mixup_alpha']
        with self.assertRaises(ValueError):
            ExecutionControls.from_dict(new)
        for field, value in [('label_smoothing', -1.), ('label_smoothing', 1.),
                             ('mixup_alpha', -1.), ('mixup_alpha', 2.), ('mixup_alpha', math.nan)]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                ExecutionControls(**{field: value})

    def test_smoothing_binary_and_ce_match_uniform_target_formula(self) -> None:
        """Both heads use epsilon/2 for the other binary class, with real gradients."""
        margin = torch.tensor([-2., .7, 1.5], requires_grad=True)
        labels = torch.tensor([0., 1., 0.])
        raw = torch.stack((torch.zeros_like(margin), margin), dim=1)
        bce = make_criterion(ExecutionControls(loss='bce', label_smoothing=.1), 7., torch.device('cpu'))
        ce = make_criterion(ExecutionControls(loss='cross_entropy', label_smoothing=.1), 7., torch.device('cpu'))
        reference = nn.BCEWithLogitsLoss()(margin, labels * .9 + .05)
        self.assertTrue(torch.allclose(bce(margin, labels), reference))
        self.assertTrue(torch.allclose(ce(raw, labels.long()), reference))
        weighted = make_criterion(ExecutionControls(label_smoothing=.1), 2., torch.device('cpu'))
        expected = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.))(margin, labels * .9 + .05)
        self.assertTrue(torch.allclose(weighted(margin, labels), expected))
        bce(margin, labels).backward()
        self.assertTrue(torch.isfinite(margin.grad).all())

    def test_mixup_reproducible_linear_pairing_and_disabled_rng(self) -> None:
        """Mix labels and images with the same permutation; disabled draws consume no RNG."""
        images = torch.arange(8.).reshape(4, 1, 1, 2)
        labels = torch.arange(4.)
        torch.manual_seed(123)
        state = torch.get_rng_state().clone()
        disabled = mixup_batch(images, labels, 0.)
        self.assertIs(disabled[0], images)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        torch.manual_seed(123)
        mixed, a, b, coefficient = mixup_batch(images, labels, .2)
        expected = coefficient * images + (1 - coefficient) * images[b.long()]
        self.assertTrue(torch.equal(mixed, expected))
        torch.manual_seed(123)
        again = mixup_batch(images, labels, .2)
        self.assertTrue(torch.equal(mixed, again[0]))
        self.assertEqual(coefficient, again[3])
        singleton = mixup_batch(images[:1], labels[:1], .2)
        self.assertEqual(singleton[3], 1.)

    def test_mixup_loss_and_gradient_equal_soft_binary_target(self) -> None:
        """Mixing two weighted smoothed BCE losses is linear in fractional targets."""
        logits = torch.tensor([-.5, 2., 1.], requires_grad=True)
        a, b, coefficient = torch.tensor([0., 1., 0.]), torch.tensor([1., 0., 0.]), .3
        criterion = make_criterion(ExecutionControls(label_smoothing=.05), 2., torch.device('cpu'))
        loss = coefficient * criterion(logits, a) + (1 - coefficient) * criterion(logits, b)
        expected = criterion(logits, coefficient * a + (1 - coefficient) * b)
        self.assertTrue(torch.allclose(loss, expected))
        self.assertTrue(torch.allclose(torch.autograd.grad(loss, logits)[0], torch.autograd.grad(expected, logits)[0]))

    def test_ce_mixup_matches_smoothed_fractional_target_gradient(self) -> None:
        """CE mixtures retain fractional labels and the analytically expected gradient."""
        raw = torch.tensor([[2., -.4], [-.1, 1.]], requires_grad=True)
        a, b = torch.tensor([0, 1]), torch.tensor([1, 0])
        criterion = make_criterion(ExecutionControls(loss='cross_entropy', label_smoothing=.1), 1., torch.device('cpu'))
        loss = .3 * criterion(raw, a) + .7 * criterion(raw, b)
        targets = .3 * torch.nn.functional.one_hot(a, 2) + .7 * torch.nn.functional.one_hot(b, 2)
        targets = targets * .9 + .05
        expected = -(targets * torch.log_softmax(raw, 1)).sum(1).mean()
        self.assertTrue(torch.allclose(loss, expected))
        self.assertTrue(torch.allclose(torch.autograd.grad(loss, raw)[0], torch.autograd.grad(expected, raw)[0]))

    def test_mixup_train_metrics_use_unmixed_inputs_and_restore_mode(self) -> None:
        """Do not label a mixed image as either constituent in ordinary accuracy."""
        class RecordingHead(nn.Module):
            """Keep a real parameter and track inputs/modes used for diagnostics."""
            def __init__(self) -> None:
                """Initialize a separable synthetic binary classifier."""
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(1.))
                self.observed = []
            def forward(self, images: torch.Tensor) -> torch.Tensor:
                """Capture mode without changing the input."""
                self.observed.append((self.training, images.detach().clone()))
                return images.flatten(1).mean(1) * self.scale
        images = torch.tensor([-2., 2.]).reshape(2, 1, 1, 1)
        labels = torch.tensor([0., 1.])
        model = RecordingHead()
        attach_controls(model, ExecutionControls(mixup_alpha=.1))
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        metrics = {}
        mixed = torch.zeros_like(images)
        with mock.patch.object(training, 'mixup_batch', return_value=(mixed, labels, labels.flip(0), .5)):
            training.train_epoch(model, [{'image': images, 'label': labels}], optimizer,
                                 nn.BCEWithLogitsLoss(), torch.device('cpu'), metrics)
        self.assertEqual(metrics['accuracy'], 1.)
        self.assertEqual([mode for mode, _ in model.observed], [True, False])
        self.assertTrue(torch.equal(model.observed[1][1], images))
        self.assertTrue(model.training)

    def test_forwarding_preserves_regularization_switches(self) -> None:
        """Preview/submission can reconstruct exactly the requested training controls."""
        args = launcher.parser().parse_args(['--mixup-alpha', '.2', '--label-smoothing', '.05'])
        launcher.validate(args)
        forward = launcher.parser().parse_args(launcher.options(args))
        self.assertEqual(vars(args), vars(forward))
        train, replay = launcher.commands(args, Path('/tmp/regularization-preview'))
        self.assertIn('--mixup-alpha', train)
        self.assertNotIn('--mixup-alpha', replay)


class RegularizationReplayTests(unittest.TestCase):
    """Train real synthetic weights and reproduce unmixed development predictions."""

    def setUp(self) -> None:
        """Reuse the established patient-disjoint synthetic manifests."""
        self.fixture = split_fixture.ADNISplitTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.prepare()

    def test_bce_and_ce_smoothed_mixed_checkpoints_replay(self) -> None:
        """Saved controls reproduce exact predictions; targets are mixed only in train."""
        for index, (loss, alpha, epsilon) in enumerate([('weighted_bce', 0., .1),
                ('weighted_bce', .2, .05), ('cross_entropy', 0., .1), ('cross_entropy', .2, .05)]):
            with self.subTest(loss=loss, alpha=alpha):
                args = launcher.parser().parse_args(['--model', 'cnn', '--data-root', str(self.fixture.root),
                    '--splits-dir', str(self.fixture.out), '--runs-root', str(self.fixture.base / 'runs'),
                    '--epochs', '1', '--workers', '0', '--threads', '1', '--batch-size', '16',
                    '--image-height', '32', '--image-width', '32', '--preprocessing', 'none',
                    '--loss', loss, '--mixup-alpha', str(alpha), '--label-smoothing', str(epsilon)])
                launcher.validate(args)
                out = self.fixture.base / ('regularization_' + str(index))
                train_args = argparse.Namespace(**vars(args), output=out / 'train', inner_only=True, device='cpu', skip_inference_profile=True)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = training.run(train_args)
                    prediction.run(argparse.Namespace(checkpoint=out / 'train/best.pt', data_root=self.fixture.root,
                        splits_dir=self.fixture.out, output=out / 'replay', device='cpu', workers=0, threads=1,
                        batch_size=16, role='early_stop', skip_inference_profile=True))
                checkpoint = torch.load(out / 'train/best.pt', weights_only=True, map_location='cpu')
                saved = checkpoint['config']['execution_controls']
                self.assertEqual(saved['mixup_alpha'], alpha)
                self.assertEqual(saved['label_smoothing'], epsilon)
                self.assertEqual(checkpoint['config']['checkpoint_format_version'], 4)
                history = split_fixture.read_csv(out / 'train/history.csv')
                self.assertEqual(history[0]['train_metrics_scope'], 'online_unmixed_eval_mode_for_mixup' if alpha else 'online_training_mode_augmented_when_configured')
                with mock.patch.object(launcher, 'frozen_seals', return_value={'synthetic': 'sealed'}):
                    verified = launcher.verify(out, args)
                self.assertTrue(verified['prediction_replay']['slice']['identical_predictions'])
                self.assertTrue(verified['prediction_replay']['patient']['identical_predictions'])
                self.assertEqual(result['epochs_completed'], 1)


if __name__ == '__main__':
    unittest.main()
