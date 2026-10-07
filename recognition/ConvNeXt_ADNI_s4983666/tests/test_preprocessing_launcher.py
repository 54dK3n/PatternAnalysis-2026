"""Check side-effect-free previews and verify actual multi-epoch synthetic replay."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from engine import prediction, training
from slurm import preprocessing_train as launcher
import test_adni_splits as split_fixture


class PreviewTests(unittest.TestCase):
    """Prevent command drift, invalid allocations and accidental submission during review."""

    def test_all_profiles_preview_without_submission_or_outputs(self) -> None:
        """Previewing four controls neither reads real data nor creates output storage."""
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            for profile in ('none', 'scan_intensity', 'scan_crop', 'scan_intensity_crop'):
                with mock.patch.object(launcher.subprocess, 'run') as process, \
                        contextlib.redirect_stdout(io.StringIO()) as stream:
                    status = launcher.main(['--dry-run', '--preprocessing', profile,
                                            '--data-root', str(base / 'missing_data'),
                                            '--splits-dir', str(base / 'missing_splits'),
                                            '--runs-root', str(base / 'runs')])
                self.assertEqual(status, 0)
                process.assert_not_called()
                self.assertIn('--epochs 30', stream.getvalue())
                self.assertIn('--patience 30', stream.getvalue())
                self.assertIn('--inner-only', stream.getvalue())
                self.assertIn('--preprocessing ' + profile, stream.getvalue())
            self.assertFalse((base / 'runs').exists())

    def test_overrides_roundtrip_and_replay_use_same_loader_controls(self) -> None:
        """An override appears once and is forwarded unchanged into the GPU job."""
        args = launcher.parser().parse_args(['--epochs', '15', '--lr', '0.0003', '--batch-size', '16',
                                            '--preprocessing', 'scan_crop', '--augmentation', 'integer_shift',
                                            '--lr-schedule', 'warmup_cosine', '--seed', '4710'])
        launcher.validate(args)
        forwarded = launcher.parser().parse_args(launcher.options(args))
        self.assertEqual(vars(args), vars(forwarded))
        train, replay = launcher.commands(args, Path('/tmp/run with spaces'))
        self.assertEqual(train.count('--epochs'), 1)
        self.assertEqual(train[train.index('--epochs') + 1], '15')
        self.assertEqual(train[train.index('--lr') + 1], '0.0003')
        for command in (train, replay):
            self.assertEqual(command[command.index('--batch-size') + 1], '16')
        self.assertIn('/tmp/run with spaces/train/best.pt', replay)
        self.assertNotIn('--preprocessing', replay)

    def test_default_patience_follows_budget_with_explicit_override(self) -> None:
        """Changing only the epoch budget keeps a full run; explicit shorter patience is respected."""
        for budget in (1, 15, 30, 60):
            args = launcher.parser().parse_args(['--epochs', str(budget)])
            launcher.validate(args)
            self.assertEqual(args.patience, budget)
            train, _ = launcher.commands(args, Path('/tmp/patience_preview'))
            self.assertEqual(train[train.index('--patience') + 1], str(budget))
        args = launcher.parser().parse_args(['--epochs', '30', '--patience', '10'])
        launcher.validate(args)
        self.assertEqual(args.patience, 10)

    def test_invalid_options_fail_before_submission(self) -> None:
        """Reject bad warmup, clipping configuration, protected output paths and CPU budgets."""
        cases = [['--epochs', '2', '--lr-schedule', 'warmup_cosine'], ['--lr', 'nan'],
                 ['--lr', '-1'], ['--crop-height', '224'], ['--preprocessing', 'none', '--crop-height', '224', '--crop-width', '192'],
                 ['--workers', '4'], ['--runs-root', str(launcher.SOURCE / 'runs')],
                 ['--time-limit', '00:00:00'], ['--data-root', 'relative']]
        for values in cases:
            with self.subTest(values=values), mock.patch.object(launcher.subprocess, 'run') as process, \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(launcher.main(['--dry-run', *values]), 1)
                process.assert_not_called()


class ReplayVerificationTests(unittest.TestCase):
    """Exercise CPU training/replay with real synthetic manifests and multiple epochs."""

    def setUp(self) -> None:
        """Build separate synthetic source and frozen split directories."""
        self.fixture = split_fixture.ADNISplitTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.prepare()

    def test_early_stopping_and_best_before_last_for_both_checkpoint_formats(self) -> None:
        """A one-epoch assumption must not invalidate multi-epoch runs or historical preprocessing."""
        for profile in ('none', 'scan_intensity_crop'):
            output = self.fixture.base / profile
            args = launcher.parser().parse_args(['--model', 'cnn', '--data-root', str(self.fixture.root),
                      '--splits-dir', str(self.fixture.out), '--runs-root', str(self.fixture.base / 'runs'),
                      '--epochs', '5', '--patience', '1', '--min-delta', '100', '--workers', '0', '--threads', '1',
                      '--batch-size', '16', '--image-height', '32', '--image-width', '32',
                      '--preprocessing', profile, '--crop-margin', '2'])
            train_args = argparse.Namespace(**vars(args), output=output / 'train', inner_only=True,
                                            device='cpu', skip_inference_profile=True)
            # Inject worse selection loss in the second epoch to force the genuine
            # first-epoch weights to remain best; final evaluation/replay use real scores.
            evaluate = training.evaluate
            calls = 0

            def controlled_selection(*values: object, **kwargs: object) -> object:
                """Only two inner selection losses are controlled; final probabilities are real."""
                nonlocal calls
                scores, slices, scans = evaluate(*values, **kwargs)
                calls += 1
                if calls <= 2:
                    scores['scan']['log_loss'] = float(calls)
                return scores, slices, scans

            with mock.patch.object(training, 'evaluate', side_effect=controlled_selection), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = training.run(train_args)
            self.assertEqual((result['epochs_completed'], result['best_epoch']), (2, 1))
            replay_args = argparse.Namespace(checkpoint=output / 'train/best.pt', data_root=self.fixture.root,
                            splits_dir=self.fixture.out, output=output / 'replay', device='cpu', workers=0,
                            threads=1, batch_size=16, role='early_stop', skip_inference_profile=True)
            with contextlib.redirect_stdout(io.StringIO()):
                prediction.run(replay_args)
            # The production seal checker binds the real Rangpur split identity;
            # replace only that identity check for this independently prepared fixture.
            with mock.patch.object(launcher, 'frozen_seals', return_value={'synthetic': 'sealed'}):
                verified = launcher.verify(output, args)
            self.assertTrue(verified['stopped_before_epoch_cap'])
            self.assertEqual(verified['best_epoch'], 1)
            self.assertTrue(verified['prediction_replay']['slice']['identical_predictions'])
            with self.assertRaisesRegex(ValueError, 'budget|seed|fold'):
                launcher.verify(output, argparse.Namespace(**{**vars(args), 'epochs': 6}))
            with self.assertRaisesRegex(ValueError, 'optimizer'):
                launcher.verify(output, argparse.Namespace(**{**vars(args), 'lr': 0.003}))
            path = output / 'replay/slice_predictions.csv'
            path.write_bytes(path.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'replay differs'):
                launcher.verify(output, args)


if __name__ == '__main__':
    unittest.main()
