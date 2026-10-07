"""Check prospective plans, serial failure isolation, source guards and genuine replay."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import torch
from slurm import preprocessing_suite as suite
from slurm import preprocessing_train as single
import test_adni_splits as split_fixture


class PlanTests(unittest.TestCase):
    """Protect full-budget development-only commands before GPU submission."""

    def test_default_plan_and_dry_run(self) -> None:
        """All 18 cases resolve before a preview; no subprocess or output is created."""
        args = suite.parser().parse_args(['--dry-run'])
        plan = suite.build_plan(args)
        self.assertEqual(len(plan), 18)
        self.assertEqual(sum(single.parser().parse_args(c['options']).epochs for c in plan), 570)
        for case in plan:
            resolved = single.parser().parse_args(case['options'])
            self.assertEqual(resolved.patience, resolved.epochs)
            train, replay = single.commands(resolved, Path('/tmp/suite_preview') / case['id'])
            self.assertIn('--inner-only', train)
            self.assertEqual(replay[replay.index('--role') + 1], 'early_stop')
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(suite.subprocess, 'run') as process, \
                contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(suite.main(['--dry-run', '--runs-root', str(Path(temp) / 'runs')]), 0)
            self.assertFalse((Path(temp) / 'runs').exists())
            process.assert_not_called()
            self.assertIn('--expected-identity', stream.getvalue())
            self.assertIn('Cases: 18; one GPU, serial', stream.getvalue())

    def test_invalid_plan_and_short_patience(self) -> None:
        """Reject duplicate/path identifiers, protected overrides, bad types and budget drift."""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'plan.json'
            case = {'id': 'A01', 'purpose': 'Synthetic control.', 'overrides': {}}
            values = [[case, case], [{**case, 'id': '../escape'}],
                      [{**case, 'overrides': {'data_root': '/tmp/other'}}],
                      [{**case, 'overrides': {'epochs': '30'}}],
                      [{**case, 'overrides': {'patience': 5}}],
                      [{**case, 'overrides': {'lr': float('nan')}}]]
            for cases in values:
                path.write_text(json.dumps({'schema_version': 1, 'name': 'synthetic', 'cases': cases}))
                args = suite.parser().parse_args(['--plan', str(path)])
                with self.assertRaises(ValueError):
                    suite.build_plan(args)


class AllocationTests(unittest.TestCase):
    """Ensure one failed case does not hide progress or stop independent later cases."""

    def test_continue_failure_and_abort_changed_source(self) -> None:
        """A partial batch has an explicit nonzero outcome; source drift stops new work."""
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            args = suite.parser().parse_args(['--runs-root', str(base / 'runs')])
            plan = suite.build_plan(args)[:2]
            records = [{'case_id': c['id'], 'purpose': c['purpose'], 'output': str(base / c['id']),
                        'options': c['options'], 'status': 'failed', 'error': 'synthetic failure',
                        'elapsed_seconds': 1.0} for c in plan]
            records[1]['status'] = 'complete'
            with mock.patch.object(torch.cuda, 'is_available', return_value=True), \
                    mock.patch.object(torch.cuda, 'get_device_name', return_value='synthetic'), \
                    mock.patch.dict('os.environ', {'SLURM_JOB_ID': '123'}), \
                    mock.patch.object(suite, 'frozen_seals', return_value={'synthetic': 'sealed'}), \
                    mock.patch.object(suite, 'run_case', side_effect=records) as run, \
                    mock.patch.object(suite, 'figures') as figures, \
                    contextlib.redirect_stdout(io.StringIO()):
                output = base / 'continued'
                self.assertEqual(suite.execute(args, plan, output), 1)
                self.assertEqual(run.call_count, 2)
                self.assertEqual(json.loads((output / 'summary.json').read_text())['status'], 'completed_with_failures')
                self.assertIn('synthetic failure', (output / 'summary.csv').read_text())
                figures.assert_called_once()
            with mock.patch.object(torch.cuda, 'is_available', return_value=True), \
                    mock.patch.object(torch.cuda, 'get_device_name', return_value='synthetic'), \
                    mock.patch.dict('os.environ', {'SLURM_JOB_ID': '123'}), \
                    mock.patch.object(suite, 'frozen_seals', return_value={'synthetic': 'sealed'}), \
                    mock.patch.object(suite, 'identity', side_effect=[{'x': 1}, {'x': 2}]), \
                    mock.patch.object(suite, 'run_case') as run:
                output = base / 'aborted'
                with self.assertRaisesRegex(ValueError, 'changed'):
                    suite.execute(args, plan, output)
                run.assert_not_called()
                self.assertEqual(json.loads((output / 'summary.json').read_text())['status'], 'aborted')

    def test_submission_identity_guard(self) -> None:
        """Changing source between sbatch submission and allocation refuses any output."""
        with tempfile.TemporaryDirectory() as temp:
            args = suite.parser().parse_args(['--expected-identity', 'changed'])
            plan = suite.build_plan(args)[:1]
            output = Path(temp) / 'not_created'
            with mock.patch.object(torch.cuda, 'is_available', return_value=True), \
                    mock.patch.object(suite, 'frozen_seals', return_value={'synthetic': 'sealed'}):
                with self.assertRaisesRegex(ValueError, 'changed after submission'):
                    suite.execute(args, plan, output)
            self.assertFalse(output.exists())


class RealCaseTests(unittest.TestCase):
    """Run the production subprocess and replay flow using synthetic CPU-only inputs."""

    def test_training_replay_failure_logging_and_figures(self) -> None:
        """One real synthetic case verifies weights/predictions; later failure retains logs."""
        fixture = split_fixture.ADNISplitTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.prepare()
        args = suite.parser().parse_args(['--model', 'cnn', '--epochs', '2', '--workers', '0',
                '--threads', '1', '--batch-size', '16', '--image-height', '32', '--image-width', '32',
                '--preprocessing', 'scan_intensity_crop', '--crop-margin', '2',
                '--data-root', str(fixture.root), '--splits-dir', str(fixture.out),
                '--runs-root', str(fixture.base / 'runs')])
        single.validate(args)
        case = {'id': 'SyntheticCPU', 'purpose': 'Synthetic training and replay.', 'options': single.options(args)}
        output = fixture.base / 'suite'
        output.mkdir()
        actual_commands = single.commands

        def cpu_commands(options: object, target: Path) -> list[list[str]]:
            """Use the identical command interface on CPU for the independent fixture."""
            commands = actual_commands(options, target)
            for command in commands:
                command[command.index('--device') + 1] = 'cpu'
            commands[0].append('--skip-inference-profile')
            return commands

        with mock.patch.object(single, 'commands', side_effect=cpu_commands), \
                mock.patch.object(single, 'frozen_seals', return_value={'synthetic': 'sealed'}):
            result = suite.run_case(output, case)
        self.assertEqual(result['status'], 'complete', str(result.get('error')) + '\n' + (output / case['id'] / 'execution.log').read_text())
        self.assertEqual(result['summary']['epochs_completed'], 2)
        self.assertTrue(result['summary']['prediction_replay']['slice']['identical_predictions'])
        suite.publish(output, [case], [result], 'complete')
        suite.figures(output, [result])
        for name in ('comparison.png', 'trajectories.png'):
            self.assertGreater((output / name).stat().st_size, 1000)
        failed_case = {**case, 'id': 'SyntheticFailure'}
        with mock.patch.object(single.subprocess, 'run', side_effect=subprocess.CalledProcessError(7, ['synthetic'])):
            failed = suite.run_case(output, failed_case)
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(failed['failed_stage'], 'train')
        self.assertTrue((output / failed_case['id'] / 'execution.log').exists())
        with self.assertRaisesRegex(ValueError, 'already exists'):
            suite.run_case(output, case)


if __name__ == '__main__':
    unittest.main()
