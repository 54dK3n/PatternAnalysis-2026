"""Run a preregistered scratch preprocessing suite serially within one GPU allocation."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from slurm import preprocessing_train as single
from slurm.deploy_feature_source import frozen_seals
from utils.artifacts import code_fingerprints, validate_output, write_csv, write_json

LAUNCHERS = ('preprocessing_suite.py', 'preprocessing_suite.sbatch',
             'submit_preprocessing_suite.sh', 'preprocessing_train.py')


def parser() -> argparse.ArgumentParser:
    """Reuse the verified single-run controls and expose a closed JSON case plan."""
    result = single.parser()
    result.description = __doc__
    result.set_defaults(time_limit='08:00:00')
    result.add_argument('--plan', type=Path, default=SOURCE / 'config/preprocessing_overnight.json')
    result.add_argument('--expected-identity', help=argparse.SUPPRESS)
    return result


def build_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Resolve every case before submission; reject unknown flags and partial budgets."""
    single.validate(args)
    if not args.plan.is_absolute():
        raise ValueError('Plan path must be absolute.')
    plan = json.loads(args.plan.read_text())
    if set(plan) != {'schema_version', 'name', 'cases'} or plan['schema_version'] != 1:
        raise ValueError('Unsupported suite plan schema.')
    if not isinstance(plan['name'], str) or not plan['name']:
        raise ValueError('Plan name must be nonempty.')
    if not isinstance(plan['cases'], list) or not 1 <= len(plan['cases']) <= 32:
        raise ValueError('Plan must contain between one and 32 cases.')
    resolved, identifiers = [], set()
    for case in plan['cases']:
        if not isinstance(case, dict) or set(case) != {'id', 'purpose', 'overrides'}:
            raise ValueError('Each case requires exactly id, purpose and overrides.')
        identifier = case['id']
        if not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', identifier):
            raise ValueError('Invalid case identifier.')
        if identifier in identifiers:
            raise ValueError('Duplicate case identifier.')
        identifiers.add(identifier)
        if not isinstance(case['purpose'], str) or not case['purpose']:
            raise ValueError('Each case needs an experimental purpose.')
        overrides = case['overrides']
        if not isinstance(overrides, dict) or not set(overrides) <= set(single.TRAIN_OPTIONS):
            raise ValueError('Unsupported case override; paths/device/evaluation roles are allocation-owned.')
        if 'recipe' in overrides:
            raise ValueError('Select the base --recipe on the suite CLI; use individual controls in case overrides.')
        # Parse through argparse again so choices/types are identical to train.py.
        tokens = single.options(args)
        for key, value in overrides.items():
            default = getattr(args, key)
            expected = type(default)
            if expected is int and type(value) is not int:
                raise ValueError(f'{key} requires an integer.')
            if expected is float and (type(value) not in (float, int)):
                raise ValueError(f'{key} requires a finite number.')
            if expected is str and type(value) is not str:
                raise ValueError(f'{key} requires a string.')
            flag = '--' + key.replace('_', '-')
            if flag in tokens:
                tokens[tokens.index(flag) + 1] = str(value)
            elif value is not None:
                tokens.extend([flag, str(value)])
        effective = single.parser().parse_args(tokens)
        single.validate(effective)
        if effective.patience != effective.epochs:
            raise ValueError('Suite cases must run the full epoch budget: patience must equal epochs.')
        resolved.append({'id': identifier, 'purpose': case['purpose'], 'options': single.options(effective)})
    return resolved


def identity(args: argparse.Namespace) -> dict[str, Any]:
    """Bind training code, executed launchers and the exact supplied plan bytes."""
    return {'training_source': code_fingerprints(),
            'launchers': {name: hashlib.sha256((SOURCE / 'slurm' / name).read_bytes()).hexdigest()
                          for name in LAUNCHERS},
            'plan_sha256': hashlib.sha256(args.plan.read_bytes()).hexdigest()}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Publish progress atomically so an interrupted job leaves readable evidence."""
    temporary = path.with_suffix('.json.tmp')
    write_json(temporary, value)
    temporary.replace(path)


def run_case(output: Path, case: dict[str, Any]) -> dict[str, Any]:
    """Train/replay one fresh case; preserve failures without suppressing later cases."""
    args = single.parser().parse_args(case['options'])
    single.validate(args)
    case_root = output / case['id']
    validate_output(case_root, args.data_root, args.splits_dir)
    case_root.mkdir(exist_ok=False)
    planned = single.commands(args, case_root)
    atomic_json(case_root / 'launch.json', {'case': case, 'commands': planned})
    record: dict[str, Any] = {'case_id': case['id'], 'purpose': case['purpose'],
                             'output': str(case_root), 'status': 'running', 'options': case['options']}
    started, stage = time.monotonic(), 'train'
    try:
        with (case_root / 'execution.log').open('w', buffering=1) as log:
            for stage, command in zip(('train', 'replay'), planned):
                print(stage + ': ' + shlex.join(command), file=log, flush=True)
                subprocess.run(command, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT, check=True)
        stage = 'verification'
        verified = single.verify(case_root, args)
        if verified['epochs_completed'] != args.epochs:
            raise ValueError('A full-budget case ended before its declared epoch cap.')
        atomic_json(case_root / 'summary.json', verified)
        record.update(status='complete', summary=verified)
    except (ValueError, OSError, RuntimeError, KeyError, subprocess.CalledProcessError) as exc:
        record.update(status='failed', failed_stage=stage, error=str(exc))
    record['elapsed_seconds'] = time.monotonic() - started
    atomic_json(case_root / 'case_result.json', record)
    return record


def summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep failed cases explicit and all three aggregation levels separate."""
    rows = []
    for record in records:
        args = single.parser().parse_args(record['options'])
        summary = record.get('summary', {})
        row = {'case_id': record['case_id'], 'status': record['status'], 'fold': args.fold,
               'seed_base': args.seed, 'seed_effective': args.seed + args.fold,
               'model': args.model, 'recipe': args.recipe, 'input_channels': args.input_channels,
               'loss': args.loss, 'precision': args.precision, 'patient_aggregation': args.patient_aggregation,
               'drop_path': args.drop_path,
               'preprocessing': args.preprocessing, 'augmentation': args.augmentation,
               'sampling': args.sampling, 'lr': args.lr, 'weight_decay': args.weight_decay,
               'lr_schedule': args.lr_schedule, 'epochs_limit': args.epochs,
               'epochs_completed': summary.get('epochs_completed'), 'best_epoch': summary.get('best_epoch'),
               'image_size': json.dumps(summary.get('image_size')), 'elapsed_seconds': record['elapsed_seconds'],
               'error': record.get('error', ''), 'output': record['output']}
        for unit in ('slice', 'scan', 'patient'):
            scores = summary.get('metrics', {}).get(unit) or {}
            for metric in ('accuracy', 'balanced_accuracy', 'macro_f1', 'auroc', 'log_loss'):
                row[f'{unit}_{metric}'] = scores.get(metric)
        resources = summary.get('resources', {})
        for metric in ('trainable_parameters', 'training_seconds', 'training_peak_cuda_allocated_mib'):
            row[metric] = resources.get(metric)
        rows.append(row)
    return rows


def publish(output: Path, plan: list[dict[str, Any]], records: list[dict[str, Any]], status: str) -> None:
    """Update a machine-readable suite index after every success or failure."""
    atomic_json(output / 'summary.json', {'status': status, 'planned_cases': len(plan),
                'finished_cases': len(records), 'complete_cases': sum(r['status'] == 'complete' for r in records),
                'failed_cases': sum(r['status'] == 'failed' for r in records), 'records': records,
                'evaluation_scope': 'inner_early_stop_only_not_outer_cv_or_final_test',
                'checkpoint_selection': 'minimum_early_stop_scan_log_loss',
                'limitations': ['Early-stop data are reused for selection and comparison; no independent accuracy claim.',
                                'Geometry profiles can have different canvas sizes.',
                                'Online training scores use training mode and configured augmentation.',
                                'Sampling changes include the declared sampler/loss class-prior behavior.']})
    if records:
        temporary = output / 'summary.csv.tmp'
        write_csv(temporary, summary_rows(records))
        temporary.replace(output / 'summary.csv')


def figures(output: Path, records: list[dict[str, Any]]) -> None:
    """Plot aggregate scores and actual epoch trajectories without MRI thumbnails."""
    os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [row for row in summary_rows(records) if row['status'] == 'complete']
    if not rows:
        return
    figure, axes = plt.subplots(1, 3, figsize=(17, max(5, len(rows) * .32)), layout='constrained')
    labels = [r['case_id'] for r in rows]
    for axis, metric, title in zip(axes, ('slice_accuracy', 'slice_auroc', 'scan_log_loss'),
                                   ('Inner slice accuracy', 'Inner slice AUROC', 'Selected scan log loss')):
        values = [r[metric] for r in rows]
        axis.barh(labels, [0 if v is None else v for v in values])
        for i, value in enumerate(values):
            if value is None:
                axis.text(0, i, 'unavailable')
        axis.invert_yaxis()
        axis.set_title(title)
        axis.grid(axis='x', alpha=.2)
        if metric == 'slice_accuracy':
            axis.axvline(.8, color='crimson', linestyle='--', label='0.80 reference (development only)')
            axis.legend(fontsize=7)
        if metric != 'scan_log_loss':
            axis.set_xlim(0, 1)
    figure.suptitle('Scratch inner development: selected by scan loss; not independent test performance')
    figure.savefig(output / 'comparison.png', dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    for row in rows:
        with (Path(row['output']) / 'train/history.csv').open(newline='') as handle:
            history = list(csv.DictReader(handle))
        epochs = [int(h['epoch']) for h in history]
        for axis, metric in zip(axes, ('train_slice_accuracy', 'early_stop_slice_accuracy', 'early_stop_scan_loss')):
            axis.plot(epochs, [float(h[metric]) for h in history], label=row['case_id'], linewidth=1)
    for axis, title in zip(axes, ('Online train slice accuracy', 'Inner slice accuracy', 'Inner scan log loss')):
        axis.set(title=title, xlabel='Epoch')
        axis.grid(alpha=.2)
    axes[-1].legend(fontsize=6, loc='upper left', bbox_to_anchor=(1, 1))
    figure.suptitle('Actual full-budget histories; online training uses training mode/augmentation')
    figure.savefig(output / 'trajectories.png', dpi=160)
    plt.close(figure)


def execute(args: argparse.Namespace, plan: list[dict[str, Any]], output: Path) -> int:
    """Keep source/splits immutable and finish a bounded serial batch on one GPU."""
    import torch
    if not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; suite was not started.')
    validate_output(output, args.data_root, args.splits_dir)
    before, seals = identity(args), frozen_seals(args.splits_dir)
    digest = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
    if args.expected_identity and args.expected_identity != digest:
        raise ValueError('Source/plan changed after submission; refusing to start.')
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'launch.json', {'job_id': os.environ['SLURM_JOB_ID'],
                'node': os.environ.get('SLURMD_NODENAME'), 'gpu': torch.cuda.get_device_name(0),
                'identity': before, 'frozen_seals': seals, 'plan': plan,
                'submission_options': single.options(args), 'plan_path': str(args.plan)})
    records: list[dict[str, Any]] = []
    publish(output, plan, records, 'running')
    try:
        for index, case in enumerate(plan, 1):
            if identity(args) != before or frozen_seals(args.splits_dir) != seals:
                raise ValueError('Source/plan/frozen manifests changed; aborting the suite.')
            print(f'[{index}/{len(plan)}] Starting {case["id"]}: {case["purpose"]}', flush=True)
            record = run_case(output, case)
            records.append(record)
            publish(output, plan, records, 'running')
            print(f'[{index}/{len(plan)}] {case["id"]}: {record["status"]}; {record["elapsed_seconds"]:.1f}s', flush=True)
            if identity(args) != before or frozen_seals(args.splits_dir) != seals:
                raise ValueError('Source/plan/frozen manifests changed during the case.')
        figures(output, records)
    except Exception as exc:
        publish(output, plan, records, 'aborted')
        atomic_json(output / 'suite_error.json', {'error': str(exc)})
        raise
    status = 'complete' if all(r['status'] == 'complete' for r in records) else 'completed_with_failures'
    publish(output, plan, records, status)
    print(f'Suite {status}: {output}', flush=True)
    return 0 if status == 'complete' else 1


def main(argv: list[str] | None = None) -> int:
    """Preview, submit, or execute the same preregistered experiment plan."""
    args = parser().parse_args(argv)
    try:
        plan = build_plan(args)
        job = os.environ.get('SLURM_JOB_ID') if args.run else '<SLURM_JOB_ID>'
        if args.run and (args.dry_run or not job or not job.isdecimal()):
            raise ValueError('Run mode requires a real numeric Slurm allocation.')
        output = args.runs_root / f'preprocessing_overnight_job{job}'
        logs = args.runs_root / 'slurm_logs'
        expected = hashlib.sha256(json.dumps(identity(args), sort_keys=True).encode()).hexdigest()
        if args.run and args.expected_identity != expected:
            raise ValueError('Missing or changed submission identity; refusing to start.')
        forwarded = [*single.options(args), '--plan', str(args.plan), '--expected-identity', expected]
        submit = ['sbatch', '--parsable', '--export=ALL', '--time=' + args.time_limit,
                  '--cpus-per-task=' + str(args.cpus_per_task), '--output=' + str(logs / 'preprocessing_overnight_%j.out'),
                  '--error=' + str(logs / 'preprocessing_overnight_%j.err'),
                  str(SOURCE / 'slurm/preprocessing_suite.sbatch'), str(SOURCE), *forwarded]
        if args.dependency is not None:
            submit.insert(1, '--dependency=' + args.dependency)
        print('Submission:', shlex.join(submit), flush=True)
        print(f'Cases: {len(plan)}; one GPU, serial; scratch; full budgets; inner early-stop only.', flush=True)
        for case in plan:
            print(case['id'] + ': ' + case['purpose'], flush=True)
            for command in single.commands(single.parser().parse_args(case['options']), output / case['id']):
                print(shlex.join(command), flush=True)
        if args.dry_run:
            return 0
        if args.run:
            return execute(args, plan, output)
        if not args.data_root.is_dir():
            raise ValueError('Source data directory missing.')
        frozen_seals(args.splits_dir)
        logs.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(submit, check=True, capture_output=True, text=True)
        print('Submitted job:', result.stdout.strip(), flush=True)
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, subprocess.CalledProcessError) as exc:
        print('ERROR:', exc, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
