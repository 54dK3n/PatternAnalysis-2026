"""Serial feature-control matrix, resource checks and prospective paired repeats."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from engine.feature_controls import eligible_repeat
from engine.feature_training import ROOT, SPEC, source_identity
from evaluation.feature_figures import paired_summary, plot_probe_summary, plot_suite, plot_spatial_features
from utils.artifacts import write_json


def marker_identity(splits: Path) -> str:
    """Hash the frozen completion marker without regenerating any assignments."""
    return hashlib.sha256((splits / 'COMPLETED.json').read_bytes()).hexdigest()


def build_plan(args: argparse.Namespace) -> dict:
    """Bind specification, operator settings, inputs and every executable source."""
    specification = json.loads(SPEC.read_text())
    marker = marker_identity(args.splits_dir)
    if args.synthetic_smoke:
        if marker == specification['manifest_sha256']:
            raise ValueError('Synthetic smoke refuses the real coursework manifests.')
    elif marker != specification['manifest_sha256']:
        raise ValueError('Frozen manifest identity differs from the preregistered design.')
    old = []
    if args.phase in ('all', 'diagnostics'):
        for diagnostic in specification['existing_diagnostics']:
            checkpoint = args.reference_suite / diagnostic['checkpoint_relative_to_suite']
            old.append({'id': diagnostic['id'], 'checkpoint': str(checkpoint.resolve()),
                        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    return {'version': 2, 'specification': specification, 'source_sha256': source_identity(),
            'manifest_sha256': marker, 'data_root': str(args.data_root.resolve()),
            'splits_dir': str(args.splits_dir.resolve()), 'device': args.device, 'workers': args.workers,
            'phase': args.phase, 'resource_smoke': not args.skip_resource_smoke,
            'paired_repeats': not args.no_repeats, 'synthetic_smoke': args.synthetic_smoke,
            'skip_inference_profile': args.skip_inference_profile, 'diagnostics': old}


def command(args: argparse.Namespace, case: str, output: Path, seed: int = 3710,
            resource_smoke: bool = False) -> list[str]:
    """Construct a fresh child process without changing declared case factors."""
    argv = [sys.executable, '-u', str(ROOT / 'train_feature_experiment.py'), '--case', case,
            '--output', str(output), '--data-root', str(args.data_root), '--splits-dir', str(args.splits_dir),
            '--seed-base', str(seed), '--workers', str(args.workers), '--device', args.device]
    if resource_smoke:
        argv.append('--smoke')
    if args.synthetic_smoke:
        argv.append('--synthetic-smoke')
    if args.skip_inference_profile:
        argv.append('--skip-inference-profile')
    return argv


def completed(path: Path, item: dict, plan: dict) -> dict | None:
    """Reuse only a completed exact source/manifest/case binding, never partials."""
    marker = path / ('metrics.json' if item['kind'] == 'training' else 'summary.json')
    if not marker.is_file():
        return None
    binding = json.loads((path.parent / (path.name + '.binding.json')).read_text())
    if binding != item:
        raise ValueError(f'Different binding at {path}')
    result = json.loads(marker.read_text())
    if result.get('status') != 'complete':
        return None
    config = json.loads((path / 'config.json').read_text())
    if config['manifest_sha256'] != plan['manifest_sha256']:
        raise ValueError('Completed result has another manifest.')
    if item['kind'] == 'training':
        expected_kind = 'synthetic_smoke' if plan['synthetic_smoke'] else 'resource_smoke' if item['smoke'] else 'fixed_budget'
        expected_epochs = 1 if expected_kind != 'fixed_budget' else 30
        if (config['code_sha256'] != plan['source_sha256'] or result['code_sha256'] != plan['source_sha256']
                or config['case_id'] != item['case'] or config['seed_base'] != item['seed_base']
                or result['epochs_completed'] != expected_epochs or result['run_kind'] != expected_kind
                or result['evaluation_role'] != 'development_inner_early_stop'):
            raise ValueError(f'Completed training violates the prospective protocol: {path}')
    return {'id': item['id'], 'status': 'complete', 'output': str(path), 'returncode': 0}


def execute(output: Path, item: dict, plan: dict, make_command: Any) -> dict:
    """Retain failed attempts and retry from scratch in new numbered directories."""
    reserved = sorted(output.glob(item['id'] + '_attempt_*.binding.json'))
    attempts = [output / p.name.removesuffix('.binding.json') for p in reserved]
    for path in reversed(attempts):
        record = completed(path, item, plan)
        if record:
            print(f"Reuse completed {item['id']}", flush=True)
            return record
    number = max((int(p.name.rsplit('_', 1)[1]) for p in attempts), default=0) + 1
    path = output / f"{item['id']}_attempt_{number:02d}"
    write_json(output / (path.name + '.binding.json'), item)
    argv = make_command(path)
    write_json(output / (path.name + '.command.json'), {'argv': argv})
    started = time.perf_counter()
    print(f"Starting {item['id']}; log={path.name}.log", flush=True)
    with (output / (path.name + '.log')).open('w') as stream:
        process = subprocess.run(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if process.returncode == 0:
        record = completed(path, item, plan)
        if record is None:
            raise ValueError('Child returned success without its completed marker.')
    else:
        record = {'id': item['id'], 'status': 'failed', 'output': str(path), 'returncode': process.returncode}
        print(f"Failed {item['id']}; preserving partials and continuing.", flush=True)
    record['duration_seconds'] = time.perf_counter() - started
    return record


def diagnostic_command(args: argparse.Namespace, item: dict, output: Path, reference: Path | None = None) -> list[str]:
    """Replay all complete train/early-stop scans using the fixed diagnostic recipe."""
    argv = [sys.executable, '-u', str(ROOT / ('diagnose_followup.py' if reference else 'diagnose.py')),
            '--checkpoint', item['checkpoint'], '--data-root', str(args.data_root), '--splits-dir', str(args.splits_dir),
            '--output', str(output), '--device', args.device, '--batch-size', '32', '--workers', str(args.workers),
            '--threads', '2', '--seed', '3710', '--max-early-patients', '0', '--max-scans-per-patient', '0']
    if reference:
        argv += ['--reference-dir', str(reference), '--routes', 'spatial', '--max-shift', '8', '--bootstrap-samples', '1000']
    else:
        argv += ['--max-train-patients', '0', '--max-transform-images', '12', '--shift-pixels', '0',
                 '--probes', '--probe-epochs', '200', '--probe-lr', '.01', '--probe-weight-decay', '.01']
    return argv


def verify_orders(records: list[dict]) -> None:
    """Fail a comparison if paired cases consumed different per-epoch slice batches."""
    references: dict[int, list] = {}
    initial_references: dict[int, str] = {}
    for record in records:
        if record['status'] != 'complete' or record['id'].startswith(('D', 'SMOKE')):
            continue
        path = Path(record['output'])
        config = json.loads((path / 'config.json').read_text())
        with (path / 'history.csv').open() as stream:
            import csv
            hashes = [(r['training_batch_order_sha256'], r['optimizer_steps']) for r in csv.DictReader(stream)]
        seed = config['seed']
        reference = references.setdefault(seed, hashes)
        initial = config['initialization_control']['reference_initial_sha256']
        if reference != hashes or initial_references.setdefault(seed, initial) != initial:
            raise ValueError('Paired initialization or training batch order changed; comparison stopped.')


def run(args: argparse.Namespace) -> int:
    """Run all cases sequentially in one GPU allocation; never auto-submit jobs."""
    plan = build_plan(args)
    if args.list:
        print(json.dumps(plan, indent=2))
        return 0
    output = args.output.resolve()
    for protected in (args.data_root.resolve(), args.splits_dir.resolve(), ROOT):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('Suite output must be separate from source/data/manifests.')
    output.mkdir(parents=True, exist_ok=True)
    plan_file = output / 'suite_plan.json'
    if plan_file.exists():
        if json.loads(plan_file.read_text()) != plan:
            raise ValueError('Existing suite has another source or plan; choose a fresh root.')
    elif any(output.iterdir()):
        raise ValueError('Nonempty suite folder has no bound plan.')
    else:
        write_json(plan_file, plan)
    records = []

    def launch(item: dict, builder: Any) -> dict:
        """Check immutable provenance before every child and persist progress."""
        if build_plan(args) != plan:
            raise ValueError('Source, manifest, or diagnostic checkpoint changed during the suite.')
        record = execute(output, item, plan, builder)
        records.append(record)
        write_json(output / 'suite_records.json', records)
        return record

    for diagnostic in plan['diagnostics']:
        item = diagnostic | {'kind': 'diagnosis', 'id': diagnostic['id'] + '_features'}
        record = launch(item, lambda p, d=diagnostic: diagnostic_command(args, d, p))
        if record['status'] != 'complete':
            continue
        reference = Path(record['output'])
        plot_probe_summary(reference)
        item = diagnostic | {'kind': 'spatial', 'id': diagnostic['id'] + '_spatial', 'reference': str(reference)}
        spatial = launch(item, lambda p, d=diagnostic, r=reference: diagnostic_command(args, d, p, r))
        if spatial['status'] == 'complete':
            plot_spatial_features(Path(spatial['output']) / 'spatial')
    if args.phase in ('all', 'core'):
        smoke_failed = set()
        if not args.skip_resource_smoke:
            for case in ('R02', 'R03', 'R05'):
                item = {'id': 'SMOKE_' + case, 'kind': 'training', 'case': case, 'seed_base': 3710, 'smoke': True}
                result = launch(item, lambda p, c=case: command(args, c, p, resource_smoke=True))
                if result['status'] != 'complete':
                    smoke_failed.add(case)
        core = {}
        for case in ('R00', 'R01', 'R02', 'R03', 'R04', 'R05'):
            if case in smoke_failed:
                records.append({'id': case, 'status': 'blocked_resource_smoke', 'output': None})
                write_json(output / 'suite_records.json', records)
                continue
            item = {'id': case, 'kind': 'training', 'case': case, 'seed_base': 3710, 'smoke': False}
            result = launch(item, lambda p, c=case: command(args, c, p))
            if result['status'] == 'complete':
                core[case] = json.loads((Path(result['output']) / 'metrics.json').read_text())
        verify_orders(records)
        gate = eligible_repeat(core) if not args.synthetic_smoke else {'status': 'not_eligible_synthetic_smoke', 'selected': None}
        if any(r['status'] != 'complete' for r in records if r['id'].startswith('D')):
            gate = {'status': 'blocked_incomplete_diagnostics', 'selected': None}
        write_json(output / 'repeat_gate.json', gate)
        if gate['selected'] and not args.no_repeats:
            candidate = gate['selected']
            for seed in (4710, 5710):
                for case in ('R00', candidate):
                    item = {'id': f'{case}_seed{seed}', 'kind': 'training', 'case': case, 'seed_base': seed, 'smoke': False}
                    launch(item, lambda p, c=case, s=seed: command(args, c, p, s))
            verify_orders(records)
            paired_summary(output, records, candidate)
        plot_suite(output, records)
    failed = [r for r in records if r['status'] != 'complete']
    write_json(output / 'suite_summary.json', {'status': 'partial_failures' if failed else 'complete',
               'manifest_sha256': plan['manifest_sha256'], 'source_sha256': plan['source_sha256'],
               'records': records, 'failed_cases': [r['id'] for r in failed],
               'protected_roles_scored': [], 'device_requested': args.device})
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    """Expose only bounded phase/resource controls; declared case settings stay fixed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/home/groups/comp3710/ADNI'))
    parser.add_argument('--splits-dir', type=Path, default=ROOT.parent / 'adni_splits_v1')
    parser.add_argument('--output', type=Path, default=ROOT.parent / 'runs/feature_suite')
    parser.add_argument('--reference-suite', type=Path, default=ROOT.parent / 'runs/convnext_suite_job633000')
    parser.add_argument('--phase', choices=('all', 'core', 'diagnostics'), default='all')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--skip-resource-smoke', action='store_true')
    parser.add_argument('--no-repeats', action='store_true', help='Stop after six core cases even if gate passes.')
    parser.add_argument('--skip-inference-profile', action='store_true')
    parser.add_argument('--synthetic-smoke', action='store_true', help='Synthetic 32x32 one-epoch checks; no repeat gate.')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args(argv)
    if args.workers < 0:
        parser.error('Workers cannot be negative.')
    try:
        return run(args)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
