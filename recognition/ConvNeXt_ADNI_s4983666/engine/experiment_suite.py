"""Run a predeclared inner-only experiment plan in independent processes.

This suite never selects a final model or scores outer/calibration/test data.
Completed attempts are reused only when their plan, code and manifests match.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from utils.artifacts import code_fingerprints, write_json


def cases() -> list[dict]:
    """Return fixed one-factor controls, two interactions and two seed repeats."""
    return [
        {'id':'E00_cnn', 'model':'cnn', 'lr':0.001, 'weight_decay':0.0001},
        {'id':'E01_lite_reference'},
        {'id':'E02_lr_low', 'lr':0.00003},
        {'id':'E03_lr_high', 'lr':0.0003},
        {'id':'E04_decay_low', 'weight_decay':0.01},
        {'id':'E05_decay_high', 'weight_decay':0.2},
        {'id':'E06_batch16', 'batch_size':16},
        {'id':'E07_warmup_cosine', 'lr_schedule':'warmup_cosine'},
        {'id':'E08_patient_sampling', 'sampling':'class_patient_balanced'},
        {'id':'E09_integer_shift', 'augmentation':'integer_shift'},
        {'id':'E10_subpixel_shift', 'augmentation':'subpixel_shift'},
        {'id':'E11_rotation_only', 'augmentation':'light', 'rotation_degrees':3.0, 'translation_fraction':0.0},
        {'id':'E12_gamma', 'augmentation':'gamma'},
        {'id':'E13_sampling_integer', 'sampling':'class_patient_balanced', 'augmentation':'integer_shift'},
        {'id':'E14_sampling_integer_gamma', 'sampling':'class_patient_balanced', 'augmentation':'integer_gamma'},
        {'id':'E15_combined_seed4710', 'sampling':'class_patient_balanced', 'augmentation':'integer_gamma', 'seed':4710},
        {'id':'E16_combined_seed5710', 'sampling':'class_patient_balanced', 'augmentation':'integer_gamma', 'seed':5710},
    ]


def build_plan(args: argparse.Namespace, root: Path) -> dict:
    """Freeze all training flags, source identities and source-manifest marker."""
    base = dict(data_root=str(args.data_root.resolve()), splits_dir=str(args.splits_dir.resolve()),
                model='convnext_lite', fold=1, epochs=args.epochs, patience=5,
                batch_size=32, workers=args.workers, threads=2, seed=3710,
                lr=0.0001, weight_decay=0.05, augmentation='none', sampling='slice_uniform',
                rotation_degrees=0.0, translation_fraction=0.0, translation_pixels=4,
                lr_schedule='constant', warmup_epochs=min(2, args.epochs - 1), min_lr_ratio=0.01,
                image_height=240, image_width=256, device=args.device,
                calibration_bins=15, reject_threshold=0.8, profile_warmup=10, profile_repeats=100)
    planned = []
    for case in cases():
        planned.append({'id':case['id'], 'flags':base | {k:v for k,v in case.items() if k != 'id'}})
    seal = args.splits_dir / 'COMPLETED.json'
    sources = code_fingerprints()
    for name in ('run_experiment_suite.py', 'slurm/submit_suite.sh', 'slurm/suite.sbatch'):
        sources[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    return {'version':1, 'scope':'development_inner_early_stop_only',
            'selection_rule':'minimum_early_stop_scan_log_loss_unchanged',
            'comparison_primary_unit':'slice', 'outer_validation_scored':False,
            'calibration_fitted':False, 'final_test_scored':False,
            'manifest_sha256':hashlib.sha256(seal.read_bytes()).hexdigest() if seal.is_file() else None,
            'code_sha256':sources, 'cases':planned}


def command(root: Path, flags: dict, output: Path) -> list[str]:
    """Construct shell-free CLI arguments with an unconditional inner-only flag."""
    argv = [sys.executable, '-u', str(root / 'train.py'), '--inner-only', '--output', str(output)]
    for key, value in flags.items():
        argv.extend(['--' + key.replace('_', '-'), str(value)])
    return argv


def write_summary(output: Path, records: list[dict]) -> None:
    """Save progress without ranking models or fitting any decision threshold."""
    write_json(output / 'suite_summary.json', records)
    columns = ['id','status','output','best_epoch','epochs_completed','slice_accuracy','slice_macro_f1',
               'slice_auroc','AD_recall','slice_ECE','slice_coverage','slice_accepted_accuracy',
               'scan_accuracy','patient_accuracy','training_peak_mib','duration_seconds','returncode']
    with (output / 'suite_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def completed_record(case: dict, output: Path, plan: dict) -> dict | None:
    """Accept only a completed result from this exact plan/code/manifest identity."""
    if not (output / 'metrics.json').is_file():
        return None
    result = json.loads((output / 'metrics.json').read_text())
    config = json.loads((output / 'config.json').read_text())
    binding = json.loads((output.parent / (output.name + '.case.json')).read_text())
    if binding != case or config['manifest_sha256'] != plan['manifest_sha256']:
        raise ValueError(f'Completed run has a different case or manifest: {output}')
    if config['code_sha256'] != {k:v for k,v in plan['code_sha256'].items()
                                 if k not in ('run_experiment_suite.py','slurm/submit_suite.sh','slurm/suite.sbatch')}:
        raise ValueError(f'Completed run has different source code: {output}')
    if result['status'] != 'complete' or result['evaluation_role'] != 'development_inner_early_stop':
        raise ValueError(f'Unexpected completed evaluation role: {output}')
    stats = result['metrics']['slice']
    raw = result['coursework_report']['confidence']['slice']
    return dict(id=case['id'], status='complete', output=str(output), best_epoch=result['best_epoch'],
                epochs_completed=result['epochs_completed'], slice_accuracy=stats['accuracy'],
                slice_macro_f1=stats['macro_f1'], slice_auroc=stats['auroc'], AD_recall=stats['per_class']['AD']['recall'],
                slice_ECE=raw['ece_predicted_class'], slice_coverage=raw['fixed_rejection']['coverage'],
                slice_accepted_accuracy=raw['fixed_rejection']['accepted_accuracy'],
                scan_accuracy=result['metrics']['scan']['accuracy'],
                patient_accuracy=result['metrics']['patient']['accuracy'] if result['metrics']['patient'] else None,
                training_peak_mib=result['resources']['training_peak_cuda_allocated_mib'], returncode=0)


def run(args: argparse.Namespace) -> int:
    """Run requested cases serially; preserve incomplete attempts for diagnosis."""
    root = Path(__file__).resolve().parents[1]
    plan = build_plan(args, root)
    if args.list:
        print(json.dumps(plan['cases'], indent=2))
        return 0
    if plan['manifest_sha256'] is None:
        raise ValueError('Frozen split completion marker is missing.')
    output = args.output.resolve()
    for protected in (args.data_root.resolve(), args.splits_dir.resolve(), root):
        if output == protected or output in protected.parents or protected in output.parents:
            # Source tree outputs are deliberately disallowed for the server suite.
            raise ValueError('Suite output must be separate from source, data and manifests.')
    selected = {c['id'] for c in plan['cases']} if not args.ids else set(args.ids.split(','))
    if not selected <= {c['id'] for c in plan['cases']}:
        raise ValueError('Unknown case ID.')
    output.mkdir(parents=True, exist_ok=True)
    plan_file = output / 'suite_plan.json'
    if plan_file.exists():
        if json.loads(plan_file.read_text()) != plan:
            raise ValueError('Existing suite has a different plan/code; use a new output root.')
    else:
        if any(output.iterdir()):
            raise ValueError('Refusing a nonempty folder without a suite plan.')
        write_json(plan_file, plan)
    records = []
    failures = 0
    for case in plan['cases']:
        if case['id'] not in selected:
            continue
        if build_plan(args, root) != plan:
            raise ValueError('Source code or frozen manifest changed during the suite.')
        attempts = sorted({p if p.is_dir() else output / p.name.removesuffix('.case.json')
                           for p in output.glob(case['id'] + '_attempt_*')
                           if p.is_dir() or p.name.endswith('.case.json')})
        done = [r for p in attempts if (r := completed_record(case, p, plan)) is not None]
        if done:
            records.append(done[-1])
            write_summary(output, records)
            print(f"Reuse completed {case['id']}", flush=True)
            continue
        numbers = [int(p.name.rsplit('_', 1)[1]) for p in attempts]
        attempt = output / f"{case['id']}_attempt_{max(numbers, default=0) + 1:02d}"
        # Reserve a sidecar before launch; training still creates its own output directory.
        write_json(output / (attempt.name + '.case.json'), case)
        record = dict(id=case['id'], status='running', output=str(attempt))
        records.append(record)
        write_summary(output, records)
        argv = command(root, case['flags'], attempt)
        write_json(output / (attempt.name + '.command.json'), {'argv':argv})
        print(f"Starting {case['id']}: {' '.join(argv)}", flush=True)
        started = time.perf_counter()
        with (output / (case['id'] + f'_attempt_{max(numbers, default=0) + 1:02d}.log')).open('w') as log:
            process = subprocess.run(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False)
        duration = time.perf_counter() - started
        if attempt.exists():
            write_json(attempt / 'suite_case.json', case)
        if process.returncode == 0:
            finished = completed_record(case, attempt, plan)
            if finished is None:
                raise ValueError('Successful process did not write completed metrics.')
            record.update(finished, duration_seconds=duration)
        else:
            record.update(status='failed', returncode=process.returncode, duration_seconds=duration)
            failures += 1
            print(f"Failed {case['id']}; continuing. Inspect its attempt log.", flush=True)
        write_summary(output, records)
    print(f'Suite finished; failures={failures}; summary={output / "suite_summary.csv"}', flush=True)
    return 1 if failures else 0


def main() -> int:
    """Parse operator paths and bounded suite settings."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/home/groups/comp3710/ADNI'))
    parser.add_argument('--splits-dir', type=Path, default=root.parent / 'adni_splits_v1')
    parser.add_argument('--output', type=Path, default=root.parent / 'runs/convnext_suite')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--ids', help='Optional comma-separated predefined case IDs.')
    parser.add_argument('--list', action='store_true', help='Print the fixed plan without running or writing files.')
    args = parser.parse_args()
    if args.epochs < 1 or args.workers < 0:
        parser.error('Epochs must be positive and workers cannot be negative.')
    try:
        return run(args)
    except (ValueError, OSError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
