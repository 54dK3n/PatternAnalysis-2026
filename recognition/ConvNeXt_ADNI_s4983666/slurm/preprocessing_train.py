"""Preview, submit and verify configurable scratch inner-development GPU training."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from dataset.augmentation import AUGMENTATION_NAMES, make_augmentation
from dataset.preprocessing import add_preprocessing_arguments, checkpoint_preprocessing, preprocessing_from_args
from dataset.sampling import SAMPLING_NAMES
from engine.scheduling import learning_rate
from models.registry import MODEL_CHOICES
from slurm.deploy_feature_source import frozen_seals
from utils.artifacts import code_fingerprints, validate_output, write_json

# These flags are the public train.py interface; output and device are allocation-owned.
TRAIN_OPTIONS = ('model', 'fold', 'epochs', 'patience', 'min_delta', 'batch_size', 'lr',
                 'weight_decay', 'seed', 'workers', 'threads', 'image_height', 'image_width',
                 'preprocessing', 'foreground_threshold', 'intensity_lower_percentile',
                 'intensity_upper_percentile', 'crop_margin', 'crop_height', 'crop_width',
                 'augmentation', 'rotation_degrees', 'translation_fraction', 'translation_pixels',
                 'sampling', 'lr_schedule', 'warmup_epochs', 'min_lr_ratio')


def parser() -> argparse.ArgumentParser:
    """Expose reproducible training options, with the tested pilot settings as defaults."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--dry-run', action='store_true', help='Print commands only; no jobs, data reads or output writes.')
    result.add_argument('--run', action='store_true', help=argparse.SUPPRESS)
    work = SOURCE.parent
    result.add_argument('--data-root', type=Path, default=Path(os.environ.get('ADNI_DATA_ROOT', '/home/groups/comp3710/ADNI')))
    result.add_argument('--splits-dir', type=Path, default=Path(os.environ.get('ADNI_SPLITS_DIR', str(work / 'adni_splits_v1'))))
    result.add_argument('--runs-root', type=Path, default=Path(os.environ.get('ADNI_RUNS_ROOT', str(work / 'runs'))))
    result.add_argument('--time-limit', default='02:00:00', help='Slurm time limit HH:MM:SS (default: two hours).')
    result.add_argument('--cpus-per-task', type=int, default=4)
    result.add_argument('--model', choices=tuple(MODEL_CHOICES), default='convnext_lite')
    result.add_argument('--fold', type=int, choices=range(1, 6), default=1)
    result.add_argument('--epochs', type=int, default=30, help='Maximum epoch budget, not a course requirement.')
    result.add_argument('--patience', type=int, default=5)
    result.add_argument('--min-delta', type=float, default=0.0001)
    result.add_argument('--batch-size', type=int, default=32)
    result.add_argument('--lr', type=float, default=0.0001)
    result.add_argument('--weight-decay', type=float, default=0.05)
    result.add_argument('--seed', type=int, default=3710, help='Base seed; train.py adds the fold number.')
    result.add_argument('--workers', type=int, default=2)
    result.add_argument('--threads', type=int, default=2)
    result.add_argument('--image-height', type=int, default=240)
    result.add_argument('--image-width', type=int, default=256)
    add_preprocessing_arguments(result)
    result.set_defaults(preprocessing='scan_intensity_crop')
    result.add_argument('--augmentation', choices=AUGMENTATION_NAMES, default='none')
    result.add_argument('--rotation-degrees', type=float, default=5.0)
    result.add_argument('--translation-fraction', type=float, default=0.03)
    result.add_argument('--translation-pixels', type=int, default=4)
    result.add_argument('--sampling', choices=SAMPLING_NAMES, default='slice_uniform')
    result.add_argument('--lr-schedule', choices=('constant', 'warmup_cosine'), default='constant')
    result.add_argument('--warmup-epochs', type=int, default=2)
    result.add_argument('--min-lr-ratio', type=float, default=0.01)
    return result


def validate(args: argparse.Namespace) -> None:
    """Reject invalid controls before submitting a costly allocation."""
    for path in (args.data_root, args.splits_dir, args.runs_root):
        if not path.is_absolute():
            raise ValueError('Data, split and run paths must be absolute.')
    # Validate the parent as well: even dry-run must not plan outputs inside data/manifests/source.
    run_parent = args.runs_root.resolve()
    for protected in (args.data_root.resolve(), args.splits_dir.resolve(), SOURCE):
        if run_parent == protected or protected in run_parent.parents or run_parent in protected.parents:
            raise ValueError('Run storage must be separate from source, data and frozen manifests.')
    if min(args.epochs, args.patience, args.batch_size, args.threads, args.cpus_per_task) < 1 or args.workers < 0:
        raise ValueError('Epochs/patience/batch/threads/CPUs must be positive; workers nonnegative.')
    if args.workers + args.threads > args.cpus_per_task:
        raise ValueError('Request enough CPUs for workers + threads.')
    if not re.fullmatch(r'[0-9]+:[0-5][0-9]:[0-5][0-9]', args.time_limit) or args.time_limit == '00:00:00':
        raise ValueError('Time limit must be a positive HH:MM:SS duration.')
    if not all(math.isfinite(v) for v in (args.lr, args.weight_decay, args.min_delta)):
        raise ValueError('Optimizer values must be finite.')
    if args.lr <= 0 or min(args.weight_decay, args.min_delta) < 0:
        raise ValueError('Learning rate must be positive; weight decay/min delta nonnegative.')
    minimum = 16 if MODEL_CHOICES[args.model] == 'small_cnn_v1' else 32
    if min(args.image_height, args.image_width) < minimum:
        raise ValueError('Image dimensions are below the model minimum.')
    preprocessing_from_args(args)
    make_augmentation(args.augmentation, rotation_degrees=args.rotation_degrees,
                      translation_fraction=args.translation_fraction, translation_pixels=args.translation_pixels)
    learning_rate(1, args.epochs, args.lr, args.lr_schedule, args.warmup_epochs, args.min_lr_ratio)


def options(args: argparse.Namespace) -> list[str]:
    """Serialize effective values exactly once so previews match the actual job."""
    result = ['--data-root', str(args.data_root), '--splits-dir', str(args.splits_dir),
              '--runs-root', str(args.runs_root), '--time-limit', args.time_limit,
              '--cpus-per-task', str(args.cpus_per_task)]
    for name in TRAIN_OPTIONS:
        result.extend(['--' + name.replace('_', '-'), str(getattr(args, name))])
    return result


def commands(args: argparse.Namespace, output: Path) -> list[list[str]]:
    """Build actual train/replay commands with development-only evaluation and CUDA."""
    common = ['--data-root', str(args.data_root), '--splits-dir', str(args.splits_dir)]
    train = [sys.executable, '-u', str(SOURCE / 'train.py'), *common,
             '--output', str(output / 'train'), '--inner-only', '--device', 'cuda']
    for name in TRAIN_OPTIONS:
        train.extend(['--' + name.replace('_', '-'), str(getattr(args, name))])
    replay = [sys.executable, '-u', str(SOURCE / 'predict.py'), *common,
              '--checkpoint', str(output / 'train/best.pt'), '--output', str(output / 'replay'),
              '--role', 'early_stop', '--device', 'cuda', '--skip-inference-profile',
              '--batch-size', str(args.batch_size), '--workers', str(args.workers), '--threads', str(args.threads)]
    return [train, replay]


def verify(output: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Accept early stopping and an earlier best epoch; reject changed checkpoints/predictions."""
    import torch
    config = json.loads((output / 'train/config.json').read_text())
    trained = json.loads((output / 'train/metrics.json').read_text())
    replayed = json.loads((output / 'replay/metrics.json').read_text())
    checkpoint = torch.load(output / 'train/best.pt', map_location='cpu', weights_only=True)
    with (output / 'train/history.csv').open(newline='') as stream:
        history = list(csv.DictReader(stream))
    if trained['status'] != 'complete' or replayed['status'] != 'complete':
        raise ValueError('Training/replay is incomplete.')
    completed, best = trained['epochs_completed'], trained['best_epoch']
    if not (1 <= best <= completed <= args.epochs and completed == len(history)
            and [int(r['epoch']) for r in history] == list(range(1, completed + 1))):
        raise ValueError('Invalid epoch history or budget.')
    if checkpoint['config'] != config or checkpoint['epoch'] != best or replayed['checkpoint_epoch'] != best:
        raise ValueError('Selected checkpoint identity differs from training/replay.')
    if config['model_name'] != MODEL_CHOICES[args.model] or config['initialization'] != 'random' or config['pretrained_weights'] is not None:
        raise ValueError('Model/initialization differs from the requested scratch architecture.')
    if (config['evaluation_mode'] != 'inner_only' or trained['evaluation_role'] != 'development_inner_early_stop'
            or replayed['evaluation_role'] != trained['evaluation_role']):
        raise ValueError('Only inner early-stop evaluation is allowed.')
    if config['epochs_limit'] != args.epochs or config['seed_base'] != args.seed or config['fold'] != args.fold:
        raise ValueError('Requested budget/seed/fold differs from the saved run.')
    if config['manifest_sha256'] != replayed['manifest_sha256'] or config['code_sha256'] != code_fingerprints():
        raise ValueError('Source or frozen manifest identity differs.')
    requested, resolved = preprocessing_from_args(args), checkpoint_preprocessing(config)
    if requested.name != resolved.name or (requested.name != 'none' and any(
            getattr(requested, name) != getattr(resolved, name) for name in
            ('foreground_threshold', 'lower_percentile', 'upper_percentile', 'crop_margin'))):
        raise ValueError('Preprocessing differs from the requested input rule.')
    if requested.crop_height and (requested.crop_height, requested.crop_width) != (resolved.crop_height, resolved.crop_width):
        raise ValueError('Explicit crop geometry differs from the requested dimensions.')
    if any(config[name] != getattr(args, name) for name in
           ('lr', 'weight_decay', 'batch_size', 'patience', 'min_delta', 'workers', 'threads')):
        raise ValueError('Saved optimizer/loader controls differ from the request.')
    if config['augmentation_config'] != make_augmentation(args.augmentation, rotation_degrees=args.rotation_degrees,
            translation_fraction=args.translation_fraction, translation_pixels=args.translation_pixels).to_dict():
        raise ValueError('Saved augmentation differs from the request.')
    if config['training_sampling']['name'] != args.sampling or config['lr_schedule']['name'] != args.lr_schedule:
        raise ValueError('Saved sampler/schedule differs from the request.')
    if config.get('preprocessing_config') != replayed['preprocessing_config']:
        raise ValueError('Replayed preprocessing differs from checkpoint settings.')
    if config['checkpoint_selection'] != 'minimum_early_stop_scan_log_loss':
        raise ValueError('Checkpoint selection policy changed.')
    selected_loss = float(history[best - 1]['early_stop_scan_loss'])
    if checkpoint['early_stop_scan_loss'] != selected_loss or trained['best_early_stop_scan_loss'] != selected_loss:
        raise ValueError('Selected checkpoint loss differs from the epoch history.')
    if selected_loss != min(float(row['early_stop_scan_loss']) for row in history):
        raise ValueError('Checkpoint is not selected by minimum early-stop scan loss.')
    comparison = {}
    for unit in ('slice', 'scan', 'patient'):
        if unit == 'patient' and trained['metrics']['patient'] is None:
            if replayed['metrics']['patient'] is not None:
                raise ValueError('Patient metric availability changed.')
            comparison[unit] = {'status': 'unavailable_mixed_diagnoses'}
            continue
        left = output / 'train' / f'early_stop_{unit}_predictions.csv'
        right = output / 'replay' / ('prediction_patient_predictions.csv' if unit == 'patient' else f'{unit}_predictions.csv')
        if left.read_bytes() != right.read_bytes() or trained['metrics'][unit] != replayed['metrics'][unit]:
            raise ValueError(f'{unit} checkpoint replay differs.')
        with left.open(newline='') as stream:
            comparison[unit] = {'identical_predictions': True, 'rows': sum(1 for _ in csv.DictReader(stream))}
    return {'status': 'complete', 'kind': 'scratch_inner_development_training_not_final_test',
            'epochs_limit': args.epochs, 'epochs_completed': completed, 'best_epoch': best,
            'stopped_before_epoch_cap': completed < args.epochs, 'prediction_replay': comparison,
            'frozen_files_verified': len(frozen_seals(args.splits_dir)),
            'checkpoint_sha256': hashlib.sha256((output / 'train/best.pt').read_bytes()).hexdigest(),
            'image_size': config['image_size'], 'preprocessing_config': config.get('preprocessing_config'),
            'metrics': trained['metrics'], 'resources': trained['resources'],
            'limitations': ['Early-stop patients select and evaluate this model; this is not independent final-test accuracy.',
                            'Preprocessing branches can use different canvas sizes; report geometry/resource differences.']}


def main(argv: list[str] | None = None) -> int:
    """Preview without side effects, submit from login, or execute within one GPU allocation."""
    args = parser().parse_args(argv)
    try:
        validate(args)
        job = os.environ.get('SLURM_JOB_ID') if args.run else '<SLURM_JOB_ID>'
        if args.run and (args.dry_run or not job or not job.isdecimal()):
            raise ValueError('Run mode requires a real Slurm allocation and cannot be combined with dry-run.')
        output = args.runs_root / f'preprocessing_train_{args.preprocessing}_fold{args.fold:02d}_job{job}'
        planned = commands(args, output)
        logs = args.runs_root / 'slurm_logs'
        submit = ['sbatch', '--parsable', '--export=ALL', '--time=' + args.time_limit,
                  '--cpus-per-task=' + str(args.cpus_per_task), '--output=' + str(logs / 'preprocessing_train_%j.out'),
                  '--error=' + str(logs / 'preprocessing_train_%j.err'), str(SOURCE / 'slurm/preprocessing_train.sbatch'),
                  str(SOURCE), *options(args)]
        print('Submission:', shlex.join(submit), flush=True)
        for stage, command in zip(('Train', 'Replay'), planned):
            print(stage + ':', shlex.join(command), flush=True)
        print('Protocol: scratch initialization; inner early-stop selection/evaluation; AdamW + weighted BCE; FP32; no label smoothing.', flush=True)
        if args.dry_run:
            return 0
        if not args.run:
            if not args.data_root.is_dir() or not (args.splits_dir / 'COMPLETED.json').is_file():
                raise ValueError('Data/frozen manifests are missing; inspect paths before submission.')
            logs.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(submit, check=True, capture_output=True, text=True)
            print('Submitted job:', result.stdout.strip(), flush=True)
            return 0
        import torch
        if not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; training was not started.')
        output = validate_output(output, args.data_root, args.splits_dir)
        output.mkdir(parents=True, exist_ok=False)
        before = frozen_seals(args.splits_dir)
        write_json(output / 'launch.json', {'job_id': job, 'node': os.environ.get('SLURMD_NODENAME'),
                   'gpu': torch.cuda.get_device_name(0), 'requested': options(args), 'commands': planned,
                   'code_sha256': code_fingerprints(),
                   'launcher_sha256': {name: hashlib.sha256((SOURCE / 'slurm' / name).read_bytes()).hexdigest()
                                      for name in ('preprocessing_train.py', 'preprocessing_train.sbatch', 'submit_preprocessing_train.sh')}})
        for command in planned:
            subprocess.run(command, cwd=SOURCE, check=True)
        summary = verify(output, args)
        if before != frozen_seals(args.splits_dir):
            raise ValueError('Frozen manifests changed during execution.')
        summary.update(job_id=job, gpu=torch.cuda.get_device_name(0), output=str(output))
        write_json(output / 'summary.json', summary)
        print('Verified training complete:', output, flush=True)
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, subprocess.CalledProcessError) as exc:
        print('ERROR:', exc, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
