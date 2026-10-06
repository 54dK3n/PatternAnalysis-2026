"""Fixed-budget inner-only feature experiments with two prospective selectors."""

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch import nn

from dataset import load_fold, make_loader
from dataset.augmentation import make_augmentation
from dataset.sampling import sampling_weights
from engine.diagnosis import collect_features
from engine.feature_controls import (MinimumCheckpoint, gradient_snapshot, matched_model,
                                     optimizer_groups, preserved_rng)
from engine.scheduling import learning_rate
from evaluation.inference import evaluate
from evaluation.metrics import binary_metrics
from evaluation.reporting import prediction_report
from evaluation.resources import profile_inference
from models import count_parameters
from utils.artifacts import code_fingerprints, validate_output, write_csv, write_json
from utils.evaluation_artifacts import export_evaluation_artifacts
from utils.runtime import environment_info, select_device, sync_device


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'config/experiment_plan_v2.json'


def source_identity() -> dict[str, str]:
    """Bind all executable experiment entries and the prospective specification."""
    fingerprints = code_fingerprints()
    for name in ('train_feature_experiment.py', 'run_feature_experiments.py',
                 'diagnose.py', 'diagnose_followup.py', 'config/experiment_plan_v2.json',
                 'slurm/feature_suite.sbatch', 'slurm/submit_feature_suite.sh'):
        fingerprints[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    return fingerprints


def train_epoch(model: nn.Module, loader: Any, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device, epoch: int,
                inspect_gradients: bool) -> tuple[float, dict, str, int, list[dict]]:
    """Train the unchanged slice-uniform stream; diagnostics read after backward."""
    model.train()
    labels_all, probabilities = [], []
    total, count, steps = 0.0, 0, 0
    digest = hashlib.sha256()
    gradients = []
    for batch in loader:
        steps += 1
        digest.update((json.dumps(list(batch['relative_path']), separators=(',', ':')) + '\n').encode())
        images = batch['image'].to(device, non_blocking=device.type == 'cuda')
        labels = batch['label'].to(device, non_blocking=device.type == 'cuda')
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss).item():
            raise ValueError('Non-finite training loss.')
        loss.backward()
        if inspect_gradients and steps <= 10:
            gradients.extend(gradient_snapshot(model, epoch, steps))
        optimizer.step()
        total += loss.detach().item() * labels.numel()
        count += labels.numel()
        labels_all.extend(int(v) for v in labels.detach().cpu().tolist())
        probabilities.extend(logits.detach().sigmoid().cpu().tolist())
    if not count:
        raise ValueError('Empty training stream.')
    return total / count, binary_metrics(labels_all, probabilities), digest.hexdigest(), steps, gradients


def memory_peaks(device: torch.device) -> dict:
    """Report both CUDA allocators, with explicit unmeasured CPU values."""
    return {name: getattr(torch.cuda, 'max_memory_' + name)(device) / 2**20
            if device.type == 'cuda' else None for name in ('allocated', 'reserved')}


def save_state(output: Path, filename: str, config: dict, model: nn.Module,
               epoch: int, scores: dict, selection: str) -> None:
    """Atomically persist a scratch state with an unambiguous selection rule."""
    bound_config = copy.deepcopy(config)
    bound_config['checkpoint_selection'] = selection
    checkpoint = {'config': bound_config, 'epoch': epoch,
                  'early_stop_scan_loss': scores['scan']['log_loss'],
                  'early_stop_slice_loss': scores['slice']['log_loss'],
                  'model_state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    temporary = output / (filename + '.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(output / filename)


def scored(model: nn.Module, loader: Any, device: torch.device, expected: int) -> tuple:
    """Run clean full-coverage scoring without changing training RNG or modes."""
    with preserved_rng(model):
        scores, slices, scans = evaluate(model, loader, device, expected)
        report, patients = prediction_report(slices, scans, 15, .8)
        scores['patient'] = report['patient_metrics']
    return scores, slices, scans, report, patients


def run(args: argparse.Namespace) -> dict:
    """Train one preregistered trajectory; never construct a protected-role loader."""
    plan = json.loads(SPEC.read_text())
    cases = {c['id']: c for c in plan['core_cases']}
    if args.case not in cases:
        raise ValueError('Unknown feature case.')
    settings = copy.deepcopy(cases[args.case]['settings'])
    synthetic = getattr(args, 'synthetic_smoke', False)
    smoke = getattr(args, 'smoke', False) or synthetic
    settings['seed_base'] = args.seed_base
    if args.seed_base not in (3710, 4710, 5710):
        raise ValueError('Only preregistered seed bases are accepted.')
    settings['workers'] = args.workers
    if smoke:
        settings['epochs'] = 1
    if synthetic:
        settings.update(image_size=[32, 32], batch_size=32, workers=0)
    if args.workers < 0:
        raise ValueError('Workers cannot be negative.')
    output = validate_output(args.output, args.data_root, args.splits_dir)
    data = load_fold(args.data_root, args.splits_dir, settings['fold'])
    if synthetic and data['manifest_sha256'] == plan['manifest_sha256']:
        raise ValueError('Synthetic smoke refuses real coursework manifests.')
    if not synthetic and data['manifest_sha256'] != plan['manifest_sha256']:
        raise ValueError('This suite requires the preregistered frozen manifest identity.')
    expected = int(data['report']['config']['expected_slices'])
    for role in ('train', 'early_stop'):
        observed = {'patients': len({r['patient_id'] for r in data[role]}),
                    'scans': len({r['image_id'] for r in data[role]}), 'slices': len(data[role])}
        if not synthetic and observed != plan['cohort'][role]:
            raise ValueError(f'Unexpected {role} cohort: {observed}')
    seed = args.seed_base + settings['fold']
    torch.set_num_threads(settings['threads'])
    device = select_device(args.device)
    model, initialization = matched_model(settings['model_identity'], seed)
    model.to(device)
    groups, group_info = optimizer_groups(model, settings['weight_decay'], settings['weight_decay_policy'])
    optimizer = torch.optim.AdamW(groups, lr=settings['lr'])
    counts = Counter(int(r['label']) for r in data['train'])
    if not counts[0] or not counts[1]:
        raise ValueError('Both training classes are required.')
    pos_weight = counts[0] / counts[1]
    if not synthetic and not math.isclose(pos_weight, settings['bce_pos_weight'], rel_tol=1e-12):
        raise ValueError('Training class weight differs from the specification.')
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    loader_args = (args.data_root, tuple(settings['image_size']), settings['batch_size'], settings['workers'], seed)
    train_loader = make_loader(data['train'], *loader_args, shuffle=True, device=device, role='train')
    # Dedicated diagnostic/evaluation generators never consume the training stream.
    clean_loader = make_loader(data['train'], *loader_args, shuffle=False, device=device, role='train')
    early_loader = make_loader(data['early_stop'], *loader_args, shuffle=False, device=device, role='early_stop')
    _, sampling_config = sampling_weights(data['train'], 'slice_uniform')
    identity = source_identity()
    config = {'checkpoint_format_version': 2, 'metrics_format_version': 3,
              'model_name': settings['model_identity'], 'model_architecture': {
                  'depths': list(model.depths), 'channels': list(model.channels),
                  'stem': settings['stem'], 'max_drop_path': model.max_drop_path,
                  'stage_effective_strides': [model.stem_stride * 2**i for i in range(4)]},
              'case_id': args.case, 'run_kind': 'synthetic_smoke' if synthetic else 'resource_smoke' if smoke else 'fixed_budget',
              'experiment_settings': settings, 'primary_evaluation_unit': 'slice',
              'patient_separation': 'frozen_patient_manifests', 'evaluation_mode': 'inner_only',
              'initialization': 'random', 'pretrained_weights': None, 'initialization_control': initialization,
              'optimizer_parameter_groups': group_info, 'fold': settings['fold'], 'seed': seed,
              'seed_base': args.seed_base, 'image_size': settings['image_size'], 'expected_slices': expected,
              'normalization': settings['normalization'], 'augmentation': 'none',
              'augmentation_config': make_augmentation('none').to_dict(), 'aggregation': 'mean_slice_AD_probability',
              'threshold': .5, 'calibration': 'not_fitted', 'calibration_bins': 15, 'reject_threshold': .8,
              'rejection_threshold_source': 'declared_before_evaluation_not_fitted',
              'checkpoint_selection': 'minimum_early_stop_slice_log_loss',
              'checkpoint_shadow_selection': 'minimum_early_stop_scan_log_loss',
              'stopping': 'fixed_budget_no_early_truncation', 'epochs_limit': settings['epochs'],
              'epoch_budget_source': 'experiment_configuration_not_course_requirement',
              'train_pos_weight': pos_weight, 'training_sampling': sampling_config,
              'manifest_sha256': data['manifest_sha256'], 'split_config': data['report']['config'],
              'code_sha256': identity, 'environment': environment_info(device),
              'data_root': str(args.data_root.resolve()), 'splits_dir': str(args.splits_dir.resolve())}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'config.json', config)
    write_json(output / 'optimizer_parameter_groups.json', group_info)
    selectors = {unit: MinimumCheckpoint() for unit in ('slice', 'scan')}
    history, epoch_metrics, gradients, milestone_stats = [], [], [], []
    peaks = {'training': {'allocated': None, 'reserved': None}, 'evaluation': {'allocated': None, 'reserved': None}}
    timings = {'training': 0.0, 'evaluation': 0.0, 'diagnostics': 0.0, 'exports_and_profiles': 0.0}
    started = time.perf_counter()
    for epoch in range(1, settings['epochs'] + 1):
        if source_identity() != identity:
            raise ValueError('Source changed during the trajectory; refusing mixed-code results.')
        rate = learning_rate(epoch, cases[args.case]['settings']['epochs'], settings['lr'], settings['lr_schedule'],
                             settings.get('warmup_epochs', 2), settings.get('min_lr_ratio', .01))
        for group in optimizer.param_groups:
            group['lr'] = rate
        sync_device(device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.perf_counter()
        loss, online, order_hash, steps, snapshots = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, epoch in plan['gradient_summary']['epochs'])
        sync_device(device)
        train_seconds = time.perf_counter() - epoch_start
        timings['training'] += train_seconds
        current_peak = memory_peaks(device)
        for key, value in current_peak.items():
            if value is not None:
                peaks['training'][key] = max(peaks['training'][key] or 0, value)
        gradients.extend(snapshots)
        if gradients:
            write_csv(output / 'gradient_batches.csv', gradients)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        phase = time.perf_counter()
        early, _, _, _, _ = scored(model, early_loader, device, expected)
        sync_device(device)
        eval_seconds = time.perf_counter() - phase
        timings['evaluation'] += eval_seconds
        current_peak = memory_peaks(device)
        for key, value in current_peak.items():
            if value is not None:
                peaks['evaluation'][key] = max(peaks['evaluation'][key] or 0, value)
        selections = {}
        for unit, selector in selectors.items():
            selections[unit] = selector.update(early[unit]['log_loss'], epoch)
            if selections[unit]:
                save_state(output, f'best_{unit}_loss.pt', config, model, epoch, early,
                           f'minimum_early_stop_{unit}_log_loss')
        clean = None
        clean_seconds = diagnostic_seconds = 0.0
        if epoch in plan['milestone_epochs'] or smoke:
            save_state(output, f'epoch_{epoch:02d}.pt', config, model, epoch, early, 'declared_fixed_epoch')
            phase = time.perf_counter()
            clean, _, _, _, _ = scored(model, clean_loader, device, expected)
            sync_device(device)
            clean_seconds = time.perf_counter() - phase
            timings['evaluation'] += clean_seconds
            phase = time.perf_counter()
            with preserved_rng(model):
                features, _, statistics, _ = collect_features(model, clean_loader, device, shift_pixels=0)
                del features
            sync_device(device)
            diagnostic_seconds = time.perf_counter() - phase
            timings['diagnostics'] += diagnostic_seconds
            milestone_stats.append({'epoch': epoch, 'role': 'train_clean', 'statistics': statistics})
            write_json(output / 'milestone_features.json', milestone_stats)
        history.append({'epoch': epoch, 'learning_rate': rate, 'train_weighted_bce': loss,
                        'train_online_slice_accuracy': online['accuracy'], 'train_online_slice_auroc': online['auroc'],
                        'early_slice_loss': early['slice']['log_loss'], 'early_scan_loss': early['scan']['log_loss'],
                        'early_slice_accuracy': early['slice']['accuracy'], 'early_slice_auroc': early['slice']['auroc'],
                        'clean_train_slice_accuracy': clean['slice']['accuracy'] if clean else None,
                        'clean_train_slice_loss': clean['slice']['log_loss'] if clean else None,
                        'selected_slice': selections['slice'], 'selected_scan': selections['scan'],
                        'training_batch_order_sha256': order_hash, 'optimizer_steps': steps,
                        'train_seconds': train_seconds, 'early_evaluation_seconds': eval_seconds,
                        'clean_train_evaluation_seconds': clean_seconds, 'diagnostics_seconds': diagnostic_seconds})
        epoch_metrics.append({'epoch': epoch, 'train_online_slice': online,
                              'early_stop': early, 'train_clean': clean, 'learning_rate': rate})
        write_json(output / 'epoch_metrics.json', epoch_metrics)
        write_csv(output / 'history.csv', history)
        print(f"{args.case} epoch {epoch:02d}/{settings['epochs']}: slice loss={early['slice']['log_loss']:.5f}, "
              f"scan loss={early['scan']['log_loss']:.5f}, order={order_hash[:12]}", flush=True)
    del optimizer, groups
    model.zero_grad(set_to_none=True)
    candidates = {}
    candidate_predictions = {}
    for unit, selector in selectors.items():
        checkpoint = torch.load(output / f'best_{unit}_loss.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model_state'])
        phase = time.perf_counter()
        scores, slices, scans, report, patients = scored(model, early_loader, device, expected)
        sync_device(device)
        timings['evaluation'] += time.perf_counter() - phase
        phase = time.perf_counter()
        candidate_dir = output / ('primary_slice' if unit == 'slice' else 'shadow_scan')
        candidate_dir.mkdir()
        write_json(candidate_dir / 'config.json', checkpoint['config'])
        write_csv(candidate_dir / 'early_stop_slice_predictions.csv', slices)
        write_csv(candidate_dir / 'early_stop_scan_predictions.csv', scans)
        failures = export_evaluation_artifacts(candidate_dir, report, slices, patients, data['early_stop'],
                                               args.data_root, tuple(settings['image_size']), 'early_stop')
        inference = profile_inference(model, tuple(settings['image_size']), device,
                                      enabled=not args.skip_inference_profile)
        result = {'status': 'complete', 'selection': checkpoint['config']['checkpoint_selection'],
                  'best_epoch': selector.epoch, 'metrics': scores, 'coursework_report': report,
                  'failure_examples': failures, 'inference_profile': inference}
        write_json(candidate_dir / 'metrics.json', result)
        candidates[unit] = result
        candidate_predictions[unit] = slices
        timings['exports_and_profiles'] += time.perf_counter() - phase
    result = {'status': 'complete', 'metrics_format_version': 3, 'run_kind': config['run_kind'],
              'evaluation_role': 'development_inner_early_stop', 'evaluation_reuses_checkpoint_selection_patients': True,
              'case_id': args.case, 'model_name': config['model_name'], 'seed_base': args.seed_base, 'seed': seed,
              'epochs_completed': len(history), 'best_epoch': selectors['slice'].epoch,
              'manifest_sha256': data['manifest_sha256'], 'code_sha256': identity,
              'metrics': candidates['slice']['metrics'], 'coursework_report': candidates['slice']['coursework_report'],
              'candidate_selections': candidates, 'resources': {
                  'trainable_parameters': count_parameters(model), 'training_seconds': timings['training'],
                  'evaluation_seconds': timings['evaluation'], 'diagnostics_seconds': timings['diagnostics'],
                  'candidate_exports_and_profiles_seconds': timings['exports_and_profiles'],
                  'total_seconds': time.perf_counter() - started,
                  'training_peak_cuda_allocated_mib': peaks['training']['allocated'],
                  'training_peak_cuda_reserved_mib': peaks['training']['reserved'],
                  'early_evaluation_peak_cuda_allocated_mib': peaks['evaluation']['allocated'],
                  'early_evaluation_peak_cuda_reserved_mib': peaks['evaluation']['reserved'],
                  'training_memory_scope': 'includes_optimizer_and_read_only_gradient_instrumentation',
                  'inference_profile': candidates['slice']['inference_profile'],
                  'device': config['environment']['device_name']}}
    from evaluation.feature_figures import plot_trajectory, compare_selections
    plot_trajectory(output, history, milestone_stats, gradients)
    result['selection_comparison'] = compare_selections(output, candidate_predictions)
    # A completed marker is written only after both selected states are exported.
    if source_identity() != identity:
        raise ValueError('Source changed before completion.')
    write_json(output / 'metrics.json', result)
    return result


def main(argv: list[str] | None = None) -> int:
    """Parse a case ID without changing the legacy training command or defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/home/groups/comp3710/ADNI'))
    parser.add_argument('--splits-dir', type=Path, default=ROOT.parent / 'adni_splits_v1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=('R00', 'R01', 'R02', 'R03', 'R04', 'R05'), required=True)
    parser.add_argument('--seed-base', type=int, default=3710)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--smoke', action='store_true', help='One native-size resource epoch; not eligible for the gate.')
    parser.add_argument('--synthetic-smoke', action='store_true', help='One 32x32 epoch on a different synthetic manifest; never real evidence.')
    parser.add_argument('--skip-inference-profile', action='store_true')
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0
