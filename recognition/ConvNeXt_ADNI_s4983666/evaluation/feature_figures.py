"""Standalone factual figures for feature-control experiments, without clinical claims."""

from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import random
import statistics
from typing import Any

from utils.artifacts import write_csv, write_json


def pyplot(output: Path) -> Any:
    """Use a noninteractive backend and keep all generated files in the run folder."""
    os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_trajectory(output: Path, history: list[dict], milestones: list[dict], gradients: list[dict]) -> None:
    """Separate weighted online BCE from comparable unweighted clean log loss."""
    plt = pyplot(output)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    epochs = [r['epoch'] for r in history]
    axes[0, 0].plot(epochs, [r['learning_rate'] for r in history])
    axes[0, 0].set(title='Declared LR trajectory', ylabel='Learning rate')
    for key, label in [('early_slice_loss', 'Early-stop slice'), ('early_scan_loss', 'Early-stop scan')]:
        axes[0, 1].plot(epochs, [r[key] for r in history], label=label)
    clean = [r for r in history if r['clean_train_slice_loss'] is not None]
    axes[0, 1].plot([r['epoch'] for r in clean], [r['clean_train_slice_loss'] for r in clean], 'o--', label='Clean train slice')
    axes[0, 1].set(title='Unweighted log loss (fixed weights per epoch)', ylabel='Log loss')
    for key, label in [('train_online_slice_accuracy', 'Online train'), ('early_slice_accuracy', 'Early-stop')]:
        axes[1, 0].plot(epochs, [r[key] for r in history], label=label)
    axes[1, 0].plot([r['epoch'] for r in clean], [r['clean_train_slice_accuracy'] for r in clean], 'o--', label='Clean train')
    axes[1, 0].set(title='Slice accuracy at full coverage', ylabel='Accuracy', ylim=(0, 1))
    axes[1, 1].plot(epochs, [r['train_weighted_bce'] for r in history], label='Online weighted BCE')
    axes[1, 1].set(title='Training objective (separate loss scale)', ylabel='Class-weighted BCE')
    for row in history:
        for key, color in [('selected_slice', 'tab:blue'), ('selected_scan', 'tab:orange')]:
            if row[key]:
                axes[0, 1].scatter(row['epoch'], row['early_slice_loss' if key == 'selected_slice' else 'early_scan_loss'], color=color, s=12)
    for axis in axes.flat:
        axis.set_xlabel('Epoch')
        axis.grid(alpha=.2)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
    figure.savefig(output / 'feature_learning_curves.png', dpi=160)
    plt.close(figure)
    if milestones:
        blocks = list(milestones[0]['statistics']['residual_branches'])
        if blocks:
            figure, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')
            for axis, key, title in zip(axes, ('mean_residual_to_skip_norm', 'layer_scale'),
                                        ('Post-LayerScale residual / skip', 'Mean absolute LayerScale')):
                matrix = []
                for block in blocks:
                    values = []
                    for row in milestones:
                        item = row['statistics']['residual_branches'][block]
                        value = item[key]['mean_absolute'] if key == 'layer_scale' else item[key]
                        values.append(math.log10(max(value, 1e-12)))
                    matrix.append(values)
                heat = axis.imshow(matrix, aspect='auto', cmap='viridis')
                axis.set_xticks(range(len(milestones)), [m['epoch'] for m in milestones])
                axis.set_yticks(range(len(blocks)), blocks, fontsize=7)
                axis.set(title=title, xlabel='Clean-train milestone epoch')
                figure.colorbar(heat, ax=axis, label='log10(value), floor 1e-12')
            figure.savefig(output / 'residual_layerscale.png', dpi=160)
            plt.close(figure)
    if gradients:
        grouped: dict[tuple, list[float]] = defaultdict(list)
        for row in gradients:
            if row['group'].endswith(('depthwise', 'mlp', 'layer_scale')) or row['group'] == 'head':
                grouped[row['epoch'], row['group']].append(row['gradient_to_parameter_l2'])
        epochs = sorted({k[0] for k in grouped})
        groups = sorted({k[1] for k in grouped})
        matrix = [[math.log10(max(statistics.mean(grouped[e, g]), 1e-12)) for e in epochs] for g in groups]
        figure, axis = plt.subplots(figsize=(10, 9), layout='constrained')
        heat = axis.imshow(matrix, aspect='auto', cmap='magma')
        axis.set_xticks(range(len(epochs)), epochs)
        axis.set_yticks(range(len(groups)), groups, fontsize=6)
        axis.set(title='Gradient / parameter L2: first 10 training batches', xlabel='Epoch')
        figure.colorbar(heat, ax=axis, label='log10(mean ratio), floor 1e-12; counters in CSV')
        figure.savefig(output / 'gradient_flow.png', dpi=160)
        plt.close(figure)


def compare_selections(output: Path, predictions: dict[str, list[dict]]) -> dict:
    """Pair both selectors on identical slices and export patient-cluster errors."""
    first = {r['relative_path']: r for r in predictions['slice']}
    second = {r['relative_path']: r for r in predictions['scan']}
    if set(first) != set(second):
        raise ValueError('Selection comparison cohorts differ.')
    patients: dict[str, dict] = {}
    for path, row in first.items():
        other = second[path]
        if any(row[k] != other[k] for k in ('patient_id', 'image_id', 'slice_index', 'label')):
            raise ValueError('Selection comparison identities differ.')
        item = patients.setdefault(row['patient_id'], {'patient_id': row['patient_id'], 'slices': 0,
                    'primary_errors': 0, 'shadow_errors': 0, 'primary_only_errors': 0,
                    'shadow_only_errors': 0, 'both_errors': 0})
        a = int((row['probability'] >= .5) != row['label'])
        b = int((other['probability'] >= .5) != row['label'])
        item['slices'] += 1
        item['primary_errors'] += a
        item['shadow_errors'] += b
        item['primary_only_errors'] += int(a and not b)
        item['shadow_only_errors'] += int(b and not a)
        item['both_errors'] += int(a and b)
    rows = []
    for patient in sorted(patients):
        item = patients[patient]
        item['primary_minus_shadow_accuracy'] = (item['shadow_errors'] - item['primary_errors']) / item['slices']
        rows.append(item)
    write_csv(output / 'selection_patient_error_comparison.csv', rows)
    plt = pyplot(output)
    figure, axis = plt.subplots(figsize=(12, 4), layout='constrained')
    axis.bar(range(len(rows)), [r['primary_minus_shadow_accuracy'] for r in rows])
    axis.axhline(0, color='black', linewidth=.5)
    axis.set(title='Same trajectory, two checkpoint rules, same patients', xlabel='Patient (CSV order)',
             ylabel='Primary minus shadow slice accuracy')
    figure.savefig(output / 'selection_comparison.png', dpi=160)
    plt.close(figure)
    result = {'cohort': 'same_early_stop_slices', 'patients': len(rows), 'slices': len(first),
              'primary_only_errors': sum(r['primary_only_errors'] for r in rows),
              'shadow_only_errors': sum(r['shadow_only_errors'] for r in rows),
              'both_errors': sum(r['both_errors'] for r in rows),
              'mean_patient_accuracy_difference': statistics.mean(r['primary_minus_shadow_accuracy'] for r in rows),
              'not_independent_evaluation': True}
    write_json(output / 'selection_comparison.json', result)
    return result


def plot_probe_summary(output: Path) -> None:
    """Plot completed train-only probe results, retaining their diagnostic units."""
    path = output / 'probes/feature_probes.json'
    if not path.is_file():
        return
    data = json.loads(path.read_text())
    plt = pyplot(output)
    stages = data['stages']
    figure, axis = plt.subplots(figsize=(9, 4), layout='constrained')
    for role in ('train', 'early_stop'):
        values = [item[role]['auroc'] for item in stages.values()]
        axis.plot(list(stages), values, 'o-', label=role)
    axis.set(title='Frozen GAP patient-mean linear probes (diagnostic only)', ylabel='Patient probe AUROC', ylim=(0, 1))
    axis.legend()
    figure.savefig(output / 'stage_probe_auroc.png', dpi=160)
    plt.close(figure)


def plot_suite(output: Path, records: list[dict]) -> None:
    """Show paired seeds descriptively without treating slices as independent trials."""
    completed = [r for r in records if r.get('status') == 'complete' and r['id'].startswith('R')]
    if not completed:
        return
    plt = pyplot(output)
    figure, axes = plt.subplots(1, 4, figsize=(16, 4), layout='constrained')
    for index, key in enumerate(('accuracy', 'auroc', 'AD_recall', 'ECE')):
        labels, values = [], []
        for row in completed:
            result = json.loads((Path(row['output']) / 'metrics.json').read_text())
            labels.append(row['id'])
            metric = result['metrics']['slice']
            values.append(metric['per_class']['AD']['recall'] if key == 'AD_recall' else
                          result['coursework_report']['confidence']['slice']['ece_predicted_class'] if key == 'ECE' else metric[key])
        axes[index].plot(range(len(labels)), values, 'o')
        axes[index].set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        axes[index].set(title=key, ylim=(0, 1))
    figure.suptitle('Primary-selected inner slice results: descriptive, full coverage')
    figure.savefig(output / 'feature_suite_comparison.png', dpi=160)
    plt.close(figure)


def paired_summary(output: Path, records: list[dict], selected: str) -> dict:
    """Export per-seed candidate-reference differences and conditional patient intervals."""
    results = {r['id']: Path(r['output']) for r in records if r.get('status') == 'complete'}
    pairs = []
    for seed in (3710, 4710, 5710):
        a_id = 'R00' if seed == 3710 else f'R00_seed{seed}'
        b_id = selected if seed == 3710 else f'{selected}_seed{seed}'
        if a_id not in results or b_id not in results:
            continue
        a = json.loads((results[a_id] / 'metrics.json').read_text())
        b = json.loads((results[b_id] / 'metrics.json').read_text())
        row = {'seed_base': seed}
        for key in ('accuracy', 'auroc', 'AD_recall', 'ECE', 'errors'):
            def value(result: dict) -> float:
                """Extract a declared full-coverage slice statistic."""
                metric = result['metrics']['slice']
                if key == 'AD_recall':
                    return metric['per_class']['AD']['recall']
                if key == 'ECE':
                    return result['coursework_report']['confidence']['slice']['ece_predicted_class']
                if key == 'errors':
                    cm = metric['confusion_matrix']
                    return cm[0][1] + cm[1][0]
                return metric[key]
            row[key + '_difference'] = value(b) - value(a)
        grouped: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        with (results[a_id] / 'primary_slice/early_stop_slice_predictions.csv').open() as stream:
            first = {r['relative_path']: r for r in csv.DictReader(stream)}
        with (results[b_id] / 'primary_slice/early_stop_slice_predictions.csv').open() as stream:
            second = {r['relative_path']: r for r in csv.DictReader(stream)}
        if set(first) != set(second):
            raise ValueError('Paired-seed cohorts differ.')
        for path, left in first.items():
            right = second[path]
            if left['label'] != right['label'] or left['patient_id'] != right['patient_id']:
                raise ValueError('Paired-seed identities disagree.')
            count = grouped[left['patient_id']]
            count[0] += int((float(right['probability']) >= .5) == int(right['label'])) - int((float(left['probability']) >= .5) == int(left['label']))
            count[1] += 1
        rng = random.Random(3710 + seed)
        values = list(grouped.values())
        samples = []
        for _ in range(1000):
            draws = [rng.choice(values) for _ in values]
            samples.append(sum(v[0] for v in draws) / sum(v[1] for v in draws))
        samples.sort()
        row['accuracy_cluster_low_conditional'] = samples[24]
        row['accuracy_cluster_high_conditional'] = samples[974]
        pairs.append(row)
    if pairs:
        write_csv(output / 'paired_seed_differences.csv', pairs)
    result = {'candidate': selected, 'complete_pairs': len(pairs), 'pairs': pairs,
              'interpretation': 'Descriptive paired seeds; intervals conditional on fitted models and this development cohort; no p-values.',
              'differences': {key: {'mean': statistics.mean(r[key] for r in pairs),
                                   'sample_sd': statistics.stdev(r[key] for r in pairs) if len(pairs) > 1 else None}
                              for key in pairs[0] if key.endswith('_difference')} if pairs else {}}
    write_json(output / 'paired_seed_summary.json', result)
    return result


def plot_spatial_features(output: Path) -> None:
    """Render equal-patient feature changes; gray cells mean undefined alignment.

    The source CSV retains reference norms and zero/valid counts. No feature
    interpolation is introduced by plotting non-stride-divisible shifts.
    """
    path = output / 'stage_patient_means.csv'
    if not path.is_file():
        return
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        for key in ('gap_absolute_l2', 'gap_relative_l2', 'aligned_relative_l2', 'reference_gap_l2', 'reference_gap_zero'):
            if row.get(key) not in (None, ''):
                grouped[row['layer'], int(row['dx']), int(row['dy']), key].append(float(row[key]))
    shifts = sorted({(int(r['dx']), int(r['dy'])) for r in rows})
    layers = list(dict.fromkeys(r['layer'] for r in rows))
    plt = pyplot(output)
    figure, axes = plt.subplots(5, 1, figsize=(14, 15), layout='constrained')
    for axis, key in zip(axes, ('gap_absolute_l2', 'gap_relative_l2', 'aligned_relative_l2', 'reference_gap_l2', 'reference_gap_zero')):
        matrix = [[statistics.mean(grouped[layer, dx, dy, key]) if grouped[layer, dx, dy, key] else math.nan
                   for dx, dy in shifts] for layer in layers]
        import numpy as np
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad('lightgray')
        heat = axis.imshow(np.ma.masked_invalid(matrix), aspect='auto', cmap=cmap)
        axis.set_xticks(range(len(shifts)), [f'{x},{y}' for x,y in shifts], rotation=90, fontsize=6)
        axis.set_yticks(range(len(layers)), layers, fontsize=7)
        axis.set(title=key + ' (equal patient means)', xlabel='Exact input displacement dx,dy')
        figure.colorbar(heat, ax=axis)
    figure.savefig(output / 'stage_feature_changes.png', dpi=160)
    plt.close(figure)
