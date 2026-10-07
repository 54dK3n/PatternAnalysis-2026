"""Package aggregate development evidence and threshold figures without patient rows."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile
from typing import Any

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from utils.artifacts import write_json

CONFIG_KEYS = ('model_name', 'model_architecture', 'execution_controls', 'training_recipe',
    'fold', 'seed', 'seed_base', 'image_size', 'expected_slices', 'initialization',
    'pretrained_weights', 'augmentation', 'augmentation_config', 'preprocessing_config',
    'preprocessing_crop_fit', 'train_slice_class_counts', 'train_pos_weight', 'epochs_limit',
    'patience', 'min_delta', 'lr', 'weight_decay', 'lr_schedule', 'training_sampling',
    'batch_size', 'workers', 'threads', 'checkpoint_selection', 'primary_evaluation_unit',
    'evaluation_mode', 'manifest_sha256', 'code_sha256', 'environment')


def assessment(summary: dict[str, Any]) -> dict[str, Any]:
    """Distinguish observed inner threshold crossings from unassessed final accuracy."""
    rows = []
    for record in summary['records']:
        metrics = record.get('summary', {}).get('metrics', {})
        rows.append({'case_id': record['case_id'], 'status': record['status'],
            'slice_accuracy': metrics.get('slice', {}).get('accuracy'),
            'scan_accuracy': metrics.get('scan', {}).get('accuracy'),
            'patient_accuracy': (metrics.get('patient') or {}).get('accuracy'),
            'slice_development_reference_met': record['status'] == 'complete'
                and metrics.get('slice', {}).get('accuracy', 0.) >= .8})
    return {'suite_status': summary['status'], 'cases': rows,
        'reference_accuracy': .8, 'reference_unit': 'slice',
        'evaluation_scope': 'inner_checkpoint_selection_cohort_not_independent_validation',
        'final_test_assessed': False, 'course_final_target_met': None,
        'note': 'Development threshold crossings require independent validation; no promotion is automatic.'}


def confusion_figure(case_id: str, scores: dict[str, Any]) -> bytes:
    """Render all available confusion units for an observed inner threshold crossing."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    units = [(unit, scores.get(unit)) for unit in ('slice', 'scan', 'patient') if scores.get(unit)]
    figure, axes = plt.subplots(1, len(units), figsize=(4.5 * len(units), 4), squeeze=False, layout='constrained')
    for axis, (unit, values) in zip(axes[0], units):
        matrix = values['confusion_matrix']
        axis.imshow(matrix, cmap='Blues')
        for i in range(2):
            for j in range(2):
                axis.text(j, i, str(matrix[i][j]), ha='center', va='center')
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['NC', 'AD'], yticklabels=['NC', 'AD'],
                 xlabel='Predicted', ylabel='Actual', title=f'{unit}: accuracy {values["accuracy"]:.3f}')
    figure.suptitle(case_id + '\nSelected inner checkpoint; final test unassessed', fontsize=10)
    stream = io.BytesIO()
    figure.savefig(stream, format='png', dpi=160, bbox_inches='tight')
    plt.close(figure)
    return stream.getvalue()


def package(suite: Path, output: Path, *, allow_partial: bool = False) -> dict[str, Any]:
    """Use a closed aggregate allowlist and bind every exported byte to an inventory."""
    suite, output = suite.resolve(), output.resolve()
    if not suite.is_dir() or output.exists() or output.suffix != '.zip':
        raise ValueError('Require an existing suite and a new .zip output.')
    summary = json.loads((suite / 'summary.json').read_text())
    if summary['status'] not in ('complete', 'completed_with_failures', 'aborted') and not allow_partial:
        raise ValueError('Running suite requires explicit --allow-partial for failure evidence.')
    report = assessment(summary)
    payload: dict[str, bytes] = {}
    for name in ('summary.csv', 'comparison.png', 'trajectories.png'):
        path = suite / name
        if path.is_file() and not path.is_symlink():
            payload[name] = path.read_bytes()
    payload['development_assessment.json'] = (json.dumps(report, indent=2, allow_nan=False) + '\n').encode()
    for record, observed in zip(summary['records'], report['cases']):
        identifier = record['case_id']
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', identifier):
            raise ValueError('Invalid case identifier.')
        if record['status'] != 'complete':
            continue
        case = suite / identifier
        if case.is_symlink() or (case / 'train').is_symlink():
            raise ValueError('Symlink in evidence source.')
        config = json.loads((case / 'train/config.json').read_text())
        public_config = {key: config[key] for key in CONFIG_KEYS if key in config}
        public_config['archive_scope'] = 'aggregate_configuration_without_patient_source_bindings'
        payload[f'{identifier}/resolved_config.json'] = (json.dumps(public_config, indent=2, allow_nan=False) + '\n').encode()
        names = ['metrics.json', 'history.csv', 'epoch_metrics.json', 'learning_curves.png', 'early_stop_confidence.png']
        names.extend(f'early_stop_{unit}_{kind}.csv' for unit in ('slice', 'scan', 'patient')
                     for kind in ('reliability', 'risk_coverage'))
        for name in names:
            path = case / 'train' / name
            if path.is_file() and not path.is_symlink():
                payload[f'{identifier}/train/{name}'] = path.read_bytes()
        verified = record['summary']
        payload[f'{identifier}/verification.json'] = (json.dumps({key: verified.get(key) for key in (
            'status', 'epochs_limit', 'epochs_completed', 'best_epoch', 'prediction_replay',
            'checkpoint_sha256', 'frozen_files_verified', 'execution_controls')}, indent=2, allow_nan=False) + '\n').encode()
        if observed['slice_development_reference_met']:
            payload[f'{identifier}/development_confusion.png'] = confusion_figure(identifier, verified['metrics'])
    manifest = {'schema_version': 1, 'suite_name': suite.name, 'suite_status': summary['status'],
        'archive_scope': 'aggregate_logs_metrics_and_non_MRI_figures',
        'sha256': {name: hashlib.sha256(value).hexdigest() for name, value in sorted(payload.items())},
        'exclusions': ['weights', 'raw_MRI', 'patient_prediction_rows', 'MRI_failure_grids',
                       'full_source_bound_configuration', 'credentials']}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in sorted(payload.items()):
            archive.writestr(name, value)
        archive.writestr('ARCHIVE_MANIFEST.json', json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    result = {'status': 'archive_complete', 'archive': str(output), 'bytes': output.stat().st_size,
        'sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'payload_entries': len(payload),
        'assessment': report}
    write_json(output.with_suffix('.json'), result)
    return result


def main(argv: list[str] | None = None) -> int:
    """Export an already completed suite; never train, score images or upload itself."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(package(args.suite, args.output, allow_partial=args.allow_partial), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
