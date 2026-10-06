"""Recover report exports from an immutable completed feature suite, without training."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

from evaluation.feature_figures import plot_suite
from utils.artifacts import validate_output, write_json


TRAINING_ID = re.compile(r'(?:SMOKE_)?R0[0-5](?:_seed(?:4710|5710))?')


def read(path: Path, fingerprints: dict[str, str]) -> Any:
    """Record exactly which historical JSON bytes were used to reconstruct reports."""
    contents = path.read_bytes()
    fingerprints[str(path)] = hashlib.sha256(contents).hexdigest()
    return json.loads(contents)


def recover(suite: Path, output: Path) -> dict:
    """Validate saved bindings and render to a fresh directory; never rewrite run data.

    Historical source/manifest bindings remain unchanged. The original Slurm exit
    status is not repaired retroactively: this marks only postprocessing recovery.
    """
    suite = suite.resolve()
    fingerprints: dict[str, str] = {}
    plan = read(suite / 'suite_plan.json', fingerprints)
    records = read(suite / 'suite_records.json', fingerprints)
    if len({record['id'] for record in records}) != len(records):
        raise ValueError('Duplicate case records.')
    output = validate_output(output, Path(plan['data_root']), Path(plan['splits_dir']))
    source = Path(__file__).resolve().parent
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Recovery output must be separate from project source.')
    training, failed = [], []
    for record in records:
        if record.get('status') != 'complete':
            failed.append(record['id'])
            continue
        path = Path(record['output']).resolve()
        if path.parent != suite:
            raise ValueError('Recorded attempt is outside this suite.')
        binding = read(suite / (path.name + '.binding.json'), fingerprints)
        if binding['id'] != record['id']:
            raise ValueError('Record and historical binding disagree.')
        kind = binding['kind']
        config = read(path / 'config.json', fingerprints)
        if config['manifest_sha256'] != plan['manifest_sha256']:
            raise ValueError('Historical manifest identity differs.')
        if kind != 'training':
            if kind not in ('diagnosis', 'spatial'):
                raise ValueError('Unsupported historical record kind.')
            summary = read(path / 'summary.json', fingerprints)
            if summary['status'] != 'complete':
                raise ValueError('Diagnostic completion marker disagrees.')
            continue
        if TRAINING_ID.fullmatch(record['id']) is None:
            raise ValueError('Unexpected training case identity.')
        result = read(path / 'metrics.json', fingerprints)
        expected_kind = 'synthetic_smoke' if plan['synthetic_smoke'] else 'resource_smoke' if binding['smoke'] else 'fixed_budget'
        expected_epochs = 1 if expected_kind != 'fixed_budget' else 30
        if (config['code_sha256'] != plan['source_sha256'] or result['code_sha256'] != plan['source_sha256']
                or config['case_id'] != binding['case'] or config['seed_base'] != binding['seed_base']
                or result['manifest_sha256'] != plan['manifest_sha256'] or result['status'] != 'complete'
                or result['evaluation_role'] != 'development_inner_early_stop'
                or result['run_kind'] != expected_kind or result['epochs_completed'] != expected_epochs):
            raise ValueError('Training result violates its historical source/case protocol.')
        for filename in ('best_slice_loss.pt', 'best_scan_loss.pt'):
            if not (path / filename).is_file():
                raise ValueError('Selected checkpoint missing; report recovery does not retrain.')
        if not binding['smoke']:
            training.append(record)
    if not training:
        raise ValueError('No completed full training cases to report.')
    gate = read(suite / 'repeat_gate.json', fingerprints)
    expected_ids = {'R00', 'R01', 'R02', 'R03', 'R04', 'R05'}
    if gate['selected'] and plan['paired_repeats']:
        expected_ids.update(f'{case}_seed{seed}' for case in ('R00', gate['selected']) for seed in (4710, 5710))
        paired = read(suite / 'paired_seed_summary.json', fingerprints)
        if paired['candidate'] != gate['selected']:
            raise ValueError('Paired-seed candidate differs from gate.')
        if not failed and paired['complete_pairs'] != 3:
            raise ValueError('Expected three completed paired seeds.')
    observed_ids = {r['id'] for r in training}
    if not failed and observed_ids != expected_ids:
        raise ValueError('Completed training inventory differs from the prospective gate.')
    output.mkdir(parents=True, exist_ok=False)
    # Mix diagnostic/resource/training records intentionally: plotting filters them.
    plot_suite(output, records)
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest for path, digest in fingerprints.items()):
        raise ValueError('Historical inputs changed during recovery.')
    root = Path(__file__).resolve().parent
    reporting_code = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                      for name in ('recover_feature_report.py', 'evaluation/feature_figures.py', 'utils/artifacts.py')}
    summary = {'status': 'partial_failures' if failed else 'complete',
               'report_status': 'recovered_from_saved_results_without_training',
               'original_scheduler_exit_status': 'unchanged_not_inferred_from_report',
               'suite_root': str(suite), 'output': str(output),
               'manifest_sha256': plan['manifest_sha256'], 'source_sha256': plan['source_sha256'],
               'reporting_code_sha256': reporting_code, 'historical_input_sha256': fingerprints,
               'records': records, 'failed_cases': failed, 'full_training_cases': len(training),
               'protected_roles_scored': [], 'training_launched': False}
    write_json(output / 'suite_summary.json', summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Recover summaries only; never submit a job, load model weights or score images."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Fresh output; default is suite-root/report_recovery_v1.')
    args = parser.parse_args(argv)
    try:
        result = recover(args.suite_root, args.output or args.suite_root / 'report_recovery_v1')
    except (ValueError, OSError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    print(json.dumps({key: result[key] for key in ('status','report_status','full_training_cases','output','training_launched')}, indent=2))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
