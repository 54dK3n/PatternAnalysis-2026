"""Check success dependencies, prospective plans and patient-free result exports."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from slurm import preprocessing_train as single, preprocessing_suite as suite
from slurm.archive_recipe_results import package, assessment


class RecipeOperationsTests(unittest.TestCase):
    """Keep expensive jobs gated and exported evidence inside the aggregate scope."""

    def test_dependency_roundtrip_and_invalid_conditions(self) -> None:
        """Only success dependencies may start the expensive comparison."""
        args = single.parser().parse_args(['--dependency', 'afterok:123:456'])
        single.validate(args)
        forwarded = single.parser().parse_args(single.options(args))
        self.assertEqual(vars(args), vars(forwarded))
        args.dependency = 'afterany:123'
        with self.assertRaises(ValueError):
            single.validate(args)

    def test_fixed_plans_keep_patient_roles_and_exact_budgets(self) -> None:
        """Resolve all GPU smoke/comparison cases through the real launcher parser."""
        for name, cases, epochs in [('recipe_gpu_smoke_20261007', 3, 3),
                                   ('recipe_comparison_20261007', 12, 360)]:
            args = suite.parser().parse_args(['--plan', str(single.SOURCE / 'config' / (name + '.json'))])
            plan = suite.build_plan(args)
            self.assertEqual(len(plan), cases)
            resolved = [single.parser().parse_args(c['options']) for c in plan]
            self.assertEqual(sum(a.epochs for a in resolved), epochs)
            self.assertTrue(all(a.fold == 1 and a.image_height == a.image_width == 224 for a in resolved))

    def test_archive_excludes_patient_rows_weights_and_MRI(self) -> None:
        """Canary files cannot enter the closed evidence allowlist; hashes are exact."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = root / 'suite/C01/train'
            case.mkdir(parents=True)
            (case / 'config.json').write_text(json.dumps({'model_name': 'test', 'data_root': 'PRIVATE',
                'preprocessing_source_parameters': {'patient': 'PRIVATE'}, 'lr': .001}))
            for name in ('early_stop_patient_predictions.csv', 'early_stop_slice_predictions.csv',
                         'early_stop_failures.png', 'best.pt', 'early_stop_failure_cases.json'):
                (case / name).write_bytes(b'PRIVATE')
            (case / 'history.csv').write_text('epoch,loss\n1,0.7\n')
            (case / 'metrics.json').write_text('{"status":"complete"}')
            summary = {'status': 'complete', 'records': [{'case_id': 'C01', 'status': 'complete',
                'summary': {'status': 'complete', 'metrics': {'slice': {'accuracy': .7},
                    'scan': {'accuracy': .9}, 'patient': {'accuracy': .9}}}}]}
            (root / 'suite/summary.json').write_text(json.dumps(summary))
            output = root / 'archive.zip'
            result = package(root / 'suite', output)
            self.assertFalse(result['assessment']['cases'][0]['slice_development_reference_met'])
            self.assertIsNone(result['assessment']['course_final_target_met'])
            with zipfile.ZipFile(output) as archive:
                manifest = json.loads(archive.read('ARCHIVE_MANIFEST.json'))
                for name, digest in manifest['sha256'].items():
                    value = archive.read(name)
                    self.assertNotIn(b'PRIVATE', value)
                    self.assertEqual(hashlib.sha256(value).hexdigest(), digest)
                self.assertFalse(any('prediction' in name or name.endswith('.pt') or 'failures' in name
                                     for name in archive.namelist()))
            with self.assertRaises(ValueError):
                package(root / 'suite', output)

    def test_threshold_crossing_is_development_only(self) -> None:
        """Crossing .80 does not establish the unscored final-test target."""
        value = assessment({'status': 'complete', 'records': [{'case_id': 'C01', 'status': 'complete',
            'summary': {'metrics': {'slice': {'accuracy': .8}, 'scan': {}, 'patient': None}}}]})
        self.assertTrue(value['cases'][0]['slice_development_reference_met'])
        self.assertFalse(value['final_test_assessed'])
        self.assertIsNone(value['course_final_target_met'])


if __name__ == '__main__':
    unittest.main()
