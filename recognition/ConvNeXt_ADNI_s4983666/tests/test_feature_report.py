"""Regression checks for mixed diagnostic/training report exports and safe recovery."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from evaluation.feature_figures import plot_suite
import recover_feature_report as RECOVERY


class FeatureReportTests(unittest.TestCase):
    """Exercise real plotting with deliberate summary-only diagnostic directories."""

    def synthetic_metrics(self) -> dict:
        """Generate declared synthetic metric inputs, never real performance claims."""
        from evaluation.metrics import binary_metrics
        from evaluation.reporting import confidence_metrics
        labels, probabilities = [0,0,1,1],[.1,.3,.6,.8]
        return {'metrics':{'slice':binary_metrics(labels,probabilities)},
                'coursework_report':{'confidence':{'slice':confidence_metrics(labels,probabilities)}}}

    def test_diagnostics_smokes_and_failed_records_are_excluded(self) -> None:
        """This reproduces the actual default all-phase crash using minimal fixtures."""
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            train=base/'R00'; train.mkdir()
            (train/'metrics.json').write_text(json.dumps(self.synthetic_metrics()))
            diagnostic=base/'D00_features'; diagnostic.mkdir()
            (diagnostic/'summary.json').write_text('{"status":"complete"}')
            records=[{'id':'D00_features','status':'complete','output':str(diagnostic)},
                     {'id':'D00_spatial','status':'complete','output':str(base/'spatial')},
                     {'id':'SMOKE_R03','status':'complete','output':str(base/'smoke')},
                     {'id':'R00','status':'complete','output':str(train)},
                     {'id':'R01','status':'failed','output':str(base/'failed')}]
            plot_suite(base,records)
            self.assertGreater((base/'feature_suite_comparison.png').stat().st_size,1000)

    def test_missing_full_case_metrics_are_not_silently_skipped(self) -> None:
        """A genuine missing training result remains an error instead of a fake plot."""
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            with self.assertRaises(FileNotFoundError):
                plot_suite(base,[{'id':'R00','status':'complete','output':str(base/'missing')}])

    def fixture(self, base: Path) -> Path:
        """Create six validated synthetic historical cases and one summary-only control."""
        suite=base/'suite'; suite.mkdir()
        plan={'data_root':str(base/'data'),'splits_dir':str(base/'splits'),'manifest_sha256':'synthetic',
              'source_sha256':{'synthetic_source':'hash'},'synthetic_smoke':False,'paired_repeats':True}
        (suite/'suite_plan.json').write_text(json.dumps(plan))
        records=[]
        for index in range(6):
            case=f'R{index:02d}'; path=suite/(case+'_attempt_01'); path.mkdir()
            binding={'id':case,'kind':'training','case':case,'seed_base':3710,'smoke':False}
            (suite/(path.name+'.binding.json')).write_text(json.dumps(binding))
            config={'manifest_sha256':'synthetic','code_sha256':plan['source_sha256'],'case_id':case,'seed_base':3710}
            (path/'config.json').write_text(json.dumps(config))
            result=self.synthetic_metrics()|{'status':'complete','run_kind':'fixed_budget','epochs_completed':30,
                'code_sha256':plan['source_sha256'],'manifest_sha256':'synthetic','evaluation_role':'development_inner_early_stop'}
            (path/'metrics.json').write_text(json.dumps(result))
            (path/'best_slice_loss.pt').write_bytes(b'synthetic placeholder not loaded')
            (path/'best_scan_loss.pt').write_bytes(b'synthetic placeholder not loaded')
            records.append({'id':case,'status':'complete','output':str(path)})
        path=suite/'D00_features_attempt_01';path.mkdir()
        (suite/(path.name+'.binding.json')).write_text(json.dumps({'id':'D00_features','kind':'diagnosis'}))
        (path/'config.json').write_text('{"manifest_sha256":"synthetic"}')
        (path/'summary.json').write_text('{"status":"complete"}')
        records.insert(0,{'id':'D00_features','status':'complete','output':str(path)})
        (suite/'suite_records.json').write_text(json.dumps(records))
        (suite/'repeat_gate.json').write_text('{"selected":null,"status":"stop_and_review_diagnostics"}')
        return suite

    def test_recovery_preserves_historical_inputs_and_uses_fresh_destination(self) -> None:
        """Recovery binds old hashes and leaves all original files/checkpoints byte-identical."""
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);suite=self.fixture(base)
            before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in suite.rglob('*') if p.is_file()}
            result=RECOVERY.recover(suite,suite/'report_recovery_v1')
            self.assertEqual(result['full_training_cases'],6)
            self.assertEqual(result['source_sha256'],{'synthetic_source':'hash'})
            self.assertFalse(result['training_launched'])
            self.assertTrue((suite/'report_recovery_v1/feature_suite_comparison.png').is_file())
            self.assertTrue(all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in before.items()))
            with self.assertRaises(ValueError):
                RECOVERY.recover(suite,suite/'report_recovery_v1')

    def test_modified_source_binding_is_rejected_before_output_creation(self) -> None:
        """Report repair cannot relabel results from a different training source."""
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);suite=self.fixture(base)
            path=suite/'R00_attempt_01/config.json'
            config=json.loads(path.read_text());config['code_sha256']={'different':'hash'}
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'historical source'):
                RECOVERY.recover(suite,suite/'report_recovery_v1')
            self.assertFalse((suite/'report_recovery_v1').exists())
