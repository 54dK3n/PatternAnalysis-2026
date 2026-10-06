"""Synthetic invariants for matched initializations and the prospective control suite."""

import argparse
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import random
import subprocess
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from dataset import make_loader, load_fold
from engine import feature_training, feature_suite, prediction
from engine.feature_controls import (MinimumCheckpoint, eligible_repeat, gradient_snapshot,
                                     matched_model, optimizer_groups, preserved_rng, state_digest)
from engine.diagnosis import _validate_checkpoint
from engine.scheduling import learning_rate
from evaluation.shift_diagnostics import spatial_observers, aligned_feature_error
from models import create_model, count_parameters
from models.registry import validate_feature_architecture
import test_adni_splits as fixtures


class FeatureControlTests(unittest.TestCase):
    """Check scientific comparison invariants before any real GPU experiments."""

    def setUp(self) -> None:
        """Keep synthetic CPU checks small."""
        torch.set_num_threads(2)

    def test_old_initialization_and_alias_meaning_are_preserved(self) -> None:
        """Reference construction still has the historical shapes/count and random stream."""
        torch.manual_seed(3711)
        expected = create_model('convnext_lite_v1').state_dict()
        model, control = matched_model('convnext_lite_v1', 3711)
        self.assertEqual(control['initial_state_sha256'], state_digest(expected))
        self.assertEqual(count_parameters(model), 4831633)
        self.assertEqual(control['independent_tensor_names'], [])
        self.assertEqual(model.stem[0].kernel_size, (4, 4))
        self.assertEqual(model.stem[0].stride, (4, 4))

    def test_variants_share_untrained_tensors_and_training_rng(self) -> None:
        """Stem shape changes must not perturb nonstem initialization or DropPath draws."""
        reference, info = matched_model('convnext_lite_v1', 3711)
        next_random = torch.rand(4)
        for name in ('convnext_lite_overlap_v1', 'convnext_lite_stride2_v1', 'convnext_lite_nodrop_v1'):
            model, control = matched_model(name, 3711)
            self.assertTrue(torch.equal(next_random, torch.rand(4)))
            self.assertEqual(control['reference_initial_sha256'], info['reference_initial_sha256'])
            for key in control['shared_tensor_names']:
                self.assertTrue(torch.equal(reference.state_dict()[key], model.state_dict()[key]), key)
            self.assertEqual(control['independent_tensor_names'], ['stem.0.weight'] if 'overlap' in name else [])
        other, other_info = matched_model('convnext_lite_v1', 4711)
        self.assertNotEqual(info['initial_state_sha256'], other_info['initial_state_sha256'])

    def test_geometry_observers_and_counts_follow_actual_stem(self) -> None:
        """Native-size maps and exact alignment differ correctly for a stride-two stem."""
        shapes = {'convnext_lite_v1': [(60,64),(30,32),(15,16),(7,8)],
                  'convnext_lite_overlap_v1': [(60,64),(30,32),(15,16),(7,8)],
                  'convnext_lite_stride2_v1': [(120,128),(60,64),(30,32),(15,16)],
                  'convnext_lite_nodrop_v1': [(60,64),(30,32),(15,16),(7,8)]}
        for name, expected in shapes.items():
            model = create_model(name)
            seen, handles = [], []
            for stage in model.stages:
                handles.append(stage.register_forward_hook(lambda m, x, y: seen.append(tuple(y.shape[-2:]))))
            try:
                with torch.inference_mode():
                    self.assertEqual(tuple(model(torch.zeros(1,1,240,256)).shape), (1,))
            finally:
                for handle in handles:
                    handle.remove()
            self.assertEqual(seen, expected)
            self.assertEqual(count_parameters(model), 4833217 if 'overlap' in name else 4831633)
            stem = 2 if 'stride2' in name else 4
            self.assertEqual([spatial_observers(model)[f'stage_{i+1}'][1] for i in range(4)], [stem*2**i for i in range(4)])
        tensor = torch.ones(1,2,4,4)
        self.assertIsNone(aligned_feature_error(tensor,tensor,1,0,2)[0])
        self.assertIsNotNone(aligned_feature_error(tensor,tensor,2,0,2)[0])

    def test_decay_groups_cover_every_parameter_once(self) -> None:
        """Norm/bias/LayerScale exclusions never omit the head or another parameter."""
        model = create_model('convnext_lite_v1')
        groups, metadata = optimizer_groups(model,.05,'zero_for_1d_parameters')
        self.assertEqual(sum(len(g['params']) for g in groups),len(list(model.parameters())))
        self.assertEqual(len({id(p) for g in groups for p in g['params']}),len(list(model.parameters())))
        decayed = next(row['parameter_names'] for row in metadata if row['weight_decay'] == .05)
        exempt = next(row['parameter_names'] for row in metadata if row['weight_decay'] == 0)
        self.assertIn('classifier.weight',decayed)
        self.assertIn('classifier.bias',exempt)
        self.assertIn('stages.0.0.layer_scale',exempt)
        groups, _ = optimizer_groups(model,.05,'all_trainable_parameters')
        self.assertEqual(len(groups),1)

    def test_rng_modes_and_state_restore_after_diagnostic_failure(self) -> None:
        """Failed diagnostics may not consume Python/tensor RNG or change mixed modes."""
        model = create_model('convnext_lite_v1')
        model.train()
        model.stages[0].eval()
        flags = [m.training for m in model.modules()]
        state = state_digest(model.state_dict())
        random.seed(98)
        torch.manual_seed(98)
        python_state, torch_state = random.getstate(), torch.get_rng_state()
        with self.assertRaisesRegex(ValueError,'diagnostic'):
            with preserved_rng(model):
                model.eval()
                random.random()
                torch.rand(6)
                raise ValueError('diagnostic')
        self.assertEqual(python_state,random.getstate())
        self.assertTrue(torch.equal(torch_state,torch.get_rng_state()))
        self.assertEqual(flags,[m.training for m in model.modules()])
        self.assertEqual(state,state_digest(model.state_dict()))

    def test_gradient_observation_does_not_change_an_optimizer_update(self) -> None:
        """Instrumentation must neither clip gradients nor consume training randomness."""
        model, _ = matched_model('convnext_lite_v1',3711)
        other = copy.deepcopy(model)
        images = torch.linspace(-1,1,2*32*32).reshape(2,1,32,32)
        expected_random = None
        for network, inspect in ((model,False),(other,True)):
            torch.manual_seed(3711)
            optimizer = torch.optim.AdamW(network.parameters(),lr=.0001,weight_decay=.05)
            nn.BCEWithLogitsLoss()(network(images),torch.tensor([0.,1.])).backward()
            if inspect:
                before = [p.grad.clone() for p in network.parameters()]
                stats = gradient_snapshot(network,1,1)
                self.assertIn('head',{r['group'] for r in stats})
                self.assertTrue(all(torch.equal(p.grad,g) for p,g in zip(network.parameters(),before)))
            optimizer.step()
            if expected_random is None:
                expected_random = torch.rand(3)
            else:
                self.assertTrue(torch.equal(expected_random,torch.rand(3)))
        self.assertEqual(state_digest(model.state_dict()),state_digest(other.state_dict()))

    def test_dual_strict_minima_can_select_different_epochs_and_ties_keep_first(self) -> None:
        """Slice and scan objectives have independent prospective selectors."""
        primary, shadow = MinimumCheckpoint(), MinimumCheckpoint()
        for epoch, (a,b) in enumerate(((.7,.6),(.5,.7),(.5,.4),(.6,.4)),1):
            primary.update(a,epoch)
            shadow.update(b,epoch)
        self.assertEqual((primary.epoch,shadow.epoch),(2,3))
        self.assertTrue(abs(learning_rate(30,30,1e-4,'warmup_cosine',2,.01)-1e-6)<1e-15)

    def test_repeat_gate_requires_all_cases_and_never_promotes_least_bad(self) -> None:
        """Engineering eligibility uses all three guardrails and deterministic ranking."""
        def result(acc: float, auc: float, recall: float, peak: float) -> dict:
            """Create an explicitly synthetic gate example."""
            return {'metrics':{'slice':{'accuracy':acc,'auroc':auc,'per_class':{'AD':{'recall':recall}}}},
                    'resources':{'training_peak_cuda_allocated_mib':peak}}
        records = {f'R{i:02d}':result(.6,.65,.5,1000) for i in range(6)}
        self.assertIsNone(eligible_repeat(records)['selected'])
        records['R01'] = result(.7,.64,.5,900)
        records['R02'] = result(.7,.7,.47,900)
        records['R03'] = result(.62,.65,.48,1100)
        records['R04'] = result(.62,.65,.48,1000)
        self.assertEqual(eligible_repeat(records)['selected'],'R04')
        self.assertEqual(eligible_repeat({k:v for k,v in records.items() if k!='R05'})['status'],'blocked_incomplete_core')


class FeatureIntegrationTests(unittest.TestCase):
    """Use real synthetic training, strict source binding and checkpoint reloads."""

    def fixture(self) -> fixtures.ADNISplitTests:
        """Prepare fresh disjoint synthetic patients without mixed diagnoses."""
        fixture = fixtures.ADNISplitTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        for relative, row in list(fixture.expected.items()):
            if row['patient_id'] in fixture.mixed_patients:
                (fixture.root/relative).unlink()
                fixture.metadata.pop(row['image_id'],None)
                del fixture.expected[relative]
        fixture.save_metadata()
        fixture.prepare()
        return fixture

    def test_real_cpu_training_dual_export_reload_and_no_protected_scoring(self) -> None:
        """Train new geometry, reload both states and audit clean diagnostics isolation."""
        fixture = self.fixture()
        args = argparse.Namespace(data_root=fixture.root,splits_dir=fixture.out,output=fixture.base/'feature',
            case='R03',seed_base=3710,workers=0,device='cpu',synthetic_smoke=True,smoke=False,skip_inference_profile=True)
        real_loader = feature_training.make_loader
        roles = []
        def observe(*values: object, **kwargs: object) -> object:
            """Reject every protected-role loader before it can read any images."""
            roles.append(kwargs['role'])
            self.assertIn(kwargs['role'],('train','early_stop'))
            return real_loader(*values,**kwargs)
        with mock.patch('engine.feature_training.make_loader',side_effect=observe),contextlib.redirect_stdout(io.StringIO()):
            result = feature_training.run(args)
        self.assertEqual(roles,['train','train','early_stop'])
        self.assertEqual(result['epochs_completed'],1)
        self.assertEqual(result['run_kind'],'synthetic_smoke')
        config = json.loads((args.output/'config.json').read_text())
        self.assertEqual(config['model_architecture']['stage_effective_strides'],[2,4,8,16])
        with (args.output/'history.csv').open() as stream:
            history = list(csv.DictReader(stream))
        data = load_fold(fixture.root,fixture.out,1)
        loader = make_loader(data['train'],fixture.root,(32,32),32,0,3711,True,torch.device('cpu'),role='train')
        import hashlib
        digest = hashlib.sha256()
        for batch in loader:
            digest.update((json.dumps(list(batch['relative_path']),separators=(',',':'))+'\n').encode())
        self.assertEqual(history[0]['training_batch_order_sha256'],digest.hexdigest())
        self.assertTrue((args.output/'gradient_flow.png').is_file())
        for unit in ('slice','scan'):
            checkpoint = torch.load(args.output/f'best_{unit}_loss.pt',weights_only=True,map_location='cpu')
            _validate_checkpoint(checkpoint)
            self.assertEqual(checkpoint['config']['checkpoint_selection'],f'minimum_early_stop_{unit}_log_loss')
            pred = argparse.Namespace(checkpoint=args.output/f'best_{unit}_loss.pt',data_root=fixture.root,splits_dir=fixture.out,
                output=fixture.base/f'reload_{unit}',batch_size=32,workers=0,threads=2,device='cpu',role='early_stop',skip_inference_profile=True)
            with contextlib.redirect_stdout(io.StringIO()):
                prediction.run(pred)
            reproduced = json.loads((pred.output/'metrics.json').read_text())
            self.assertEqual(result['candidate_selections'][unit]['metrics']['slice'],reproduced['metrics']['slice'])
            invalid = copy.deepcopy(checkpoint['config'])
            invalid['model_architecture']['stem']['stride']=4
            with self.assertRaises(ValueError):
                validate_feature_architecture(invalid)

    def test_suite_failure_preserves_attempts_and_source_change_blocks_reuse(self) -> None:
        """A failed job must not be silently reused or turn into a repeat winner."""
        fixture = self.fixture()
        args = argparse.Namespace(data_root=fixture.root,splits_dir=fixture.out,output=fixture.base/'suite',
            reference_suite=fixture.base/'unused',phase='core',device='cpu',workers=0,
            skip_resource_smoke=True,no_repeats=False,synthetic_smoke=True,skip_inference_profile=True,list=False)
        with mock.patch('engine.feature_suite.subprocess.run',return_value=mock.Mock(returncode=1)) as child,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(feature_suite.run(args),1)
            self.assertEqual(child.call_count,6)
            self.assertEqual(feature_suite.run(args),1)
        self.assertEqual(len(list(args.output.glob('*.binding.json'))),12)
        self.assertIsNone(json.loads((args.output/'repeat_gate.json').read_text())['selected'])
        args.no_repeats=True
        with self.assertRaisesRegex(ValueError,'another source or plan'):
            feature_suite.run(args)


    def test_actual_six_case_suite_reuses_completed_results_and_matches_orders(self) -> None:
        """Exercise all factors on real synthetic tensors, then forbid second launches."""
        fixture = self.fixture()
        args = argparse.Namespace(data_root=fixture.root,splits_dir=fixture.out,output=fixture.base/'matrix',
            reference_suite=fixture.base/'unused',phase='core',device='cpu',workers=0,
            skip_resource_smoke=True,no_repeats=False,synthetic_smoke=True,skip_inference_profile=True,list=False)
        with contextlib.redirect_stdout(io.StringIO()):
            status = feature_suite.run(args)
        errors = '\n'.join(p.name + ': ' + p.read_text()[-1600:] for p in args.output.glob('*.log')) if status else ''
        self.assertEqual(status,0,errors)
        records = json.loads((args.output/'suite_records.json').read_text())
        self.assertEqual(len(records),6)
        self.assertTrue(all(r['status']=='complete' for r in records))
        feature_suite.verify_orders(records)
        self.assertEqual(json.loads((args.output/'repeat_gate.json').read_text())['status'],'not_eligible_synthetic_smoke')
        with mock.patch('engine.feature_suite.subprocess.run') as launch,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(feature_suite.run(args),0)
            launch.assert_not_called()

    def test_fixed_budget_does_not_obey_legacy_patience(self) -> None:
        """A synthetic three-epoch protocol still completes with patience one."""
        fixture = self.fixture()
        specification = json.loads(feature_training.SPEC.read_text())
        data = load_fold(fixture.root,fixture.out,1)
        specification['manifest_sha256'] = data['manifest_sha256']
        for role in ('train','early_stop'):
            specification['cohort'][role] = {'patients':len({r['patient_id'] for r in data[role]}),
                'scans':len({r['image_id'] for r in data[role]}),'slices':len(data[role])}
        counts = [sum(int(r['label'])==v for r in data['train']) for v in (0,1)]
        specification['core_cases'][0]['settings'].update(epochs=3,patience=1,image_size=[32,32],
            workers=0,batch_size=32,bce_pos_weight=counts[0]/counts[1])
        spec_path = fixture.base/'synthetic_protocol.json'
        spec_path.write_text(json.dumps(specification))
        args = argparse.Namespace(data_root=fixture.root,splits_dir=fixture.out,output=fixture.base/'fixed',
            case='R00',seed_base=3710,workers=0,device='cpu',synthetic_smoke=False,smoke=False,skip_inference_profile=True)
        with mock.patch.object(feature_training,'SPEC',spec_path),contextlib.redirect_stdout(io.StringIO()):
            result = feature_training.run(args)
        self.assertEqual(result['epochs_completed'],3)
        self.assertEqual(json.loads((args.output/'config.json').read_text())['stopping'],'fixed_budget_no_early_truncation')


    def test_clean_scoring_leaves_training_loader_rng_and_modes_unchanged(self) -> None:
        """Extra clean evaluation cannot reshuffle the next epoch or alter DropPath RNG."""
        fixture = self.fixture()
        data = load_fold(fixture.root,fixture.out,1)
        train_loader = make_loader(data['train'],fixture.root,(32,32),32,0,3711,True,torch.device('cpu'),role='train')
        early_loader = make_loader(data['early_stop'],fixture.root,(32,32),32,0,3711,False,torch.device('cpu'),role='early_stop')
        model, _ = matched_model('convnext_lite_v1',3711)
        model.train()
        model.stages[0].eval()
        modes = [m.training for m in model.modules()]
        torch_state = torch.get_rng_state().clone()
        loader_state = train_loader.generator.get_state().clone()
        python_state = random.getstate()
        state = state_digest(model.state_dict())
        feature_training.scored(model,early_loader,torch.device('cpu'),2)
        self.assertTrue(torch.equal(torch_state,torch.get_rng_state()))
        self.assertTrue(torch.equal(loader_state,train_loader.generator.get_state()))
        self.assertEqual(python_state,random.getstate())
        self.assertEqual(modes,[m.training for m in model.modules()])
        self.assertEqual(state,state_digest(model.state_dict()))

    def test_submit_wrapper_dry_run_is_nonmutating(self) -> None:
        """Dry-run must not create logs or allocate a GPU."""
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(['bash',str(root/'slurm/submit_feature_suite.sh'),'--dry-run'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('feature_suite.sbatch',result.stdout)
        self.assertIn('--time=06:00:00',result.stdout)


if __name__ == '__main__':
    unittest.main()
