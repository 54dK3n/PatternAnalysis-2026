"""Synthetic checks for train-only controls and resumable batch orchestration."""
import argparse
from collections import defaultdict
import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

from PIL import Image
import torch

from dataset.augmentation import AUGMENTATION_NAMES, AugmentationConfig, augment_image, make_augmentation
from dataset.loaders import make_loader
from dataset.sampling import sampling_weights
from engine.scheduling import learning_rate
from engine import training, prediction, experiment_suite
import test_adni_splits as fixtures


class NewTransformTests(unittest.TestCase):
    """Test interpolation distinction, unchanged sources and serialized contracts."""

    def test_all_profiles_round_trip_and_reject_modified_metadata(self) -> None:
        """New metadata must not silently alter historical profiles."""
        for name in AUGMENTATION_NAMES:
            c = make_augmentation(name)
            self.assertEqual(AugmentationConfig.from_dict(c.to_dict()), c)
            invalid = c.to_dict() | {'algorithm':'changed'}
            with self.assertRaises(ValueError):
                AugmentationConfig.from_dict(invalid)
        self.assertEqual(make_augmentation('none').to_dict()['algorithm'], 'pil_affine_v1')
        self.assertNotIn('translation_pixels', make_augmentation('light').to_dict())
        with self.assertRaises(ValueError):
            make_augmentation('integer_shift', translation_pixels=True)
        with self.assertRaises(ValueError):
            make_augmentation('integer_shift', translation_pixels=9)

    def test_integer_shift_moves_exact_values_and_does_not_wrap_edges(self) -> None:
        """An integer shift must not interpolate or wrap cropped content."""
        image = Image.new('L', (8,8))
        image.putpixel((3,3), 123)
        image.putpixel((7,7), 254)
        before = image.tobytes()
        rng = mock.Mock()
        rng.randint.side_effect = [1,2]
        with mock.patch.object(Image.Image, 'transform', side_effect=AssertionError('Interpolation')):
            shifted = augment_image(image, make_augmentation('integer_shift'), rng)
        self.assertEqual(shifted.getpixel((4,5)), 123)
        self.assertNotIn(254, shifted.tobytes())
        self.assertEqual(shifted.getpixel((0,0)), 0)
        self.assertEqual(image.tobytes(), before)

    def test_fractional_shift_creates_intermediate_values(self) -> None:
        """A half-pixel shift differs from exact crop/paste with the same bound."""
        image = Image.new('L', (8,8))
        for y in range(8):
            for x in range(4,8):
                image.putpixel((x,y),255)
        rng = mock.Mock()
        rng.uniform.side_effect = [0.5,0.0]
        shifted = augment_image(image, make_augmentation('subpixel_shift'), rng)
        self.assertTrue(any(0 < v < 255 for v in shifted.tobytes()))
        self.assertEqual(shifted.size, image.size)

    def test_gamma_preserves_geometry_black_and_white(self) -> None:
        """Gamma changes intensities and leaves background/spatial positions intact."""
        image = Image.new('L', (3,1))
        image.putdata([0,128,255])
        rng = mock.Mock()
        rng.uniform.return_value = 0.9
        transformed = augment_image(image, make_augmentation('gamma'), rng)
        self.assertEqual(list(transformed.tobytes())[::2], [0,255])
        self.assertGreater(transformed.getpixel((1,0)),128)
        self.assertEqual(list(image.tobytes()),[0,128,255])


class SamplingTests(unittest.TestCase):
    """Check expected mass instead of relying only on noisy empirical counts."""

    def rows(self) -> list[dict]:
        """Construct unequal patient and class slice counts."""
        return [dict(patient_id=p, label=str(y), partition='development')
                for p,y,n in [('nc_a',0,2),('nc_b',0,8),('ad_a',1,20)] for _ in range(n)]

    def test_each_class_and_patient_has_declared_mass(self) -> None:
        """Repeated scans/slices must not increase a patient's expected weight."""
        rows = self.rows()
        weights, info = sampling_weights(rows, 'class_patient_balanced')
        mass = defaultdict(float)
        for r,w in zip(rows,weights):
            mass[r['patient_id']] += w
        self.assertAlmostEqual(mass['nc_a'],0.25)
        self.assertAlmostEqual(mass['nc_b'],0.25)
        self.assertAlmostEqual(mass['ad_a'],0.5)
        self.assertEqual(info['draws_per_epoch'],len(rows))
        self.assertEqual(sampling_weights(rows,'slice_uniform')[0],None)

    def test_mixed_or_reserved_patients_and_eval_sampling_are_rejected(self) -> None:
        """Only single-diagnosis development training patients may be balanced."""
        rows = self.rows()
        for changed in (rows + [dict(rows[0],label='1')], [dict(rows[0],partition='test')], rows[:2]):
            with self.assertRaises(ValueError):
                sampling_weights(changed,'class_patient_balanced')
        with self.assertRaises(ValueError):
            make_loader(rows, '.', (32,32), 2,0,1,False,torch.device('cpu'),
                        role='early_stop',sampling='class_patient_balanced')


class ScheduleTests(unittest.TestCase):
    """Check endpoints and historical constant-learning-rate behavior."""

    def test_constant_and_warmup_cosine_values(self) -> None:
        """Warmup starts at half LR and cosine ends at its declared floor."""
        self.assertEqual(learning_rate(1,30,0.001,'constant'),0.001)
        self.assertEqual(learning_rate(1,30,0.001,'warmup_cosine'),0.0005)
        self.assertEqual(learning_rate(2,30,0.001,'warmup_cosine'),0.001)
        self.assertAlmostEqual(learning_rate(30,30,0.001,'warmup_cosine'),0.00001)
        with self.assertRaises(ValueError):
            learning_rate(1,2,0.001,'warmup_cosine',warmup=2)


class ControlIntegrationTests(unittest.TestCase):
    """Run actual synthetic CPU training and reproduce selected predictions."""

    def test_new_controls_train_without_holdouts_and_checkpoint_reloads(self) -> None:
        """Balanced sampling and new transform metadata remain reload-compatible."""
        fixture = fixtures.ADNISplitTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        # Production manifests have single-diagnosis patients; remove mixed synthetic cases.
        for relative, record in list(fixture.expected.items()):
            if record['patient_id'] in fixture.mixed_patients:
                (fixture.root / relative).unlink()
                fixture.metadata.pop(record['image_id'], None)
                del fixture.expected[relative]
        fixture.save_metadata()
        fixture.prepare()
        args = argparse.Namespace(data_root=fixture.root,splits_dir=fixture.out,output=fixture.base/'controlled',
            model='convnext_lite', fold=1,epochs=2,patience=5,batch_size=16,workers=0,threads=1,
            lr=0.0001,weight_decay=0.05,min_delta=0,seed=3710,image_height=32,image_width=32,device='cpu',
            inner_only=True,sampling='class_patient_balanced',augmentation='integer_gamma',translation_pixels=4,
            lr_schedule='warmup_cosine',warmup_epochs=1,min_lr_ratio=0.01)
        real_loader = training.make_loader
        seen = []
        def observe(*values: object, **kwargs: object) -> object:
            """Observe actual loaders without permitting outer or reserved roles."""
            seen.append(kwargs['role'])
            self.assertIn(kwargs['role'],('train','early_stop'))
            return real_loader(*values,**kwargs)
        with mock.patch('engine.training.make_loader',side_effect=observe), contextlib.redirect_stdout(io.StringIO()):
            result = training.run(args)
        self.assertEqual(seen,['train','early_stop'])
        config = json.loads((args.output/'config.json').read_text())
        self.assertEqual(config['train_pos_weight'],1.0)
        self.assertEqual(config['training_sampling']['name'],'class_patient_balanced')
        self.assertEqual(config['lr_schedule']['name'],'warmup_cosine')
        self.assertEqual(result['evaluation_role'],'development_inner_early_stop')
        checkpoint = torch.load(args.output/'best.pt',map_location='cpu',weights_only=True)
        self.assertTrue(all(torch.isfinite(v).all() for v in checkpoint['model_state'].values()))
        pred = argparse.Namespace(checkpoint=args.output/'best.pt',data_root=fixture.root,splits_dir=fixture.out,
            output=fixture.base/'reloaded',batch_size=16,workers=0,threads=1,device='cpu',role='early_stop')
        with contextlib.redirect_stdout(io.StringIO()):
            prediction.run(pred)
        reproduced = json.loads((pred.output / "metrics.json").read_text())
        self.assertEqual(result['metrics']['slice'],reproduced['metrics']['slice'])
        self.assertEqual(result['metrics']['scan'],reproduced['metrics']['scan'])


class SuiteTests(unittest.TestCase):
    """Verify the plan and guarded failure/retry behavior without GPU jobs."""

    def test_seventeen_commands_are_always_inner_only(self) -> None:
        """Each case has a stable identity and never invokes a holdout predictor."""
        root = Path(__file__).resolve().parents[1]
        args = argparse.Namespace(data_root=Path('/data'),splits_dir=Path('/splits'),epochs=30,workers=2,device='cuda')
        plan = experiment_suite.build_plan(args,root)
        self.assertEqual(len(plan['cases']),17)
        self.assertEqual(len({c['id'] for c in plan['cases']}),17)
        for case in plan['cases']:
            argv = experiment_suite.command(root,case['flags'],Path('/output'))
            self.assertIn('--inner-only',argv)
            self.assertNotIn('--role',argv)
        self.assertEqual(plan['cases'][9]['flags']['translation_pixels'],plan['cases'][10]['flags']['translation_pixels'])

    def test_failed_attempts_continue_and_retry_in_new_folders(self) -> None:
        """Failed pre-audit launches remain identifiable and are never overwritten."""
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            splits = base/'splits'
            splits.mkdir()
            (splits/'COMPLETED.json').write_text('{}')
            args = argparse.Namespace(data_root=base/'data',splits_dir=splits,output=base/'suite',
                epochs=30,workers=0,device='cpu',list=False,ids='E00_cnn,E01_lite_reference')
            with mock.patch('engine.experiment_suite.subprocess.run',return_value=mock.Mock(returncode=1)) as call, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(experiment_suite.run(args),1)
                self.assertEqual(call.call_count,2)
                self.assertEqual(experiment_suite.run(args),1)
            attempts = list(args.output.glob('*.case.json'))
            self.assertEqual(len(attempts),4)
            self.assertTrue(any('attempt_02' in p.name for p in attempts))
            args.epochs=31
            with self.assertRaisesRegex(ValueError,'different plan'):
                experiment_suite.run(args)

    def test_actual_suite_completion_is_reused_without_retraining(self) -> None:
        """Run a complete synthetic CPU case, then verify completion reuse."""
        fixture = fixtures.ADNISplitTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.prepare()
        args = argparse.Namespace(data_root=fixture.root,splits_dir=fixture.out,output=fixture.base/'suite',
            epochs=1,workers=0,device='cpu',list=False,ids='E00_cnn')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(experiment_suite.run(args),0)
        stats = json.loads((args.output/'suite_summary.json').read_text())
        self.assertEqual(stats[0]['status'],'complete')
        with mock.patch('engine.experiment_suite.subprocess.run') as launch, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(experiment_suite.run(args),0)
            launch.assert_not_called()
