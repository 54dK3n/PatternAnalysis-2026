"""Check train-only transforms without using real ADNI images or model weights."""

import copy
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

from PIL import Image, ImageOps
import torch

from dataset.augmentation import AugmentationConfig, augment_image, make_augmentation
from dataset import loaders
from dataset.loaders import make_loader
from dataset.slices import ADNISliceDataset
from dataset.splits import AuditError, digest


class AugmentationConfigTests(unittest.TestCase):
    def test_tracker_descriptor_fix_is_limited_to_affected_platforms(self):
        for platform, version in (("linux", (3, 12)), ("win32", (3, 12)), ("darwin", (3, 11))):
            with self.subTest(platform=platform, version=version), \
                    mock.patch.object(loaders.sys, "platform", platform), \
                    mock.patch.object(loaders.sys, "version_info", version), \
                    mock.patch("multiprocessing.resource_tracker.getfd") as getfd, \
                    mock.patch.object(loaders.os, "set_inheritable") as inheritable:
                loaders._prevent_macos_tracker_inheritance()
                getfd.assert_not_called()
                inheritable.assert_not_called()
        with mock.patch.object(loaders.sys, "platform", "darwin"), \
                mock.patch.object(loaders.sys, "version_info", (3, 12)), \
                mock.patch("multiprocessing.resource_tracker.getfd", return_value=123) as getfd, \
                mock.patch.object(loaders.os, "set_inheritable") as inheritable:
            loaders._prevent_macos_tracker_inheritance()
            getfd.assert_called_once_with()
            inheritable.assert_called_once_with(123, False)

    def test_serialized_profiles_round_trip_with_explicit_transform_metadata(self):
        for config in (make_augmentation(), make_augmentation("light"),
                       make_augmentation("light", rotation_degrees=15, translation_fraction=0.1)):
            with self.subTest(name=config.name, rotation=config.rotation_degrees):
                self.assertEqual(AugmentationConfig.from_dict(config.to_dict()), config)
                self.assertEqual(set(config.to_dict()), {
                    "name", "rotation_degrees", "translation_fraction", "algorithm",
                    "scope", "interpolation", "fill",
                })
        self.assertEqual(make_augmentation().rotation_degrees, 0.0)
        self.assertEqual(make_augmentation().translation_fraction, 0.0)

    def test_invalid_names_bounds_and_checkpoint_contracts_are_rejected(self):
        for name in ("flip", "LIGHT", None, True):
            with self.subTest(name=name), self.assertRaises(ValueError):
                make_augmentation(name)
        for field, invalid in (
            ("rotation_degrees", (float("nan"), float("inf"), -1, 15.1, True, "5", None)),
            ("translation_fraction", (float("nan"), float("inf"), -0.1, 0.1001, True, "0.03", None)),
        ):
            for value in invalid:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    make_augmentation("light", **{field: value})
        with self.assertRaises(ValueError):
            AugmentationConfig("none", 1, 0)
        with self.assertRaises(ValueError):
            AugmentationConfig("none", 0, 0.01)

        valid = make_augmentation("light").to_dict()
        mutations = [None, "light", {}, dict(valid, unexpected=True),
                     {key: value for key, value in valid.items() if key != "algorithm"}]
        for field, value in (("algorithm", "unknown"), ("scope", "all"),
                             ("interpolation", "nearest"), ("fill", 255), ("fill", False),
                             ("rotation_degrees", float("nan")), ("translation_fraction", True)):
            mutations.append(dict(valid, **{field: value}))
        for value in mutations:
            with self.subTest(value=value), self.assertRaises(ValueError):
                AugmentationConfig.from_dict(value)

    def test_identity_profile_consumes_no_random_numbers_or_resampling(self):
        image = Image.new("L", (20, 16), color=128)
        rng = random.Random(13)
        before = rng.getstate()
        with mock.patch.object(Image.Image, "rotate", side_effect=AssertionError("Unexpected resampling")):
            actual = augment_image(image, make_augmentation(), rng)
        self.assertIs(actual, image)
        self.assertEqual(rng.getstate(), before)

    def test_rotation_and_translation_use_one_bilinear_operation_with_black_fill(self):
        image = Image.new("L", (100, 80), color=255)
        rng = mock.Mock()
        rng.uniform.side_effect = [5.0, 0.03, -0.03]
        with mock.patch.object(image, "rotate", wraps=image.rotate) as rotate:
            result = augment_image(image, make_augmentation("light"), rng)
        rotate.assert_called_once_with(
            5.0, resample=Image.Resampling.BILINEAR, expand=False,
            translate=(3.0, -2.4), fillcolor=0,
        )
        self.assertEqual(result.size, image.size)
        self.assertIn(0, result.tobytes())
        self.assertEqual(set(image.tobytes()), {255})


class SliceAugmentationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="adni_augmentation_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.rows = []
        for index in range(8):
            relative = f"slices/{100 + index // 2}_{index % 2}.jpeg"
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            image = Image.new("L", (64, 48))
            image.putdata([(x * 7 + y * 11 + index * 29) % 256
                           for y in range(48) for x in range(64)])
            image.save(path, quality=95)
            self.rows.append({
                "relative_path": relative, "label": str((index // 2) % 2),
                "patient_id": f"001_S_{4000 + index // 2}", "image_id": str(100 + index // 2),
                "slice_index": str(index % 2), "partition": "development",
                "file_sha256": digest(path.read_bytes()),
                "pixel_sha256": digest(Image.open(path).tobytes()),
            })

    def dataset(self, seed=3710, **kwargs):
        return ADNISliceDataset(
            self.rows, self.root, image_size=(48, 64), role="train",
            augmentation=make_augmentation("light"), augmentation_seed=seed, **kwargs,
        )

    def test_actual_augmentation_changes_only_the_returned_image(self):
        original_rows = copy.deepcopy(self.rows)
        source_bytes = [(self.root / row["relative_path"]).read_bytes() for row in self.rows]
        baseline = ADNISliceDataset(self.rows, self.root, image_size=(48, 64))
        augmented = self.dataset()
        for index in range(len(self.rows)):
            actual, original = augmented[index], baseline[index]
            self.assertFalse(torch.equal(actual["image"], original["image"]))
            self.assertEqual(tuple(actual["image"].shape), (1, 48, 64))
            self.assertTrue(torch.isfinite(actual["image"]).all())
            self.assertGreaterEqual(actual["image"].min().item(), -1.0)
            self.assertLessEqual(actual["image"].max().item(), 1.0)
            for key in ("patient_id", "image_id", "slice_index", "relative_path"):
                self.assertEqual(actual[key], original[key])
            self.assertTrue(torch.equal(actual["label"], original["label"]))
        self.assertEqual(self.rows, original_rows)
        self.assertEqual(augmented.rows, original_rows)
        for row, content in zip(self.rows, source_bytes):
            path = self.root / row["relative_path"]
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(digest(path.read_bytes()), row["file_sha256"])
            with Image.open(path) as image:
                self.assertEqual(digest(image.tobytes()), row["pixel_sha256"])

    def test_non_training_roles_and_non_development_manifests_reject_augmentation(self):
        for role in ("evaluation", "early_stop", "val", "calibration", "test"):
            with self.subTest(role=role), self.assertRaisesRegex(AuditError, "train role"):
                ADNISliceDataset(self.rows, self.root, role=role, augmentation=make_augmentation("light"))
        for partition in ("calibration", "test", None):
            rows = [dict(self.rows[0], partition=partition)]
            with self.subTest(partition=partition), self.assertRaisesRegex(AuditError, "development"):
                ADNISliceDataset(rows, self.root, role="train", augmentation=make_augmentation("light"))
        with self.assertRaisesRegex(AuditError, "development"):
            ADNISliceDataset(self.rows + [dict(self.rows[0], partition="test")], self.root,
                             role="train", augmentation=make_augmentation("light"))
        with self.assertRaisesRegex(AuditError, "Unsupported dataset role"):
            ADNISliceDataset(self.rows, self.root, role="unknown")
        with self.assertRaisesRegex(AuditError, "AugmentationConfig"):
            ADNISliceDataset(self.rows, self.root, role="train", augmentation="light")
        with self.assertRaisesRegex(AuditError, "integer"):
            self.dataset(seed=True)

    def test_none_preserves_original_grayscale_resize_and_tensor_values(self):
        for role in ("train", "evaluation", "early_stop", "val", "calibration", "test"):
            dataset = ADNISliceDataset(self.rows, self.root, image_size=(24, 40),
                                       role=role, augmentation=make_augmentation())
            with Image.open(self.root / self.rows[0]["relative_path"]) as image:
                processed = ImageOps.exif_transpose(image).convert("L").resize(
                    (40, 24), resample=Image.Resampling.BILINEAR,
                )
                expected = torch.tensor(list(processed.tobytes()), dtype=torch.float32).reshape(1, 24, 40)
                expected = expected.div(255.0).sub(0.5).div(0.5)
            with self.subTest(role=role):
                self.assertTrue(torch.equal(dataset[0]["image"], expected))
                self.assertTrue(torch.equal(dataset[0]["image"], expected))
                self.assertIsNone(dataset._augmentation_rng)

    def test_source_hash_is_verified_before_randomness_or_transformation(self):
        dataset = self.dataset()
        path = self.root / self.rows[0]["relative_path"]
        Image.new("L", (64, 48), 128).save(path)
        with mock.patch("dataset.slices.augment_image") as transform, \
                self.assertRaisesRegex(AuditError, "Source image changed"):
            dataset[0]
        transform.assert_not_called()
        self.assertIsNone(dataset._augmentation_rng)

    def test_reproducible_private_stream_varies_across_draws_and_seeds(self):
        first, repeated, different = self.dataset(33), self.dataset(33), self.dataset(34)
        stream_a = [first[0]["image"] for _ in range(4)]
        stream_b = [repeated[0]["image"] for _ in range(4)]
        stream_c = [different[0]["image"] for _ in range(4)]
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(stream_a, stream_b)))
        self.assertTrue(any(not torch.equal(a, c) for a, c in zip(stream_a, stream_c)))
        self.assertTrue(any(not torch.equal(stream_a[0], value) for value in stream_a[1:]))

    def test_augmentation_does_not_consume_global_python_or_torch_randomness(self):
        python_state, torch_state = random.getstate(), torch.random.get_rng_state().clone()
        dataset = self.dataset()
        for _ in range(3):
            dataset[0]
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))

    def test_shuffling_does_not_implicitly_enable_augmentation(self):
        baseline = ADNISliceDataset(self.rows, self.root, image_size=(48, 64))
        expected = {row["relative_path"]: baseline[index]["image"] for index, row in enumerate(self.rows)}
        loader = make_loader(self.rows, self.root, (48, 64), 3, 0, 3710, True, torch.device("cpu"))
        actual_paths = []
        for batch in loader:
            for path, image in zip(batch["relative_path"], batch["image"]):
                self.assertTrue(torch.equal(image, expected[path]))
                actual_paths.append(path)
        self.assertEqual(set(actual_paths), set(expected))
        self.assertEqual(len(actual_paths), len(expected))
        with self.assertRaisesRegex(AuditError, "train role"):
            make_loader(self.rows, self.root, (48, 64), 3, 0, 3710, True,
                        torch.device("cpu"), augmentation=make_augmentation("light"))

    def test_two_worker_loaders_reproduce_augmented_epochs_with_the_same_seed(self):
        def create():
            return make_loader(self.rows, self.root, (48, 64), 4, 2, 8123, True,
                               torch.device("cpu"), role="train", augmentation=make_augmentation("light"))

        def collect(loader):
            paths, images = [], []
            for batch in loader:
                paths.extend(batch["relative_path"])
                images.extend(batch["image"].unbind(0))
            return paths, torch.stack(images)

        first, repeated = create(), create()
        first_epoch, first_epoch_repeat = collect(first), collect(repeated)
        second_epoch, second_epoch_repeat = collect(first), collect(repeated)
        self.assertEqual(first_epoch[0], first_epoch_repeat[0])
        self.assertTrue(torch.equal(first_epoch[1], first_epoch_repeat[1]))
        self.assertEqual(second_epoch[0], second_epoch_repeat[0])
        self.assertTrue(torch.equal(second_epoch[1], second_epoch_repeat[1]))
        original = dict(zip(first_epoch[0], first_epoch[1]))
        updated = dict(zip(second_epoch[0], second_epoch[1]))
        self.assertEqual(set(original), set(updated))
        self.assertTrue(any(not torch.equal(original[path], updated[path]) for path in original))


if __name__ == "__main__":
    unittest.main()
