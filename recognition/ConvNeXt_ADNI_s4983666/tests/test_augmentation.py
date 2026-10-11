"""Training-only augmentation profiles and their effect on images."""

import random
import unittest

from PIL import Image

from dataset.augmentation import AUGMENTATION_NAMES, AugmentationConfig, augment_image, make_augmentation


def gradient_image(width: int = 32, height: int = 24) -> Image.Image:
    """Left-to-right brightness ramp on a black border, like a head on background."""
    image = Image.new("L", (width, height), 0)
    for x in range(4, width - 4):
        for y in range(4, height - 4):
            image.putpixel((x, y), 40 + 6 * x)
    return image


class AugmentationTests(unittest.TestCase):
    def test_profiles_and_round_trip(self):
        self.assertEqual(AUGMENTATION_NAMES, ("none", "light", "strong"))
        for name in AUGMENTATION_NAMES:
            config = make_augmentation(name)
            self.assertEqual(AugmentationConfig.from_dict(config.to_dict()), config)
        strong = make_augmentation("strong")
        self.assertEqual(strong.flip_probability, 0.5)
        self.assertLess(strong.min_crop_scale, 1.0)
        with self.assertRaises(ValueError):
            make_augmentation("vertical_flip")

    def test_none_is_identity_and_strong_keeps_size_mode_and_black_background(self):
        image = gradient_image()
        self.assertIs(augment_image(image, make_augmentation("none"), random.Random(0)), image)
        for seed in range(20):
            out = augment_image(image, make_augmentation("strong"), random.Random(seed))
            self.assertEqual((out.size, out.mode), (image.size, image.mode))
            self.assertEqual(out.getpixel((0, 0)), 0)  # Corners stay background.
        self.assertEqual(list(image.getdata()), list(gradient_image().getdata()))  # Input untouched.

    def test_same_seed_gives_same_result_and_flip_reverses_brightness_ramp(self):
        image = gradient_image()
        strong = make_augmentation("strong")
        first = augment_image(image, strong, random.Random(7))
        self.assertEqual(list(first.getdata()), list(augment_image(image, strong, random.Random(7)).getdata()))
        flip_only = AugmentationConfig("strong", flip_probability=1.0)
        flipped = augment_image(image, flip_only, random.Random(0))
        self.assertGreater(flipped.getpixel((6, 12)), flipped.getpixel((25, 12)))

    def test_stacked_context_bands_receive_identical_geometry(self):
        band = gradient_image()
        stacked = Image.merge("RGB", [band, band, band])
        out = augment_image(stacked, make_augmentation("strong"), random.Random(3))
        red, green, blue = out.split()
        self.assertEqual(list(red.getdata()), list(green.getdata()))
        self.assertEqual(list(red.getdata()), list(blue.getdata()))

    def test_ranges_are_bounded(self):
        for field, value in (("rotation_degrees", 45.0), ("min_crop_scale", 0.2),
                             ("flip_probability", 1.5), ("gamma_range", 0.9)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                AugmentationConfig("strong", **{field: value})


if __name__ == "__main__":
    unittest.main()
