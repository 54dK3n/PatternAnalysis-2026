"""Versioned geometric augmentation applied only to development training slices.

The light profile combines rotation and translation in one Pillow bilinear
resampling operation on a fixed canvas. It adds no flip, random crop, scale,
intensity adjustment, or statistics estimated from any data partition.

The added experimental profiles separate integer crop/paste shifts from
continuous affine shifts using the same pixel bound, and mild gamma LUT changes.
They do not establish clinical invariance or performance benefits.

Pillow transform reference:
https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.rotate
"""

from dataclasses import dataclass
import math

from PIL import Image


AUGMENTATION_NAMES = ("none", "light", "integer_shift", "subpixel_shift", "gamma", "integer_gamma")
_METADATA = {
    "algorithm": "pil_affine_v1",
    "scope": "train_only",
    "interpolation": "bilinear",
    "fill": 0,
}
_FIELDS = {"name", "rotation_degrees", "translation_fraction", *_METADATA}


def _validate_bounds(rotation_degrees, translation_fraction):
    for name, value, maximum in (("rotation_degrees", rotation_degrees, 15.0),
                                 ("translation_fraction", translation_fraction, 0.1)):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0.0 <= value <= maximum:
            raise ValueError(f"{name} must be a finite number between 0 and {maximum}.")


@dataclass(frozen=True)
class AugmentationConfig:
    """An immutable augmentation profile with strict checkpoint serialization."""

    name: str = "none"
    rotation_degrees: float = 0.0
    translation_fraction: float = 0.0
    translation_pixels: int = 4

    def __post_init__(self):
        if type(self.name) is not str or self.name not in AUGMENTATION_NAMES:
            raise ValueError(f"Unsupported augmentation profile: {self.name!r}")
        _validate_bounds(self.rotation_degrees, self.translation_fraction)
        if self.name == "none" and (self.rotation_degrees != 0 or self.translation_fraction != 0):
            raise ValueError("The none augmentation profile must have zero rotation and translation.")
        if type(self.translation_pixels) is not int or not 0 <= self.translation_pixels <= 8:
            raise ValueError("Translation pixels must be an integer in [0, 8].")
        if self.name not in ("none", "light") and (self.rotation_degrees or self.translation_fraction):
            raise ValueError("New profiles use pixel bounds, not rotation/fraction settings.")
        if self.name in ("none", "light", "gamma") and self.translation_pixels != 4:
            raise ValueError("Translation pixel overrides apply only to shift profiles.")
        object.__setattr__(self, "rotation_degrees", float(self.rotation_degrees))
        object.__setattr__(self, "translation_fraction", float(self.translation_fraction))

    def to_dict(self):
        """Record every supported transform setting rather than only a profile name."""
        value = {
            "name": self.name,
            "rotation_degrees": self.rotation_degrees,
            "translation_fraction": self.translation_fraction,
            **_METADATA,
        }
        if self.name not in ("none", "light"):
            value.update(algorithm={"integer_shift":"crop_paste_integer_v1", "subpixel_shift":"pil_affine_pixel_v1",
                                    "gamma":"pil_gamma_lut_v1", "integer_gamma":"crop_paste_integer_gamma_v1"}[self.name],
                         interpolation="bilinear" if self.name == "subpixel_shift" else "none")
            if self.name != "gamma":
                value["translation_pixels"] = self.translation_pixels
            if self.name in ("gamma", "integer_gamma"):
                value["gamma_range"] = [0.9, 1.1]
        return value

    @classmethod
    def from_dict(cls, value):
        """Reject missing, unknown, or changed algorithm settings in a checkpoint."""
        if type(value) is not dict:
            raise ValueError("Augmentation configuration must be a dict.")
        config = cls(value.get("name"), value.get("rotation_degrees"),
                     value.get("translation_fraction"), value.get("translation_pixels", 4))
        expected = config.to_dict()
        if set(value) != set(expected):
            raise ValueError("Augmentation configuration has unsupported fields.")
        for key in set(expected) - {"name", "rotation_degrees", "translation_fraction", "translation_pixels"}:
            if type(value[key]) is not type(expected[key]) or value[key] != expected[key]:
                raise ValueError(f"Unsupported augmentation setting: {key}={value[key]!r}")
        return config


def make_augmentation(name="none", rotation_degrees=5.0, translation_fraction=0.03, translation_pixels=4):
    """Resolve a named profile, keeping the default pipeline an exact identity."""
    if type(name) is not str or name not in AUGMENTATION_NAMES:
        raise ValueError(f"Unsupported augmentation profile: {name!r}")
    _validate_bounds(rotation_degrees, translation_fraction)
    if name == "none":
        return AugmentationConfig()
    if name == "light":
        return AugmentationConfig(name, rotation_degrees, translation_fraction)
    return AugmentationConfig(name, translation_pixels=translation_pixels)


def augment_image(image, config, rng):
    """Sample from a caller-owned RNG; never modify an image or source file in place.

    Rotation is around the image center. Translation bounds are fractions of
    the corresponding width and height. Pixels outside the fixed output canvas
    are discarded, and newly exposed positions use black fill (0 before
    normalization). There is no additional crop or resize in this operation.
    """
    if not isinstance(config, AugmentationConfig):
        raise ValueError("augmentation must be an AugmentationConfig.")
    if config.name == "none":
        return image
    if config.name in ("integer_shift", "integer_gamma"):
        dx = rng.randint(-config.translation_pixels, config.translation_pixels)
        dy = rng.randint(-config.translation_pixels, config.translation_pixels)
        shifted = Image.new("L", image.size, 0)
        shifted.paste(image, (dx, dy))  # No resampling; edges can still be clipped.
        image = shifted
    elif config.name == "subpixel_shift":
        dx = rng.uniform(-config.translation_pixels, config.translation_pixels)
        dy = rng.uniform(-config.translation_pixels, config.translation_pixels)
        return image.transform(image.size, Image.Transform.AFFINE, (1, 0, -dx, 0, 1, -dy),
                               resample=Image.Resampling.BILINEAR, fillcolor=0)
    if config.name in ("gamma", "integer_gamma"):
        gamma = rng.uniform(0.9, 1.1)
        lut = [round(255 * (i / 255) ** gamma) for i in range(256)]
        return image.point(lut)  # Preserves black background and spatial coordinates.
    if config.name == "integer_shift":
        return image
    angle = rng.uniform(-config.rotation_degrees, config.rotation_degrees)
    horizontal = rng.uniform(-config.translation_fraction, config.translation_fraction) * image.width
    vertical = rng.uniform(-config.translation_fraction, config.translation_fraction) * image.height
    return image.rotate(
        angle, resample=Image.Resampling.BILINEAR, expand=False,
        translate=(horizontal, vertical), fillcolor=0,
    )
