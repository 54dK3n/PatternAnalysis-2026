"""Versioned geometric augmentation applied only to development training slices.

The light profile combines rotation and translation in one Pillow bilinear
resampling operation on a fixed canvas. It adds no flip, random crop, scale,
intensity adjustment, or statistics estimated from any data partition.

Pillow transform reference:
https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.rotate
"""

from dataclasses import dataclass
import math

from PIL import Image


AUGMENTATION_NAMES = ("none", "light")
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

    def __post_init__(self):
        if type(self.name) is not str or self.name not in AUGMENTATION_NAMES:
            raise ValueError(f"Unsupported augmentation profile: {self.name!r}")
        _validate_bounds(self.rotation_degrees, self.translation_fraction)
        if self.name == "none" and (self.rotation_degrees != 0 or self.translation_fraction != 0):
            raise ValueError("The none augmentation profile must have zero rotation and translation.")
        object.__setattr__(self, "rotation_degrees", float(self.rotation_degrees))
        object.__setattr__(self, "translation_fraction", float(self.translation_fraction))

    def to_dict(self):
        """Record every supported transform setting rather than only a profile name."""
        return {
            "name": self.name,
            "rotation_degrees": self.rotation_degrees,
            "translation_fraction": self.translation_fraction,
            **_METADATA,
        }

    @classmethod
    def from_dict(cls, value):
        """Reject missing, unknown, or changed algorithm settings in a checkpoint."""
        if type(value) is not dict or set(value) != _FIELDS:
            raise ValueError("Augmentation configuration must contain exactly the supported fields.")
        for key, expected in _METADATA.items():
            if type(value[key]) is not type(expected) or value[key] != expected:
                raise ValueError(f"Unsupported augmentation setting: {key}={value[key]!r}")
        return cls(value["name"], value["rotation_degrees"], value["translation_fraction"])


def make_augmentation(name="none", rotation_degrees=5.0, translation_fraction=0.03):
    """Resolve a named profile, keeping the default pipeline an exact identity."""
    if type(name) is not str or name not in AUGMENTATION_NAMES:
        raise ValueError(f"Unsupported augmentation profile: {name!r}")
    _validate_bounds(rotation_degrees, translation_fraction)
    if name == "none":
        return AugmentationConfig()
    return AugmentationConfig(name, rotation_degrees, translation_fraction)


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
    angle = rng.uniform(-config.rotation_degrees, config.rotation_degrees)
    horizontal = rng.uniform(-config.translation_fraction, config.translation_fraction) * image.width
    vertical = rng.uniform(-config.translation_fraction, config.translation_fraction) * image.height
    return image.rotate(
        angle, resample=Image.Resampling.BILINEAR, expand=False,
        translate=(horizontal, vertical), fillcolor=0,
    )
