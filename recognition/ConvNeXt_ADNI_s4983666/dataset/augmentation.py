"""Training-only augmentation for grayscale MRI slices.

Profiles:

* ``none``   - identity (used for every evaluation role).
* ``light``  - small rotation and translation; the original profile.
* ``strong`` - horizontal flip, random-resized crop, rotation, translation and a
  mild intensity change. It targets the memorisation seen when the scratch
  ConvNeXt reached ~100% training accuracy within a few epochs.

Why each transform keeps the AD/NC label valid:

* Horizontal flip: the slices are axial and AD atrophy is bilateral, so a
  left-right mirror is still a plausible brain with the same diagnosis.
* Crop, rotation and translation: head size, position and tilt vary between
  patients and scanners; the diagnosis does not depend on them.
* Intensity gain and gamma: brightness and contrast vary between acquisitions.
  Both map black to black, so the background stays zero.

Vertical flips and large rotations are not used because they produce views that
never occur in this dataset.

When neighbouring slices are stacked as the bands of one PIL image, every band
receives exactly the same random transform, so the context stays aligned.

References: Pillow image API
https://pillow.readthedocs.io/en/stable/reference/Image.html ; random-resized
cropping follows Szegedy et al., "Going Deeper with Convolutions", CVPR 2015.
"""

from dataclasses import asdict, dataclass
import math
import random

from PIL import Image


AUGMENTATION_NAMES = ("none", "light", "strong")


@dataclass(frozen=True)
class AugmentationConfig:
    """Ranges of the random training transforms; a zero range disables one.

    ``intensity_gain=g`` samples a gain from [1 - g, 1 + g] and
    ``gamma_range=r`` samples a gamma from [1 - r, 1 + r].
    """

    name: str = "none"
    flip_probability: float = 0.0
    min_crop_scale: float = 1.0
    rotation_degrees: float = 0.0
    translation_fraction: float = 0.0
    intensity_gain: float = 0.0
    gamma_range: float = 0.0

    def __post_init__(self) -> None:
        """Reject unknown profiles and ranges that would distort the anatomy."""
        if self.name not in AUGMENTATION_NAMES:
            raise ValueError(f"Unsupported augmentation profile: {self.name!r}")
        limits = {"flip_probability": (0.0, 1.0), "min_crop_scale": (0.5, 1.0),
                  "rotation_degrees": (0.0, 15.0), "translation_fraction": (0.0, 0.1),
                  "intensity_gain": (0.0, 0.3), "gamma_range": (0.0, 0.3)}
        for field, (low, high) in limits.items():
            value = getattr(self, field)
            if not isinstance(value, (int, float)) or not low <= value <= high:
                raise ValueError(f"{field} must be between {low} and {high}.")

    def to_dict(self) -> dict:
        """Serialize every range so a checkpoint records the exact transform."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "AugmentationConfig":
        """Rebuild a configuration saved by ``to_dict``."""
        return cls(**value)


PROFILES = {
    "none": AugmentationConfig(),
    "light": AugmentationConfig("light", rotation_degrees=5.0, translation_fraction=0.03),
    "strong": AugmentationConfig("strong", flip_probability=0.5, min_crop_scale=0.85,
                                 rotation_degrees=10.0, translation_fraction=0.05,
                                 intensity_gain=0.1, gamma_range=0.15),
}


def make_augmentation(name: str = "none") -> AugmentationConfig:
    """Return the fixed configuration of a named profile."""
    if name not in PROFILES:
        raise ValueError(f"Unsupported augmentation profile: {name!r}")
    return PROFILES[name]


def _black(image: Image.Image) -> int | tuple[int, ...]:
    """Fill value for pixels exposed by rotation or translation."""
    bands = len(image.getbands())
    return 0 if bands == 1 else (0,) * bands


def _random_resized_crop(image: Image.Image, min_scale: float, rng: random.Random) -> Image.Image:
    """Crop a random region covering [min_scale, 1] of the area and resize it back."""
    width, height = image.size
    side = math.sqrt(rng.uniform(min_scale, 1.0))  # Same factor on both sides keeps the aspect ratio.
    crop_width, crop_height = max(1, round(width * side)), max(1, round(height * side))
    left = rng.randint(0, width - crop_width)
    top = rng.randint(0, height - crop_height)
    box = (left, top, left + crop_width, top + crop_height)
    return image.resize((width, height), Image.Resampling.BILINEAR, box=box)


def augment_image(image: Image.Image, config: AugmentationConfig, rng: random.Random) -> Image.Image:
    """Apply one random draw of every enabled transform; the input is not modified."""
    if config.name == "none":
        return image
    if rng.random() < config.flip_probability:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if config.min_crop_scale < 1.0:
        image = _random_resized_crop(image, config.min_crop_scale, rng)
    if config.rotation_degrees or config.translation_fraction:
        angle = rng.uniform(-config.rotation_degrees, config.rotation_degrees)
        shift_x = rng.uniform(-config.translation_fraction, config.translation_fraction) * image.width
        shift_y = rng.uniform(-config.translation_fraction, config.translation_fraction) * image.height
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False,
                             translate=(shift_x, shift_y), fillcolor=_black(image))
    if config.intensity_gain or config.gamma_range:
        gain = rng.uniform(1 - config.intensity_gain, 1 + config.intensity_gain)
        gamma = rng.uniform(1 - config.gamma_range, 1 + config.gamma_range)
        table = [min(255, round(255 * gain * (value / 255) ** gamma)) for value in range(256)]
        image = image.point(table * len(image.getbands()))  # One table per band.
    return image
