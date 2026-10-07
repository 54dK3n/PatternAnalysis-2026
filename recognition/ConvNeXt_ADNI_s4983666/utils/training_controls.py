"""Versioned, opt-in scratch execution controls and overridable recipe defaults.

Cross entropy / BCE definitions: https://docs.pytorch.org/docs/stable/nn.html
AMP: https://docs.pytorch.org/docs/stable/amp.html
Peer recipe values are reported by Ken's supplied screenshots, not audited code.
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math
import sys
from typing import Any

import torch
from torch import nn

RECIPES = {
    'custom': {},
    'lite_reference': {'model': 'convnext_lite', 'preprocessing': 'none', 'image_height': 240,
        'image_width': 256, 'lr': 1e-4, 'weight_decay': .05, 'batch_size': 32,
        'lr_schedule': 'constant', 'epochs': 30, 'patience': None, 'augmentation': 'none',
        'sampling': 'slice_uniform', 'input_channels': 1, 'loss': 'weighted_bce',
        'precision': 'fp32', 'patient_aggregation': 'mean_probability', 'drop_path': None},
    'lite_augmented': {'model': 'convnext_lite', 'preprocessing': 'scan_intensity_crop',
        'image_height': 240, 'image_width': 256, 'lr': 1e-4, 'weight_decay': .05,
        'batch_size': 32, 'lr_schedule': 'constant', 'epochs': 30, 'patience': None,
        'augmentation': 'integer_gamma', 'sampling': 'slice_uniform', 'input_channels': 1,
        'loss': 'weighted_bce', 'precision': 'fp32', 'patient_aggregation': 'mean_probability', 'drop_path': None},
    'peer_tiny': {'model': 'convnext_tiny', 'preprocessing': 'none', 'image_height': 224,
        'image_width': 224, 'lr': 3e-4, 'weight_decay': 1e-4, 'batch_size': 64,
        'lr_schedule': 'cosine', 'warmup_epochs': 0, 'min_lr_ratio': 0.0,
        'epochs': 30, 'patience': None, 'augmentation': 'none', 'sampling': 'slice_uniform',
        'input_channels': 3, 'loss': 'cross_entropy', 'precision': 'amp_bf16',
        'patient_aggregation': 'mean_logit', 'drop_path': .1},
}


class RecipeParser(argparse.ArgumentParser):
    """Apply recipe defaults after parsing, preserving every explicit CLI override."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Require exact flag spelling so precedence is unambiguous."""
        kwargs['allow_abbrev'] = False
        super().__init__(*args, **kwargs)

    def parse_args(self, args: list[str] | None = None, namespace: argparse.Namespace | None = None) -> argparse.Namespace:
        """Use the same recipe resolution for previews, submission and real execution."""
        tokens = list(sys.argv[1:] if args is None else args)
        result = super().parse_args(tokens, namespace)
        explicit = {self._option_string_actions[token.split('=', 1)[0]].dest
                    for token in tokens if token.split('=', 1)[0] in self._option_string_actions}
        for key, value in RECIPES[getattr(result, 'recipe', 'custom')].items():
            if key not in explicit:
                setattr(result, key, value)
        return result


@dataclass(frozen=True)
class ExecutionControls:
    """Define tensor shape, objective, compute dtype and patient decision rule."""

    input_channels: int = 1
    loss: str = 'weighted_bce'
    precision: str = 'fp32'
    patient_aggregation: str = 'mean_probability'
    drop_path: float | None = None
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0

    def __post_init__(self) -> None:
        """Reject unsupported shapes/objectives rather than silently falling back."""
        if type(self.input_channels) is not int or self.input_channels not in (1, 3):
            raise ValueError('Input channels must be one or three repeated grayscale channels.')
        if self.loss not in ('bce', 'weighted_bce', 'cross_entropy', 'weighted_cross_entropy'):
            raise ValueError('Unsupported loss.')
        if self.precision not in ('fp32', 'amp_fp16', 'amp_bf16'):
            raise ValueError('Unsupported precision.')
        if self.patient_aggregation not in ('mean_probability', 'mean_logit'):
            raise ValueError('Unsupported patient aggregation.')
        if self.drop_path is not None and (type(self.drop_path) not in (int, float)
                or not math.isfinite(self.drop_path) or not 0 <= self.drop_path < 1):
            raise ValueError('DropPath must be finite in [0, 1).')

        if (type(self.label_smoothing) not in (int, float) or not math.isfinite(self.label_smoothing)
                or not 0 <= self.label_smoothing < 1):
            raise ValueError('Label smoothing must be finite in [0, 1).')
        if (type(self.mixup_alpha) not in (int, float) or not math.isfinite(self.mixup_alpha)
                or not 0 <= self.mixup_alpha <= 1):
            raise ValueError('Mixup alpha must be finite in [0, 1]; zero disables mixing.')

    @property
    def output_classes(self) -> int:
        """CE has two raw logits; BCE has one AD-versus-NC logit."""
        return 2 if 'cross_entropy' in self.loss else 1

    def to_dict(self) -> dict[str, Any]:
        """Bind executable controls to a closed algorithm version."""
        values = asdict(self)
        if self.label_smoothing == 0 and self.mixup_alpha == 0:
            # Retain exact legacy metadata and RNG behavior with switches disabled.
            del values['label_smoothing'], values['mixup_alpha']
            return {**values, 'algorithm': 'scratch_execution_v1'}
        return {**values, 'algorithm': 'scratch_execution_v2'}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> 'ExecutionControls':
        """Reject missing/extra fields and changed control versions in checkpoints."""
        if type(value) is not dict:
            raise ValueError('Incomplete or unsupported execution controls.')
        fields = set(cls.__dataclass_fields__)
        if value.get('algorithm') == 'scratch_execution_v1':
            fields -= {'label_smoothing', 'mixup_alpha'}
        if set(value) != fields | {'algorithm'}:
            raise ValueError('Incomplete or unsupported execution controls.')
        controls = cls(**{name: value[name] for name in fields})
        if value != controls.to_dict():
            raise ValueError('Changed execution control version.')
        return controls


def add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose orthogonal switches; defaults retain historical execution."""
    parser.add_argument('--recipe', choices=tuple(RECIPES), default='custom',
                        help='Recipe defaults; explicit options always win. peer_tiny is an approximation.')
    parser.add_argument('--input-channels', type=int, choices=(1, 3), default=1)
    parser.add_argument('--loss', choices=('bce', 'weighted_bce', 'cross_entropy', 'weighted_cross_entropy'), default='weighted_bce')
    parser.add_argument('--precision', choices=('fp32', 'amp_fp16', 'amp_bf16'), default='fp32')
    parser.add_argument('--patient-aggregation', choices=('mean_probability', 'mean_logit'), default='mean_probability')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='Training targets: (1-epsilon)*target + epsilon/2; zero disables smoothing.')
    parser.add_argument('--mixup-alpha', type=float, default=0.0,
                        help='Train-only batch Mixup Beta(alpha, alpha), range [0,1]; zero disables mixing.')
    parser.add_argument('--drop-path', type=float, default=None, help='Omitted: architecture default; zero disables stochastic depth.')


def controls_from_args(args: argparse.Namespace) -> ExecutionControls:
    """Keep legacy Python callers without the new fields compatible."""
    return ExecutionControls(**{name: getattr(args, name, field.default)
                                for name, field in ExecutionControls.__dataclass_fields__.items()})


def checkpoint_controls(config: dict[str, Any]) -> ExecutionControls:
    """New shapes require format four; older checkpoints keep their old semantics."""
    value = config.get('execution_controls')
    controls = ExecutionControls.from_dict(value) if value is not None else ExecutionControls()
    if config['checkpoint_format_version'] == 4:
        if value is None:
            raise ValueError('Format four requires execution controls.')
    elif controls != ExecutionControls():
        raise ValueError('Historical checkpoint cannot redefine execution controls.')
    return controls


def attach_controls(model: nn.Module, controls: ExecutionControls) -> None:
    """Attach runtime metadata without adding weights or changing state-dict names."""
    model.execution_controls = controls


def model_controls(model: nn.Module) -> ExecutionControls:
    """Unmodified diagnostic/legacy models execute using historical defaults."""
    return getattr(model, 'execution_controls', ExecutionControls())


def validate_device(controls: ExecutionControls, device: torch.device) -> None:
    """AMP is explicit CUDA-only; unsupported hardware never falls back to FP32."""
    if controls.precision != 'fp32':
        if device.type != 'cuda':
            raise ValueError('AMP controls require CUDA; choose --precision fp32 for CPU.')
        if controls.precision == 'amp_bf16' and not torch.cuda.is_bf16_supported():
            raise ValueError('This GPU does not support BF16 AMP.')


def autocast_context(model: nn.Module, device: torch.device) -> Any:
    """Use the configured autocast dtype consistently in training/replay/profiling."""
    precision = model_controls(model).precision
    if precision == 'fp32':
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16 if precision == 'amp_fp16' else torch.bfloat16)


def model_input(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Repeat normalized grayscale channels without altering pixels or geometry."""
    return images.repeat(1, 3, 1, 1) if model_controls(model).input_channels == 3 else images


def ad_logits(model: nn.Module, logits: torch.Tensor) -> torch.Tensor:
    """Convert raw CE logits to the AD-minus-NC margin in FP32."""
    logits = logits.float()
    return logits[:, 1] - logits[:, 0] if model_controls(model).output_classes == 2 else logits


def make_criterion(controls: ExecutionControls, pos_weight: float, device: torch.device) -> nn.Module:
    """Use original training counts; balanced sampling supplies a unit class prior."""
    if controls.output_classes == 2:
        weight = torch.tensor([1., pos_weight], device=device) if controls.loss == 'weighted_cross_entropy' else None
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=controls.label_smoothing)
    base = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight if controls.loss == 'weighted_bce' else 1., device=device))
    return SmoothedBinaryLoss(base, controls.label_smoothing) if controls.label_smoothing else base


class SmoothedBinaryLoss(nn.Module):
    """Smooth a binary target toward a uniform prior before weighted BCE.

    Uses the same two-class uniform-mixture convention as PyTorch CE:
    https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    Class weight is applied by BCE after smoothing, never fitted on evaluation data.
    """

    def __init__(self, criterion: nn.Module, epsilon: float) -> None:
        """Keep the original BCE reduction and positive-class weighting."""
        super().__init__()
        self.criterion, self.epsilon = criterion, epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Use fractional targets without rounding them to class indices."""
        return self.criterion(logits, targets * (1 - self.epsilon) + self.epsilon / 2)


def mixup_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Mix only the current training minibatch, with one Beta draw and permutation.

    Zhang et al., mixup: Beyond Empirical Risk Minimization (ICLR 2018):
    https://arxiv.org/abs/1710.09412
    No sample/label is fetched from another loader or role. Disabled mixing
    consumes no RNG draws, preserving legacy runs; singleton batches are unchanged.
    """
    if alpha == 0 or labels.numel() < 2:
        return images, labels, labels, 1.0
    coefficient = float(torch.distributions.Beta(alpha, alpha).sample().item())
    permutation = torch.randperm(labels.numel(), device=images.device)
    return coefficient * images + (1 - coefficient) * images[permutation], labels, labels[permutation], coefficient
