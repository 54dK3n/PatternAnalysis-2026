"""Prospective controls for matched scratch experiments; no trained weights reused."""

from collections import defaultdict
from contextlib import contextmanager
import hashlib
import math
import random
from typing import Any, Iterator

import torch
from torch import nn

from models import create_model
from utils.runtime import seed_everything


@contextmanager
def preserved_rng(model: nn.Module | None = None) -> Iterator[None]:
    """Restore Python/CPU/CUDA RNG and every module mode even after exceptions.

    Call only with dedicated nontraining loaders. A loader's private generator
    is intentionally not rewound; the training loader is never used here.
    """
    python_state = random.getstate()
    flags = [(module, module.training) for module in model.modules()] if model is not None else []
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        for module, mode in flags:
            module.training = mode


def state_digest(state: dict[str, torch.Tensor]) -> str:
    """Hash names, shapes, dtypes and contiguous CPU tensor bytes deterministically."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        cpu = value.detach().cpu().contiguous()
        digest.update(f'{name}:{tuple(cpu.shape)}:{cpu.dtype}\n'.encode())
        digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def matched_model(name: str, seed: int) -> tuple[nn.Module, dict]:
    """Share compatible tensors with a freshly generated reference for this seed.

    The overlapping stem uses a separate seed stream (seed + 2000003). Its
    different shape cannot share reference weights. No checkpoint is read.
    Reset training RNG after construction to keep DropPath draws comparable.
    """
    with preserved_rng():
        seed_everything(seed)
        reference = create_model('convnext_lite_v1').state_dict()
        seed_everything(seed + 2000003)
        model = create_model(name)
        state = model.state_dict()
        shared, independent = [], []
        for key, value in state.items():
            if key in reference and value.shape == reference[key].shape:
                state[key] = reference[key].clone()
                shared.append(key)
            else:
                independent.append(key)
        model.load_state_dict(state)
        info = {'reference_initial_sha256': state_digest(reference),
                'initial_state_sha256': state_digest(model.state_dict()),
                'shared_tensor_names': shared, 'independent_tensor_names': independent,
                'independent_initialization_seed': seed + 2000003,
                'reference_source': 'fresh_random_initialization_no_checkpoint',
                'training_rng_reset_seed': seed}
    seed_everything(seed)
    return model, info


def optimizer_groups(model: nn.Module, weight_decay: float, policy: str) -> tuple[list[dict], list[dict]]:
    """Assign each trainable parameter exactly once, exporting all names/counts."""
    if policy not in ('all_trainable_parameters', 'zero_for_1d_parameters'):
        raise ValueError('Unknown weight-decay policy.')
    groups: dict[float, list] = defaultdict(list)
    names: dict[float, list[str]] = defaultdict(list)
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError('Unexpected aliased trainable parameter.')
        seen.add(id(parameter))
        decay = 0.0 if policy == 'zero_for_1d_parameters' and parameter.ndim == 1 else weight_decay
        groups[decay].append(parameter)
        names[decay].append(name)
    if len(seen) != sum(p.requires_grad for p in model.parameters()):
        raise ValueError('Optimizer parameter coverage failed.')
    metadata = [{'weight_decay': d, 'parameter_names': names[d],
                 'parameter_tensors': len(groups[d]),
                 'scalar_parameters': sum(p.numel() for p in groups[d])} for d in groups]
    return [{'params': values, 'weight_decay': d} for d, values in groups.items()], metadata


class MinimumCheckpoint:
    """Keep the earliest epoch attaining each strict finite minimum."""

    def __init__(self) -> None:
        """Initialize an empty prospective selector."""
        self.loss = math.inf
        self.epoch = 0

    def update(self, loss: float, epoch: int) -> bool:
        """Return whether the candidate strictly improves the recorded loss."""
        if not math.isfinite(loss):
            raise ValueError('Non-finite checkpoint-selection loss.')
        if loss < self.loss:
            self.loss, self.epoch = loss, epoch
            return True
        return False


def gradient_snapshot(model: nn.Module, epoch: int, batch: int) -> list[dict]:
    """Read gradient/parameter L2 without modifying tensors or optimizer state."""
    buckets: dict[str, list] = defaultdict(list)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('classifier.'):
            group = 'head'
        elif name.startswith('stages.'):
            pieces = name.split('.')
            component = 'mlp' if pieces[3] in ('expand', 'project') else pieces[3]
            group = '.'.join(pieces[:3] + [component])
        else:
            group = name.rsplit('.', 1)[0]
        buckets[group].append(parameter)
    result = []
    for name, parameters in sorted(buckets.items()):
        parameter_sq = gradient_sq = 0.0
        missing = zero = scalars = 0
        for parameter in parameters:
            value = parameter.detach().double()
            parameter_sq += value.square().sum().item()
            scalars += parameter.numel()
            if parameter.grad is None:
                missing += 1
            else:
                grad = parameter.grad.detach().double()
                if not torch.isfinite(grad).all().item():
                    raise ValueError('Non-finite gradient in diagnostic snapshot.')
                norm_sq = grad.square().sum().item()
                gradient_sq += norm_sq
                zero += int(norm_sq == 0)
        pn, gn = math.sqrt(parameter_sq), math.sqrt(gradient_sq)
        result.append({'epoch': epoch, 'batch': batch, 'group': name, 'gradient_l2': gn,
                       'parameter_l2': pn, 'gradient_to_parameter_l2': gn / max(pn, 1e-12),
                       'parameter_tensors': len(parameters), 'scalar_parameters': scalars,
                       'missing_grad_tensors': missing, 'zero_grad_tensors': zero,
                       'normalization': 'gradient_l2 / max(parameter_l2, 1e-12)'})
    return result


def eligible_repeat(records: dict[str, dict]) -> dict:
    """Apply the preregistered engineering gate to primary slice selections only."""
    required = {'R00', 'R01', 'R02', 'R03', 'R04', 'R05'}
    if set(records) != required:
        return {'status': 'blocked_incomplete_core', 'selected': None}
    reference = records['R00']['metrics']['slice']
    eligible = []
    for case in sorted(required - {'R00'}):
        result = records[case]
        metric = result['metrics']['slice']
        gain = metric['accuracy'] - reference['accuracy']
        auc = metric['auroc']
        recall_drop = reference['per_class']['AD']['recall'] - metric['per_class']['AD']['recall']
        if (gain >= .02 - 1e-12 and auc is not None and reference['auroc'] is not None
                and auc >= reference['auroc'] and recall_drop <= .02 + 1e-12):
            peak = result['resources']['training_peak_cuda_allocated_mib']
            eligible.append((case, metric['accuracy'], auc, peak))
    ranked = sorted(eligible, key=lambda v: (-v[1], -v[2], v[3] if v[3] is not None else math.inf, v[0]))
    return {'status': 'eligible' if ranked else 'stop_and_review_diagnostics',
            'selected': ranked[0][0] if ranked else None,
            'eligible_ranked': [v[0] for v in ranked], 'not_a_significance_test': True,
            'criteria': {'min_accuracy_gain': .02, 'min_auroc_gain': 0, 'max_AD_recall_drop': .02}}
