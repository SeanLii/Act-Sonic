"""Training helpers shared by the CLI and smoke tests."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .model import ACTPolicy


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_loss(
    model: ACTPolicy,
    outputs: dict[str, Tensor | None],
    target_actions: Tensor,
    is_pad: Tensor,
    kl_weight: float,
    pad_weight: float,
) -> dict[str, Tensor]:
    predicted = outputs["actions"]
    pad_logits = outputs["pad_logits"]
    mu, logvar = outputs["mu"], outputs["logvar"]
    assert predicted is not None and pad_logits is not None and mu is not None and logvar is not None
    valid = (~is_pad).to(dtype=predicted.dtype)
    per_token_l1 = (predicted - target_actions).abs().mean(dim=-1)
    l1 = (per_token_l1 * valid).sum() / valid.sum().clamp_min(1.0)
    pad_bce = F.binary_cross_entropy_with_logits(pad_logits, is_pad.to(pad_logits.dtype))
    kl = model.kl_divergence(mu, logvar)
    total = l1 + kl_weight * kl + pad_weight * pad_bce
    return {"loss": total, "l1": l1, "kl": kl, "pad_bce": pad_bce}
