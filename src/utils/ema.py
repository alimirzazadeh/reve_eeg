"""Exponential moving average of model parameters and buffers.

Update rule: ema = decay * ema + (1 - decay) * model, applied per
optimizer step. Effective decay is warmed up early via
`min(decay, (1 + step) / (10 + step))` so the EMA tracks fresh weights
before saturating at the target decay.
"""

from copy import deepcopy

import torch
import torch.nn as nn


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.module = deepcopy(_unwrap(model))
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        msd = _unwrap(model).state_dict()
        for k, v in self.module.state_dict().items():
            mv = msd[k]
            if v.is_floating_point():
                v.mul_(d).add_(mv.detach(), alpha=1.0 - d)
            else:
                v.copy_(mv)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict)
