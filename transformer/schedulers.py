import math
from typing import Any, Dict, Protocol

from torch.optim import Optimizer


class Scheduler(Protocol):
    def state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None: ...
    def step(self) -> None: ...


class Noam:
    """
    Transformer-style noam schedule:
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """

    def __init__(self, optimizer: Optimizer, d_model: int, warmup_steps: int, last_epoch: int = -1):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = max(1, warmup_steps)
        self.last_epoch = max(0, last_epoch)

    def step(self):
        self.last_epoch += 1
        lr = self.d_model ** (-0.5) * min(
            self.last_epoch ** (-0.5), self.last_epoch * self.warmup_steps ** (-1.5)
        )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def state_dict(self):
        return {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state_dict: dict):
        self.d_model = state_dict["d_model"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.last_epoch = state_dict["last_epoch"]


class Cosine:
    """
    Cosine schedule with warmup:
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))
    progress linearly increases from 0 to 1 except warmup
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_min: float,
        lr_max: float,
        warmup_steps: int,
        max_steps: int,
        last_epoch: int = -1,
    ):
        self.optimizer = optimizer
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warmup_steps = max(1, warmup_steps)
        self.max_steps = max(warmup_steps, max_steps)
        self.last_epoch = max(0, last_epoch)

    def step(self):
        self.last_epoch += 1
        if self.last_epoch <= self.warmup_steps:
            lr = self.lr_max * float(self.last_epoch) / float(self.warmup_steps)
        else:
            progress = float(self.last_epoch - self.warmup_steps) / float(
                max(1, self.max_steps - self.warmup_steps)
            )
            progress = min(progress, 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                1.0 + math.cos(math.pi * progress)
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def state_dict(self):
        return {
            "lr_min": self.lr_min,
            "lr_max": self.lr_max,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state_dict: dict):
        self.lr_min = state_dict["lr_min"]
        self.lr_max = state_dict["lr_max"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.max_steps = state_dict["max_steps"]
        self.last_epoch = state_dict["last_epoch"]
