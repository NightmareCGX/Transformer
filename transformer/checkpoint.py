import glob
import os
from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer

from transformer.schedulers import Scheduler


def save_checkpoint(
    path: str,
    step: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[Scheduler] = None,
    scaler: Optional[GradScaler] = None,
    extra: Optional[Dict[str, Any]] = None,
    keep_last: int = 0,
) -> None:
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "extra": extra or {},
    }

    os.makedirs(path, exist_ok=True)
    checkpoint_path = os.path.join(path, f"checkpoint_{step:07d}.pt")
    torch.save(payload, checkpoint_path)

    if keep_last > 0:
        checkpoint_files = sorted(glob.glob(os.path.join(path, "checkpoint_*.pt")))
        if len(checkpoint_files) > keep_last:
            for checkpoint_file in checkpoint_files[:-keep_last]:
                try:
                    os.remove(checkpoint_file)
                except FileNotFoundError:
                    pass  # File doesn't exist - nothing to remove
                except PermissionError:
                    print(f"Permission denied: cannot remove {checkpoint_file}")
                except OSError as e:
                    print(f"Error removing file: {e}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Scheduler] = None,
    scaler: Optional[GradScaler] = None,
    map_location: Union[str, torch.device] = "cpu",
) -> tuple:
    payload = torch.load(checkpoint_path, map_location=map_location)

    model.load_state_dict(payload["model"])

    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])

    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])

    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])

    step = payload.get("step")
    extra = payload.get("extra", {}) or {}

    return step, extra
