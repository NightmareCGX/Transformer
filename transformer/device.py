from __future__ import annotations

from typing import Union

import torch

DeviceLike = Union[str, torch.device]


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0" if torch.cuda.device_count() > 0 else "cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(device: DeviceLike = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    s = str(device).lower()
    if s == "auto":
        return _auto_device()
    if "cuda" in s:
        return torch.device(s) if torch.cuda.is_available() else torch.device("cpu")
    if s == "mps":
        return torch.device(s) if torch.backends.mps.is_available() else torch.device("cpu")
    if s == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device string: {device}")


def setup_global_device(device: torch.device):
    torch.set_default_device(device)

    print(f"Default device set to: {torch.get_default_device()}")
