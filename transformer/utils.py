import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_posenc(max_len: int, d_model: int):
    posenc = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len).unsqueeze(1)
    denominator_inverse = torch.pow(10000, -1 * torch.arange(0, d_model, 2) / d_model)
    posenc[:, 0::2] = torch.sin(pos * denominator_inverse)
    posenc[:, 1::2] = torch.cos(pos * denominator_inverse)

    return posenc.unsqueeze(0)


def causal_mask(L: int, dtype: torch.dtype = torch.float32):
    matrix = torch.zeros((1, 1, L, L), dtype=dtype)
    matrix = matrix.masked_fill(
        torch.triu(torch.ones((1, 1, L, L), dtype=torch.bool), diagonal=1),
        float("-inf"),
    )

    return matrix


def padding_mask(key_padding: torch.Tensor, dtype: torch.dtype = torch.float32):
    assert key_padding.dtype == torch.bool, "key_padding must be a bool tensor"
    B, L = key_padding.shape
    matrix = torch.zeros((B, 1, 1, L), dtype=dtype)
    matrix = matrix.masked_fill(key_padding[:, None, None, :], float("-inf"))

    return matrix
