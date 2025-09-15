from typing import Union

import torch


def setup_global_dtype(dtype: Union[str, torch.dtype]) -> None:
    dtype_mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float64": torch.float64,
        "double": torch.float64,
        "fp64": torch.float64,
    }

    if isinstance(dtype, str):
        resolved_dtype = dtype_mapping.get(dtype.lower())
        if resolved_dtype is None:
            raise ValueError(f"Unsupported dtype string. Supported: {list(dtype_mapping.keys())}")
    else:
        resolved_dtype = dtype

    torch.set_default_dtype(resolved_dtype)

    print(f"Default dtype set to: {torch.get_default_dtype()}")
