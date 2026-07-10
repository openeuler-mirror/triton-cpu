"""Kunpeng backend override for rwkv_mm_sparsity.

This module is loaded automatically by replace_customized_ops() when
GEMS_VENDOR=kunpeng. The rwkv_mm_sparsity function defined here replaces
the default implementation in flag_gems.fused.rwkv_mm_sparsity at runtime,
without modifying the original operator file.

The override tries the ArmPL cblas_*gemv path first (on compressed non-zero
entries), and falls back to the default Triton kernel if ArmPL is unavailable
or the dtype is unsupported.
"""

import torch

from ..rwkv_mm_sparsity_armpl import rwkv_mm_sparsity_armpl
from flag_gems.fused.rwkv_mm_sparsity import (
    rwkv_mm_sparsity as _rwkv_mm_sparsity_triton,
)


def rwkv_mm_sparsity(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """ArmPL when available, else the default Triton kernel."""
    try:
        return rwkv_mm_sparsity_armpl(k, v)
    except RuntimeError as err:
        # Graceful fallback when ArmPL is unavailable.
        if "Could not load ArmPL" not in str(err):
            raise
    except TypeError:
        # Unsupported dtype for ArmPL path, try Triton path.
        pass
    return _rwkv_mm_sparsity_triton(k, v)
