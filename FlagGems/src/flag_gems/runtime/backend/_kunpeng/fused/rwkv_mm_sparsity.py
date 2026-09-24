"""Kunpeng backend override for rwkv_mm_sparsity.

This module is loaded automatically by replace_customized_ops() when
GEMS_VENDOR=kunpeng. The rwkv_mm_sparsity function defined here replaces
the default implementation in flag_gems.fused.rwkv_mm_sparsity at runtime,
without modifying the original operator file.

The override runs a sparse GEMV kernel that reads only the rows of v that k
selects, fusing the gather into the accumulation. ArmPL remains the path for
float64, and the default Triton kernel is the last resort.
"""

import os

import torch

from .. import rwkv_mm_sparsity_triton as _triton_mod
from ..rwkv_mm_sparsity_armpl import rwkv_mm_sparsity_armpl
from flag_gems.fused.rwkv_mm_sparsity import (
    rwkv_mm_sparsity as _rwkv_mm_sparsity_triton_default,
)


def _fallback(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    try:
        return rwkv_mm_sparsity_armpl(k, v)
    except RuntimeError as err:
        # Graceful fallback when ArmPL is unavailable.
        if "Could not load ArmPL" not in str(err):
            raise
    except TypeError:
        # Unsupported dtype for ArmPL path, try Triton path.
        pass
    return _rwkv_mm_sparsity_triton_default(k, v)


_triton_mod._fallback = _fallback

# TRITON_USE_ARMPL=1 sends every dtype through ArmPL.
if os.environ.get("TRITON_USE_ARMPL", "0") != "0":
    rwkv_mm_sparsity = _fallback
else:
    rwkv_mm_sparsity = _triton_mod.rwkv_mm_sparsity_triton
