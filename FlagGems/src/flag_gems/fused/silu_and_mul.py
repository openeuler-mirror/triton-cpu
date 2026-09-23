import logging
from dataclasses import replace

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import get_codegen_config

logger = logging.getLogger(__name__)


_SILU_AND_MUL_CONFIG = replace(
    get_codegen_config(), contiguous_tiles_per_cta=True
)


@pointwise_dynamic(
    promotion_methods=[(0, 1, "DEFAULT")], config=_SILU_AND_MUL_CONFIG
)
@triton.jit
def silu_and_mul_kernel(x, y):
    x_fp32 = x.to(tl.float32)
    x_silu = tl.fdiv(x_fp32, (1.0 + tl.exp(-x_fp32)))
    return x_silu * y


# ---------------------------------------------------------------------------
# Optimized contiguous float32 fast path.
#
# On the triton-shared CPU backend a scalar range-loop lowers to a plain C loop
# that the host compiler (LLVM) auto-vectorizes.  Tuning (see silu_tune.py)
# shows this is ~4x faster than the masked tensor-vector code the generic
# pointwise_dynamic kernel emits, while producing identical float32 results.
# We only take this path for the common contiguous, same-shape, float32 CPU
# case and fall back to silu_and_mul_kernel for everything else.
# ---------------------------------------------------------------------------
_SILU_AND_MUL_FAST_BLOCK = 1024


@triton.jit
def _silu_and_mul_fast_kernel(X, Y, O, N, BLOCK: tl.constexpr):
    start = tl.program_id(0) * BLOCK
    end = tl.minimum(start + BLOCK, N)
    for i in range(start, end):
        a = tl.load(X + i).to(tl.float32)
        b = tl.load(Y + i).to(tl.float32)
        tl.store(O + i, tl.fdiv(a, 1.0 + tl.exp(-a)) * b)


def _can_use_fast_path(A, B):
    return (
        A.is_cpu
        and B.is_cpu
        and A.dtype == torch.float32
        and B.dtype == torch.float32
        and A.shape == B.shape
        and A.is_contiguous()
        and B.is_contiguous()
    )


def _silu_and_mul_fast(A, B, out=None):
    if out is None:
        out = torch.empty_like(A)
    n = A.numel()
    if n:
        grid = (triton.cdiv(n, _SILU_AND_MUL_FAST_BLOCK),)
        _silu_and_mul_fast_kernel[grid](A, B, out, n, _SILU_AND_MUL_FAST_BLOCK)
    return out


@pointwise_dynamic(
    promotion_methods=[(0, 1, 2, "DEFAULT"), (0, 1, 2, "DEFAULT")], num_outputs=2
)
@triton.jit
def silu_and_mul_grad_kernel(x, y, dgrad):
    x_fp32 = x.to(tl.float32)
    sig = 1 / (1 + tl.exp(-x_fp32))
    x_silu = x_fp32 * sig
    d_x_silu = sig * (1 + x_fp32 * (1 - sig))
    dx = d_x_silu * dgrad * y
    dy = dgrad * x_silu
    return dx, dy


class SiluAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        logger.debug("GEMS SILU AND MUL FORWARD")
        if _can_use_fast_path(A, B):
            return _silu_and_mul_fast(A, B)
        return silu_and_mul_kernel(A, B)

    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A, grad_B = silu_and_mul_grad_kernel(A, B, grad_output)
        return grad_A, grad_B


def silu_and_mul(A, B):
    return SiluAndMul.apply(A, B)


def silu_and_mul_out(A, B, out):
    if (
        _can_use_fast_path(A, B)
        and out.dtype == torch.float32
        and out.shape == A.shape
        and out.is_contiguous()
    ):
        return _silu_and_mul_fast(A, B, out)
    silu_and_mul_kernel(A, B, out0=out)
    return out
