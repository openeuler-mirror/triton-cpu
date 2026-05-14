import logging
import math

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_kernel(
    input_ptr,  # pointer to the input
    residual_ptr,  # pointer to the residual
    w_ptr,  # pointer to the weights
    in_stride_r,  # how much to increase the pointer when moving by 1 row
    in_stride_c,  # how much to increase the pointer when moving by 1 col
    r_stride_r,  # how much to increase the pointer when moving by 1 row
    r_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in in_ptr
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid = tle.program_id(0)
    input_ptr += pid * in_stride_r
    residual_ptr += pid * r_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(input_ptr + cols * in_stride_c, mask, other=0.0).to(cdtype)
    r = tl.load(residual_ptr + cols * r_stride_c, mask, other=0.0).to(cdtype)

    x += r
    # write back to residual
    tl.store(residual_ptr + cols * r_stride_c, x, mask=mask)

    var = tl.sum(x * x / N, axis=0)
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms * w).to(cdtype)
    # write back to input
    tl.store(input_ptr + cols * in_stride_c, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_loop_kernel(
    input_ptr,  # pointer to the input
    residual_ptr,  # pointer to the residual
    w_ptr,  # pointer to the weights
    in_stride_r,  # how much to increase the pointer when moving by 1 row
    in_stride_c,  # how much to increase the pointer when moving by 1 col
    r_stride_r,  # how much to increase the pointer when moving by 1 row
    r_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in in_ptr
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid = tle.program_id(0)
    input_ptr += pid * in_stride_r
    residual_ptr += pid * r_stride_r

    # First pass: compute variance (sum of squares) after adding residual
    var_sum = 0.0
    num_steps = tl.cdiv(N, BLOCK_SIZE)

    for step in range(0, num_steps, 1):
        start_n = step * BLOCK_SIZE
        n_offsets = start_n + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N
        x = tl.load(input_ptr + n_offsets * in_stride_c, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(residual_ptr + n_offsets * r_stride_c, mask=mask, other=0.0).to(tl.float32)
        x = tl.where(mask, x + r, 0.0)
        var_sum += tl.sum(x * x).to(tl.float32)

    var = var_sum / N
    rrms = 1 / tl.sqrt(var + eps)

    # Second pass: normalize, write back to residual and input
    for step in range(0, num_steps, 1):
        start_n = step * BLOCK_SIZE
        n_offsets = start_n + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N
        x = tl.load(input_ptr + n_offsets * in_stride_c, mask=mask, other=0.0).to(cdtype)
        r = tl.load(residual_ptr + n_offsets * r_stride_c, mask=mask, other=0.0).to(cdtype)
        x = x + r
        # write back to residual
        tl.store(residual_ptr + n_offsets * r_stride_c, x, mask=mask)
        w = tl.load(w_ptr + n_offsets, mask=mask, other=0.0)
        y = (x * rrms * w).to(cdtype)
        # write back to input
        tl.store(input_ptr + n_offsets * in_stride_c, y, mask=mask)


def fused_add_rms_norm(x, residual, normalized_shape, weight, eps=1e-5):
    """
    This function performs fused residual addition and RMS normalization **in-place**.
    Both `x` and `residual` tensors will be modified. Use with caution if these tensors
    are reused elsewhere or require gradients.
    """
    logger.debug(
        "GEMS FUSED_ADD_RMS_NORM FORWARD, [input shape]: %s, [residual shape]: %s, [weight shape]: %s",
        x.size(),
        residual.size(),
        weight.size(),
    )
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()

    with torch_device_fn.device(x.device):
        if N < 4096:
            BLOCK_SIZE = triton.next_power_of_2(N)
            fused_add_rms_norm_kernel[M,](
                x, residual, weight, N, 1, N, 1, N, eps, BLOCK_SIZE
            )
        else:
            BLOCK_SIZE = 4096
            fused_add_rms_norm_loop_kernel[M,](
                x, residual, weight, N, 1, N, 1, N, eps, BLOCK_SIZE
            )
    return x, residual
