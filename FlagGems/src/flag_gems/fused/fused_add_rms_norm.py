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
    input_ptr,
    residual_ptr,
    w_ptr,
    in_stride_r,
    in_stride_c,
    r_stride_r,
    r_stride_c,
    N,
    eps,
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

    for step in range(0, num_steps, 1):
        start_n = step * BLOCK_SIZE
        n_offsets = start_n + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N
        x = tl.load(input_ptr + n_offsets * in_stride_c, mask=mask, other=0.0).to(cdtype)
        r = tl.load(residual_ptr + n_offsets * r_stride_c, mask=mask, other=0.0).to(cdtype)
        x = x + r
        tl.store(residual_ptr + n_offsets * r_stride_c, x, mask=mask)
        w = tl.load(w_ptr + n_offsets, mask=mask, other=0.0)
        y = (x * rrms * w).to(cdtype)
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

    BLOCK_SIZE = 4096

    with torch_device_fn.device(x.device):
        fused_add_rms_norm_kernel[M,](
            x, residual, weight, N, 1, N, 1, N, eps, BLOCK_SIZE
        )
    return x, residual
