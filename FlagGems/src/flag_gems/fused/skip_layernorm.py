import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def skip_layer_norm_kernel(
    Y,
    X,
    R,
    W,
    B,
    y_stride_r,
    y_stride_c,
    x_stride_r,
    x_stride_c,
    r_stride_r,
    r_stride_c,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    Y += pid * y_stride_r
    X += pid * x_stride_r
    R += pid * r_stride_r

    m = 0.0
    s = 0.0
    cnt = 0
    num_steps = tl.cdiv(N, BLOCK_SIZE)

    for step in range(0, num_steps, 1):
        start_n = step * BLOCK_SIZE
        n_offsets = start_n + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N
        x = tl.load(X + n_offsets * x_stride_c, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R + n_offsets * r_stride_c, mask=mask, other=0.0).to(tl.float32)
        x = tl.where(mask, x + r, 0.0)
        valid_cnt = tl.sum(mask.to(tl.int32))
        if step == 0:
            m = tl.sum(x) / valid_cnt
            s = tl.sum(tl.where(mask, (x - m) * (x - m), 0.0))
            cnt = valid_cnt
        else:
            new_m = m + tl.sum(tl.where(mask, x - m, 0.0)) / (cnt + valid_cnt)
            new_s = s + tl.sum(tl.where(mask, (x - new_m) * (x - m), 0.0))
            cnt += valid_cnt
            m = new_m
            s = new_s

    mean = m
    var = s / cnt
    rstd = 1 / tl.sqrt(var + eps)

    for step in range(0, num_steps, 1):
        start_n = step * BLOCK_SIZE
        n_offsets = start_n + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N
        x = tl.load(X + n_offsets * x_stride_c, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R + n_offsets * r_stride_c, mask=mask, other=0.0).to(tl.float32)
        x = tl.where(mask, x + r, 0.0)
        tl.store(R + n_offsets * r_stride_c, x, mask=mask)
        w = tl.load(W + n_offsets, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + n_offsets, mask=mask, other=0.0).to(tl.float32)
        y = w * (x - mean) * rstd + b
        y = y.to(Y.dtype.element_ty)
        tl.store(Y + n_offsets * y_stride_c, y, mask=mask)


class SkipLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, normalized_shape, weight, bias, eps=1e-5):
        logger.debug("GEMS SKIP LAYERNORM FORWARD")
        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)

        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        y = torch.empty_like(x)

        BLOCK_SIZE = 4096

        with torch_device_fn.device(x.device):
            skip_layer_norm_kernel[M,](
                y, x, residual, weight, bias, N, 1, N, 1, N, 1, N, eps, BLOCK_SIZE
            )
        return y


def skip_layer_norm(x, residual, normalized_shape, weight, bias, eps=1e-5):
    return SkipLayerNorm.apply(x, residual, normalized_shape, weight, bias, eps)
