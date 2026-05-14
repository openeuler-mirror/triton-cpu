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
    Y,  # pointer to the output
    X,  # pointer to the input
    R,  # pointer to the residual
    W,  # pointer to the weights
    B,  # pointer to the biases
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    r_stride_r,  # how much to increase the pointer when moving by 1 row
    r_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    Y += pid * y_stride_r
    X += pid * x_stride_r
    R += pid * r_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    r = tl.load(R + cols * r_stride_c, mask, other=0.0).to(tl.float32)

    x += r

    mean = tl.sum(x, axis=0) / N

    # Compute variance
    _var = tl.where(mask, x - mean, 0.0)
    _var = _var * _var
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0).to(tl.float32)

    x_hat = (x - mean) * rstd
    y = w * x_hat + b
    y = y.to(Y.dtype.element_ty)
    tl.store(Y + cols * y_stride_c, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def skip_layer_norm_loop_kernel(
    Y,  # pointer to the output
    X,  # pointer to the input
    R,  # pointer to the residual
    W,  # pointer to the weights
    B,  # pointer to the biases
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    r_stride_r,  # how much to increase the pointer when moving by 1 row
    r_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    Y += pid * y_stride_r
    X += pid * x_stride_r
    R += pid * r_stride_r

    # First pass: compute mean and variance with Welford's online algorithm
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
        # First batch: initialize directly
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

    # Second pass: normalize and apply weight/bias, also store residual
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

        with torch_device_fn.device(x.device):
            if N < 4096:
                BLOCK_SIZE = triton.next_power_of_2(N)
                skip_layer_norm_kernel[M,](
                    y, x, residual, weight, bias, N, 1, N, 1, N, 1, N, eps, BLOCK_SIZE
                )
            else:
                BLOCK_SIZE = 4096
                skip_layer_norm_loop_kernel[M,](
                    y, x, residual, weight, bias, N, 1, N, 1, N, 1, N, eps, BLOCK_SIZE
                )
        return y


def skip_layer_norm(x, residual, normalized_shape, weight, bias, eps=1e-5):
    return SkipLayerNorm.apply(x, residual, normalized_shape, weight, bias, eps)
