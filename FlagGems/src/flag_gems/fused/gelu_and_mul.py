import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

erf = tl_extra_shim.erf
tanh = tl_extra_shim.tanh
logger = logging.getLogger(__name__)

from dataclasses import replace
from flag_gems.utils.codegen_config_utils import get_codegen_config

_GELU_BLOCK_SIZE = 512
_GELU_CHUNK_SIZE = 64
_GELU_MAX_GRID_SIZE = 65536

_GELU_AND_MUL_CONFIG = replace(
    get_codegen_config(),
    max_tile_size=_GELU_BLOCK_SIZE,
    contiguous_tiles_per_cta=True,
)

@triton.jit
def gelu_none_and_mul_contiguous_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    tiles_per_program,
    chunks_per_tile,
    BLOCK_SIZE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    FULL_TILES: tl.constexpr,
):
    pid = tl.program_id(0)

    rcp_sqrt_2: tl.constexpr = 0.7071067811

    for tile_offset in tl.range(0, tiles_per_program, loop_unroll_factor = 1):
        tile_id = pid * tiles_per_program + tile_offset
        tile_base = tile_id.to(tl.int64) * BLOCK_SIZE

        for chunk_id in tl.range(0, chunks_per_tile, loop_unroll_factor=1):
            offsets = tile_base + chunk_id.to(tl.int64) * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)

            if FULL_TILES:
                x = tl.load(x_ptr + offsets)
                y = tl.load(y_ptr + offsets)
            else:
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

            x_fp32 = x.to(tl.float32)
            erf_x = erf(x_fp32 * rcp_sqrt_2)
            gelu_x = 0.5 * x_fp32 * (1.0 + erf_x)
            output = gelu_x * y

            if FULL_TILES:
                tl.store(output_ptr + offsets, output)
            else:
                tl.store(output_ptr + offsets, output, mask=mask)

@pointwise_dynamic(
    promotion_methods=[(0, 1, "DEFAULT")],
    config = _GELU_AND_MUL_CONFIG,
)
@triton.jit
def gelu_none_and_mul_kernel(x, y):
    x_fp32 = x.to(tl.float32)
    RCP_SQRT_2: tl.constexpr = 0.7071067811
    x_gelu = 0.5 * x_fp32 * (1 + erf(x_fp32 * RCP_SQRT_2))
    return x_gelu * y

@pointwise_dynamic(
    promotion_methods=[(0, 1, 2, "DEFAULT"), (0, 1, 2, "DEFAULT")], num_outputs=2,
    config = _GELU_AND_MUL_CONFIG,
)
@triton.jit
def gelu_none_and_mul_grad_kernel(x, y, dgrad):
    RCP_SQRT_2: tl.constexpr = 0.7071067811
    COEFF: tl.constexpr = 0.7978845608028654

    x_fp32 = x.to(tl.float32)
    x_erf = 1.0 + erf(x_fp32 * RCP_SQRT_2)
    x_gelu = 0.5 * x_fp32 * x_erf

    d_gelu = dgrad * y
    dx = (
        d_gelu
        * 0.5
        * (
            x_erf
            + x_fp32 * COEFF * tl.exp(-0.5 * x_fp32 * x_fp32)
        )
    )

    dy = dgrad * x_gelu

    return dx, dy


@pointwise_dynamic(
    promotion_methods=[(0, 1, "DEFAULT")],
    config = _GELU_AND_MUL_CONFIG,
)
@triton.jit
def gelu_tanh_and_mul_kernel(x, y):
    x_fp32 = x.to(tl.float32)
    x_gelu = (
        0.5
        * x_fp32
        * (
            1
            + tanh(x_fp32 * 0.79788456 * (1 + 0.044715 * x_fp32 * x_fp32))
        )
    )
    return x_gelu * y


@pointwise_dynamic(
    promotion_methods=[(0, 1, 2, "DEFAULT"), (0, 1, 2, "DEFAULT")], num_outputs=2,
    config = _GELU_AND_MUL_CONFIG,
)
@triton.jit
def gelu_tanh_and_mul_grad_kernel(x, y, dgrad):
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)

    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    a_cubed = x_fp32 * x_fp32 * x_fp32
    tanh_arg = sqrt_2_over_pi * (x_fp32 + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * x_fp32 * (1 + tanh_result)
    dy = geglu_a * dgrad

    term1 = 0.5 * (1 + tanh_result)
    tanh_sq = tanh_result * tanh_result
    term2 = (
        0.5
        * x_fp32
        * (1 - tanh_sq)
        * (sqrt_2_over_pi * (1 + 3 * 0.044715 * x_fp32 * x_fp32))
    )
    dx = dgrad * y_fp32 * (term1 + term2)

    return dx, dy

def _can_use_contiguous_kernel(x, y):
    return (
        isinstance(x, torch.Tensor)
        and isinstance(y, torch.Tensor)
        and x.device == y.device
        and x.dtype == y.dtype
        and x.shape == y.shape
        and x.is_contiguous()
        and y.is_contiguous()
    )

def _contiguous_forward(x, y):
    output = torch.empty_like(x)
    n_elements = x.numel()

    if n_elements == 0:
        return output

    block_size = _GELU_BLOCK_SIZE
    chunk_size = _GELU_CHUNK_SIZE

    assert block_size % chunk_size == 0

    num_tiles = triton.cdiv(n_elements, block_size)
    grid_size = min(num_tiles, _GELU_MAX_GRID_SIZE)
    tiles_per_program = triton.cdiv(num_tiles, grid_size)
    chunks_per_tile = block_size // chunk_size

    full_tiles = (n_elements % block_size == 0 and num_tiles % grid_size == 0)

    grid = (grid_size,)

    gelu_none_and_mul_contiguous_kernel[grid](
        x,
        y,
        output,
        n_elements,
        tiles_per_program,
        chunks_per_tile,
        BLOCK_SIZE = block_size,
        CHUNK_SIZE = chunk_size,
        FULL_TILES = full_tiles,
    )

    return output

class GeluAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, approximate="none"):
        logger.debug("GEMS GELU AND MUL FORWARD")
        ctx.save_for_backward(x, y)
        ctx.approximate = approximate

        if approximate not in ("none", "tanh"):
            raise ValueError(f"Invalid approximate value: {approximate}")

        if approximate == "tanh":
            return gelu_tanh_and_mul_kernel(x, y)
        elif _can_use_contiguous_kernel(x, y):
            return _contiguous_forward(x, y)
        elif approximate == "none":
            return gelu_none_and_mul_kernel(x, y)

    @staticmethod
    def backward(ctx, dgrad):
        logger.debug("GEMS GELU AND MUL BACKWARD")
        x, y = ctx.saved_tensors
        if ctx.approximate == "none":
            dx, dy = gelu_none_and_mul_grad_kernel(x, y, dgrad)
        else:
            dx, dy = gelu_tanh_and_mul_grad_kernel(x, y, dgrad)
        return dx, dy, None


def gelu_and_mul(x, y, approximate="none"):
    return GeluAndMul.apply(x, y, approximate)