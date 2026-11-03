import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    ],
    key=["num_tokens", "topk", "hidden_size"],
)
@triton.jit
def moe_sum_cpu_kernel(
    input_ptr,
    output_ptr,
    num_tokens,
    topk: tl.constexpr,
    hidden_size: tl.constexpr,
    input_stride_token: tl.constexpr,
    input_stride_topk: tl.constexpr,
    output_stride_token: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    hidden_offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if token_idx >= num_tokens:
        return
    if hidden_size % BLOCK_SIZE == 0:
        hidden_mask = tl.full((BLOCK_SIZE,), True, tl.int1)
    else:
        hidden_mask = hidden_offsets < hidden_size
    input_base = input_ptr + token_idx * input_stride_token + hidden_offsets
    acc = tl.zeros((BLOCK_SIZE,), tl.float32)

    # Expose independent loads while preserving the expert accumulation order.
    for group_idx in range(topk // 4):
        expert_base = input_base + group_idx * 4 * input_stride_topk
        data0 = tl.load(expert_base, mask=hidden_mask, other=0.0)
        data1 = tl.load(
            expert_base + input_stride_topk, mask=hidden_mask, other=0.0
        )
        data2 = tl.load(
            expert_base + 2 * input_stride_topk, mask=hidden_mask, other=0.0
        )
        data3 = tl.load(
            expert_base + 3 * input_stride_topk, mask=hidden_mask, other=0.0
        )
        acc += data0
        acc += data1
        acc += data2
        acc += data3
    for expert_idx in range(topk // 4 * 4, topk):
        acc += tl.load(
            input_base + expert_idx * input_stride_topk,
            mask=hidden_mask,
            other=0.0,
        )
    tl.store(
        output_ptr + token_idx * output_stride_token + hidden_offsets,
        acc,
        mask=hidden_mask,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["hidden_size", "topk"],
)
@triton.jit
def moe_sum_kernel(
    input_ptr,
    output_ptr,
    num_tokens,
    topk,
    hidden_size,
    input_stride_token,
    input_stride_topk,
    input_stride_hidden,
    output_stride_token,
    output_stride_hidden,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    hidden_start = block_idx * BLOCK_SIZE
    hidden_offsets = hidden_start + tl.arange(0, BLOCK_SIZE)
    hidden_mask = hidden_offsets < hidden_size
    if token_idx >= num_tokens:
        return
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    input_base = input_ptr + token_idx * input_stride_token

    for expert_idx in range(topk):
        expert_ptr = input_base + expert_idx * input_stride_topk
        expert_data = tl.load(expert_ptr + hidden_offsets, mask=hidden_mask, other=0.0)
        acc += expert_data
    output_ptr_pos = output_ptr + token_idx * output_stride_token + hidden_offsets

    tl.store(
        output_ptr_pos,
        acc.to(tl.float16) if input_ptr.dtype.element_ty == tl.float16 else acc,
        mask=hidden_mask,
    )


def moe_sum(
    input: torch.Tensor,
    output: torch.Tensor,
):
    num_tokens, topk, hidden_size = input.shape
    input_strides = input.stride()
    output_strides = output.stride()
    grid = lambda meta: (num_tokens, triton.cdiv(hidden_size, meta["BLOCK_SIZE"]))
    if (
        input.device.type == "cpu"
        and input.dtype == torch.float32
        and topk >= 32
        and hidden_size >= 64
        and input_strides[2] == 1
        and output_strides[1] == 1
    ):
        moe_sum_cpu_kernel[grid](
            input,
            output,
            num_tokens,
            topk,
            hidden_size,
            input_strides[0],
            input_strides[1],
            output_strides[0],
        )
        return
    moe_sum_kernel[grid](
        input,
        output,
        num_tokens,
        topk,
        hidden_size,
        input_strides[0],
        input_strides[1],
        input_strides[2],
        output_strides[0],
        output_strides[1],
    )
