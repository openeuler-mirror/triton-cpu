import pytest
import torch

import flag_gems

from . import performance_utils as utils

try:
    import os

    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
    import vllm._custom_ops as vllm_ops

    HAS_VLLM = True
    WARP_SIZE = 32
except ImportError:
    HAS_VLLM = False
    WARP_SIZE = 0


# Modified from: https://github.com/vllm-project/vllm/blob/main/tests/kernels/moe/test_moe_align_block_size.py
def torch_moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Golden torch implementation of moe_align_block_size.

    This function aligns the token distribution across experts to be compatible
    with block size for matrix multiplication by sorting tokens by expert and
    padding to block boundaries.
    """
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)

    # if topk_ids.numel() < num_experts:
    #     max_num_tokens_padded = topk_ids.numel() * block_size

    flattened_token_indices = torch.arange(
        topk_ids.numel(), device=topk_ids.device, dtype=torch.int32
    )
    flattened_expert_ids = topk_ids.flatten()
    sorted_expert_ids, sort_indices = torch.sort(flattened_expert_ids, stable=True)
    sorted_token_indices = flattened_token_indices[sort_indices]

    expert_token_counts = torch.zeros(
        num_experts, dtype=torch.int64, device=topk_ids.device
    )
    for expert_id in range(num_experts):
        mask = sorted_expert_ids == expert_id
        expert_token_counts[expert_id] = mask.sum()

    expert_padded_counts = torch.zeros(
        num_experts, dtype=torch.int64, device=topk_ids.device
    )
    for expert_id in range(num_experts):
        original_count = expert_token_counts[expert_id]
        if expert_map is not None and expert_map[expert_id] == -1:
            continue
        if original_count > 0:
            expert_padded_counts[expert_id] = (
                (original_count + block_size - 1) // block_size
            ) * block_size

    in_sorted_token_ids = torch.full(
        (max_num_tokens_padded,),
        topk_ids.numel(),
        dtype=torch.int32,
        device=topk_ids.device,
    )

    # max_num_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    max_num_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.zeros(max_num_blocks, dtype=torch.int32, device=topk_ids.device)

    current_pos = 0
    current_block = 0
    for expert_id in range(num_experts):
        if expert_map is not None and expert_map[expert_id] == -1:
            continue

        expert_mask = sorted_expert_ids == expert_id
        expert_tokens = sorted_token_indices[expert_mask]
        num_expert_tokens = expert_tokens.shape[0]

        if num_expert_tokens > 0:
            in_sorted_token_ids[
                current_pos : current_pos + num_expert_tokens
            ] = expert_tokens

            expert_blocks_needed = expert_padded_counts[expert_id] // block_size

            expert_id_new = expert_id
            if expert_map is not None:
                expert_id_new = expert_map[expert_id]
            expert_ids[
                current_block : current_block + expert_blocks_needed
            ] = expert_id_new

            current_pos += expert_padded_counts[expert_id]
            current_block += expert_blocks_needed

    total_padded_tokens = expert_padded_counts.sum()
    in_num_tokens_post_pad = torch.tensor(
        [total_padded_tokens], dtype=torch.int32, device=topk_ids.device
    )
    sorted_token_ids.copy_(in_sorted_token_ids)
    experts_ids.copy_(expert_ids)
    num_tokens_post_pad.copy_(in_num_tokens_post_pad)

    return in_sorted_token_ids, expert_ids, num_tokens_post_pad

def _input_fn(shape, dtype, device):
    num_experts = shape[0]
    block_size = shape[1]
    dtype = torch.int32
    topk_ids = torch.randint(
        0, num_experts, (shape[2], shape[3]), dtype=dtype, device=device
    )

    if HAS_VLLM:
        max_num_tokens_padded = ((num_experts + WARP_SIZE - 1) // WARP_SIZE) * WARP_SIZE

        # padded_num_experts in vllm._custom_ops.moe_align_block_size
        # must be less than 1024
        if max_num_tokens_padded >= 1024:
            return
    else:
        max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)

    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=dtype, device=device)
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty((max_num_m_blocks,), dtype=dtype, device=device)
    num_tokens_post_pad = torch.empty(1, dtype=dtype, device=device)

    yield (
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )


class MoeAlignBlockSizeBenchmark(utils.GenericBenchmark4DOnly):
    def set_shapes(self, shape_file_path: None):
        moe_align_block_size_shape = [
            (512, 64, 16384, 10),
            (512, 64, 6152, 10),
            (512, 64, 4727, 10),
            (512, 64, 1905, 10),
            (512, 64, 11575, 10),
            (512, 64, 1032, 10),
            (512, 64, 4201, 10),
            (512, 64, 2056, 10),
            (512, 64, 7561, 10),
            (512, 64, 4104, 10),
            (512, 64, 14281, 10),
        ]
        self.shapes = moe_align_block_size_shape

    def set_more_shapes(self):
        return None


@pytest.mark.moe_align_block_size_triton
def test_moe_align_block_size_triton():
    gems_op = flag_gems.moe_align_block_size_triton
    bench = MoeAlignBlockSizeBenchmark(
        op_name="moe_align_block_size_triton",
        input_fn=_input_fn,
        torch_op=vllm_ops.moe_align_block_size if HAS_VLLM else torch_moe_align_block_size,
        dtypes=[
            torch.int32,
        ],
    )

    bench.set_gems(gems_op)
    bench.run()
