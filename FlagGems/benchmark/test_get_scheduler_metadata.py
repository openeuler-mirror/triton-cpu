import math
from typing import Optional

import pytest
import torch

import flag_gems

from .performance_utils import GenericBenchmark


# Following cpu_get_scheduler_metadata is copied from
# triton-cpu/FlagGems/tests/test_attention_ops.py


def cpu_get_scheduler_metadata(
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads: int,
    num_heads_k: int,
    headdim: int,
    headdim_v: int,
    qkv_dtype: torch.dtype,
    seqused_k: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    page_size: Optional[int] = None,
    max_seqlen_k_new: int = 0,
    is_causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    has_softcap: bool = False,
    num_splits: int = 0,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
) -> torch.Tensor:
    from flag_gems.ops.get_scheduler_metadata import (
        get_num_splits,
        get_optimal_block_mn,
        get_pack_gqa,
        get_pagedkv_tma,
        round_up_headdim,
        round_up_headdimv,
    )

    device = seqused_k.device
    if device.type != "cpu":
        raise ValueError("cpu_get_scheduler_metadata only supports CPU tensors")
    dtype = torch.int32

    supported_dtypes = (torch.half, torch.bfloat16)
    assert (
        qkv_dtype in supported_dtypes
    ), "FlashAttention only supports fp16 and bf16 data type"
    assert (
        num_heads % num_heads_k == 0
    ), "Number of heads in key/value must divide number of heads in query"

    effective_is_causal = is_causal
    effective_window_left = window_size_left if window_size_left >= 0 else -1
    effective_window_right = window_size_right

    if effective_window_left >= max_seqlen_k - 1:
        effective_window_left = -1
    if effective_window_right >= max_seqlen_q - 1:
        effective_window_right = -1

    if (
        max_seqlen_q == 1
        and effective_window_left == -1
        and effective_window_right == -1
    ):
        if (headdim <= 64 or headdim > 128) or page_size is None:
            effective_is_causal = False

    if effective_is_causal:
        effective_window_right = 0

    final_is_causal = effective_window_left < 0 and effective_window_right == 0
    final_is_local = (
        effective_window_left >= 0 or effective_window_right >= 0
    ) and not final_is_causal

    arch = 0
    num_sm = torch.get_num_threads() - sm_margin

    softcap = 1.0 if has_softcap else 0.0
    element_size = qkv_dtype.itemsize
    has_page_table = page_size is not None
    d_rounded = round_up_headdim(headdim)
    dv_rounded = round_up_headdimv(headdim_v)

    pagedkv_tma = get_pagedkv_tma(
        arch=arch,
        page_size=page_size if page_size is not None else 1,
        has_page_table=has_page_table,
        leftpad_k=leftpad_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k_new=max_seqlen_k_new,
        num_heads=num_heads,
        num_heads_k=num_heads_k,
        d_rounded=d_rounded,
        dv_rounded=dv_rounded,
        is_causal=final_is_causal,
        is_local=final_is_local,
        element_size=element_size,
        softcap=has_softcap,
    )

    varlen_q_flag = cu_seqlens_q is not None or seqused_q is not None
    pack_gqa = (
        pack_gqa
        if pack_gqa is not None
        else get_pack_gqa(
            arch=arch,
            has_page_table=has_page_table,
            pagedkv_tma=pagedkv_tma,
            num_splits=num_splits,
            num_heads=num_heads,
            num_heads_k=num_heads_k,
            varlen_q=varlen_q_flag,
            seqlen_q=max_seqlen_q,
            d_rounded=d_rounded,
            dv_rounded=dv_rounded,
            is_causal=final_is_causal,
            is_local=final_is_local,
            element_size=element_size,
            softcap=has_softcap,
        )
    )

    use_dynamic_split = batch_size <= 992

    if num_splits <= 0:
        eff_num_splits = get_num_splits(
            batch_size=batch_size,
            num_heads=num_heads,
            num_heads_k=num_heads_k,
            headdim=headdim,
            headdim_v=headdim_v,
            d_rounded=d_rounded,
            dv_rounded=dv_rounded,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_k_new=max_seqlen_k_new,
            arch=arch,
            num_sm=num_sm,
            is_causal=final_is_causal,
            is_local=final_is_local,
            has_softcap=softcap,
            is_varlen=True,
            has_page_table=has_page_table,
            pack_gqa=pack_gqa,
            window_size_left=effective_window_left,
            window_size_right=effective_window_right,
            element_size=element_size,
            use_dynamic_split=use_dynamic_split,
        )
    else:
        eff_num_splits = num_splits

    eff_num_splits = min(eff_num_splits, 256, num_sm)

    pack_gqa = True if eff_num_splits > 1 else pack_gqa

    qhead_per_khead = (
        1 if not pack_gqa else (num_heads + num_heads_k - 1) // num_heads_k
    )
    num_head_k = num_heads_k if pack_gqa else num_heads

    blockM, blockN = get_optimal_block_mn(
        device=device,
        headdim=headdim,
        headdim_v=headdim_v,
        is_causal=final_is_causal,
        is_local=final_is_local,
        has_softcap=has_softcap,
        element_size=element_size,
        paged_kv=has_page_table,
        pagedkv_tma=pagedkv_tma,
        varlen_and_split=eff_num_splits > 1,
        append_kv=max_seqlen_k_new > 0,
    )

    if seqused_q is not None:
        seqlen_q = seqused_q[:batch_size]
    elif cu_seqlens_q is not None:
        seqlen_q = cu_seqlens_q[1 : batch_size + 1] - cu_seqlens_q[:batch_size]
    else:
        seqlen_q = torch.full(
            (batch_size,), max_seqlen_q, dtype=dtype, device=device
        )

    num_m_blocks = (seqlen_q * qhead_per_khead + blockM - 1) // blockM

    seqlen_k = seqused_k[:batch_size]
    if max_seqlen_k_new > 0:
        if cu_seqlens_k_new is not None:
            seqlen_k = seqlen_k + (
                cu_seqlens_k_new[1 : batch_size + 1]
                - cu_seqlens_k_new[:batch_size]
            )
        else:
            seqlen_k = seqlen_k + max_seqlen_k_new
    if leftpad_k is not None:
        seqlen_k = seqlen_k - leftpad_k[:batch_size]

    num_n_blocks = (seqlen_k + blockN - 1) // blockN
    if use_dynamic_split:
        total_blocks = (num_m_blocks * num_n_blocks).sum().item()
        blocks_per_sm = max(
            1,
            math.ceil(total_blocks * 1.1 * num_head_k / num_sm),
        )
        num_splits_dynamic = (
            (num_n_blocks + blocks_per_sm - 1) // blocks_per_sm
        ).clamp_max_(eff_num_splits)
        num_splits_dynamic.clamp_min_(1)

    scheduler_needs_semaphore = eff_num_splits > 1
    alloc_size = int(scheduler_needs_semaphore) + int(use_dynamic_split) * batch_size
    scheduler_metadata = torch.empty(alloc_size, dtype=dtype, device=device)
    offset = 0
    if scheduler_needs_semaphore:
        scheduler_metadata[0] = 0
        offset = 1
    if use_dynamic_split:
        scheduler_metadata[offset:] = num_splits_dynamic
    return scheduler_metadata


class GetSchedulerMetadataBenchmark(GenericBenchmark):
    DEFAULT_METRICS = GenericBenchmark.DEFAULT_METRICS[:] + ["gbps"]

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (8, 8, 1024, 16, 4, 128, 128),
            (32, 32, 512, 8, 8, 64, 64),
            (256, 256, 2048, 32, 32, 128, 128),
            (512, 512, 4096, 32, 8, 128, 128),
            (1024, 1024, 8192, 64, 16, 128, 128),
        ]

    def set_more_shapes(self):
        return None

    def get_gbps(self, args, latency):
        cache_sequence_lengths = args[6]
        logical_bytes = (
            cache_sequence_lengths.numel()
            * cache_sequence_lengths.element_size()
        )
        return logical_bytes / latency / 1e6


@pytest.mark.get_scheduler_metadata
def test_get_scheduler_metadata(monkeypatch):
    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")

    def input_kwargs(shape, dtype, device):
        (
            batch_size,
            max_seqlen_q,
            max_seqlen_k,
            num_heads_q,
            num_heads_kv,
            headdim,
            headdim_v,
        ) = shape
        cache_seqlens = torch.randint(
            1, max_seqlen_k + 1, (batch_size,), dtype=torch.int32, device=device
        )

        yield (
            batch_size,
            max_seqlen_q,
            max_seqlen_k,
            num_heads_q,
            num_heads_kv,
            headdim,
            cache_seqlens,
            dtype,  # qkv_dtype
            headdim_v,  # headdim_v
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k_new
            None,  # cache_leftpad
            None,  # page_size
            0,  # max_seqlen_k_new
            False,  # causal
            (-1, -1),  # window_size
            False,  # has_softcap
            0,  # num_splits
            None,  # pack_gqa
            0,  # sm_margin
        )

    def make_wrapper(op):
        def wrapper(
            batch_size,
            max_seqlen_q,
            max_seqlen_k,
            num_heads_q,
            num_heads_kv,
            headdim,
            cache_seqlens,
            qkv_dtype=torch.bfloat16,
            headdim_v=None,
            cu_seqlens_q=None,
            cu_seqlens_k_new=None,
            cache_leftpad=None,
            page_size=None,
            max_seqlen_k_new=0,
            causal=False,
            window_size=(-1, -1),
            has_softcap=False,
            num_splits=0,
            pack_gqa=None,
            sm_margin=0,
        ):
            return op(
                batch_size=batch_size,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                num_heads=num_heads_q,
                num_heads_k=num_heads_kv,
                headdim=headdim,
                headdim_v=headdim_v or headdim,
                qkv_dtype=qkv_dtype,
                seqused_k=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=None,
                cu_seqlens_k_new=cu_seqlens_k_new,
                seqused_q=None,
                leftpad_k=cache_leftpad,
                page_size=page_size,
                max_seqlen_k_new=max_seqlen_k_new,
                is_causal=causal,
                window_size_left=window_size[0],
                window_size_right=window_size[1],
                has_softcap=has_softcap,
                num_splits=num_splits,
                pack_gqa=pack_gqa,
                sm_margin=sm_margin,
            )

        return wrapper

    flaggems_wrapper = make_wrapper(flag_gems.ops.get_scheduler_metadata)
    if flag_gems.device == "cpu":
        reference_op = make_wrapper(cpu_get_scheduler_metadata)
    else:
        try:
            from vllm.vllm_flash_attn.flash_attn_interface import (
                get_scheduler_metadata as reference_op,
            )
        except ImportError:
            pytest.skip("vLLM is not available, skipping performance test")

    bench = GetSchedulerMetadataBenchmark(
        op_name="get_scheduler_metadata",
        input_fn=input_kwargs,
        torch_op=reference_op,
        gems_op=flaggems_wrapper,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.run()
