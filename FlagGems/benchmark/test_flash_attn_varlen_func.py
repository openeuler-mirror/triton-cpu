from typing import Any, List, Optional

import pytest
import torch

import flag_gems

from . import performance_utils as base
from . import utils

vendor_name = flag_gems.vendor_name


class FlashAttnVarlenBenchmark(base.Benchmark):
    """
    benchmark for flash_attn_varlen_func
    """

    DEFAULT_METRICS = base.Benchmark.DEFAULT_METRICS[:] + ["tflops"]

    @staticmethod
    def _attention_pair_count(query_length, key_length, causal, left, right):
        alignment = (
            key_length - query_length
            if causal or (left >= 0 and right >= 0)
            else 0
        )
        pairs = 0
        for query_index in range(query_length):
            center = query_index + alignment
            start = 0 if left < 0 else max(0, center - left)
            end = min(key_length, center + 1) if causal else key_length
            if right >= 0:
                end = min(end, center + right + 1)
            pairs += max(0, end - start)
        return pairs

    def get_tflops(self, op, *args, **kwargs):
        query = args[0]
        cumulative_query_lengths = args[4].tolist()
        used_key_lengths = args[7].tolist()
        causal = args[11]
        window_size = args[12]
        left, right = (-1, -1) if window_size is None else window_size
        pairs = sum(
            self._attention_pair_count(
                cumulative_query_lengths[index + 1]
                - cumulative_query_lengths[index],
                used_key_lengths[index],
                causal,
                left,
                right,
            )
            for index in range(len(used_key_lengths))
        )
        return 4 * query.shape[1] * query.shape[2] * pairs

    def set_shapes(self, shape_file_path: Optional[List[Any]] = None):
        # Collecting from qwen/Qwen3-1.7B
        # --random-input 512 --random-output 2048 --num-prompts 200 --request-rate inf
        # Format: (cu_seq_lens_q, seqused_k, num_heads, head_size, block_size,
        # num_blocks, alibi, soft_cap)

        all_cu_seq_lens_q = [
            (
                0,
                512,
            ),
            (
                0,
                1,
                2,
                72,
            ),
            tuple(range(0, 45))
            + (
                105,
                121,
                137,
                153,
                169,
                185,
                201,
                217,
                233,
                249,
                265,
            ),
            tuple(range(0, 196))
            + (
                211,
                226,
                240,
                253,
                265,
            ),
        ]
        all_seqused_k = [
            (512,),
            (
                1,
                1,
                70,
            ),
            (515,) + (514,) * 20 + (513,) * 20 + (512,) * 14,
            (2333,)
            + (2331,) * 20
            + (2330,) * 20
            + (2329,) * 14
            + (2328,) * 18
            + (2327,) * 15
            + (2326,) * 17
            + (2325,) * 18
            + (2324,) * 21
            + (2323,) * 22
            + (2322,) * 24
            + (2321,) * 5
            + (
                2320,
                2319,
                2318,
                2317,
                2316,
            ),
        ]

        num_heads = 16
        num_heads_k = 8
        head_dim = 128
        block_size = 16
        num_blocks = 2000
        alibi = False
        soft_cap = None

        all_configs = [
            (
                cu_seq_lens_q,
                seqused_k,
                num_heads,
                num_heads_k,
                head_dim,
                block_size,
                num_blocks,
                alibi,
                soft_cap,
            )
            for cu_seq_lens_q, seqused_k in zip(all_cu_seq_lens_q, all_seqused_k)
        ]

        self.shapes = all_configs

    def get_input_iter(self, dtype):
        for config in self.shapes:
            yield self.flash_attn_varlen_input_fn(config, dtype, self.device)

    def flash_attn_varlen_input_fn(self, config, dtype, device):
        """Input function for flash attention varlen benchmark"""
        (
            cu_query_lens,
            seqused_k,
            num_query_heads,
            num_kv_heads,
            head_size,
            block_size,
            num_blocks,
            alibi,
            soft_cap,
        ) = config

        if alibi is True and soft_cap is not None:
            return

        num_seqs = len(cu_query_lens) - 1
        max_query_len = max(
            map(lambda x, y: x - y, cu_query_lens[1:], cu_query_lens[:-1])
        )
        max_kv_len = max(seqused_k)
        window_size = (-1, -1)
        scale = head_size**-0.5

        assert num_seqs == len(seqused_k)

        with torch.device(device):
            query = torch.randn(
                cu_query_lens[-1],
                num_query_heads,
                head_size,
                dtype=dtype,
                device=device,
            )
            out = torch.empty_like(query)
            key_cache = torch.randn(
                num_blocks,
                block_size,
                num_kv_heads,
                head_size,
                dtype=dtype,
                device=device,
            )
            value_cache = torch.randn_like(key_cache)
            cu_query_lens = torch.tensor(
                cu_query_lens, dtype=torch.int32, device=device
            )
            seqused_k = torch.tensor(seqused_k, dtype=torch.int32, device=device)

            max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
            block_tables = torch.randint(
                0,
                num_blocks,
                (num_seqs, max_num_blocks_per_seq),
                dtype=torch.int32,
                device=device,
            )

            causal = True

            if alibi:
                alibi_slopes = (
                    torch.ones(
                        num_seqs, num_query_heads, device=device, dtype=torch.float32
                    )
                    * 0.3
                )
            else:
                alibi_slopes = None

        return (
            query,
            key_cache,
            value_cache,
            max_query_len,
            cu_query_lens,
            max_kv_len,
            None,
            seqused_k,
            None,
            0.0,
            scale,
            causal,
            window_size,
            soft_cap if soft_cap is not None else 0,
            alibi_slopes,
            False,
            False,
            block_tables,
            False,
            out,
            None,
            None,
            None,
            None,
            {
                "s_aux": None,
                "num_splits": 0,
                "cp_world_size": 1,
                "cp_rank": 0,
                "cp_tot_seqused_k": None,
                "fa_version": 2,
            },
        )


def flash_attn_varlen_legacy(*args, **kwargs):
    """
    Compatibility wrapper for running old flash_attn_varlen_func.
    """
    (
        query,
        key_cache,
        value_cache,
        max_query_len,
        cu_query_lens,
        max_kv_len,
        _,
        seqused_k,
        _,
        dropout_p,
        scale,
        causal,
        window_size,
        soft_cap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        block_tables,
        _,
        out,
        *_,
    ) = args

    k_flat = key_cache.reshape(-1, key_cache.shape[2], key_cache.shape[3])
    v_flat = value_cache.reshape(-1, value_cache.shape[2], value_cache.shape[3])
    cu_seqlens_k = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=seqused_k.device),
            torch.cumsum(seqused_k, dim=0),
        ]
    ).to(torch.int32)

    from flash_attn import flash_attn_varlen_func

    result = flash_attn_varlen_func(
        query,  # q
        k_flat,  # k (flattened from key_cache)
        v_flat,  # v (flattened from value_cache)
        cu_query_lens,  # cu_seqlens_q
        cu_seqlens_k,  # cu_seqlens_k (constructed from seqused_k)
        max_query_len,  # max_seqlen_q
        max_kv_len,  # max_seqlen_k
        dropout_p,  # dropout_p
        scale,  # softmax_scale
        causal,  # causal
        tuple(window_size),  # window_size
        float(soft_cap),  # softcap
        alibi_slopes,  # alibi_slopes
        deterministic,  # deterministic
        return_attn_probs,  # return_attn_probs
        block_tables,  # block_table
        alibi_slopes is not None,  # use_alibi (derived from alibi_slopes)
        0,  # alibi_mode
        1,  # imp_mode
        out=out,  # out
        bias=None,  # bias
    )
    return result


# Following attn_bias_from_alibi_slopes and ref_paged_attn are copied from
# triton-cpu/FlagGems/tests/test_attention_ops.py


def attn_bias_from_alibi_slopes(slopes, seqlen_q, seqlen_k, causal=False):
    batch, nheads = slopes.shape
    device = slopes.device
    slopes = slopes.unsqueeze(-1).unsqueeze(-1)
    if causal:
        return (
            torch.arange(-seqlen_k + 1, 1, device=device, dtype=torch.float32) * slopes
        )

    row_idx = torch.arange(seqlen_q, device=device, dtype=torch.long).unsqueeze(-1)
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    relative_pos = torch.abs(row_idx + seqlen_k - seqlen_q - col_idx)
    return -slopes * relative_pos.to(dtype=slopes.dtype)


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: List[int],
    kv_lens: List[int],
    block_tables: torch.Tensor,
    scale: float,
    attn_bias: torch.Tensor = None,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: List[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        # clone to avoid clobbering the query tensor
        q = query[start_idx : start_idx + query_len].clone()
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        attn = torch.einsum("qhd,khd->hqk", q, k)
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))

        if attn_bias is not None:
            attn = attn + attn_bias[i, :, :query_len, :kv_len]

        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


def torch_flash_attn_varlen_func(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    s_aux=None,
    num_splits: int = 0,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k=None,
    fa_version: int = 2,
):
    """Adapt flash_attn_varlen_func arguments for the PyTorch paged-attention ref."""
    query_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    kv_lens = seqused_k.tolist()

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    window_size = (-1, -1) if window_size is None else window_size
    sliding_window = None if window_size == (-1, -1) else window_size[0]

    if alibi_slopes is None:
        attn_bias = None
    else:
        if alibi_slopes.ndim == 1:
            alibi_slopes = alibi_slopes.unsqueeze(0).expand(len(query_lens), -1)
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes, max_seqlen_q, max_seqlen_k, causal=causal
        )

    result = ref_paged_attn(
        query=q,
        key_cache=k,
        value_cache=v,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_table,
        scale=softmax_scale,
        attn_bias=attn_bias,
        sliding_window=sliding_window,
        soft_cap=softcap if softcap > 0 else None,
    )
    if out is not None:
        out.copy_(result)
        result = out
    return result


@pytest.mark.skipif(
    flag_gems.device != "cpu" and utils.SkipVersion("vllm", "<0.9"),
    reason="vLLM version prior to 0.9 does not include the flash_attn_varlen_func API.",
)
@pytest.mark.skipif(
    flag_gems.device != "cpu" and utils.SkipVersion("torch", "<2.7"),
    reason="Torch version prior to 2.7 is not compatible with VLLM.",
)
@pytest.mark.skipif(vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(vendor_name == "mthreads", reason="Torch < 2.7")
@pytest.mark.skipif(vendor_name == "cambricon", reason="TypeError")
@pytest.mark.flash_attn_varlen_func
def test_flash_attn_varlen_func(monkeypatch):
    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")

    if flag_gems.device == "cpu":
        flash_attn_varlen_func = torch_flash_attn_varlen_func
    elif vendor_name == "iluvatar":
        # iluvatar does not have updated vllm_flash_attn, use conversion wrapper
        flash_attn_varlen_func = flash_attn_varlen_legacy
    else:
        from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

    bench = FlashAttnVarlenBenchmark(
        op_name="flash_attn_varlen_func",
        torch_op=flash_attn_varlen_func,
        gems_op=flag_gems.ops.flash_attn_varlen_func,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.run()
