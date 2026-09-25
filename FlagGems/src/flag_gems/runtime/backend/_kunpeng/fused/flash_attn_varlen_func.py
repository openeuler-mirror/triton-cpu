"""Kunpeng backend override for flash_attn_varlen_func.

Loaded automatically by replace_customized_ops() when GEMS_VENDOR=kunpeng.

A mixed serving batch (some sequences decoding with q_len==1, others prefilling
with q_len>1) never satisfies the global max_seqlen_q==1 precondition of the
seqlenq_ngroups group-swap in mha_varlan_fwd, so the swap never fires and every
query head re-gathers its KV head.  This override splits such a batch into a
pure-decode sub-batch (all q_len==1) and a prefill sub-batch, runs each through
the default flash_attn_varlen_func, and scatters the results back.  The decode
sub-call now triggers the group-swap, sharing each paged KV head across its query
group (~halves the decode gathers).  The BLOCK_N/BLOCK_M tiling for the two
sub-calls comes from the Kunpeng mha_block_* heuristics config; the default
operator implementation is not modified.
"""

import torch

import flag_gems.ops.attention as _attention
from flag_gems.runtime import torch_device_fn

# Captured before we replace the public/ops-level references below.
_default_flash_attn_varlen_func = _attention.flash_attn_varlen_func


def flash_attn_varlen_func(
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
    passthrough = dict(
        q_v=q_v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        scheduler_metadata=scheduler_metadata,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        s_aux=s_aux,
        num_splits=num_splits,
        cp_world_size=cp_world_size,
        cp_rank=cp_rank,
        cp_tot_seqused_k=cp_tot_seqused_k,
        fa_version=fa_version,
    )

    no_window = window_size is None or (window_size[0] < 0 and window_size[1] < 0)
    if (
        str(torch_device_fn.__name__).endswith("cpu")
        and block_table is not None
        and seqused_k is not None
        and cu_seqlens_q is not None
        and not return_softmax_lse
        and q.ndim == 3
        and q.shape[1] > k.shape[2]  # GQA (more query heads than KV heads)
        and alibi_slopes is None
        and no_window
        and dropout_p == 0
        and softcap == 0
        and int(max_seqlen_q) > 1
    ):
        num_heads_k = k.shape[2]
        q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        decode_seq = (q_lens == 1).nonzero(as_tuple=True)[0]
        prefill_seq = (q_lens > 1).nonzero(as_tuple=True)[0]
        n_decode = decode_seq.numel()
        batch = q_lens.numel()
        # Only worth the two extra launches when the decode sub-batch alone fills
        # the cores; otherwise the split overhead dominates (small mixed batches).
        if 0 < n_decode < batch and n_decode * num_heads_k >= torch.get_num_threads():
            total_q = q.shape[0]
            if out is None:
                out = torch.empty_like(q)
            decode_tok = cu_seqlens_q[decode_seq].to(torch.long)
            is_decode_tok = torch.zeros(total_q, dtype=torch.bool, device=q.device)
            is_decode_tok[decode_tok] = True
            prefill_tok = (~is_decode_tok).nonzero(as_tuple=True)[0]

            def run_sub(seq_idx, tok_idx, sub_max_q, sub_cu_q):
                sub_seqused_k = seqused_k[seq_idx].contiguous()
                sub_out = torch.empty(
                    (tok_idx.numel(), q.shape[1], q.shape[2]),
                    dtype=q.dtype,
                    device=q.device,
                )
                _default_flash_attn_varlen_func(
                    q[tok_idx].contiguous(),
                    k,
                    v,
                    sub_max_q,
                    sub_cu_q,
                    int(sub_seqused_k.max()),
                    seqused_k=sub_seqused_k,
                    block_table=block_table[seq_idx].contiguous(),
                    out=sub_out,
                    **passthrough,
                )
                out[tok_idx] = sub_out

            run_sub(
                decode_seq,
                decode_tok,
                1,
                torch.arange(n_decode + 1, dtype=torch.int32, device=q.device),
            )
            prefill_cu = torch.zeros(
                prefill_seq.numel() + 1, dtype=torch.int32, device=q.device
            )
            prefill_cu[1:] = torch.cumsum(q_lens[prefill_seq], 0)
            run_sub(
                prefill_seq,
                prefill_tok,
                int(q_lens[prefill_seq].max()),
                prefill_cu,
            )
            return out

    return _default_flash_attn_varlen_func(
        q,
        k,
        v,
        max_seqlen_q,
        cu_seqlens_q,
        max_seqlen_k,
        cu_seqlens_k=cu_seqlens_k,
        seqused_k=seqused_k,
        block_table=block_table,
        out=out,
        return_softmax_lse=return_softmax_lse,
        **passthrough,
    )


# replace_customized_ops() rebinds the public flag_gems.flash_attn_varlen_func
# (the API vLLM calls).  Benchmarks/tests reach the op through the internal
# flag_gems.ops.* reference, so rebind that here too, without touching the
# default operator module.
import flag_gems.ops as _ops  # noqa: E402

_ops.flash_attn_varlen_func = flash_attn_varlen_func
