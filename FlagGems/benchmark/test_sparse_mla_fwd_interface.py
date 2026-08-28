import random

import pytest
import torch

import flag_gems
from flag_gems.fused.DSA.sparse_mla import triton_sparse_mla_fwd_interface

from benchmark.performance_utils import GenericBenchmark


def _init_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def make_sparse_mla_input(
    batch_size,
    seq_len_q,
    seq_len_kv,
    num_heads,
    num_kv_heads,
    qk_dim,
    topk,
    dtype,
    device,
):
    _init_seed(42)
    B = batch_size
    S = seq_len_q
    H = num_heads
    DQK = qk_dim
    SKV = seq_len_kv
    HKV = num_kv_heads

    q = torch.randn((B, S, H, DQK), dtype=dtype, device=device)
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device=device)

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device=device)
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, : len(i_i)] = i_i

    return q, kv, indices


SPARSE_MLA_PARAMS = [
    {"seq_len_q": 64, "seq_len_kv": 1024, "topk": 64, "num_heads": 128},
    {"seq_len_q": 128, "seq_len_kv": 2048, "topk": 128, "num_heads": 128},
    {"seq_len_q": 256, "seq_len_kv": 4096, "topk": 256, "num_heads": 128},
    {"seq_len_q": 512, "seq_len_kv": 8192, "topk": 512, "num_heads": 128},
]


def sparse_mla_input_fn(param, dtype, device):
    q, kv, indices = make_sparse_mla_input(
        batch_size=1,
        seq_len_q=param["seq_len_q"],
        seq_len_kv=param["seq_len_kv"],
        num_heads=param["num_heads"],
        num_kv_heads=1,
        qk_dim=576,
        topk=param["topk"],
        dtype=dtype,
        device=device,
    )
    yield (q, kv, indices, {"d_v": 512})


class SparseMlaFwdBenchmark(GenericBenchmark):
    DEFAULT_METRICS = GenericBenchmark.DEFAULT_METRICS[:] + ["tflops"]

    def set_shapes(self, shape_file_path=None):
        self.shapes = SPARSE_MLA_PARAMS

    def set_more_shapes(self):
        return []

    def get_tflops(self, op, *args, **kwargs):
        query, _, indices = args[:3]
        value_dim = kwargs.get("d_v", args[4] if len(args) > 4 else None)
        batch = 1 if query.ndim == 3 else query.shape[0]
        query_length, query_heads, query_dim = query.shape[-3:]
        key_value_heads, topk = indices.shape[-2:]
        if key_value_heads == 0 or query_heads % key_value_heads:
            raise ValueError(
                "query heads must be divisible by key/value heads"
            )
        return (
            2
            * batch
            * query_length
            * query_heads
            * topk
            * (query_dim + value_dim)
        )


@pytest.mark.sparse_mla_fwd_interface
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_sparse_mla_fwd_interface():
    bench = SparseMlaFwdBenchmark(
        op_name="sparse_mla_fwd_interface",
        torch_op=triton_sparse_mla_fwd_interface,
        input_fn=sparse_mla_input_fn,
        dtypes=[torch.bfloat16],
    )
    bench.run()