import math

import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import GenericBenchmarkExcluse1D


class FusedAddRMSNormBenchmark(GenericBenchmarkExcluse1D):
    DEFAULT_METRICS = GenericBenchmarkExcluse1D.DEFAULT_METRICS[:] + ["gbps"]

    def get_gbps(self, args, latency):
        inp, residual, normalized_shape, weight = args[:4]
        passes = 3 if math.prod(normalized_shape) >= 4096 else 2
        logical_bytes = (
            passes * inp.numel() * inp.element_size()
            + passes * residual.numel() * residual.element_size()
            + weight.numel() * weight.element_size()
        )
        return logical_bytes / latency / 1e6


def _input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    residual = torch.randn(shape, dtype=dtype, device=device)
    layer_shape = (shape[-1],)
    weight = torch.randn(layer_shape, dtype=dtype, device=device)
    yield inp, residual, layer_shape, weight, 1e-5


def torch_op(x, residual, layer_shape, weight, eps):
    x = x + residual
    variance = x.pow(2).mean(-1, keepdim=True)
    hidden_states = x * torch.rsqrt(variance + eps)
    return weight * hidden_states


@pytest.mark.fused_add_rms_norm
def test_fused_add_rms_norm():
    bench = FusedAddRMSNormBenchmark(
        input_fn=_input_fn,
        op_name="fused_add_rms_norm",
        torch_op=torch_op,
        gems_op=flag_gems.fused_add_rms_norm,
        dtypes=FLOAT_DTYPES,
    )

    bench.run()
