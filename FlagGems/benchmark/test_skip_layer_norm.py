import math

import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import GenericBenchmarkExcluse1D


class SkipLayerNormBenchmark(GenericBenchmarkExcluse1D):
    DEFAULT_METRICS = GenericBenchmarkExcluse1D.DEFAULT_METRICS[:] + ["gbps"]

    def get_gbps(self, args, latency):
        inp, residual, normalized_shape, weight, bias = args[:5]
        large_normalization = math.prod(normalized_shape) >= 4096
        input_passes = 3 if large_normalization else 2
        residual_passes = 3 if large_normalization else 1
        logical_bytes = (
            input_passes * inp.numel() * inp.element_size()
            + residual_passes * residual.numel() * residual.element_size()
            + weight.numel() * weight.element_size()
            + bias.numel() * bias.element_size()
        )
        return logical_bytes / latency / 1e6


def _input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    residual = torch.randn(shape, dtype=dtype, device=device)
    layer_shape = (shape[-1],)
    weight = torch.randn(layer_shape, dtype=dtype, device=device)
    bias = torch.randn(layer_shape, dtype=dtype, device=device)

    yield inp, residual, layer_shape, weight, bias


def torch_op(inp, residual, layer_shape, weight, bias):
    return torch.layer_norm(inp + residual, layer_shape, weight, bias)


@pytest.mark.skip_layer_norm
def test_skip_layernorm():
    bench = SkipLayerNormBenchmark(
        input_fn=_input_fn,
        op_name="skip_layernorm",
        gems_op=flag_gems.skip_layer_norm,
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
