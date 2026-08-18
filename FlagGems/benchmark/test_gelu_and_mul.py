import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import GenericBenchmark, binary_input_fn


class GeluAndMulBenchmark(GenericBenchmark):
    DEFAULT_METRICS = GenericBenchmark.DEFAULT_METRICS[:] + ["gbps"]

    def get_gbps(self, args, latency):
        inp, other = args[:2]
        logical_bytes = (
            2 * inp.numel() * inp.element_size()
            + other.numel() * other.element_size()
        )
        return logical_bytes / latency / 1e6


@pytest.mark.gelu_and_mul
def test_gelu_and_mul():
    def torch_op(x, y):
        return torch.mul(torch.nn.functional.gelu(x), y)

    bench = GeluAndMulBenchmark(
        input_fn=binary_input_fn,
        op_name="gelu_and_mul",
        torch_op=torch_op,
        gems_op=flag_gems.gelu_and_mul,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
