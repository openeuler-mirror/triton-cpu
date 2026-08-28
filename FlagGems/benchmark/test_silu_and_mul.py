import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import GenericBenchmark, binary_input_fn


class SiluAndMulBenchmark(GenericBenchmark):
    DEFAULT_METRICS = GenericBenchmark.DEFAULT_METRICS[:] + ["gbps"]

    def get_gbps(self, args, latency):
        inp, other = args[:2]
        logical_bytes = (
            2 * inp.numel() * inp.element_size()
            + other.numel() * other.element_size()
        )
        return logical_bytes / latency / 1e6


@pytest.mark.silu_and_mul
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_silu_and_mul():
    def torch_op(x, y):
        return torch.mul(torch.nn.functional.silu(x), y)

    bench = SiluAndMulBenchmark(
        input_fn=binary_input_fn,
        op_name="silu_and_mul",
        gems_op=flag_gems.silu_and_mul,
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.silu_and_mul_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_silu_and_mul_out():
    def gems_op(x, y):
        out = torch.empty_like(x)
        return flag_gems.silu_and_mul_out(x, y, out)

    def torch_op(x, y):
        out = torch.empty_like(x)
        return torch.mul(torch.nn.functional.silu(x), y, out=out)

    bench = SiluAndMulBenchmark(
        input_fn=binary_input_fn,
        op_name="silu_and_mul_out",
        gems_op=gems_op,
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()