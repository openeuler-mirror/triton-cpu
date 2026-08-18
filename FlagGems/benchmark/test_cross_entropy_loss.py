import pytest
import torch

import flag_gems

from . import attri_util as attr_utils
from . import performance_utils as utils


def cross_entropy_loss_input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    target = torch.randint(0, shape[-1], (shape[0],), device=device)
    yield inp, target

    if utils.Config.bench_level == utils.BenchLevel.COMPREHENSIVE:
        weight = torch.randn(shape[-1], dtype=cur_dtype, device=device)
        yield inp, target, {"weight": weight, "ignore_index": 1, "reduction": "none"}
        yield inp, target, {
            "weight": weight,
            "reduction": "sum",
            "label_smoothing": 0.1,
        }


class CrossEntropyLossBenchmark(utils.GenericBenchmark2DOnly):
    DEFAULT_METRICS = utils.Benchmark.DEFAULT_METRICS[:] + ["gbps"]

    def unpack_to_args_kwargs(self, input_tuple):
        args, kwargs = super().unpack_to_args_kwargs(input_tuple)
        self.metric_kwargs = kwargs
        return args, kwargs

    def get_gbps(self, args, latency):
        logits, target = args[:2]
        weight = self.metric_kwargs.get("weight")
        reduction = self.metric_kwargs.get("reduction", "mean")
        output_elements = target.numel() if reduction == "none" else 1
        logical_bytes = (
            logits.numel() * logits.element_size()
            + target.numel() * target.element_size()
            + (0 if weight is None else weight.numel() * weight.element_size())
            + output_elements * logits.element_size()
        )
        return logical_bytes / latency / 1e6


@pytest.mark.cross_entropy_loss
def test_cross_entropy_loss():
    bench = CrossEntropyLossBenchmark(
        input_fn=cross_entropy_loss_input_fn,
        op_name="cross_entropy_loss",
        torch_op=torch.nn.functional.cross_entropy,
        dtypes=attr_utils.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.cross_entropy_loss)
    bench.run()
