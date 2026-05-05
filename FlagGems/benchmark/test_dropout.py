import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.dropout
def test_dropout():
    bench = base.UnaryPointwiseBenchmark(
        op_name="dropout",
        torch_op=lambda x: torch.ops.aten.native_dropout(x, 0.5, True),
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
