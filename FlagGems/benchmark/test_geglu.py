import pytest

import flag_gems
from benchmark.attri_util import DEFAULT_METRICS, FLOAT_DTYPES
from benchmark.performance_utils import TexGluForwardBenchmark

# Note: Importing transformer_engine (especially in some versions like py 3.10) may automatically
# configure the Root Logger (adding handlers). This may cause subsequent `logging.basicConfig`
# calls (used by FlagGems benchmark) to be ignored/no-op, leading to missing result log files.
# See: https://github.com/NVIDIA/TransformerEngine/issues/1065
GEMS_OP = getattr(flag_gems, "geglu")
try:
    from transformer_engine.pytorch import cpp_extensions as tex

    TE_OP = getattr(tex, "geglu")
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False
    TE_OP = None


class GegluBenchmark(TexGluForwardBenchmark):
    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["gbps"]

    def get_gbps(self, args, latency):
        inp = args[0]
        logical_bytes = (
            inp.numel() + inp.numel() // 2
        ) * inp.element_size()
        return logical_bytes / latency / 1e6


@pytest.mark.geglu
@pytest.mark.skipif(flag_gems.device !="cpu" and not TE_AVAILABLE, reason="TransformerEngine not installed")
@pytest.mark.skipif(flag_gems.device !="cpu" and TE_OP is None, reason="'geglu' not found in TransformerEngine")
@pytest.mark.skipif(GEMS_OP is None, reason="'geglu' not found in FlagGems")
def test_geglu():
    if flag_gems.device == "cpu":
        TE_OP = GEMS_OP
    bench = GegluBenchmark(
        op_name="geglu",
        torch_op=TE_OP,
        gems_op=GEMS_OP,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
