import pytest

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
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


@pytest.mark.geglu
@pytest.mark.skipif(flag_gems.device !="cpu" and not TE_AVAILABLE, reason="TransformerEngine not installed")
@pytest.mark.skipif(flag_gems.device !="cpu" and TE_OP is None, reason="'geglu' not found in TransformerEngine")
@pytest.mark.skipif(GEMS_OP is None, reason="'geglu' not found in FlagGems")
def test_geglu():
    if flag_gems.device == "cpu":
        TE_OP = GEMS_OP
    bench = TexGluForwardBenchmark(
        op_name="geglu",
        torch_op=TE_OP,
        gems_op=GEMS_OP,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
