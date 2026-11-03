import importlib

from .unique import _unique2

# flag_gems.ops.isin is shadowed by the isin function, so import the module.
_isin = importlib.import_module("flag_gems.ops.isin")

# Reuse the generic isin, but make its unique step use the kunpeng _unique2.
_isin._unique2 = _unique2
isin = _isin.isin
