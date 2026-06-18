import ctypes
import os

_armpl_lib = None
_armpl_load_error = None


def load_armpl():
    """Return cached ArmPL handle configured for cblas_sgemv/cblas_dgemv."""
    global _armpl_lib, _armpl_load_error
    if _armpl_lib is not None:
        return _armpl_lib
    if _armpl_load_error is not None:
        raise RuntimeError(_armpl_load_error)

    candidates = ["libarmpl_lp64.so", "libarmpl.so", "libarmpl_lp64_mp.so"]
    armpl_dir = os.environ.get("ARMPL_DIR", "")
    if armpl_dir:
        candidates = [os.path.join(armpl_dir, "lib", n) for n in candidates] + candidates

    for name in candidates:
        try:
            lib = ctypes.CDLL(name)
            lib.cblas_sgemv.restype = None
            lib.cblas_sgemv.argtypes = [
                ctypes.c_int,    # order
                ctypes.c_int,    # transA
                ctypes.c_int,    # M
                ctypes.c_int,    # N
                ctypes.c_float,  # alpha
                ctypes.c_void_p, # A
                ctypes.c_int,    # lda
                ctypes.c_void_p, # x
                ctypes.c_int,    # incx
                ctypes.c_float,  # beta
                ctypes.c_void_p, # y
                ctypes.c_int,    # incy
            ]
            lib.cblas_dgemv.restype = None
            lib.cblas_dgemv.argtypes = [
                ctypes.c_int,    # order
                ctypes.c_int,    # transA
                ctypes.c_int,    # M
                ctypes.c_int,    # N
                ctypes.c_double, # alpha
                ctypes.c_void_p, # A
                ctypes.c_int,    # lda
                ctypes.c_void_p, # x
                ctypes.c_int,    # incx
                ctypes.c_double, # beta
                ctypes.c_void_p, # y
                ctypes.c_int,    # incy
            ]
            _armpl_lib = lib
            return lib
        except OSError:
            continue

    _armpl_load_error = (
        "Could not load ArmPL. Install ArmPL and either set ARMPL_DIR or "
        "ensure libarmpl_lp64.so is on LD_LIBRARY_PATH."
    )
    raise RuntimeError(_armpl_load_error)


# Optional eager preload: set FLAG_GEMS_PRELOAD_ARMPL=1 to resolve ArmPL once
# at import time and cache both success/failure globally for this process.
if os.environ.get("FLAG_GEMS_PRELOAD_ARMPL", "0") == "1":
    try:
        load_armpl()
    except RuntimeError:
        pass
