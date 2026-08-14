# Tests for the print functionality of the Triton-CPU backend's PrintOpConverter.

import os
import random
import re
import sys

import pytest

import torch

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Print kernel definitions
# ---------------------------------------------------------------------------

@triton.jit
def kernel_device_print(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    tl.device_print("x:", x)
    tl.store(Y + tl.arange(0, BLOCK), x)


@triton.jit
def kernel_device_print_scalar(SCALAR):
    x = tl.load(SCALAR)
    tl.device_print("x:", x)


@triton.jit
def kernel_print(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    print("x:", x)
    tl.store(Y + tl.arange(0, BLOCK), x)


@triton.jit
def kernel_device_print_multiple_args(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    y = tl.full((BLOCK, ), 1, tl.int32)
    tl.device_print("", x, y)
    tl.store(Y + tl.arange(0, BLOCK), y)


# ---------------------------------------------------------------------------
# Test execution and output capture helpers
# ---------------------------------------------------------------------------

# List of kernels participating in the combinatoric (func, dtype) tests.
func_list = ["device_print", "device_print_scalar", "print", "device_print_multiple_args"]

# Data types covered by the combinatoric tests.
torch_types = ["int8", "uint8", "int16", "int32", "long", "float16", "float32", "float64"]

# Integer types (printed via %PRId64 at runtime, comparable exactly).
INT_TYPES = {"int8", "uint8", "int16", "int32", "long"}


def _make_value(data_type: str) -> float:
    """
    Generate a random "real" value per data type, used for equality assertions.
    For integers, pick a non-zero random integer within the type's range; for
    floats, pick a random fractional value.
    """
    if data_type == "int8":
        return random.randint(-100, 100)
    if data_type == "uint8":
        return random.randint(0, 200)
    if data_type == "int16":
        return random.randint(-30000, 30000)
    if data_type in ("int32", "long"):
        return random.randint(-1000000, 1000000)
    # Float types: random fractional value.
    return round(random.uniform(1.0, 1000.0), 3)


def _extract_number(out: str, marker: str):
    """
    Locate the first numeric string after marker in the print output.
    Returns None if the marker is not found.
    """
    idx = out.splitlines()[0].rfind(marker)
    if idx == -1:
        return None
    tail = out[idx + len(marker):]
    m = re.search(r"(-?[\d.eE+]+)", tail)
    return m.group(1) if m else None


def _assert_value(out: str, marker: str, expected: float, data_type: str):
    """
    Parse the printed numeric value and assert it equals the expected value.
    Integers are compared exactly; floats are compared approximately (since %lg
    prints the shortest representation of the promoted float32/float64 value,
    which may differ from the Python-side value by tiny rounding).
    """
    token = _extract_number(out, marker)
    assert token is not None, f"no numeric value parsed after {marker}, output:\n{out}"
    if data_type in INT_TYPES:
        assert int(token) == int(expected), \
            f"integer print mismatch: expected {expected}, got {token}, output:\n{out}"
    else:
        assert float(token) == pytest.approx(expected, rel=1e-4, abs=1e-4), \
            f"float print mismatch: expected {expected}, got {token}, output:\n{out}"


def _run_kernel(func_type: str, data_type: str, device: str, N: int, v):
    """
    Build inputs per (kernel, dtype) and run the corresponding kernel.
    Since PrintOpConverter only prints the element at index 0 of a tensor, the
    random value v is placed at index 0. Returns the actually stored value
    (low-precision types such as float16 round on write), used as the printed
    expected value.
    """
    x = torch.zeros((N, ), dtype=getattr(torch, data_type), device=device)
    x[0] = v
    y = torch.zeros((N, ), dtype=x.dtype, device=device)
    if func_type == "device_print":
        kernel_device_print[(1, )](x, y, BLOCK=N)
        stored = x[0].item()
    elif func_type == "device_print_scalar":
        scalar = torch.tensor(v, dtype=x.dtype, device=device)
        kernel_device_print_scalar[(1, )](scalar)
        stored = scalar.item()
    elif func_type == "print":
        kernel_print[(1, )](x, y, BLOCK=N)
        stored = x[0].item()
    elif func_type == "device_print_multiple_args":
        kernel_device_print_multiple_args[(1, )](x, y, BLOCK=N)
        stored = x[0].item()
    else:
        raise AssertionError(f"unknown kernel: {func_type}")
    return stored


def _run_and_capture(func_type: str, data_type: str, device: str, v):
    """
    Run a print kernel and capture standard output, returning (output, stored value).

    Since the println_* runtime helpers write directly to fd 1 (bypassing
    sys.stdout), the real output is captured by redirecting fd 1 to a temp file.
    The stored value is used for the printed-value assertion.
    """
    import tempfile

    N = 128
    saved_fd = os.dup(1)
    tmp = tempfile.TemporaryFile("w+t")
    try:
        sys.stdout.flush()
        os.dup2(tmp.fileno(), 1)
        stored = _run_kernel(func_type, data_type, device, N, v)
        torch.cpu.synchronize()
        sys.stdout.flush()
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        tmp.seek(0)
        out = tmp.read()
        tmp.close()
    return out, stored


# ---------------------------------------------------------------------------
# Test cases (combinatoric parametrization)
# ---------------------------------------------------------------------------
@pytest.mark.cpu
@pytest.mark.interpreter
@pytest.mark.parametrize("func_type, data_type", [
    (fn, data_type)
    for fn in func_list
    for data_type in torch_types
])
def test_print(func_type: str, data_type: str, device: str = "cpu"):
    """
    Run (kernel, dtype) cases, print a random real value and assert equality.
    The stored value after write-back is used as the expected value to eliminate
    the storage rounding of low-precision types (such as float16).
    """
    v = _make_value(data_type)
    out, stored = _run_and_capture(func_type, data_type, device, v)

    if func_type == "device_print_multiple_args":
        # Two operands: the prefix is normalized to ": ", the first is x(stored)
        # and the second is y(1).
        _assert_value(out, ":", stored, data_type)
        assert "1" in out, f"integer argument y not printed, output:\n{out}"
    else:
        # device_print / print / device_print_scalar all use the "x:" prefix.
        _assert_value(out, "x:", stored, data_type)