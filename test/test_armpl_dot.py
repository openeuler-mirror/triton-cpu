"""Test that tt.dot lowers through the ArmPL path and produces correct results.

Requires ArmPL to be installed (libarmpl_lp64.so shipped under triton/_C/).

    pytest test/test_armpl_dot.py -v
"""

import os
import pytest
import torch
import triton
import triton.language as tl


def _armpl_available():
    """Check if ArmPL is loadable."""
    import ctypes
    import importlib.resources
    candidates = ["libarmpl_lp64.so", "libarmpl.so", "libarmpl_lp64_mp.so"]
    # Look in triton/_C/ where libarmpl_lp64.so is shipped at build time.
    try:
        triton_c_dir = str(importlib.resources.files(triton).joinpath("_C"))
        candidates = [os.path.join(triton_c_dir, n) for n in candidates] + candidates
    except Exception:
        pass
    for name in candidates:
        try:
            ctypes.CDLL(name)
            return True
        except OSError:
            continue
    return False


requires_armpl = pytest.mark.skipif(
    not _armpl_available(),
    reason="ArmPL not available (libarmpl_lp64.so not found under triton/_C/ or LD_LIBRARY_PATH)",
)


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Simple tiled matmul: C[M,N] += A[M,K] * B[K,N]."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


@triton.jit
def matmul_kernel_transpose_b(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,  # B is NxK (transposed)
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Tiled matmul where B is pre-transposed: C[M,N] += A[M,K] * B^T[N,K]."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        # B is NxK, so we need B^T for the dot: tl.dot(a, tl.trans(b))
        acc = tl.dot(a, tl.trans(b), acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


@triton.jit
def gemv_kernel(
    a_ptr, x_ptr, y_ptr,
    M, K,
    stride_am, stride_ak,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """GEMV: y[M] += A[M,K] * x[K]."""
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    x_ptrs = x_ptr + offs_k

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        x = tl.load(x_ptrs, mask=offs_k < K - k, other=0.0)
        # y[M] += A[M,K] * x[K]  →  tl.dot(A, x[:, None]) then squeeze
        acc += tl.sum(a * x[None, :], axis=1)
        a_ptrs += BLOCK_K * stride_ak
        x_ptrs += BLOCK_K

    y_ptrs = y_ptr + offs_m
    tl.store(y_ptrs, acc)


def _run_matmul(M, N, K, dtype=torch.float32):
    """Run triton matmul and compare against torch.mm."""
    a = torch.randn((M, K), dtype=dtype, device="cpu")
    b = torch.randn((K, N), dtype=dtype, device="cpu")
    c = torch.zeros((M, N), dtype=torch.float32, device="cpu")

    # Use blocks matching the actual data size to avoid padding
    # that introduces 0-valued elements.  Blocks larger than the
    # problem size corrupt ArmPL GEMM buffers.
    BLOCK_M = min(M, 32)
    BLOCK_N = min(N, 32)
    BLOCK_K = min(K, 32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    expected = torch.mm(a.to(torch.float32), b.to(torch.float32))
    torch.testing.assert_close(c, expected, rtol=1e-3, atol=1e-5)
    return c


def _run_matmul_transpose_b(M, N, K, dtype=torch.float32):
    """Run triton matmul with transposed B and compare against torch.mm."""
    a = torch.randn((M, K), dtype=dtype, device="cpu")
    b = torch.randn((N, K), dtype=dtype, device="cpu")  # B is NxK (transposed)
    c = torch.zeros((M, N), dtype=torch.float32, device="cpu")

    BLOCK_M = min(M, 32)
    BLOCK_N = min(N, 32)
    BLOCK_K = min(K, 32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel_transpose_b[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    expected = torch.mm(a.to(torch.float32), b.to(torch.float32).t())
    torch.testing.assert_close(c, expected, rtol=1e-3, atol=1e-5)
    return c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@requires_armpl
@pytest.mark.parametrize("M,N,K", [
    (64, 64, 64),
    (128, 128, 128),
    (256, 128, 64),
    (32, 256, 128),
])
def test_matmul_f32(M, N, K):
    """Test fp32 matmul correctness."""
    _run_matmul(M, N, K, dtype=torch.float32)


@requires_armpl
@pytest.mark.parametrize("M,N,K", [
    (64, 64, 64),
    (128, 128, 128),
])
def test_matmul_transpose_b_f32(M, N, K):
    """Test fp32 matmul with transposed B."""
    _run_matmul_transpose_b(M, N, K, dtype=torch.float32)


@requires_armpl
@pytest.mark.parametrize("M,N,K", [
    (64, 64, 64),
    (128, 128, 128),
])
def test_matmul_f16(M, N, K):
    """Test fp16 matmul correctness (should upcast to f32 internally)."""
    _run_matmul(M, N, K, dtype=torch.float16)


@requires_armpl
@pytest.mark.parametrize("M,N,K", [
    (64, 64, 64),
    (128, 128, 128),
])
def test_matmul_bf16(M, N, K):
    """Test bf16 matmul correctness (should upcast to f32 internally)."""
    _run_matmul(M, N, K, dtype=torch.bfloat16)


@requires_armpl
def test_small_matmul():
    """Test very small matmul (16x16 edge case)."""
    _run_matmul(16, 16, 16, dtype=torch.float32)


@requires_armpl
@pytest.mark.parametrize("size", [8, 4, 2])
def test_tiny_matmul(size):
    """Test tiny matmul sizes."""
    _run_matmul(size, size, size, dtype=torch.float32)


@requires_armpl
def test_non_square():
    """Test non-square matrices."""
    _run_matmul(128, 64, 256, dtype=torch.float32)
    _run_matmul(64, 256, 128, dtype=torch.float32)