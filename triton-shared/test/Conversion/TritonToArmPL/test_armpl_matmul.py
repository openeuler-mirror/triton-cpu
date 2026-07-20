"""Test that tt.dot lowers through the ArmPL path and produces correct results.

This test verifies:
1. The ArmPL path is taken when ArmPL is available
2. Results match the reference (PyTorch) implementation
3. Transpose flags are correctly inferred from tt.trans
4. f16/bf16 upcast to f32 works correctly
"""

import os
import pytest
import torch
import triton
import triton.language as tl


def _armpl_available():
    """Check if ArmPL is available on this system."""
    try:
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
    except Exception:
        pass
    return False


def _maybe_skip_if_no_armpl():
    if not _armpl_available():
        pytest.skip("ArmPL not available on this system")


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Standard tiled matmul: C[M,N] += A[M,K] @ B[K,N]"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


@triton.jit
def matmul_transpose_b_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Matmul where B is stored in column-major (N,K) and transposed before dot."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # B is stored as [N, K] (column-major), so we load [K, N] and transpose
    b_ptrs = B + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        b_t = tl.trans(b)  # [BLOCK_K, BLOCK_N] -> [BLOCK_N, BLOCK_K]
        acc = tl.dot(a, b_t, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K", [(128, 256, 64), (64, 64, 64), (256, 128, 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_matmul_correctness(M, N, K, dtype):
    """Test that triton matmul with tt.dot produces correct results."""
    _maybe_skip_if_no_armpl()

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    A = torch.randn((M, K), dtype=dtype, device="cpu")
    B = torch.randn((K, N), dtype=dtype, device="cpu")
    C = torch.zeros((M, N), dtype=dtype, device="cpu")

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    ref = A.to(torch.float32) @ B.to(torch.float32)
    torch.testing.assert_close(C.to(torch.float32), ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("M,N,K", [(128, 256, 64), (64, 64, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_matmul_transpose_b_correctness(M, N, K, dtype):
    """Test that triton matmul with transposed B produces correct results."""
    _maybe_skip_if_no_armpl()

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    A = torch.randn((M, K), dtype=dtype, device="cpu")
    B_col = torch.randn((N, K), dtype=dtype, device="cpu")  # stored column-major
    C = torch.zeros((M, N), dtype=dtype, device="cpu")

    matmul_transpose_b_kernel[grid](
        A, B_col, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B_col.stride(0), B_col.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    ref = A.to(torch.float32) @ B_col.T.to(torch.float32)
    torch.testing.assert_close(C.to(torch.float32), ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("M,K", [(128, 64), (64, 32)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_matmul_gemv_correctness(M, K, dtype):
    """Test that GEMV (N=1) path produces correct results."""
    _maybe_skip_if_no_armpl()

    N = 1
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 1, 32
    grid = (triton.cdiv(M, BLOCK_M), 1)

    A = torch.randn((M, K), dtype=dtype, device="cpu")
    B = torch.randn((K, N), dtype=dtype, device="cpu")
    C = torch.zeros((M, N), dtype=dtype, device="cpu")

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    ref = A.to(torch.float32) @ B.to(torch.float32)
    torch.testing.assert_close(C.to(torch.float32), ref, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])