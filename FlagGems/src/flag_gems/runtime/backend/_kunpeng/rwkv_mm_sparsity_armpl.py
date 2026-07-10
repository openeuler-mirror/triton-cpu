import ctypes

import torch

from .armpl import load_armpl

# ---------------------------------------------------------------------------
# ArmPL sparse-aware GEMV
# ---------------------------------------------------------------------------
# The operation is output = v.T @ k where k is ~90% sparse.
# ArmPL's armpl_spgemv_exec operates on sparse_matrix × dense_vector and is
# not a direct fit because here v.T is the dense matrix and k is the sparse
# vector.  The most efficient approach is to compress k and the corresponding
# rows of v to only the non-zero entries, then call cblas_sgemv on the reduced
# system (CblasTrans: A = v_nz [nnz × emb_dim], y = A^T * k_nz).
# With 90% sparsity this reduces the GEMV to 10% of the original work.


def rwkv_mm_sparsity_armpl(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute v.T @ k with dtype-aware backend selection.

    - fp32/fp64: ArmPL cblas_{s,d}gemv on compressed nonzero rows.
    - fp16/bf16: upcast inputs to fp32, then run ArmPL cblas_sgemv.
    """
    assert k.dim() == 1 and v.dim() == 2
    assert k.size(0) == v.size(0)

    orig_dtype = k.dtype

    # Fail fast when ArmPL is unavailable so caller can cleanly fallback
    # without executing extra tensor ops under use_gems() interception.
    armpl = load_armpl()

    if orig_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError(
            f"rwkv_mm_sparsity_armpl only supports float16/bfloat16/float32/float64, got {orig_dtype}"
        )

    emb_dim = v.size(1)

    # Compress to non-zero entries only.
    nz_mask = k != 0
    k_nz = k[nz_mask]
    v_nz = v[nz_mask]  # shape: [nnz, emb_dim]
    nnz = k_nz.shape[0]

    if nnz == 0:
        return torch.zeros(emb_dim, dtype=orig_dtype, device=k.device)

    # ArmPL BLAS path for float32/float64. For fp16/bf16, inputs are upcasted
    # to float32 for cblas_sgemv, then cast back to original dtype.
    use_double = orig_dtype == torch.float64
    work_dtype = torch.float64 if use_double else torch.float32

    k_nz = k_nz.to(work_dtype).contiguous()
    v_nz = v_nz.to(work_dtype).contiguous()  # row-major [nnz, emb_dim]
    output = torch.zeros(emb_dim, dtype=work_dtype, device=k.device)

    # We call cblas_*gemv directly via ctypes instead of going through tl.dot,
    # the reason being that tl.dot materializes its operands as
    # Triton block tensors, which are capped at 1M elements (1048576). For the
    # RWKV workload (n=16384, emb_dim=4096, ~90% sparse → nnz≈1638), the v tile
    # alone is [BLOCK_K, BLOCK_M] = [2048, 4096] = 8M elements — well over the
    # limit — so the tl.dot kernel fails to compile and silently falls back to
    # the slower pure-Triton path. The direct ctypes call has no such limit: it
    # passes raw data_ptr() values and integer dimensions to ArmPL, letting
    # cblas_*gemv handle internal tiling on the full compressed operands in a
    # single call. The tl.dot → ArmPL pass remains the right choice for tiled
    # matmul kernels whose block sizes fit under the numel cap.
    CblasRowMajor = 101
    CblasTrans = 112  # compute A^T * x, i.e. v_nz.T @ k_nz

    if use_double:
        armpl.cblas_dgemv(
            CblasRowMajor,
            CblasTrans,
            nnz,
            emb_dim,
            1.0,
            v_nz.data_ptr(),
            emb_dim,
            k_nz.data_ptr(),
            1,
            0.0,
            output.data_ptr(),
            1,
        )
    else:
        armpl.cblas_sgemv(
            CblasRowMajor,
            CblasTrans,
            nnz,
            emb_dim,
            ctypes.c_float(1.0),
            v_nz.data_ptr(),
            emb_dim,
            k_nz.data_ptr(),
            1,
            ctypes.c_float(0.0),
            output.data_ptr(),
            1,
        )

    return output.to(orig_dtype)
