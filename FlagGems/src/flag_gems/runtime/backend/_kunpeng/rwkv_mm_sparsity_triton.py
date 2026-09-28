"""Kunpeng sparse GEMV for rwkv_mm_sparsity, fused gather + accumulate.

out[j] = sum_i k[i] * v[i][j], with k ~90-95% zero.

"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@triton.jit
def _collect_rows(k_ptr, idx_ptr, n, SCAN: tl.constexpr, EXACT: tl.constexpr):
    """Write the index of every non-zero of k to idx_ptr, return how many.
    """
    lane = tl.arange(0, SCAN)
    bit = tl.full((SCAN,), 1, tl.int32) << lane
    c = 0
    for b0 in range(0, n, SCAN):
        if EXACT:
            kv = tl.load(k_ptr + b0 + lane)
        else:
            kv = tl.load(k_ptr + b0 + lane, mask=b0 + lane < n, other=0.0)
        bits = tl.sum(tl.where(kv != 0, bit, 0), axis=0)
        while bits != 0:
            low = bits & (-bits)
            # low is a power of two; as a float its exponent is its bit index.
            u = (low.to(tl.uint32).to(tl.float32).to(tl.int32, bitcast=True)
                 >> 23) - 127
            tl.store(idx_ptr + c, b0 + u)
            c += 1
            bits = bits ^ low
    return c


@triton.jit
def _accumulate_rows(
    k_ptr, idx_ptr, v_ptr, out_ptr, offs, mask, nnz, emb,
    UNROLL: tl.constexpr,
    EVEN: tl.constexpr,
    STREAM: tl.constexpr,
):
    main = (nnz // UNROLL) * UNROLL
    for i0 in range(0, main, UNROLL):
        if EVEN:
            acc = tl.load(out_ptr + offs)
        else:
            acc = tl.load(out_ptr + offs, mask=mask, other=0.0)
        for u in tl.static_range(UNROLL):
            r = tl.load(idx_ptr + i0 + u)
            ptrs = v_ptr + r.to(tl.int64) * emb + offs
            if EVEN:
                if STREAM:
                    row = tl.load(ptrs, eviction_policy="evict_first")
                else:
                    row = tl.load(ptrs)
            else:
                if STREAM:
                    row = tl.load(ptrs, mask=mask, other=0.0,
                                  eviction_policy="evict_first")
                else:
                    row = tl.load(ptrs, mask=mask, other=0.0)
            acc += row.to(tl.float32) * tl.load(k_ptr + r).to(tl.float32)
        if EVEN:
            tl.store(out_ptr + offs, acc)
        else:
            tl.store(out_ptr + offs, acc, mask=mask)
    for i in range(main, nnz):
        r = tl.load(idx_ptr + i)
        ptrs = v_ptr + r.to(tl.int64) * emb + offs
        if EVEN:
            acc = tl.load(out_ptr + offs)
            if STREAM:
                row = tl.load(ptrs, eviction_policy="evict_first")
            else:
                row = tl.load(ptrs)
        else:
            acc = tl.load(out_ptr + offs, mask=mask, other=0.0)
            if STREAM:
                row = tl.load(ptrs, mask=mask, other=0.0,
                              eviction_policy="evict_first")
            else:
                row = tl.load(ptrs, mask=mask, other=0.0)
        acc += row.to(tl.float32) * tl.load(k_ptr + r).to(tl.float32)
        if EVEN:
            tl.store(out_ptr + offs, acc)
        else:
            tl.store(out_ptr + offs, acc, mask=mask)


@triton.jit
def _spmv_kernel(
    k_ptr,        # float [n]
    idx_ptr,      # int32 [programs, n + SCAN]   scratch, one row per program
    v_ptr,        # float [n, emb]
    v_stream_ptr, # float [n, emb]   the same buffer as v_ptr, see below
    out_ptr,      # float [emb]
    n,
    emb,
    BLOCK_E: tl.constexpr,
    UNROLL: tl.constexpr,
    SCAN: tl.constexpr,
    EVEN: tl.constexpr,   # emb is a multiple of BLOCK_E
    EXACT: tl.constexpr,  # n is a multiple of SCAN
):
    """One launch: every program finds the non-zeros of k on its own.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    mask = offs < emb
    idx_ptr += pid * (n + SCAN)
    nnz = _collect_rows(k_ptr, idx_ptr, n, SCAN, EXACT)

    if EVEN:
        tl.store(out_ptr + offs, tl.zeros((BLOCK_E,), dtype=tl.float32))
    else:
        tl.store(out_ptr + offs, tl.zeros((BLOCK_E,), dtype=tl.float32),
                 mask=mask)
    if pid == 0:
        _accumulate_rows(k_ptr, idx_ptr, v_stream_ptr, out_ptr, offs, mask,
                         nnz, emb, UNROLL, EVEN, True)
    else:
        _accumulate_rows(k_ptr, idx_ptr, v_ptr, out_ptr, offs, mask,
                         nnz, emb, UNROLL, EVEN, False)


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

# Slice of the output each program owns.
_BLOCK_E = 128

# Rows per trip.
_UNROLL = 16

# k entries per bitmask; at most 32 so the mask fits an int32.
_SCAN = 32


_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _fallback(k, v):
    raise TypeError(f"rwkv_mm_sparsity: unsupported dtype {k.dtype}")


_scratch = {}
_plans = {}
_runners = {}


def _index_buffer(size, device):
    buf = _scratch.get(device)
    if buf is None or buf.numel() < size:
        buf = torch.empty(size, dtype=torch.int32, device=device)
        _scratch[device] = buf
    return buf


def _plan(n, emb):
    plan = _plans.get((n, emb))
    if plan is None:
        grid = triton.cdiv(emb, _BLOCK_E)
        plan = (
            grid,
            grid * (n + _SCAN),
            emb % _BLOCK_E == 0,
            n % _SCAN == 0,
        )
        _plans[(n, emb)] = plan
    return plan


def _compile(k, v, out, idx, n, emb, plan):
    grid, _, even, exact = plan
    ck = _spmv_kernel[(grid,)](
        k, idx, v, v, out, n, emb,
        BLOCK_E=_BLOCK_E, UNROLL=_UNROLL, SCAN=_SCAN, EVEN=even, EXACT=exact,
    )
    try:
        return ck[(grid, 1, 1)]
    except (TypeError, IndexError, AttributeError):
        return None


def rwkv_mm_sparsity_triton(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if k.dtype not in _FAST_DTYPES:
        return _fallback(k, v)
    ks = k.shape
    vs = v.shape
    assert len(ks) == 1 and len(vs) == 2 and ks[0] == vs[0]

    n = ks[0]
    emb = vs[1]
    out_dtype = k.dtype

    v = v.contiguous()
    k = k.contiguous()
    device = k.device
    plan = _plan(n, emb)
    idx = _index_buffer(plan[1], device)
    out = torch.empty(emb, dtype=torch.float32, device=device)

    if (k.data_ptr() | v.data_ptr() | out.data_ptr()) & 15 == 0:
        key = (id(idx), out_dtype, v.dtype, n, emb)
    else:
        key = (
            id(idx), out_dtype, v.dtype, n, emb,
            k.data_ptr() % 16 == 0,
            v.data_ptr() % 16 == 0,
            out.data_ptr() % 16 == 0,
        )
    run = _runners.get(key)
    if run is None:
        run = _compile(k, v, out, idx, n, emb, plan)
        if run is None:
            return out if out_dtype is torch.float32 else out.to(out_dtype)
        _runners[key] = run
    else:
        run(k, idx, v, v, out, n, emb, stream=0)

    return out if out_dtype is torch.float32 else out.to(out_dtype)
