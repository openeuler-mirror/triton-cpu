"""Kunpeng sparse GEMV for rwkv_mm_sparsity, fused gather + accumulate.

out[j] = sum_i k[i] * v[i][j], with k ~90-95% zero.

The ArmPL path this replaces compresses first -- v[nz_mask] copies every
selected row into a new buffer -- and then hands that buffer to cblas_sgemv,
so the selected rows are read twice and written once.  There is no need for
the copy: the accumulation touches each selected row exactly once and every
access along j is contiguous, so the kernel can read v in place.

"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@triton.jit
def _compact_kernel(
    k_ptr,        # float [n]    the sparse vector
    off_ptr,      # int64 [n]    out: element offset of each selected row in v
    w_ptr,        # float [n]    out: the non-zero values of k
    cnt_ptr,      # int32 [1]    out: how many were selected
    n,
    emb,
    CHUNK: tl.constexpr,   # k entries per program, a multiple of BLOCK
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,   # n is exactly num_programs * CHUNK
):
    pid = tl.program_id(0)
    start = pid * CHUNK

    accv = tl.zeros((CHUNK,), dtype=tl.int32)
    if EXACT:
        for i0 in range(0, start, CHUNK):
            kv = tl.load(k_ptr + i0 + tl.arange(0, CHUNK))
            accv += (kv != 0).to(tl.int32)

        c = tl.sum(accv, axis=0)
        for i0 in range(start, start + CHUNK, BLOCK):
            ii = i0 + tl.arange(0, BLOCK)
            kv = tl.load(k_ptr + ii)
            nz = kv != 0
            inz = nz.to(tl.int32)
            pos = tl.cumsum(inz, axis=0) - inz
            tl.store(off_ptr + c + pos, ii.to(tl.int64) * emb, mask=nz)
            tl.store(w_ptr + c + pos, kv.to(tl.float32), mask=nz)
            c += tl.sum(inz, axis=0)
    else:
        for i0 in range(0, start, CHUNK):
            ii = i0 + tl.arange(0, CHUNK)
            kv = tl.load(k_ptr + ii, mask=ii < n, other=0.0)
            accv += (kv != 0).to(tl.int32)

        c = tl.sum(accv, axis=0)
        stop = tl.minimum(start + CHUNK, n)
        for i0 in range(start, stop, BLOCK):
            ii = i0 + tl.arange(0, BLOCK)
            m = ii < n
            kv = tl.load(k_ptr + ii, mask=m, other=0.0)
            nz = (kv != 0) & m
            inz = nz.to(tl.int32)
            pos = tl.cumsum(inz, axis=0) - inz
            tl.store(off_ptr + c + pos, ii.to(tl.int64) * emb, mask=nz)
            tl.store(w_ptr + c + pos, kv.to(tl.float32), mask=nz)
            c += tl.sum(inz, axis=0)

    if pid == tl.num_programs(0) - 1:
        tl.store(cnt_ptr, c)


@triton.jit
def _spmv_slice(
    off_ptr, w_ptr, v_ptr, out_ptr, offs, mask, nnz,
    UNROLL: tl.constexpr,
    EVEN: tl.constexpr,
    STREAM: tl.constexpr,
):
    main = (nnz // UNROLL) * UNROLL
    if EVEN:
        for i0 in range(0, main, UNROLL):
            acc = tl.load(out_ptr + offs)
            for u in tl.static_range(UNROLL):
                off = tl.load(off_ptr + i0 + u)
                if STREAM:
                    row = tl.load(v_ptr + off + offs, eviction_policy="evict_first")
                else:
                    row = tl.load(v_ptr + off + offs)
                acc += row.to(tl.float32) * tl.load(w_ptr + i0 + u)
            tl.store(out_ptr + offs, acc)
        for i in range(main, nnz):
            acc = tl.load(out_ptr + offs)
            off = tl.load(off_ptr + i)
            if STREAM:
                row = tl.load(v_ptr + off + offs, eviction_policy="evict_first")
            else:
                row = tl.load(v_ptr + off + offs)
            acc += row.to(tl.float32) * tl.load(w_ptr + i)
            tl.store(out_ptr + offs, acc)
    else:
        for i0 in range(0, main, UNROLL):
            acc = tl.load(out_ptr + offs, mask=mask, other=0.0)
            for u in tl.static_range(UNROLL):
                off = tl.load(off_ptr + i0 + u)
                if STREAM:
                    row = tl.load(v_ptr + off + offs, mask=mask, other=0.0,
                                  eviction_policy="evict_first")
                else:
                    row = tl.load(v_ptr + off + offs, mask=mask, other=0.0)
                acc += row.to(tl.float32) * tl.load(w_ptr + i0 + u)
            tl.store(out_ptr + offs, acc, mask=mask)
        for i in range(main, nnz):
            acc = tl.load(out_ptr + offs, mask=mask, other=0.0)
            off = tl.load(off_ptr + i)
            if STREAM:
                row = tl.load(v_ptr + off + offs, mask=mask, other=0.0,
                              eviction_policy="evict_first")
            else:
                row = tl.load(v_ptr + off + offs, mask=mask, other=0.0)
            acc += row.to(tl.float32) * tl.load(w_ptr + i)
            tl.store(out_ptr + offs, acc, mask=mask)


@triton.jit
def _spmv_kernel(
    off_ptr,      # int64 [nnz]
    w_ptr,        # float [nnz]
    v_ptr,        # float [n, emb]
    v_stream_ptr, # float [n, emb]   the same buffer as v_ptr, see below
    out_ptr,      # float [emb]
    cnt_ptr,      # int32 [1]    nnz, read on device so the host never syncs
    emb,
    BLOCK_E: tl.constexpr,
    UNROLL: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    nnz = tl.load(cnt_ptr)
    if EVEN:
        mask = offs < emb          # unused on this path; keeps one signature
        tl.store(out_ptr + offs, tl.zeros((BLOCK_E,), dtype=tl.float32))
    else:
        mask = offs < emb
        tl.store(out_ptr + offs, tl.zeros((BLOCK_E,), dtype=tl.float32),
                 mask=mask)
    if pid == 0:
        _spmv_slice(off_ptr, w_ptr, v_stream_ptr, out_ptr, offs, mask, nnz,
                    UNROLL, EVEN, True)
    else:
        _spmv_slice(off_ptr, w_ptr, v_ptr, out_ptr, offs, mask, nnz,
                    UNROLL, EVEN, False)


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

# Slice of the output each program owns.
_BLOCK_E = 128

# Rows per trip.
_UNROLL = 16

# k entries per compaction program.
_COMPACT_PROGS = 32
_COMPACT_BLOCK = 128


_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _fallback(k, v):
    raise TypeError(f"rwkv_mm_sparsity: unsupported dtype {k.dtype}")


_scratch = {}
_plans = {}
_runners = {}


def _index_buffers(n, device):
    buf = _scratch.get(device)
    if buf is None or buf[3] < n:
        buf = (
            torch.empty(n, dtype=torch.int64, device=device),
            torch.empty(n, dtype=torch.float32, device=device),
            torch.empty(1, dtype=torch.int32, device=device),
            n,
        )
        _scratch[device] = buf
    return buf


def _plan(n, emb):
    plan = _plans.get((n, emb))
    if plan is None:
        chunk = triton.cdiv(
            triton.cdiv(n, _COMPACT_PROGS), _COMPACT_BLOCK
        ) * _COMPACT_BLOCK
        g_compact = triton.cdiv(n, chunk)
        plan = (
            chunk,
            g_compact,
            triton.cdiv(emb, _BLOCK_E),
            emb % _BLOCK_E == 0,
            g_compact * chunk == n,
        )
        _plans[(n, emb)] = plan
    return plan


def _compile(k, v, out, off, w, cnt, n, emb, plan):
    chunk, g_compact, g_spmv, even, exact = plan
    _compact_kernel[(g_compact,)](
        k, off, w, cnt, n, emb,
        CHUNK=chunk, BLOCK=_COMPACT_BLOCK, EXACT=exact,
    )
    compact_ck = getattr(_compact_kernel, "_fast_kernel", None)
    _spmv_kernel[(g_spmv,)](
        off, w, v, v, out, cnt, emb,
        BLOCK_E=_BLOCK_E, UNROLL=_UNROLL, EVEN=even,
    )
    spmv_ck = getattr(_spmv_kernel, "_fast_kernel", None)
    if compact_ck is None or spmv_ck is None:
        return None
    try:
        return compact_ck[(g_compact, 1, 1)], spmv_ck[(g_spmv, 1, 1)]
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
    off, w, cnt, _ = _index_buffers(n, device)
    out = torch.empty(emb, dtype=torch.float32, device=device)
    plan = _plan(n, emb)

    if (k.data_ptr() | v.data_ptr() | out.data_ptr()) & 15 == 0:
        key = (id(off), out_dtype, v.dtype, n, emb)
    else:
        key = (
            id(off), out_dtype, v.dtype, n, emb,
            k.data_ptr() % 16 == 0,
            v.data_ptr() % 16 == 0,
            out.data_ptr() % 16 == 0,
        )
    runners = _runners.get(key)
    if runners is None:
        runners = _compile(k, v, out, off, w, cnt, n, emb, plan)
        if runners is None:
            return out if out_dtype is torch.float32 else out.to(out_dtype)
        _runners[key] = runners
    else:
        compact_run, spmv_run = runners
        compact_run(k, off, w, cnt, n, emb, stream=0)
        spmv_run(off, w, v, v, out, cnt, emb, stream=0)

    return out if out_dtype is torch.float32 else out.to(out_dtype)
