// Flash-attention forward for Arm SME (bf16 / fp16 inputs, fp32 accumulation).
//
// Both GEMMs (S = Q K^T and O += P V) run as widening outer products
// (BFMOPA / FMOPA za32) into the four 32-bit ZA tiles; softmax runs as
// streaming SVE in the same streaming region.  Requires SVL = 512 bit
// (16 fp32 lanes); the entry points return -3 on any other vector length.
//
// Part of libkunpeng_llm.so.  The triton-shared compiler replaces a recognised
// Triton flash-attention forward kernel by one call per program (one block of
// query rows of one head) to kunpeng_flash_attention_forward_{bf16,f16}, see
// kunpeng_llm.h.  The Triton launcher provides the parallelism.
//
// Data layouts (L = 16 = fp32 lanes per vector, "pair" = two adjacent 16-bit
// elements along the contraction dimension, viewed as one uint32):
//   Qp [rt][p][L]      u32  pair p of query rows rt*L .. rt*L+15
//   Kp [kt][p][L]      u32  pair p of keys kt*L .. kt*L+15
//   Vp [kp][dt][2L]    u16  keys (2kp, 2kp+1) interleaved, d in dt*L .. dt*L+15
//   ST [j][BM]         f32  scores, transposed: lane = query row
//   Pp [rt][kp][2L]    u16  probabilities, keys (2kp, 2kp+1) interleaved
//   OT [d][BM]         f32  output accumulator, transposed
// The transposed ST/OT layouts make the softmax row statistics plain
// lane-wise vector ops (no horizontal reductions).

#include <arm_sme.h>
#include <arm_sve.h>
#include <math.h>
#include "kunpeng_llm.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define L 16
#define MAX_D 256

// One (batch, head): the pointers address row 0 of that head.
typedef struct {
  const uint16_t *q, *k, *v;
  uint16_t *o;
  // row strides, in elements
  int64_t q_rs, k_rs, v_rs, o_rs;
  int64_t Sq, Sk;
  // key j is visible from row r iff 0 <= j < Sk and, for a bounded side,
  // r + left <= j (has_left) and j <= r + right (has_right)
  int64_t left, right;
  int32_t has_left, has_right;
  int32_t D;
  int32_t block_m; // multiple of 16, <= 128
  int32_t chunk_n; // keys per online-softmax chunk, multiple of 16
  float scale_log2; // softmax_scale * log2(e)
} fa_args;

int kunpeng_llm_sme_vector_bytes(void) { return (int)svcntsb(); }

// Per-thread scratch, kept across calls so steady-state calls do not fault in
// fresh pages.
//
// Packed K/V tiles (16 keys each) live in a direct-mapped cache of `ntile`
// slots: slot = key_tile % ntile, tag[slot] = key tile held.  Sliding-window
// and causal blocks reuse the tiles of the previous block; long rows simply
// repack.  The cache is sized to stay L2-resident.
//
// Consecutive calls on a thread are usually consecutive row blocks of the
// same head (the launcher hands out chunks of consecutive programs), so the
// cache survives across calls while the head (key_*) and the launch (epoch)
// stay the same.  K and V are read-only within one launch but may be
// rewritten in place between launches, hence the epoch.
typedef struct {
  uint32_t *Qp, *Kp;
  uint16_t *Vp, *Pp;
  float *ST, *OT, *m, *l, *inv;
  int64_t *tag;
  int64_t ntile;
  size_t bytes;
  void *mem;
  // what the tags refer to; key_valid = 0 after (re)allocation
  int key_valid;
  const void *key_k, *key_v;
  int64_t key_k_rs, key_v_rs, key_Sk;
  int32_t key_D;
  uint64_t key_epoch;
} fa_scratch;

static _Atomic uint64_t launch_epoch = 1;

void kunpeng_llm_begin_launch(void) {
  atomic_fetch_add_explicit(&launch_epoch, 1, memory_order_relaxed);
}

static __thread fa_scratch tls_scratch;

#define KV_CACHE_BYTES (384 << 10)

static fa_scratch *get_scratch(const fa_args *a, int64_t rows) {
  fa_scratch *s = &tls_scratch;
  const int64_t BM = a->block_m, NC = a->chunk_n, D = a->D;
  int64_t ntile = KV_CACHE_BYTES / (L * D * 2 * 2);
  if (ntile < NC / L + 1)
    ntile = NC / L + 1;
  // The K/V tile cache comes first: its layout depends only on D and ntile
  // (both part of the cache key), not on the row count or block size of the
  // call, so cached tiles stay where the next call looks for them.
  size_t sz[10] = {
      ntile * L * D * 2, // Kp
      ntile * L * D * 2, // Vp
      ntile * 8,         // tag
      BM * D * 2,        // Qp
      BM * NC * 2,       // Pp
      NC * BM * 4,       // ST
      D * BM * 4,        // OT
      (rows + BM) * 4,   // m
      (rows + BM) * 4,   // l
      BM * 4,            // inv
  };
  size_t total = 0;
  for (int i = 0; i < 10; i++)
    total += (sz[i] + 255) & ~(size_t)255;
  if (total > s->bytes) {
    free(s->mem);
    s->mem = aligned_alloc(256, total);
    s->bytes = s->mem ? total : 0;
    s->key_valid = 0;
    if (!s->mem)
      return NULL;
  }
  char *p = (char *)s->mem;
  void *ptr[10];
  for (int i = 0; i < 10; i++) {
    ptr[i] = p;
    p += (sz[i] + 255) & ~(size_t)255;
  }
  s->Kp = ptr[0];
  s->Vp = ptr[1];
  s->tag = ptr[2];
  s->Qp = ptr[3];
  s->Pp = ptr[4];
  s->ST = ptr[5];
  s->OT = ptr[6];
  s->m = ptr[7];
  s->l = ptr[8];
  s->inv = ptr[9];
  if (s->ntile != ntile)
    s->key_valid = 0;
  s->ntile = ntile;

  const uint64_t epoch =
      atomic_load_explicit(&launch_epoch, memory_order_relaxed);
  if (!s->key_valid || s->key_epoch != epoch || s->key_k != a->k ||
      s->key_v != a->v || s->key_k_rs != a->k_rs || s->key_v_rs != a->v_rs ||
      s->key_Sk != a->Sk || s->key_D != a->D) {
    for (int64_t i = 0; i < ntile; i++)
      s->tag[i] = -1;
    s->key_valid = 1;
    s->key_epoch = epoch;
    s->key_k = a->k;
    s->key_v = a->v;
    s->key_k_rs = a->k_rs;
    s->key_v_rs = a->v_rs;
    s->key_Sk = a->Sk;
    s->key_D = a->D;
  }
  return s;
}

// The streaming-mode helpers take fa_args / fa_scratch by value so that, once
// inlined, the fields stay in registers instead of being reloaded after every
// vector store.
static inline int64_t row_lb(const fa_args a,
                             int64_t r) __arm_streaming_compatible {
  if (!a.has_left)
    return 0;
  int64_t x = r + a.left;
  return x < 0 ? 0 : x;
}

static inline int64_t row_rb(const fa_args a,
                             int64_t r) __arm_streaming_compatible {
  if (!a.has_right)
    return a.Sk - 1;
  int64_t x = r + a.right;
  return x > a.Sk - 1 ? a.Sk - 1 : x;
}

// ---------------------------------------------------------------------------
// Packing
// ---------------------------------------------------------------------------

#define FA_LD_ROWS(t, u)                                                       \
  do {                                                                         \
    svbool_t pw = svwhilelt_b32((u), npair);                                   \
    for (int r = 0; r < L; r++)                                                \
      svld1_hor_za32(t, r, (rowok >> r) & 1 ? pw : svpfalse(), src[r] + (u));  \
  } while (0)
#define FA_ST_COLS(t, u)                                                       \
  do {                                                                         \
    int nu = npair - (u) < L ? npair - (u) : L;                                \
    for (int p = 0; p < nu; p++)                                               \
      svst1_ver_za32(t, p, svptrue_b32(), dst + (int64_t)((u) + p) * L);       \
  } while (0)

// Transpose L rows of `npair` u32 pairs into dst[p][L], rotating through the
// four ZA tiles so the loads of chunk i overlap the stores of chunk i - 1.
// Rows outside [0, nrows_valid) are packed as zeros.  Clobbers all of ZA.
static inline void pack_rows_u32(uint32_t *dst, const uint16_t *base,
                                 int64_t row_stride, int64_t row0,
                                 int64_t nrows_valid,
                                 int npair) __arm_streaming __arm_inout("za") {
  const uint32_t *src[L];
  unsigned rowok = 0;
  for (int r = 0; r < L; r++) {
    const int64_t row = row0 + r;
    const int ok = row >= 0 && row < nrows_valid;
    rowok |= (unsigned)ok << r;
    src[r] = (const uint32_t *)(base + (ok ? row : 0) * row_stride);
  }
  for (int u = 0; u < npair; u += 4 * L) {
    FA_LD_ROWS(0, u);
    if (u > 0)
      FA_ST_COLS(3, u - L);
    if (u + L >= npair) {
      FA_ST_COLS(0, u);
      break;
    }
    FA_LD_ROWS(1, u + L);
    FA_ST_COLS(0, u);
    if (u + 2 * L >= npair) {
      FA_ST_COLS(1, u + L);
      break;
    }
    FA_LD_ROWS(2, u + 2 * L);
    FA_ST_COLS(1, u + L);
    if (u + 3 * L >= npair) {
      FA_ST_COLS(2, u + 2 * L);
      break;
    }
    FA_LD_ROWS(3, u + 3 * L);
    FA_ST_COLS(2, u + 2 * L);
    if (u + 4 * L >= npair) {
      FA_ST_COLS(3, u + 3 * L);
      break;
    }
  }
}
#undef FA_LD_ROWS
#undef FA_ST_COLS

// Pack values [k0, k0 + nk) (nk even) of one head.
static inline void pack_v(uint16_t *Vp, const uint16_t *vbase, int64_t v_rs,
                          int64_t k0, int nk, int64_t Sk,
                          int D) __arm_streaming {
  const svbool_t ph = svwhilelt_b16(0, L), pt = svptrue_b16();
  const int ndt = D / L;
  for (int kp = 0; kp < nk / 2; kp++) {
    const int64_t k = k0 + 2 * kp;
    const int ok0 = k >= 0 && k < Sk, ok1 = k + 1 >= 0 && k + 1 < Sk;
    const svbool_t p0 = ok0 ? ph : svpfalse(), p1 = ok1 ? ph : svpfalse();
    const uint16_t *r0 = vbase + (ok0 ? k : 0) * v_rs;
    const uint16_t *r1 = vbase + (ok1 ? k + 1 : 0) * v_rs;
    uint16_t *dst = Vp + (int64_t)kp * ndt * 2 * L;
    int dt = 0;
    for (; dt + 4 <= ndt; dt += 4) {
      svuint16_t a0 = svld1_u16(p0, r0 + dt * L);
      svuint16_t b0 = svld1_u16(p1, r1 + dt * L);
      svuint16_t a1 = svld1_u16(p0, r0 + (dt + 1) * L);
      svuint16_t b1 = svld1_u16(p1, r1 + (dt + 1) * L);
      svuint16_t a2 = svld1_u16(p0, r0 + (dt + 2) * L);
      svuint16_t b2 = svld1_u16(p1, r1 + (dt + 2) * L);
      svuint16_t a3 = svld1_u16(p0, r0 + (dt + 3) * L);
      svuint16_t b3 = svld1_u16(p1, r1 + (dt + 3) * L);
      svst1_u16(pt, dst + dt * 2 * L, svzip1_u16(a0, b0));
      svst1_u16(pt, dst + (dt + 1) * 2 * L, svzip1_u16(a1, b1));
      svst1_u16(pt, dst + (dt + 2) * 2 * L, svzip1_u16(a2, b2));
      svst1_u16(pt, dst + (dt + 3) * 2 * L, svzip1_u16(a3, b3));
    }
    for (; dt < ndt; dt++) {
      svuint16_t a = svld1_u16(p0, r0 + dt * L);
      svuint16_t b = svld1_u16(p1, r1 + dt * L);
      svst1_u16(pt, dst + dt * 2 * L, svzip1_u16(a, b));
    }
  }
}

// ---------------------------------------------------------------------------
// exp2 for fp32 vectors.  x <= -127 (including -inf) gives 2^-127 rather
// than 0: masked scores contribute ~1e-38 relative weight, and rows with no
// visible key at all are detected through their running max instead.
// ---------------------------------------------------------------------------
static inline svfloat32_t exp2v(svbool_t pg, svfloat32_t x) __arm_streaming {
  x = svmax_n_f32_x(pg, x, -127.0f);
  svfloat32_t n = svrintn_f32_x(pg, x);
  svfloat32_t f = svsub_f32_x(pg, x, n);
  svfloat32_t r = svdup_f32(1.535336188319500e-4f);
  r = svmad_f32_x(pg, r, f, svdup_f32(1.339887440266574e-3f));
  r = svmad_f32_x(pg, r, f, svdup_f32(9.618437357674640e-3f));
  r = svmad_f32_x(pg, r, f, svdup_f32(5.550332471162809e-2f));
  r = svmad_f32_x(pg, r, f, svdup_f32(2.402264791363012e-1f));
  r = svmad_f32_x(pg, r, f, svdup_f32(6.931472028550421e-1f));
  r = svmad_f32_x(pg, r, f, svdup_f32(1.0f));
  return svscale_f32_x(pg, r, svcvt_s32_f32_x(pg, n));
}

// Prefetch rows [r0, r1) (clipped to [0, hi)) of a [rows][D] 16-bit matrix.
static inline void prefetch_rows(const uint16_t *base, int64_t rs, int64_t r0,
                                 int64_t r1, int64_t hi,
                                 int D) __arm_streaming {
  if (r0 < 0)
    r0 = 0;
  if (r1 > hi)
    r1 = hi;
  for (int64_t r = r0; r < r1; r++)
    for (int off = 0; off < D * 2; off += 64)
      __builtin_prefetch((const char *)(base + r * rs) + off, 0, 2);
}

#define FA_SUFFIX bf16
#define FA_MOPA(t, a, b)                                                       \
  svmopa_za32_bf16_m(t, svptrue_b16(), svptrue_b16(),                          \
                     svreinterpret_bf16_u16(a), svreinterpret_bf16_u16(b))
#define FA_CVT_PAIR(pg, lo, hi)                                                \
  svreinterpret_u16_bf16(svcvtnt_bf16_f32_x(svcvt_bf16_f32_x(pg, lo), pg, hi))
#include "flash_attn_fwd_sme_impl.h"
#undef FA_SUFFIX
#undef FA_MOPA
#undef FA_CVT_PAIR

#define FA_SUFFIX f16
#define FA_MOPA(t, a, b)                                                       \
  svmopa_za32_f16_m(t, svptrue_b16(), svptrue_b16(), svreinterpret_f16_u16(a), \
                    svreinterpret_f16_u16(b))
#define FA_CVT_PAIR(pg, lo, hi)                                                \
  svreinterpret_u16_f16(svcvtnt_f16_f32_x(svcvt_f16_f32_x(pg, lo), pg, hi))
#include "flash_attn_fwd_sme_impl.h"
#undef FA_SUFFIX
#undef FA_MOPA
#undef FA_CVT_PAIR

// Sliding windows: small query blocks keep the computed key range close to
// the window (16 + W keys per row instead of BM + W).  Otherwise larger
// blocks amortise the K/V packing over more rows.
static void set_blocking(fa_args *a, int64_t num_rows) {
  int64_t bm = a->has_left ? 16 : 64;
  const int64_t rows16 = (num_rows + L - 1) / L * L;
  if (bm > rows16)
    bm = rows16;
  a->block_m = (int32_t)bm;
  a->chunk_n = 512;
}

static void fail(const char *what) {
  fprintf(stderr, "libkunpeng_llm: kunpeng_flash_attention_forward: %s\n",
          what);
  abort();
}

#define FA_ENTRY(suffix)                                                       \
  void kunpeng_flash_attention_forward_##suffix(                               \
      const void *q, int64_t q_rs, const void *k, int64_t k_rs, const void *v, \
      int64_t v_rs, void *o, int64_t o_rs, float *lse, int64_t row_start,      \
      int64_t num_rows, int64_t seqlen_q, int64_t seqlen_k,                    \
      int64_t window_flags, int64_t left_shift, int64_t right_shift,           \
      int64_t head_dim, float scale_log2) {                                    \
    if (head_dim % L != 0 || head_dim <= 0 || head_dim > MAX_D)                \
      fail("unsupported head_dim");                                            \
    if (svcntsb() != L * 4)                                                    \
      fail("needs 512-bit SME vectors "                                        \
           "(/proc/sys/abi/sme_default_vector_length = 64)");                  \
    if (num_rows <= 0 || row_start >= seqlen_q)                                \
      return;                                                                  \
    fa_args a;                                                                 \
    a.q = (const uint16_t *)q;                                                 \
    a.k = (const uint16_t *)k;                                                 \
    a.v = (const uint16_t *)v;                                                 \
    a.o = (uint16_t *)o;                                                       \
    a.q_rs = q_rs;                                                             \
    a.k_rs = k_rs;                                                             \
    a.v_rs = v_rs;                                                             \
    a.o_rs = o_rs;                                                             \
    a.Sq = seqlen_q;                                                           \
    a.Sk = seqlen_k;                                                           \
    a.left = left_shift;                                                       \
    a.right = right_shift;                                                     \
    a.has_left = (window_flags & KUNPENG_FA_WINDOW_LEFT) != 0;                 \
    a.has_right = (window_flags & KUNPENG_FA_WINDOW_RIGHT) != 0;               \
    a.D = (int32_t)head_dim;                                                   \
    a.scale_log2 = scale_log2;                                                 \
    set_blocking(&a, num_rows);                                                \
    if (fa_block_##suffix(&a, row_start, num_rows, lse) != 0)                  \
      fail("out of memory");                                                   \
  }

FA_ENTRY(bf16)
FA_ENTRY(f16)
