// Flash-MLA decode forward for Arm SME (bf16 inputs, fp32 accumulation).
//
// Multi-head latent attention (MLA) decode step with paged KV, one program =
// num_heads query heads of one batch for a single query position (s_q == 1),
// as used by the FlagGems flash_mla kernel.  Per chunk of page_size keys
// (gathered through the batch's page table) it computes
//
//   S   = Q [K_ckv K_pe]^T * scale        one GEMM over the full head_dim d
//   online softmax (the Triton kernel uses natural exp; exp2 internally with
//   scale_log2 = scale * log2(e), which is algebraically identical)
//   acc += P V_ckv                         the first dv elements of a key row
//   o = acc / sum(P)
//
// Both GEMMs run as widening outer products (BFMOPA za32) into the four
// 32-bit ZA tiles; softmax runs as streaming SVE in the same streaming
// region.  Requires SVL = 512 bit (16 fp32 lanes); the entry point aborts
// with a message on any other vector length.
//
// Part of libkunpeng_llm.so.  The triton-shared compiler replaces a
// recognised Triton flash-MLA kernel by one call per program to
// kunpeng_flash_mla_bf16 (see kunpeng_llm.h); the Triton launcher provides
// the parallelism, one program per batch (every head of the batch is in the
// same program, so the paged keys are gathered once per batch).
//
// Keys are processed in chunks of MLA_CHUNK_KEYS (one online-softmax step:
// QK^T, softmax, PV); the QK product stages two 16-key tiles at a time into a
// small L1-resident pair-packed buffer and prefetches the next block into L2,
// and the PV product builds its FMOPA operand directly from the kv rows
// (two rows zipped), so no packed K/V copy of the cache is made.
//
// Data layouts (L = 16 = fp32 lanes per vector, "pair" = two adjacent 16-bit
// elements along the contraction dimension, viewed as one uint32):
//   Qp [rt][p][L]      u32  pair p of query heads rt*L .. rt*L+15
//   Kst [slot][p][L]   u32  pair p of the 16 keys of staged tile `slot`
//   ST [j][BH]         f32  scores, transposed: lane = query head
//   Pp [rt][kp][2L]    u16  probabilities, key pairs interleaved
//   OT [dtv][BH]       f32  output accumulator, transposed
// The transposed ST/OT layouts make the softmax row statistics plain
// lane-wise vector ops (no horizontal reductions).

#include <arm_sme.h>
#include <arm_sve.h>
#include <math.h>
#include "kunpeng_llm.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define L 16
#define MAX_D 1024
#define MAX_BH 128
#define MAX_PAGE 512

// One (batch, head block): the pointers address row 0 of the head block.
typedef struct {
  const uint16_t *q, *kv;
  const int32_t *bt; // page table of this batch
  uint16_t *o;
  // head strides, in elements; kv rows (one per cached token) are kv_rs
  // elements apart, head_dim contiguous
  int64_t q_hs, kv_rs, o_hs;
  int32_t seqlen;
  int32_t BH;  // query heads per program, multiple of L, <= MAX_BH
  int32_t page;  // keys per page == softmax chunk granularity, multiple of L
  int32_t D, DV; // qk contraction dim, output/value dim
  float scale_log2; // softmax_scale * log2(e)
} mla_args;

// kunpeng_llm_sme_vector_bytes and kunpeng_llm_begin_launch (the launch
// epoch) are library-wide; they live in flash_attn_fwd_sme.c.

// Per-thread scratch, kept across calls so steady-state calls do not fault in
// fresh pages.  Size depends only on the program's shape (D, DV, BH, chunk),
// never on the data, so the buffers sit in the same place on every call.
//
// There is no cross-call K/V cache: keys are staged per QK block into Kst,
// which is small enough to stay in L1 while the block is consumed, and the
// PV product reads the value columns of the kv rows directly (see
// mla_pv_gemm).  One program covers every head of its batch, so a packed tile
// would never be reused by another program anyway; staging it into a large
// buffer would only add an L2/L3 round trip.
typedef struct {
  uint32_t *Kst; // [MLA_STAGE_TILES][npair][L] pair-packed key tile staging
  uint32_t *Qp;  // [nrt][npair][L] pair-packed query heads
  uint16_t *Pp;  // [nrt][chunk keys / 2][2L] probabilities, key pairs zipped
  float *ST;     // [chunk keys][BH] scores, transposed: lane = query head
  float *OT;     // [DV][BH] output accumulator, transposed
  float *m, *l, *inv;
  size_t bytes;
  void *mem;
} mla_scratch;

static __thread mla_scratch tls_scratch;

// Keys per QK block: two 16-key tiles.  Their staged form is 2 * 16 * D * 2
// bytes (36 KiB for D = 576), which stays L1-resident across the head tiles
// of the block; the block loop prefetches one block ahead into L2.
#define MLA_STAGE_TILES 2

// Online-softmax chunk in keys (rounded down to whole pages): large enough to
// amortise the output rescale, small enough that the score buffer stays
// L2-resident.
#define MLA_CHUNK_KEYS 512

static mla_scratch *get_scratch(const mla_args *a) {
  mla_scratch *s = &tls_scratch;
  const int64_t BH = a->BH, D = a->D, DV = a->DV, page = a->page;
  const int64_t ppc = MLA_CHUNK_KEYS / page < 1 ? 1 : MLA_CHUNK_KEYS / page;
  const int64_t nk_max = ppc * page;
  size_t sz[8] = {
      MLA_STAGE_TILES * (D / 2) * L * 4, // Kst
      BH * D * 2,                        // Qp
      BH * nk_max * 2,                   // Pp
      nk_max * BH * 4,                   // ST
      DV * BH * 4,                       // OT
      BH * 4,                            // m
      2 * BH * 4,                        // l + inv
      0,
  };
  size_t total = 0;
  for (int i = 0; i < 8; i++)
    total += (sz[i] + 255) & ~(size_t)255;
  if (total > s->bytes) {
    free(s->mem);
    s->mem = aligned_alloc(256, total);
    s->bytes = s->mem ? total : 0;
    if (!s->mem)
      return NULL;
  }
  char *p = (char *)s->mem;
  void *ptr[8];
  for (int i = 0; i < 8; i++) {
    ptr[i] = p;
    p += (sz[i] + 255) & ~(size_t)255;
  }
  s->Kst = ptr[0];
  s->Qp = ptr[1];
  s->Pp = ptr[2];
  s->ST = ptr[3];
  s->OT = ptr[4];
  s->m = ptr[5];
  s->l = ptr[6];
  s->inv = s->l + BH;
  return s;
}

// The streaming-mode helpers take mla_args / mla_scratch by value so that,
// once inlined, the fields stay in registers instead of being reloaded after
// every vector store.

// ---------------------------------------------------------------------------
// Packing
// ---------------------------------------------------------------------------

#define MLA_LD_ROWS(t, u)                                                      \
  do {                                                                         \
    svbool_t pw = svwhilelt_b32((u), npair);                                   \
    for (int r = 0; r < L; r++)                                                \
      svld1_hor_za32(t, r, (rowok >> r) & 1 ? pw : svpfalse(), src[r] + (u));  \
  } while (0)
#define MLA_ST_COLS(t, u)                                                      \
  do {                                                                         \
    int nu = npair - (u) < L ? npair - (u) : L;                                \
    for (int p = 0; p < nu; p++)                                               \
      svst1_ver_za32(t, p, svptrue_b32(), dst + (int64_t)((u) + p) * L);       \
  } while (0)

// Transpose L rows of `npair` u32 pairs into dst[p][L], rotating through the
// four ZA tiles so the loads of chunk i overlap the stores of chunk i - 1.
// Rows at or after nrows_valid are packed as zeros.  Clobbers all of ZA.
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
    MLA_LD_ROWS(0, u);
    if (u > 0)
      MLA_ST_COLS(3, u - L);
    if (u + L >= npair) {
      MLA_ST_COLS(0, u);
      break;
    }
    MLA_LD_ROWS(1, u + L);
    MLA_ST_COLS(0, u);
    if (u + 2 * L >= npair) {
      MLA_ST_COLS(1, u + L);
      break;
    }
    MLA_LD_ROWS(2, u + 2 * L);
    MLA_ST_COLS(1, u + L);
    if (u + 3 * L >= npair) {
      MLA_ST_COLS(2, u + 2 * L);
      break;
    }
    MLA_LD_ROWS(3, u + 3 * L);
    MLA_ST_COLS(2, u + 2 * L);
    if (u + 4 * L >= npair) {
      MLA_ST_COLS(3, u + 3 * L);
      break;
    }
  }
}
#undef MLA_LD_ROWS
#undef MLA_ST_COLS

// Fast path of pack_rows_u32: all L rows valid and consecutive (a whole
// 16-key kv tile).  The rows are walked with a single pointer, so the loop
// needs one live address instead of L and nothing spills.
static inline void pack_tile_u32(uint32_t *dst, const uint16_t *base,
                                 int64_t row_stride, int npair)
    __arm_streaming __arm_inout("za") {
  const svbool_t pt = svptrue_b32();
  const int64_t rs32 = row_stride >> 1;
  for (int u = 0; u < npair; u += L) {
    const int nu = npair - u < L ? npair - u : L;
    const svbool_t pw = nu == L ? pt : svwhilelt_b32(u, npair);
    const uint32_t *s = (const uint32_t *)base + u;
    for (int r = 0; r < L; r++) {
      svld1_hor_za32(0, r, pw, s);
      s += rs32;
    }
    uint32_t *d = dst + (int64_t)u * L;
    if (nu == L) {
      for (int p = 0; p < L; p++)
        svst1_ver_za32(0, p, pt, d + p * L);
    } else {
      for (int p = 0; p < nu; p++)
        svst1_ver_za32(0, p, pt, d + p * L);
    }
  }
}

// ---------------------------------------------------------------------------
// exp2 for fp32 vectors.  x <= -127 (including -inf) gives 2^-127 rather than
// 0.  Masked scores land there and contribute ~1e-38 relative weight; their
// key rows are skipped by the PV product and never enter the accumulator.
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

// Pull the rows of key tiles [t0, t0 + ntiles) into L2 ahead of the block that
// will stage them.  Rows of one page are contiguous, so this walks whole
// pages; both the K and the V columns of a row are touched (the PV product
// reads the value columns later).
static inline void mla_prefetch_block(const mla_args a, int64_t t0, int ntiles)
    __arm_streaming {
  const int tpp = a.page / L;
  for (int i = 0; i < ntiles; i++) {
    const int64_t t = t0 + i;
    const int64_t pg = a.bt[t / tpp];
    const int64_t off = t - (t / tpp) * tpp;
    const uint16_t *rows = a.kv + (pg * tpp + off) * L * a.kv_rs;
    for (int r = 0; r < L; r++)
      for (int o = 0; o < a.D * 2; o += 128)
        __builtin_prefetch((const char *)(rows + r * a.kv_rs) + o, 0, 1);
  }
}

#define MLA_MOPA(t, a, b)                                                      \
  svmopa_za32_bf16_m(t, svptrue_b16(), svptrue_b16(),                          \
                     svreinterpret_bf16_u16(a), svreinterpret_bf16_u16(b))
#define MLA_CVT_PAIR(pg, lo, hi)                                               \
  svreinterpret_u16_bf16(svcvtnt_bf16_f32_x(svcvt_bf16_f32_x(pg, lo), pg, hi))

// Stage keys [t0, t0 + nct) (nct <= MLA_STAGE_TILES) into the pair-packed
// staging buffer.  Tile i holds keys (t0 + i) * 16 .. +15, gathered from page
// bt[(t0 + i) / (page/16)]; rows at or after seqlen are packed as zeros (their
// scores are masked in the softmax).
static inline void mla_stage(const mla_args a, const mla_scratch s, int64_t t0,
                            int nct) __arm_streaming __arm_inout("za") {
  const int tpp = a.page / L;
  const int npair = a.D / 2;
  for (int64_t i = 0; i < nct; i++) {
    const int64_t t = t0 + i;
    const int64_t pg = a.bt[t / tpp];
    const int64_t off = t - (t / tpp) * tpp;
    const int nvalid = a.seqlen - t * L < L ? (int)(a.seqlen - t * L) : L;
    const uint16_t *rows = a.kv + (pg * tpp + off) * L * a.kv_rs;
    if (nvalid == L)
      pack_tile_u32(s.Kst + i * (int64_t)npair * L, rows, a.kv_rs, npair);
    else
      pack_rows_u32(s.Kst + i * (int64_t)npair * L, rows, a.kv_rs, 0, nvalid,
                    npair);
  }
}

// S^T[j][BH] = (Q K^T)^T for nrt head tiles x nct key tiles (staged in Kst,
// key tile i at slot i), 2x2 ZA tiles at a time.  Key tiles stay outermost so
// each staged K pair is loaded once per two head tiles (L1-resident) instead
// of once per head tile.
static inline void mla_s_gemm(const mla_scratch s, int D, int nct, float *ST,
                              int nrt, int BH) __arm_streaming
    __arm_inout("za") {
  const svbool_t pt = svptrue_b32(), ph = svptrue_b16();
  const int npair = D / 2;
  const int64_t tile_stride = (int64_t)npair * 2 * L; // in u16
  for (int ct = 0; ct < nct; ct += 2) {
    const int two_c = ct + 1 < nct;
    const uint16_t *b0 = (const uint16_t *)s.Kst + ct * tile_stride;
    const uint16_t *b1 = b0 + tile_stride;
    for (int rt = 0; rt < nrt; rt += 2) {
      const int two_r = rt + 1 < nrt;
      const uint16_t *a0 = (const uint16_t *)s.Qp + rt * tile_stride;
      const uint16_t *a1 = a0 + tile_stride;
      svzero_za();
      if (two_r && two_c) {
        for (int p = 0; p < npair; p++) {
          svuint16_t va0 = svld1_u16(ph, a0 + p * 2 * L);
          svuint16_t va1 = svld1_u16(ph, a1 + p * 2 * L);
          svuint16_t vb0 = svld1_u16(ph, b0 + p * 2 * L);
          svuint16_t vb1 = svld1_u16(ph, b1 + p * 2 * L);
          MLA_MOPA(0, va0, vb0);
          MLA_MOPA(1, va0, vb1);
          MLA_MOPA(2, va1, vb0);
          MLA_MOPA(3, va1, vb1);
        }
      } else if (two_r) {
        for (int p = 0; p < npair; p++) {
          svuint16_t va0 = svld1_u16(ph, a0 + p * 2 * L);
          svuint16_t va1 = svld1_u16(ph, a1 + p * 2 * L);
          svuint16_t vb0 = svld1_u16(ph, b0 + p * 2 * L);
          MLA_MOPA(0, va0, vb0);
          MLA_MOPA(2, va1, vb0);
        }
      } else if (two_c) {
        for (int p = 0; p < npair; p++) {
          svuint16_t va0 = svld1_u16(ph, a0 + p * 2 * L);
          svuint16_t vb0 = svld1_u16(ph, b0 + p * 2 * L);
          svuint16_t vb1 = svld1_u16(ph, b1 + p * 2 * L);
          MLA_MOPA(0, va0, vb0);
          MLA_MOPA(1, va0, vb1);
        }
      } else {
        for (int p = 0; p < npair; p++) {
          svuint16_t va0 = svld1_u16(ph, a0 + p * 2 * L);
          svuint16_t vb0 = svld1_u16(ph, b0 + p * 2 * L);
          MLA_MOPA(0, va0, vb0);
        }
      }
      // Vertical slice j of a tile holds the scores of key j for 16 heads.
      float *s00 = ST + (int64_t)(ct * L) * BH + rt * L;
      for (int j = 0; j < L; j++)
        svst1_ver_za32(0, j, pt, s00 + (int64_t)j * BH);
      if (two_c)
        for (int j = 0; j < L; j++)
          svst1_ver_za32(1, j, pt, s00 + (int64_t)(L + j) * BH);
      if (two_r)
        for (int j = 0; j < L; j++)
          svst1_ver_za32(2, j, pt, s00 + (int64_t)j * BH + L);
      if (two_r && two_c)
        for (int j = 0; j < L; j++)
          svst1_ver_za32(3, j, pt, s00 + (int64_t)(L + j) * BH + L);
    }
  }
}

// Online-softmax update for keys [c, c + nk): masks keys at or after c +
// nvis (the tail page), writes the packed probabilities Pp, updates the
// running max (scaled, log2 domain) and sum, and rescales OT.
static inline void mla_softmax(const mla_args a, float *ST, uint16_t *Pp,
                               float *OT, float *mrow, float *lrow, int nrt,
                               int64_t c, int nk, int nvis,
                               int first, int BH) __arm_streaming {
  const svbool_t pt = svptrue_b32();
  const svfloat32_t ninf = svdup_f32(-INFINITY);
  const svfloat32_t scale = svdup_f32(a.scale_log2);
  const int nkp = nk / 2;
  for (int rt = 0; rt < nrt; rt++) {
    float *st = ST + rt * L;
    svfloat32_t mx0 = ninf, mx1 = ninf;
    // Keys [0, nvisfull) are visible to all heads; the boundary key may be
    // visible for some lanes only when nvis is not a multiple of the key
    // tiling, which for whole chunks it never is (nvis == nk), so the masked
    // path only runs on the tail chunk.
    int j = 0;
    if (nvis >= nk) {
      svfloat32_t mx2 = ninf, mx3 = ninf;
      for (; j + 3 < nk; j += 4) {
        mx0 = svmax_f32_x(pt, mx0, svld1_f32(pt, st + (int64_t)j * BH));
        mx1 = svmax_f32_x(pt, mx1, svld1_f32(pt, st + (int64_t)(j + 1) * BH));
        mx2 = svmax_f32_x(pt, mx2, svld1_f32(pt, st + (int64_t)(j + 2) * BH));
        mx3 = svmax_f32_x(pt, mx3, svld1_f32(pt, st + (int64_t)(j + 3) * BH));
      }
      for (; j < nk; j++)
        mx0 = svmax_f32_x(pt, mx0, svld1_f32(pt, st + (int64_t)j * BH));
      mx0 = svmax_f32_x(pt, mx0, mx2);
      mx1 = svmax_f32_x(pt, mx1, mx3);
    } else {
      for (; j < nvis; j++) {
        svfloat32_t v = svld1_f32(pt, st + (int64_t)j * BH);
        svst1_f32(pt, st + (int64_t)j * BH, v);
        mx0 = svmax_f32_x(pt, mx0, v);
      }
      for (; j < nk; j++) {
        svst1_f32(pt, st + (int64_t)j * BH, ninf);
      }
    }
    svfloat32_t m_old = svld1_f32(pt, mrow + rt * L);
    svfloat32_t mx = svmax_f32_x(
        pt, m_old, svmul_f32_x(pt, svmax_f32_x(pt, mx0, mx1), scale));
    svfloat32_t msafe =
        svsel_f32(svcmpeq_f32(pt, mx, ninf), svdup_f32(0.0f), mx);
    svfloat32_t nmsafe = svneg_f32_x(pt, msafe);
    svfloat32_t alpha = exp2v(pt, svsub_f32_x(pt, m_old, msafe));
    // Four key pairs per iteration: eight independent exp2 chains.
    svfloat32_t l0 = svmul_f32_x(pt, svld1_f32(pt, lrow + rt * L), alpha);
    svfloat32_t l1 = svdup_f32(0.0f), l2 = svdup_f32(0.0f),
                l3 = svdup_f32(0.0f);
    uint16_t *pp = Pp + (int64_t)rt * nkp * 2 * L;
    for (int kp = 0; kp < nkp; kp += 4) {
      const float *sk = st + (int64_t)(2 * kp) * BH;
      svfloat32_t p0 =
          exp2v(pt, svmad_f32_x(pt, svld1_f32(pt, sk), scale, nmsafe));
      svfloat32_t p1 =
          exp2v(pt, svmad_f32_x(pt, svld1_f32(pt, sk + BH), scale, nmsafe));
      svfloat32_t p2 =
          exp2v(pt, svmad_f32_x(pt, svld1_f32(pt, sk + 2 * BH), scale, nmsafe));
      svfloat32_t p3 =
          exp2v(pt, svmad_f32_x(pt, svld1_f32(pt, sk + 3 * BH), scale, nmsafe));
      svfloat32_t p4 =
          exp2v(pt, svmad_f32_x(pt, svld1_f32(pt, sk + 4 * BH), scale, nmsafe));
      svfloat32_t p5 =
          exp2v(pt, svmad_f32_x(pt, svld1_f32(pt, sk + 5 * BH), scale, nmsafe));
      svfloat32_t p6 =
          exp2v(pt, svmad_f32_x(pt, svld1_f32(pt, sk + 6 * BH), scale, nmsafe));
      svfloat32_t p7 =
          exp2v(pt, svmad_f32_x(pt, svld1_f32(pt, sk + 7 * BH), scale, nmsafe));
      l0 = svadd_f32_x(pt, l0, svadd_f32_x(pt, p0, p1));
      l1 = svadd_f32_x(pt, l1, svadd_f32_x(pt, p2, p3));
      l2 = svadd_f32_x(pt, l2, svadd_f32_x(pt, p4, p5));
      l3 = svadd_f32_x(pt, l3, svadd_f32_x(pt, p6, p7));
      uint16_t *pk = pp + (int64_t)kp * 2 * L;
      svst1_u16(svptrue_b16(), pk, MLA_CVT_PAIR(pt, p0, p1));
      svst1_u16(svptrue_b16(), pk + 2 * L, MLA_CVT_PAIR(pt, p2, p3));
      svst1_u16(svptrue_b16(), pk + 4 * L, MLA_CVT_PAIR(pt, p4, p5));
      svst1_u16(svptrue_b16(), pk + 6 * L, MLA_CVT_PAIR(pt, p6, p7));
    }
    l0 = svadd_f32_x(pt, svadd_f32_x(pt, l0, l1), svadd_f32_x(pt, l2, l3));
    svst1_f32(pt, mrow + rt * L, mx);
    svst1_f32(pt, lrow + rt * L, l0);
    if (!first) {
      float *ot = OT + rt * L;
      for (int d = 0; d < a.DV; d++)
        svst1_f32(pt, ot + (int64_t)d * BH,
                  svmul_f32_x(pt, svld1_f32(pt, ot + (int64_t)d * BH), alpha));
    }
  }
}

// 1 / row sum, or 0 for heads that saw no key (running max still -inf).
static inline void mla_row_inv(const float *mrow, const float *lrow,
                               float *inv, int nrt) __arm_streaming {
  const svbool_t pt = svptrue_b32();
  for (int rt = 0; rt < nrt; rt++) {
    svfloat32_t m = svld1_f32(pt, mrow + rt * L);
    svfloat32_t l = svld1_f32(pt, lrow + rt * L);
    svbool_t ok = svand_b_z(pt, svcmpgt_n_f32(pt, l, 0.0f),
                            svcmpne_n_f32(pt, m, -INFINITY));
    svst1_f32(pt, inv + rt * L,
              svsel_f32(ok, svdivr_n_f32_x(pt, l, 1.0f), svdup_f32(0.0f)));
  }
}

// Heads of ZA tile `t` scaled by inv[] -> output heads r .. r + nrow.
#define MLA_WRITE_HEAD(t, i, inv_v, out_p, o_hs)                               \
  do {                                                                         \
    svfloat32_t x = svread_hor_za32_f32_m(svdup_f32(0.0f), pt, t, i);          \
    x = svmul_f32_x(pt, x, svdup_lane_f32(inv_v, i));                          \
    svuint16_t hv = MLA_CVT_PAIR(pt, x, x); /* value in even lanes */          \
    svst1_u16(ph16, (out_p) + (i) * (o_hs), svuzp1_u16(hv, hv));               \
  } while (0)
#define MLA_WRITE_TILE(t, inv_v, out_p, nrow, o_hs)                            \
  do {                                                                         \
    int i = 0;                                                                 \
    for (; i + 4 <= (nrow); i += 4) {                                          \
      svfloat32_t x0 = svread_hor_za32_f32_m(svdup_f32(0.0f), pt, t, i);       \
      svfloat32_t x1 = svread_hor_za32_f32_m(svdup_f32(0.0f), pt, t, i + 1);   \
      svfloat32_t x2 = svread_hor_za32_f32_m(svdup_f32(0.0f), pt, t, i + 2);   \
      svfloat32_t x3 = svread_hor_za32_f32_m(svdup_f32(0.0f), pt, t, i + 3);   \
      x0 = svmul_f32_x(pt, x0, svdup_lane_f32(inv_v, i));                      \
      x1 = svmul_f32_x(pt, x1, svdup_lane_f32(inv_v, i + 1));                  \
      x2 = svmul_f32_x(pt, x2, svdup_lane_f32(inv_v, i + 2));                  \
      x3 = svmul_f32_x(pt, x3, svdup_lane_f32(inv_v, i + 3));                  \
      svuint16_t h01 = MLA_CVT_PAIR(pt, x0, x1);                               \
      svuint16_t h23 = MLA_CVT_PAIR(pt, x2, x3);                               \
      /* even lanes = head i (i + 2), odd lanes = head i + 1 (i + 3) */        \
      svst1_u16(ph16, (out_p) + i * (o_hs), svuzp1_u16(h01, h01));             \
      svst1_u16(ph16, (out_p) + (i + 1) * (o_hs), svuzp2_u16(h01, h01));       \
      svst1_u16(ph16, (out_p) + (i + 2) * (o_hs), svuzp1_u16(h23, h23));       \
      svst1_u16(ph16, (out_p) + (i + 3) * (o_hs), svuzp2_u16(h23, h23));       \
    }                                                                          \
    for (; i < (nrow); i++)                                                    \
      MLA_WRITE_HEAD(t, i, inv_v, out_p, o_hs);                                \
  } while (0)

// One key pair of the PV product: rows (2kp, 2kp + 1) of the value columns,
// zipped into the two key-pair FMOPA operands.  Reference uses vp, kp, pa0,
// pa1, pcol, ph, two_r, two_c from the enclosing scope.
#define MLA_PV_PAIR(P0, P1)                                                    \
  do {                                                                         \
    svuint16_t vr0 = svld1_u16((P0), vp + (int64_t)(2 * kp) * a.kv_rs);         \
    svuint16_t vr1 = svld1_u16((P1), vp + (int64_t)(2 * kp + 1) * a.kv_rs);     \
    svuint16_t vb0 = svzip1_u16(vr0, vr1);                                     \
    svuint16_t vb1 = svzip2_u16(vr0, vr1);                                     \
    svuint16_t va0 = svld1_u16(ph, pa0 + kp * 2 * L);                          \
    MLA_MOPA(0, va0, vb0);                                                     \
    if (two_c)                                                                 \
      MLA_MOPA(1, va0, vb1);                                                   \
    if (two_r) {                                                               \
      svuint16_t va1 = svld1_u16(ph, pa1 + kp * 2 * L);                        \
      MLA_MOPA(2, va1, vb0);                                                   \
      if (two_c)                                                               \
        MLA_MOPA(3, va1, vb1);                                                 \
    }                                                                          \
  } while (0)

// OT[dtv][BH] += (P V)^T over key tiles [t0, t0 + nkt), 2x2 ZA tiles at a
// time.  Value-dim tiles stay outermost so the value columns of a kv row are
// touched once per head tile pair and stay in L1 across the head tiles; the
// operand is built straight from the kv rows: two consecutive rows are loaded
// (32 value columns each) and zipped into the key-pair form FMOPA wants, so
// no packed V copy is needed.  On the last chunk the result is normalised and
// written straight from ZA to the output heads.
static inline void mla_pv_gemm(const mla_args a, const mla_scratch s,
                               int64_t t0, int nkt, int nrt, int first,
                               int last, int BH) __arm_streaming
    __arm_inout("za") {
  const svbool_t pt = svptrue_b32(), ph = svptrue_b16();
  const svbool_t ph16 = svwhilelt_b16(0, L);
  const int DV = a.DV, ndt = DV / L, tpp = a.page / L;
  const int64_t ptile = (int64_t)nkt * L * L;   // P of one head tile, in u16
  float *OT = s.OT;
  for (int dt = 0; dt < ndt; dt += 2) {
    const int two_c = dt + 1 < ndt;
    // Upper half of the load is unused when DV is an odd multiple of L.
    const svbool_t pcol = two_c ? ph : svwhilelt_b16(0, DV - dt * L);
    for (int rt = 0; rt < nrt; rt += 2) {
      const int two_r = rt + 1 < nrt;
      const uint16_t *a0 = s.Pp + rt * ptile;
      const uint16_t *a1 = a0 + ptile;
      float *o00 = OT + (int64_t)(dt * L) * BH + rt * L;
      if (first) {
        svzero_za();
      } else {
        for (int j = 0; j < L; j++)
          svld1_ver_za32(0, j, pt, o00 + (int64_t)j * BH);
        if (two_c)
          for (int j = 0; j < L; j++)
            svld1_ver_za32(1, j, pt, o00 + (int64_t)(L + j) * BH);
        if (two_r)
          for (int j = 0; j < L; j++)
            svld1_ver_za32(2, j, pt, o00 + (int64_t)j * BH + L);
        if (two_r && two_c)
          for (int j = 0; j < L; j++)
            svld1_ver_za32(3, j, pt, o00 + (int64_t)(L + j) * BH + L);
      }
      for (int kt = 0; kt < nkt; kt++) {
        const int64_t t = t0 + kt;
        const int64_t pg = a.bt[t / tpp];
        const int64_t off = t - (t / tpp) * tpp;
        const uint16_t *vp =
            a.kv + (pg * tpp + off) * L * a.kv_rs + dt * L;
        const uint16_t *pa0 = a0 + kt * L * L;
        const uint16_t *pa1 = a1 + kt * L * L;
        // Keys at or after seqlen have (near-)zero probabilities, so their
        // rows are simply not accumulated; the odd tail row of a masked tile
        // is loaded under a false predicate instead (it may hold stale data).
        const int nvalid = a.seqlen - t * L < L ? (int)(a.seqlen - t * L) : L;
        const int nkp = nvalid >> 1;
        int kp = 0;
        for (; kp < nkp; kp++)
          MLA_PV_PAIR(pcol, pcol);
        if (nvalid & 1)
          MLA_PV_PAIR(pcol, svpfalse());
      }
      if (last) {
        const int64_t o_hs = a.o_hs;
        uint16_t *out0 = a.o + rt * L * o_hs + dt * L;
        const svfloat32_t inv0 = svld1_f32(pt, s.inv + rt * L);
        const svfloat32_t inv1 = svld1_f32(pt, s.inv + rt * L + L);
        MLA_WRITE_TILE(0, inv0, out0, L, o_hs);
        if (two_c)
          MLA_WRITE_TILE(1, inv0, out0 + L, L, o_hs);
        if (two_r)
          MLA_WRITE_TILE(2, inv1, out0 + L * o_hs, L, o_hs);
        if (two_r && two_c)
          MLA_WRITE_TILE(3, inv1, out0 + L * o_hs + L, L, o_hs);
      } else {
        for (int j = 0; j < L; j++)
          svst1_ver_za32(0, j, pt, o00 + (int64_t)j * BH);
        if (two_c)
          for (int j = 0; j < L; j++)
            svst1_ver_za32(1, j, pt, o00 + (int64_t)(L + j) * BH);
        if (two_r)
          for (int j = 0; j < L; j++)
            svst1_ver_za32(2, j, pt, o00 + (int64_t)j * BH + L);
        if (two_r && two_c)
          for (int j = 0; j < L; j++)
            svst1_ver_za32(3, j, pt, o00 + (int64_t)(L + j) * BH + L);
      }
    }
  }
}

// Q K^T for the nct key tiles of a chunk starting at key tile t0, staged and
// consumed in blocks of MLA_STAGE_TILES tiles, prefetching one block ahead.
static inline void mla_qk_chunk(const mla_args a, const mla_scratch s,
                                int64_t t0, int nct, int nrt, int BH)
    __arm_streaming __arm_inout("za") {
  for (int i = 0; i < nct; i += MLA_STAGE_TILES) {
    const int ncs = nct - i < MLA_STAGE_TILES ? nct - i : MLA_STAGE_TILES;
    mla_stage(a, s, t0 + i, ncs);
    if (i + MLA_STAGE_TILES < nct)
      mla_prefetch_block(a, t0 + i + MLA_STAGE_TILES,
                         nct - i - MLA_STAGE_TILES < MLA_STAGE_TILES
                             ? nct - i - MLA_STAGE_TILES
                             : MLA_STAGE_TILES);
    mla_s_gemm(s, a.D, ncs, s.ST + (int64_t)(i * L) * BH, nrt, BH);
  }
}

// The whole head block of one batch: s_q == 1, so there is exactly one block
// of BH query heads.
__arm_locally_streaming __arm_new("za") static void
mla_task(const mla_args *ap, const mla_scratch *sp) {
  const mla_args a = *ap;
  mla_scratch s = *sp;
  const int BH = a.BH, D = a.D;
  const int npair = D / 2, nrt = BH / L, tpp = a.page / L;

  for (int rt = 0; rt < nrt; rt++) {
    uint32_t *qp = s.Qp + (int64_t)rt * npair * L;
    if (rt * L + L <= BH)
      pack_tile_u32(qp, a.q + (int64_t)(rt * L) * a.q_hs, a.q_hs, npair);
    else
      pack_rows_u32(qp, a.q, a.q_hs, rt * L, BH, npair);
    svst1_f32(svptrue_b32(), s.m + rt * L, svdup_f32(-INFINITY));
    svst1_f32(svptrue_b32(), s.l + rt * L, svdup_f32(0.0f));
  }

  const int64_t npage = a.seqlen / a.page; // full pages
  const int tail = (int)(a.seqlen - npage * a.page);
  // Chunks are whole pages (the page table indirection makes key ranges that
  // straddle pages expensive); pages per chunk amortises the OT save/restore
  // and the softmax pass over more keys.
  const int ppc = MLA_CHUNK_KEYS / a.page < 1 ? 1 : MLA_CHUNK_KEYS / a.page;

  int64_t t = 0;
  int first = 1;
  while (t < npage * tpp) {
    int nct = npage * tpp - t < (int64_t)ppc * tpp ? (int)(npage * tpp - t)
                                                   : ppc * tpp;
    const int last = !tail && t + nct >= npage * tpp;
    mla_qk_chunk(a, s, t, nct, nrt, BH);
    mla_softmax(a, s.ST, s.Pp, s.OT, s.m, s.l, nrt, t * L, nct * L, nct * L,
                first, BH);
    if (last)
      mla_row_inv(s.m, s.l, s.inv, nrt);
    mla_pv_gemm(a, s, t, nct, nrt, first, last, BH);
    first = 0;
    t += nct;
  }
  if (tail) {
    const int nct = (tail + L - 1) / L;
    mla_qk_chunk(a, s, t, nct, nrt, BH);
    mla_softmax(a, s.ST, s.Pp, s.OT, s.m, s.l, nrt, t * L, nct * L, tail, 0,
                BH);
    mla_row_inv(s.m, s.l, s.inv, nrt);
    mla_pv_gemm(a, s, t, nct, nrt, first, 1, BH);
  }
}

#undef MLA_MOPA
#undef MLA_CVT_PAIR
#undef MLA_WRITE_HEAD
#undef MLA_WRITE_TILE

static void fail(const char *what) {
  fprintf(stderr, "libkunpeng_llm: kunpeng_flash_mla_bf16: %s\n", what);
  abort();
}

// Query heads [0, num_heads) of one batch (the head block offset is already
// folded into q / o by the caller); seqlen keys gathered through block_table.
void kunpeng_flash_mla_bf16(const void *q, int64_t q_hs, const void *kv,
                            int64_t kv_rs, const int32_t *block_table,
                            int64_t seqlen, void *o, int64_t o_hs,
                            int64_t num_heads, int64_t page_size, int64_t dv,
                            int64_t d, float scale) {
  if (num_heads % L != 0 || num_heads <= 0 || num_heads > MAX_BH)
    fail("unsupported num_heads");
  if (page_size % L != 0 || page_size <= 0 || page_size > MAX_PAGE)
    fail("unsupported page_size");
  if (d % L != 0 || dv % L != 0 || dv <= 0 || dv > d || d > MAX_D)
    fail("unsupported head_dim");
  if (seqlen < 0)
    fail("negative seqlen");
  if (svcntsb() != L * 4)
    fail("needs 512-bit SME vectors "
         "(/proc/sys/abi/sme_default_vector_length = 64)");

  mla_args a;
  a.q = (const uint16_t *)q;
  a.kv = (const uint16_t *)kv;
  a.bt = block_table;
  a.o = (uint16_t *)o;
  a.q_hs = q_hs;
  a.kv_rs = kv_rs;
  a.o_hs = o_hs;
  a.seqlen = (int32_t)seqlen;
  a.BH = (int32_t)num_heads;
  a.page = (int32_t)page_size;
  a.D = (int32_t)d;
  a.DV = (int32_t)dv;
  a.scale_log2 = scale * 1.4426950408889634f;
  if (seqlen == 0) {
    for (int64_t h = 0; h < num_heads; h++)
      memset((char *)o + h * o_hs * 2, 0, (size_t)dv * 2);
    return;
  }
  mla_scratch *s = get_scratch(&a);
  if (!s)
    fail("out of memory");
  mla_task(&a, s);
}
