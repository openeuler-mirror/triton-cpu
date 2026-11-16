// Sparse MLA, one query and 64 heads per Triton program. BF16 inputs/outputs,
// FP32 accumulation. Packing and the 2x2 SME outer-product scheme follow the
// Kunpeng flash-attention library; scratch belongs to each calling thread.
#include "kunpeng_llm.h"
#include <algorithm>
#include <arm_sme.h>
#include <arm_sve.h>
#include <cmath>
#include <cstring>
#include <memory>
#include <new>

__attribute__((always_inline)) static inline uint16_t
bf16(float f) __arm_streaming_compatible {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (u + 0x7fff + ((u >> 16) & 1)) >> 16;
}

// Both operands contain 32-row tiles, each reduction pair followed by its
// 32 adjacent row pairs. Four ZA tiles compute a 32x32 output block.
__arm_locally_streaming __arm_new("za") static void multiply(const uint16_t *a,
                                                             const uint16_t *b,
                                                             float *c, int m,
                                                             int n, int k,
                                                             int b_stride = 0) {
  if (!b_stride)
    b_stride = k;
  auto ph = svptrue_b16();
  auto pf = svptrue_b32();
  for (int i = 0; i < m; i += 32)
    for (int j = 0; j < n; j += 32) {
      svzero_za();
      const uint16_t *ap = a + i * k, *bp = b + j * b_stride;
      for (int p = 0; p < k / 2; ++p) {
        auto a0 = svreinterpret_bf16_u16(svld1_u16(ph, ap + p * 64));
        auto a1 = svreinterpret_bf16_u16(svld1_u16(ph, ap + p * 64 + 32));
        auto b0 = svreinterpret_bf16_u16(svld1_u16(ph, bp + p * 64));
        auto b1 = svreinterpret_bf16_u16(svld1_u16(ph, bp + p * 64 + 32));
        svmopa_za32_bf16_m(0, ph, ph, a0, b0);
        svmopa_za32_bf16_m(1, ph, ph, a0, b1);
        svmopa_za32_bf16_m(2, ph, ph, a1, b0);
        svmopa_za32_bf16_m(3, ph, ph, a1, b1);
      }
      for (int r = 0; r < 16; ++r) {
        svst1_hor_za32(0, r, pf, c + (i + r) * n + j);
        svst1_hor_za32(1, r, pf, c + (i + r) * n + j + 16);
        svst1_hor_za32(2, r, pf, c + (i + r + 16) * n + j);
        svst1_hor_za32(3, r, pf, c + (i + r + 16) * n + j + 16);
      }
    }
}

// PR #269's bounded polynomial exp2, used on nonpositive softmax arguments.
static inline svfloat32_t exp2v(svbool_t pg, svfloat32_t x) __arm_streaming {
  auto underflow = svcmple_n_f32(pg, x, -150.0f);
  x = svmax_n_f32_x(pg, x, -150.0f);
  svfloat32_t n = svrintn_f32_x(pg, x);
  svfloat32_t f = svsub_f32_x(pg, x, n);
  svfloat32_t r = svdup_f32(1.535336188319500e-4f);
  r = svmad_f32_x(pg, r, f, svdup_f32(1.339887440266574e-3f));
  r = svmad_f32_x(pg, r, f, svdup_f32(9.618437357674640e-3f));
  r = svmad_f32_x(pg, r, f, svdup_f32(5.550332471162809e-2f));
  r = svmad_f32_x(pg, r, f, svdup_f32(2.402264791363012e-1f));
  r = svmad_f32_x(pg, r, f, svdup_f32(6.931472028550421e-1f));
  r = svmad_f32_x(pg, r, f, svdup_f32(1.0f));
  auto result = svscale_f32_x(pg, r, svcvt_s32_f32_x(pg, n));
  return svsel_f32(underflow, svdup_f32(0), result);
}

// Guard results of FP32 arithmetic, rather than imposing a magnitude limit
// on Q, KV or scale. A rejected fast-path result is recomputed by Triton.
static inline bool finite_lanes(svbool_t pg, svfloat32_t value)
    __arm_streaming {
  auto bits = svreinterpret_u32_f32(value);
  auto exponent = svand_n_u32_x(pg, bits, 0x7f800000);
  return !svptest_any(pg, svcmpeq_n_u32(pg, exponent, 0x7f800000));
}

__arm_locally_streaming static bool finite_results(const float *values,
                                                   size_t count) {
  for (size_t i = 0; i < count; i += svcntw()) {
    auto pg = svwhilelt_b32(uint64_t(i), uint64_t(count));
    if (!finite_lanes(pg, svld1_f32(pg, values + i)))
      return false;
  }
  return true;
}

// Transpose 16 rows of 32-bit BF16 pairs through ZA, as in PR #269.
static void pack_pairs(const uint16_t *const *src, uint16_t *dst, int pairs,
                       int pair_stride = 64) __arm_streaming __arm_inout("za") {
  auto pg = svptrue_b32();
  for (int p = 0; p < pairs; p += 32) {
    for (int r = 0; r < 16; ++r) {
      svld1_hor_za32(0, r, pg, src[r] + 2 * p);
      svld1_hor_za32(1, r, pg, src[r] + 2 * (p + 16));
    }
    for (int c = 0; c < 16; ++c) {
      svst1_ver_za32(0, c, pg, dst + (p + c) * pair_stride);
      svst1_ver_za32(1, c, pg, dst + (p + c + 16) * pair_stride);
    }
  }
}

template <typename T> struct ScratchBuffer {
  std::unique_ptr<T[]> data;
  size_t capacity = 0;

  bool reserve(size_t size) {
    if (size <= capacity)
      return true;
    std::unique_ptr<T[]> buffer(new (std::nothrow) T[size]);
    if (!buffer)
      return false;
    data.swap(buffer);
    capacity = size;
    return true;
  }
};

struct KVCache {
  const uint16_t *source = nullptr;
  int rows = 0;
  ScratchBuffer<uint16_t> packed;
};

// Each worker only reuses packed KV from its current launch.
static thread_local KVCache dense_cache;

extern "C" __attribute__((visibility("default"))) void
kunpeng_sparse_mla_begin_launch(void) {
  dense_cache.source = nullptr;
}

__arm_locally_streaming __arm_new("za") static void pack_dense(
    const uint16_t *kv, uint16_t *packed, uint16_t *values, int rows) {
  const uint16_t *ptr[16];
  for (int t = 0; t < rows; t += 16) {
    for (int r = 0; r < 16; ++r)
      ptr[r] = kv + (t + r) * 576;
    pack_pairs(ptr, packed + (t / 32) * 32 * 576 + (t % 32) * 2, 288);
  }
  auto ph = svptrue_b16();
  for (int d = 0; d < 512; d += 32)
    for (int p = 0; p < rows / 2; ++p) {
      auto a = svld1_u16(ph, kv + p * 2 * 576 + d),
           b = svld1_u16(ph, kv + (p * 2 + 1) * 576 + d);
      svst1_u16(ph, values + d * rows + p * 64, svzip1_u16(a, b));
      svst1_u16(ph, values + d * rows + p * 64 + 32, svzip2_u16(a, b));
    }
}
__arm_locally_streaming __arm_new("za") static void prepare(
    const uint16_t *q, const uint16_t *kv, const int32_t *ids, uint16_t *qp,
    uint16_t *kp, uint16_t *vp, int keys, int row, bool use_dense) {
  auto ph = svptrue_b16();

  const uint16_t *rows[16];
  for (int h = 0; h < 64; h += 16) {
    for (int r = 0; r < 16; ++r)
      rows[r] = q + (h + r) * 576;
    pack_pairs(rows, qp + (h / 32) * 32 * 576 + (h % 32) * 2, 288);
  }
  if (use_dense)
    return;
  for (int t = 0; t < keys; t += 16) {
    for (int r = 0; r < 16; ++r)
      rows[r] =
          kv + ((ids[t + r] >= 0 && ids[t + r] <= row) ? ids[t + r] : 0) * 576;
    pack_pairs(rows, kp + (t / 32) * 32 * 576 + (t % 32) * 2, 288);
  }
  for (int d = 0; d < 512; d += 32)
    for (int p = 0; p < keys / 2; ++p) {
      int i0 = ids[p * 2], i1 = ids[p * 2 + 1];
      auto v0 = svld1_u16(ph, kv + ((i0 >= 0 && i0 <= row) ? i0 : 0) * 576 + d);
      auto v1 = svld1_u16(ph, kv + ((i1 >= 0 && i1 <= row) ? i1 : 0) * 576 + d);
      svst1_u16(ph, vp + d * keys + p * 64, svzip1_u16(v0, v1));
      svst1_u16(ph, vp + d * keys + p * 64 + 32, svzip2_u16(v0, v1));
    }
}
__arm_locally_streaming __arm_new("za") static bool probabilities(
    float *scores, uint16_t *pp, float *den, float *logs, const int32_t *ids,
    int keys, int row, float scale) {
  auto pg = svptrue_b32();
  for (int h = 0; h < 64; ++h) {
    float *s = scores + h * keys;
    auto mx = svdup_f32(-INFINITY);
    for (int t = 0; t < keys; t += 16) {
      auto id = svld1_s32(pg, ids + t);
      auto valid =
          svand_b_z(pg, svcmpge_n_s32(pg, id, 0), svcmple_n_s32(pg, id, row));
      auto v =
          svmul_n_f32_x(pg, svld1_f32(pg, s + t), scale * 1.4426950408889634f);
      if (!finite_lanes(valid, v))
        return false;
      v = svsel_f32(valid, v, svdup_f32(-INFINITY));
      svst1_f32(pg, s + t, v);
      mx = svmax_f32_x(pg, mx, v);
    }
    float m = svmaxv_f32(pg, mx);
    auto sum = svdup_f32(0);
    for (int t = 0; t < keys; t += 16) {
      auto x = svld1_f32(pg, s + t);
      auto e = exp2v(pg, svsub_n_f32_x(pg, x, m));
      e = svsel_f32(svcmpeq_n_f32(pg, x, -INFINITY), svdup_f32(0), e);
      sum = svadd_f32_x(pg, sum, e);
      svst1_f32(pg, s + t, e);
    }
    den[h] = svaddv_f32(pg, sum);
    if (!finite_lanes(pg, svdup_f32(den[h])) || !(den[h] > 0))
      return false;
    logs[h] = m;
  }
  for (int h = 0; h < 64; h += 16)
    for (int t = 0; t < keys; t += 32) {
      for (int r = 0; r < 16; ++r) {
        svld1_hor_za32(0, r, pg, scores + (h + r) * keys + t);
        svld1_hor_za32(1, r, pg, scores + (h + r) * keys + t + 16);
      }
      for (int c = 0; c < 16; c += 2) {
#define PACK_PROB(tile, offset)                                                \
  {                                                                            \
    auto x = svread_ver_za32_f32_m(svdup_f32(0), pg, tile, c);                 \
    auto y = svread_ver_za32_f32_m(svdup_f32(0), pg, tile, c + 1);             \
    auto lo = svand_n_u32_x(                                                   \
        pg, svreinterpret_u32_bf16(svcvt_bf16_f32_x(pg, x)), 0xffff);            \
    auto hi = svand_n_u32_x(                                                   \
        pg, svreinterpret_u32_bf16(svcvt_bf16_f32_x(pg, y)), 0xffff);            \
    auto lost_x = svand_b_z(pg, svcmpgt_n_f32(pg, x, 0),                       \
                            svcmpeq_n_u32(pg, lo, 0));                        \
    auto lost_y = svand_b_z(pg, svcmpgt_n_f32(pg, y, 0),                       \
                            svcmpeq_n_u32(pg, hi, 0));                        \
    if (svptest_any(pg, svorr_b_z(pg, lost_x, lost_y)))                        \
      return false;                                                          \
    auto x16 = svreinterpret_u16_u32(lo);                                      \
    auto y16 = svreinterpret_u16_u32(hi);                                      \
    auto pair = svzip1_u16(svuzp1_u16(x16, x16),                               \
                           svuzp1_u16(y16, y16));                              \
    auto *dest = pp + (h / 32) * 32 * keys +                                   \
                 (t + c + offset) / 2 * 64 + (h % 32) * 2;                     \
    svst1_u16(svptrue_b16(), dest, pair);                                      \
  }
        PACK_PROB(0, 0);
        PACK_PROB(1, 16);
#undef PACK_PROB
      }
    }
  return true;
}
__arm_locally_streaming static void finish(const float *result,
                                           const float *den, const float *logs,
                                           uint16_t *out, uint16_t *lse) {
  auto pg = svptrue_b32();
  auto ph = svwhilelt_b16(0, 16);
  // log2(x) = exponent + 2/ln(2) * atanh((mantissa-1)/(mantissa+1)).
  // The normalized mantissa is in [1,2), so the series argument is <= 1/3.
  for (int h = 0; h < 64; h += 16) {
    auto bits = svreinterpret_u32_f32(svld1_f32(pg, den + h));
    auto exponent = svsub_n_s32_x(
        pg, svreinterpret_s32_u32(svlsr_n_u32_x(pg, bits, 23)), 127);
    auto mantissa = svreinterpret_f32_u32(
        svorr_n_u32_x(pg, svand_n_u32_x(pg, bits, 0x7fffff), 0x3f800000));
    auto y = svdiv_f32_x(pg, svsub_n_f32_x(pg, mantissa, 1.f),
                         svadd_n_f32_x(pg, mantissa, 1.f));
    auto z = svmul_f32_x(pg, y, y);
    auto p = svdup_f32(1.f / 11.f);
    p = svmla_f32_x(pg, svdup_f32(1.f / 9.f), z, p);
    p = svmla_f32_x(pg, svdup_f32(1.f / 7.f), z, p);
    p = svmla_f32_x(pg, svdup_f32(1.f / 5.f), z, p);
    p = svmla_f32_x(pg, svdup_f32(1.f / 3.f), z, p);
    p = svmla_f32_x(pg, svdup_f32(1.f), z, p);
    auto value = svmla_n_f32_x(pg, svcvt_f32_s32_x(pg, exponent),
                               svmul_f32_x(pg, y, p), 2.8853900817779268f);
    value = svadd_f32_x(pg, value, svld1_f32(pg, logs + h));
    auto half = svreinterpret_u16_bf16(svcvt_bf16_f32_x(pg, value));
    svst1_u16(ph, lse + h, svuzp1_u16(half, half));
  }
  for (int h = 0; h < 64; ++h) {
    float inv = 1.f / den[h];
    for (int d = 0; d < 512; d += 16) {
      auto v = svmul_n_f32_x(pg, svld1_f32(pg, result + h * 512 + d), inv);
      auto bits = svreinterpret_u16_bf16(svcvt_bf16_f32_x(pg, v));
      svst1_u16(ph, out + h * 512 + d, svuzp1_u16(bits, bits));
    }
  }
}
extern "C" __attribute__((visibility("default"))) int
kunpeng_sparse_mla_forward_bf16(const uint16_t *q, const uint16_t *kv,
                                const int32_t *ids, uint16_t *out,
                                uint16_t *lse, int row, int topk, float scale,
                                int query_rows) {
  unsigned long vl;
  asm("rdsvl %0, #1" : "=r"(vl));
  if (vl != 64)
    return 1;
  if (row < 0 || row >= query_rows || topk <= 0 || topk > 4096 ||
      topk % 32)
    return 1;
  int keys = 32 * std::min(topk / 32, row / 32 + 1);
  bool any = false;
  for (int t = 0; t < 32; ++t)
    any |= ids[t] >= 0 && ids[t] <= row;
  if (!any)
    return 1; // Preserve the original online-softmax NaN behavior.

  thread_local ScratchBuffer<uint16_t> storage;
  thread_local ScratchBuffer<float> floats;
  size_t packed_size = 64 * 576 + keys * (576 + 512 + 64);
  size_t float_size = 64 * keys + 64 * 512 + 128;
  if (!storage.reserve(packed_size) || !floats.reserve(float_size))
    return 1;
  auto qp = storage.data.get(), kp = qp + 64 * 576, vp = kp + keys * 576,
       pp = vp + 512 * keys;
  auto scores = floats.data.get(), result = scores + 64 * keys,
       den = result + 64 * 512, logs = den + 64;

  constexpr int max_dense_rows = 2048;
  thread_local unsigned char seen[max_dense_rows];
  thread_local int32_t ordered[max_dense_rows];
  // A permutation of a complete key prefix can reuse packed dense tiles.
  // Detect duplicates and gaps; arbitrary sparse selections keep their order.
  bool dense_set = query_rows % 32 == 0 && query_rows <= max_dense_rows;
  int count = 0, maximum = -1;
  if (dense_set) {
    std::fill_n(seen, query_rows, 0);
    for (int t = 0; t < keys; ++t)
      if (ids[t] >= 0 && ids[t] <= row) {
        int id = ids[t];
        if (seen[id])
          dense_set = false;
        seen[id] = 1;
        ++count;
        maximum = std::max(maximum, id);
      }
    dense_set = dense_set && maximum + 1 == count;
  }
  const int32_t *active_ids = ids;
  const uint16_t *packed_keys = kp, *packed_values = vp;
  if (dense_set) {
    auto &cache = dense_cache;
    if (cache.source != kv || cache.rows != query_rows) {
      if (!cache.packed.reserve(query_rows * (576 + 512)))
        return 1;
      auto *data = cache.packed.data.get();
      pack_dense(kv, data, data + query_rows * 576, query_rows);
      cache.source = kv;
      cache.rows = query_rows;
    }
    keys = (count + 31) / 32 * 32;
    for (int t = 0; t < keys; ++t)
      ordered[t] = t < count ? t : -1;
    active_ids = ordered;
    packed_keys = cache.packed.data.get();
    packed_values = packed_keys + query_rows * 576;
  }
  prepare(q, kv, active_ids, qp, kp, vp, keys, row, dense_set);
  multiply(qp, packed_keys, scores, 64, keys, 576);
  if (!probabilities(scores, pp, den, logs, active_ids, keys, row, scale))
    return 1;
  multiply(pp, packed_values, result, 64, 512, keys,
           dense_set ? query_rows : keys);
  if (!finite_results(result, size_t(64) * 512))
    return 1;
  finish(result, den, logs, out, lse);
  return 0;
}
