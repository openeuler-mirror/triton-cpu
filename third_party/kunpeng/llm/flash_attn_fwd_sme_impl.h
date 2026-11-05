// Element-type specific part of flash_attn_fwd_sme.c, included once per input
// type.  Expects FA_SUFFIX, FA_MOPA(tile, a, b) and FA_CVT_PAIR(pg, lo, hi).

#define FA_CAT2(a, b) a##_##b
#define FA_CAT(a, b) FA_CAT2(a, b)
#define FN(name) FA_CAT(name, FA_SUFFIX)

// Make key tiles [t0, t1) of one head resident in the K/V tile cache.
static inline void
FN(ensure_tiles)(const fa_args a, fa_scratch s, const uint16_t *kb,
                 const uint16_t *vb, int64_t t0,
                 int64_t t1) __arm_streaming __arm_inout("za") {
  const int D = a.D;
  for (int64_t t = t0; t < t1; t++) {
    const int64_t slot = t % s.ntile;
    if (s.tag[slot] == t)
      continue;
    pack_rows_u32(s.Kp + slot * L * D / 2, kb, a.k_rs, t * L, a.Sk, D / 2);
    pack_v(s.Vp + slot * L * D, vb, a.v_rs, t * L, L, a.Sk, D);
    s.tag[slot] = t;
  }
}

// S^T[j][BM] = (Q K^T)^T for nrt row tiles x nct key tiles (starting at key
// tile t0), 2x2 ZA tiles at a time.
static inline void FN(s_gemm)(const fa_scratch s, int D, int64_t t0, int nct,
                              float *ST, int nrt,
                              int BM) __arm_streaming __arm_inout("za") {
  const svbool_t pt = svptrue_b32(), ph = svptrue_b16();
  const int npair = D / 2;
  const int64_t tile_stride = (int64_t)npair * 2 * L; // in u16
  for (int rt = 0; rt < nrt; rt += 2) {
    const int two_r = rt + 1 < nrt;
    const uint16_t *a0 = (const uint16_t *)s.Qp + rt * tile_stride;
    const uint16_t *a1 = a0 + tile_stride;
    for (int ct = 0; ct < nct; ct += 2) {
      const int two_c = ct + 1 < nct;
      const uint16_t *b0 =
          (const uint16_t *)s.Kp + ((t0 + ct) % s.ntile) * tile_stride;
      const uint16_t *b1 =
          (const uint16_t *)s.Kp + ((t0 + ct + 1) % s.ntile) * tile_stride;
      svzero_za();
      if (two_r && two_c) {
        for (int p = 0; p < npair; p++) {
          svuint16_t va0 = svld1_u16(ph, a0 + p * 2 * L);
          svuint16_t va1 = svld1_u16(ph, a1 + p * 2 * L);
          svuint16_t vb0 = svld1_u16(ph, b0 + p * 2 * L);
          svuint16_t vb1 = svld1_u16(ph, b1 + p * 2 * L);
          FA_MOPA(0, va0, vb0);
          FA_MOPA(1, va0, vb1);
          FA_MOPA(2, va1, vb0);
          FA_MOPA(3, va1, vb1);
        }
      } else if (two_r) {
        for (int p = 0; p < npair; p++) {
          svuint16_t va0 = svld1_u16(ph, a0 + p * 2 * L);
          svuint16_t va1 = svld1_u16(ph, a1 + p * 2 * L);
          svuint16_t vb0 = svld1_u16(ph, b0 + p * 2 * L);
          FA_MOPA(0, va0, vb0);
          FA_MOPA(2, va1, vb0);
        }
      } else if (two_c) {
        for (int p = 0; p < npair; p++) {
          svuint16_t va0 = svld1_u16(ph, a0 + p * 2 * L);
          svuint16_t vb0 = svld1_u16(ph, b0 + p * 2 * L);
          svuint16_t vb1 = svld1_u16(ph, b1 + p * 2 * L);
          FA_MOPA(0, va0, vb0);
          FA_MOPA(1, va0, vb1);
        }
      } else {
        for (int p = 0; p < npair; p++) {
          svuint16_t va0 = svld1_u16(ph, a0 + p * 2 * L);
          svuint16_t vb0 = svld1_u16(ph, b0 + p * 2 * L);
          FA_MOPA(0, va0, vb0);
        }
      }
      // Vertical slice j of a tile holds the scores of key j for 16 rows.
      float *s00 = ST + (int64_t)(ct * L) * BM + rt * L;
      for (int j = 0; j < L; j++)
        svst1_ver_za32(0, j, pt, s00 + (int64_t)j * BM);
      if (two_c)
        for (int j = 0; j < L; j++)
          svst1_ver_za32(1, j, pt, s00 + (int64_t)(L + j) * BM);
      if (two_r)
        for (int j = 0; j < L; j++)
          svst1_ver_za32(2, j, pt, s00 + (int64_t)j * BM + L);
      if (two_r && two_c)
        for (int j = 0; j < L; j++)
          svst1_ver_za32(3, j, pt, s00 + (int64_t)(L + j) * BM + L);
    }
  }
}

// Online-softmax update for keys [c, c + nk): masks ST, writes the packed
// probabilities Pp, updates the running max (scaled, log2 domain) and sum,
// and rescales OT.
static inline void FN(softmax)(const fa_args a, float *ST, uint16_t *Pp,
                               float *OT, float *mrow, float *lrow, int64_t r0,
                               int nrt, int64_t c, int nk, int first,
                               int BM) __arm_streaming {
  const svbool_t pt = svptrue_b32();
  const svfloat32_t ninf = svdup_f32(-INFINITY);
  const svfloat32_t scale = svdup_f32(a.scale_log2);
  const int nkp = nk / 2;
  for (int rt = 0; rt < nrt; rt++) {
    const int64_t rbase = r0 + rt * L;
    // Keys [c + f0, c + f1) are visible to all 16 rows; only the rest needs
    // the per-lane window test.
    int64_t f0 = row_lb(a, rbase + L - 1) - c;
    int64_t f1 = row_rb(a, rbase) + 1 - c;
    f0 = f0 < 0 ? 0 : (f0 > nk ? nk : f0);
    f1 = f1 < f0 ? f0 : (f1 > nk ? nk : f1);
    svint32_t rows = svindex_s32((int32_t)rbase, 1);
    svint32_t lb = svdup_s32(0), rb = svdup_s32(a.Sk - 1);
    if (a.has_left)
      lb = svmax_n_s32_x(pt, svadd_n_s32_x(pt, rows, (int32_t)a.left), 0);
    if (a.has_right)
      rb = svmin_n_s32_x(pt, svadd_n_s32_x(pt, rows, (int32_t)a.right),
                         a.Sk - 1);

    float *st = ST + rt * L;
    svfloat32_t mx0 = ninf, mx1 = ninf;
#define FA_MASK_KEY(j, mx)                                                     \
  do {                                                                         \
    svint32_t kj = svdup_s32((int32_t)(c + (j)));                              \
    svbool_t ok =                                                              \
        svand_b_z(pt, svcmpge_s32(pt, kj, lb), svcmple_s32(pt, kj, rb));       \
    svfloat32_t v =                                                            \
        svsel_f32(ok, svld1_f32(pt, st + (int64_t)(j) * BM), ninf);            \
    svst1_f32(pt, st + (int64_t)(j) * BM, v);                                  \
    mx = svmax_f32_x(pt, mx, v);                                               \
  } while (0)
    int jm = 0;
    for (; jm + 1 < f0; jm += 2) {
      FA_MASK_KEY(jm, mx0);
      FA_MASK_KEY(jm + 1, mx1);
    }
    if (jm < f0)
      FA_MASK_KEY(jm, mx0);
    int j = (int)f0;
    {
      svfloat32_t mx2 = ninf, mx3 = ninf;
      for (; j + 3 < f1; j += 4) {
        mx0 = svmax_f32_x(pt, mx0, svld1_f32(pt, st + (int64_t)j * BM));
        mx1 = svmax_f32_x(pt, mx1, svld1_f32(pt, st + (int64_t)(j + 1) * BM));
        mx2 = svmax_f32_x(pt, mx2, svld1_f32(pt, st + (int64_t)(j + 2) * BM));
        mx3 = svmax_f32_x(pt, mx3, svld1_f32(pt, st + (int64_t)(j + 3) * BM));
      }
      for (; j < f1; j++)
        mx0 = svmax_f32_x(pt, mx0, svld1_f32(pt, st + (int64_t)j * BM));
      mx0 = svmax_f32_x(pt, mx0, mx2);
      mx1 = svmax_f32_x(pt, mx1, mx3);
    }
    for (j = (int)f1; j + 1 < nk; j += 2) {
      FA_MASK_KEY(j, mx0);
      FA_MASK_KEY(j + 1, mx1);
    }
    if (j < nk)
      FA_MASK_KEY(j, mx0);
#undef FA_MASK_KEY
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
      const float *sk = st + (int64_t)(2 * kp) * BM;
      svfloat32_t p0 =
          exp2v(pt, svmad_f32_x(pt, svld1_f32(pt, sk), scale, nmsafe));
      svfloat32_t p1 =
          exp2v(pt, svmad_f32_x(pt, svld1_f32(pt, sk + BM), scale, nmsafe));
      svfloat32_t p2 =
          exp2v(pt, svmad_f32_x(pt, svld1_f32(pt, sk + 2 * BM), scale, nmsafe));
      svfloat32_t p3 =
          exp2v(pt, svmad_f32_x(pt, svld1_f32(pt, sk + 3 * BM), scale, nmsafe));
      svfloat32_t p4 =
          exp2v(pt, svmad_f32_x(pt, svld1_f32(pt, sk + 4 * BM), scale, nmsafe));
      svfloat32_t p5 =
          exp2v(pt, svmad_f32_x(pt, svld1_f32(pt, sk + 5 * BM), scale, nmsafe));
      svfloat32_t p6 =
          exp2v(pt, svmad_f32_x(pt, svld1_f32(pt, sk + 6 * BM), scale, nmsafe));
      svfloat32_t p7 =
          exp2v(pt, svmad_f32_x(pt, svld1_f32(pt, sk + 7 * BM), scale, nmsafe));
      l0 = svadd_f32_x(pt, l0, svadd_f32_x(pt, p0, p1));
      l1 = svadd_f32_x(pt, l1, svadd_f32_x(pt, p2, p3));
      l2 = svadd_f32_x(pt, l2, svadd_f32_x(pt, p4, p5));
      l3 = svadd_f32_x(pt, l3, svadd_f32_x(pt, p6, p7));
      uint16_t *pk = pp + (int64_t)kp * 2 * L;
      svst1_u16(svptrue_b16(), pk, FA_CVT_PAIR(pt, p0, p1));
      svst1_u16(svptrue_b16(), pk + 2 * L, FA_CVT_PAIR(pt, p2, p3));
      svst1_u16(svptrue_b16(), pk + 4 * L, FA_CVT_PAIR(pt, p4, p5));
      svst1_u16(svptrue_b16(), pk + 6 * L, FA_CVT_PAIR(pt, p6, p7));
    }
    l0 = svadd_f32_x(pt, svadd_f32_x(pt, l0, l1), svadd_f32_x(pt, l2, l3));
    svst1_f32(pt, mrow + rt * L, mx);
    svst1_f32(pt, lrow + rt * L, l0);
    if (!first) {
      float *ot = OT + rt * L;
      for (int d = 0; d < a.D; d++)
        svst1_f32(pt, ot + (int64_t)d * BM,
                  svmul_f32_x(pt, svld1_f32(pt, ot + (int64_t)d * BM), alpha));
    }
  }
}

// 1 / row sum, or 0 for rows that saw no key (running max still -inf).
static inline void FN(row_inv)(const float *mrow, const float *lrow, float *inv,
                               int nrt) __arm_streaming {
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

// Rows of ZA tile `t` scaled by inv[] -> output rows r .. r + nrow.
#define FA_WRITE_ROW(t, i, inv_v, out_p, o_rs)                                 \
  do {                                                                         \
    svfloat32_t x = svread_hor_za32_f32_m(svdup_f32(0.0f), pt, t, i);          \
    x = svmul_f32_x(pt, x, svdup_lane_f32(inv_v, i));                          \
    svuint16_t hv = FA_CVT_PAIR(pt, x, x); /* value in even lanes */           \
    svst1_u16(ph16, (out_p) + (i) * (o_rs), svuzp1_u16(hv, hv));               \
  } while (0)
#define FA_WRITE_TILE(t, inv_v, out_p, nrow, o_rs)                             \
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
      svuint16_t h01 = FA_CVT_PAIR(pt, x0, x1);                                \
      svuint16_t h23 = FA_CVT_PAIR(pt, x2, x3);                                \
      /* even lanes = row i (i + 2), odd lanes = row i + 1 (i + 3) */          \
      svst1_u16(ph16, (out_p) + i * (o_rs), svuzp1_u16(h01, h01));             \
      svst1_u16(ph16, (out_p) + (i + 1) * (o_rs), svuzp2_u16(h01, h01));       \
      svst1_u16(ph16, (out_p) + (i + 2) * (o_rs), svuzp1_u16(h23, h23));       \
      svst1_u16(ph16, (out_p) + (i + 3) * (o_rs), svuzp2_u16(h23, h23));       \
    }                                                                          \
    for (; i < (nrow); i++)                                                    \
      FA_WRITE_ROW(t, i, inv_v, out_p, o_rs);                                  \
  } while (0)

// OT[d][BM] += (P V)^T over key tiles [t0, t0 + nkt), 2x2 ZA tiles at a time.
// On the last chunk the result is normalised and written straight from ZA to
// the output rows [r0, r0 + nr).
static inline void FN(pv_gemm)(const fa_args a, const fa_scratch s, int64_t t0,
                               int nkt, int nrt, int first, int last,
                               uint16_t *ob, int64_t r0, int nr,
                               int BM) __arm_streaming __arm_inout("za") {
  const svbool_t pt = svptrue_b32(), ph = svptrue_b16();
  const svbool_t ph16 = svwhilelt_b16(0, L);
  const int D = a.D, ndt = D / L;
  const int64_t vstride = (int64_t)ndt * 2 * L; // one key pair, in u16
  const int64_t vtile = (int64_t)L * D;         // one key tile, in u16
  const int64_t ptile = (int64_t)nkt * L * L;   // P of one row tile, in u16
  float *OT = s.OT;
  for (int rt = 0; rt < nrt; rt += 2) {
    const int two_r = rt + 1 < nrt;
    const uint16_t *a0 = s.Pp + rt * ptile;
    const uint16_t *a1 = a0 + ptile;
    for (int dt = 0; dt < ndt; dt += 2) {
      const int two_c = dt + 1 < ndt;
      float *o00 = OT + (int64_t)(dt * L) * BM + rt * L;
      if (first) {
        svzero_za();
      } else {
        for (int j = 0; j < L; j++)
          svld1_ver_za32(0, j, pt, o00 + (int64_t)j * BM);
        if (two_c)
          for (int j = 0; j < L; j++)
            svld1_ver_za32(1, j, pt, o00 + (int64_t)(L + j) * BM);
        if (two_r)
          for (int j = 0; j < L; j++)
            svld1_ver_za32(2, j, pt, o00 + (int64_t)j * BM + L);
        if (two_r && two_c)
          for (int j = 0; j < L; j++)
            svld1_ver_za32(3, j, pt, o00 + (int64_t)(L + j) * BM + L);
      }
      for (int kt = 0; kt < nkt; kt++) {
        const uint16_t *b0 = s.Vp + ((t0 + kt) % s.ntile) * vtile + dt * 2 * L;
        const uint16_t *b1 = b0 + 2 * L;
        const uint16_t *pa0 = a0 + kt * L * L;
        const uint16_t *pa1 = a1 + kt * L * L;
        if (two_r && two_c) {
          for (int kp = 0; kp < L / 2; kp++) {
            svuint16_t va0 = svld1_u16(ph, pa0 + kp * 2 * L);
            svuint16_t va1 = svld1_u16(ph, pa1 + kp * 2 * L);
            svuint16_t vb0 = svld1_u16(ph, b0 + kp * vstride);
            svuint16_t vb1 = svld1_u16(ph, b1 + kp * vstride);
            FA_MOPA(0, va0, vb0);
            FA_MOPA(1, va0, vb1);
            FA_MOPA(2, va1, vb0);
            FA_MOPA(3, va1, vb1);
          }
        } else if (two_r) {
          for (int kp = 0; kp < L / 2; kp++) {
            svuint16_t va0 = svld1_u16(ph, pa0 + kp * 2 * L);
            svuint16_t va1 = svld1_u16(ph, pa1 + kp * 2 * L);
            svuint16_t vb0 = svld1_u16(ph, b0 + kp * vstride);
            FA_MOPA(0, va0, vb0);
            FA_MOPA(2, va1, vb0);
          }
        } else if (two_c) {
          for (int kp = 0; kp < L / 2; kp++) {
            svuint16_t va0 = svld1_u16(ph, pa0 + kp * 2 * L);
            svuint16_t vb0 = svld1_u16(ph, b0 + kp * vstride);
            svuint16_t vb1 = svld1_u16(ph, b1 + kp * vstride);
            FA_MOPA(0, va0, vb0);
            FA_MOPA(1, va0, vb1);
          }
        } else {
          for (int kp = 0; kp < L / 2; kp++) {
            svuint16_t va0 = svld1_u16(ph, pa0 + kp * 2 * L);
            svuint16_t vb0 = svld1_u16(ph, b0 + kp * vstride);
            FA_MOPA(0, va0, vb0);
          }
        }
      }
      if (last) {
        const int64_t o_rs = a.o_rs;
        const int n0 = nr - rt * L < L ? nr - rt * L : L;
        const int n1 = nr - rt * L - L < L ? nr - rt * L - L : L;
        uint16_t *out0 = ob + (r0 + rt * L) * o_rs + dt * L;
        const svfloat32_t inv0 = svld1_f32(pt, s.inv + rt * L);
        const svfloat32_t inv1 = svld1_f32(pt, s.inv + rt * L + L);
        FA_WRITE_TILE(0, inv0, out0, n0, o_rs);
        if (two_c)
          FA_WRITE_TILE(1, inv0, out0 + L, n0, o_rs);
        if (two_r)
          FA_WRITE_TILE(2, inv1, out0 + L * o_rs, n1, o_rs);
        if (two_r && two_c)
          FA_WRITE_TILE(3, inv1, out0 + L * o_rs + L, n1, o_rs);
      } else {
        for (int j = 0; j < L; j++)
          svst1_ver_za32(0, j, pt, o00 + (int64_t)j * BM);
        if (two_c)
          for (int j = 0; j < L; j++)
            svst1_ver_za32(1, j, pt, o00 + (int64_t)(L + j) * BM);
        if (two_r)
          for (int j = 0; j < L; j++)
            svst1_ver_za32(2, j, pt, o00 + (int64_t)j * BM + L);
        if (two_r && two_c)
          for (int j = 0; j < L; j++)
            svst1_ver_za32(3, j, pt, o00 + (int64_t)(L + j) * BM + L);
      }
    }
  }
}

static inline void FN(zero_rows)(uint16_t *ob, int64_t o_rs, int64_t r0, int nr,
                                 int D) __arm_streaming {
  for (int i = 0; i < nr; i++)
    for (int d = 0; d < D; d += (int)svcnth())
      svst1_u16(svwhilelt_b16(d, D), ob + (r0 + i) * o_rs + d, svdup_u16(0));
}

// Query rows [rbeg, rend) of the head addressed by *ap.
__arm_locally_streaming __arm_new("za") static void FN(task)(
    const fa_args *ap, int64_t rbeg, int64_t rend, const fa_scratch *sp) {
  const fa_args a = *ap;
  const fa_scratch s = *sp;
  const int D = a.D, BM = a.block_m, NC = a.chunk_n;
  const int npair = D / 2;
  const uint16_t *qb = a.q, *kb = a.k, *vb = a.v;
  uint16_t *ob = a.o;

  for (int64_t r0 = rbeg; r0 < rend; r0 += BM) {
    const int nr = rend - r0 < BM ? (int)(rend - r0) : BM;
    const int nrt = (nr + L - 1) / L;
    float *mrow = s.m + (r0 - rbeg);
    float *lrow = s.l + (r0 - rbeg);
    const int64_t lo = row_lb(a, r0) & ~(int64_t)(L - 1);
    const int64_t hi = (row_rb(a, r0 + nr - 1) + 1 + L - 1) & ~(int64_t)(L - 1);

    // Pull in what the next block adds while this one computes.
    if (r0 + BM < rend) {
      const int64_t r1 = r0 + BM;
      const int64_t nr1 = rend - r1 < BM ? rend - r1 : BM;
      const int64_t hi1 = row_rb(a, r1 + nr1 - 1) + 1;
      const int64_t lo1 = row_lb(a, r1);
      prefetch_rows(qb, a.q_rs, r1, r1 + nr1, a.Sq, D);
      prefetch_rows(kb, a.k_rs, lo1 > hi ? lo1 : hi, hi1, a.Sk, D);
      prefetch_rows(vb, a.v_rs, lo1 > hi ? lo1 : hi, hi1, a.Sk, D);
    }

    for (int rt = 0; rt < nrt; rt++) {
      pack_rows_u32(s.Qp + (int64_t)rt * npair * L, qb, a.q_rs, r0 + rt * L,
                    a.Sq, npair);
      svst1_f32(svptrue_b32(), mrow + rt * L, svdup_f32(-INFINITY));
      svst1_f32(svptrue_b32(), lrow + rt * L, svdup_f32(0.0f));
    }

    if (hi <= lo) {
      FN(zero_rows)(ob, a.o_rs, r0, nr, D);
      continue;
    }
    for (int64_t c = lo; c < hi; c += NC) {
      const int nk = hi - c < NC ? (int)(hi - c) : NC;
      const int64_t t0 = c / L;
      const int first = c == lo, last = c + nk >= hi;
      FN(ensure_tiles)(a, s, kb, vb, t0, t0 + nk / L);
      FN(s_gemm)(s, D, t0, nk / L, s.ST, nrt, BM);
      FN(softmax)(a, s.ST, s.Pp, s.OT, mrow, lrow, r0, nrt, c, nk, first, BM);
      if (last)
        FN(row_inv)(mrow, lrow, s.inv, nrt);
      FN(pv_gemm)(a, s, t0, nk / L, nrt, first, last, ob, r0, nr, BM);
    }
  }
}

// Query rows [row_start, row_start + num_rows) (clipped to Sq) of one head;
// lse points at row 0 of that head.
static int FN(fa_block)(const fa_args *a, int64_t row_start, int64_t num_rows,
                        float *lse) {
  const int64_t rbeg = row_start < 0 ? 0 : row_start;
  const int64_t rend =
      row_start + num_rows < a->Sq ? row_start + num_rows : a->Sq;
  if (rbeg >= rend)
    return 0;
  fa_scratch *s = get_scratch(a, rend - rbeg);
  if (!s)
    return -2;
  FN(task)(a, rbeg, rend, s);
  for (int64_t r = rbeg; r < rend; r++) {
    const float m = s->m[r - rbeg], l = s->l[r - rbeg];
    lse[r] = (l > 0.0f && m != -INFINITY) ? m * 0.69314718055994531f + logf(l)
                                          : -INFINITY;
  }
  return 0;
}

#undef FN
#undef FA_CAT
#undef FA_CAT2
