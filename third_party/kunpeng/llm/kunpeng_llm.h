// libkunpeng_llm.so: LLM kernels for Kunpeng CPUs, called from code generated
// by the triton-shared compiler (triton-shared/lib/Conversion/
// TritonToKunpengLLM).  Keep these signatures in sync with that pass.

#ifndef KUNPENG_LLM_H
#define KUNPENG_LLM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KUNPENG_FA_WINDOW_LEFT 1
#define KUNPENG_FA_WINDOW_RIGHT 2

// Flash-attention forward for query rows [row_start, row_start + num_rows)
// (clipped to [0, seqlen_q)) of one (batch, head):
//
//   o[r]   = sum over visible j of softmax_j(scale * q[r] . k[j]) * v[j]
//   lse[r] = log(sum over visible j of exp(scale * q[r] . k[j]))
//
// Key j is visible from row r iff 0 <= j < seqlen_k and
//   r + left_shift <= j    if window_flags & KUNPENG_FA_WINDOW_LEFT
//   j <= r + right_shift   if window_flags & KUNPENG_FA_WINDOW_RIGHT
// e.g. a (wl, wr) sliding window is left_shift = seqlen_k - seqlen_q - wl,
// right_shift = seqlen_k - seqlen_q + wr; causal is right-bounded only, with
// wr = 0.  Rows without a visible key get o = 0 and lse = -inf.
//
// q, k, v and o point at row 0 of the head, rows are *_rs elements apart and
// the head_dim elements of a row are contiguous; lse points at row 0 of the
// head and is contiguous.  scale_log2 = scale * log2(e).  Inputs and o are
// bf16 or fp16; accumulation is fp32.
//
// Needs head_dim % 16 == 0, head_dim <= 256 and 512-bit SME vectors; aborts
// with a message otherwise or if scratch memory cannot be allocated.
void kunpeng_flash_attention_forward_bf16(
    const void *q, int64_t q_rs, const void *k, int64_t k_rs, const void *v,
    int64_t v_rs, void *o, int64_t o_rs, float *lse, int64_t row_start,
    int64_t num_rows, int64_t seqlen_q, int64_t seqlen_k, int64_t window_flags,
    int64_t left_shift, int64_t right_shift, int64_t head_dim,
    float scale_log2);
void kunpeng_flash_attention_forward_f16(
    const void *q, int64_t q_rs, const void *k, int64_t k_rs, const void *v,
    int64_t v_rs, void *o, int64_t o_rs, float *lse, int64_t row_start,
    int64_t num_rows, int64_t seqlen_q, int64_t seqlen_k, int64_t window_flags,
    int64_t left_shift, int64_t right_shift, int64_t head_dim,
    float scale_log2);

// Called by the Triton launcher before every launch of a kernel that uses
// this library: per-thread caches of packed inputs are dropped across
// launches, because inputs may be rewritten in place between them.
void kunpeng_llm_begin_launch(void);

// SME streaming vector length in bytes (64 is required).
int kunpeng_llm_sme_vector_bytes(void);

#ifdef __cplusplus
}
#endif

#endif // KUNPENG_LLM_H
