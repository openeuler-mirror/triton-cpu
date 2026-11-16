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

// As above, but K/V come from a paged cache (flash_attn_varlen_func): key j
// of this sequence is row (j % page_size) of page page_table[j / page_size],
//
//   k + page_table[j / page_size] * k_page_stride + (j % page_size) * k_rs
//
// with page_table pointing at the first page of this sequence and k, v at
// row 0 of the KV head.  q, o and lse address this sequence's rows of this
// query head, as above; seqlen_q and seqlen_k are the sequence's own lengths.
// Needs page_size > 0 and a non-null page_table.
void kunpeng_flash_attention_varlen_forward_bf16(
    const void *q, int64_t q_rs, const void *k, int64_t k_page_stride,
    int64_t k_rs, const void *v, int64_t v_page_stride, int64_t v_rs, void *o,
    int64_t o_rs, float *lse, const int32_t *page_table, int64_t page_size,
    int64_t row_start, int64_t num_rows, int64_t seqlen_q, int64_t seqlen_k,
    int64_t window_flags, int64_t left_shift, int64_t right_shift,
    int64_t head_dim, float scale_log2);
void kunpeng_flash_attention_varlen_forward_f16(
    const void *q, int64_t q_rs, const void *k, int64_t k_page_stride,
    int64_t k_rs, const void *v, int64_t v_page_stride, int64_t v_rs, void *o,
    int64_t o_rs, float *lse, const int32_t *page_table, int64_t page_size,
    int64_t row_start, int64_t num_rows, int64_t seqlen_q, int64_t seqlen_k,
    int64_t window_flags, int64_t left_shift, int64_t right_shift,
    int64_t head_dim, float scale_log2);

// Flash-MLA decode forward (multi-head latent attention with paged KV) for
// one (batch, head block) program with a single query position (s_q == 1):
//
//   o[h] = sum over keys j < seqlen of
//          softmax_j(scale * q[h] . k[j]) * v[j]
//
// with q[h] a d-element row (the first dv elements are the latent value
// part, the remaining d - dv the rope part; k[j] is the matching d-element
// key row), v[j] the first dv elements of key row j, and softmax over all
// seqlen keys (a decode step sees every cached key, no mask).
//
// Key rows are gathered through the batch's page table: key j of page p is
// kv + (p * page_size + j % page_size) * kv_rs, where p = block_table[j /
// page_size] and kv rows are kv_rs elements apart with d contiguous.  q and
// o point at head 0 of the batch's head block (the head-block offset is
// already folded in by the caller), heads are q_hs / o_hs elements apart,
// and the head_dim elements of a row are contiguous.  scale is the softmax
// scale; inputs and o are bf16, accumulation is fp32.  No lse is produced.
//
// Needs num_heads, page_size, d and dv multiples of 16 (num_heads <= 128,
// page_size <= 512, dv <= d <= 1024) and 512-bit SME vectors; aborts with a
// message otherwise or if scratch memory cannot be allocated.
void kunpeng_flash_mla_bf16(const void *q, int64_t q_hs, const void *kv,
                            int64_t kv_rs, const int32_t *block_table,
                            int64_t seqlen, void *o, int64_t o_hs,
                            int64_t num_heads, int64_t page_size, int64_t dv,
                            int64_t d, float scale);

// Called by the Triton launcher before every launch of a kernel that uses
// this library: per-thread caches of packed inputs are dropped across
// launches, because inputs may be rewritten in place between them.
void kunpeng_llm_begin_launch(void);

// Sparse MLA: one query row and 64 heads, DQK=576, DV=512.
// Returns nonzero when the original Triton program must handle this tile.
int kunpeng_sparse_mla_forward_bf16(const uint16_t *q, const uint16_t *kv,
    const int32_t *indices, uint16_t *output, uint16_t *lse,
    int row, int topk, float scale, int query_rows);
// Invalidates sparse MLA packed KV before each sparse MLA launch.
void kunpeng_sparse_mla_begin_launch(void);

// SME streaming vector length in bytes (64 is required).
int kunpeng_llm_sme_vector_bytes(void);

#ifdef __cplusplus
}
#endif

#endif // KUNPENG_LLM_H
