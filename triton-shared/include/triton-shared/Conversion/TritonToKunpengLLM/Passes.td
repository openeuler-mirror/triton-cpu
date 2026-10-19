#ifndef TRITON_TO_KUNPENG_LLM_CONVERSION_PASSES
#define TRITON_TO_KUNPENG_LLM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonFlashAttentionToKunpengLLM
    : Pass<"triton-flash-attention-to-kunpeng-llm", "mlir::ModuleOp"> {
  let summary = "Replace a flash-attention forward kernel by a libkunpeng_llm call";
  let description = [{
    Recognises Triton kernels whose program computes one block of query rows
    of one attention head with the online-softmax (flash-attention) loop nest:

      S = dot(Q, K^T); S = select(window mask, -inf, S)
      m' = max(m, rowmax(S)); P = exp2(S * c - m' * c)
      l' = l * exp2((m - m') * c) + rowsum(P)
      acc' = acc * exp2((m - m') * c) + dot(P, V)
      o = acc / l; lse = m * scale + log(l)

    over one or two chained key loops, and replaces the whole kernel body by
    one call to kunpeng_flash_attention_forward_{bf16,f16} per program.

    The match is semantic rather than syntactic: load/store addresses, row
    and key indices, the mask and the key-loop bounds are evaluated as
    integer polynomials over SSA values and tensor indices, and the pass
    checks that the pointers address [rows, head_dim] row-major tiles of one
    head, that the loops visit every key the mask admits, and that keys
    visited by an unmasked loop are visible to every row.  Kernels that do
    anything else (dropout, alibi, softcap, extra stores, paged K/V, ...)
    are left untouched.
  }];
  let constructor = "triton::createTritonFlashAttentionToKunpengLLMPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::arith::ArithDialect"];
}

#endif
