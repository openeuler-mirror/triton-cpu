#ifndef TRITON_TO_KUNPENG_LLM_CONVERSION_PASSES_H
#define TRITON_TO_KUNPENG_LLM_CONVERSION_PASSES_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createTritonFlashAttentionToKunpengLLMPass();

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/TritonToKunpengLLM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
