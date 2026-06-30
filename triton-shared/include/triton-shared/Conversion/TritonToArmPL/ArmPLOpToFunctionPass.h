#ifndef TRITON_CONVERSION_TRITONTOARMPL_ARMPLOPTOFUNCTIONPASS_H
#define TRITON_CONVERSION_TRITONTOARMPL_ARMPLOPTOFUNCTIONPASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/TritonToArmPL/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createTritonDotToArmPLOpPass();
std::unique_ptr<OperationPass<ModuleOp>> createArmPLOpToFunctionPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONTOARMPL_ARMPLOPTOFUNCTIONPASS_H