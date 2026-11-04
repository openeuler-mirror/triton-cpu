//===----------------------------------------------------------------------===//
// Double-buffer loop-carried allocations
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITONTOLINALG_EXPANDEXP2FEXPA_H
#define TRITON_CONVERSION_TRITONTOLINALG_EXPANDEXP2FEXPA_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createExpandExp2FexpaPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONTOLINALG_EXPANDEXP2FEXPA_H
