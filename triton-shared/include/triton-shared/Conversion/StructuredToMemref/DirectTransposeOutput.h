// Rewrites eligible buffered transposes to write directly to their outputs.
#ifndef TRITON_SHARED_CONVERSION_DIRECT_TRANSPOSE_OUTPUT_H
#define TRITON_SHARED_CONVERSION_DIRECT_TRANSPOSE_OUTPUT_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::triton {
std::unique_ptr<OperationPass<func::FuncOp>> createDirectTransposeOutput();
}

#endif
