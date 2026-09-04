//===----------------------------------------------------------------------===//
//
// Copyright (c) OpenEuler.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITONTOLINALG_HoistStaticAllocs
#define TRITON_CONVERSION_TRITONTOLINALG_HoistStaticAllocs

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createHoistStaticAllocsPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONTOLINALG_HoistStaticAllocs