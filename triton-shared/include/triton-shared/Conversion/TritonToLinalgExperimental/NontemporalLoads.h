//===----------------------------------------------------------------------===//
//
// Copyright (c) OpenEuler.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITONTOLINALG_NontemporalLoads
#define TRITON_CONVERSION_TRITONTOLINALG_NontemporalLoads

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

constexpr const char *kNontemporalLoadArgsAttr = "tts.nontemporal_load_args";
constexpr const char *kNontemporalStoreArgsAttr = "tts.nontemporal_store_args";
constexpr const char *kPtrArgCountAttr = "tts.ptr_arg_count";

std::unique_ptr<OperationPass<ModuleOp>>
createTritonAnnotateNontemporalArgsPass();
std::unique_ptr<OperationPass<ModuleOp>> createMarkNontemporalLoadsPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONTOLINALG_NontemporalLoads
