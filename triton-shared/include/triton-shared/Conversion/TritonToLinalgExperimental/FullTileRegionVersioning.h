//===----------------------------------------------------------------------===//
//
// Copyright (c) OpenEuler.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITONTOLINALG_FullTileRegionVersioning_H
#define TRITON_CONVERSION_TRITONTOLINALG_FullTileRegionVersioning_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<Pass> createFullTileVersioningPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONTOLINALG_FullTileRegionVersioning_H
