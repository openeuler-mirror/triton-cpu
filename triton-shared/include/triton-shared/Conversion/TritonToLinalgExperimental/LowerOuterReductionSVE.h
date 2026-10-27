//===----------------------------------------------------------------------===//
// Lower tiled outer vector reductions in SVE-width chunks
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITONTOLINALG_LOWEROUTERREDUCTIONSVE_H
#define TRITON_CONVERSION_TRITONTOLINALG_LOWEROUTERREDUCTIONSVE_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<Pass> createLowerOuterReductionSVEPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONTOLINALG_LOWEROUTERREDUCTIONSVE_H
