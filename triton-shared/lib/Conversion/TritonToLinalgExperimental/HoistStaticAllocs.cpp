//===----------------------------------------------------------------------===//
// Static scratch allocations to the function entry block
//===----------------------------------------------------------------------===//
//
// LLVM's SROA only ever looks at allocas in a function's entry block.
// A statically sized scratch tile that ends up in any other block --
// inside a loop, or inside the guard will not be handled by LLVM SROA.
// Hoist these statically sized stratch allocas to the entry block for later
// optimizations.
//
#include "triton-shared/Conversion/TritonToLinalgExperimental/HoistStaticAllocs.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"

using namespace mlir;
using namespace triton;

namespace {

// Derive from the tablegen base rather than PassWrapper: the pass is declared
// in Passes.td, so it goes through registerPass(), which rejects a pass whose
// getArgument() is empty.  The base supplies the "hoist-static-allocs" argument
// and the description.
struct HoistStaticAllocsPass
    : public HoistStaticAllocsBase<HoistStaticAllocsPass> {
  /// Matches the per-buffer ceiling promote-buffers-to-stack is given, so this
  /// only touches allocations that are going to become stack slots anyway.
  static constexpr int64_t kMaxBytes = 16 * 1024;

  static bool smallStatic(MemRefType type) {
    if (!type.hasStaticShape())
      return false;
    auto elem = type.getElementType();
    if (!elem.isIntOrFloat())
      return false;
    return type.getNumElements() * elem.getIntOrFloatBitWidth() / 8 <= kMaxBytes;
  }

  /// Hoist every candidate in `body` to the front of its entry block.  Reversed
  /// so the original relative order is preserved.
  template <typename OpT, typename PredT>
  static void hoistInto(Operation *scope, Block &entry, PredT pred) {
    SmallVector<OpT> toHoist;
    scope->walk([&](OpT op) {
      if (op->getBlock() != &entry && pred(op))
        toHoist.push_back(op);
    });
    for (auto op : llvm::reverse(toHoist))
      op->moveBefore(&entry, entry.begin());
  }

  void runOnOperation() override {
    // Use an env variable to control this pass as too many hoists may increase
    // the pressure of registre.
    if (const char *off = getenv("TRITON_SHARED_HOIST_STATIC_ALLOCS"))
      if (off[0] == '0')
        return;
    // Before bufferization: memref.alloc straight out of StructuredToMemref.
    getOperation()->walk([&](func::FuncOp func) {
      if (func.isExternal())
        return;
      Block &entry = func.getBody().front();
      hoistInto<memref::AllocOp>(func, entry, [](memref::AllocOp op) {
        return op.getDynamicSizes().empty() && smallStatic(op.getType());
      });
      hoistInto<memref::AllocaOp>(func, entry, [](memref::AllocaOp op) {
        return op.getDynamicSizes().empty() && smallStatic(op.getType());
      });
    });

    // After bufferization and the transform schedule the scratch tiles are
    // llvm.alloca.  Their size is an operand, so only constant ones can move;
    // the constant is re-materialised in the entry block.
    getOperation()->walk([&](LLVM::LLVMFuncOp func) {
      if (func.isExternal())
        return;
      Block &entry = func.getBody().front();
      SmallVector<LLVM::AllocaOp> toHoist;
      func.walk([&](LLVM::AllocaOp op) {
        if (op->getBlock() == &entry)
          return;
        APInt n;
        if (!matchPattern(op.getArraySize(), m_ConstantInt(&n)))
          return;
        auto elem = op.getElemType();
        if (!elem.isIntOrFloat())
          return;
        if (n.getZExtValue() * elem.getIntOrFloatBitWidth() / 8 > kMaxBytes)
          return;
        toHoist.push_back(op);
      });
      for (auto op : llvm::reverse(toHoist)) {
        OpBuilder b(&entry, entry.begin());
        Value size = b.clone(*op.getArraySize().getDefiningOp())->getResult(0);
        op->moveBefore(&entry, entry.begin());
        op.getArraySizeMutable().assign(size);
        size.getDefiningOp()->moveBefore(op);
      }
    });
  }
};
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createHoistStaticAllocsPass() {
  return std::make_unique<HoistStaticAllocsPass>();
}
