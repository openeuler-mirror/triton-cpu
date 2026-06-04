//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Conversion/TritonArithToLinalg/TritonArithToLinalg.h"
#include "triton-shared/Conversion/TritonPtrToMemref/TritonPtrToMemref.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/CollapseShape.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/ReconcilePtrCasts.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimental.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToPtr.h"
#include "triton-shared/Conversion/TritonToStructured/TritonToStructured.h"
#include "triton-shared/Conversion/TritonToUnstructured/TritonToUnstructured.h"
#include "triton-shared/Conversion/UnstructuredToMemref/UnstructuredToMemref.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"

namespace {

/// Folds `arith.cmpi slt/ult (tt.make_range {0, N}, splat<M>)` into
/// `arith.constant dense<true>` when M >= N.  The range [0, N) is always
/// strictly less than M, so the comparison is trivially always-true.
/// Running this before TritonToUnstructured ensures that GatherOps created
/// from always-masked loads carry a concrete dense<true> mask, which lets
/// GatherConverter skip the dead scf::IfOp.
struct FoldAlwaysTrueMakeRangeMaskPattern
    : public OpRewritePattern<arith::CmpIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpIOp cmpOp,
                                PatternRewriter &rewriter) const override {
    if (cmpOp.getPredicate() != arith::CmpIPredicate::slt &&
        cmpOp.getPredicate() != arith::CmpIPredicate::ult)
      return failure();

    auto makeRange =
        cmpOp.getLhs().getDefiningOp<triton::MakeRangeOp>();
    if (!makeRange)
      return failure();

    DenseIntElementsAttr rhsAttr;
    if (!matchPattern(cmpOp.getRhs(), m_Constant(&rhsAttr)))
      return failure();

    int64_t rangeEnd = static_cast<int64_t>(makeRange.getEnd());
    for (APInt val : rhsAttr.getValues<APInt>()) {
      // For signed slt: rhs must be >= rangeEnd; for unsigned ult same check.
      int64_t bound = static_cast<int64_t>(val.getSExtValue());
      if (bound < rangeEnd)
        return failure();
    }

    // Every element of range [0, N) is < bound, mask is always-true.
    auto resultType = cast<RankedTensorType>(cmpOp.getResult().getType());
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        cmpOp, DenseIntElementsAttr::get(resultType, true));
    return success();
  }
};

/// Lightweight pass that applies FoldAlwaysTrueMakeRangeMaskPattern.
/// Must run before TritonToUnstructuredPass so that GatherOps are created
/// with the already-folded (constant dense<true>) mask.
struct FoldAlwaysTrueMasksPass
    : public PassWrapper<FoldAlwaysTrueMasksPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldAlwaysTrueMasksPass)

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldAlwaysTrueMakeRangeMaskPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns))))
      signalPassFailure();
  }
};

static std::unique_ptr<Pass> createFoldAlwaysTrueMasksPass() {
  return std::make_unique<FoldAlwaysTrueMasksPass>();
}

class TritonToLinalgExperimentalPass
    : public TritonToLinalgExperimentalBase<TritonToLinalgExperimentalPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {

    registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                    linalg::LinalgDialect, affine::AffineDialect,
                    scf::SCFDialect, tensor::TensorDialect,
                    bufferization::BufferizationDialect, memref::MemRefDialect,
                    ttx::TritonTilingExtDialect, tts::TritonStructuredDialect,
                    tptr::TPtrDialect, ptr::PtrDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(createTritonToStructuredPass(enableMakeGatherScatterTensorPtr));

    // Erase dead code and fold constants created during lowering
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());

    // Fold trivially-always-true masks (e.g. make_range < its own upper bound)
    // before TritonToUnstructured creates GatherOps.  Doing so here means the
    // GatherOp will carry a concrete dense<true> mask, enabling GatherConverter
    // to skip dead scf::IfOp emission entirely.
    pm.addPass(createFoldAlwaysTrueMasksPass());

    pm.addPass(createTritonToUnstructuredPass());
    pm.addPass(createTritonArithToLinalgPass(/*tensorPtrToLinalg=*/true));

    pm.addPass(createStructuredToMemrefPass());
    pm.addPass(createUnstructuredToMemrefPass());
    pm.addPass(createTritonPtrToMemrefPass());
    pm.addPass(createTritonToPtrPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createReconcilePtrCastsPass());

    pm.addPass(createRemoveDeadValuesPass());
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    if (enableCollapseShape) {
      // Canonicalizer pass will rewrite tensor.expand_shape(linalg.fill) to
      // linalg.fill(tensor.expand_shape) so we need to run it before
      // collapseShape pass
      pm.addPass(createCollapseShapePass());
    }

    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonToLinalgExperimentalPass() {
  return std::make_unique<TritonToLinalgExperimentalPass>();
}
