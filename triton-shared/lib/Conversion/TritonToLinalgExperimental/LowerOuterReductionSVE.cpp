//===----------------------------------------------------------------------===//
// Lower outer vector reductions and transposes in SVE-width chunks
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/TritonToLinalgExperimental/LowerOuterReductionSVE.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <numeric>

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"

using namespace mlir;

namespace {

/// Turn a vector<MxN> reduction along dimension zero into N independent
/// reductions over native SVE vectors. Each MxN input is sliced into
/// vector<SVE_LANESxN> chunks; a shuffle gathers one column from each chunk,
/// and the columns are accumulated using native-width vector operations.
class LowerOuterReductionPattern
    : public OpRewritePattern<vector::MultiDimReductionOp> {
public:
  LowerOuterReductionPattern(MLIRContext *context, int64_t vscale)
      : OpRewritePattern(context), vscale(vscale) {}

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp op,
                                PatternRewriter &rewriter) const override {
    auto sourceType = op.getSourceVectorType();
    auto destType = dyn_cast<VectorType>(op.getDestType());
    if (sourceType.getRank() != 2 || !destType || destType.getRank() != 1 ||
        sourceType.isScalable() || destType.isScalable())
      return failure();

    ArrayRef<int64_t> shape = sourceType.getShape();
    if (destType.getDimSize(0) != shape[1] ||
        op.getReductionDims().size() != 1 || op.getReductionDims().front() != 0)
      return failure();

    auto maskable = cast<vector::MaskableOpInterface>(op.getOperation());
    if (maskable.isMasked())
      return failure();

    Type elementType = sourceType.getElementType();
    if (!elementType.isIntOrFloat())
      return failure();
    unsigned elementBits = elementType.getIntOrFloatBitWidth();
    int64_t sveLanes = (128 * vscale) / elementBits;
    if (vscale < 1 || sveLanes < 1 || sveLanes > shape[0] ||
        shape[0] % sveLanes != 0)
      return failure();

    Location loc = op.getLoc();
    int64_t columns = shape[1];
    auto flatType = VectorType::get({sveLanes * columns}, elementType);
    SmallVector<Value> columnAccumulators(columns);

    for (int64_t row = 0; row < shape[0]; row += sveLanes) {
      Value slice = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, op.getSource(), ArrayRef<int64_t>{row, 0},
          ArrayRef<int64_t>{sveLanes, columns}, ArrayRef<int64_t>{1, 1});
      Value flat = rewriter.create<vector::ShapeCastOp>(loc, flatType, slice);

      for (int64_t column = 0; column < columns; ++column) {
        SmallVector<int64_t> shuffleMask;
        shuffleMask.reserve(sveLanes);
        for (int64_t lane = 0; lane < sveLanes; ++lane)
          shuffleMask.push_back(lane * columns + column);

        Value columnVector =
            rewriter.create<vector::ShuffleOp>(loc, flat, flat, shuffleMask);
        Value &accumulator = columnAccumulators[column];
        accumulator =
            accumulator
                ? vector::makeArithReduction(rewriter, loc, op.getKind(),
                                             columnVector, accumulator)
                : columnVector;
      }
    }

    Value result = op.getAcc();
    for (int64_t column = 0; column < columns; ++column) {
      Value scalarAcc = rewriter.create<vector::ExtractOp>(
          loc, op.getAcc(), ArrayRef<int64_t>{column});
      Value reduced = rewriter.create<vector::ReductionOp>(
          loc, op.getKind(), columnAccumulators[column], scalarAcc);
      result = rewriter.create<vector::InsertOp>(loc, reduced, result, column);
    }
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  int64_t vscale;
};

/// Fuse the rank-2 transpose produced by InnerReduction with each extracted
/// row reduction. Materializing the vector<NxM> transpose would otherwise
/// scalarize into a large extract/insert network before LLVM reconstructs
/// vectors from it.
class FuseTransposeReductionPattern
    : public OpRewritePattern<vector::ReductionOp> {
public:
  FuseTransposeReductionPattern(MLIRContext *context, int64_t vscale)
      : OpRewritePattern(context), vscale(vscale) {}

  LogicalResult matchAndRewrite(vector::ReductionOp op,
                                PatternRewriter &rewriter) const override {
    auto maskable = cast<vector::MaskableOpInterface>(op.getOperation());
    if (maskable.isMasked())
      return failure();

    auto extract = op.getVector().getDefiningOp<vector::ExtractOp>();
    if (!extract || extract.hasDynamicPosition() ||
        extract.getStaticPosition().size() != 1)
      return failure();
    auto transpose = extract.getVector().getDefiningOp<vector::TransposeOp>();
    if (!transpose || transpose.getPermutation() != ArrayRef<int64_t>({1, 0}))
      return failure();

    VectorType sourceType = transpose.getSourceVectorType();
    ArrayRef<int64_t> shape = sourceType.getShape();
    int64_t column = extract.getStaticPosition().front();
    if (sourceType.getRank() != 2 || sourceType.isScalable() || column < 0 ||
        column >= shape[1] || op.getSourceVectorType().getRank() != 1 ||
        op.getSourceVectorType().getDimSize(0) != shape[0])
      return failure();

    Type elementType = sourceType.getElementType();
    if (!elementType.isIntOrFloat())
      return failure();
    unsigned elementBits = elementType.getIntOrFloatBitWidth();
    int64_t sveLanes = (128 * vscale) / elementBits;
    if (vscale < 1 || sveLanes < 1 || sveLanes > shape[0] ||
        shape[0] % sveLanes != 0)
      return failure();

    Location loc = op.getLoc();
    int64_t columns = shape[1];
    auto flatType = VectorType::get({sveLanes * columns}, elementType);
    Value accumulator;
    SmallVector<int64_t> shuffleMask;
    shuffleMask.reserve(sveLanes);
    for (int64_t lane = 0; lane < sveLanes; ++lane)
      shuffleMask.push_back(lane * columns + column);

    for (int64_t row = 0; row < shape[0]; row += sveLanes) {
      Value slice = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, transpose.getVector(), ArrayRef<int64_t>{row, 0},
          ArrayRef<int64_t>{sveLanes, columns}, ArrayRef<int64_t>{1, 1});
      Value flat = rewriter.create<vector::ShapeCastOp>(loc, flatType, slice);
      Value columnVector =
          rewriter.create<vector::ShuffleOp>(loc, flat, flat, shuffleMask);
      accumulator = accumulator ? vector::makeArithReduction(
                                      rewriter, loc, op.getKind(), columnVector,
                                      accumulator, op.getFastmathAttr())
                                : columnVector;
    }

    Value reduced =
        op.getAcc()
            ? rewriter.create<vector::ReductionOp>(
                  loc, op.getKind(), accumulator, op.getAcc(), op.getFastmath())
            : rewriter.create<vector::ReductionOp>(
                  loc, op.getKind(), accumulator, op.getFastmath());
    rewriter.replaceOp(op, reduced);
    return success();
  }

private:
  int64_t vscale;
};

/// Split a transpose whose long dimension contains multiple SVE vectors into
/// native-width transposes. The native chunks are subsequently rewritten to
/// interleave/deinterleave operations by LowerNativeTransposePattern.
class ChunkTransposePattern : public OpRewritePattern<vector::TransposeOp> {
public:
  ChunkTransposePattern(MLIRContext *context, int64_t vscale)
      : OpRewritePattern(context), vscale(vscale) {}

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    VectorType sourceType = op.getSourceVectorType();
    VectorType resultType = op.getResultVectorType();
    if (sourceType.getRank() != 2 || sourceType.isScalable() ||
        op.getPermutation() != ArrayRef<int64_t>({1, 0}))
      return failure();

    Type elementType = sourceType.getElementType();
    if (!elementType.isIntOrFloat() || vscale < 1)
      return failure();
    int64_t sveLanes = (128 * vscale) / elementType.getIntOrFloatBitWidth();
    ArrayRef<int64_t> shape = sourceType.getShape();

    bool rowsAreChunked =
        shape[1] == 2 && shape[0] > sveLanes && shape[0] % sveLanes == 0;
    bool columnsAreChunked =
        shape[0] == 2 && shape[1] > sveLanes && shape[1] % sveLanes == 0;
    if (!rowsAreChunked && !columnsAreChunked)
      return failure();

    Location loc = op.getLoc();
    Value result = rewriter.create<arith::ConstantOp>(
        loc, resultType, rewriter.getZeroAttr(resultType));
    int64_t longDimension = rowsAreChunked ? shape[0] : shape[1];
    for (int64_t offset = 0; offset < longDimension; offset += sveLanes) {
      SmallVector<int64_t> offsets = rowsAreChunked
                                         ? SmallVector<int64_t>{offset, 0}
                                         : SmallVector<int64_t>{0, offset};
      SmallVector<int64_t> sizes = rowsAreChunked
                                       ? SmallVector<int64_t>{sveLanes, 2}
                                       : SmallVector<int64_t>{2, sveLanes};
      Value slice = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, op.getVector(), offsets, sizes, ArrayRef<int64_t>{1, 1});
      Value transposed = rewriter.create<vector::TransposeOp>(
          loc, slice, ArrayRef<int64_t>{1, 0});
      SmallVector<int64_t> resultOffsets =
          rowsAreChunked ? SmallVector<int64_t>{0, offset}
                         : SmallVector<int64_t>{offset, 0};
      result = rewriter.create<vector::InsertStridedSliceOp>(
          loc, transposed, result, resultOffsets, ArrayRef<int64_t>{1, 1});
    }

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  int64_t vscale;
};

/// Lower the native chunks left by shape-cast lowering to explicit
/// interleave/deinterleave operations. These retain vector semantics through
/// LLVM lowering and map naturally to AArch64 ZIP/UZP instructions.
class LowerNativeTransposePattern
    : public OpRewritePattern<vector::TransposeOp> {
public:
  LowerNativeTransposePattern(MLIRContext *context, int64_t vscale)
      : OpRewritePattern(context), vscale(vscale) {}

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    VectorType sourceType = op.getSourceVectorType();
    VectorType resultType = op.getResultVectorType();
    if (sourceType.getRank() != 2 || sourceType.isScalable() ||
        op.getPermutation() != ArrayRef<int64_t>({1, 0}))
      return failure();

    Type elementType = sourceType.getElementType();
    if (!elementType.isIntOrFloat() || vscale < 1)
      return failure();
    int64_t sveLanes = (128 * vscale) / elementType.getIntOrFloatBitWidth();
    ArrayRef<int64_t> shape = sourceType.getShape();
    if (shape != ArrayRef<int64_t>({sveLanes, 2}) &&
        shape != ArrayRef<int64_t>({2, sveLanes}))
      return failure();

    Location loc = op.getLoc();
    auto flatType = VectorType::get({2 * sveLanes}, elementType);
    Value flatResult;
    if (shape[0] == sveLanes) {
      Value flat =
          rewriter.create<vector::ShapeCastOp>(loc, flatType, op.getVector());
      auto deinterleave = rewriter.create<vector::DeinterleaveOp>(loc, flat);
      SmallVector<int64_t> identity(2 * sveLanes);
      std::iota(identity.begin(), identity.end(), 0);
      flatResult = rewriter.create<vector::ShuffleOp>(
          loc, deinterleave.getRes1(), deinterleave.getRes2(), identity);
    } else {
      Value row0 = rewriter.create<vector::ExtractOp>(loc, op.getVector(), 0);
      Value row1 = rewriter.create<vector::ExtractOp>(loc, op.getVector(), 1);
      flatResult = rewriter.create<vector::InterleaveOp>(loc, row0, row1);
    }

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, resultType,
                                                     flatResult);
    return success();
  }

private:
  int64_t vscale;
};

class LowerOuterReductionSVEPass
    : public LowerOuterReductionSVEBase<LowerOuterReductionSVEPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerOuterReductionPattern>(&getContext(), vscale);
    patterns.add<FuseTransposeReductionPattern>(&getContext(), vscale);
    patterns.add<ChunkTransposePattern>(&getContext(), vscale);
    patterns.add<LowerNativeTransposePattern>(&getContext(), vscale);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::triton::createLowerOuterReductionSVEPass() {
  return std::make_unique<LowerOuterReductionSVEPass>();
}
