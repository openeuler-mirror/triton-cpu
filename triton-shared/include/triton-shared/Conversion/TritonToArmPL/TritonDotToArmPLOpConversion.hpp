//===----------------------------------------------------------------------===//
//
// Pattern that converts tt.dot to tts.armpl_matmul / tts.armpl_gemv.
// The ArmPL ops are side-effecting (no results) — they write into c/y.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITONTOARMPL_TRITONDOTTOARMPLOPCONVERSION_HPP
#define TRITON_CONVERSION_TRITONTOARMPL_TRITONDOTTOARMPLOPCONVERSION_HPP

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

namespace mlir {
namespace triton {

static constexpr int32_t kCblasNoTrans = 111;
static constexpr int32_t kCblasTrans = 112;

static bool isZeroTensor(Value v, bool integers) {
  if (auto splatOp = v.getDefiningOp<triton::SplatOp>()) {
    if (auto constOp = splatOp.getSrc().getDefiningOp<arith::ConstantOp>()) {
      if (auto val = dyn_cast<FloatAttr>(constOp.getValue()))
        return val.getValueAsDouble() == 0.;
      if (auto val = dyn_cast<IntegerAttr>(constOp.getValue()))
        return val.getValue() == 0;
    }
    return false;
  }
  if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
      if (denseAttr.isSplat()) {
        if (integers) return denseAttr.getSplatValue<APInt>().isZero();
        return denseAttr.getSplatValue<APFloat>().isZero();
      }
    }
  }
  return false;
}

static Value getTransposedSource(Value v) {
  auto transOp = v.getDefiningOp<triton::TransOp>();
  if (!transOp) return nullptr;
  auto order = transOp.getOrder();
  if (order.size() == 2 && order[0] == 1 && order[1] == 0)
    return transOp.getSrc();
  return nullptr;
}

static std::pair<Value, Type>
upcastToF32IfNeeded(Value tensor, Location loc, OpBuilder &builder) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto elemType = tensorType.getElementType();
  if (!elemType.isF16() && !elemType.isBF16()) return {tensor, elemType};
  auto f32Type = tensorType.clone(builder.getF32Type());
  return {builder.create<arith::ExtFOp>(loc, f32Type, tensor), elemType};
}

static Value downcastFromF32(Value tensor, Type origElemType, Location loc,
                             OpBuilder &builder) {
  if (origElemType.isF32() || origElemType.isF64()) return tensor;
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto dstType = tensorType.clone(origElemType);
  return builder.create<arith::TruncFOp>(loc, dstType, tensor);
}

struct ArmPLOpPattern : public OpRewritePattern<triton::DotOp> {
  using OpRewritePattern<triton::DotOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::DotOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // ArmPL BLAS routines only support 2D matrices with f32/f64 element types
    // (f16/bf16 are upcast to f32 before the call).  Integer dot products
    // (e.g. int8 x int8 -> int32), float8 dot products, and batched/3D dot
    // operations are left to the standard TritonArithToLinalg lowering.
    // bf16 is also left to the standard lowering because the
    // bf16→f32 upcast generates excessive conversion IR that inflates
    // llc compile time.
    auto aType = cast<RankedTensorType>(op.getA().getType());
    if (aType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "non-2D tensor");
    auto aElemType = aType.getElementType();
    if (!aElemType.isa<FloatType>() || type::isFloat8(aElemType))
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    if (aElemType.isBF16())
      return rewriter.notifyMatchFailure(op, "bf16 not supported by ArmPL");

    Value aSrc = op.getA();
    Value bSrc = op.getB();
    int32_t transA = kCblasNoTrans;
    int32_t transB = kCblasNoTrans;

    if (Value t = getTransposedSource(aSrc)) { transA = kCblasTrans; aSrc = t; }
    if (Value t = getTransposedSource(bSrc)) { transB = kCblasTrans; bSrc = t; }

    auto cType = cast<RankedTensorType>(op.getC().getType());
    int64_t M = cType.getDimSize(0);
    int64_t N = cType.getDimSize(1);

    bool integers = cType.getElementType().isInteger();
    double alpha = 1.0;
    double beta = isZeroTensor(op.getC(), integers) ? 0.0 : 1.0;

    auto [aF32, aOrig] = upcastToF32IfNeeded(aSrc, loc, rewriter);
    auto [bF32, bOrig] = upcastToF32IfNeeded(bSrc, loc, rewriter);
    auto [cF32, cOrig] = upcastToF32IfNeeded(op.getC(), loc, rewriter);

    // Emit the ArmPL op (returns the result tensor)
    Value armplResult;
    if (N == 1) {
      armplResult = rewriter.create<tts::ArmPLGemvOp>(
          loc, cF32.getType(),
          aF32, bF32, cF32,
          rewriter.getI32IntegerAttr(transA),
          rewriter.getF64FloatAttr(alpha),
          rewriter.getF64FloatAttr(beta));
    } else if (M == 1) {
      int32_t gvT = (transB == kCblasNoTrans) ? kCblasTrans : kCblasNoTrans;
      armplResult = rewriter.create<tts::ArmPLGemvOp>(
          loc, cF32.getType(),
          bF32, aF32, cF32,
          rewriter.getI32IntegerAttr(gvT),
          rewriter.getF64FloatAttr(alpha),
          rewriter.getF64FloatAttr(beta));
    } else {
      armplResult = rewriter.create<tts::ArmPLMatmulOp>(
          loc, cF32.getType(),
          aF32, bF32, cF32,
          rewriter.getI32IntegerAttr(transA),
          rewriter.getI32IntegerAttr(transB),
          rewriter.getF64FloatAttr(alpha),
          rewriter.getF64FloatAttr(beta));
    }

    auto resultType = cast<RankedTensorType>(op.getType());
    Value result = armplResult;

    // Downcast if needed
    if (cast<RankedTensorType>(armplResult.getType()).getElementType() != resultType.getElementType())
      result = downcastFromF32(result, resultType.getElementType(), loc, rewriter);

    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONTOARMPL_TRITONDOTTOARMPLOPCONVERSION_HPP