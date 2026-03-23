#ifndef TRITON_CONVERSION_PATTERNS
#define TRITON_CONVERSION_PATTERNS

//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Analysis/MaskAnalysis.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Analysis/PtrAnalysis.h"
#include "triton-shared/Conversion/TritonArithToLinalg/ConversionTools.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton-shared/Utils/Utils.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"

#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include <numeric>
#include <optional>
#include <type_traits>

using namespace mlir;
using namespace triton;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Extract a scalar value from v.
// If v is a scalar, return that directly. Otherwise, parse through operations
// (currently only support splat, sitofp, and truncf) that produce it to
// extract the underlying scalar value. We then reconstruct the chain of
// operations that can produce this constant with the original type. If no
// scalar value can be extracted, a nullptr is returned.
static Value getScalarValue(Value operand, Location loc,
                            ConversionPatternRewriter &rewriter) {
  SmallVector<Operation *> ops;

  auto reconstructScalarValue = [&](Value src) {
    for (auto op = ops.rbegin(); op != ops.rend(); ++op) {
      src = TypeSwitch<Operation *, Value>(*op)
                .Case<arith::SIToFPOp>([&](Operation *op) {
                  auto resType = op->getResults()[0].getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resType)) {
                    resType = shapedType.getElementType();
                  }
                  return rewriter.create<arith::SIToFPOp>(loc, resType, src);
                })
                .Case<arith::TruncFOp>([&](Operation *op) {
                  auto resType = op->getResults()[0].getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resType)) {
                    resType = shapedType.getElementType();
                  }
                  return rewriter.create<arith::TruncFOp>(loc, resType, src);
                })
                .Default([](Operation *op) {
                  llvm_unreachable("unsupported op in generating ");
                  return nullptr;
                });
    }
    return src;
  };

  while (true) {
    if (!dyn_cast<ShapedType>(operand.getType())) {
      return reconstructScalarValue(operand);
    } else if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = dyn_cast<DenseElementsAttr>(op.getValue())) {
        if (!attr.isSplat()) {
          InFlightDiagnostic diag = emitError(loc)
                                    << "other value used in masked load "
                                       "produced by unsupported instruction";
          return nullptr;
        }
        auto elemValue = attr.getSplatValue<Attribute>();
        auto constOp = arith::ConstantOp::materialize(
            rewriter, elemValue, attr.getElementType(), op.getLoc());
        return reconstructScalarValue(constOp.getResult());
      }
    } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
      operand = op.getSrc();
    } else if (auto op = operand.getDefiningOp<arith::SIToFPOp>()) {
      ops.push_back(op.getOperation());
      operand = op.getIn();
    } else if (auto op = operand.getDefiningOp<arith::TruncFOp>()) {
      ops.push_back(op.getOperation());
      operand = op.getIn();
    } else {
      InFlightDiagnostic diag = emitError(loc)
                                << "other value used in masked load produced "
                                   "by unsupported instruction";
      return nullptr;
    }
  }
  return nullptr;
}

static Value getTransposedValue(Value source, const Location loc,
                                ConversionPatternRewriter &rewriter,
                                llvm::ArrayRef<int32_t> order = {}) {
  auto sourceType = cast<RankedTensorType>(source.getType());
  auto sourceRank = sourceType.getRank();

  SmallVector<int64_t> perm(sourceRank);
  SmallVector<int64_t> transposedShape(sourceType.getShape());
  if (order.empty()) {
    std::iota(std::begin(perm), std::end(perm), 0);
    std::swap(perm[sourceRank - 1], perm[sourceRank - 2]);
    std::swap(transposedShape[sourceRank - 1], transposedShape[sourceRank - 2]);
  } else {
    // Use the provided order
    assert(order.size() == sourceRank && "Order size must match source rank");
    for (unsigned i = 0; i < sourceRank; ++i) {
      perm[i] = order[i];
      transposedShape[i] = sourceType.getShape()[order[i]];
    }
  }

  Value transposeInit = rewriter.create<tensor::EmptyOp>(
      loc, transposedShape, sourceType.getElementType());

  Value transpose =
      rewriter.create<linalg::TransposeOp>(loc, source, transposeInit, perm)
          .getResults()[0];

  return transpose;
}

// for IntLike and FloatLike types
static std::optional<unsigned> getBitWidth(Type a) {
  if (auto type = dyn_cast<TensorType>(a)) {
    auto elementType = type.getElementType();
    if (elementType.isIntOrFloat()) {
      return type.getElementType().getIntOrFloatBitWidth();
    }
    return std::nullopt;
  }

  if (a.isIntOrFloat())
    return a.getIntOrFloatBitWidth();

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Op Lowering Patterns
//===----------------------------------------------------------------------===//

namespace {

//-----------------------------
// Begin of monolithic only
//-----------------------------
struct AdvanceConverter : public OpConversionPattern<triton::AdvanceOp> {
  using OpConversionPattern<triton::AdvanceOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::AdvanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallDenseMap<Value, PtrState> knownPtrs;
    PtrState pointerState;
    PtrAnalysis::rewriteAdvanceOp(op, rewriter, knownPtrs);
    return success();
  }
};

struct MakeTensorPtrConverter
    : public OpConversionPattern<triton::MakeTensorPtrOp> {
  using OpConversionPattern<triton::MakeTensorPtrOp>::OpConversionPattern;

  void populateVectorAsIndex(SmallVector<OpFoldResult> &vec,
                             Operation::operand_range ops,
                             ConversionPatternRewriter &rewriter,
                             Location loc) const {
    for (auto opnd : ops) {
      if (isa<IntegerType>(opnd.getType())) {
        auto castOp = rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getIndexType(), opnd);
        vec.push_back(castOp.getResult());
      } else {
        assert(isa<IndexType>(opnd.getType()));
        vec.push_back(opnd);
      }
    }
  }

  LogicalResult
  matchAndRewrite(triton::MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    PtrState pointerState;

    auto orderSize = op.getOrder().size();
    if (orderSize > 1) {
      for (auto [first, second] :
           llvm::zip(op.getOrder().slice(0, orderSize - 2),
                     op.getOrder().slice(1, orderSize - 1))) {
        assert(first == second + 1 &&
               "Currently only support default order on block pointers");
      }
    }

    pointerState.source = rewriter.getRemappedValue(op.getBase());
    populateVectorAsIndex(pointerState.offsets, op.getOffsets(), rewriter, loc);
    populateVectorAsIndex(pointerState.strides, op.getStrides(), rewriter, loc);

    SmallVector<Value> newOffsets;
    for (auto [offset, stride] :
         llvm::zip(pointerState.offsets, pointerState.strides)) {
      auto mulOp = rewriter.create<arith::MulIOp>(loc, offset.get<Value>(),
                                                  stride.get<Value>());
      newOffsets.push_back(mulOp.getResult());
    }

    pointerState.offsets.clear();

    for (auto offset : newOffsets) {
      pointerState.offsets.push_back(offset);
    }

    ArrayRef<int64_t> resultShape;
    auto pointerType =
        cast<mlir::triton::PointerType>(op.getResult().getType());
    if (auto shapedType = dyn_cast<ShapedType>(pointerType.getPointeeType())) {
      resultShape = shapedType.getShape();
      for (auto dim_size : resultShape) {
        pointerState.sizes.push_back(
            IntegerAttr::get(IntegerType::get(op.getContext(), 64), dim_size));
      }
    } else {
      // scalar pointer, should produce a one dimensional memref
      SmallVector<int64_t> scalarShape(1, 1);
      resultShape = scalarShape;
      assert(pointerState.getRank() == 1);
    }

    auto castOp = pointerState.createCastOp(resultShape, loc, rewriter);
    rewriter.replaceOp(op, castOp.getResult());
    return success();
  }
};

struct LegacyAddPtrConverter : public OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern<triton::AddPtrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallDenseMap<Value, PtrState> knownPtrs;
    PtrAnalysis::rewriteAddptrOp(op, rewriter, knownPtrs);
    return success();
  }
};

struct LoadConverter : public OpConversionPattern<triton::LoadOp> {
private:
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

  void createSideBySideCopies(Value block1, Value block2, Value dst,
                              Location loc,
                              ConversionPatternRewriter &rewriter) const {

    auto zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));

    auto one =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    Value block1Row = rewriter.create<memref::DimOp>(loc, block1, 0);
    Value block1Col = rewriter.create<memref::DimOp>(loc, block1, 1);

    Value block2Row = rewriter.create<memref::DimOp>(loc, block2, 0);
    Value block2Col = rewriter.create<memref::DimOp>(loc, block2, 1);

    auto block1Dst =
        rewriter.create<memref::SubViewOp>(loc, dst, /* offsets */
                                           ValueRange{zero, zero},
                                           /* sizes */
                                           ValueRange{block1Row, block1Col},
                                           /* strides */
                                           ValueRange{one, one});

    auto block2Dst =
        rewriter.create<memref::SubViewOp>(loc, dst,
                                           /* offsets */
                                           ValueRange{zero, block1Col},
                                           /* sizes */
                                           ValueRange{block2Row, block2Col},
                                           /* strides */
                                           ValueRange{one, one});

    rewriter.create<memref::CopyOp>(loc, block1, block1Dst);
    rewriter.create<memref::CopyOp>(loc, block2, block2Dst);
  }

  void createStackedCopies(Value block1, Value block2, Value dst, Location loc,
                           ConversionPatternRewriter &rewriter) const {

    auto zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    auto one =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    Value block1Row = rewriter.create<memref::DimOp>(loc, block1, 0);
    Value block1Col = rewriter.create<memref::DimOp>(loc, block1, 1);

    Value block2Row = rewriter.create<memref::DimOp>(loc, block2, 0);
    Value block2Col = rewriter.create<memref::DimOp>(loc, block2, 1);

    auto block1Dst =
        rewriter.create<memref::SubViewOp>(loc, dst, /* offsets */
                                           ValueRange{zero, zero},
                                           /* sizes */
                                           ValueRange{block1Row, block1Col},
                                           /* strides */
                                           ValueRange{one, one});

    auto block2Dst =
        rewriter.create<memref::SubViewOp>(loc, dst,
                                           /* offsets */
                                           ValueRange{block1Row, zero},
                                           /* sizes */
                                           ValueRange{block2Row, block2Col},
                                           /* strides */
                                           ValueRange{one, one});

    rewriter.create<memref::CopyOp>(loc, block1, block1Dst);
    rewriter.create<memref::CopyOp>(loc, block2, block2Dst);
  }

public:
  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptr = adaptor.getPtr();
    auto mask = op.getMask();
    auto other = op.getOther();
    auto loc = op.getLoc();

    // 0. Shortcut for scalar loads
    if (!isa<ShapedType>(op.getResult().getType())) {
      auto sMemRef = PtrAnalysis::getScalarMemRef(op.getPtr(), adaptor.getPtr(),
                                                  loc, rewriter);

      auto zeroMap = AffineMap::getConstantMap(0, rewriter.getContext());
      auto loadOp = rewriter.create<affine::AffineLoadOp>(
          op.getLoc(), sMemRef, zeroMap, std::nullopt);
      rewriter.replaceOp(op, loadOp.getResult());
      return success();
    }

    // 1. Simple case where no mask is used.
    auto type = dyn_cast<MemRefType>(ptr.getType());
    if (!type) {
      // Seen when implicit broadcasting is done late in a chain of operations.
      // The workaround is to broadcast the pointers early in the address
      // calculation. A proper fix is complicated, but at least we can provide a
      // better error message.
      return rewriter.notifyMatchFailure(
          op, "LoadOp expects a memref, not a memref of pointers");
    }

    auto tensorType =
        RankedTensorType::get(type.getShape(), type.getElementType());
    auto alloc = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(type.getShape(), type.getElementType()));

    if (!mask) {
      assert(!other && "other value used in non-masked load");
      if (auto unrealizedCast =
              ptr.getDefiningOp<UnrealizedConversionCastOp>()) {
        if (auto wrapType = unrealizedCast->getAttrOfType<StringAttr>(
                ModuloState::WraparoundAttr)) {

          auto memrefs = unrealizedCast.getOperands();
          auto block1 = memrefs[0];
          auto block2 = memrefs[1];

          if (wrapType.getValue() == ModuloState::WraparoundSideBySide) {
            createSideBySideCopies(block1, block2, alloc, loc, rewriter);
          } else if (wrapType.getValue() == ModuloState::WraparoundStacked) {
            createStackedCopies(block1, block2, alloc, loc, rewriter);
          } else {
            llvm_unreachable("unexpected wraparound type");
          }
        } else {
          llvm_unreachable("unexpected unrealized cast op");
        }

      } else {
        rewriter.create<memref::CopyOp>(loc, ptr, alloc);
      }

      Value tensor = rewriter.create<bufferization::ToTensorOp>(
          loc, tensorType, alloc, true /* restrict */, true /* writable */);
      rewriter.replaceOp(op, tensor);

      return success();
    }

    // 2. Continuous masked loads.
    // Analyze the mask operand to determine at runtime the size of the data we
    // are moving.
    MaskState mstate;
    auto isContMask = mstate.parse(mask, loc, rewriter);

    if (isContMask.failed()) {
      return rewriter.notifyMatchFailure(
          op, "Cannot lower continuous masked loads");
    }

    // fill load destination with other value
    if (other) {
      auto scalarOther = getScalarValue(other, loc, rewriter);
      assert(scalarOther && "other value used in masked load produced by "
                            "unsupported instruction");

      // For each dimension check if mstate.dims[i] < shape[i], or-accumulate
      // the result
      auto shape = type.getShape();
      auto accBase =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(false))
              .getResult();
      for (size_t i = 0; i < type.getShape().size(); i++) {
        auto shapei = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIndexAttr(shape[i]));

        Value dimi = dyn_cast<Value>(mstate.dims[i]);
        if (!dimi) {
          dimi = rewriter.create<arith::ConstantOp>(
              loc, cast<IntegerAttr>(mstate.dims[i].get<Attribute>()));
        }

        auto cmpOp = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, dimi, shapei);
        accBase = rewriter.create<arith::OrIOp>(loc, accBase, cmpOp.getResult())
                      .getResult();
      }

      // condition the memset on the or-accumulation
      // initialize with padding prior to CopyOp
      rewriter.create<scf::IfOp>(
          loc, accBase, [&](OpBuilder &builder, Location loc) {
            builder.create<linalg::FillOp>(loc, ValueRange{scalarOther},
                                           ValueRange{alloc});
            builder.create<scf::YieldOp>(loc);
          });
    }

    if (auto unrealizedCast = ptr.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (auto wrapType = unrealizedCast->getAttrOfType<StringAttr>(
              ModuloState::WraparoundAttr)) {

        auto memrefs = unrealizedCast.getOperands();
        auto block1 = memrefs[0];
        auto block2 = memrefs[1];

        if (wrapType.getValue() == ModuloState::WraparoundSideBySide) {
          auto [subview1, subview2] =
              mstate.getSideBySideSubviews(block1, block2, loc, rewriter);

          createSideBySideCopies(subview1, subview2, alloc, loc, rewriter);
        } else if (wrapType.getValue() == ModuloState::WraparoundStacked) {
          auto [subview1, subview2] =
              mstate.getStackedSubviews(block1, block2, loc, rewriter);

          createStackedCopies(subview1, subview2, alloc, loc, rewriter);
        } else {
          llvm_unreachable("unexpected wraparound type");
        }

      } else {
        llvm_unreachable("unexpected unrealized cast op");
      }

    } else {
      memref::SubViewOp srcSubview = mstate.getSubview(ptr, loc, rewriter);
      memref::SubViewOp dstSubview = mstate.getSubview(alloc, loc, rewriter);
      rewriter.create<memref::CopyOp>(loc, srcSubview, dstSubview);
    }

    Value tensor = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorType, alloc, true /* restrict */, true /* writable */);
    rewriter.replaceOp(op, tensor);

    return success();
  }
};

struct StoreConverter : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptr = adaptor.getPtr();
    auto val = adaptor.getValue();
    auto mask = op.getMask();
    auto loc = op.getLoc();

    // 0. Shortcut for scalar stores
    if (!isa<ShapedType>(val.getType())) {
      auto sMemRef =
          PtrAnalysis::getScalarMemRef(op.getPtr(), ptr, loc, rewriter);

      auto index =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0))
              .getResult();
      auto zeroMap = AffineMap::getConstantMap(0, rewriter.getContext());
      rewriter.create<affine::AffineStoreOp>(loc, val, sMemRef, zeroMap,
                                             std::nullopt);
      rewriter.eraseOp(op);
      return success();
    }

    // 1. Simple case where no mask is used.
    if (!mask) {
      auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
          loc, val, ptr);
      storeOp.setWritable(true);
      rewriter.eraseOp(op);
      return success();
    }

    // 2. Continuous masked stores.
    // Analyze the mask operand to determine at runtime the size of the data we
    // are moving.
    MaskState mstate;
    auto isContMask = mstate.parse(mask, loc, rewriter);

    if (isContMask.failed())
      return failure();

    auto srcSlice = mstate.getExtractSlice(val, loc, rewriter);
    auto dstSubview = mstate.getSubview(ptr, loc, rewriter);

    auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
        loc, srcSlice, dstSubview);
    storeOp.setWritable(true);
    rewriter.eraseOp(op);

    return success();
  }
};

struct LoopConverter : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallDenseMap<Value, PtrState> knownPtrs;
    PtrAnalysis::IndexMapSet
        levelToBlockArgIndex; // level -> set of block arg index to be replaced

    PtrAnalysis::rewriteForOp(op, rewriter, levelToBlockArgIndex, 0, knownPtrs);
    return success();
  }
};

struct YieldConverter : public OpConversionPattern<scf::YieldOp> {
  using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
    return success();
  }
};

// Remove all Meta ops except for AddPtr which is handled by AddPtrConverter.
// Use benefit == 10 to ensure that this pattern always takes precedence over
// other patterns.
struct MetaOpConverter : public RewritePattern {
private:
  // UseAnalysis will tag operations whose results are used only as meta-data
  // with "MetaUse" tag.
  bool isMetaUse(Operation *op) const { return op->hasAttr("MetaUse"); }

public:
  MetaOpConverter(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/10, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {

    if (isa<triton::AddPtrOp>(op)) {
      return rewriter.notifyMatchFailure(op,
                                         "AddPtrOp will be handled separately");
    }

    if (isMetaUse(op)) {
      rewriter.eraseOp(op);
      return success();
    }

    return rewriter.notifyMatchFailure(op, "requires meta ops");
  }
};

struct UnrealizedCastConverter
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern<UnrealizedConversionCastOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

//-----------------------------
// End of monolithic only
//-----------------------------

struct SplatConverter : public OpConversionPattern<triton::SplatOp> {
  using OpConversionPattern<triton::SplatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opType = cast<TensorType>(op.getType());
    auto loc = op.getLoc();

    auto init = rewriter.create<tensor::EmptyOp>(loc, opType.getShape(),
                                                 opType.getElementType());

    auto filledTensor =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{adaptor.getSrc()},
                                    ValueRange{init})
            .result();

    rewriter.replaceOp(op, filledTensor);
    return success();
  }
};

struct BroadcastConverter : public OpConversionPattern<triton::BroadcastOp> {
private:
  using OpConversionPattern<triton::BroadcastOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    assert(op->getNumResults() == 1 && "code assumes single result!");
    RankedTensorType sourceType =
        cast<RankedTensorType>(adaptor.getSrc().getType());
    RankedTensorType resultType = cast<RankedTensorType>(op.getType());
    auto elementType = resultType.getElementType();
    size_t resultRank = resultType.getRank();

    SmallVector<AffineMap> indexingMaps;
    indexingMaps.reserve(op->getNumOperands() + op->getNumResults());

    indexingMaps.push_back(getBroadcastAffineMap(
        op->getContext(), sourceType.getShape(), resultType.getShape()));
    indexingMaps.append(op->getNumResults(),
                        rewriter.getMultiDimIdentityMap(resultRank));

    assert(op->getNumResults() == 1 && "code assumes single result!");
    auto init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                 elementType);

    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, op->getResultTypes(), ValueRange{adaptor.getSrc()},
        ValueRange{init}, indexingMaps, getNParallelLoopsAttrs(resultRank),
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          Value opResult = blockArgs[0];
          nestedBuilder.create<linalg::YieldOp>(loc, opResult);
        });

    linalgOp->setAttr("broadcastDims",
                      rewriter.getDenseI64ArrayAttr(
                          getBroadcastDims(sourceType, resultType)));

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};

struct ExpandDimsConverter : public OpConversionPattern<triton::ExpandDimsOp> {
  using OpConversionPattern<triton::ExpandDimsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getSrc();
    auto srcRank = cast<RankedTensorType>(src.getType()).getRank();
    auto resType = cast<RankedTensorType>(op->getResultTypes()[0]);
    SmallVector<ReassociationIndices> reassoc;
    int64_t c = 0;
    for (int64_t i = 0; i < srcRank; i++) {
      ReassociationIndices g;
      g.push_back(c++);
      if (op.getAxis() == i) {
        g.push_back(c++);
      } else if (op.getAxis() == i + 1 && i == srcRank - 1) {
        g.push_back(c++);
      }
      reassoc.push_back(g);
    }

    auto expandShapeOp = rewriter.create<tensor::ExpandShapeOp>(
        op.getLoc(), resType, src, reassoc);

    rewriter.replaceOp(op, expandShapeOp.getResult());
    return success();
  }
};

struct TransposeConverter : public OpConversionPattern<triton::TransOp> {
  using OpConversionPattern<triton::TransOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto res = getTransposedValue(adaptor.getSrc(), op.getLoc(), rewriter,
                                  op.getOrder());
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct MakeRangeConverter : public OpConversionPattern<triton::MakeRangeOp> {
  using OpConversionPattern<triton::MakeRangeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = cast<TensorType>(op.getResult().getType());
    auto shape = type.getShape();
    auto elementType = type.getElementType();
    auto context = rewriter.getContext();

    assert(type.getShape().size() == 1 &&
           type.getElementType().getIntOrFloatBitWidth() == 32 &&
           "make range can only return 1D int32 tensor");

    SmallVector<AffineMap> indexingMaps{AffineMap::get(
        /* dimCount */ 1, /* symbolCount */ 0,
        SmallVector<AffineExpr>{mlir::getAffineDimExpr(0, context)}, context)};

    auto init = rewriter.create<tensor::EmptyOp>(loc, shape, elementType);
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, op->getResultTypes(), /* operands */ ValueRange{},
        ValueRange{init}, indexingMaps, getNParallelLoopsAttrs(1),
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          Value index = nestedBuilder.create<linalg::IndexOp>(loc, 0);
          Value res = nestedBuilder.create<arith::IndexCastOp>(
              loc, type.getElementType(), index);
          if (op.getStart()) {
            auto start = rewriter.create<mlir::arith::ConstantIntOp>(
                op.getLoc(), op.getStart(),
                type.getElementType().getIntOrFloatBitWidth());
            res = nestedBuilder.create<arith::AddIOp>(loc, res, start);
          }
          nestedBuilder.create<linalg::YieldOp>(loc, res);
        });

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};

struct AssertConverter : public OpConversionPattern<triton::AssertOp> {
  using OpConversionPattern<triton::AssertOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value condVal = op.getCondition();

    auto assertMessage =
        llvm::formatv("Assertion `{0}` failed", op.getMessage());
    // The condition can only be I1 or I1Tensor (integer or tensor) from
    // TritonOps.td.
    // Tensors will always be RankedTensorType.
    if (isa<mlir::IntegerType>(condVal.getType())) {
      // handle scalar case
      rewriter.create<mlir::cf::AssertOp>(op.getLoc(), condVal,
                                          assertMessage.str());
    } else if (auto tensorType =
                   dyn_cast<RankedTensorType>(condVal.getType())) {
      // handle tensor case
      int64_t rank = tensorType.getRank();
      // create identity mapping for access pattern
      SmallVector<AffineMap, 3> indexingMaps{
          AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext())};
      // loops do not depend on each other
      SmallVector<utils::IteratorType, 3> iteratorTypes(
          rank, utils::IteratorType::parallel);
      rewriter.create<linalg::GenericOp>(
          op.getLoc(), TypeRange{}, condVal, ValueRange{},
          ArrayRef<AffineMap>{indexingMaps},
          ArrayRef<utils::IteratorType>{iteratorTypes},
          [&](OpBuilder &b, Location loc, ValueRange args) {
            // obtain the element in the tensor
            Value element = args[0];
            // make a cf.assert for the current element
            b.create<mlir::cf::AssertOp>(loc, element, assertMessage.str());
            b.create<linalg::YieldOp>(loc);
          });
    } else {
      op.emitError("Unexpected type in triton::AssertOp");
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct BitcastConverter : public OpConversionPattern<triton::BitcastOp> {
  using OpConversionPattern<triton::BitcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // arith::bitcast does not support casting pointers
    if (triton::isPtrTypeLike(op.getType())) {
      return failure();
    }

    auto arithBitcast = rewriter.create<arith::BitcastOp>(
        op.getLoc(), op.getType(), op.getOperand());

    rewriter.replaceOp(op, arithBitcast.getResult());
    return success();
  }
};

struct CallConverter : public OpConversionPattern<triton::CallOp> {
  using OpConversionPattern<triton::CallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> args = adaptor.getOperands();

    // We need to pass extra arguments added by addProgramInfo which are
    // num_programs and program_ids
    if (FuncOp parentFunc = op->getParentOfType<triton::FuncOp>()) {
      SymbolRefAttr calleeAttr = op.getCalleeAttr();
      StringRef calleeName = calleeAttr.getRootReference();

      if (ModuleOp module = op->getParentOfType<ModuleOp>()) {
        if (FuncOp calleeFunc = module.lookupSymbol<FuncOp>(calleeName)) {
          size_t argsNeed = calleeFunc.getFunctionType().getInputs().size();
          Block &entryBlock = parentFunc.front();
          auto parentInputs = entryBlock.getArguments();
          size_t argsParent = parentInputs.size();

          if (argsNeed > args.size()) {
            int missing = argsNeed - args.size();
            int missingArgsStart = argsParent - missing;
            for (int i = 0; i < missing; i++) {
              args.push_back(parentInputs[missingArgsStart + i]);
            }
          }
        }
      }
    }

    auto call = rewriter.create<func::CallOp>(op.getLoc(), op.getCallee(),
                                              op.getResultTypes(), args);

    if (!call) {
      op.emitError("Failed to create func::CallOp");
      return failure();
    }

    rewriter.replaceOp(op, call);
    return success();
  }
};

struct FpToFpConverter : public OpConversionPattern<triton::FpToFpOp> {
  using OpConversionPattern<triton::FpToFpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::FpToFpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto roundingMode = triton::RoundingMode::RTNE; // default

    auto roundingModeAttr = op.getRounding();
    if (roundingModeAttr.has_value()) {
      roundingMode = roundingModeAttr.value();
    }

    assert(roundingMode != triton::RoundingMode::RTZ &&
           "Rounding Towards Zero is not supported");

    Type resultType = op.getResult().getType();

    auto operandWidth = getBitWidth(op.getOperand().getType());
    auto resultWidth = getBitWidth(resultType);

    assert(operandWidth.has_value() && resultWidth.has_value() &&
           "Not a float-like operand or result");

    if (operandWidth.value() > resultWidth.value()) {
      Value truncatedValue = rewriter.create<arith::TruncFOp>(
          op.getLoc(), resultType, op.getOperand());
      rewriter.replaceOp(op, truncatedValue);
      return success();
    }

    Value extendedValue = rewriter.create<arith::ExtFOp>(
        op.getLoc(), resultType, op.getOperand());
    rewriter.replaceOp(op, extendedValue);

    return success();
  }
};

struct ClampConverter : public OpConversionPattern<triton::ClampFOp> {
  using OpConversionPattern<triton::ClampFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ClampFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    bool propagateNan = op.getPropagateNan() == triton::PropagateNan::ALL;

    Location loc = op.getLoc();
    Value x = adaptor.getOperands()[0];
    Value min = adaptor.getOperands()[1];
    Value max = adaptor.getOperands()[2];

    Value clamp;

    if (propagateNan) {
      Value maxMin = rewriter.create<arith::MaximumFOp>(loc, x, min);
      clamp = rewriter.create<arith::MinimumFOp>(loc, maxMin, max);
    } else {
      Value maxMin = rewriter.create<arith::MaxNumFOp>(loc, x, min);
      clamp = rewriter.create<arith::MinNumFOp>(loc, maxMin, max);
    }
    rewriter.replaceOp(op, clamp);

    return success();
  }
};

struct PreciseSqrtConverter
    : public OpConversionPattern<triton::PreciseSqrtOp> {
  using OpConversionPattern<triton::PreciseSqrtOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::PreciseSqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto replacement =
        rewriter.create<math::SqrtOp>(op.getLoc(), adaptor.getOperands());

    rewriter.replaceOp(op, replacement);
    return success();
  }
};

struct PreciseDivConverter : public OpConversionPattern<triton::PreciseDivFOp> {
  using OpConversionPattern<triton::PreciseDivFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::PreciseDivFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto replacement =
        rewriter.create<arith::DivFOp>(op.getLoc(), adaptor.getOperands());

    rewriter.replaceOp(op, replacement);
    return success();
  }
};

struct CatConverter : public OpConversionPattern<triton::CatOp> {
  using OpConversionPattern<triton::CatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto replacement = rewriter.create<tensor::ConcatOp>(
        op.getLoc(), 0 /* concat dimension */, adaptor.getOperands());

    rewriter.replaceOp(op, replacement);

    return success();
  }
};

struct SplitConverter : public OpConversionPattern<triton::SplitOp> {
  using OpConversionPattern<triton::SplitOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getOperand();
    auto inputType = cast<RankedTensorType>(input.getType());

    Type resultType = op.getResults().front().getType();
    auto resultTensor = cast<RankedTensorType>(resultType);
    auto shape = inputType.getShape();

    SmallVector<OpFoldResult> offsets(shape.size(), rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(shape.size(), rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes = llvm::to_vector(
        llvm::map_range(shape, [&](int64_t dim) -> OpFoldResult {
          return rewriter.getIndexAttr(dim);
        }));

    SmallVector<Value> results;

    for (int i = 0; i < 2; ++i) {
      offsets.pop_back();
      sizes.pop_back();

      offsets.push_back(rewriter.getIndexAttr(i));
      sizes.push_back(rewriter.getIndexAttr(1));
      Value slice = rewriter.create<tensor::ExtractSliceOp>(
          loc, resultTensor, input, offsets, sizes, strides);
      results.push_back(slice);
    }

    rewriter.replaceOp(op, results);
    return success();
  }
};

struct JoinConverter : public OpConversionPattern<triton::JoinOp> {
  using OpConversionPattern<triton::JoinOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::JoinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange inputs = op.getOperands();

    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    auto loc = op.getLoc();
    Value result = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    auto shape = resultType.getShape();

    SmallVector<OpFoldResult> offsets(shape.size(), rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(shape.size(), rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes = llvm::to_vector(
        llvm::map_range(shape, [&](int64_t dim) -> OpFoldResult {
          return rewriter.getIndexAttr(dim);
        }));

    for (int i = 0; i < 2; ++i) {
      offsets.pop_back();
      sizes.pop_back();

      offsets.push_back(rewriter.getIndexAttr(i));
      sizes.push_back(rewriter.getIndexAttr(1));
      result = rewriter.create<tensor::InsertSliceOp>(loc, inputs[i], result,
                                                      offsets, sizes, strides);
    }

    rewriter.replaceOp(op, result);

    return success();
  }
};

struct MulHiUIOpConverter : public OpConversionPattern<triton::MulhiUIOp> {
  using OpConversionPattern<triton::MulhiUIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MulhiUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto mulResult =
        rewriter.create<arith::MulUIExtendedOp>(loc, adaptor.getOperands());
    rewriter.replaceOp(op, mulResult.getHigh());

    return success();
  }
};

struct MatmulConverter : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern<triton::DotOp>::OpConversionPattern;

  // true means tensor elements are zeros
  // false means not zero or it cannot be determined
  bool isZeroTensor(Value &v, bool integers) const {
    if (auto splatOp = v.getDefiningOp<triton::SplatOp>()) {
      if (auto constOp = splatOp.getSrc().getDefiningOp<arith::ConstantOp>()) {
        if (auto val = dyn_cast<FloatAttr>(constOp.getValue())) {
          return val.getValueAsDouble() == 0.;
        }
        if (auto val = dyn_cast<IntegerAttr>(constOp.getValue())) {
          return val.getValue() == 0;
        }
      }
      return false;
    }

    if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
        if (denseAttr.isSplat()) {
          if (integers)
            return denseAttr.getSplatValue<APInt>().isZero();
          return denseAttr.getSplatValue<APFloat>().isZero();
        }
      }
    }

    return false;
  }

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto opa = op.getA();
    auto opb = op.getB();
    auto opc = op.getC();

    auto dstType = cast<RankedTensorType>(op.getType());
    auto elementType = dstType.getElementType();
    bool integers = elementType.isInteger();
    bool skipC = isZeroTensor(opc, integers);

    // Initialize result tensor with zeros
    auto init =
        rewriter.create<tensor::EmptyOp>(loc, dstType.getShape(), elementType);
    TypedAttr constantAttr =
        integers
            ? static_cast<TypedAttr>(rewriter.getIntegerAttr(elementType, 0))
            : static_cast<TypedAttr>(rewriter.getFloatAttr(elementType, 0));

    auto zero = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), elementType, constantAttr);

    auto zeroes =
        rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{init})
            .result();

    Value res;
    auto rank = dstType.getRank();

    if (rank == 2) {
      // Standard matmul
      res = rewriter
                .create<linalg::MatmulOp>(loc, ValueRange{opa, opb},
                                          ValueRange{zeroes})
                .getResult(0);
    } else if (rank == 3) {
      // Batched matmul
      res = rewriter
                .create<linalg::BatchMatmulOp>(loc, ValueRange{opa, opb},
                                               ValueRange{zeroes})
                .getResult(0);
    } else {
      return rewriter.notifyMatchFailure(
          op, "Only 2D or 3D inputs supported for tt.dot lowering");
    }

    // Add C if it's not zero
    if (!skipC) {
      if (integers) {
        res = rewriter.create<arith::AddIOp>(loc, opc, res);
      } else {
        res = rewriter.create<arith::AddFOp>(loc, opc, res);
      }
    }

    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ReduceConverter : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern<triton::ReduceOp>::OpConversionPattern;

private:
  llvm::SmallVector<Operation *> getRedOps(triton::ReduceOp redOp) const {
    auto reduceBlock = redOp.getBody();
    return llvm::map_to_vector(reduceBlock->without_terminator(),
                               [](Operation &op) { return &op; });
  }

  bool requiresF32Conversion(const Type elemType, Operation *redOp) const {
    // Only if it is a binaryOp and types mismatch
    if (isa<arith::AddFOp>(redOp) || isa<arith::SubFOp>(redOp) ||
        isa<arith::MulFOp>(redOp) || isa<arith::DivFOp>(redOp) ||
        isa<triton::ReduceReturnOp>(redOp)) {
      auto lhsType = elemType;
      auto rhsType = (isa<triton::ReduceReturnOp>(redOp)) ?
                        redOp->getOperand(0).getType():
                        redOp->getOperand(1).getType();
      return (lhsType != rhsType) && (isa<FloatType>(rhsType)) &&
             rhsType.getIntOrFloatBitWidth() == 32;
    }
  }

  Value getRedElement(Value lhs, Value rhs, const Location loc,
                      Operation *redOp, OpBuilder &b,
                      const bool convertLhsToF32Precision) const {
    return llvm::TypeSwitch<Operation *, Value>(redOp)
        .Case([&](arith::AddFOp) {
          if (convertLhsToF32Precision) {
            lhs = b.create<arith::ExtFOp>(loc, Float32Type::get(b.getContext()),
                                          lhs);
          }
          return b.create<arith::AddFOp>(loc, lhs, rhs);
        })
        .Case<arith::AddIOp, arith::MaximumFOp, arith::MaxNumFOp,
              arith::MinimumFOp, arith::MinNumFOp, arith::MinSIOp,
              arith::MinUIOp, arith::MaxSIOp, arith::MaxUIOp, arith::SubIOp,
              arith::MulIOp, arith::DivSIOp, arith::SubFOp, arith::MulFOp,
              arith::DivFOp>([&](auto redOp) {
          return b.create<decltype(redOp)>(loc, lhs, rhs);
        })
        .Case<arith::CmpFOp, arith::CmpIOp>([&](auto redOp) {
          return b.create<decltype(redOp)>(loc, redOp.getPredicate(), lhs, rhs);
        })
        .Default([](Operation *op) {
          op->dump();
          llvm_unreachable("Reduction op not yet supported");
          return nullptr;
        });
  }

  static Value getFirstElementAlongAxis(PatternRewriter &rewriter, Value tensor,
                                        int64_t axis, Location loc) {
    auto rankedTy = cast<RankedTensorType>(tensor.getType());
    SmallVector<Value> idxs(rankedTy.getRank(),
                            rewriter.create<arith::ConstantIndexOp>(loc, axis));
    // Extract the first element along reduction axis
    return rewriter.create<tensor::ExtractOp>(loc, rankedTy.getElementType(),
                                              tensor, idxs);
  }

  static Operation *findFirstAccumulatorUserAt(Block &block,
                                               unsigned accArgIdx) {
    Value accArg = block.getArgument(accArgIdx);
    for (Operation &opInner : block.getOperations()) {
      if(isa<triton::ReduceReturnOp>(&opInner))
          return &opInner;
      if (&opInner == block.getTerminator())
        break;
      for (Value operand : opInner.getOperands())
        if (operand == accArg)
          return &opInner;
    }
    return nullptr;
  }

  arith::ConstantOp getRedBaseConstOp(ConversionPatternRewriter &rewriter,
                                      Operation *redOp,
                                      Type constantType) const {
    const int64_t bitWidth = constantType.getIntOrFloatBitWidth();

    auto attr =
        llvm::TypeSwitch<Operation *, TypedAttr>(redOp)
            .Case([&](arith::AddFOp) {
              return rewriter.getFloatAttr(constantType, 0.f);
            })
            .Case([&](arith::AddIOp) {
              return rewriter.getIntegerAttr(constantType, 0);
            })
            .Case<arith::MaximumFOp, arith::MaxNumFOp>([&](auto) {
              return rewriter.getFloatAttr(
                  constantType, -std::numeric_limits<float>::infinity());
            })
            .Case<arith::MinimumFOp, arith::MinNumFOp>([&](auto) {
              return rewriter.getFloatAttr(
                  constantType, std::numeric_limits<float>::infinity());
            })
            .Case([&](arith::MinSIOp) {
              return rewriter.getIntegerAttr(constantType,
                                             llvm::maxIntN(bitWidth));
            })
            .Case([&](arith::MinUIOp) {
              return rewriter.getIntegerAttr(constantType,
                                             llvm::maxUIntN(bitWidth));
            })
            .Case([&](arith::MaxSIOp) {
              return rewriter.getIntegerAttr(constantType,
                                             llvm::minIntN(bitWidth));
            })
            .Case<arith::MaxUIOp, arith::XOrIOp>(
                [&](auto) { return rewriter.getIntegerAttr(constantType, 0); })
            .Case([&](arith::MulFOp) {
              return rewriter.getFloatAttr(constantType, 1.f);
            })
            .Case<arith::MulIOp, arith::AndIOp>(
                [&](auto) { return rewriter.getIntegerAttr(constantType, 1); })
            .Case([&](arith::OrIOp) {
              return rewriter.getIntegerAttr(constantType, 0);
            })
            .Case<triton::ReduceReturnOp>(
                [&](auto) { return rewriter.getIntegerAttr(constantType, 0); })
            .Default([&](Operation *op) { return nullptr; });

    if (!attr) {
      TypedAttr attr =
        constantType.isIntOrIndex()
            ? cast<TypedAttr>(rewriter.getIntegerAttr(constantType, -1))
            : cast<TypedAttr>(rewriter.getFloatAttr(constantType, -1.0f));

      auto constOp = rewriter.create<arith::ConstantOp>(redOp->getLoc(),
                                                        constantType, attr);
      constOp->setAttr("invalid.constant", rewriter.getUnitAttr());
      return constOp;
    }

    return rewriter.create<arith::ConstantOp>(redOp->getLoc(), constantType,
                                              attr);
  }

  static bool isPlaceholder(Value v) {
    if (auto constOp = v.getDefiningOp<arith::ConstantOp>())
      return constOp->hasAttr("invalid.constant");
    return false;
  }

  LogicalResult
  convertToLinalgReduce(triton::ReduceOp op,
                        typename triton::ReduceOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const {

    auto loc = op.getLoc();
    SmallVector<Value> sources(adaptor.getOperands().begin(),
                               adaptor.getOperands().end());
    unsigned numReductions = sources.size();

    if (numReductions == 0)
      return rewriter.notifyMatchFailure(op, "no inputs to reduce");

    SmallVector<Type> resTypes;
    for (Value r : op.getResult()) {
      resTypes.push_back(r.getType());
    }
    if (resTypes.size() != numReductions) {
      return rewriter.notifyMatchFailure(
          op, "number of results must match number of inputs");
    }

    // We'll use the first source's ranked tensor type for rank
    // checks/transposes.
    auto firstSourceType = cast<RankedTensorType>(sources.front().getType());
    auto elemType = firstSourceType.getElementType();
    unsigned rank = firstSourceType.getRank();

    // Get the reduction region's block
    Block *redBlock = op.getBody();
    if (!redBlock)
      return rewriter.notifyMatchFailure(op, "reduce has no body");

    auto axis = op.getAxis();
    auto isVectorReduce = firstSourceType.getRank() == 1;

    if (axis == firstSourceType.getRank() - 1 && !isVectorReduce) {
      for (unsigned i = 0; i < numReductions; ++i) {
        sources[i] = getTransposedValue(sources[i], op.getLoc(), rewriter);
      }
      axis = static_cast<int64_t>(rank) - 2;
    }

    SmallVector<Operation *> accUsers(numReductions, nullptr);
    SmallVector<bool> convertToF32(numReductions, false);
    SmallVector<Type> constantTypes(numReductions);
    SmallVector<Value> accBaseConstOps(numReductions);

    for (unsigned i = 0; i < numReductions; ++i) {
      unsigned accArgIdx =
          numReductions + i; // Triton ordering: inputs[0..N-1], outputs[0..N-1]
      Operation *accUser = findFirstAccumulatorUserAt(*redBlock, accArgIdx);
      if (!accUser)
        return rewriter.notifyMatchFailure(
            op, "Expected reduction block to have an accumulator user");
      accUsers[i] = accUser;

      // Decide if we need f32 promotion for this reduction
      convertToF32[i] = requiresF32Conversion(resTypes[i], accUser);

      auto srcType = cast<RankedTensorType>(sources[i].getType());
      Type srcElemType = srcType.getElementType();
      constantTypes[i] = convertToF32[i]
                             ? Float32Type::get(rewriter.getContext())
                             : srcElemType;
      accBaseConstOps[i] =
          getRedBaseConstOp(rewriter, accUser, constantTypes[i]);
    }

    // To be able to support any type of body in the reduce op, we
    // initialize the linalg.reduce with the first element of each
    // source tensor. This ensures that the accumulator type matches
    // the source element type, and that the reduction region body
    // is valid without any modification and no need for constants.
    SmallVector<Value> initTensors(numReductions);

    for (unsigned i = 0; i < numReductions; ++i) {
      Value firstElem =
          getFirstElementAlongAxis(rewriter, sources[i], axis, loc);
      Value initVal =
          isPlaceholder(accBaseConstOps[i]) ? firstElem : accBaseConstOps[i];
      Value initTensor;
      if (isVectorReduce) {
        // First element is scalar already, so wrap in rank-0 tensor
        Value alloc = rewriter.create<bufferization::AllocTensorOp>(
            loc, RankedTensorType::get({}, initVal.getType()), ValueRange{});
        initTensor = rewriter.create<tensor::InsertOp>(loc, initVal, alloc,
                                                       ValueRange{});
      } else {
        // Create output shape and fill from initVal
        auto resRanked = cast<RankedTensorType>(op.getResult()[i].getType());
        SmallVector<int64_t> resShape(resRanked.getShape().begin(),
                                      resRanked.getShape().end());
        Value empty =
            rewriter.create<tensor::EmptyOp>(loc, resShape, constantTypes[i]);
        initTensor = rewriter
                         .create<linalg::FillOp>(loc, ValueRange{initVal},
                                                 ValueRange{empty})
                         .result();
      }
      initTensors[i] = initTensor;
    }

    // Create the linalg.reduce op
    auto reduceOp = rewriter.create<linalg::ReduceOp>(
        loc, sources, initTensors,
        axis, // one axis for all reductions
        [&](OpBuilder &opBuilder, Location innerLoc, ValueRange inputs) {
          // expected inputs.size() == 2 * numReductions
          assert(inputs.size() == 2 * numReductions &&
                 "reduce region expects 2*N inputs");

          IRMapping bvm;

          // Map Triton region args to inputs:
          // Triton region args: [in0, in1, ..., inN-1, out0, out1, ...,
          // outN-1] linalg.reduce inputs:  same ordering (ins..., outs...)
          for (unsigned i = 0; i < numReductions; ++i) {
            Value inVal = inputs[i];                  // element
            Value accVal = inputs[numReductions + i]; // accumulator

            // If this reduction wants f32 compute and element isn't f32,
            // extend it now.
            if (convertToF32[i]) {
              if (!inVal.getType().isa<Float32Type>()) {
                inVal = opBuilder.create<arith::ExtFOp>(
                    innerLoc, constantTypes[i], inVal);
              }
              // accumulator (accVal) should already be of constantTypes[i]
              // because init tensors were created that way
            }

            // Map block args:
            bvm.map(redBlock->getArgument(i), inVal); // input arg i
            bvm.map(redBlock->getArgument(numReductions + i),
                    accVal); // accumulator arg i
          }

          // Ensure clone insertion happens inside the reduce region builder.
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(opBuilder.getBlock(),
                                     opBuilder.getInsertionPoint());

          // Clone body ops except the terminator; when we hit the terminator,
          // yield N values.
          for (Operation &innerOp : redBlock->getOperations()) {
            if (&innerOp == redBlock->getTerminator()) {
              // Triton terminator yields N values (one per
              // accumulator/output)
              SmallVector<Value> yields;
              yields.reserve(numReductions);
              for (unsigned i = 0; i < numReductions; ++i) {
                // the terminator operand i corresponds to the i-th output
                Value termOperand = innerOp.getOperand(i);
                Value mapped = bvm.lookupOrNull(termOperand);
                yields.push_back(mapped);
              }
              opBuilder.create<linalg::YieldOp>(innerLoc, yields);
              break;
            }
            // Clone the op into the reduce region
            rewriter.clone(innerOp, bvm);
          }
        });

    SmallVector<Value> finalResults;
    finalResults.reserve(numReductions);
    ValueRange reducedVals = reduceOp.getResults(); // returns N Values
    for (unsigned i = 0; i < numReductions; ++i) {
      Value v = reducedVals[i];
      // If the source/result was rank-1 originally, extract scalar
      if (isVectorReduce) {
        v = rewriter.create<tensor::ExtractOp>(loc, constantTypes[i], v);
      }

      // If we promoted to f32 for compute, truncate back to the original
      // result type
      if (convertToF32[i]) {
        v = rewriter.create<arith::TruncFOp>(loc, resTypes[i], v);
      }

      finalResults.push_back(v);
    }

    rewriter.replaceOp(op, finalResults);
    return success();
  }

public:
  LogicalResult
  matchAndRewrite(triton::ReduceOp op,
                  typename triton::ReduceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType =
        cast<RankedTensorType>(adaptor.getOperands().front().getType());
    assert(sourceType.hasRank() && "Expected input is "
                                   "ranked");

    int64_t axis = op.getAxis();
    assert(axis >= 0 && axis < sourceType.getRank() &&
           "Expected reduction "
           "axis is within "
           "operand's rank");

    return convertToLinalgReduce(op, adaptor, rewriter);
  }
};

template <typename T>
class ArgMinMaxBaseConverter : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern<triton::ReduceOp>::OpConversionPattern;

  // We're looking for an op that looks like this:
  //
  // %9:2 = "tt.reduce"(%8, %3) <{axis = 0 : i32}> ({
  // ^bb0(%arg9: f32, %arg10: i32, %arg11: f32, %arg12: i32):
  // -------------------------------------------------
  // `matchTieBreakValue`                                |
  //   %11 = arith.cmpf oeq, %arg9, %arg11 : f32         |
  //   %12 = arith.cmpi slt, %arg10, %arg12 : i32        |   1.
  //   %13 = arith.andi %11, %12 : i1                    |
  // -------------------------------------------------   |-> `matchShouldUpdate`
  // `matchUpdateCondition`                              |
  //   %14 = arith.cmpf ogt, %arg9, %arg11 : f32         |   2.
  // -------------------------------------------------   |
  //   %15 = arith.ori %14, %13 : i1                     |
  // -------------------------------------------------
  //   %16 = arith.select %15, %arg9, %arg11 : f32
  //   %17 = arith.select %15, %arg10, %arg12 : i32
  //   tt.reduce.return %16, %17 : f32, i32
  // }) : (tensor<4096xf32>, tensor<4096xi32>) -> (f32, i32)
  //
  // The above mlir code is lowered from this combinator in triton's
  // standard.py:
  //
  //  def _argmax_combine(value1, index1, value2, index2, tie_break_left):
  //    if tie_break_left:
  //        tie = value1 == value2 and index1 < index2
  //    else:
  //        tie = False
  //    gt = value1 > value2 or tie
  //    v_ret = core.where(gt, value1, value2)
  //    i_ret = core.where(gt, index1, index2)
  //    return v_ret, i_ret

  LogicalResult matchTieBreakResult(Value currValue, Value currIndex,
                                    Value reduceValue, Value reduceIndex,
                                    mlir::Block::iterator &it,
                                    Value &tileBreakValue) const {
    // Match the following (section 1. of the above)
    //
    //   %11 = arith.cmpf oeq, %arg9, %arg11 : f32
    //   %12 = arith.cmpi slt, %arg10, %arg12 : i32
    //   %13 = arith.andi %11, %12 : i1
    //
    // which is equivalent to the following python code
    //
    //   tie = value1 == value2 and index1 < index2

    // matching: %11 = arith.cmpf oeq, %arg9, %arg11 : f32
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *it << "\n");
    auto eqCmpOp = dyn_cast<arith::CmpFOp>(*it++);
    if (eqCmpOp) {
      if (eqCmpOp.getPredicate() != arith::CmpFPredicate::OEQ) {
        return failure();
      }
      if (currValue != eqCmpOp.getLhs() || reduceValue != eqCmpOp.getRhs()) {
        return failure();
      }
    } else {
      return failure();
    }

    // matching: %12 = arith.cmpi slt, %arg10, %arg12 : i32
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *it << "\n");
    auto sltCmpOp = dyn_cast<arith::CmpIOp>(*it++);
    if (sltCmpOp) {
      if (sltCmpOp.getPredicate() != arith::CmpIPredicate::slt) {
        return failure();
      }
      if (currIndex != sltCmpOp.getLhs() || reduceIndex != sltCmpOp.getRhs()) {
        return failure();
      }
    } else {
      return failure();
    }

    // matching: %13 = arith.andi %11, %12 : i1
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *it << "\n");
    auto andOp = dyn_cast<arith::AndIOp>(*it++);
    if (andOp) {
      if (andOp.getLhs() != eqCmpOp || andOp.getRhs() != sltCmpOp) {
        return failure();
      }
    } else {
      return failure();
    }

    tileBreakValue = andOp;
    return success();
  }

  LogicalResult matchShouldUpdateValue(Value currValue, Value currIndex,
                                       Value reduceValue, Value reduceIndex,
                                       mlir::Block::iterator &it,
                                       Value &shouldUpdate) const {
    Value tieResult;
    if (failed(matchTieBreakResult(currValue, currIndex, reduceValue,
                                   reduceIndex, it, tieResult))) {
      LLVM_DEBUG(llvm::dbgs() << "Tie break result match failed\n");
      return failure();
    }

    Value comparisonResult;
    if (failed(T::matchComparisonResult(currValue, currIndex, reduceValue,
                                        reduceIndex, it, comparisonResult))) {
      LLVM_DEBUG(llvm::dbgs() << "Comparison result match failed\n");
      return failure();
    }

    // matching: %15 = arith.ori %14, %13 : i1
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *it << "\n");
    auto orOp = dyn_cast<arith::OrIOp>(*it++);
    if (orOp) {
      if (orOp.getLhs() != comparisonResult || orOp.getRhs() != tieResult) {
        return failure();
      }
    } else {
      return failure();
    }

    shouldUpdate = orOp;
    return success();
  }

  Value getInitTensor(ConversionPatternRewriter &rewriter,
                      ArrayRef<int64_t> shape, Value fillValue,
                      Location loc) const {
    Value initTensor =
        rewriter.create<tensor::EmptyOp>(loc, shape, fillValue.getType());
    return rewriter
        .create<linalg::FillOp>(loc, ValueRange{fillValue},
                                ValueRange{initTensor})
        .result();
  }

public:
  ArgMinMaxBaseConverter(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult match(ReduceOp op) const override final {
    if (op.getBody()->getNumArguments() != 4) {
      return failure();
    }

    auto block = op.getBody();
    auto ops = block->without_terminator();

    Value currValue = block->getArgument(0);
    Value currIndex = block->getArgument(1);
    Value reduceValue = block->getArgument(2);
    Value reduceIndex = block->getArgument(3);

    auto opsIt = ops.begin();
    Value shouldUpdate;
    if (failed(matchShouldUpdateValue(currValue, currIndex, reduceValue,
                                      reduceIndex, opsIt, shouldUpdate))) {
      return failure();
    }

    // matching: %16 = arith.select %15, %arg9, %arg11 : f32
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *opsIt << "\n");
    auto valueSelectOp = dyn_cast<arith::SelectOp>(*opsIt++);
    if (valueSelectOp) {
      if (valueSelectOp.getCondition() != shouldUpdate ||
          currValue != valueSelectOp.getTrueValue() ||
          reduceValue != valueSelectOp.getFalseValue()) {
        return failure();
      }
    } else {
      return failure();
    }

    // matching:%17 = arith.select %15, %arg10, %arg12 : i32
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *opsIt << "\n");
    auto indexSelectOp = dyn_cast<arith::SelectOp>(*opsIt++);
    if (indexSelectOp) {
      if (indexSelectOp.getCondition() != shouldUpdate ||
          currIndex != indexSelectOp.getTrueValue() ||
          reduceIndex != indexSelectOp.getFalseValue()) {
        return failure();
      }
    } else {
      return failure();
    }

    // matching: tt.reduce.return %16, %17 : f32, i32
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *opsIt << "\n");
    auto termOp = dyn_cast<triton::ReduceReturnOp>(*opsIt++);
    if (termOp && termOp == block->getTerminator()) {
      auto opnds = termOp.getOperands();
      if (opnds != ArrayRef<Value>{valueSelectOp, indexSelectOp}) {
        return failure();
      }
    } else {
      return failure();
    }

    return success();
  }

  void rewrite(ReduceOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override final {
    auto loc = op.getLoc();

    auto elemTypes = op.getElementTypes();

    // Set the initial value of the rank-0 tensor containing
    // the result value to either -inf or +inf depending on
    // whether we're dealing with argmax or argmin
    auto valueType = elemTypes[0];
    auto valuesAccBaseVal = rewriter.create<arith::ConstantOp>(
        loc, valueType,
        rewriter.getFloatAttr(valueType, T::getBaseReductionValue()));

    // Set the initial value of the rank-0 tensor containing the index of the
    // min or max value to -1
    auto indexType = elemTypes[1];
    auto indicesAccBaseVal = rewriter.create<arith::ConstantOp>(
        loc, indexType, rewriter.getIntegerAttr(indexType, -1));

    // Get the shape of the resulting tensors (both for values and indices). If
    // we are reducing to a single scalar, then the result's type is a tensor of
    // rank-0, otherwise we can reuse the original result shape
    auto valueResultType = dyn_cast<RankedTensorType>(op.getType(0));
    const auto isScalarReduce = valueResultType == nullptr;
    SmallVector<int64_t> reductionResultShape{
        isScalarReduce ? SmallVector<int64_t>{}
                       : SmallVector<int64_t>(valueResultType.getShape())};

    SmallVector<Value> outputs{
        getInitTensor(rewriter, reductionResultShape, valuesAccBaseVal, loc),
        getInitTensor(rewriter, reductionResultShape, indicesAccBaseVal, loc)};

    auto linalgOp = rewriter.create<linalg::ReduceOp>(
        loc, adaptor.getOperands(), outputs,
        SmallVector<int64_t>{adaptor.getAxis()},
        [&](OpBuilder &b, Location loc, ValueRange inputs) {
          assert(inputs.size() == 4);

          auto tritonReduceBlock = op.getBody();
          IRMapping mapping;
          mapping.map(tritonReduceBlock->getArguments(), inputs);

          for (auto &op : tritonReduceBlock->without_terminator()) {
            b.clone(op, mapping);
          }

          auto tritonYield = tritonReduceBlock->getTerminator();
          auto results =
              llvm::map_to_vector(tritonYield->getOperands(), [&](Value val) {
                return mapping.lookup(val);
              });
          b.create<linalg::YieldOp>(loc, results);
        });

    if (isScalarReduce) {
      SmallVector<Value> reduceResults{
          rewriter.create<tensor::ExtractOp>(
              loc, valueType, linalgOp.getResults()[0], ValueRange{}),
          rewriter.create<tensor::ExtractOp>(
              loc, indexType, linalgOp.getResults()[1], ValueRange{})};
      rewriter.replaceOp(op, reduceResults);
    } else {
      rewriter.replaceOp(op, linalgOp);
    }
  }
};

struct ArgMaxConverter : public ArgMinMaxBaseConverter<ArgMaxConverter> {
  static LogicalResult matchComparisonResult(Value currValue, Value currIndex,
                                             Value reduceValue,
                                             Value reduceIndex,
                                             mlir::Block::iterator &it,
                                             Value &comparisonResult) {
    // %14 = arith.cmpf ogt, %arg9, %arg11 : f32
    // This corresponds to section 2. of the sample snippet in
    // ArgMinMaxBaseConverter
    auto cmpOp = dyn_cast<arith::CmpFOp>(*it++);
    if (cmpOp) {
      if (cmpOp.getPredicate() != arith::CmpFPredicate::OGT ||
          currValue != cmpOp.getLhs() || reduceValue != cmpOp.getRhs()) {
        return failure();
      }
    } else {
      return failure();
    }

    comparisonResult = cmpOp;
    return success();
  }

  static float getBaseReductionValue() {
    return -std::numeric_limits<float>::infinity();
  }

  ArgMaxConverter(MLIRContext *context) : ArgMinMaxBaseConverter(context) {}
};

struct ArgMinConverter : public ArgMinMaxBaseConverter<ArgMinConverter> {
  static LogicalResult matchComparisonResult(Value currValue, Value currIndex,
                                             Value reduceValue,
                                             Value reduceIndex,
                                             mlir::Block::iterator &it,
                                             Value &comparisonResult) {
    // %14 = arith.cmpf olt, %arg9, %arg11 : f32
    // This corresponds to section 2. of the sample snippet in
    // ArgMinMaxBaseConverter
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *it << "\n");
    auto cmpOp = dyn_cast<arith::CmpFOp>(*it++);
    if (cmpOp) {
      if (cmpOp.getPredicate() != arith::CmpFPredicate::OLT ||
          currValue != cmpOp.getLhs() || reduceValue != cmpOp.getRhs()) {
        return failure();
      }
    } else {
      return failure();
    }

    comparisonResult = cmpOp;
    return success();
  }

  static float getBaseReductionValue() {
    return std::numeric_limits<float>::infinity();
  }

  ArgMinConverter(MLIRContext *context) : ArgMinMaxBaseConverter(context) {}
};

// get_program_id and get_num_programs:
// When launching triton kernels, we pass 6 additional arguments to indicate
// num_programs and program_id. Amongst those six, we have 3 arguments
// correspond to each axis for num_programs followed by 3 additional arguments
// for program_id.
//
// For instance, with triton kernel example_kernel(a, b, c), we have:
//  example_kernel(
//    a, b, c,
//    num_programs_axis_0,
//    num_programs_axis_1,
//    num_programs_axis_2,
//    program_id_axis_0,
//    program_id_axis_1,
//    program_id_axis_2,
//   )
//
struct GetProgramIDConverter
    : public OpConversionPattern<triton::GetProgramIdOp> {
  using OpConversionPattern<triton::GetProgramIdOp>::OpConversionPattern;
  static uint32_t constexpr LAUNCH_GRID_RANK =
      getMaxEnumValForProgramIDDim() + 1;

public:
  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto axis = (uint32_t)op.getAxis();
    assert(axis < LAUNCH_GRID_RANK && "program_id expects "
                                      "axis to be either 0, "
                                      "1, or 2");

    auto func = op->getParentOfType<FunctionOpInterface>();
    auto numArgs = func.getNumArguments();
    auto id = func.getArgument(numArgs - LAUNCH_GRID_RANK + axis);

    rewriter.replaceOp(op, id);
    return success();
  }
};

struct GetNumProgramsConverter
    : public OpConversionPattern<triton::GetNumProgramsOp> {
  using OpConversionPattern<triton::GetNumProgramsOp>::OpConversionPattern;

private:
  static uint32_t constexpr LAUNCH_GRID_RANK =
      getMaxEnumValForProgramIDDim() + 1;

public:
  GetNumProgramsConverter(MLIRContext *context)
      : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto axis = (uint32_t)op.getAxis();
    assert(axis < LAUNCH_GRID_RANK && "program_id expects "
                                      "axis to be either 0, "
                                      "1, or 2");

    auto func = op->getParentOfType<FunctionOpInterface>();
    auto numArgs = func.getNumArguments();
    auto id = func.getArgument(numArgs - LAUNCH_GRID_RANK * 2 + axis);

    rewriter.replaceOp(op, id);
    return success();
  }
};

// Convert a pair of cmpf and select to either min or max.
// Leave the pattern as simple as possible because triton has plans to emit
// min and max directly.
template <typename CmpOp>
struct MinMaxConverter : public OpRewritePattern<CmpOp> {
  using OpRewritePattern<CmpOp>::OpRewritePattern;

  MinMaxConverter(MLIRContext *context)
      : OpRewritePattern<CmpOp>(context, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(CmpOp cmpOp,
                                PatternRewriter &rewriter) const final {
    if (!cmpOp.getResult().hasOneUse()) {
      return failure();
    }
    auto selectOp =
        dyn_cast<arith::SelectOp>(*cmpOp.getResult().getUsers().begin());
    if (!selectOp) {
      return failure();
    }

    if (!(cmpOp.getResult() == selectOp.getCondition() &&
          cmpOp.getLhs() == selectOp.getTrueValue() &&
          cmpOp.getRhs() == selectOp.getFalseValue())) {
      return failure();
    }

    rewriteOpWithMinMax(rewriter, cmpOp, selectOp, cmpOp.getPredicate());
    rewriter.eraseOp(cmpOp);

    return success();
  }

  void rewriteOpWithMinMax(PatternRewriter &rewriter, arith::CmpFOp cmpOp,
                           arith::SelectOp selectOp,
                           arith::CmpFPredicate pred) const {
    switch (pred) {
    case arith::CmpFPredicate::OGT:
    case arith::CmpFPredicate::OGE:
      rewriter.replaceOpWithNewOp<arith::MaximumFOp>(selectOp, cmpOp.getLhs(),
                                                     cmpOp.getRhs());
      break;
    case arith::CmpFPredicate::OLT:
    case arith::CmpFPredicate::OLE:
      rewriter.replaceOpWithNewOp<arith::MinimumFOp>(selectOp, cmpOp.getLhs(),
                                                     cmpOp.getRhs());
      break;
    default:
      llvm_unreachable("Unhandled predicate");
    }
  }

  void rewriteOpWithMinMax(PatternRewriter &rewriter, arith::CmpIOp cmpOp,
                           arith::SelectOp selectOp,
                           arith::CmpIPredicate pred) const {
    switch (pred) {
    case arith::CmpIPredicate::sgt:
      rewriter.replaceOpWithNewOp<arith::MaxSIOp>(selectOp, cmpOp.getLhs(),
                                                  cmpOp.getRhs());
      break;
    case arith::CmpIPredicate::ugt:
      rewriter.replaceOpWithNewOp<arith::MaxUIOp>(selectOp, cmpOp.getLhs(),
                                                  cmpOp.getRhs());
      break;
    case arith::CmpIPredicate::slt:
      rewriter.replaceOpWithNewOp<arith::MinSIOp>(selectOp, cmpOp.getLhs(),
                                                  cmpOp.getRhs());
      break;
    case arith::CmpIPredicate::ult:
      rewriter.replaceOpWithNewOp<arith::MinUIOp>(selectOp, cmpOp.getLhs(),
                                                  cmpOp.getRhs());
      break;
    default:
      llvm_unreachable("Unhandled predicate");
    }
  }
};

struct DenseConstantConverter : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto attr = cast<DenseElementsAttr>(op.getValue());
    auto loc = op.getLoc();

    auto splatConst = arith::ConstantOp::materialize(
        rewriter, attr.getSplatValue<Attribute>(), attr.getElementType(), loc);

    auto init = rewriter.create<tensor::EmptyOp>(
        loc, cast<RankedTensorType>(op.getResult().getType()).getShape(),
        attr.getElementType());

    rewriter.replaceOpWithNewOp<linalg::FillOp>(op, ValueRange{splatConst},
                                                ValueRange{init});

    return success();
  }
};

class CumSumConverter : public OpConversionPattern<triton::ScanOp> {
  using OpConversionPattern<triton::ScanOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value> sources(adaptor.getOperands().begin(),
                               adaptor.getOperands().end());
    unsigned numReductions = sources.size();

    if (numReductions == 0) {
      return rewriter.notifyMatchFailure(
          op, "Expected at least one source operand for scan");
    }

    SmallVector<RankedTensorType> inputTypes;
    inputTypes.reserve(numReductions);
    for (Value source : sources) {
      auto rankedType = dyn_cast<RankedTensorType>(source.getType());
      if (!rankedType || !rankedType.hasRank()) {
        return rewriter.notifyMatchFailure(
            op, "Expected all source operands to be ranked tensors");
      }
      inputTypes.push_back(rankedType);
    }

    // All sources must have the same rank (ScanOp requires that).
    int64_t rank = inputTypes[0].getRank();

    int axis = op.getAxis();
    bool reverse = op.getReverse();

    // 2. Allocate output memrefs
    SmallVector<Value> outputMemrefs;
    outputMemrefs.reserve(numReductions);
    for (unsigned i = 0; i < numReductions; ++i) {
      outputMemrefs.push_back(rewriter.create<memref::AllocOp>(
          loc, MemRefType::get(inputTypes[i].getShape(),
                               inputTypes[i].getElementType())));
    }

    // 3. Build loop nest
    SmallVector<Value> lbs, ubs, steps;
    lbs.reserve(rank);
    ubs.reserve(rank);
    steps.reserve(rank);

    for (int64_t i = 0; i < rank; ++i) {
      Value lb = rewriter.create<arith::ConstantIndexOp>(loc, 0).getResult();
      lbs.push_back(lb);
      int64_t dimSize = inputTypes[0].getDimSize(i);
      Value ub =
          rewriter.create<arith::ConstantIndexOp>(loc, dimSize).getResult();
      ubs.push_back(ub);
      Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1).getResult();
      steps.push_back(step);
    }

    // capture ubs, axis, reverse from outer scope
    auto loopNest = scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps,
        [&](OpBuilder &b, Location loc, ValueRange ivs) {
          Value ivAxis = ivs[axis]; // 0..N-1
          Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
          Value one = b.create<arith::ConstantIndexOp>(loc, 1);
          Value dimSize = ubs[axis]; // outer ubs vector

          // Compute pos = logical index along scan axis
          Value pos;
          if (reverse) {
            // pos = (dimSize - 1) - ivAxis
            Value dimMinus1 = b.create<arith::SubIOp>(loc, dimSize, one);
            pos = b.create<arith::SubIOp>(loc, dimMinus1, ivAxis);
          } else {
            pos = ivAxis;
          }

          // Base case check: first iteration of the loop (ivAxis == 0)
          Value isFirst = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  ivAxis, zero);

          // Build index vector that uses `pos` on the scan axis
          SmallVector<Value> idxs(ivs.begin(), ivs.end());
          idxs[axis] = pos;

          b.create<scf::IfOp>(
              loc, isFirst,
              [&](OpBuilder &b, Location loc) {
                // first element (pos is either 0 or dimSize-1 depending on
                // reverse)
                for (unsigned i = 0; i < numReductions; ++i) {
                  Value v = b.create<tensor::ExtractOp>(loc, sources[i], idxs);
                  b.create<memref::StoreOp>(loc, v, outputMemrefs[i], idxs);
                }
                b.create<scf::YieldOp>(loc);
              },
              [&](OpBuilder &b, Location loc) {
                // general case: read accumulator from previous logical
                // position
                SmallVector<Value> prevIdx = idxs;
                Value prevPos;
                if (reverse) {
                  // prev = pos + 1
                  prevPos = b.create<arith::AddIOp>(loc, pos, one);
                } else {
                  // prev = pos - 1
                  prevPos = b.create<arith::SubIOp>(loc, pos, one);
                }
                prevIdx[axis] = prevPos;

                SmallVector<Value> accs(numReductions), curs(numReductions);
                for (unsigned i = 0; i < numReductions; ++i) {
                  // read the previous value from the output memref
                  accs[i] =
                      b.create<memref::LoadOp>(loc, outputMemrefs[i], prevIdx);
                  // read the current value from the input memref
                  curs[i] = b.create<tensor::ExtractOp>(loc, sources[i], idxs);
                }

                // Map ALL 2*N args: [outputs..., inputs...]
                IRMapping mapping;
                Block &reg = op.getRegion().front();
                // NOTE: accs come before the inputs which is counterintutive
                // Map input args (regionArg[0..N-1]) -> accs
                for (unsigned i = 0; i < numReductions; ++i)
                  mapping.map(reg.getArgument(i), accs[i]);
                // Map accumulator args (regionArg[N..2N-1]) -> curs
                for (unsigned i = 0; i < numReductions; ++i)
                  mapping.map(reg.getArgument(numReductions + i), curs[i]);

                for (Operation &innerOp :
                     op.getRegion().front().without_terminator())
                  b.clone(innerOp, mapping);

                Operation *term = reg.getTerminator();
                SmallVector<Value> results(numReductions);
                for (unsigned i = 0; i < numReductions; ++i) {
                  results[i] = mapping.lookupOrDefault(term->getOperand(i));
                }

                // Store each result
                for (unsigned i = 0; i < numReductions; ++i)
                  b.create<memref::StoreOp>(loc, results[i], outputMemrefs[i],
                                            idxs);

                b.create<scf::YieldOp>(loc);
              });
        });

    SmallVector<Value> resultTensors;
    resultTensors.reserve(numReductions);
    for (unsigned i = 0; i < numReductions; ++i) {
      resultTensors.push_back(rewriter.create<bufferization::ToTensorOp>(
          loc, outputMemrefs[i], true, true));
    }

    rewriter.replaceOp(op, ValueRange{resultTensors});
    return success();
  }
};

class AddPtrConverter : public OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern<triton::AddPtrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resType = op.getResult().getType();
    assert(isa<ShapedType>(resType));
    auto rank = cast<RankedTensorType>(resType).getRank();
    SmallVector<AffineMap, 3> indexingMaps(
        /*numResult + numOperands*/ 3, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType, 6> iteratorTypes(
        rank, utils::IteratorType::parallel);
    SmallVector<Value> outputs = {op.getPtr()};
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, op->getResultTypes(), op->getOperands(), outputs, indexingMaps,
        iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
          auto resultTypes =
              llvm::map_to_vector(op->getResultTypes(), [](Type type) {
                return cast<TensorType>(type).getElementType();
              });
          auto *scalarOp =
              builder.create(loc, op->getName().getIdentifier(),
                             regionArgs.take_front(op->getNumOperands()),
                             resultTypes, op->getAttrs());
          builder.create<linalg::YieldOp>(loc, scalarOp->getResults());
        });
    return success();
  }
};

// Convert triton op X operating on tensors of pointers to a linalg.generic
// wrapping op X to operate on single pointer.
// This pattern rewriter is almost identical to AddPtrConverter above, except
// that the out param for the linalg op is an empty op instead of reusing one
// of the existing operands. This is because depending on the templatized op,
// the type of the operands might be different, so we cannot pick a default
// operand to reuse for all cases.
template <typename OpType>
class TensorOpConverter : public OpConversionPattern<OpType> {
  using OpConversionPattern<OpType>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTensorType =
        dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultTensorType) {
      return failure();
    }
    auto rank = resultTensorType.getRank();
    SmallVector<AffineMap> indexingMaps(
        /*numResult + numOperands*/ op->getNumResults() + op->getNumOperands(),
        rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    SmallVector<Value> outputs = {rewriter.create<tensor::EmptyOp>(
        op->getLoc(), resultTensorType.getShape(),
        resultTensorType.getElementType())};
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, op->getResultTypes(), op->getOperands(), outputs, indexingMaps,
        iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
          auto resultTypes =
              llvm::map_to_vector(op->getResultTypes(), [](Type type) {
                return cast<TensorType>(type).getElementType();
              });
          auto *scalarOp =
              builder.create(loc, op->getName().getIdentifier(),
                             regionArgs.take_front(op->getNumOperands()),
                             resultTypes, op->getAttrs());
          builder.create<linalg::YieldOp>(loc, scalarOp->getResults());
        });
    return success();
  }
};

// Convert triton store op operating on tensors of pointers to a linalg.generic
// wrapping op a triton store op on single pointer.
// Note that this linalg.generic op has an empty `out` param.
class StorePtrToLinalgConverter : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto storeTensorType = dyn_cast<RankedTensorType>(op.getValue().getType());
    if (!storeTensorType) {
      return failure();
    }
    auto rank = storeTensorType.getRank();
    SmallVector<AffineMap> indexingMaps(
        /*numResult + numOperands*/ op->getNumResults() + op.getNumOperands(),
        rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    SmallVector<Value> outputs;
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, op->getResultTypes(), op->getOperands(), outputs, indexingMaps,
        iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
          auto resultTypes =
              llvm::map_to_vector(op->getResultTypes(), [](Type type) {
                return cast<TensorType>(type).getElementType();
              });
          auto *scalarOp =
              builder.create(loc, op->getName().getIdentifier(),
                             regionArgs.take_front(op->getNumOperands()),
                             resultTypes, op->getAttrs());
          builder.create<linalg::YieldOp>(loc, scalarOp->getResults());
        });
    return success();
  }
};

class ReshapeConverter : public OpConversionPattern<triton::ReshapeOp> {
  using OpConversionPattern<triton::ReshapeOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(triton::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = op.getSrc();
    auto output = op.getResult();

    auto inputType = input.getType();
    auto outputType = output.getType();
    if (!outputType.hasStaticShape()) {
      return failure();
    }

    if (auto maybeReassociationMap =
            getReassociationIndicesForReshape(inputType, outputType)) {
      auto reassociationMap = *maybeReassociationMap;
      if (outputType.getRank() < inputType.getRank()) {
        rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
            op, outputType, input, reassociationMap);
      } else {
        rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
            op, outputType, input, reassociationMap);
      }
      return success();
    }

    ArrayRef<int64_t> outputShape = outputType.getShape();

    auto shape = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64TensorAttr(outputShape));
    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(op, outputType, input,
                                                   shape);

    return success();
  }
};

class BarrierConverter : public OpConversionPattern<gpu::BarrierOp> {
  using OpConversionPattern<gpu::BarrierOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (dyn_cast<RegionBranchOpInterface>(op->getParentOp())) {
      // If possible, gpu.barrier is replaced with llvm.fence Op.
      rewriter.replaceOpWithNewOp<LLVM::FenceOp>(op,
          mlir::LLVM::AtomicOrdering::seq_cst, "crossthread");
      return success();
    }
    // All other cases remove gpu.barrier
    rewriter.eraseOp(op);
    return success();
  }
};

class HistogramConverter : public OpConversionPattern<triton::HistogramOp> {
  using OpConversionPattern<triton::HistogramOp>::OpConversionPattern;
public:
  LogicalResult
  matchAndRewrite(HistogramOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto input = op.getSrc();
    auto ResultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    SmallVector<AffineMap> affineMaps(1,
        rewriter.getMultiDimIdentityMap(ResultType.getRank()));
    auto Zero = rewriter.create<arith::ConstantOp>(op->getLoc(),
        rewriter.getIntegerAttr(rewriter.getIntegerType(32), 0));

    auto alloc = rewriter.create<memref::AllocOp>(loc,
        MemRefType::get(ResultType.getShape(), ResultType.getElementType()));

    auto Zeros = rewriter.create<linalg::FillOp>(loc, ValueRange{Zero},
                                                 ValueRange{alloc});

    auto genericOp = rewriter.create<linalg::GenericOp>(loc, TypeRange{},
        ValueRange{input}, ValueRange{}, affineMaps,
        SmallVector<utils::IteratorType>(ResultType.getRank(),
                                         utils::IteratorType::parallel),
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange regionArgs) {
          // triton supports histogram calculation only from 0 and width of 1.
          // Each input data is considered as index for output array.
          // Increment the output array will get the result.
          auto loadValue = regionArgs[0];
          auto One = nestedBuilder.create<arith::ConstantOp>(nestedLoc,
              rewriter.getIntegerAttr(nestedBuilder.getIntegerType(32), 1));
          Value histogramArrayIndex = nestedBuilder.create<arith::IndexCastOp>(
              nestedLoc, rewriter.getIndexType(), loadValue);
          auto kindAttr = arith::AtomicRMWKindAttr::get(
              nestedBuilder.getContext(), arith::AtomicRMWKind::addi);

            SmallVector<Value, 1> memIndices{histogramArrayIndex};
            auto atomic = nestedBuilder.create<memref::AtomicRMWOp>(nestedLoc,
                kindAttr, One, alloc, memIndices);

            nestedBuilder.create<linalg::YieldOp>(nestedLoc);
        });

    SmallVector<Value> resultTensors {rewriter.create<bufferization::ToTensorOp>(
        loc, alloc, true, true)};
    rewriter.replaceOp(op, ValueRange{resultTensors});
    return success();
  }
};

class ExternElementwiseBinaryOpConverter
    : public OpConversionPattern<triton::ExternElementwiseOp> {
  using OpConversionPattern<triton::ExternElementwiseOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(triton::ExternElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // sleef function name is like: Sleef_xxx(numel), need to get rid of the
    // placeholder
    StringRef sym = op.getSymbol().split('(').first;
    if (!op.getPure() || op.getSrcs().size() != 2)
      return failure();

    // Calls to sleef math library
    if (sym.starts_with("Sleef_")) {
      auto moduleOp = op->getParentOfType<ModuleOp>();
      auto operands = adaptor.getOperands();
      auto funcType =
          rewriter.getFunctionType(operands.getTypes(), op.getType());
      auto funcOp = moduleOp.lookupSymbol<func::FuncOp>(sym);

      if (!funcOp) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        funcOp = rewriter.create<func::FuncOp>(loc, sym, funcType);
        funcOp.setPrivate();
      }
      rewriter.replaceOpWithNewOp<func::CallOp>(op, funcOp, operands);
      return success();
    }
#define POPULATE_BINARY_OP(FUNC_NAME, DST_OP)                                  \
  if (!sym.compare(FUNC_NAME)) {                                               \
    rewriter.replaceOpWithNewOp<DST_OP>(op, op.getSrcs()[0], op.getSrcs()[1]); \
    return success();                                                          \
  }

    POPULATE_BINARY_OP("__nv_atan2f", math::Atan2Op);
    POPULATE_BINARY_OP("__nv_atan2", math::Atan2Op);
    POPULATE_BINARY_OP("__nv_powf", math::PowFOp);
    POPULATE_BINARY_OP("__nv_pow", math::PowFOp);

#undef POPULATE_BINARY_OP
    return failure();
  }
};

class ExternElementwiseUnaryOpConverter
    : public OpConversionPattern<triton::ExternElementwiseOp> {
  using OpConversionPattern<triton::ExternElementwiseOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(triton::ExternElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // sleef function name is like: Sleef_rintf(numel), need to get rid of the
    // placeholder
    StringRef sym = op.getSymbol().split('(').first;
    if (!op.getPure() || op.getSrcs().size() != 1)
      return failure();

    // Calls to sleef math library
    if (sym.starts_with("Sleef_")) {
      auto moduleOp = op->getParentOfType<ModuleOp>();
      auto funcType =
          rewriter.getFunctionType(op.getSrcs()[0].getType(), op.getType());
      auto funcOp = moduleOp.lookupSymbol<func::FuncOp>(sym);
      if (!funcOp) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        funcOp = rewriter.create<func::FuncOp>(loc, sym, funcType);
        funcOp.setPrivate();
      }
      rewriter.replaceOpWithNewOp<func::CallOp>(op, funcOp,
                                                adaptor.getOperands());
      return success();
    }
#define POPULATE_UNARY_OP(FUNC_NAME, DST_OP)                                   \
  if (!sym.compare(FUNC_NAME)) {                                               \
    rewriter.replaceOpWithNewOp<DST_OP>(op, op.getSrcs()[0]);                  \
    return success();                                                          \
  }

    POPULATE_UNARY_OP("__nv_fabsf", math::AbsFOp);
    POPULATE_UNARY_OP("__nv_fabs", math::AbsFOp);
    POPULATE_UNARY_OP("__nv_sinf", math::SinOp);
    POPULATE_UNARY_OP("__nv_sin", math::SinOp);
    POPULATE_UNARY_OP("__nv_cosf", math::CosOp);
    POPULATE_UNARY_OP("__nv_cos", math::CosOp);
    POPULATE_UNARY_OP("__nv_tanf", math::TanOp);
    POPULATE_UNARY_OP("__nv_tan", math::TanOp);
    POPULATE_UNARY_OP("__nv_asinf", math::AsinOp);
    POPULATE_UNARY_OP("__nv_asin", math::AsinOp);
    POPULATE_UNARY_OP("__nv_acosf", math::AcosOp);
    POPULATE_UNARY_OP("__nv_acos", math::AcosOp);
    POPULATE_UNARY_OP("__nv_atanf", math::AtanOp);
    POPULATE_UNARY_OP("__nv_atan", math::AtanOp);
    POPULATE_UNARY_OP("__nv_sinhf", math::SinhOp);
    POPULATE_UNARY_OP("__nv_sinh", math::SinhOp);
    POPULATE_UNARY_OP("__nv_coshf", math::CoshOp);
    POPULATE_UNARY_OP("__nv_cosh", math::CoshOp);
    POPULATE_UNARY_OP("__nv_tanhf", math::TanhOp);
    POPULATE_UNARY_OP("__nv_tanhf", math::TanhOp);
    POPULATE_UNARY_OP("__nv_acoshf", math::AcoshOp);
    POPULATE_UNARY_OP("__nv_acosh", math::AcoshOp);
    POPULATE_UNARY_OP("__nv_asinhf", math::AsinhOp);
    POPULATE_UNARY_OP("__nv_asinh", math::AsinhOp);
    POPULATE_UNARY_OP("__nv_atanhf", math::AtanhOp);
    POPULATE_UNARY_OP("__nv_atanhf", math::AtanhOp);
    POPULATE_UNARY_OP("__nv_logf", math::LogOp);
    POPULATE_UNARY_OP("__nv_log", math::LogOp);
    POPULATE_UNARY_OP("__nv_log10f", math::Log10Op);
    POPULATE_UNARY_OP("__nv_log10", math::Log10Op);
    POPULATE_UNARY_OP("__nv_log1pf", math::Log1pOp);
    POPULATE_UNARY_OP("__nv_log1p", math::Log1pOp);
    POPULATE_UNARY_OP("__nv_expf", math::ExpOp);
    POPULATE_UNARY_OP("__nv_exp", math::ExpOp);
    POPULATE_UNARY_OP("__nv_exp2f", math::Exp2Op);
    POPULATE_UNARY_OP("__nv_exp2", math::Exp2Op);
    POPULATE_UNARY_OP("__nv_erff", math::ErfOp);
    POPULATE_UNARY_OP("__nv_erf", math::ErfOp);
    POPULATE_UNARY_OP("__nv_sqrtf", math::SqrtOp);
    POPULATE_UNARY_OP("__nv_sqrt", math::SqrtOp);
    POPULATE_UNARY_OP("__nv_rsqrtf", math::RsqrtOp);
    POPULATE_UNARY_OP("__nv_rsqrt", math::RsqrtOp);
    POPULATE_UNARY_OP("__nv_ceilf", math::CeilOp);
    POPULATE_UNARY_OP("__nv_ceil", math::CeilOp);
    POPULATE_UNARY_OP("__nv_floorf", math::FloorOp);
    POPULATE_UNARY_OP("__nv_floor", math::FloorOp);
    POPULATE_UNARY_OP("__nv_truncf", math::TruncOp);
    POPULATE_UNARY_OP("__nv_trunc", math::TruncOp);

#undef POPULATE_UNARY_OP
    return failure();
  }
};

static void populateExternElementwiseOpToMLIROps(RewritePatternSet &patterns) {
  patterns.add<ExternElementwiseBinaryOpConverter,
               ExternElementwiseUnaryOpConverter>(patterns.getContext());
}

} // namespace

#endif
