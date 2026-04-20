//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//
// Throughout the conversion process, we convert !tt.ptr -> {!ptr.ptr or
// memref<*>}. This process leaves around unrealized_conversion_cast ops between
// these types. We want to remove these unrealized casts and use the proper
// conversion ops in the PtrDialect: to_memref or from_memref. To do this, we
// use a pattern that simplifies the chain of conversions by removing
// intermediate conversion cast ops. At the end, we are left with just pointer
// to memref or vice versa. We then convert the unrealized cast to to_memref or
// from_memref accordingly.
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrAttrs.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/ReconcilePtrCasts.h"

#include "triton-shared/Conversion/TritonToLinalgExperimental/ReconcilePtrCasts.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"

#define DEBUG_TYPE "triton-to-linalg"

namespace {

static bool isOneToOneCast(UnrealizedConversionCastOp op) {
  return (op.getInputs().size() == 1 && op->getNumResults() == 1);
}

struct SimplifyUnrealizedCast
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  SimplifyUnrealizedCast(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<UnrealizedConversionCastOp>(context, benefit) {}

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (!isOneToOneCast(op)) {
      return failure();
    }
    auto in = op.getInputs().front();

    if (auto unrealizedCast = in.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (!isOneToOneCast(unrealizedCast)) {
        return failure();
      }

      auto prevInput = unrealizedCast.getInputs().front();
      auto newCast = rewriter.create<UnrealizedConversionCastOp>(
          op->getLoc(), op->getResultTypes(), ValueRange{prevInput});

      rewriter.replaceOp(op, newCast);
      return success();
    }
    return failure();
  }
};

struct FromMemrefConverter
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  FromMemrefConverter(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<UnrealizedConversionCastOp>(context, benefit) {}

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (!isOneToOneCast(op)) {
      return failure();
    }

    Location loc = op.getLoc();
    auto input = op.getInputs().front();
    auto unrankedInput = dyn_cast<UnrankedMemRefType>(input.getType());
    auto output = op.getResult(0);
    auto outType = output.getType();

    if (unrankedInput && isa<triton::PointerType, ptr::PtrType>(outType)) {
      // from_memref only takes ranked memref, cast the unranked memref to
      // ranked memref first.
      Type elementTy;
      if (auto ttPtr = dyn_cast<triton::PointerType>(outType))
        elementTy = ttPtr.getPointeeType();
      else if (auto genericPtr = dyn_cast<ptr::PtrType>(outType))
        elementTy = genericPtr.getElementType();
      else
        return failure();

      Value rankedMemref = rewriter.create<memref::CastOp>(
          op.getLoc(), MemRefType::get({1}, unrankedInput.getElementType()),
          input);

      if (elementTy && elementTy != unrankedInput.getElementType()) {
        // Insert an unrealized conversion cast to match element type
        auto castOp = rewriter.create<UnrealizedConversionCastOp>(
            loc, MemRefType::get({1}, elementTy), rankedMemref);

        // Use the result of the cast op
        rankedMemref = castOp.getResult(0);
      }

      auto memrefToPtr = rewriter.create<tptr::FromMemrefOp>(
          op->getLoc(),
          ptr::PtrType::get(rewriter.getContext(),
                            ptr::GenericSpaceAttr::get(rewriter.getContext())),
          rankedMemref);

      rewriter.replaceAllUsesWith(output, memrefToPtr);
      rewriter.eraseOp(op);

      return success();
    }

    return failure();
  }
};

struct ToMemrefConverter : public OpRewritePattern<UnrealizedConversionCastOp> {
  ToMemrefConverter(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<UnrealizedConversionCastOp>(context, benefit) {}

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (!isOneToOneCast(op)) {
      return failure();
    }
    auto input = op.getInputs().front();
    auto inType = input.getType();
    auto output = op.getResult(0);
    auto outUnrankedMemrefType = dyn_cast<UnrankedMemRefType>(output.getType());
    if (isa<triton::PointerType, ptr::PtrType>(inType) &&
        outUnrankedMemrefType) {
      // to_memref can only cast to ranked static shape memref, we have to cast
      // the resulting memref back to unranked
      auto elemType = outUnrankedMemrefType.getElementType();
      auto ptrToMemref = rewriter.create<tptr::ToMemrefOp>(
          op->getLoc(), MemRefType::get({1}, elemType), input);

      SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(1)};
      SmallVector<OpFoldResult> newStrides = {rewriter.getIndexAttr(1)};
      auto newUnrankedMemref = rewriter.create<memref::ReinterpretCastOp>(
          op->getLoc(), MemRefType::get({ShapedType::kDynamic}, elemType),
          ptrToMemref, rewriter.getIndexAttr(0), sizes, newStrides);

      rewriter.replaceAllUsesWith(output, newUnrankedMemref);
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};

static std::optional<arith::AtomicRMWKind> mapTritonToMLIR(uint32_t triKind) {
  switch (triKind) {
  case 1:
    return arith::AtomicRMWKind::andi; // Triton AND
  case 2:
    return arith::AtomicRMWKind::ori; // Triton OR
  case 3:
    return std::nullopt; // Triton XOR not supported
  case 4:
    return arith::AtomicRMWKind::addi; // Triton ADD
  case 5:
    return arith::AtomicRMWKind::addf; // Triton FADD
  case 6:
    return arith::AtomicRMWKind::maxs; // Triton signed max
  case 7:
    return arith::AtomicRMWKind::mins; // Triton signed min
  case 8:
    return arith::AtomicRMWKind::maxu; // Triton unsigned max
  case 9:
    return arith::AtomicRMWKind::minu; // Triton unsigned min
  case 10:
    return arith::AtomicRMWKind::assign; // Triton XCHG → assign
  default:
    return std::nullopt;
  }
}

struct AtomicrmwConverter : public OpRewritePattern<triton::AtomicRMWOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::AtomicRMWOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto val = op.getVal();
    Value tritonPtr = op.getPtr();
    auto mask = op.getMask();

    // Get Triton's atomic kind integer
    auto kindIntAttr = op->getAttrOfType<IntegerAttr>("atomic_rmw_op");
    if (!kindIntAttr)
      return rewriter.notifyMatchFailure(op, "missing Triton atomic kind");

    uint32_t triKind = kindIntAttr.getInt();
    auto mlirKindOpt = mapTritonToMLIR(triKind);
    if (!mlirKindOpt)
      return rewriter.notifyMatchFailure(op, "unsupported Triton atomic kind");

    auto kindAttr =
        arith::AtomicRMWKindAttr::get(rewriter.getContext(), *mlirKindOpt);

    // Recover memref from Triton pointer
    Value memref;
    SmallVector<Value, 1> indices;
    auto resolvePtr = [&](Value ptr, Value offset) -> LogicalResult {
      if (auto castOp = ptr.getDefiningOp<UnrealizedConversionCastOp>()) {
        memref = castOp.getInputs()[0];
        indices.push_back(rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getIndexType(), offset));
        return success();
      }
      if (auto fromMemRef = ptr.getDefiningOp<tptr::FromMemrefOp>()) {
        memref = fromMemRef.getOperand();
        indices.push_back(rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getIndexType(), offset));
        return success();
      }
      return rewriter.notifyMatchFailure(op,
                                         "unsupported base ptr for addptr op");
    };

    if (auto fromMemref = tritonPtr.getDefiningOp<tptr::FromMemrefOp>()) {
      indices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
      memref = fromMemref.getOperand();
    } else if (auto tensorType =
                   dyn_cast<RankedTensorType>(tritonPtr.getType())) {
      // shapes and rank
      SmallVector<int64_t> shape(tensorType.getShape().begin(),
                                 tensorType.getShape().end());
      unsigned rank = tensorType.getRank();

      // pointer element type (e.g. !tt.ptr<f32> or !ptr.ptr<#...>)
      Type lanePtrElemTy = tensorType.getElementType();
      Type pointeeTy = nullptr;
      if (auto ttPtr = mlir::dyn_cast<triton::PointerType>(lanePtrElemTy)) {
        pointeeTy = ttPtr.getPointeeType();
      } else if (auto genericPtr =
                     mlir::dyn_cast<ptr::PtrType>(lanePtrElemTy)) {
        pointeeTy = genericPtr.getElementType();
      } else {
        return rewriter.notifyMatchFailure(op,
                                           "unsupported pointer element type");
      }

      // the result tensor type should match 'val'
      auto resultTensorTy = val.getType().dyn_cast<RankedTensorType>();
      if (!resultTensorTy)
        return rewriter.notifyMatchFailure(
            op, "expected ranked tensor as value/result");

      // === 2) Prepare loop bounds as Values (ValueRange expected by
      // buildLoopNest) ===
      SmallVector<Value> lbs, ubs, steps;
      lbs.reserve(rank);
      ubs.reserve(rank);
      steps.reserve(rank);
      for (unsigned i = 0; i < rank; ++i) {
        lbs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
        // shape[i] may be dynamic (-1) — for static shapes we use constant
        // index; if dynamic, you'd need to materialize the dynamic bound. Here
        // we assume static.
        if (shape[i] == ShapedType::kDynamic) {
          return rewriter.notifyMatchFailure(
              op, "dynamic shapes not supported in lowering yet");
        }
        ubs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, shape[i]));
        steps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
      }

      // === 3) Allocate a result memref to collect scalar results per-lane ===
      Type resultElemTy = resultTensorTy.getElementType();

      // 1) Create an empty tensor to accumulate results
      Value resultEmpty = rewriter.create<tensor::EmptyOp>(
          loc,
          SmallVector<int64_t>(tensorType.getShape().begin(),
                               tensorType.getShape().end()),
          resultTensorTy.getElementType());

      scf::LoopNest nest = scf::buildLoopNest(
          rewriter, loc, lbs, ubs, steps,
          /*iterArgs=*/ValueRange{resultEmpty},
          [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange ivs,
              ValueRange iterArgs) -> scf::ValueVector {
            // iterArgs[0] = current accumulator tensor
            Value currentTensor = iterArgs[0];

            // Unwrap pointer tensor if necessary
            Value ttPtr = tritonPtr;
            if (auto cast =
                    tritonPtr.getDefiningOp<UnrealizedConversionCastOp>())
              ttPtr = cast.getOperand(0);

            // 1. Extract pointer element for this lane
            Value lanePtr =
                nestedBuilder.create<tensor::ExtractOp>(nestedLoc, ttPtr, ivs);

            // 2. Convert lane pointer to memref<1xf32>
            auto elemTy = val.getType().cast<RankedTensorType>().getElementType();
            Value laneMemref = nestedBuilder.create<tptr::ToMemrefOp>(
                nestedLoc, MemRefType::get({1}, elemTy), lanePtr);

            // 3. Extract the value element for this lane
            Value laneVal =
                nestedBuilder.create<tensor::ExtractOp>(nestedLoc, val, ivs);

            // 4. Perform the atomic RMW op
            Value zeroIdx =
                nestedBuilder.create<arith::ConstantIndexOp>(nestedLoc, 0);
            SmallVector<Value, 1> memIndices{zeroIdx};
            auto atomic = nestedBuilder.create<memref::AtomicRMWOp>(
                nestedLoc, kindAttr, laneVal, laneMemref, memIndices);

            // 5. Insert the atomic result back into the current tensor
            Value updatedTensor = nestedBuilder.create<tensor::InsertOp>(
                nestedLoc, atomic.getResult(), currentTensor, ivs);

            // Yield the updated tensor as the new iter_arg
            return {updatedTensor};
          });

      // Use the loop result as the final tensor
      Value resultTensor = nest.results.front();

      // Replace the Triton op with the result tensor
      rewriter.replaceOp(op, resultTensor);
      return success();

    } else if (auto castOp =
                   tritonPtr.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (auto addPtrOp =
              castOp.getOperand(0).getDefiningOp<triton::AddPtrOp>()) {
        if (failed(resolvePtr(addPtrOp.getPtr(), addPtrOp.getOffset())))
          return rewriter.notifyMatchFailure(op, "unsupported addptr");
      } else if (auto addPtrOp =
                     castOp.getOperand(0).getDefiningOp<tptr::PtrAddOp>()) {
        if (failed(resolvePtr(addPtrOp.getOperand(0), addPtrOp.getOffset())))
          return rewriter.notifyMatchFailure(op, "unsupported addptr");
      } else
        return rewriter.notifyMatchFailure(op, "unsupported cast");
    } else if (auto addPtrOp = tritonPtr.getDefiningOp<triton::AddPtrOp>()) {
      if (failed(resolvePtr(addPtrOp.getPtr(), addPtrOp.getOffset())))
        return rewriter.notifyMatchFailure(op, "unsupported addptr");
    } else if (auto addPtrOp = tritonPtr.getDefiningOp<tptr::PtrAddOp>()) {
      if (failed(resolvePtr(addPtrOp.getOperand(0), addPtrOp.getOffset())))
        return rewriter.notifyMatchFailure(op, "unsupported addptr");
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported Triton pointer");
    }

    if (mask) {
      auto ifOp = rewriter.create<scf::IfOp>(
          loc, mask,
          /*thenBuilder=*/
          [&](OpBuilder &b, Location l) {
            Value atomic = b.create<memref::AtomicRMWOp>(l, kindAttr, val,
                                                         memref, indices);
            b.create<scf::YieldOp>(l, atomic);
          },
          /*elseBuilder=*/
          // Yield a default value to make sure ifOp is defined in every branch.
          [&](OpBuilder &b, Location l) {
            Value defaultVal = b.create<arith::ConstantOp>(
                loc, rewriter.getZeroAttr(val.getType()));
            b.create<scf::YieldOp>(l, defaultVal);
          });

      rewriter.replaceOp(op, ifOp.getResult(0));
    } else {
      // Replace with memref.atomic_rmw
      rewriter.replaceOpWithNewOp<memref::AtomicRMWOp>(op, kindAttr, val,
                                                       memref, indices);
    }

    return success();
  }
};

struct AtomicCASOpConversion : public OpConversionPattern<AtomicCASOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto replaceVal = op.getVal();
    auto compareVal = op.getCmp();
    Value tritonPtr = op.getPtr();

    // Recover memref from Triton pointer
    Value memref;
    if (auto fromMemref = tritonPtr.getDefiningOp<tptr::FromMemrefOp>())
      memref = fromMemref.getOperand();
    else if (auto tensorType =
                 dyn_cast<RankedTensorType>(tritonPtr.getType())) {
      // shapes and rank
      SmallVector<int64_t> shape(tensorType.getShape().begin(),
                                 tensorType.getShape().end());
      unsigned rank = tensorType.getRank();

      // the result tensor type should match 'val'
      auto resultTensorTy = replaceVal.getType().dyn_cast<RankedTensorType>();
      if (!resultTensorTy)
        return rewriter.notifyMatchFailure(
            op, "expected ranked tensor as value/result");

      // ==Prepare loop bounds as Values (ValueRange expected by
      // buildLoopNest) ===
      SmallVector<Value> lbs, ubs, steps;
      lbs.reserve(rank);
      ubs.reserve(rank);
      steps.reserve(rank);
      for (unsigned i = 0; i < rank; ++i) {
        lbs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
        // shape[i] may be dynamic (-1) — for static shapes we use constant
        // index; if dynamic, you'd need to materialize the dynamic bound. Here
        // we assume static.
        if (shape[i] == ShapedType::kDynamic) {
          return rewriter.notifyMatchFailure(
              op, "dynamic shapes not supported in lowering yet");
        }
        ubs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, shape[i]));
        steps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
      }

      // Unwrap pointer tensor if necessary
      Value ttPtr = tritonPtr;
      if (auto cast =
                  tritonPtr.getDefiningOp<UnrealizedConversionCastOp>())
          ttPtr = cast.getOperand(0);

      // Allocate an empty result tensor to carry per-lane CAS old-values
      // through the loop as an iter-arg.
      Value resultEmpty = rewriter.create<tensor::EmptyOp>(
          loc, shape, resultTensorTy.getElementType());

      scf::LoopNest nest = scf::buildLoopNest(
          rewriter, loc, lbs, ubs, steps, ValueRange{resultEmpty},
          [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange ivs,
              ValueRange iterArgs) -> scf::ValueVector {
            Value currentTensor = iterArgs[0];

            // 1. Extract pointer element for this lane
            Value lanePtr =
                nestedBuilder.create<tensor::ExtractOp>(nestedLoc, ttPtr, ivs);

            // 2. Convert lane pointer to memref<1xtype>
            auto elemTy = replaceVal.getType().cast<RankedTensorType>().getElementType();
            Value laneMemref = nestedBuilder.create<tptr::ToMemrefOp>(
                nestedLoc, MemRefType::get({1}, elemTy), lanePtr);

            // 3. Extract the replace value element for this lane
            Value laneReplaceVal =
                nestedBuilder.create<tensor::ExtractOp>(nestedLoc, replaceVal, ivs);

            // 4. Extract the compare value for this lane
            Value laneCmpVal =
                nestedBuilder.create<tensor::ExtractOp>(nestedLoc, compareVal, ivs);

            Value zeroIdx =
                nestedBuilder.create<arith::ConstantIndexOp>(nestedLoc, 0);

            // 5. Create the GenericAtomicRMWOp for cmpxchg atomic operation
            SmallVector<Value, 1> memIndices{zeroIdx};
            auto genericOp = nestedBuilder.create<memref::GenericAtomicRMWOp>(
                   nestedLoc, laneMemref, memIndices);
            Value currentValue = genericOp.getCurrentValue();

            OpBuilder bodyBuilder =
              OpBuilder::atBlockEnd(genericOp.getBody(), rewriter.getListener());

            // arith.cmpi requires integer operands. For float element types,
            // bitcast to an equal-width integer before comparing.  This also
            // matches hardware cmpxchg bit-pattern equality semantics.
            Value isEqual;
            if (auto floatTy = dyn_cast<FloatType>(elemTy)) {
              Type intTy = IntegerType::get(bodyBuilder.getContext(),
                                            floatTy.getWidth());
              Value curInt = bodyBuilder.create<arith::BitcastOp>(
                  nestedLoc, intTy, currentValue);
              Value cmpInt = bodyBuilder.create<arith::BitcastOp>(
                  nestedLoc, intTy, laneCmpVal);
              isEqual = bodyBuilder.create<arith::CmpIOp>(
                  nestedLoc, arith::CmpIPredicate::eq, curInt, cmpInt);
            } else {
              isEqual = bodyBuilder.create<arith::CmpIOp>(
                  nestedLoc, arith::CmpIPredicate::eq, currentValue,
                  laneCmpVal);
            }

            auto result = bodyBuilder.create<arith::SelectOp>(nestedLoc, isEqual,
                laneReplaceVal, currentValue);

            bodyBuilder.create<memref::AtomicYieldOp>(nestedLoc, result);

            // 6. Insert the scalar CAS result into the accumulating tensor.
            Value updatedTensor = nestedBuilder.create<tensor::InsertOp>(
                nestedLoc, genericOp.getResult(), currentTensor, ivs);
            return {updatedTensor};
          });
      rewriter.replaceOp(op, nest.results.front());
      return success();
    } else if (auto castOp =
                   tritonPtr.getDefiningOp<UnrealizedConversionCastOp>()) {
      // Scalar pointer via TritonToPtr: unrealized_conversion_cast from
      // !ptr.ptr back to !tt.ptr<T>.  Recover a memref via ToMemrefOp.
      Value ptrVal = castOp.getInputs().front();
      auto pointeeTy =
          cast<triton::PointerType>(tritonPtr.getType()).getPointeeType();
      memref = rewriter.create<tptr::ToMemrefOp>(
          loc, MemRefType::get({1}, pointeeTy), ptrVal);
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported Triton pointer");
    }
    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value, 1> memIndices{zeroIdx};
    auto genericOp = rewriter.create<memref::GenericAtomicRMWOp>(
                   loc, memref, memIndices);
    Value currentValue = genericOp.getCurrentValue();

    OpBuilder bodyBuilder =
        OpBuilder::atBlockEnd(genericOp.getBody(), rewriter.getListener());

    // arith.cmpi requires integer operands (see tensor path comment above).
    auto elemTy = replaceVal.getType();
    Value isEqual;
    if (auto floatTy = dyn_cast<FloatType>(elemTy)) {
      Type intTy =
          IntegerType::get(bodyBuilder.getContext(), floatTy.getWidth());
      Value curInt =
          bodyBuilder.create<arith::BitcastOp>(loc, intTy, currentValue);
      Value cmpInt =
          bodyBuilder.create<arith::BitcastOp>(loc, intTy, compareVal);
      isEqual = bodyBuilder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  curInt, cmpInt);
    } else {
      isEqual = bodyBuilder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  currentValue, compareVal);
    }

    auto result = bodyBuilder.create<arith::SelectOp>(loc, isEqual,
        replaceVal, currentValue);
    bodyBuilder.create<memref::AtomicYieldOp>(loc, result);
    rewriter.replaceOp(op, genericOp.getResult());
    return success();
  }
};

class ReconcilePtrCastsPass
    : public ReconcilePtrCastsBase<ReconcilePtrCastsPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tptr::TPtrDialect, memref::MemRefDialect, BuiltinDialect,
                    arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // === Phase 1: Greedy rewrites ===
    {
      RewritePatternSet greedyPatterns(&getContext());
      greedyPatterns
          .add<SimplifyUnrealizedCast, FromMemrefConverter, ToMemrefConverter>(
              &getContext());

      if (failed(applyPatternsAndFoldGreedily(moduleOp,
                                              std::move(greedyPatterns)))) {
        signalPassFailure();
        return;
      }
    }

    // === Phase 2: Conversion patterns ===
    {
      RewritePatternSet conversionPatterns(&getContext());
      conversionPatterns.add<AtomicrmwConverter>(&getContext());
      conversionPatterns.add<AtomicCASOpConversion>(&getContext());

      ConversionTarget target(getContext());
      target.addIllegalOp<triton::AtomicRMWOp>();
      target.addLegalDialect<arith::ArithDialect>();
      target.addLegalDialect<memref::MemRefDialect>();
      target.addLegalDialect<scf::SCFDialect>();
      target.addLegalDialect<tensor::TensorDialect>();
      target.addLegalDialect<tptr::TPtrDialect>();
      target.addLegalDialect<LLVM::LLVMDialect>();
      target.addLegalDialect<BuiltinDialect>();

      if (failed(applyPartialConversion(moduleOp, target,
                                        std::move(conversionPatterns)))) {
        signalPassFailure();
        return;
      }
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createReconcilePtrCastsPass() {
  return std::make_unique<ReconcilePtrCastsPass>();
}
