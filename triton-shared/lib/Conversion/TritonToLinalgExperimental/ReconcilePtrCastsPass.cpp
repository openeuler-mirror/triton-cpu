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

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"

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
    if (auto fromMemref = tritonPtr.getDefiningOp<tptr::FromMemrefOp>())
      memref = fromMemref.getOperand();
    else if (auto tensorType =
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

    } else {
      return rewriter.notifyMatchFailure(op, "unsupported Triton pointer");
    }

    // Use index 0 for rank-1 memrefs for now
    Value idx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value, 1> indices{idx};

    Value atomic;

    if (mask) {
      auto ifOp = rewriter.create<scf::IfOp>(
          loc, mask,
          /*thenBuilder=*/
          [&](OpBuilder &b, Location l) {
            atomic = b.create<memref::AtomicRMWOp>(l, kindAttr, val, memref,
                                                   indices);
            b.create<scf::YieldOp>(l);
          },
          /*elseBuilder=*/
          [&](OpBuilder &b, Location l) { b.create<scf::YieldOp>(l); });

      rewriter.replaceOp(op, atomic);
    } else {
      // Replace with memref.atomic_rmw
      rewriter.replaceOpWithNewOp<memref::AtomicRMWOp>(op, kindAttr, val,
                                                       memref, indices);
    }

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

      ConversionTarget target(getContext());
      target.addIllegalOp<triton::AtomicRMWOp>();
      target.addLegalDialect<arith::ArithDialect>();
      target.addLegalDialect<memref::MemRefDialect>();
      target.addLegalDialect<scf::SCFDialect>();
      target.addLegalDialect<tensor::TensorDialect>();
      target.addLegalDialect<tptr::TPtrDialect>();
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
