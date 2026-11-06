//===- ExpandExp2Fexpa.cpp - Expand math.exp2 into SVE FEXPA sequences ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Rewrites math.exp2 on fixed-length f32 vectors into a 1-ULP-accurate
// FEXPA-based sequence, chunked to SVE container width. The mathematical
// sequence lives here (in the frontend/backend pipeline) instead of in the
// LLVM AArch64 backend: the FEXPA table lookup is issued as a direct call to
// the upstream llvm.aarch64.sve.fexpa.x intrinsic (declared in the module),
// while staging between fixed-length chunks and SVE container registers uses
// the standard vector.scalable.insert/extract operations
// (llvm.vector.insert/extract), which the AArch64 backend folds into pure
// register aliasing when the chunk fits its SVE container under the
// function's vscale_range() guarantee.
//
// This expansion targets the SVE pipeline and must only be applied to code
// that will be compiled with a vscale_range() guarantee covering the chunk
// width (the triton-shared pipeline picks chunk-width = vscale_min * 4).
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/TritonToLinalgExperimental/ExpandExp2Fexpa.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"

namespace {

// The upstream unpredicated FEXPA intrinsic (see e.g.
// llvm/test/CodeGen/AArch64/sve-intrinsics-fexpa.ll): one SVE register of
// i32 lanes in, one SVE register of f32 lanes out. The expansion only ever
// targets f32 container chunks, so a single mangled name covers all uses.
constexpr StringRef kFexpaIntrinsic = "llvm.aarch64.sve.fexpa.x.nxv4f32";

static VectorType fexpaResultType(MLIRContext *ctx) {
  return VectorType::get({4}, Float32Type::get(ctx), /*numScalableDims=*/1);
}

static VectorType fexpaOperandType(MLIRContext *ctx) {
  return VectorType::get({4}, IntegerType::get(ctx, 32), /*numScalableDims=*/1);
}

static bool canExpand(VectorType vecTy, int64_t chunkWidth) {
  if (vecTy.isScalable() || vecTy.getRank() != 1 ||
      !vecTy.getElementType().isF32())
    return false;
  int64_t numElts = vecTy.getNumElements();
  return chunkWidth > 0 && numElts >= chunkWidth &&
         numElts % chunkWidth == 0;
}

// Expand math.exp2 on a fixed-length f32 vector into an FEXPA-based sequence
// on SVE container chunks.
struct ExpandExp2FexpaPattern : public OpRewritePattern<math::Exp2Op> {
  ExpandExp2FexpaPattern(MLIRContext *context, int64_t chunkWidth,
                         PatternBenefit benefit = 1)
      : OpRewritePattern<math::Exp2Op>(context, benefit),
        chunkWidth(chunkWidth) {}

  LogicalResult matchAndRewrite(math::Exp2Op op,
                                PatternRewriter &rewriter) const override {
    auto vecTy = dyn_cast<VectorType>(op.getType());
    if (!vecTy || !canExpand(vecTy, chunkWidth))
      return failure();

    int64_t numElts = vecTy.getNumElements();

    Location loc = op.getLoc();
    Value x = op.getOperand();

    // A vector<[4]xf32> container holds one SVE register's worth of f32
    // lanes; chunkWidth lanes must fit into it under the function's
    // vscale_range guarantee.
    VectorType chunkTy = VectorType::get({chunkWidth}, rewriter.getF32Type());
    VectorType containerTy = fexpaResultType(rewriter.getContext());
    VectorType containerI32Ty = fexpaOperandType(rewriter.getContext());

    auto makeSplat = [&](double v) -> Value {
      return rewriter.create<arith::ConstantOp>(
          loc, DenseFPElementsAttr::get(containerTy,
                                        APFloat(static_cast<float>(v))));
    };

    constexpr double shift = 196735.0; // 1.5 * 2^17 + 127

    // An SVE container register to stage fixed-length chunks in. The insert
    // below fully covers the container's minimum size, so the undefined base
    // is dropped by the backend.
    Value undefContainer = rewriter.create<LLVM::UndefOp>(loc, containerTy);

    // Expand a single chunk of chunkWidth fixed f32 lanes into the FEXPA
    // sequence operating on one SVE register.
    auto expandChunk = [&](Value chunkFixed) -> Value {
      // Stage the fixed-length chunk in its SVE container register.
      Value chunk = rewriter.create<vector::ScalableInsertOp>(
          loc, chunkFixed, undefContainer, 0);

      // Clamp to [-126, 127] so that z = x + 196735.0 = 1.5 * 2^17 + 127 + r
      // carries r in the mantissa bits and n in the exponent bits.
      Value clamped = rewriter.create<arith::MinimumFOp>(loc, chunk,
                                                         makeSplat(127.0));
      clamped =
          rewriter.create<arith::MaximumFOp>(loc, clamped, makeSplat(-126.0));
      Value z = rewriter.create<arith::AddFOp>(loc, clamped, makeSplat(shift));
      Value zi =
          rewriter.create<arith::BitcastOp>(loc, containerI32Ty, z);
      Value scale = rewriter.create<LLVM::CallOp>(loc, containerTy,
                                                  kFexpaIntrinsic, zi)
                        ->getResult(0);

      // Correction: 2^r - 1 = t + t^2/2 + t^3/6 with t = r * ln2,
      // |t| < ln2 / 128.
      Value n =
          rewriter.create<arith::SubFOp>(loc, z, makeSplat(shift));
      Value r = rewriter.create<arith::SubFOp>(loc, clamped, n);
      Value t = rewriter.create<arith::MulFOp>(
          loc, r, makeSplat(0.6931471805599453));
      Value p = rewriter.create<math::FmaOp>(loc, containerTy, t,
                                             makeSplat(1.0 / 6.0),
                                             makeSplat(0.5));
      p = rewriter.create<math::FmaOp>(loc, containerTy, t, p,
                                       makeSplat(1.0));
      p = rewriter.create<arith::MulFOp>(loc, t, p);
      Value res =
          rewriter.create<math::FmaOp>(loc, containerTy, scale, p, scale);

      // Move the result back into the fixed-length domain.
      return rewriter.create<vector::ScalableExtractOp>(loc, chunkTy, res, 0);
    };

    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(rewriter.getF32Type(), 0.0));
    Value result = rewriter.create<vector::SplatOp>(loc, vecTy, zero);
    for (int64_t i = 0; i < numElts; i += chunkWidth) {
      Value chunk = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, x, ArrayRef<int64_t>{i}, ArrayRef<int64_t>{chunkWidth},
          ArrayRef<int64_t>{1});
      chunk = expandChunk(chunk);
      result = rewriter.create<vector::InsertStridedSliceOp>(
          loc, chunk, result, ArrayRef<int64_t>{i}, ArrayRef<int64_t>{1});
    }

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  int64_t chunkWidth;
};

struct ExpandExp2FexpaPass: public ExpandExp2FexpaBase<ExpandExp2FexpaPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
      ModuleOp module = getOperation();
      MLIRContext *ctx = &getContext();

      // Declare the FEXPA intrinsic once, if any op in the module will be
      // expanded.
      bool anyExp2 = false;
      module->walk([&](math::Exp2Op op) {
        if (auto vecTy = dyn_cast<VectorType>(op.getType()))
          anyExp2 |= canExpand(vecTy, chunkWidth);
      });
      if (anyExp2 && !module.lookupSymbol<LLVM::LLVMFuncOp>(kFexpaIntrinsic)) {
        OpBuilder builder(ctx);
        builder.setInsertionPointToStart(module.getBody());
        builder.create<LLVM::LLVMFuncOp>(
            builder.getUnknownLoc(), kFexpaIntrinsic,
            LLVM::LLVMFunctionType::get(fexpaResultType(ctx),
                                        {fexpaOperandType(ctx)}));
      }

      RewritePatternSet patterns(ctx);
      patterns.add<ExpandExp2FexpaPattern>(ctx, chunkWidth);
      GreedyRewriteConfig config;
      if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                       config)))
        signalPassFailure();
    }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createExpandExp2FexpaPass() {
  return std::make_unique<ExpandExp2FexpaPass>();
}
