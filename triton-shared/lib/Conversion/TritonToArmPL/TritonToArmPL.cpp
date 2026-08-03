//===----------------------------------------------------------------------===//
//
// Two passes: TritonDotToArmPLOp converts tt.dot to tts.armpl_matmul/gemv,
// and ArmPLOpToFunction lowers those ops to func.call @cblas_*.
//
// ArmPLOpToFunction does its own bufferization (alloc + materialize) because
// the generic OneShotBufferize pass doesn't understand custom tts ops.
// A future improvement would implement BufferizableOpInterface to defer
// bufferization to the standard pipeline. This would require the tts.armpl_matmul/gemv
// to be legal to the point where they can be bufferized, which is not
// currently the case.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton-shared/Conversion/TritonToArmPL/TritonDotToArmPLOpConversion.hpp"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonToArmPL/Passes.h.inc"

namespace {

/// Emit an LLVM function declaration for the ArmPL BLAS function.
///
/// We create an llvm.func (not func.func) because the argument types already
/// use LLVM pointer types (!llvm.ptr).  The ConvertFuncToLLVM pass skips
/// func.func declarations whose signatures are already in LLVM types, leaving
/// them as func.func — which mlir-translate cannot lower.  By emitting
/// llvm.func / llvm.call directly we avoid that gap entirely.
static LLVM::LLVMFuncOp getOrCreateBLASFunc(ModuleOp module, const char *name,
                                            Type returnType,
                                            ArrayRef<Type> argTypes,
                                            OpBuilder &builder) {
  if (auto existing = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return existing;
  auto funcType = LLVM::LLVMFunctionType::get(returnType, argTypes);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto funcOp = builder.create<LLVM::LLVMFuncOp>(
      module.getLoc(), name, funcType);
  funcOp.setPrivate();
  return funcOp;
}

/// Allocate a memref and copy tensor data into it.
static Value bufferizeTensor(Value tensor, Location loc, OpBuilder &builder) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto memrefType =
      MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  auto alloc = builder.create<memref::AllocOp>(loc, memrefType);
  auto store = builder.create<bufferization::MaterializeInDestinationOp>(
      loc, tensor, alloc);
  store.setWritable(true);
  return alloc;
}

/// Copy a memref back to a tensor.  Do NOT dealloc here — the bufferization
/// framework (e.g. OneShotBufferize) manages deallocation later.
static Value memrefToTensor(Value memref, Location loc, OpBuilder &builder) {
  auto memrefType = cast<MemRefType>(memref.getType());
  auto tensorType = RankedTensorType::get(memrefType.getShape(),
                                          memrefType.getElementType());
  return builder.create<bufferization::ToTensorOp>(
      loc, tensorType, memref, /*restrict=*/true, /*writable=*/false);
}

static Value extractPtr(Value memref, Location loc, OpBuilder &builder) {
  Value alignedPtr = builder.create<memref::ExtractAlignedPointerAsIndexOp>(
      loc, memref);
  Value i64Ptr = builder.create<arith::IndexCastOp>(
      loc, builder.getI64Type(), alignedPtr);
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  return builder.create<LLVM::IntToPtrOp>(loc, ptrType, i64Ptr);
}

static Value getLeadingDimI32(Value memref, Location loc, OpBuilder &builder) {
  auto metadata = builder.create<memref::ExtractStridedMetadataOp>(loc, memref);
  Value stride0 = metadata.getStrides()[0];
  return builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), stride0);
}

static Value castToI32(Value v, Location loc, OpBuilder &builder) {
  return builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), v);
}

static Value getOneI32(Location loc, OpBuilder &builder) {
  return builder.create<arith::ConstantIntOp>(loc, 1, builder.getI32Type());
}

static Value createFloatConst(Location loc, const llvm::APFloat &value,
                              Type type, OpBuilder &builder) {
  llvm::APFloat converted(value);
  bool losesInfo;
  auto &targetSem = type.isF64() ? llvm::APFloat::IEEEdouble()
                                 : llvm::APFloat::IEEEsingle();
  converted.convert(targetSem, llvm::APFloat::rmNearestTiesToEven, &losesInfo);
  return builder.create<arith::ConstantOp>(
      loc, type, builder.getFloatAttr(type, converted));
}

//===----------------------------------------------------------------------===//
// ArmPLMatmulLowering
//===----------------------------------------------------------------------===//
struct ArmPLMatmulLowering : public OpRewritePattern<tts::ArmPLMatmulOp> {
  using OpRewritePattern<tts::ArmPLMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tts::ArmPLMatmulOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto aType = cast<RankedTensorType>(op.getA().getType());
    bool isDouble = aType.getElementType().isF64();
    auto i32Type = rewriter.getI32Type();

    // Bufferize tensors → memrefs
    Value aMem = bufferizeTensor(op.getA(), loc, rewriter);
    Value bMem = bufferizeTensor(op.getB(), loc, rewriter);
    Value cMem = bufferizeTensor(op.getC(), loc, rewriter);

    // Extract pointers
    Value aPtr = extractPtr(aMem, loc, rewriter);
    Value bPtr = extractPtr(bMem, loc, rewriter);
    Value cPtr = extractPtr(cMem, loc, rewriter);

    // Get leading dimensions as i32
    Value lda = getLeadingDimI32(aMem, loc, rewriter);
    Value ldb = getLeadingDimI32(bMem, loc, rewriter);
    Value ldc = getLeadingDimI32(cMem, loc, rewriter);

    // Get M, N, K as i32
    Value M = rewriter.create<memref::DimOp>(loc, cMem, 0);
    Value N = rewriter.create<memref::DimOp>(loc, cMem, 1);
    int32_t transA = op.getTransA();
    Value K = rewriter.create<memref::DimOp>(loc, aMem, (transA == 112) ? 0 : 1);
    Value m32 = castToI32(M, loc, rewriter);
    Value n32 = castToI32(N, loc, rewriter);
    Value k32 = castToI32(K, loc, rewriter);

    // BLAS enum constants
    auto cblasRowMajor = rewriter.create<arith::ConstantIntOp>(loc, 101, i32Type);
    auto transAVal = rewriter.create<arith::ConstantIntOp>(loc, op.getTransA(), i32Type);
    auto transBVal = rewriter.create<arith::ConstantIntOp>(loc, op.getTransB(), i32Type);

    // Alpha and beta
    auto scalarType = isDouble ? rewriter.getF64Type() : rewriter.getF32Type();
    Value alphaVal = createFloatConst(loc, op.getAlpha(), scalarType, rewriter);
    Value betaVal = createFloatConst(loc, op.getBeta(), scalarType, rewriter);

    // Create cblas_sgemm / cblas_dgemm call
    const char *funcName = isDouble ? "cblas_dgemm" : "cblas_sgemm";
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto funcOp = getOrCreateBLASFunc(
        module, funcName, voidType,
        {i32Type, i32Type, i32Type, i32Type, i32Type, i32Type,
         scalarType, ptrType, i32Type, ptrType, i32Type,
         scalarType, ptrType, i32Type},
        rewriter);

    SmallVector<Value> args = {cblasRowMajor, transAVal, transBVal,
                               m32, n32, k32,
                               alphaVal, aPtr, lda,
                               bPtr, ldb,
                               betaVal, cPtr, ldc};
    rewriter.create<LLVM::CallOp>(
        loc, funcOp.getFunctionType(),
        SymbolRefAttr::get(rewriter.getContext(), funcName),
        args);

    // Convert c memref back to tensor
    Value result = memrefToTensor(cMem, loc, rewriter);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ArmPLGemvLowering
//===----------------------------------------------------------------------===//
struct ArmPLGemvLowering : public OpRewritePattern<tts::ArmPLGemvOp> {
  using OpRewritePattern<tts::ArmPLGemvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tts::ArmPLGemvOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto aType = cast<RankedTensorType>(op.getA().getType());
    bool isDouble = aType.getElementType().isF64();
    auto i32Type = rewriter.getI32Type();

    // Bufferize tensors → memrefs
    Value aMem = bufferizeTensor(op.getA(), loc, rewriter);
    Value xMem = bufferizeTensor(op.getX(), loc, rewriter);
    Value yMem = bufferizeTensor(op.getY(), loc, rewriter);

    // Extract pointers
    Value aPtr = extractPtr(aMem, loc, rewriter);
    Value xPtr = extractPtr(xMem, loc, rewriter);
    Value yPtr = extractPtr(yMem, loc, rewriter);

    // Get leading dimension for A as i32
    Value lda = getLeadingDimI32(aMem, loc, rewriter);

    // Get M, N as i32
    Value M = rewriter.create<memref::DimOp>(loc, aMem, 0);
    Value N = rewriter.create<memref::DimOp>(loc, aMem, 1);
    Value m32 = castToI32(M, loc, rewriter);
    Value n32 = castToI32(N, loc, rewriter);

    // BLAS constants
    auto cblasRowMajor = rewriter.create<arith::ConstantIntOp>(loc, 101, i32Type);
    auto transAVal = rewriter.create<arith::ConstantIntOp>(loc, op.getTransA(), i32Type);
    auto incx = getOneI32(loc, rewriter);
    auto incy = getOneI32(loc, rewriter);

    // Alpha and beta
    auto scalarType = isDouble ? rewriter.getF64Type() : rewriter.getF32Type();
    Value alphaVal = createFloatConst(loc, op.getAlpha(), scalarType, rewriter);
    Value betaVal = createFloatConst(loc, op.getBeta(), scalarType, rewriter);

    // Create cblas_sgemv / cblas_dgemv call
    const char *funcName = isDouble ? "cblas_dgemv" : "cblas_sgemv";
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto funcOp = getOrCreateBLASFunc(
        module, funcName, voidType,
        {i32Type, i32Type, i32Type, i32Type,
         scalarType, ptrType, i32Type, ptrType, i32Type,
         scalarType, ptrType, i32Type},
        rewriter);

    SmallVector<Value> args = {cblasRowMajor, transAVal, m32, n32,
                               alphaVal, aPtr, lda,
                               xPtr, incx,
                               betaVal, yPtr, incy};
    rewriter.create<LLVM::CallOp>(
        loc, funcOp.getFunctionType(),
        SymbolRefAttr::get(rewriter.getContext(), funcName),
        args);

    // Convert y memref back to tensor
    Value result = memrefToTensor(yMem, loc, rewriter);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct TritonDotToArmPLOpPass
    : public TritonDotToArmPLOpBase<TritonDotToArmPLOpPass> {
  void runOnOperation() override {
    auto module = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<ArmPLOpPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

struct ArmPLOpToFunctionPass : public ArmPLOpToFunctionBase<ArmPLOpToFunctionPass> {
  void runOnOperation() override {
    auto module = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<ArmPLMatmulLowering, ArmPLGemvLowering>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createTritonDotToArmPLOpPass() {
  return std::make_unique<TritonDotToArmPLOpPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createArmPLOpToFunctionPass() {
  return std::make_unique<ArmPLOpToFunctionPass>();
}

} // namespace triton
} // namespace mlir