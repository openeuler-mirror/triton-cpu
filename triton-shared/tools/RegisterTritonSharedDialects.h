#pragma once
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "triton-shared/Conversion/StructuredToMemref/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include "triton/Conversion/TritonToTritonGPU/Passes.h"

#include "triton-shared/Conversion/StructuredToMemref/Passes.h"
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonPtrToMemref/Passes.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h"
#include "triton-shared/Conversion/TritonToStructured/Passes.h"
#include "triton-shared/Conversion/UnstructuredToMemref/Passes.h"
#include "triton-shared/Conversion/TritonToUnstructured/Passes.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton-shared/Conversion/TPtrToLLVM/Passes.h"


#include "mlir/Config/mlir-config.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir {
void registerCloneTestPasses();
void registerConvertToTargetEnvPass();
void registerLazyLoadingTestPasses();
void registerLoopLikeInterfaceTestPasses();
void registerPassManagerTestPass();
void registerPrintSpirvAvailabilityPass();
void registerRegionTestPasses();
void registerShapeFunctionTestPasses();
void registerSideEffectTestPasses();
void registerSliceAnalysisTestPass();
void registerSymbolTestPasses();
void registerTestAffineAccessAnalysisPass();
void registerTestAffineDataCopyPass();
void registerTestAffineLoopUnswitchingPass();
void registerTestAffineReifyValueBoundsPass();
void registerTestAffineWalk();
void registerTestBytecodeRoundtripPasses();
void registerTestDecomposeAffineOpPass();
void registerTestFunc();
void registerTestGpuLoweringPasses();
void registerTestGpuMemoryPromotionPass();
void registerTestLoopPermutationPass();
void registerTestMatchers();
void registerTestOperationEqualPass();
void registerTestPreserveUseListOrders();
void registerTestPrintDefUsePass();
void registerTestPrintInvalidPass();
void registerTestPrintNestingPass();
void registerTestReducer();
void registerTestSpirvEntryPointABIPass();
void registerTestSpirvModuleCombinerPass();
void registerTestTraitsPass();
void registerTosaTestQuantUtilAPIPass();
void registerVectorizerTestPass();

namespace test {
void registerCommutativityUtils();
void registerConvertCallOpPass();
void registerConvertFuncOpPass();
void registerInliner();
void registerMemRefBoundCheck();
void registerPatternsTestPass();
void registerSimpleParametricTilingPass();
void registerTestAffineLoopParametricTilingPass();
void registerTestAliasAnalysisPass();
void registerTestArithEmulateWideIntPass();
void registerTestBuiltinAttributeInterfaces();
void registerTestBuiltinDistinctAttributes();
void registerTestCallGraphPass();
void registerTestCfAssertPass();
void registerTestCFGLoopInfoPass();
void registerTestComposeSubView();
void registerTestCompositePass();
void registerTestConstantFold();
void registerTestControlFlowSink();
void registerTestDataLayoutPropagation();
void registerTestDataLayoutQuery();
void registerTestDeadCodeAnalysisPass();
void registerTestDecomposeCallGraphTypes();
void registerTestDiagnosticsPass();
void registerTestDominancePass();
void registerTestDynamicPipelinePass();
void registerTestEmulateNarrowTypePass();
void registerTestExpandMathPass();
void registerTestFooAnalysisPass();
void registerTestComposeSubView();
void registerTestMultiBuffering();
void registerTestIRVisitorsPass();
void registerTestGenericIRVisitorsPass();
void registerTestInterfaces();
void registerTestIRVisitorsPass();
void registerTestLastModifiedPass();
void registerTestLinalgDecomposeOps();
void registerTestLinalgDropUnitDims();
void registerTestLinalgElementwiseFusion();
void registerTestLinalgGreedyFusion();
void registerTestLinalgRankReduceContractionOps();
void registerTestLinalgTransforms();
void registerTestLivenessAnalysisPass();
void registerTestLivenessPass();
void registerTestLoopFusion();
void registerTestLoopMappingPass();
void registerTestLoopUnrollingPass();
void registerTestLowerToArmNeon();
void registerTestLowerToArmSME();
void registerTestLowerToLLVM();
void registerTestMakeIsolatedFromAbovePass();
void registerTestMatchReductionPass();
void registerTestMathAlgebraicSimplificationPass();
void registerTestMathPolynomialApproximationPass();
void registerTestMathToVCIXPass();
void registerTestMemRefDependenceCheck();
void registerTestMemRefStrideCalculation();
void registerTestMeshReshardingSpmdizationPass();
void registerTestMeshSimplificationsPass();
void registerTestMultiBuffering();
void registerTestNextAccessPass();
void registerTestNVGPULowerings();
void registerTestOneToNTypeConversionPass();
void registerTestOpaqueLoc();
void registerTestOpLoweringPasses();
void registerTestPadFusion();
void registerTestRecursiveTypesPass();
void registerTestSCFUpliftWhileToFor();
void registerTestSCFUtilsPass();
void registerTestSCFWhileOpBuilderPass();
void registerTestSCFWrapInZeroTripCheckPasses();
void registerTestShapeMappingPass();
void registerTestSliceAnalysisPass();
void registerTestSPIRVFuncSignatureConversion();
void registerTestTensorCopyInsertionPass();
void registerTestTensorTransforms();
void registerTestTopologicalSortAnalysisPass();
void registerTestTransformDialectEraseSchedulePass();
void registerTestVectorLowerings();
void registerTestVectorReductionToSPIRVDotProd();
void registerTestWrittenToPass();
void registerTestDialectConversionPasses();
void registerTestPDLByteCodePass();
void registerTestPDLLPasses();
} // namespace test
} // namespace mlir

namespace test {
void registerTestDialect(mlir::DialectRegistry &);
void registerTestDynDialect(mlir::DialectRegistry &);
void registerTestTilingInterfaceTransformDialectExtension(mlir::DialectRegistry &);
void registerTestTransformDialectExtension(mlir::DialectRegistry &);
} // namespace test

void registerTestPasses() {
  mlir::registerCloneTestPasses();
  mlir::registerConvertToTargetEnvPass();
  mlir::registerLazyLoadingTestPasses();
  mlir::registerLoopLikeInterfaceTestPasses();
  mlir::registerPassManagerTestPass();
  mlir::registerPrintSpirvAvailabilityPass();
  mlir::registerRegionTestPasses();
  mlir::registerShapeFunctionTestPasses();
  mlir::registerSideEffectTestPasses();
  mlir::registerSliceAnalysisTestPass();
  mlir::registerSymbolTestPasses();
  mlir::registerTestAffineAccessAnalysisPass();
  mlir::registerTestAffineDataCopyPass();
  mlir::registerTestAffineLoopUnswitchingPass();
  mlir::registerTestAffineReifyValueBoundsPass();
  mlir::registerTestAffineWalk();
  mlir::registerTestBytecodeRoundtripPasses();
  mlir::registerTestDecomposeAffineOpPass();
  mlir::registerTestFunc();
  mlir::registerTestGpuLoweringPasses();
  mlir::registerTestGpuMemoryPromotionPass();
  mlir::registerTestLoopPermutationPass();
  mlir::registerTestMatchers();
  mlir::registerTestOperationEqualPass();
  mlir::registerTestPreserveUseListOrders();
  mlir::registerTestPrintDefUsePass();
  mlir::registerTestPrintInvalidPass();
  mlir::registerTestPrintNestingPass();
  mlir::registerTestReducer();
  mlir::registerTestSpirvEntryPointABIPass();
  mlir::registerTestSpirvModuleCombinerPass();
  mlir::registerTestTraitsPass();
  mlir::registerTosaTestQuantUtilAPIPass();
  mlir::registerVectorizerTestPass();

  mlir::test::registerCommutativityUtils();
  mlir::test::registerConvertCallOpPass();
  mlir::test::registerConvertFuncOpPass();
  mlir::test::registerInliner();
  mlir::test::registerMemRefBoundCheck();
  mlir::test::registerPatternsTestPass();
  mlir::test::registerSimpleParametricTilingPass();
  mlir::test::registerTestAffineLoopParametricTilingPass();
  mlir::test::registerTestAliasAnalysisPass();
  mlir::test::registerTestArithEmulateWideIntPass();
  mlir::test::registerTestBuiltinAttributeInterfaces();
  mlir::test::registerTestBuiltinDistinctAttributes();
  mlir::test::registerTestCallGraphPass();
  mlir::test::registerTestCfAssertPass();
  mlir::test::registerTestCFGLoopInfoPass();
  mlir::test::registerTestComposeSubView();
  mlir::test::registerTestCompositePass();
  mlir::test::registerTestConstantFold();
  mlir::test::registerTestControlFlowSink();
  mlir::test::registerTestDataLayoutPropagation();
  mlir::test::registerTestDataLayoutQuery();
  mlir::test::registerTestDeadCodeAnalysisPass();
  mlir::test::registerTestDecomposeCallGraphTypes();
  mlir::test::registerTestDiagnosticsPass();
  mlir::test::registerTestDominancePass();
  mlir::test::registerTestDynamicPipelinePass();
  mlir::test::registerTestEmulateNarrowTypePass();
  mlir::test::registerTestExpandMathPass();
  mlir::test::registerTestFooAnalysisPass();
  mlir::test::registerTestComposeSubView();
  mlir::test::registerTestMultiBuffering();
  mlir::test::registerTestIRVisitorsPass();
  mlir::test::registerTestGenericIRVisitorsPass();
  mlir::test::registerTestInterfaces();
  mlir::test::registerTestIRVisitorsPass();
  mlir::test::registerTestLastModifiedPass();
  mlir::test::registerTestLinalgDecomposeOps();
  mlir::test::registerTestLinalgDropUnitDims();
  mlir::test::registerTestLinalgElementwiseFusion();
  mlir::test::registerTestLinalgGreedyFusion();
  mlir::test::registerTestLinalgRankReduceContractionOps();
  mlir::test::registerTestLinalgTransforms();
  mlir::test::registerTestLivenessAnalysisPass();
  mlir::test::registerTestLivenessPass();
  mlir::test::registerTestLoopFusion();
  mlir::test::registerTestLoopMappingPass();
  mlir::test::registerTestLoopUnrollingPass();
  mlir::test::registerTestLowerToArmNeon();
  mlir::test::registerTestLowerToArmSME();
  mlir::test::registerTestLowerToLLVM();
  mlir::test::registerTestMakeIsolatedFromAbovePass();
  mlir::test::registerTestMatchReductionPass();
  mlir::test::registerTestMathAlgebraicSimplificationPass();
  mlir::test::registerTestMathPolynomialApproximationPass();
  mlir::test::registerTestMathToVCIXPass();
  mlir::test::registerTestMemRefDependenceCheck();
  mlir::test::registerTestMemRefStrideCalculation();
  mlir::test::registerTestMeshReshardingSpmdizationPass();
  mlir::test::registerTestMeshSimplificationsPass();
  mlir::test::registerTestMultiBuffering();
  mlir::test::registerTestNextAccessPass();
  mlir::test::registerTestNVGPULowerings();
  mlir::test::registerTestOneToNTypeConversionPass();
  mlir::test::registerTestOpaqueLoc();
  mlir::test::registerTestOpLoweringPasses();
  mlir::test::registerTestPadFusion();
  mlir::test::registerTestRecursiveTypesPass();
  mlir::test::registerTestSCFUpliftWhileToFor();
  mlir::test::registerTestSCFUtilsPass();
  mlir::test::registerTestSCFWhileOpBuilderPass();
  mlir::test::registerTestSCFWrapInZeroTripCheckPasses();
  mlir::test::registerTestShapeMappingPass();
  mlir::test::registerTestSliceAnalysisPass();
  mlir::test::registerTestSPIRVFuncSignatureConversion();
  mlir::test::registerTestTensorCopyInsertionPass();
  mlir::test::registerTestTensorTransforms();
  mlir::test::registerTestTopologicalSortAnalysisPass();
  mlir::test::registerTestTransformDialectEraseSchedulePass();
  mlir::test::registerTestVectorLowerings();
  mlir::test::registerTestVectorReductionToSPIRVDotProd();
  mlir::test::registerTestWrittenToPass();
  mlir::test::registerTestDialectConversionPasses();
  mlir::test::registerTestPDLByteCodePass();
  mlir::test::registerTestPDLLPasses();
}

inline void registerTritonSharedDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllPasses();
  mlir::registerTritonPasses();
  mlir::triton::gpu::registerTritonGPUPasses();
  mlir::triton::registerTritonToLinalgPass();
  mlir::triton::registerTritonToLinalgExperimentalPasses();
  mlir::triton::registerTritonToStructuredPass();
  mlir::triton::registerTritonPtrToMemref();
  mlir::triton::registerUnstructuredToMemref();
  mlir::triton::registerTritonToUnstructuredPasses();
  mlir::triton::registerTritonArithToLinalgPasses();
  mlir::triton::registerConvertTritonToTritonGPUPass();
  mlir::triton::registerStructuredToMemrefPasses();
  mlir::tptr::registerTPtrToLLVM();


  // TODO: register Triton & TritonGPU passes
  registry.insert<
      mlir::tptr::TPtrDialect, mlir::ptr::PtrDialect,
      mlir::ttx::TritonTilingExtDialect, mlir::tts::TritonStructuredDialect,
      mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
      mlir::triton::gpu::TritonGPUDialect, mlir::math::MathDialect,
      mlir::arith::ArithDialect, mlir::scf::SCFDialect, mlir::gpu::GPUDialect,
      mlir::linalg::LinalgDialect, mlir::func::FuncDialect,
      mlir::tensor::TensorDialect, mlir::memref::MemRefDialect,
      mlir::bufferization::BufferizationDialect, mlir::LLVM::LLVMDialect>();

    ::mlir::registerAllDialects(registry);
    ::mlir::registerAllExtensions(registry);
    registerTestPasses();

}
