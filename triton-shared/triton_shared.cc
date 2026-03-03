// LLVM
#include "llvm/IR/Constants.h"
#include "llvm/Support/TargetSelect.h"

// MLIR: Conversion Passes
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"

// MLIR: Dialects
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/LegalizeFloat8Types.h"
#include "mlir/Dialect/LLVMIR/Transforms/PromoteI1ToI8.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
// #include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"

// MLIR: Core IR and Passes
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

// MLIR: Target and Translation
// #include "mlir/Target/LLVMIR/Dialect/AMX/AMXToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

// LLVM: Debug
#include "llvm/Support/Debug.h" // Key header file

// MLIR: Top-level Transforms
#include "mlir/Transforms/Passes.h"

// Triton and other third-party dialects
#include "triton-shared/Conversion/StructuredToMemref/Passes.h"
#include "triton-shared/Conversion/TPtrToLLVM/Passes.h"
#include "triton-shared/Conversion/TPtrToLLVM/TPtrToLLVM.h"
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonPtrToMemref/Passes.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h"
#include "triton-shared/Conversion/TritonToStructured/Passes.h"
#include "triton-shared/Conversion/TritonToUnstructured/Passes.h"
#include "triton-shared/Conversion/UnstructuredToMemref/Passes.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// Test
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

namespace py = pybind11;

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
void registerTestTilingInterfaceTransformDialectExtension(
    mlir::DialectRegistry &);
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

#define ADD_PASS_WRAPPER_0(name, builder)                                      \
  m.def(name, [](mlir::PassManager &pm) { pm.addPass(builder()); })

#define ADD_PASS_WRAPPER_1(name, builder, ty0)                                 \
  m.def(name,                                                                  \
        [](mlir::PassManager &pm, ty0 val0) { pm.addPass(builder(val0)); })

#define ADD_PASS_WRAPPER_1_ARG(name, builder, ty0, arg0, val0)                 \
  m.def(                                                                       \
      name,                                                                    \
      [](mlir::PassManager &pm, ty0 arg0) { pm.addPass(builder(val0)); },      \
      py::arg("pm"), py::arg(#arg0) = val0);

void enable_mlir_debug(const std::string &debug_type) {
  ::llvm::DebugFlag = true;
  llvm::setCurrentDebugType(debug_type.c_str());
}

namespace {
using namespace mlir;
template <typename Derived>
class OpPassWrapper : public PassWrapper<Derived, OperationPass<>> {};

struct TestTransformDialectEraseSchedulePass
    : public PassWrapper<TestTransformDialectEraseSchedulePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestTransformDialectEraseSchedulePass)

  StringRef getArgument() const final {
    return "test-transform-dialect-erase-schedule";
  }

  StringRef getDescription() const final {
    return "erase transform dialect schedule from the IR";
  }

  void runOnOperation() override {
    getOperation()->walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
      if (isa<transform::TransformOpInterface>(nestedOp)) {
        nestedOp->erase();
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createTestTransformDialectEraseSchedulePass() {
  return std::make_unique<TestTransformDialectEraseSchedulePass>();
}

void init_to_ttsharedir(py::module &&m) {
  using namespace mlir::triton;

  ADD_PASS_WRAPPER_0("add_triton_to_linalg_experimental",
                     createTritonToLinalgExperimentalPass);
}
void init_to_llvm(py::module &&m) {
  using namespace mlir;

  ADD_PASS_WRAPPER_0("add_transform_interpreter",
                     transform::createInterpreterPass);

  ADD_PASS_WRAPPER_0("add_convert_linalg_to_affine_loops",
                     createConvertLinalgToAffineLoopsPass);
  ADD_PASS_WRAPPER_0("add_empty_tensor_to_alloc_tensor",
                     bufferization::createEmptyTensorToAllocTensorPass);

  ADD_PASS_WRAPPER_1_ARG(
      "add_convert_vector_to_llvm_with_sve",
      [](bool enableSVE) {
        mlir::ConvertVectorToLLVMPassOptions options;
        options.armSVE = enableSVE;
        return mlir::createConvertVectorToLLVMPass(options);
      },
      bool, enable_sve, true);

  ADD_PASS_WRAPPER_0("add_canonicalizer", createCanonicalizerPass);
  ADD_PASS_WRAPPER_0("add_cse", createCSEPass);
  ADD_PASS_WRAPPER_0("add_one_shot_bufferize",
                     bufferization::createOneShotBufferizePass);
  ADD_PASS_WRAPPER_0("add_lower_affine", createLowerAffinePass);
  ADD_PASS_WRAPPER_0("add_convert_linalg_to_loops",
                     createConvertLinalgToLoopsPass);
  ADD_PASS_WRAPPER_0("add_expand_strided_metadata",
                     memref::createExpandStridedMetadataPass);
  ADD_PASS_WRAPPER_0("add_convert_scf_to_cf", createConvertSCFToCFPass);
  ADD_PASS_WRAPPER_0("add_convert_arith_to_llvm",
                     createArithToLLVMConversionPass);
  ADD_PASS_WRAPPER_0("add_convert_math_to_llvm", createConvertMathToLLVMPass);
  ADD_PASS_WRAPPER_0("add_convert_math_to_libm", createConvertMathToLibmPass);
  ADD_PASS_WRAPPER_0("add_convert_complex_to_llvm",
                     createConvertComplexToLLVMPass);
  ADD_PASS_WRAPPER_0("add_convert_vector_to_llvm",
                     createConvertVectorToLLVMPass);
  ADD_PASS_WRAPPER_0("add_convert_index_to_llvm", createConvertIndexToLLVMPass);
  ADD_PASS_WRAPPER_0("add_memref_expand", memref::createExpandOpsPass);
  ADD_PASS_WRAPPER_0("add_finalize_memref_to_llvm",
                     createFinalizeMemRefToLLVMConversionPass);
  ADD_PASS_WRAPPER_0("add_convert_func_to_llvm", createConvertFuncToLLVMPass);
  ADD_PASS_WRAPPER_0("add_convert_tptr_to_llvm", tptr::createTPtrToLLVMPass);
  ADD_PASS_WRAPPER_0("add_convert_cf_to_llvm",
                     createConvertControlFlowToLLVMPass);
  ADD_PASS_WRAPPER_0("add_reconcile_unrealized_casts",
                     createReconcileUnrealizedCastsPass);
  ADD_PASS_WRAPPER_0("add_convert_vector_to_scf", createConvertVectorToSCFPass);
  ADD_PASS_WRAPPER_0("add_test_transform_dialect_erase_schedule",
                     createTestTransformDialectEraseSchedulePass);
  ADD_PASS_WRAPPER_0("add_strip_debug_info", createStripDebugInfoPass);
  ADD_PASS_WRAPPER_0("add_llvm_legalize_float8_types",
                     LLVM::createLegalizeFloat8TypesPass);
  ADD_PASS_WRAPPER_0("add_convert_to_llvm", createConvertToLLVMPass);
  ADD_PASS_WRAPPER_0("add_promote_i1_to_i8", LLVM::createPromoteI1ToI8Pass);
}

// Loading passes is necessary when using the transform schedule, however, they
// cannnot be loaded more than once and they are not bound to the context or to
// a registry so we need to keep this global flag
bool loaded = false;

void init_triton_shared_ir(py::module &&m) {

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;

    // Register interfaces and translations
    registry.insert<mlir::triton::TritonDialect, mlir::tptr::TPtrDialect,
                    mlir::ttx::TritonTilingExtDialect,
                    mlir::tts::TritonStructuredDialect,
                    mlir::triton::gpu::TritonGPUDialect>();
    mlir::registerAllDialects(registry);
    mlir::registerAllExtensions(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
    if (!loaded) {
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
      registerTestPasses();
    }

    loaded = true;
  });
}

void print_context_ops(py::module &&m) {

  m.def("print_context_ops", [](mlir::MLIRContext &context) {
    llvm::errs() << "=== Registered ops ===\n";
    for (const auto op : context.getRegisteredOperations()) {
      llvm::errs() << op.getStringRef() << "\n";
    }
  });
}

void init_triton_shared_debug(py::module &&m) {
  m.def("enable_mlir_debug", enable_mlir_debug,
        "Enables a specific MLIR/LLVM debug type (e.g., 'pattern-rewrite'). "
        "Pass an empty string to disable.",
        py::arg("debug_type"));
}

void init_triton_triton_shared(py::module &&m) {
  init_to_ttsharedir(m.def_submodule("to_ttsharedir"));
  init_to_llvm(m.def_submodule("to_llir"));
  init_triton_shared_ir(m.def_submodule("ir"));
  init_triton_shared_debug(m.def_submodule("debug"));
  print_context_ops(m.def_submodule("debug"));
}
