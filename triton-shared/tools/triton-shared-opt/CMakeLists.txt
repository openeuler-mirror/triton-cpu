get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_executable(triton-shared-opt triton-shared-opt.cpp PARTIAL_SOURCES_INTENDED)

 set(test_libs
    ${cuda_test_libs}
    MLIRTestFuncToLLVM
    MLIRAffineTransformsTestPasses
    MLIRArithTestPasses
    MLIRArmNeonTestPasses
    MLIRArmSMETestPasses
    MLIRBufferizationTestPasses
    MLIRControlFlowTestPasses
    MLIRDLTITestPasses
    MLIRFuncTestPasses
    MLIRGPUTestPasses
    MLIRLinalgTestPasses
    MLIRLoopLikeInterfaceTestPasses
    MLIRMathTestPasses
    MLIRTestMathToVCIX
    MLIRMemRefTestPasses
    MLIRMeshTest
    MLIRNVGPUTestPasses
    MLIRSCFTestPasses
    MLIRShapeTestPasses
    MLIRSPIRVTestPasses
    MLIRTensorTestPasses
    MLIRTestAnalysis
    MLIRTestConvertToSPIRV
    MLIRTestDialect
    MLIRTestDynDialect
    MLIRTestIR
    MLIRTestOneToNTypeConversionPass
    MLIRTestPass
    MLIRTestReducer
    MLIRTestTransforms
    MLIRTilingInterfaceTestPasses
    MLIRVectorTestPasses
    MLIRTestVectorToSPIRV
    MLIRLLVMTestPasses
    )
  set(test_libs ${test_libs}
    MLIRTestPDLL
    MLIRTestTransformDialect
    )

  set(test_libs ${test_libs}
      MLIRTestPDLL
      MLIRTestRewrite
      )

# TODO: what's this?
llvm_update_compile_flags(triton-shared-opt)
target_link_libraries(triton-shared-opt PRIVATE
  TritonAnalysis
  TritonTransforms
  TritonGPUTransforms
  TritonSharedAnalysis
  ${dialect_libs}
  ${extension_libs}
  ${conversion_libs}
  ${test_libs}
  # tests
  TritonTestAnalysis
  # MLIR core
  MLIROptLib
  MLIRLLVMDialect
  MLIRPass
  MLIRTransforms
)

mlir_check_all_link_libraries(triton-shared-opt)
