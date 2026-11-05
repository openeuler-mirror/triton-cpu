add_triton_library(StructuredToMemref
  StructuredToMemref.cpp
  StructuredToMemrefPass.cpp
  DirectTransposeOutput.cpp

  DEPENDS
  StructuredToMemrefConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRSCFTransforms
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRLinalgDialect
  MLIRVectorDialect
  MLIRSideEffectInterfaces
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonTransforms
  TritonTilingExtIR
  TritonStructuredIR
)
