add_triton_library(TritonToArmPL
  TritonToArmPL.cpp

  DEPENDS
  TritonToArmPLConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  TritonStructuredIR
  TritonIR
)