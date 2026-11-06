add_triton_library(TritonToKunpengLLM
  TritonToKunpengLLM.cpp

  DEPENDS
  TritonToKunpengLLMConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRLLVMDialect
  MLIRFuncDialect
  MLIRMathDialect
  MLIRSCFDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  TritonIR
)
