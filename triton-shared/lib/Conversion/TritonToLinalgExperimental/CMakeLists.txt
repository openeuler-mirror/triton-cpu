#===------------------------------------------------------------------------===#
#
# Copyright (c) Triton Project Contributors.
#
#===------------------------------------------------------------------------===#

add_triton_library(TritonToLinalgExperimental
  TritonToLinalgExperimentalPass.cpp
  ReconcilePtrCastsPass.cpp
  TritonToPtrPass.cpp
  CollapseShape.cpp
  FullTileRegionVersioning.cpp
  HoistStaticAllocs.cpp
  NontemporalLoads.cpp
  DoubleBufferLoopAllocations.cpp

  DEPENDS
  TritonToLinalgExperimentalConversionPassIncGen

  LINK_LIBS PUBLIC
  TritonTilingExtIR
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRLLVMDialect
  MLIRSCFDialect
  MLIRControlFlowInterfaces
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  MLIRVectorDialect
  TPtrIR
  TritonAnalysis
  TritonIR
  TritonTransforms
  TritonSharedAnalysis
  TritonArithToLinalg
  StructuredToMemref
  UnstructuredToMemref
  TritonPtrToMemref
  TritonToArmPL
  TritonSharedUtils

  TritonArithToLinalg
  StructuredToMemref
  TritonToStructured
  TritonToUnstructured
  TritonPtrToMemref
  UnstructuredToMemref
)
