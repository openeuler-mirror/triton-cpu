#ifndef STRUCTURED_TO_MEMREF_CONVERSION_PASSES
#define STRUCTURED_TO_MEMREF_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def StructuredToMemref : Pass<"structured-to-memref", "mlir::ModuleOp"> {
  let summary = "Convert triton structured pointer ops to memref";
}

def DirectTransposeOutput : Pass<"triton-cpu-direct-transpose-output", "mlir::func::FuncOp"> {
  let summary = "Write proven full transpose tiles directly to the output view";
  let constructor = "triton::createDirectTransposeOutput()";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::bufferization::BufferizationDialect",
                           "mlir::linalg::LinalgDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::tensor::TensorDialect",
                           "mlir::vector::VectorDialect"];
}

#endif
