include "mlir/Pass/PassBase.td"

def TritonDotToArmPLOp : Pass<"triton-dot-to-armpl-op", "mlir::ModuleOp"> {
  let summary = "Convert tt.dot to tts.armpl_matmul/tts.armpl_gemv";
  let constructor = "mlir::triton::createTritonDotToArmPLOpPass()";
}

def ArmPLOpToFunction : Pass<"armpl-op-to-function", "mlir::ModuleOp"> {
  let summary = "Lower tts.armpl_matmul and tts.armpl_gemv to func.call @cblas_sgemm/@cblas_dgemm/@cblas_sgemv/@cblas_dgemv";
  let constructor = "mlir::triton::createArmPLOpToFunctionPass()";
}