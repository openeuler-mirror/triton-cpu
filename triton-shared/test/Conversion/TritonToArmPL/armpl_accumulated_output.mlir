// RUN: env TRITON_SHARED_ENABLE_ACL_OPT=0 triton-shared-opt --triton-to-linalg-experimental=enable-armpl=true %s | FileCheck %s --check-prefixes=CHECK,OFF
// RUN: env TRITON_SHARED_ENABLE_ACL_OPT=1 triton-shared-opt --triton-to-linalg-experimental=enable-armpl=true %s | FileCheck %s --check-prefixes=CHECK,ON
// RUN: env -u TRITON_SHARED_ENABLE_ACL_OPT triton-shared-opt --triton-to-linalg-experimental=enable-armpl=true %s | FileCheck %s --check-prefixes=CHECK,ON

// Exercise the production entry point. Only small accumulated output tiles
// fall back; zero accumulators and the 4096-element threshold retain ArmPL.
module {
  // CHECK-LABEL: func.func @small_accum
  // OFF: llvm.call @cblas_sgemm
  // ON-NOT: llvm.call @cblas
  // ON: linalg.matmul
  // ON-NOT: llvm.call @cblas
  tt.func @small_accum(%a: tensor<32x16xf32>, %b: tensor<16x32xf32>, %c: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %d = tt.dot %a, %b, %c {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<32x16xf32> * tensor<16x32xf32> -> tensor<32x32xf32>
    tt.return %d : tensor<32x32xf32>
  }

  // CHECK-LABEL: func.func @below_threshold
  // OFF: llvm.call @cblas_sgemm
  // ON-NOT: llvm.call @cblas
  // ON: linalg.matmul
  // ON-NOT: llvm.call @cblas
  tt.func @below_threshold(%a: tensor<32x16xf32>, %b: tensor<16x64xf32>, %c: tensor<32x64xf32>) -> tensor<32x64xf32> {
    %d = tt.dot %a, %b, %c {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
    tt.return %d : tensor<32x64xf32>
  }

  // CHECK-LABEL: func.func @at_threshold
  // CHECK: llvm.call @cblas_sgemm
  tt.func @at_threshold(%a: tensor<64x16xf32>, %b: tensor<16x64xf32>, %c: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %d = tt.dot %a, %b, %c {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<64x16xf32> * tensor<16x64xf32> -> tensor<64x64xf32>
    tt.return %d : tensor<64x64xf32>
  }

  // CHECK-LABEL: func.func @large_accum
  // CHECK: llvm.call @cblas_sgemm
  tt.func @large_accum(%a: tensor<64x16xf32>, %b: tensor<16x128xf32>, %c: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %d = tt.dot %a, %b, %c {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<64x16xf32> * tensor<16x128xf32> -> tensor<64x128xf32>
    tt.return %d : tensor<64x128xf32>
  }

  // CHECK-LABEL: func.func @small_zero
  // CHECK: llvm.call @cblas_sgemm
  tt.func @small_zero(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %c = arith.constant dense<0.0> : tensor<16x16xf32>
    %d = tt.dot %a, %b, %c {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    tt.return %d : tensor<16x16xf32>
  }

  // CHECK-LABEL: func.func @small_f16_accum
  // OFF: llvm.call @cblas_sgemm
  // ON-NOT: llvm.call @cblas
  // ON: linalg.matmul
  // ON-NOT: llvm.call @cblas
  tt.func @small_f16_accum(%a: tensor<16x16xf16>, %b: tensor<16x16xf16>, %c: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %d = tt.dot %a, %b, %c {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
    tt.return %d : tensor<16x16xf32>
  }

  // CHECK-LABEL: func.func @small_bf16_accum
  // OFF: llvm.call @cblas_sgemm
  // ON-NOT: llvm.call @cblas
  // ON: linalg.matmul
  // ON-NOT: llvm.call @cblas
  tt.func @small_bf16_accum(%a: tensor<16x16xbf16>, %b: tensor<16x16xbf16>, %c: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %d = tt.dot %a, %b, %c {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<16x16xbf16> * tensor<16x16xbf16> -> tensor<16x16xf32>
    tt.return %d : tensor<16x16xf32>
  }
}
