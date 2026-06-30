// RUN: triton-shared-opt --split-input-file --armpl-op-to-function %s | FileCheck %s

// Test 1: Lower tts.armpl_matmul (f32) to func.call @cblas_sgemm
module {
  func.func @armpl_matmul_f32(%a: memref<128x64xf32>, %b: memref<64x256xf32>, %c: memref<128x256xf32>) {
    tts.armpl_matmul %a, %b, %c transA = 111 transB = 111 alpha = 1.000000e+00 beta = 0.000000e+00 : memref<128x64xf32>, memref<64x256xf32>, memref<128x256xf32>
    return
  }
}

// CHECK-LABEL: func.func private @cblas_sgemm
// CHECK-SAME:  (i32, i32, i32, i32, i32, i32, f32, !llvm.ptr, i32, !llvm.ptr, i32, f32, !llvm.ptr, i32)
// CHECK-LABEL: func.func @armpl_matmul_f32
// CHECK:       memref.extract_aligned_pointer_as_index
// CHECK:       memref.extract_strided_metadata
// CHECK:       func.call @cblas_sgemm

// -----

// Test 2: Lower tts.armpl_matmul (f64) to func.call @cblas_dgemm
module {
  func.func @armpl_matmul_f64(%a: memref<128x64xf64>, %b: memref<64x256xf64>, %c: memref<128x256xf64>) {
    tts.armpl_matmul %a, %b, %c transA = 111 transB = 112 alpha = 1.000000e+00 beta = 1.000000e+00 : memref<128x64xf64>, memref<64x256xf64>, memref<128x256xf64>
    return
  }
}

// CHECK-LABEL: func.func private @cblas_dgemm
// CHECK-SAME:  (i32, i32, i32, i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32)
// CHECK-LABEL: func.func @armpl_matmul_f64
// CHECK:       func.call @cblas_dgemm

// -----

// Test 3: Lower tts.armpl_gemv (f32) to func.call @cblas_sgemv
module {
  func.func @armpl_gemv_f32(%a: memref<128x64xf32>, %x: memref<64xf32>, %y: memref<128xf32>) {
    tts.armpl_gemv %a, %x, %y transA = 111 alpha = 1.000000e+00 beta = 0.000000e+00 : memref<128x64xf32>, memref<64xf32>, memref<128xf32>
    return
  }
}

// CHECK-LABEL: func.func private @cblas_sgemv
// CHECK-SAME:  (i32, i32, i32, i32, f32, !llvm.ptr, i32, !llvm.ptr, i32, f32, !llvm.ptr, i32)
// CHECK-LABEL: func.func @armpl_gemv_f32
// CHECK:       func.call @cblas_sgemv

// -----

// Test 4: Lower tts.armpl_gemv (f64) to func.call @cblas_dgemv
module {
  func.func @armpl_gemv_f64(%a: memref<128x64xf64>, %x: memref<64xf64>, %y: memref<128xf64>) {
    tts.armpl_gemv %a, %x, %y transA = 112 alpha = 2.000000e+00 beta = 1.000000e+00 : memref<128x64xf64>, memref<64xf64>, memref<128xf64>
    return
  }
}

// CHECK-LABEL: func.func private @cblas_dgemv
// CHECK-SAME:  (i32, i32, i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, i32, f64, !llvm.ptr, i32)
// CHECK-LABEL: func.func @armpl_gemv_f64
// CHECK:       func.call @cblas_dgemv