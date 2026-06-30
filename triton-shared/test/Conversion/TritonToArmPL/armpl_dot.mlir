// RUN: triton-shared-opt --split-input-file --triton-dot-to-armpl-op %s | FileCheck %s

// Test 1: Basic tt.dot with f32 operands, no transpose, zero accumulator
// Should produce tts.armpl_matmul with transA=111, transB=111, alpha=1.0, beta=0.0
module {
  tt.func @dot_f32_no_trans(%a: tensor<128x64xf32>, %b: tensor<64x256xf32>) -> tensor<128x256xf32> {
    %cst = arith.constant dense<0.0> : tensor<128x256xf32>
    %d = tt.dot %a, %b, %cst {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf32> * tensor<64x256xf32> -> tensor<128x256xf32>
    tt.return %d : tensor<128x256xf32>
  }
}

// CHECK-LABEL: func.func @dot_f32_no_trans
// CHECK:       tts.armpl_matmul
// CHECK-SAME:  transA = 111 transB = 111
// CHECK-SAME:  alpha = 1.000000e+00 beta = 0.000000e+00
// CHECK-SAME:  tensor<128x64xf32>, tensor<64x256xf32>, tensor<128x256xf32>

// -----

// Test 2: tt.dot with transposed B operand (tt.trans before dot)
// Should produce tts.armpl_matmul with transA=111, transB=112
module {
  tt.func @dot_f32_trans_b(%a: tensor<128x64xf32>, %b: tensor<256x64xf32>) -> tensor<128x256xf32> {
    %bt = tt.trans %b {order = array<i32: 1, 0>} : tensor<256x64xf32> -> tensor<64x256xf32>
    %cst = arith.constant dense<0.0> : tensor<128x256xf32>
    %d = tt.dot %a, %bt, %cst {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf32> * tensor<64x256xf32> -> tensor<128x256xf32>
    tt.return %d : tensor<128x256xf32>
  }
}

// CHECK-LABEL: func.func @dot_f32_trans_b
// CHECK:       tts.armpl_matmul
// CHECK-SAME:  transA = 111 transB = 112

// -----

// Test 3: tt.dot with transposed A operand
// Should produce tts.armpl_matmul with transA=112, transB=111
module {
  tt.func @dot_f32_trans_a(%a: tensor<64x128xf32>, %b: tensor<64x256xf32>) -> tensor<128x256xf32> {
    %at = tt.trans %a {order = array<i32: 1, 0>} : tensor<64x128xf32> -> tensor<128x64xf32>
    %cst = arith.constant dense<0.0> : tensor<128x256xf32>
    %d = tt.dot %at, %b, %cst {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf32> * tensor<64x256xf32> -> tensor<128x256xf32>
    tt.return %d : tensor<128x256xf32>
  }
}

// CHECK-LABEL: func.func @dot_f32_trans_a
// CHECK:       tts.armpl_matmul
// CHECK-SAME:  transA = 112 transB = 111

// -----

// Test 4: tt.dot with non-zero accumulator (beta=1.0)
module {
  tt.func @dot_f32_with_acc(%a: tensor<128x64xf32>, %b: tensor<64x256xf32>, %c: tensor<128x256xf32>) -> tensor<128x256xf32> {
    %d = tt.dot %a, %b, %c {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf32> * tensor<64x256xf32> -> tensor<128x256xf32>
    tt.return %d : tensor<128x256xf32>
  }
}

// CHECK-LABEL: func.func @dot_f32_with_acc
// CHECK:       tts.armpl_matmul
// CHECK-SAME:  alpha = 1.000000e+00 beta = 1.000000e+00

// -----

// Test 5: tt.dot with f16 operands (should upcast to f32)
module {
  tt.func @dot_f16(%a: tensor<128x64xf16>, %b: tensor<64x256xf16>) -> tensor<128x256xf16> {
    %cst = arith.constant dense<0.0> : tensor<128x256xf16>
    %d = tt.dot %a, %b, %cst {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf16> * tensor<64x256xf16> -> tensor<128x256xf16>
    tt.return %d : tensor<128x256xf16>
  }
}

// CHECK-LABEL: func.func @dot_f16
// CHECK:       arith.extf
// CHECK:       tts.armpl_matmul
// CHECK-SAME:  tensor<128x64xf32>, tensor<64x256xf32>, tensor<128x256xf32>
// CHECK:       arith.truncf