// RUN: triton-shared-opt --split-input-file --verify-each --triton-cpu-direct-transpose-output %s | FileCheck %s

// Full tiles, tails, permutation checks, and rectangular tiles.
// Check write-contiguous direct-output loop order and preserve the dynamic boundary fallback.
module {
func.func @direct_transpose_output_32x32(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
  %cst = arith.constant dense<0.000000e+00> : vector<16x16xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c16_i32 = arith.constant 16 : i32
  %c32 = arith.constant 32 : index
  %c16 = arith.constant 16 : index
  %0 = arith.muli %arg5, %c16_i32 : i32
  %1 = arith.index_cast %0 : i32 to index
  %2 = arith.muli %arg6, %c16_i32 : i32
  %3 = arith.index_cast %2 : i32 to index
  %4 = arith.muli %1, %c32 : index
  %5 = arith.addi %4, %3 : index
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%5], sizes: [16, 16], strides: [32, 1] : memref<*xf32> to memref<16x16xf32, strided<[32, 1], offset: ?>>
  %6 = arith.addi %1, %c16 : index
  %7 = arith.minsi %6, %c32 : index
  %8 = arith.maxsi %7, %1 : index
  %9 = arith.subi %8, %1 : index
  %10 = arith.addi %3, %c16 : index
  %11 = arith.minsi %10, %c32 : index
  %12 = arith.maxsi %11, %3 : index
  %13 = arith.subi %12, %3 : index
  %14 = arith.minsi %9, %c16 : index
  %15 = arith.minsi %13, %c16 : index
  %alloca = memref.alloca() : memref<16x16xf32>
  %16 = arith.cmpi eq, %14, %c16 : index
  %17 = arith.cmpi eq, %15, %c16 : index
  %18 = arith.andi %16, %17 : i1
  scf.if %18 {
    %23 = vector.transfer_read %reinterpret_cast[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf32, strided<[32, 1], offset: ?>>, vector<16x16xf32>
    vector.transfer_write %23, %alloca[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
  } else {
    %23 = arith.cmpi slt, %14, %c16 : index
    %24 = arith.cmpi slt, %15, %c16 : index
    %25 = arith.ori %23, %24 : i1
    scf.if %25 {
      vector.transfer_write %cst, %alloca[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
    }
    %subview = memref.subview %reinterpret_cast[0, 0] [%14, %15] [1, 1] : memref<16x16xf32, strided<[32, 1], offset: ?>> to memref<?x?xf32, strided<[32, 1], offset: ?>>
    %subview_3 = memref.subview %alloca[0, 0] [%14, %15] [1, 1] : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1]>>
    memref.copy %subview, %subview_3 : memref<?x?xf32, strided<[32, 1], offset: ?>> to memref<?x?xf32, strided<[16, 1]>>
  }
  %alloca_1 = memref.alloca() {alignment = 64 : i64} : memref<16x16xf32>
  %19 = vector.transfer_read %alloca[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf32>, vector<16x16xf32>
  %20 = vector.transpose %19, [1, 0] : vector<16x16xf32> to vector<16x16xf32>
  vector.transfer_write %20, %alloca_1[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
  %21 = arith.muli %3, %c32 : index
  %22 = arith.addi %21, %1 : index
  %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%22], sizes: [16, 16], strides: [32, 1] : memref<*xf32> to memref<16x16xf32, strided<[32, 1], offset: ?>>
  scf.if %18 {
    memref.copy %alloca_1, %reinterpret_cast_2 : memref<16x16xf32> to memref<16x16xf32, strided<[32, 1], offset: ?>>
  } else {
    %subview = memref.subview %alloca_1[0, 0] [%15, %14] [1, 1] : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1]>>
    %subview_3 = memref.subview %reinterpret_cast_2[0, 0] [%15, %14] [1, 1] : memref<16x16xf32, strided<[32, 1], offset: ?>> to memref<?x?xf32, strided<[32, 1], offset: ?>>
    memref.copy %subview, %subview_3 : memref<?x?xf32, strided<[16, 1]>> to memref<?x?xf32, strided<[32, 1], offset: ?>>
  }
  return
}
func.func @direct_transpose_output_31x31(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
  %cst = arith.constant dense<0.000000e+00> : vector<16x16xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c16_i32 = arith.constant 16 : i32
  %c31 = arith.constant 31 : index
  %c16 = arith.constant 16 : index
  %0 = arith.muli %arg5, %c16_i32 : i32
  %1 = arith.index_cast %0 : i32 to index
  %2 = arith.muli %arg6, %c16_i32 : i32
  %3 = arith.index_cast %2 : i32 to index
  %4 = arith.muli %1, %c31 : index
  %5 = arith.addi %4, %3 : index
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%5], sizes: [16, 16], strides: [31, 1] : memref<*xf32> to memref<16x16xf32, strided<[31, 1], offset: ?>>
  %6 = arith.addi %1, %c16 : index
  %7 = arith.minsi %6, %c31 : index
  %8 = arith.maxsi %7, %1 : index
  %9 = arith.subi %8, %1 : index
  %10 = arith.addi %3, %c16 : index
  %11 = arith.minsi %10, %c31 : index
  %12 = arith.maxsi %11, %3 : index
  %13 = arith.subi %12, %3 : index
  %14 = arith.minsi %9, %c16 : index
  %15 = arith.minsi %13, %c16 : index
  %alloca = memref.alloca() : memref<16x16xf32>
  %16 = arith.cmpi eq, %14, %c16 : index
  %17 = arith.cmpi eq, %15, %c16 : index
  %18 = arith.andi %16, %17 : i1
  scf.if %18 {
    %23 = vector.transfer_read %reinterpret_cast[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf32, strided<[31, 1], offset: ?>>, vector<16x16xf32>
    vector.transfer_write %23, %alloca[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
  } else {
    %23 = arith.cmpi slt, %14, %c16 : index
    %24 = arith.cmpi slt, %15, %c16 : index
    %25 = arith.ori %23, %24 : i1
    scf.if %25 {
      vector.transfer_write %cst, %alloca[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
    }
    %subview = memref.subview %reinterpret_cast[0, 0] [%14, %15] [1, 1] : memref<16x16xf32, strided<[31, 1], offset: ?>> to memref<?x?xf32, strided<[31, 1], offset: ?>>
    %subview_3 = memref.subview %alloca[0, 0] [%14, %15] [1, 1] : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1]>>
    memref.copy %subview, %subview_3 : memref<?x?xf32, strided<[31, 1], offset: ?>> to memref<?x?xf32, strided<[16, 1]>>
  }
  %alloca_1 = memref.alloca() {alignment = 64 : i64} : memref<16x16xf32>
  %19 = vector.transfer_read %alloca[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf32>, vector<16x16xf32>
  %20 = vector.transpose %19, [1, 0] : vector<16x16xf32> to vector<16x16xf32>
  vector.transfer_write %20, %alloca_1[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
  %21 = arith.muli %3, %c31 : index
  %22 = arith.addi %21, %1 : index
  %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%22], sizes: [16, 16], strides: [31, 1] : memref<*xf32> to memref<16x16xf32, strided<[31, 1], offset: ?>>
  scf.if %18 {
    memref.copy %alloca_1, %reinterpret_cast_2 : memref<16x16xf32> to memref<16x16xf32, strided<[31, 1], offset: ?>>
  } else {
    %subview = memref.subview %alloca_1[0, 0] [%15, %14] [1, 1] : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1]>>
    %subview_3 = memref.subview %reinterpret_cast_2[0, 0] [%15, %14] [1, 1] : memref<16x16xf32, strided<[31, 1], offset: ?>> to memref<?x?xf32, strided<[31, 1], offset: ?>>
    memref.copy %subview, %subview_3 : memref<?x?xf32, strided<[16, 1]>> to memref<?x?xf32, strided<[31, 1], offset: ?>>
  }
  return
}
func.func @direct_transpose_output_identity_permutation(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
  %cst = arith.constant dense<0.000000e+00> : vector<16x16xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c16_i32 = arith.constant 16 : i32
  %c32 = arith.constant 32 : index
  %c16 = arith.constant 16 : index
  %0 = arith.muli %arg5, %c16_i32 : i32
  %1 = arith.index_cast %0 : i32 to index
  %2 = arith.muli %arg6, %c16_i32 : i32
  %3 = arith.index_cast %2 : i32 to index
  %4 = arith.muli %1, %c32 : index
  %5 = arith.addi %4, %3 : index
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%5], sizes: [16, 16], strides: [32, 1] : memref<*xf32> to memref<16x16xf32, strided<[32, 1], offset: ?>>
  %6 = arith.addi %1, %c16 : index
  %7 = arith.minsi %6, %c32 : index
  %8 = arith.maxsi %7, %1 : index
  %9 = arith.subi %8, %1 : index
  %10 = arith.addi %3, %c16 : index
  %11 = arith.minsi %10, %c32 : index
  %12 = arith.maxsi %11, %3 : index
  %13 = arith.subi %12, %3 : index
  %14 = arith.minsi %9, %c16 : index
  %15 = arith.minsi %13, %c16 : index
  %alloca = memref.alloca() : memref<16x16xf32>
  %16 = arith.cmpi eq, %14, %c16 : index
  %17 = arith.cmpi eq, %15, %c16 : index
  %18 = arith.andi %16, %17 : i1
  scf.if %18 {
    %23 = vector.transfer_read %reinterpret_cast[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf32, strided<[32, 1], offset: ?>>, vector<16x16xf32>
    vector.transfer_write %23, %alloca[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
  } else {
    %23 = arith.cmpi slt, %14, %c16 : index
    %24 = arith.cmpi slt, %15, %c16 : index
    %25 = arith.ori %23, %24 : i1
    scf.if %25 {
      vector.transfer_write %cst, %alloca[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
    }
    %subview = memref.subview %reinterpret_cast[0, 0] [%14, %15] [1, 1] : memref<16x16xf32, strided<[32, 1], offset: ?>> to memref<?x?xf32, strided<[32, 1], offset: ?>>
    %subview_3 = memref.subview %alloca[0, 0] [%14, %15] [1, 1] : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1]>>
    memref.copy %subview, %subview_3 : memref<?x?xf32, strided<[32, 1], offset: ?>> to memref<?x?xf32, strided<[16, 1]>>
  }
  %alloca_1 = memref.alloca() {alignment = 64 : i64} : memref<16x16xf32>
  %19 = vector.transfer_read %alloca[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf32>, vector<16x16xf32>
  %20 = vector.transpose %19, [0, 1] : vector<16x16xf32> to vector<16x16xf32>
  vector.transfer_write %20, %alloca_1[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
  %21 = arith.muli %3, %c32 : index
  %22 = arith.addi %21, %1 : index
  %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%22], sizes: [16, 16], strides: [32, 1] : memref<*xf32> to memref<16x16xf32, strided<[32, 1], offset: ?>>
  scf.if %18 {
    memref.copy %alloca_1, %reinterpret_cast_2 : memref<16x16xf32> to memref<16x16xf32, strided<[32, 1], offset: ?>>
  } else {
    %subview = memref.subview %alloca_1[0, 0] [%15, %14] [1, 1] : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1]>>
    %subview_3 = memref.subview %reinterpret_cast_2[0, 0] [%15, %14] [1, 1] : memref<16x16xf32, strided<[32, 1], offset: ?>> to memref<?x?xf32, strided<[32, 1], offset: ?>>
    memref.copy %subview, %subview_3 : memref<?x?xf32, strided<[16, 1]>> to memref<?x?xf32, strided<[32, 1], offset: ?>>
  }
  return
}
// Independent allocation order in a real nonsquare versioned full-tile chain.
func.func @direct_transpose_output_versioned_nonsquare(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
  %cst = arith.constant dense<0.000000e+00> : vector<16x8xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c16_i32 = arith.constant 16 : i32
  %c8_i32 = arith.constant 8 : i32
  %c17 = arith.constant 17 : index
  %c16 = arith.constant 16 : index
  %c33 = arith.constant 33 : index
  %c8 = arith.constant 8 : index
  %0 = arith.muli %arg5, %c16_i32 : i32
  %1 = arith.index_cast %0 : i32 to index
  %2 = arith.muli %arg6, %c8_i32 : i32
  %3 = arith.index_cast %2 : i32 to index
  %4 = arith.muli %1, %c17 : index
  %5 = arith.addi %4, %3 : index
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%5], sizes: [16, 8], strides: [17, 1] : memref<*xf32> to memref<16x8xf32, strided<[17, 1], offset: ?>>
  %6 = arith.addi %1, %c16 : index
  %7 = arith.minsi %6, %c33 : index
  %8 = arith.maxsi %7, %1 : index
  %9 = arith.subi %8, %1 : index
  %10 = arith.addi %3, %c8 : index
  %11 = arith.minsi %10, %c17 : index
  %12 = arith.maxsi %11, %3 : index
  %13 = arith.subi %12, %3 : index
  %14 = arith.minsi %9, %c16 : index
  %15 = arith.minsi %13, %c8 : index
  %16 = arith.cmpi eq, %14, %c16 : index
  %17 = arith.cmpi eq, %15, %c8 : index
  %18 = arith.andi %16, %17 : i1
  scf.if %18 {
    %alloca = memref.alloca() : memref<16x8xf32>
    %19 = vector.transfer_read %reinterpret_cast[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x8xf32, strided<[17, 1], offset: ?>>, vector<16x8xf32>
    vector.transfer_write %19, %alloca[%c0, %c0] {in_bounds = [true, true]} : vector<16x8xf32>, memref<16x8xf32>
    %20 = vector.transfer_read %alloca[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x8xf32>, vector<16x8xf32>
    %21 = vector.transpose %20, [1, 0] : vector<16x8xf32> to vector<8x16xf32>
    %alloca_1 = memref.alloca() {alignment = 64 : i64} : memref<8x16xf32>
    vector.transfer_write %21, %alloca_1[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
    %22 = arith.muli %3, %c33 : index
    %23 = arith.addi %22, %1 : index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%23], sizes: [8, 16], strides: [33, 1] : memref<*xf32> to memref<8x16xf32, strided<[33, 1], offset: ?>>
    linalg.copy ins(%alloca_1 : memref<8x16xf32>) outs(%reinterpret_cast_2 : memref<8x16xf32, strided<[33, 1], offset: ?>>)
  } else {
    %alloca = memref.alloca() : memref<16x8xf32>
    %19 = arith.cmpi slt, %14, %c16 : index
    %20 = arith.cmpi slt, %15, %c8 : index
    %21 = arith.ori %19, %20 : i1
    scf.if %21 {
      vector.transfer_write %cst, %alloca[%c0, %c0] {in_bounds = [true, true]} : vector<16x8xf32>, memref<16x8xf32>
    }
    %subview = memref.subview %reinterpret_cast[0, 0] [%14, %15] [1, 1] : memref<16x8xf32, strided<[17, 1], offset: ?>> to memref<?x?xf32, strided<[17, 1], offset: ?>>
    %subview_1 = memref.subview %alloca[0, 0] [%14, %15] [1, 1] : memref<16x8xf32> to memref<?x?xf32, strided<[8, 1]>>
    memref.copy %subview, %subview_1 : memref<?x?xf32, strided<[17, 1], offset: ?>> to memref<?x?xf32, strided<[8, 1]>>
    %22 = vector.transfer_read %alloca[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x8xf32>, vector<16x8xf32>
    %23 = vector.transpose %22, [1, 0] : vector<16x8xf32> to vector<8x16xf32>
    %alloca_2 = memref.alloca() {alignment = 64 : i64} : memref<8x16xf32>
    vector.transfer_write %23, %alloca_2[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
    %24 = arith.muli %3, %c33 : index
    %25 = arith.addi %24, %1 : index
    %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [8, 16], strides: [33, 1] : memref<*xf32> to memref<8x16xf32, strided<[33, 1], offset: ?>>
    %subview_4 = memref.subview %alloca_2[0, 0] [%15, %14] [1, 1] : memref<8x16xf32> to memref<?x?xf32, strided<[16, 1]>>
    %subview_5 = memref.subview %reinterpret_cast_3[0, 0] [%15, %14] [1, 1] : memref<8x16xf32, strided<[33, 1], offset: ?>> to memref<?x?xf32, strided<[33, 1], offset: ?>>
    linalg.copy ins(%subview_4 : memref<?x?xf32, strided<[16, 1]>>) outs(%subview_5 : memref<?x?xf32, strided<[33, 1], offset: ?>>)
  } {tts.full_tile_versioned}
  return
}

}

// CHECK-LABEL: func.func @direct_transpose_output_32x32
// CHECK: scf.for [[ROW32:%[a-zA-Z0-9_]+]] =
// CHECK: scf.for [[COL32:%[a-zA-Z0-9_]+]] =
// CHECK: [[VALUE32:%[a-zA-Z0-9_]+]] = memref.load {{.*}}[[[COL32]], [[ROW32]]]
// CHECK: memref.store [[VALUE32]], {{.*}}[[[ROW32]], [[COL32]]]
// CHECK: return

// CHECK-LABEL: func.func @direct_transpose_output_31x31
// CHECK: scf.for [[ROW31:%[a-zA-Z0-9_]+]] =
// CHECK: scf.for [[COL31:%[a-zA-Z0-9_]+]] =
// CHECK: [[VALUE31:%[a-zA-Z0-9_]+]] = memref.load {{.*}}[[[COL31]], [[ROW31]]]
// CHECK: memref.store [[VALUE31]], {{.*}}[[[ROW31]], [[COL31]]]
// CHECK: } else {
// CHECK: vector.transpose
// CHECK: memref.copy
// CHECK: return

// CHECK-LABEL: func.func @direct_transpose_output_identity_permutation
// CHECK-NOT: scf.for
// CHECK: vector.transpose {{.*}}, [0, 1]
// CHECK: return

// CHECK-LABEL: func.func @direct_transpose_output_versioned_nonsquare
// CHECK: scf.if
// CHECK: scf.for [[NROW:%.*]] =
// CHECK: scf.for [[NCOL:%.*]] =
// CHECK: [[NVALUE:%.*]] = memref.load {{.*}}[[[NCOL]], [[NROW]]]
// CHECK: memref.store [[NVALUE]], {{.*}}[[[NROW]], [[NCOL]]]
// CHECK: } else {
// CHECK: vector.transpose
// CHECK: return

// -----

// Buffered copy forms and exclusive-use checks.
// Match the current master bufferization layout without changing copy policy.
module {
func.func @direct_transpose_output_master_copy_forms(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
  %cst = arith.constant dense<0.000000e+00> : vector<16x16xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c16_i32 = arith.constant 16 : i32
  %c32 = arith.constant 32 : index
  %c16 = arith.constant 16 : index
  %0 = arith.muli %arg5, %c16_i32 : i32
  %1 = arith.index_cast %0 : i32 to index
  %2 = arith.muli %arg6, %c16_i32 : i32
  %3 = arith.index_cast %2 : i32 to index
  %4 = arith.muli %1, %c32 : index
  %5 = arith.addi %4, %3 : index
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%5], sizes: [16, 16], strides: [32, 1] : memref<*xf32> to memref<16x16xf32, strided<[32, 1], offset: ?>>
  %6 = arith.addi %1, %c16 : index
  %7 = arith.minsi %6, %c32 : index
  %8 = arith.maxsi %7, %1 : index
  %9 = arith.subi %8, %1 : index
  %10 = arith.addi %3, %c16 : index
  %11 = arith.minsi %10, %c32 : index
  %12 = arith.maxsi %11, %3 : index
  %13 = arith.subi %12, %3 : index
  %14 = arith.minsi %9, %c16 : index
  %15 = arith.minsi %13, %c16 : index
  %alloca = memref.alloca() : memref<16x16xf32>
  %16 = arith.cmpi eq, %14, %c16 : index
  %17 = arith.cmpi eq, %15, %c16 : index
  %18 = arith.andi %16, %17 : i1
  scf.if %18 {
    %23 = vector.transfer_read %reinterpret_cast[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf32, strided<[32, 1], offset: ?>>, vector<16x16xf32>
    vector.transfer_write %23, %alloca[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
  } else {
    %23 = arith.cmpi slt, %14, %c16 : index
    %24 = arith.cmpi slt, %15, %c16 : index
    %25 = arith.ori %23, %24 : i1
    scf.if %25 {
      vector.transfer_write %cst, %alloca[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
    }
    %subview = memref.subview %reinterpret_cast[0, 0] [%14, %15] [1, 1] : memref<16x16xf32, strided<[32, 1], offset: ?>> to memref<?x?xf32, strided<[32, 1], offset: ?>>
    %subview_3 = memref.subview %alloca[0, 0] [%14, %15] [1, 1] : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1]>>
    memref.copy %subview, %subview_3 : memref<?x?xf32, strided<[32, 1], offset: ?>> to memref<?x?xf32, strided<[16, 1]>>
  }
  %19 = vector.transfer_read %alloca[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf32>, vector<16x16xf32>
  %20 = vector.transpose %19, [1, 0] : vector<16x16xf32> to vector<16x16xf32>
  %alloca_1 = memref.alloca() {alignment = 64 : i64} : memref<16x16xf32>
  vector.transfer_write %20, %alloca_1[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
  %21 = arith.muli %3, %c32 : index
  %22 = arith.addi %21, %1 : index
  %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%22], sizes: [16, 16], strides: [32, 1] : memref<*xf32> to memref<16x16xf32, strided<[32, 1], offset: ?>>
  scf.if %18 {
    linalg.copy ins(%alloca_1 : memref<16x16xf32>) outs(%reinterpret_cast_2 : memref<16x16xf32, strided<[32, 1], offset: ?>>)
  } else {
    %subview = memref.subview %alloca_1[0, 0] [%15, %14] [1, 1] : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1]>>
    %subview_3 = memref.subview %reinterpret_cast_2[0, 0] [%15, %14] [1, 1] : memref<16x16xf32, strided<[32, 1], offset: ?>> to memref<?x?xf32, strided<[32, 1], offset: ?>>
    linalg.copy ins(%subview : memref<?x?xf32, strided<[16, 1]>>) outs(%subview_3 : memref<?x?xf32, strided<[32, 1], offset: ?>>)
  }
  return
}

  func.func @keep_unrelated_copy(%input: memref<2x2xf32>, %output: memref<2x2xf32>) {
    linalg.copy ins(%input : memref<2x2xf32>) outs(%output : memref<2x2xf32>)
    return
  }

func.func @reject_extra_scratch_user(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
  %cst = arith.constant dense<0.000000e+00> : vector<16x16xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c16_i32 = arith.constant 16 : i32
  %c32 = arith.constant 32 : index
  %c16 = arith.constant 16 : index
  %0 = arith.muli %arg5, %c16_i32 : i32
  %1 = arith.index_cast %0 : i32 to index
  %2 = arith.muli %arg6, %c16_i32 : i32
  %3 = arith.index_cast %2 : i32 to index
  %4 = arith.muli %1, %c32 : index
  %5 = arith.addi %4, %3 : index
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%5], sizes: [16, 16], strides: [32, 1] : memref<*xf32> to memref<16x16xf32, strided<[32, 1], offset: ?>>
  %6 = arith.addi %1, %c16 : index
  %7 = arith.minsi %6, %c32 : index
  %8 = arith.maxsi %7, %1 : index
  %9 = arith.subi %8, %1 : index
  %10 = arith.addi %3, %c16 : index
  %11 = arith.minsi %10, %c32 : index
  %12 = arith.maxsi %11, %3 : index
  %13 = arith.subi %12, %3 : index
  %14 = arith.minsi %9, %c16 : index
  %15 = arith.minsi %13, %c16 : index
  %alloca = memref.alloca() : memref<16x16xf32>
  %16 = arith.cmpi eq, %14, %c16 : index
  %17 = arith.cmpi eq, %15, %c16 : index
  %18 = arith.andi %16, %17 : i1
  scf.if %18 {
    %23 = vector.transfer_read %reinterpret_cast[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf32, strided<[32, 1], offset: ?>>, vector<16x16xf32>
    vector.transfer_write %23, %alloca[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
  } else {
    %23 = arith.cmpi slt, %14, %c16 : index
    %24 = arith.cmpi slt, %15, %c16 : index
    %25 = arith.ori %23, %24 : i1
    scf.if %25 {
      vector.transfer_write %cst, %alloca[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
    }
    %subview = memref.subview %reinterpret_cast[0, 0] [%14, %15] [1, 1] : memref<16x16xf32, strided<[32, 1], offset: ?>> to memref<?x?xf32, strided<[32, 1], offset: ?>>
    %subview_3 = memref.subview %alloca[0, 0] [%14, %15] [1, 1] : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1]>>
    memref.copy %subview, %subview_3 : memref<?x?xf32, strided<[32, 1], offset: ?>> to memref<?x?xf32, strided<[16, 1]>>
  }
  %extra_user = memref.dim %alloca, %c0 : memref<16x16xf32>
  %19 = vector.transfer_read %alloca[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf32>, vector<16x16xf32>
  %20 = vector.transpose %19, [1, 0] : vector<16x16xf32> to vector<16x16xf32>
  %alloca_1 = memref.alloca() {alignment = 64 : i64} : memref<16x16xf32>
  vector.transfer_write %20, %alloca_1[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
  %21 = arith.muli %3, %c32 : index
  %22 = arith.addi %21, %1 : index
  %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%22], sizes: [16, 16], strides: [32, 1] : memref<*xf32> to memref<16x16xf32, strided<[32, 1], offset: ?>>
  scf.if %18 {
    linalg.copy ins(%alloca_1 : memref<16x16xf32>) outs(%reinterpret_cast_2 : memref<16x16xf32, strided<[32, 1], offset: ?>>)
  } else {
    %subview = memref.subview %alloca_1[0, 0] [%15, %14] [1, 1] : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1]>>
    %subview_3 = memref.subview %reinterpret_cast_2[0, 0] [%15, %14] [1, 1] : memref<16x16xf32, strided<[32, 1], offset: ?>> to memref<?x?xf32, strided<[32, 1], offset: ?>>
    linalg.copy ins(%subview : memref<?x?xf32, strided<[16, 1]>>) outs(%subview_3 : memref<?x?xf32, strided<[32, 1], offset: ?>>)
  }
  return
}

func.func @reject_extra_transpose_user(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
  %cst = arith.constant dense<0.000000e+00> : vector<16x16xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c16_i32 = arith.constant 16 : i32
  %c32 = arith.constant 32 : index
  %c16 = arith.constant 16 : index
  %0 = arith.muli %arg5, %c16_i32 : i32
  %1 = arith.index_cast %0 : i32 to index
  %2 = arith.muli %arg6, %c16_i32 : i32
  %3 = arith.index_cast %2 : i32 to index
  %4 = arith.muli %1, %c32 : index
  %5 = arith.addi %4, %3 : index
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%5], sizes: [16, 16], strides: [32, 1] : memref<*xf32> to memref<16x16xf32, strided<[32, 1], offset: ?>>
  %6 = arith.addi %1, %c16 : index
  %7 = arith.minsi %6, %c32 : index
  %8 = arith.maxsi %7, %1 : index
  %9 = arith.subi %8, %1 : index
  %10 = arith.addi %3, %c16 : index
  %11 = arith.minsi %10, %c32 : index
  %12 = arith.maxsi %11, %3 : index
  %13 = arith.subi %12, %3 : index
  %14 = arith.minsi %9, %c16 : index
  %15 = arith.minsi %13, %c16 : index
  %alloca = memref.alloca() : memref<16x16xf32>
  %16 = arith.cmpi eq, %14, %c16 : index
  %17 = arith.cmpi eq, %15, %c16 : index
  %18 = arith.andi %16, %17 : i1
  scf.if %18 {
    %23 = vector.transfer_read %reinterpret_cast[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf32, strided<[32, 1], offset: ?>>, vector<16x16xf32>
    vector.transfer_write %23, %alloca[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
  } else {
    %23 = arith.cmpi slt, %14, %c16 : index
    %24 = arith.cmpi slt, %15, %c16 : index
    %25 = arith.ori %23, %24 : i1
    scf.if %25 {
      vector.transfer_write %cst, %alloca[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
    }
    %subview = memref.subview %reinterpret_cast[0, 0] [%14, %15] [1, 1] : memref<16x16xf32, strided<[32, 1], offset: ?>> to memref<?x?xf32, strided<[32, 1], offset: ?>>
    %subview_3 = memref.subview %alloca[0, 0] [%14, %15] [1, 1] : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1]>>
    memref.copy %subview, %subview_3 : memref<?x?xf32, strided<[32, 1], offset: ?>> to memref<?x?xf32, strided<[16, 1]>>
  }
  %19 = vector.transfer_read %alloca[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf32>, vector<16x16xf32>
  %20 = vector.transpose %19, [1, 0] : vector<16x16xf32> to vector<16x16xf32>
  %extra_user = arith.addf %20, %20 : vector<16x16xf32>
  %alloca_1 = memref.alloca() {alignment = 64 : i64} : memref<16x16xf32>
  vector.transfer_write %20, %alloca_1[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
  %21 = arith.muli %3, %c32 : index
  %22 = arith.addi %21, %1 : index
  %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%22], sizes: [16, 16], strides: [32, 1] : memref<*xf32> to memref<16x16xf32, strided<[32, 1], offset: ?>>
  scf.if %18 {
    linalg.copy ins(%alloca_1 : memref<16x16xf32>) outs(%reinterpret_cast_2 : memref<16x16xf32, strided<[32, 1], offset: ?>>)
  } else {
    %subview = memref.subview %alloca_1[0, 0] [%15, %14] [1, 1] : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1]>>
    %subview_3 = memref.subview %reinterpret_cast_2[0, 0] [%15, %14] [1, 1] : memref<16x16xf32, strided<[32, 1], offset: ?>> to memref<?x?xf32, strided<[32, 1], offset: ?>>
    linalg.copy ins(%subview : memref<?x?xf32, strided<[16, 1]>>) outs(%subview_3 : memref<?x?xf32, strided<[32, 1], offset: ?>>)
  }
  return
}
}

// CHECK-LABEL: func.func @direct_transpose_output_master_copy_forms
// CHECK: scf.for [[ROW:%[a-zA-Z0-9_]+]] =
// CHECK: scf.for [[COL:%[a-zA-Z0-9_]+]] =
// CHECK: [[VALUE:%[a-zA-Z0-9_]+]] = memref.load {{.*}}[[[COL]], [[ROW]]]
// CHECK: memref.store [[VALUE]], {{.*}}[[[ROW]], [[COL]]]
// CHECK: } else {
// CHECK: vector.transpose
// CHECK: linalg.copy
// CHECK: return
// CHECK-LABEL: func.func @keep_unrelated_copy
// CHECK-NOT: memref.copy
// CHECK: linalg.copy
// CHECK: return

// CHECK-LABEL: func.func @reject_extra_scratch_user
// CHECK-NOT: scf.for
// CHECK: vector.transpose
// CHECK-NOT: scf.for
// CHECK: linalg.copy
// CHECK-NOT: scf.for
// CHECK: return

// CHECK-LABEL: func.func @reject_extra_transpose_user
// CHECK-NOT: scf.for
// CHECK: vector.transpose
// CHECK-NOT: scf.for
// CHECK: linalg.copy
// CHECK-NOT: scf.for
// CHECK: return

// -----

// Aliasing, operation order, element types, and scratch ownership.
// Reject output alias effects, cross-block setup, and converting copies.
module {

// Do not move the final output write across an aliasing write.
  func.func @scalar_alias_write(%input: memref<*xf32>, %output: memref<*xf32>) {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %input_parent = memref.reinterpret_cast %input to offset: [0], sizes: [2, 2], strides: [4, 1] : memref<*xf32> to memref<2x2xf32, strided<[4, 1]>>
    %alias = memref.reinterpret_cast %output to offset: [0], sizes: [2, 2], strides: [4, 1] : memref<*xf32> to memref<2x2xf32, strided<[4, 1]>>
    %ninety_nine = arith.constant 99.0 : f32
    %scratch = memref.alloc() : memref<2x2xf32>
    %source = memref.subview %input_parent[0, 0] [2, 2] [1, 1] : memref<2x2xf32, strided<[4, 1]>> to memref<2x2xf32, strided<[4, 1]>>
    %staged = memref.subview %scratch[0, 0] [2, 2] [1, 1] : memref<2x2xf32> to memref<2x2xf32, strided<[2, 1]>>
    memref.copy %source, %staged : memref<2x2xf32, strided<[4, 1]>> to memref<2x2xf32, strided<[2, 1]>>
    %transposed = memref.alloc() : memref<2x2xf32>
    linalg.transpose ins(%scratch : memref<2x2xf32>) outs(%transposed : memref<2x2xf32>) permutation = [1, 0]
    memref.store %ninety_nine, %alias[%c0, %c0] : memref<2x2xf32, strided<[4, 1]>>
    %destination_parent = memref.reinterpret_cast %output to offset: [0], sizes: [2, 2], strides: [4, 1] : memref<*xf32> to memref<2x2xf32, strided<[4, 1]>>
    %materialized = memref.alloc() : memref<2x2xf32>
    linalg.copy ins(%transposed : memref<2x2xf32>) outs(%materialized : memref<2x2xf32>)
    %source_view = memref.subview %materialized[0, 0] [2, 2] [1, 1] : memref<2x2xf32> to memref<2x2xf32, strided<[2, 1]>>
    %destination_view = memref.subview %destination_parent[0, 0] [2, 2] [1, 1] : memref<2x2xf32, strided<[4, 1]>> to memref<2x2xf32, strided<[4, 1]>>
    memref.copy %source_view, %destination_view : memref<2x2xf32, strided<[2, 1]>> to memref<2x2xf32, strided<[4, 1]>>
    memref.dealloc %scratch : memref<2x2xf32>
    memref.dealloc %transposed : memref<2x2xf32>
    memref.dealloc %materialized : memref<2x2xf32>
    return
  }
// CHECK-LABEL: func.func @scalar_alias_write
// CHECK-NOT: scf.if
// CHECK: linalg.transpose
// CHECK-NOT: scf.if
// CHECK: return

// An aliasing read must still observe the output before the copy.
  func.func @scalar_alias_read(%input: memref<*xf32>, %output: memref<*xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %input_parent = memref.reinterpret_cast %input to offset: [0], sizes: [2, 2], strides: [4, 1] : memref<*xf32> to memref<2x2xf32, strided<[4, 1]>>
    %alias = memref.reinterpret_cast %output to offset: [0], sizes: [2, 2], strides: [4, 1] : memref<*xf32> to memref<2x2xf32, strided<[4, 1]>>
    %ninety_nine = arith.constant 99.0 : f32
    %scratch = memref.alloc() : memref<2x2xf32>
    %source = memref.subview %input_parent[0, 0] [2, 2] [1, 1] : memref<2x2xf32, strided<[4, 1]>> to memref<2x2xf32, strided<[4, 1]>>
    %staged = memref.subview %scratch[0, 0] [2, 2] [1, 1] : memref<2x2xf32> to memref<2x2xf32, strided<[2, 1]>>
    memref.copy %source, %staged : memref<2x2xf32, strided<[4, 1]>> to memref<2x2xf32, strided<[2, 1]>>
    %transposed = memref.alloc() : memref<2x2xf32>
    linalg.transpose ins(%scratch : memref<2x2xf32>) outs(%transposed : memref<2x2xf32>) permutation = [1, 0]
    %seen = memref.load %alias[%c0, %c0] : memref<2x2xf32, strided<[4, 1]>>
    %destination_parent = memref.reinterpret_cast %output to offset: [0], sizes: [2, 2], strides: [4, 1] : memref<*xf32> to memref<2x2xf32, strided<[4, 1]>>
    %materialized = memref.alloc() : memref<2x2xf32>
    linalg.copy ins(%transposed : memref<2x2xf32>) outs(%materialized : memref<2x2xf32>)
    %source_view = memref.subview %materialized[0, 0] [2, 2] [1, 1] : memref<2x2xf32> to memref<2x2xf32, strided<[2, 1]>>
    %destination_view = memref.subview %destination_parent[0, 0] [2, 2] [1, 1] : memref<2x2xf32, strided<[4, 1]>> to memref<2x2xf32, strided<[4, 1]>>
    memref.copy %source_view, %destination_view : memref<2x2xf32, strided<[2, 1]>> to memref<2x2xf32, strided<[4, 1]>>
    memref.dealloc %scratch : memref<2x2xf32>
    memref.dealloc %transposed : memref<2x2xf32>
    memref.dealloc %materialized : memref<2x2xf32>
    return %seen : f32
  }
// CHECK-LABEL: func.func @scalar_alias_read
// CHECK-NOT: scf.if
// CHECK: linalg.transpose
// CHECK-NOT: scf.if
// CHECK: return

// Keep valid nested setup unchanged instead of comparing blocks.
  func.func @scalar_outer_offset(%input: memref<*xf32>, %output: memref<*xf32>) {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %c1 = arith.constant 1 : index
    %outer_offset = arith.constant 0 : index
    scf.for %iteration = %c0 to %c1 step %c1 {
    %input_parent = memref.reinterpret_cast %input to offset: [0], sizes: [2, 2], strides: [4, 1] : memref<*xf32> to memref<2x2xf32, strided<[4, 1]>>
    %scratch = memref.alloc() : memref<2x2xf32>
    %source = memref.subview %input_parent[0, 0] [2, 2] [1, 1] : memref<2x2xf32, strided<[4, 1]>> to memref<2x2xf32, strided<[4, 1]>>
    %staged = memref.subview %scratch[0, 0] [2, 2] [1, 1] : memref<2x2xf32> to memref<2x2xf32, strided<[2, 1]>>
    memref.copy %source, %staged : memref<2x2xf32, strided<[4, 1]>> to memref<2x2xf32, strided<[2, 1]>>
    %transposed = memref.alloc() : memref<2x2xf32>
    linalg.transpose ins(%scratch : memref<2x2xf32>) outs(%transposed : memref<2x2xf32>) permutation = [1, 0]
    %destination_parent = memref.reinterpret_cast %output to offset: [%outer_offset], sizes: [2, 2], strides: [4, 1] : memref<*xf32> to memref<2x2xf32, strided<[4, 1], offset: ?>>
    %materialized = memref.alloc() : memref<2x2xf32>
    linalg.copy ins(%transposed : memref<2x2xf32>) outs(%materialized : memref<2x2xf32>)
    %source_view = memref.subview %materialized[0, 0] [2, 2] [1, 1] : memref<2x2xf32> to memref<2x2xf32, strided<[2, 1]>>
    %destination_view = memref.subview %destination_parent[0, 0] [2, 2] [1, 1] : memref<2x2xf32, strided<[4, 1], offset: ?>> to memref<2x2xf32, strided<[4, 1], offset: ?>>
    memref.copy %source_view, %destination_view : memref<2x2xf32, strided<[2, 1]>> to memref<2x2xf32, strided<[4, 1], offset: ?>>
    memref.dealloc %scratch : memref<2x2xf32>
    memref.dealloc %transposed : memref<2x2xf32>
    memref.dealloc %materialized : memref<2x2xf32>
    }
    return
  }
// CHECK-LABEL: func.func @scalar_outer_offset
// CHECK-NOT: scf.if
// CHECK: linalg.transpose
// CHECK-NOT: scf.if
// CHECK: return

// Keep the f16-to-f32 conversion in the materialization copy.
  func.func @scalar_converting_copy(%input: memref<*xf16>, %output: memref<*xf32>) {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %input_parent = memref.reinterpret_cast %input to offset: [0], sizes: [2, 2], strides: [4, 1] : memref<*xf16> to memref<2x2xf16, strided<[4, 1]>>
    %scratch = memref.alloc() : memref<2x2xf16>
    %source = memref.subview %input_parent[0, 0] [2, 2] [1, 1] : memref<2x2xf16, strided<[4, 1]>> to memref<2x2xf16, strided<[4, 1]>>
    %staged = memref.subview %scratch[0, 0] [2, 2] [1, 1] : memref<2x2xf16> to memref<2x2xf16, strided<[2, 1]>>
    memref.copy %source, %staged : memref<2x2xf16, strided<[4, 1]>> to memref<2x2xf16, strided<[2, 1]>>
    %transposed = memref.alloc() : memref<2x2xf16>
    linalg.transpose ins(%scratch : memref<2x2xf16>) outs(%transposed : memref<2x2xf16>) permutation = [1, 0]
    %destination_parent = memref.reinterpret_cast %output to offset: [0], sizes: [2, 2], strides: [4, 1] : memref<*xf32> to memref<2x2xf32, strided<[4, 1]>>
    %materialized = memref.alloc() : memref<2x2xf32>
    linalg.copy ins(%transposed : memref<2x2xf16>) outs(%materialized : memref<2x2xf32>)
    %source_view = memref.subview %materialized[0, 0] [2, 2] [1, 1] : memref<2x2xf32> to memref<2x2xf32, strided<[2, 1]>>
    %destination_view = memref.subview %destination_parent[0, 0] [2, 2] [1, 1] : memref<2x2xf32, strided<[4, 1]>> to memref<2x2xf32, strided<[4, 1]>>
    memref.copy %source_view, %destination_view : memref<2x2xf32, strided<[2, 1]>> to memref<2x2xf32, strided<[4, 1]>>
    memref.dealloc %scratch : memref<2x2xf16>
    memref.dealloc %transposed : memref<2x2xf16>
    memref.dealloc %materialized : memref<2x2xf32>
    return
  }
// CHECK-LABEL: func.func @scalar_converting_copy
// CHECK-NOT: scf.if
// CHECK: linalg.transpose
// CHECK-NOT: scf.if
// CHECK: return

// Do not move the final output write across an aliasing write.
  func.func @vector_alias_write(%input: memref<*xf32>, %output: memref<*xf32>) {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %input_parent = memref.reinterpret_cast %input to offset: [0], sizes: [2, 2], strides: [4, 1] : memref<*xf32> to memref<2x2xf32, strided<[4, 1]>>
    %alias = memref.reinterpret_cast %output to offset: [0], sizes: [2, 2], strides: [4, 1] : memref<*xf32> to memref<2x2xf32, strided<[4, 1]>>
    %ninety_nine = arith.constant 99.0 : f32
    %scratch = memref.alloc() : memref<2x2xf32>
    %source = memref.subview %input_parent[0, 0] [2, 2] [1, 1] : memref<2x2xf32, strided<[4, 1]>> to memref<2x2xf32, strided<[4, 1]>>
    %staged = memref.subview %scratch[0, 0] [2, 2] [1, 1] : memref<2x2xf32> to memref<2x2xf32, strided<[2, 1]>>
    memref.copy %source, %staged : memref<2x2xf32, strided<[4, 1]>> to memref<2x2xf32, strided<[2, 1]>>
    %snapshot = vector.transfer_read %scratch[%c0, %c0], %zero {in_bounds = [true, true]} : memref<2x2xf32>, vector<2x2xf32>
    %transpose = vector.transpose %snapshot, [1, 0] : vector<2x2xf32> to vector<2x2xf32>
    %transposed = memref.alloc() : memref<2x2xf32>
    vector.transfer_write %transpose, %transposed[%c0, %c0] {in_bounds = [true, true]} : vector<2x2xf32>, memref<2x2xf32>
    memref.store %ninety_nine, %alias[%c0, %c0] : memref<2x2xf32, strided<[4, 1]>>
    %destination_parent = memref.reinterpret_cast %output to offset: [0], sizes: [2, 2], strides: [4, 1] : memref<*xf32> to memref<2x2xf32, strided<[4, 1]>>
    %materialized = memref.alloc() : memref<2x2xf32>
    linalg.copy ins(%transposed : memref<2x2xf32>) outs(%materialized : memref<2x2xf32>)
    %source_view = memref.subview %materialized[0, 0] [2, 2] [1, 1] : memref<2x2xf32> to memref<2x2xf32, strided<[2, 1]>>
    %destination_view = memref.subview %destination_parent[0, 0] [2, 2] [1, 1] : memref<2x2xf32, strided<[4, 1]>> to memref<2x2xf32, strided<[4, 1]>>
    memref.copy %source_view, %destination_view : memref<2x2xf32, strided<[2, 1]>> to memref<2x2xf32, strided<[4, 1]>>
    memref.dealloc %scratch : memref<2x2xf32>
    memref.dealloc %transposed : memref<2x2xf32>
    memref.dealloc %materialized : memref<2x2xf32>
    return
  }
// CHECK-LABEL: func.func @vector_alias_write
// CHECK-NOT: scf.if
// CHECK: vector.transpose
// CHECK-NOT: scf.if
// CHECK: return

// An aliasing read must still observe the output before the copy.
  func.func @vector_alias_read(%input: memref<*xf32>, %output: memref<*xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %input_parent = memref.reinterpret_cast %input to offset: [0], sizes: [2, 2], strides: [4, 1] : memref<*xf32> to memref<2x2xf32, strided<[4, 1]>>
    %alias = memref.reinterpret_cast %output to offset: [0], sizes: [2, 2], strides: [4, 1] : memref<*xf32> to memref<2x2xf32, strided<[4, 1]>>
    %ninety_nine = arith.constant 99.0 : f32
    %scratch = memref.alloc() : memref<2x2xf32>
    %source = memref.subview %input_parent[0, 0] [2, 2] [1, 1] : memref<2x2xf32, strided<[4, 1]>> to memref<2x2xf32, strided<[4, 1]>>
    %staged = memref.subview %scratch[0, 0] [2, 2] [1, 1] : memref<2x2xf32> to memref<2x2xf32, strided<[2, 1]>>
    memref.copy %source, %staged : memref<2x2xf32, strided<[4, 1]>> to memref<2x2xf32, strided<[2, 1]>>
    %snapshot = vector.transfer_read %scratch[%c0, %c0], %zero {in_bounds = [true, true]} : memref<2x2xf32>, vector<2x2xf32>
    %transpose = vector.transpose %snapshot, [1, 0] : vector<2x2xf32> to vector<2x2xf32>
    %transposed = memref.alloc() : memref<2x2xf32>
    vector.transfer_write %transpose, %transposed[%c0, %c0] {in_bounds = [true, true]} : vector<2x2xf32>, memref<2x2xf32>
    %seen = memref.load %alias[%c0, %c0] : memref<2x2xf32, strided<[4, 1]>>
    %destination_parent = memref.reinterpret_cast %output to offset: [0], sizes: [2, 2], strides: [4, 1] : memref<*xf32> to memref<2x2xf32, strided<[4, 1]>>
    %materialized = memref.alloc() : memref<2x2xf32>
    linalg.copy ins(%transposed : memref<2x2xf32>) outs(%materialized : memref<2x2xf32>)
    %source_view = memref.subview %materialized[0, 0] [2, 2] [1, 1] : memref<2x2xf32> to memref<2x2xf32, strided<[2, 1]>>
    %destination_view = memref.subview %destination_parent[0, 0] [2, 2] [1, 1] : memref<2x2xf32, strided<[4, 1]>> to memref<2x2xf32, strided<[4, 1]>>
    memref.copy %source_view, %destination_view : memref<2x2xf32, strided<[2, 1]>> to memref<2x2xf32, strided<[4, 1]>>
    memref.dealloc %scratch : memref<2x2xf32>
    memref.dealloc %transposed : memref<2x2xf32>
    memref.dealloc %materialized : memref<2x2xf32>
    return %seen : f32
  }
// CHECK-LABEL: func.func @vector_alias_read
// CHECK-NOT: scf.if
// CHECK: vector.transpose
// CHECK-NOT: scf.if
// CHECK: return

// Keep valid nested setup unchanged instead of comparing blocks.
  func.func @vector_outer_offset(%input: memref<*xf32>, %output: memref<*xf32>) {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %c1 = arith.constant 1 : index
    %outer_offset = arith.constant 0 : index
    scf.for %iteration = %c0 to %c1 step %c1 {
    %input_parent = memref.reinterpret_cast %input to offset: [0], sizes: [2, 2], strides: [4, 1] : memref<*xf32> to memref<2x2xf32, strided<[4, 1]>>
    %scratch = memref.alloc() : memref<2x2xf32>
    %source = memref.subview %input_parent[0, 0] [2, 2] [1, 1] : memref<2x2xf32, strided<[4, 1]>> to memref<2x2xf32, strided<[4, 1]>>
    %staged = memref.subview %scratch[0, 0] [2, 2] [1, 1] : memref<2x2xf32> to memref<2x2xf32, strided<[2, 1]>>
    memref.copy %source, %staged : memref<2x2xf32, strided<[4, 1]>> to memref<2x2xf32, strided<[2, 1]>>
    %snapshot = vector.transfer_read %scratch[%c0, %c0], %zero {in_bounds = [true, true]} : memref<2x2xf32>, vector<2x2xf32>
    %transpose = vector.transpose %snapshot, [1, 0] : vector<2x2xf32> to vector<2x2xf32>
    %transposed = memref.alloc() : memref<2x2xf32>
    vector.transfer_write %transpose, %transposed[%c0, %c0] {in_bounds = [true, true]} : vector<2x2xf32>, memref<2x2xf32>
    %destination_parent = memref.reinterpret_cast %output to offset: [%outer_offset], sizes: [2, 2], strides: [4, 1] : memref<*xf32> to memref<2x2xf32, strided<[4, 1], offset: ?>>
    %materialized = memref.alloc() : memref<2x2xf32>
    linalg.copy ins(%transposed : memref<2x2xf32>) outs(%materialized : memref<2x2xf32>)
    %source_view = memref.subview %materialized[0, 0] [2, 2] [1, 1] : memref<2x2xf32> to memref<2x2xf32, strided<[2, 1]>>
    %destination_view = memref.subview %destination_parent[0, 0] [2, 2] [1, 1] : memref<2x2xf32, strided<[4, 1], offset: ?>> to memref<2x2xf32, strided<[4, 1], offset: ?>>
    memref.copy %source_view, %destination_view : memref<2x2xf32, strided<[2, 1]>> to memref<2x2xf32, strided<[4, 1], offset: ?>>
    memref.dealloc %scratch : memref<2x2xf32>
    memref.dealloc %transposed : memref<2x2xf32>
    memref.dealloc %materialized : memref<2x2xf32>
    }
    return
  }
// CHECK-LABEL: func.func @vector_outer_offset
// CHECK-NOT: scf.if
// CHECK: vector.transpose
// CHECK-NOT: scf.if
// CHECK: return

// Keep the f16-to-f32 conversion in the materialization copy.
  func.func @vector_converting_copy(%input: memref<*xf16>, %output: memref<*xf32>) {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f16
    %input_parent = memref.reinterpret_cast %input to offset: [0], sizes: [2, 2], strides: [4, 1] : memref<*xf16> to memref<2x2xf16, strided<[4, 1]>>
    %scratch = memref.alloc() : memref<2x2xf16>
    %source = memref.subview %input_parent[0, 0] [2, 2] [1, 1] : memref<2x2xf16, strided<[4, 1]>> to memref<2x2xf16, strided<[4, 1]>>
    %staged = memref.subview %scratch[0, 0] [2, 2] [1, 1] : memref<2x2xf16> to memref<2x2xf16, strided<[2, 1]>>
    memref.copy %source, %staged : memref<2x2xf16, strided<[4, 1]>> to memref<2x2xf16, strided<[2, 1]>>
    %snapshot = vector.transfer_read %scratch[%c0, %c0], %zero {in_bounds = [true, true]} : memref<2x2xf16>, vector<2x2xf16>
    %transpose = vector.transpose %snapshot, [1, 0] : vector<2x2xf16> to vector<2x2xf16>
    %transposed = memref.alloc() : memref<2x2xf16>
    vector.transfer_write %transpose, %transposed[%c0, %c0] {in_bounds = [true, true]} : vector<2x2xf16>, memref<2x2xf16>
    %destination_parent = memref.reinterpret_cast %output to offset: [0], sizes: [2, 2], strides: [4, 1] : memref<*xf32> to memref<2x2xf32, strided<[4, 1]>>
    %materialized = memref.alloc() : memref<2x2xf32>
    linalg.copy ins(%transposed : memref<2x2xf16>) outs(%materialized : memref<2x2xf32>)
    %source_view = memref.subview %materialized[0, 0] [2, 2] [1, 1] : memref<2x2xf32> to memref<2x2xf32, strided<[2, 1]>>
    %destination_view = memref.subview %destination_parent[0, 0] [2, 2] [1, 1] : memref<2x2xf32, strided<[4, 1]>> to memref<2x2xf32, strided<[4, 1]>>
    memref.copy %source_view, %destination_view : memref<2x2xf32, strided<[2, 1]>> to memref<2x2xf32, strided<[4, 1]>>
    memref.dealloc %scratch : memref<2x2xf16>
    memref.dealloc %transposed : memref<2x2xf16>
    memref.dealloc %materialized : memref<2x2xf32>
    return
  }
// CHECK-LABEL: func.func @vector_converting_copy
// CHECK-NOT: scf.if
// CHECK: vector.transpose
// CHECK-NOT: scf.if
// CHECK: return

// A scratch outside the loop cannot be ordered against operations inside it.
func.func @reject_hoisted_transpose_scratch(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
  %outer0 = arith.constant 0 : index
  %outer1 = arith.constant 1 : index
  %alloca_1 = memref.alloca() {alignment = 64 : i64} : memref<16x16xf32>
  scf.for %iteration = %outer0 to %outer1 step %outer1 {
  %cst = arith.constant dense<0.000000e+00> : vector<16x16xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c16_i32 = arith.constant 16 : i32
  %c32 = arith.constant 32 : index
  %c16 = arith.constant 16 : index
  %0 = arith.muli %arg5, %c16_i32 : i32
  %1 = arith.index_cast %0 : i32 to index
  %2 = arith.muli %arg6, %c16_i32 : i32
  %3 = arith.index_cast %2 : i32 to index
  %4 = arith.muli %1, %c32 : index
  %5 = arith.addi %4, %3 : index
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%5], sizes: [16, 16], strides: [32, 1] : memref<*xf32> to memref<16x16xf32, strided<[32, 1], offset: ?>>
  %6 = arith.addi %1, %c16 : index
  %7 = arith.minsi %6, %c32 : index
  %8 = arith.maxsi %7, %1 : index
  %9 = arith.subi %8, %1 : index
  %10 = arith.addi %3, %c16 : index
  %11 = arith.minsi %10, %c32 : index
  %12 = arith.maxsi %11, %3 : index
  %13 = arith.subi %12, %3 : index
  %14 = arith.minsi %9, %c16 : index
  %15 = arith.minsi %13, %c16 : index
  %alloca = memref.alloca() : memref<16x16xf32>
  %16 = arith.cmpi eq, %14, %c16 : index
  %17 = arith.cmpi eq, %15, %c16 : index
  %18 = arith.andi %16, %17 : i1
  scf.if %18 {
    %23 = vector.transfer_read %reinterpret_cast[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf32, strided<[32, 1], offset: ?>>, vector<16x16xf32>
    vector.transfer_write %23, %alloca[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
  } else {
    %23 = arith.cmpi slt, %14, %c16 : index
    %24 = arith.cmpi slt, %15, %c16 : index
    %25 = arith.ori %23, %24 : i1
    scf.if %25 {
      vector.transfer_write %cst, %alloca[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
    }
    %subview = memref.subview %reinterpret_cast[0, 0] [%14, %15] [1, 1] : memref<16x16xf32, strided<[32, 1], offset: ?>> to memref<?x?xf32, strided<[32, 1], offset: ?>>
    %subview_3 = memref.subview %alloca[0, 0] [%14, %15] [1, 1] : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1]>>
    memref.copy %subview, %subview_3 : memref<?x?xf32, strided<[32, 1], offset: ?>> to memref<?x?xf32, strided<[16, 1]>>
  }
  %19 = vector.transfer_read %alloca[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf32>, vector<16x16xf32>
  %20 = vector.transpose %19, [1, 0] : vector<16x16xf32> to vector<16x16xf32>
  vector.transfer_write %20, %alloca_1[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
  %21 = arith.muli %3, %c32 : index
  %22 = arith.addi %21, %1 : index
  %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%22], sizes: [16, 16], strides: [32, 1] : memref<*xf32> to memref<16x16xf32, strided<[32, 1], offset: ?>>
  scf.if %18 {
    linalg.copy ins(%alloca_1 : memref<16x16xf32>) outs(%reinterpret_cast_2 : memref<16x16xf32, strided<[32, 1], offset: ?>>)
  } else {
    %subview = memref.subview %alloca_1[0, 0] [%15, %14] [1, 1] : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1]>>
    %subview_3 = memref.subview %reinterpret_cast_2[0, 0] [%15, %14] [1, 1] : memref<16x16xf32, strided<[32, 1], offset: ?>> to memref<?x?xf32, strided<[32, 1], offset: ?>>
    linalg.copy ins(%subview : memref<?x?xf32, strided<[16, 1]>>) outs(%subview_3 : memref<?x?xf32, strided<[32, 1], offset: ?>>)
  }
  }
  return
}

// CHECK-LABEL: func.func @reject_hoisted_transpose_scratch
// CHECK-NOT: memref.load
// CHECK-NOT: memref.store
// CHECK: vector.transpose
// CHECK-NOT: memref.load
// CHECK-NOT: memref.store
// CHECK: return

// A nonsquare positive control prevents blanket rejection of this copy form.
func.func @scalar_direct_output(%input: memref<*xf32>, %output: memref<*xf32>) {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %input_parent = memref.reinterpret_cast %input to offset: [0], sizes: [2, 3], strides: [4, 1] : memref<*xf32> to memref<2x3xf32, strided<[4, 1]>>
  %scratch = memref.alloc() : memref<2x3xf32>
  %source = memref.subview %input_parent[0, 0] [2, 3] [1, 1] : memref<2x3xf32, strided<[4, 1]>> to memref<2x3xf32, strided<[4, 1]>>
  %staged = memref.subview %scratch[0, 0] [2, 3] [1, 1] : memref<2x3xf32> to memref<2x3xf32, strided<[3, 1]>>
  memref.copy %source, %staged : memref<2x3xf32, strided<[4, 1]>> to memref<2x3xf32, strided<[3, 1]>>
  %transposed = memref.alloc() : memref<3x2xf32>
  linalg.transpose ins(%scratch : memref<2x3xf32>) outs(%transposed : memref<3x2xf32>) permutation = [1, 0]
  %destination_parent = memref.reinterpret_cast %output to offset: [0], sizes: [3, 2], strides: [4, 1] : memref<*xf32> to memref<3x2xf32, strided<[4, 1]>>
  %materialized = memref.alloc() : memref<3x2xf32>
  linalg.copy ins(%transposed : memref<3x2xf32>) outs(%materialized : memref<3x2xf32>)
  %source_view = memref.subview %materialized[0, 0] [3, 2] [1, 1] : memref<3x2xf32> to memref<3x2xf32, strided<[2, 1]>>
  %destination_view = memref.subview %destination_parent[0, 0] [3, 2] [1, 1] : memref<3x2xf32, strided<[4, 1]>> to memref<3x2xf32, strided<[4, 1]>>
  memref.copy %source_view, %destination_view : memref<3x2xf32, strided<[2, 1]>> to memref<3x2xf32, strided<[4, 1]>>
  memref.dealloc %scratch : memref<2x3xf32>
  memref.dealloc %transposed : memref<3x2xf32>
  memref.dealloc %materialized : memref<3x2xf32>
  return
}
// CHECK-LABEL: func.func @scalar_direct_output
// CHECK: scf.if
// CHECK: scf.for [[SROW:%.*]] =
// CHECK: scf.for [[SCOL:%.*]] =
// CHECK: [[SVALUE:%.*]] = memref.load {{.*}}[[[SCOL]], [[SROW]]]
// CHECK: memref.store [[SVALUE]], {{.*}}[[[SROW]], [[SCOL]]]
// CHECK: } else {
// CHECK: linalg.transpose
// CHECK: return

// A nonsquare positive control prevents blanket rejection of this copy form.
func.func @vector_direct_output(%input: memref<*xf32>, %output: memref<*xf32>) {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %input_parent = memref.reinterpret_cast %input to offset: [0], sizes: [2, 3], strides: [4, 1] : memref<*xf32> to memref<2x3xf32, strided<[4, 1]>>
  %scratch = memref.alloc() : memref<2x3xf32>
  %source = memref.subview %input_parent[0, 0] [2, 3] [1, 1] : memref<2x3xf32, strided<[4, 1]>> to memref<2x3xf32, strided<[4, 1]>>
  %staged = memref.subview %scratch[0, 0] [2, 3] [1, 1] : memref<2x3xf32> to memref<2x3xf32, strided<[3, 1]>>
  memref.copy %source, %staged : memref<2x3xf32, strided<[4, 1]>> to memref<2x3xf32, strided<[3, 1]>>
  %snapshot = vector.transfer_read %scratch[%c0, %c0], %zero {in_bounds = [true, true]} : memref<2x3xf32>, vector<2x3xf32>
  %transpose = vector.transpose %snapshot, [1, 0] : vector<2x3xf32> to vector<3x2xf32>
  %transposed = memref.alloc() : memref<3x2xf32>
  vector.transfer_write %transpose, %transposed[%c0, %c0] {in_bounds = [true, true]} : vector<3x2xf32>, memref<3x2xf32>
  %destination_parent = memref.reinterpret_cast %output to offset: [0], sizes: [3, 2], strides: [4, 1] : memref<*xf32> to memref<3x2xf32, strided<[4, 1]>>
  %materialized = memref.alloc() : memref<3x2xf32>
  linalg.copy ins(%transposed : memref<3x2xf32>) outs(%materialized : memref<3x2xf32>)
  %source_view = memref.subview %materialized[0, 0] [3, 2] [1, 1] : memref<3x2xf32> to memref<3x2xf32, strided<[2, 1]>>
  %destination_view = memref.subview %destination_parent[0, 0] [3, 2] [1, 1] : memref<3x2xf32, strided<[4, 1]>> to memref<3x2xf32, strided<[4, 1]>>
  memref.copy %source_view, %destination_view : memref<3x2xf32, strided<[2, 1]>> to memref<3x2xf32, strided<[4, 1]>>
  memref.dealloc %scratch : memref<2x3xf32>
  memref.dealloc %transposed : memref<3x2xf32>
  memref.dealloc %materialized : memref<3x2xf32>
  return
}
// CHECK-LABEL: func.func @vector_direct_output
// CHECK: scf.if
// CHECK: scf.for [[VROW:%.*]] =
// CHECK: scf.for [[VCOL:%.*]] =
// CHECK: [[VVALUE:%.*]] = memref.load {{.*}}[[[VCOL]], [[VROW]]]
// CHECK: memref.store [[VVALUE]], {{.*}}[[[VROW]], [[VCOL]]]
// CHECK: } else {
// CHECK: vector.transpose
// CHECK: return
}

// -----

// Preserve vector snapshots across later scratch initialization.
// A vector read snapshots the scratch before the later fill. The rewrite must
// not replace that snapshot with scalar reads after the fill.
module {
  func.func @fill_after_snapshot(%input: memref<*xf32>, %output: memref<*xf32>) {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %input_parent = memref.reinterpret_cast %input to offset: [0], sizes: [2, 2], strides: [4, 1] : memref<*xf32> to memref<2x2xf32, strided<[4, 1]>>
    %scratch = memref.alloc() : memref<2x2xf32>
    %source = memref.subview %input_parent[0, 0] [2, 2] [1, 1] : memref<2x2xf32, strided<[4, 1]>> to memref<2x2xf32, strided<[4, 1]>>
    %staged = memref.subview %scratch[0, 0] [2, 2] [1, 1] : memref<2x2xf32> to memref<2x2xf32, strided<[2, 1]>>
    memref.copy %source, %staged : memref<2x2xf32, strided<[4, 1]>> to memref<2x2xf32, strided<[2, 1]>>
    %snapshot = vector.transfer_read %scratch[%c0, %c0], %zero {in_bounds = [true, true]} : memref<2x2xf32>, vector<2x2xf32>
    %transpose = vector.transpose %snapshot, [1, 0] : vector<2x2xf32> to vector<2x2xf32>
    linalg.fill ins(%zero : f32) outs(%scratch : memref<2x2xf32>)
    %transposed = memref.alloc() : memref<2x2xf32>
    vector.transfer_write %transpose, %transposed[%c0, %c0] {in_bounds = [true, true]} : vector<2x2xf32>, memref<2x2xf32>
    %destination_parent = memref.reinterpret_cast %output to offset: [0], sizes: [2, 2], strides: [4, 1] : memref<*xf32> to memref<2x2xf32, strided<[4, 1]>>
    %materialized = memref.alloc() : memref<2x2xf32>
    linalg.copy ins(%transposed : memref<2x2xf32>) outs(%materialized : memref<2x2xf32>)
    %source_view = memref.subview %materialized[0, 0] [2, 2] [1, 1] : memref<2x2xf32> to memref<2x2xf32, strided<[2, 1]>>
    %destination_view = memref.subview %destination_parent[0, 0] [2, 2] [1, 1] : memref<2x2xf32, strided<[4, 1]>> to memref<2x2xf32, strided<[4, 1]>>
    memref.copy %source_view, %destination_view : memref<2x2xf32, strided<[2, 1]>> to memref<2x2xf32, strided<[4, 1]>>
    memref.dealloc %scratch : memref<2x2xf32>
    memref.dealloc %transposed : memref<2x2xf32>
    memref.dealloc %materialized : memref<2x2xf32>
    return
  }
}

// CHECK-LABEL: func.func @fill_after_snapshot
// CHECK: [[SNAPSHOT:%.*]] = vector.transfer_read
// CHECK: vector.transpose [[SNAPSHOT]], [1, 0]
// CHECK: linalg.fill
// CHECK-NOT: scf.for
// CHECK: return

// -----

// Preserve scalar scratch initialization before the new branch.
// A fill before the original transpose can still follow the new branch's
// insertion point. Keep that chain unchanged to preserve initialization order.
module {
  func.func @fill_after_branch_insertion(%input: memref<*xf32>, %output: memref<*xf32>) {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %input_parent = memref.reinterpret_cast %input to offset: [0], sizes: [2, 2], strides: [4, 1] : memref<*xf32> to memref<2x2xf32, strided<[4, 1]>>
    %scratch = memref.alloc() : memref<2x2xf32>
    %source = memref.subview %input_parent[0, 0] [2, 2] [1, 1] : memref<2x2xf32, strided<[4, 1]>> to memref<2x2xf32, strided<[4, 1]>>
    %staged = memref.subview %scratch[0, 0] [2, 2] [1, 1] : memref<2x2xf32> to memref<2x2xf32, strided<[2, 1]>>
    memref.copy %source, %staged : memref<2x2xf32, strided<[4, 1]>> to memref<2x2xf32, strided<[2, 1]>>
    %transposed = memref.alloc() : memref<2x2xf32>
    linalg.fill ins(%zero : f32) outs(%scratch : memref<2x2xf32>)
    linalg.transpose ins(%scratch : memref<2x2xf32>) outs(%transposed : memref<2x2xf32>) permutation = [1, 0]
    %destination_parent = memref.reinterpret_cast %output to offset: [0], sizes: [2, 2], strides: [4, 1] : memref<*xf32> to memref<2x2xf32, strided<[4, 1]>>
    %materialized = memref.alloc() : memref<2x2xf32>
    linalg.copy ins(%transposed : memref<2x2xf32>) outs(%materialized : memref<2x2xf32>)
    %source_view = memref.subview %materialized[0, 0] [2, 2] [1, 1] : memref<2x2xf32> to memref<2x2xf32, strided<[2, 1]>>
    %destination_view = memref.subview %destination_parent[0, 0] [2, 2] [1, 1] : memref<2x2xf32, strided<[4, 1]>> to memref<2x2xf32, strided<[4, 1]>>
    memref.copy %source_view, %destination_view : memref<2x2xf32, strided<[2, 1]>> to memref<2x2xf32, strided<[4, 1]>>
    memref.dealloc %scratch : memref<2x2xf32>
    memref.dealloc %transposed : memref<2x2xf32>
    memref.dealloc %materialized : memref<2x2xf32>
    return
  }
}


// CHECK-LABEL: func.func @fill_after_branch_insertion
// CHECK-NOT: scf.for
// CHECK: linalg.fill
// CHECK-NOT: scf.for
// CHECK: linalg.transpose
// CHECK-NOT: scf.for
// CHECK: return
