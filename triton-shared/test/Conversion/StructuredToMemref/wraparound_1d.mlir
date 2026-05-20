// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s

module {
  tt.func @wrap_1d(%arg0 : !tt.ptr<f32>) {
    %cst = arith.constant dense<-9.900000e+01> : tensor<8xf32>
    %cst_5 = arith.constant dense<5> : tensor<8xi32>
    %c3_i32 = arith.constant 3 : i32
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = arith.addi %0, %cst_5 : tensor<8xi32>
    %2 = tt.splat %c3_i32 : i32 -> tensor<8xi32>
    %3 = arith.remsi %1, %2 : tensor<8xi32>
    %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %5 = tt.addptr %4, %3 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %mask = arith.cmpi slt, %0, %2 : tensor<8xi32>
    %6 = tt.load %5, %mask, %cst : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL:  func.func @wrap_1d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[WRAPPED_:%.+]] = arith.remsi [[CST_5_]], [[CST_3_]] : index
// CHECK:           [[REMAINING_:%.+]] = arith.subi [[CST_3_]], [[WRAPPED_]] : index
// CHECK:           [[D1_:%.+]] = arith.minsi {{%.+}}, [[CST_8_]] : index
// CHECK:           [[CAST_0_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[WRAPPED_]]{{.}}, sizes: {{.}}[[D1_]]{{.}}, strides: {{.}}[[CST_1_]]{{.}} : memref<*xf32> to memref<?xf32, strided<[?], offset: ?>>
// CHECK:           arith.subi [[CST_8_]], [[D1_]] : index
// CHECK:           [[CAST_1_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: {{.}}{{%.+}}{{.}}, strides: {{.}}[[CST_1_]]{{.}} : memref<*xf32> to memref<?xf32, strided<[?], offset: ?>>
// CHECK:           [[ALLOC_:%.+]] = memref.alloc() : memref<8xf32>
// CHECK:           linalg.fill
// CHECK:           memref.copy {{%.+}}, {{%.+}} : memref<?xf32, strided<[?], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:           scf.for {{%.+}} = [[CST_0_]] to {{%.+}} step {{%.+}} {
// CHECK:             memref.copy {{%.+}}, {{%.+}} : memref<?xf32, strided<[?], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK:           }
// CHECK:           bufferization.to_tensor [[ALLOC_]] restrict writable : memref<8xf32>
// CHECK:           return
// CHECK:         }
