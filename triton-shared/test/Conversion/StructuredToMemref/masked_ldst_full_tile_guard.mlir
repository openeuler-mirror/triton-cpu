// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s

// Two masked load/store kernels over a tile of size 128:
//  * full_tile has a dynamic bound `idx < %arg2` (a standard boundary-check
//    mask), so the pass inserts a runtime guard `dim == 128` and copies the
//    whole tile directly when it holds (fast path).
//  * partial_tile has a static bound `idx < 96`, which can never be a full
//    tile, so no guard is inserted and the general fill + subview path is used.
module {
  tt.func @masked_load_store_full_tile(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %1 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %ldptr = tt.addptr %0, %2 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %stptr = tt.addptr %1, %2 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %other = arith.constant dense<0.000000e+00> : tensor<128xf32>
    %5 = tt.splat %arg2 : i32 -> tensor<128xi32>
    %mask = arith.cmpi slt, %2, %5 : tensor<128xi32>
    %buff = tt.load %ldptr, %mask, %other : tensor<128x!tt.ptr<f32>>
    tt.store %stptr, %buff, %mask : tensor<128x!tt.ptr<f32>>
    tt.return
  }

  tt.func @masked_load_store_partial_tile(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %1 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %ldptr = tt.addptr %0, %2 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %stptr = tt.addptr %1, %2 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %other = arith.constant dense<0.000000e+00> : tensor<128xf32>
    %cst = arith.constant dense<96> : tensor<128xi32>
    %mask = arith.cmpi slt, %2, %cst : tensor<128xi32>
    %buff = tt.load %ldptr, %mask, %other : tensor<128x!tt.ptr<f32>>
    tt.store %stptr, %buff, %mask : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}

// Fast path: the dynamic bound yields a `dim == 128` guard; the then-branch
// copies the whole tile, the else-branch keeps the partial fill + subview path.
// CHECK-LABEL:  func.func @masked_load_store_full_tile
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_128_:%.+]] = arith.constant 128 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [128], strides: [1] : memref<*xf32> to memref<128xf32, strided<[1]>>
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [128], strides: [1] : memref<*xf32> to memref<128xf32, strided<[1]>>
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_1_:%.+]] = arith.minsi [[VAR_0_]], [[CST_128_]] : index
// CHECK:           [[VAR_2_:%.+]] = arith.maxsi [[VAR_1_]], [[CST_0_]] : index
// CHECK:           [[RES_:%.+]] = memref.alloc() : memref<128xf32>
// CHECK:           [[VAR_3_:%.+]] = arith.cmpi eq, [[VAR_2_]], [[CST_128_]] : index
// CHECK:           scf.if [[VAR_3_]] {
// CHECK:             memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<128xf32, strided<[1]>> to memref<128xf32>
// CHECK:             [[VAR_4_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<128xf32> to tensor<128xf32>
// CHECK:             bufferization.materialize_in_destination [[VAR_4_]] in writable [[VAR_reinterpret_cast_0_]] : (tensor<128xf32>, memref<128xf32, strided<[1]>>) -> ()
// CHECK:           } else {
// CHECK:             [[VAR_5_:%.+]] = arith.cmpi slt, [[VAR_2_]], [[CST_128_]] : index
// CHECK:             scf.if [[VAR_5_]] {
// CHECK:               linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[RES_]] : memref<128xf32>)
// CHECK:             }
// CHECK-DAG:         [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_2_]]{{.}} [1] : memref<128xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
// CHECK-DAG:         [[VAR_subview_1_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_2_]]{{.}} [1] : memref<128xf32> to memref<?xf32, strided<[1]>>
// CHECK:             memref.copy [[VAR_subview_]], [[VAR_subview_1_]] : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
// CHECK:             [[VAR_6_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<128xf32> to tensor<128xf32>
// CHECK-DAG:         [[VAR_extracted_slice_:%.+]] = tensor.extract_slice [[VAR_6_]][0] {{.}}[[VAR_2_]]{{.}} [1] : tensor<128xf32> to tensor<?xf32>
// CHECK-DAG:         [[VAR_subview_2_:%.+]] = memref.subview [[VAR_reinterpret_cast_0_]][0] {{.}}[[VAR_2_]]{{.}} [1] : memref<128xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
// CHECK:             bufferization.materialize_in_destination [[VAR_extracted_slice_]] in writable [[VAR_subview_2_]] : (tensor<?xf32>, memref<?xf32, strided<[1]>>) -> ()
// CHECK:           }
// CHECK:           return

// Original path: the static bound folds away the guard, so no `arith.cmpi eq`
// and no `scf.if` appear; only the plain fill + static `[96]` subview copy.
// CHECK-LABEL:  func.func @masked_load_store_partial_tile
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [128], strides: [1] : memref<*xf32> to memref<128xf32, strided<[1]>>
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [128], strides: [1] : memref<*xf32> to memref<128xf32, strided<[1]>>
// CHECK:           [[RES_:%.+]] = memref.alloc() : memref<128xf32>
// CHECK:           linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[RES_]] : memref<128xf32>)
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] [96] [1] : memref<128xf32, strided<[1]>> to memref<96xf32, strided<[1]>>
// CHECK-DAG:       [[VAR_subview_1_:%.+]] = memref.subview [[RES_]][0] [96] [1] : memref<128xf32> to memref<96xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_1_]] : memref<96xf32, strided<[1]>> to memref<96xf32, strided<[1]>>
// CHECK:           [[VAR_0_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<128xf32> to tensor<128xf32>
// CHECK-DAG:       [[VAR_extracted_slice_:%.+]] = tensor.extract_slice [[VAR_0_]][0] [96] [1] : tensor<128xf32> to tensor<96xf32>
// CHECK-DAG:       [[VAR_subview_2_:%.+]] = memref.subview [[VAR_reinterpret_cast_0_]][0] [96] [1] : memref<128xf32, strided<[1]>> to memref<96xf32, strided<[1]>>
// CHECK:           bufferization.materialize_in_destination [[VAR_extracted_slice_]] in writable [[VAR_subview_2_]] : (tensor<96xf32>, memref<96xf32, strided<[1]>>) -> ()
// CHECK:           return
