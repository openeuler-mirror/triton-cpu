// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s

// A statically partial mask must stage the complete value before a bounded copy.
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


// CHECK-LABEL: func.func @masked_load_store_full_tile
// CHECK: scf.if
// CHECK: bufferization.materialize_in_destination
// CHECK-LABEL: func.func @masked_load_store_partial_tile
// CHECK: [[VALUE:%.*]] = bufferization.to_tensor
// CHECK: [[STAGING:%.*]] = memref.alloc() : memref<128xf32>
// CHECK: bufferization.materialize_in_destination [[VALUE]] in writable [[STAGING]] : (tensor<128xf32>, memref<128xf32>) -> ()
// CHECK: [[SRC:%.*]] = memref.subview [[STAGING]][0] [96] [1]
// CHECK: [[DST:%.*]] = memref.subview {{.*}}[0] [96] [1]
// CHECK: memref.copy [[SRC]], [[DST]]
// CHECK-NOT: tensor.extract_slice
// CHECK: return
