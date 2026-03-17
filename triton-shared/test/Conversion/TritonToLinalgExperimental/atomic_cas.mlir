// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
module {
  tt.func public @kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.atomic_cas acq_rel, gpu, %arg0, %c0_i32_0, %c1_i32 : (!tt.ptr<i32>, i32, i32) -> i32
    tt.return
  }
// CHECK-LABEL: func.func @kernel
// CHECK:    %c0 = arith.constant 0 : index
// CHECK:    %c1_i32 = arith.constant 1 : i32
// CHECK:    %c0_i32 = arith.constant 0 : i32
// CHECK:    %cast = memref.cast %arg0 : memref<*xi32> to memref<1xi32>
// CHECK:    %0 = memref.generic_atomic_rmw %cast[%c0] : memref<1xi32> {
// CHECK:    ^bb0(%arg7: i32):
// CHECK:      %1 = arith.cmpi eq, %arg7, %c0_i32 : i32
// CHECK:      %2 = arith.select %1, %c1_i32, %arg7 : i32
// CHECK:      memref.atomic_yield %2 : i32
// CHECK:    }
// CHECK:    return
  tt.func public @kernel_tensor(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32} loc(unknown)) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<4xi64>
    %cst_0 = arith.constant dense<0> : tensor<4xi64>
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %7 = tt.atomic_cas acq_rel, gpu, %6, %cst_0, %cst : (tensor<4x!tt.ptr<i64>>, tensor<4xi64>, tensor<4xi64>) -> tensor<4xi64>
    tt.return
  }
}

// CHECK-LABEL:  tt.func public @kernel_tensor
// CHECK:    %c1 = arith.constant 1 : index
// CHECK:    %c4 = arith.constant 4 : index
// CHECK:    %c0 = arith.constant 0 : index
// CHECK:    %9 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%8, %6 : tensor<4x!ptr.ptr<#ptr.generic_space>>, tensor<4xi32>) outs(%8 : tensor<4x!ptr.ptr<#ptr.generic_space>>) {
// CHECK:    ^bb0(%in: !ptr.ptr<#ptr.generic_space>, %in_0: i32, %out: !ptr.ptr<#ptr.generic_space>):
// CHECK:      %10 = arith.muli %in_0, %0 : i32
// CHECK:      %11 = tptr.ptradd %in %10 : <#ptr.generic_space>, i32 to <#ptr.generic_space>
// CHECK:      linalg.yield %11 : !ptr.ptr<#ptr.generic_space>
// CHECK:    } -> tensor<4x!ptr.ptr<#ptr.generic_space>>
// CHECK: scf.for %arg7 = %c0 to %c4 step %c1 {
// CHECK:      %extracted = tensor.extract %9[%arg7] : tensor<4x!ptr.ptr<#ptr.generic_space>>
// CHECK:      %10 = tptr.to_memref %extracted : <#ptr.generic_space> to memref<1xi64>
// CHECK:      %11 = memref.generic_atomic_rmw %10[%c0] : memref<1xi64> {
// CHECK:      ^bb0(%arg8: i64):
// CHECK:        %12 = arith.cmpi eq, %arg8, %c0_i64 : i64
// CHECK:        %13 = arith.select %12, %c2_i64, %arg8 : i64
// CHECK:        memref.atomic_yield %13 : i64
// CHECK:      }
// CHECK:    }
