// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s
// RUN: env TRITON_SHARED_VERSION_TILE_BYTES=0 triton-shared-opt \
// RUN:   --triton-to-linalg-experimental %s | FileCheck %s --check-prefix=NOVER

// Two kernels over 64-float tiles, each run twice: once normally and once with
// TRITON_SHARED_VERSION_TILE_BYTES=0, so the before and after are one diff.
// The tile is 256 B, under the per-tile cap.  Both outputs are checked in full
// with CHECK-NEXT, so anything the pass adds, drops or reorders shows up.
//
// Without versioning the guard sits on each access and yields a tensor, so both
// of its arms write the same scratch slot -- one with a whole-tile memref.copy,
// one with linalg.fill plus a copy of a runtime number of elements.  SROA
// cannot split a slot written that way, which is why the scratch survives all
// the way to the assembly.
//
// With versioning the guard is hoisted around the region and yields nothing.
// Each arm gets its own scratch, and the fast arm's is written by a single
// unconditional constant-size copy, which SROA does promote.  Note in the
// output below that the versioned form allocates one buffer per arm
// (%alloc/%alloc_0 fast, %alloc_1/%alloc_2 slow) rather than sharing one.

module {
  tt.func @masked_load_versioned(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %2 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %3 = tt.addptr %1, %0 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %4 = tt.addptr %2, %0 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<64xi32>
    %mask = arith.cmpi slt, %0, %5 : tensor<64xi32>
    %a = tt.load %3, %mask, %cst : tensor<64x!tt.ptr<f32>>
    tt.store %4, %a : tensor<64x!tt.ptr<f32>>
    tt.return
  }

  tt.func @two_masked_loads_versioned(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>,
                                      %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %2 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %3 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %4 = tt.addptr %1, %0 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %5 = tt.addptr %2, %0 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %6 = tt.addptr %3, %0 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %7 = tt.splat %arg3 : i32 -> tensor<64xi32>
    %8 = tt.splat %arg4 : i32 -> tensor<64xi32>
    %maskA = arith.cmpi slt, %0, %7 : tensor<64xi32>
    %maskB = arith.cmpi slt, %0, %8 : tensor<64xi32>
    %a = tt.load %4, %maskA, %cst : tensor<64x!tt.ptr<f32>>
    %b = tt.load %5, %maskB, %cst : tensor<64x!tt.ptr<f32>>
    %sum = arith.addf %a, %b : tensor<64xf32>
    tt.store %6, %sum : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}

// Versioned output, in full.

// CHECK:       #map = affine_map<(d0) -> (d0)>
// CHECK-NEXT:  module {
// CHECK-NEXT:    func.func @masked_load_versioned(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
// CHECK-NEXT:      %alloc = memref.alloc() : memref<64xf32>
// CHECK-NEXT:      %alloc_0 = memref.alloc() : memref<64xf32>
// CHECK-NEXT:      %c64 = arith.constant 64 : index
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [64], strides: [1] : memref<*xf32> to memref<64xf32, strided<[1]>>
// CHECK-NEXT:      %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [64], strides: [1] : memref<*xf32> to memref<64xf32, strided<[1]>>
// CHECK-NEXT:      %0 = arith.index_cast %arg2 : i32 to index
// CHECK-NEXT:      %1 = arith.minsi %0, %c64 : index
// CHECK-NEXT:      %2 = arith.maxsi %1, %c0 : index
// CHECK-NEXT:      %3 = arith.cmpi eq, %2, %c64 : index
// CHECK-NEXT:      scf.if %3 {
// CHECK-NEXT:        memref.copy %reinterpret_cast, %alloc : memref<64xf32, strided<[1]>> to memref<64xf32>
// CHECK-NEXT:        %4 = bufferization.to_tensor %alloc restrict writable : memref<64xf32> to tensor<64xf32>
// CHECK-NEXT:        bufferization.materialize_in_destination %4 in writable %reinterpret_cast_1 : (tensor<64xf32>, memref<64xf32, strided<[1]>>) -> ()
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %4 = arith.cmpi slt, %2, %c64 : index
// CHECK-NEXT:        scf.if %4 {
// CHECK-NEXT:          linalg.fill ins(%cst : f32) outs(%alloc_0 : memref<64xf32>)
// CHECK-NEXT:        }
// CHECK-NEXT:        %subview = memref.subview %reinterpret_cast[0] [%2] [1] : memref<64xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
// CHECK-NEXT:        %subview_2 = memref.subview %alloc_0[0] [%2] [1] : memref<64xf32> to memref<?xf32, strided<[1]>>
// CHECK-NEXT:        memref.copy %subview, %subview_2 : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
// CHECK-NEXT:        %5 = bufferization.to_tensor %alloc_0 restrict writable : memref<64xf32> to tensor<64xf32>
// CHECK-NEXT:        bufferization.materialize_in_destination %5 in writable %reinterpret_cast_1 : (tensor<64xf32>, memref<64xf32, strided<[1]>>) -> ()
// CHECK-NEXT:      } {tts.full_tile_versioned}
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @two_masked_loads_versioned(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
// CHECK-NEXT:      %alloc = memref.alloc() : memref<64xf32>
// CHECK-NEXT:      %alloc_0 = memref.alloc() : memref<64xf32>
// CHECK-NEXT:      %alloc_1 = memref.alloc() : memref<64xf32>
// CHECK-NEXT:      %alloc_2 = memref.alloc() : memref<64xf32>
// CHECK-NEXT:      %c64 = arith.constant 64 : index
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [64], strides: [1] : memref<*xf32> to memref<64xf32, strided<[1]>>
// CHECK-NEXT:      %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [64], strides: [1] : memref<*xf32> to memref<64xf32, strided<[1]>>
// CHECK-NEXT:      %reinterpret_cast_4 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [64], strides: [1] : memref<*xf32> to memref<64xf32, strided<[1]>>
// CHECK-NEXT:      %0 = arith.index_cast %arg3 : i32 to index
// CHECK-NEXT:      %1 = arith.minsi %0, %c64 : index
// CHECK-NEXT:      %2 = arith.maxsi %1, %c0 : index
// CHECK-NEXT:      %3 = arith.index_cast %arg4 : i32 to index
// CHECK-NEXT:      %4 = arith.minsi %3, %c64 : index
// CHECK-NEXT:      %5 = arith.maxsi %4, %c0 : index
// CHECK-NEXT:      %6 = arith.cmpi eq, %2, %c64 : index
// CHECK-NEXT:      %7 = arith.cmpi eq, %5, %c64 : index
// CHECK-NEXT:      %8 = arith.andi %6, %7 : i1
// CHECK-NEXT:      scf.if %8 {
// CHECK-NEXT:        memref.copy %reinterpret_cast, %alloc : memref<64xf32, strided<[1]>> to memref<64xf32>
// CHECK-NEXT:        %9 = bufferization.to_tensor %alloc restrict writable : memref<64xf32> to tensor<64xf32>
// CHECK-NEXT:        memref.copy %reinterpret_cast_3, %alloc_0 : memref<64xf32, strided<[1]>> to memref<64xf32>
// CHECK-NEXT:        %10 = bufferization.to_tensor %alloc_0 restrict writable : memref<64xf32> to tensor<64xf32>
// CHECK-NEXT:        %11 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%9, %10 : tensor<64xf32>, tensor<64xf32>) outs(%9 : tensor<64xf32>) {
// CHECK-NEXT:        ^bb0(%in: f32, %in_5: f32, %out: f32):
// CHECK-NEXT:          %12 = arith.addf %in, %in_5 : f32
// CHECK-NEXT:          linalg.yield %12 : f32
// CHECK-NEXT:        } -> tensor<64xf32>
// CHECK-NEXT:        bufferization.materialize_in_destination %11 in writable %reinterpret_cast_4 : (tensor<64xf32>, memref<64xf32, strided<[1]>>) -> ()
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %9 = scf.if %6 -> (tensor<64xf32>) {
// CHECK-NEXT:          memref.copy %reinterpret_cast, %alloc_1 : memref<64xf32, strided<[1]>> to memref<64xf32>
// CHECK-NEXT:          %12 = bufferization.to_tensor %alloc_1 restrict writable : memref<64xf32> to tensor<64xf32>
// CHECK-NEXT:          scf.yield %12 : tensor<64xf32>
// CHECK-NEXT:        } else {
// CHECK-NEXT:          %12 = arith.cmpi slt, %2, %c64 : index
// CHECK-NEXT:          scf.if %12 {
// CHECK-NEXT:            linalg.fill ins(%cst : f32) outs(%alloc_1 : memref<64xf32>)
// CHECK-NEXT:          }
// CHECK-NEXT:          %subview = memref.subview %reinterpret_cast[0] [%2] [1] : memref<64xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
// CHECK-NEXT:          %subview_5 = memref.subview %alloc_1[0] [%2] [1] : memref<64xf32> to memref<?xf32, strided<[1]>>
// CHECK-NEXT:          memref.copy %subview, %subview_5 : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
// CHECK-NEXT:          %13 = bufferization.to_tensor %alloc_1 restrict writable : memref<64xf32> to tensor<64xf32>
// CHECK-NEXT:          scf.yield %13 : tensor<64xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        %10 = scf.if %7 -> (tensor<64xf32>) {
// CHECK-NEXT:          memref.copy %reinterpret_cast_3, %alloc_2 : memref<64xf32, strided<[1]>> to memref<64xf32>
// CHECK-NEXT:          %12 = bufferization.to_tensor %alloc_2 restrict writable : memref<64xf32> to tensor<64xf32>
// CHECK-NEXT:          scf.yield %12 : tensor<64xf32>
// CHECK-NEXT:        } else {
// CHECK-NEXT:          %12 = arith.cmpi slt, %5, %c64 : index
// CHECK-NEXT:          scf.if %12 {
// CHECK-NEXT:            linalg.fill ins(%cst : f32) outs(%alloc_2 : memref<64xf32>)
// CHECK-NEXT:          }
// CHECK-NEXT:          %subview = memref.subview %reinterpret_cast_3[0] [%5] [1] : memref<64xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
// CHECK-NEXT:          %subview_5 = memref.subview %alloc_2[0] [%5] [1] : memref<64xf32> to memref<?xf32, strided<[1]>>
// CHECK-NEXT:          memref.copy %subview, %subview_5 : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
// CHECK-NEXT:          %13 = bufferization.to_tensor %alloc_2 restrict writable : memref<64xf32> to tensor<64xf32>
// CHECK-NEXT:          scf.yield %13 : tensor<64xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        %11 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%9, %10 : tensor<64xf32>, tensor<64xf32>) outs(%9 : tensor<64xf32>) {
// CHECK-NEXT:        ^bb0(%in: f32, %in_5: f32, %out: f32):
// CHECK-NEXT:          %12 = arith.addf %in, %in_5 : f32
// CHECK-NEXT:          linalg.yield %12 : f32
// CHECK-NEXT:        } -> tensor<64xf32>
// CHECK-NEXT:        bufferization.materialize_in_destination %11 in writable %reinterpret_cast_4 : (tensor<64xf32>, memref<64xf32, strided<[1]>>) -> ()
// CHECK-NEXT:      } {tts.full_tile_versioned}
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// Not versioned, in full: no arith.andi, no tts.full_tile_versioned,
// and one guard per access yielding the tile.

// NOVER:       #map = affine_map<(d0) -> (d0)>
// NOVER-NEXT:  module {
// NOVER-NEXT:    func.func @masked_load_versioned(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
// NOVER-NEXT:      %c64 = arith.constant 64 : index
// NOVER-NEXT:      %c0 = arith.constant 0 : index
// NOVER-NEXT:      %cst = arith.constant 0.000000e+00 : f32
// NOVER-NEXT:      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [64], strides: [1] : memref<*xf32> to memref<64xf32, strided<[1]>>
// NOVER-NEXT:      %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [64], strides: [1] : memref<*xf32> to memref<64xf32, strided<[1]>>
// NOVER-NEXT:      %0 = arith.index_cast %arg2 : i32 to index
// NOVER-NEXT:      %1 = arith.minsi %0, %c64 : index
// NOVER-NEXT:      %2 = arith.maxsi %1, %c0 : index
// NOVER-NEXT:      %alloc = memref.alloc() : memref<64xf32>
// NOVER-NEXT:      %3 = arith.cmpi eq, %2, %c64 : index
// NOVER-NEXT:      %4 = scf.if %3 -> (tensor<64xf32>) {
// NOVER-NEXT:        memref.copy %reinterpret_cast, %alloc : memref<64xf32, strided<[1]>> to memref<64xf32>
// NOVER-NEXT:        %5 = bufferization.to_tensor %alloc restrict writable : memref<64xf32> to tensor<64xf32>
// NOVER-NEXT:        scf.yield %5 : tensor<64xf32>
// NOVER-NEXT:      } else {
// NOVER-NEXT:        %5 = arith.cmpi slt, %2, %c64 : index
// NOVER-NEXT:        scf.if %5 {
// NOVER-NEXT:          linalg.fill ins(%cst : f32) outs(%alloc : memref<64xf32>)
// NOVER-NEXT:        }
// NOVER-NEXT:        %subview = memref.subview %reinterpret_cast[0] [%2] [1] : memref<64xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
// NOVER-NEXT:        %subview_1 = memref.subview %alloc[0] [%2] [1] : memref<64xf32> to memref<?xf32, strided<[1]>>
// NOVER-NEXT:        memref.copy %subview, %subview_1 : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
// NOVER-NEXT:        %6 = bufferization.to_tensor %alloc restrict writable : memref<64xf32> to tensor<64xf32>
// NOVER-NEXT:        scf.yield %6 : tensor<64xf32>
// NOVER-NEXT:      }
// NOVER-NEXT:      bufferization.materialize_in_destination %4 in writable %reinterpret_cast_0 : (tensor<64xf32>, memref<64xf32, strided<[1]>>) -> ()
// NOVER-NEXT:      return
// NOVER-NEXT:    }
// NOVER-NEXT:    func.func @two_masked_loads_versioned(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
// NOVER-NEXT:      %c64 = arith.constant 64 : index
// NOVER-NEXT:      %c0 = arith.constant 0 : index
// NOVER-NEXT:      %cst = arith.constant 0.000000e+00 : f32
// NOVER-NEXT:      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [64], strides: [1] : memref<*xf32> to memref<64xf32, strided<[1]>>
// NOVER-NEXT:      %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [64], strides: [1] : memref<*xf32> to memref<64xf32, strided<[1]>>
// NOVER-NEXT:      %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [64], strides: [1] : memref<*xf32> to memref<64xf32, strided<[1]>>
// NOVER-NEXT:      %0 = arith.index_cast %arg3 : i32 to index
// NOVER-NEXT:      %1 = arith.minsi %0, %c64 : index
// NOVER-NEXT:      %2 = arith.maxsi %1, %c0 : index
// NOVER-NEXT:      %alloc = memref.alloc() : memref<64xf32>
// NOVER-NEXT:      %3 = arith.cmpi eq, %2, %c64 : index
// NOVER-NEXT:      %4 = scf.if %3 -> (tensor<64xf32>) {
// NOVER-NEXT:        memref.copy %reinterpret_cast, %alloc : memref<64xf32, strided<[1]>> to memref<64xf32>
// NOVER-NEXT:        %11 = bufferization.to_tensor %alloc restrict writable : memref<64xf32> to tensor<64xf32>
// NOVER-NEXT:        scf.yield %11 : tensor<64xf32>
// NOVER-NEXT:      } else {
// NOVER-NEXT:        %11 = arith.cmpi slt, %2, %c64 : index
// NOVER-NEXT:        scf.if %11 {
// NOVER-NEXT:          linalg.fill ins(%cst : f32) outs(%alloc : memref<64xf32>)
// NOVER-NEXT:        }
// NOVER-NEXT:        %subview = memref.subview %reinterpret_cast[0] [%2] [1] : memref<64xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
// NOVER-NEXT:        %subview_3 = memref.subview %alloc[0] [%2] [1] : memref<64xf32> to memref<?xf32, strided<[1]>>
// NOVER-NEXT:        memref.copy %subview, %subview_3 : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
// NOVER-NEXT:        %12 = bufferization.to_tensor %alloc restrict writable : memref<64xf32> to tensor<64xf32>
// NOVER-NEXT:        scf.yield %12 : tensor<64xf32>
// NOVER-NEXT:      }
// NOVER-NEXT:      %5 = arith.index_cast %arg4 : i32 to index
// NOVER-NEXT:      %6 = arith.minsi %5, %c64 : index
// NOVER-NEXT:      %7 = arith.maxsi %6, %c0 : index
// NOVER-NEXT:      %alloc_2 = memref.alloc() : memref<64xf32>
// NOVER-NEXT:      %8 = arith.cmpi eq, %7, %c64 : index
// NOVER-NEXT:      %9 = scf.if %8 -> (tensor<64xf32>) {
// NOVER-NEXT:        memref.copy %reinterpret_cast_0, %alloc_2 : memref<64xf32, strided<[1]>> to memref<64xf32>
// NOVER-NEXT:        %11 = bufferization.to_tensor %alloc_2 restrict writable : memref<64xf32> to tensor<64xf32>
// NOVER-NEXT:        scf.yield %11 : tensor<64xf32>
// NOVER-NEXT:      } else {
// NOVER-NEXT:        %11 = arith.cmpi slt, %7, %c64 : index
// NOVER-NEXT:        scf.if %11 {
// NOVER-NEXT:          linalg.fill ins(%cst : f32) outs(%alloc_2 : memref<64xf32>)
// NOVER-NEXT:        }
// NOVER-NEXT:        %subview = memref.subview %reinterpret_cast_0[0] [%7] [1] : memref<64xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
// NOVER-NEXT:        %subview_3 = memref.subview %alloc_2[0] [%7] [1] : memref<64xf32> to memref<?xf32, strided<[1]>>
// NOVER-NEXT:        memref.copy %subview, %subview_3 : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
// NOVER-NEXT:        %12 = bufferization.to_tensor %alloc_2 restrict writable : memref<64xf32> to tensor<64xf32>
// NOVER-NEXT:        scf.yield %12 : tensor<64xf32>
// NOVER-NEXT:      }
// NOVER-NEXT:      %10 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%4, %9 : tensor<64xf32>, tensor<64xf32>) outs(%4 : tensor<64xf32>) {
// NOVER-NEXT:      ^bb0(%in: f32, %in_3: f32, %out: f32):
// NOVER-NEXT:        %11 = arith.addf %in, %in_3 : f32
// NOVER-NEXT:        linalg.yield %11 : f32
// NOVER-NEXT:      } -> tensor<64xf32>
// NOVER-NEXT:      bufferization.materialize_in_destination %10 in writable %reinterpret_cast_1 : (tensor<64xf32>, memref<64xf32, strided<[1]>>) -> ()
// NOVER-NEXT:      return
// NOVER-NEXT:    }
// NOVER-NEXT:  }
