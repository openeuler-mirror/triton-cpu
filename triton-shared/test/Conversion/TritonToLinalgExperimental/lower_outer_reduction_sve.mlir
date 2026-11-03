// RUN: triton-shared-opt %s --lower-outer-reduction-sve="vscale=1" | FileCheck %s

// A 128-bit SVE vector contains four f32 lanes. The transpose chunks produced
// while lowering a tiled outer reduction should retain vector semantics.

// CHECK-LABEL: func.func @deinterleave_chunk
// CHECK-NOT: vector.transpose
// CHECK: vector.shape_cast {{.*}} : vector<4x2xf32> to vector<8xf32>
// CHECK: %[[EVEN:.*]], %[[ODD:.*]] = vector.deinterleave
// CHECK: vector.shuffle %[[EVEN]], %[[ODD]]
func.func @deinterleave_chunk(%arg: vector<4x2xf32>) -> vector<2x4xf32> {
  %0 = vector.transpose %arg, [1, 0] : vector<4x2xf32> to vector<2x4xf32>
  return %0 : vector<2x4xf32>
}

// CHECK-LABEL: func.func @interleave_chunk
// CHECK-NOT: vector.transpose
// CHECK: %[[ROW0:.*]] = vector.extract {{.*}}[0]
// CHECK: %[[ROW1:.*]] = vector.extract {{.*}}[1]
// CHECK: vector.interleave %[[ROW0]], %[[ROW1]]
func.func @interleave_chunk(%arg: vector<2x4xf32>) -> vector<4x2xf32> {
  %0 = vector.transpose %arg, [1, 0] : vector<2x4xf32> to vector<4x2xf32>
  return %0 : vector<4x2xf32>
}

// Transposes below the large-reduction tiling threshold are split into native
// SVE chunks as well.

// CHECK-LABEL: func.func @small_transpose
// CHECK-NOT: vector.transpose
// CHECK-COUNT-2: vector.deinterleave
func.func @small_transpose(%arg: vector<8x2xf32>) -> vector<2x8xf32> {
  %0 = vector.transpose %arg, [1, 0] : vector<8x2xf32> to vector<2x8xf32>
  return %0 : vector<2x8xf32>
}

// CHECK-LABEL: func.func @small_transpose_reverse
// CHECK-NOT: vector.transpose
// CHECK-COUNT-2: vector.interleave
func.func @small_transpose_reverse(%arg: vector<2x8xf32>) -> vector<8x2xf32> {
  %0 = vector.transpose %arg, [1, 0] : vector<2x8xf32> to vector<8x2xf32>
  return %0 : vector<8x2xf32>
}

// The reduction lowering is likewise independent of the 64-element tiling.

// CHECK-LABEL: func.func @small_outer_reduction
// CHECK-NOT: vector.multi_reduction
// CHECK-COUNT-4: vector.shuffle
// CHECK-COUNT-2: vector.reduction
func.func @small_outer_reduction(%arg: vector<8x2xf32>, %acc: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.multi_reduction <add>, %arg, %acc [0] : vector<8x2xf32> to vector<2xf32>
  return %0 : vector<2xf32>
}

// Transposes without a two-element dimension remain outside this targeted
// lowering.

// CHECK-LABEL: func.func @unrelated_transpose
// CHECK: vector.transpose
func.func @unrelated_transpose(%arg: vector<8x3xf32>) -> vector<3x8xf32> {
  %0 = vector.transpose %arg, [1, 0] : vector<8x3xf32> to vector<3x8xf32>
  return %0 : vector<3x8xf32>
}
