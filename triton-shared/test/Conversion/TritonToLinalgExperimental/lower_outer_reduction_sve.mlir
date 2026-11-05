// RUN: triton-shared-opt %s --lower-outer-reduction-sve="vscale=1" | FileCheck %s
// RUN: triton-shared-opt %s --lower-outer-reduction-sve="vscale=4" | FileCheck %s --check-prefix=VSCALE4

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

// A wide second dimension must not be flattened with an SVE chunk. At
// vscale=4 this would otherwise create vector<1024xf32> shuffle operands.

// CHECK-LABEL: func.func @wide_outer_reduction
// CHECK: vector.multi_reduction
// VSCALE4-LABEL: func.func @wide_outer_reduction
// VSCALE4-NOT: vector<1024xf32>
// VSCALE4: vector.multi_reduction
func.func @wide_outer_reduction(%arg: vector<64x64xf32>, %acc: vector<64xf32>) -> vector<64xf32> {
  %0 = vector.multi_reduction <add>, %arg, %acc [0] : vector<64x64xf32> to vector<64xf32>
  return %0 : vector<64xf32>
}

// The fused transpose/reduction path has the same restriction.

// CHECK-LABEL: func.func @wide_transpose_reduction
// CHECK: vector.transpose
// CHECK: vector.reduction
// VSCALE4-LABEL: func.func @wide_transpose_reduction
// VSCALE4-NOT: vector<1024xf32>
// VSCALE4: vector.transpose
// VSCALE4: vector.reduction
func.func @wide_transpose_reduction(%arg: vector<64x64xf32>, %acc: f32) -> f32 {
  %0 = vector.transpose %arg, [1, 0] : vector<64x64xf32> to vector<64x64xf32>
  %1 = vector.extract %0[0] : vector<64xf32> from vector<64x64xf32>
  %2 = vector.reduction <add>, %1, %acc : vector<64xf32> into f32
  return %2 : f32
}


// A wide reduction keeps every reduction in its function on the standard
// lowering path, including otherwise-supported two-column reductions.

// CHECK-LABEL: func.func @wide_reduction_bypasses_function
// CHECK-COUNT-2: vector.multi_reduction
// VSCALE4-LABEL: func.func @wide_reduction_bypasses_function
// VSCALE4-COUNT-2: vector.multi_reduction
func.func @wide_reduction_bypasses_function(
    %wide: vector<64x64xf32>, %wide_acc: vector<64xf32>,
    %narrow: vector<64x2xf32>, %narrow_acc: vector<2xf32>)
    -> (vector<64xf32>, vector<2xf32>) {
  %0 = vector.multi_reduction <add>, %wide, %wide_acc [0] : vector<64x64xf32> to vector<64xf32>
  %1 = vector.multi_reduction <add>, %narrow, %narrow_acc [0] : vector<64x2xf32> to vector<2xf32>
  return %0, %1 : vector<64xf32>, vector<2xf32>
}
// Transposes without a two-element dimension remain outside this targeted
// lowering.

// CHECK-LABEL: func.func @unrelated_transpose
// CHECK: vector.transpose
func.func @unrelated_transpose(%arg: vector<8x3xf32>) -> vector<3x8xf32> {
  %0 = vector.transpose %arg, [1, 0] : vector<8x3xf32> to vector<3x8xf32>
  return %0 : vector<3x8xf32>
}
