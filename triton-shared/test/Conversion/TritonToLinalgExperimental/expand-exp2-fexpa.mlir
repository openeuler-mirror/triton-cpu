// RUN: triton-shared-opt --expand-exp2-fexpa=chunk-width=16 %s | FileCheck %s

// CHECK-DAG: llvm.func @llvm.aarch64.sve.fexpa.x.nxv4f32(vector<[4]xi32>) -> vector<[4]xf32>

// A 512-lane f32 vector splits into 32 chunks of 16 lanes; each chunk is
// staged in a vector<[4]xf32> SVE register via vector.scalable.insert, runs
// the 1-ULP FEXPA sequence via a direct call to the upstream
// llvm.aarch64.sve.fexpa.x intrinsic and is read back via
// vector.scalable.extract.

// CHECK-LABEL: func.func @exp2_v512f32
// CHECK-DAG: arith.constant dense<1.967350e+05> : vector<[4]xf32>
// CHECK-DAG: arith.constant dense<1.270000e+02> : vector<[4]xf32>
// CHECK-DAG: arith.constant dense<-1.260000e+02> : vector<[4]xf32>
// CHECK-DAG: arith.constant dense<0.693147182> : vector<[4]xf32>
// CHECK-DAG: llvm.mlir.undef : vector<[4]xf32>
func.func @exp2_v512f32(%arg0: memref<512xf32>) {
  %c0 = arith.constant 0 : index
  %0 = vector.load %arg0[%c0] : memref<512xf32>, vector<512xf32>
  // CHECK: vector.extract_strided_slice {{.*}} {offsets = [0], sizes = [16], strides = [1]} : vector<512xf32> to vector<16xf32>
  // CHECK: vector.scalable.insert {{.*}} : vector<16xf32> into vector<[4]xf32>
  // CHECK: arith.minimumf
  // CHECK: arith.maximumf
  // CHECK: arith.addf
  // CHECK: arith.bitcast {{.*}} : vector<[4]xf32> to vector<[4]xi32>
  // CHECK: llvm.call @llvm.aarch64.sve.fexpa.x.nxv4f32({{.*}}) : (vector<[4]xi32>) -> vector<[4]xf32>
  // CHECK: math.fma
  // CHECK: vector.scalable.extract {{.*}} : vector<16xf32> from vector<[4]xf32>
  // CHECK: vector.insert_strided_slice {{.*}} : vector<16xf32> into vector<512xf32>
  %1 = math.exp2 %0 : vector<512xf32>
  // CHECK-COUNT-31: llvm.call @llvm.aarch64.sve.fexpa.x.nxv4f32
  // CHECK-NOT: math.exp2
  vector.store %1, %arg0[%c0] : memref<512xf32>, vector<512xf32>
  return
}

// Sizes that are not a multiple of the chunk width are left untouched.

// CHECK-LABEL: func.func @exp2_v20f32
func.func @exp2_v20f32(%arg0: memref<20xf32>) {
  %c0 = arith.constant 0 : index
  %0 = vector.load %arg0[%c0] : memref<20xf32>, vector<20xf32>
  // CHECK: math.exp2 {{.*}} : vector<20xf32>
  // CHECK-NOT: llvm.aarch64.sve.fexpa
  %1 = math.exp2 %0 : vector<20xf32>
  vector.store %1, %arg0[%c0] : memref<20xf32>, vector<20xf32>
  return
}

// Non-f32 element types and scalars are left untouched.

// CHECK-LABEL: func.func @exp2_v64f64
func.func @exp2_v64f64(%arg0: memref<64xf64>) {
  %c0 = arith.constant 0 : index
  %0 = vector.load %arg0[%c0] : memref<64xf64>, vector<64xf64>
  // CHECK: math.exp2 {{.*}} : vector<64xf64>
  // CHECK-NOT: llvm.aarch64.sve.fexpa
  %1 = math.exp2 %0 : vector<64xf64>
  vector.store %1, %arg0[%c0] : memref<64xf64>, vector<64xf64>
  return
}

// CHECK-LABEL: func.func @exp2_scalar
func.func @exp2_scalar(%arg0: f32) -> f32 {
  // CHECK: math.exp2 {{.*}} : f32
  %0 = math.exp2 %arg0 : f32
  return %0 : f32
}
