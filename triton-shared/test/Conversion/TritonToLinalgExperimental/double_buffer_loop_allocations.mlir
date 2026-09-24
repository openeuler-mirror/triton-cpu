// RUN: triton-shared-opt %s --split-input-file | FileCheck %s --check-prefix=BEFORE
// RUN: triton-shared-opt %s --double-buffer-loop-allocations --split-input-file | FileCheck %s

// Reduced instance_norm statistics loop: update the per-lane mean (m), sum of
// squared deviations (s), and count (cnt) for each 64-element input tile.
// Before: allocate three new state buffers on every iteration.
// After: allocate each state buffer once outside the loop. Each old state is
// read before its buffer is overwritten; yield the scratch buffers for reuse.
//
// BEFORE-LABEL: func.func @instance_norm_stats(
// BEFORE-NOT: memref.alloc
// BEFORE: scf.for
// BEFORE: %[[M:.*]] = memref.alloc() {{.*}} : memref<64xf32>
// BEFORE: vector.transfer_write {{.*}}, %[[M]]
// BEFORE: %[[S:.*]] = memref.alloc() {{.*}} : memref<64xf32>
// BEFORE: vector.transfer_write {{.*}}, %[[S]]
// BEFORE: %[[CNT:.*]] = memref.alloc() {{.*}} : memref<64xi32>
// BEFORE: vector.transfer_write {{.*}}, %[[CNT]]
// BEFORE-NEXT: scf.yield %[[CNT]], %[[M]], %[[S]]
//
// CHECK-LABEL: func.func @instance_norm_stats(
// CHECK: %[[CNT_BUF:.*]] = memref.alloc() {alignment = 64 : i64} : memref<64xi32>
// CHECK-NEXT: %[[M_BUF:.*]] = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
// CHECK-NEXT: %[[S_BUF:.*]] = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
// CHECK-NEXT: {{.*}}scf.for
// CHECK-SAME: iter_args(%[[CNT:.*]] = {{.*}}, %[[M:.*]] = {{.*}}, %[[S:.*]] = {{.*}})
// CHECK-NOT: memref.alloc
// CHECK: %[[OLD_M:.*]] = vector.transfer_read %[[M]]
// CHECK-NOT: memref.alloc
// CHECK: %[[NEW_M:.*]] = arith.addf %[[OLD_M]],
// CHECK-NEXT: vector.transfer_write %[[NEW_M]], %[[M_BUF]]
// CHECK-NEXT: %[[OLD_S:.*]] = vector.transfer_read %[[S]]
// CHECK-NOT: memref.alloc
// CHECK: %[[NEW_S:.*]] = arith.addf %[[OLD_S]],
// CHECK-NEXT: vector.transfer_write %[[NEW_S]], %[[S_BUF]]
// CHECK-NEXT: %[[OLD_CNT:.*]] = vector.transfer_read %[[CNT]]
// CHECK-NEXT: %[[NEW_CNT:.*]] = arith.addi %[[OLD_CNT]],
// CHECK-NEXT: vector.transfer_write %[[NEW_CNT]], %[[CNT_BUF]]
// CHECK-NEXT: scf.yield %[[CNT_BUF]], %[[M_BUF]], %[[S_BUF]]
// CHECK-NEXT: }
func.func @instance_norm_stats(%input: memref<?xf32>, %init_cnt: memref<64xi32>,
    %init_m: memref<64xf32>, %init_s: memref<64xf32>, %steps: i32)
    -> (memref<64xi32>, memref<64xf32>, memref<64xf32>) {
  %c0 = arith.constant 0 : index
  %zero_i32 = arith.constant 0 : i32
  %one_i32 = arith.constant 1 : i32
  %tile = arith.constant 64 : i32
  %zero = arith.constant 0.0 : f32
  %ones = arith.constant dense<1> : vector<64xi32>
  %result:3 = scf.for %i = %zero_i32 to %steps step %one_i32
      iter_args(%cnt = %init_cnt, %m = %init_m, %s = %init_s)
      -> (memref<64xi32>, memref<64xf32>, memref<64xf32>) : i32 {
    %offset_i32 = arith.muli %i, %tile : i32
    %offset = arith.index_cast %offset_i32 : i32 to index
    %x = vector.transfer_read %input[%offset], %zero {in_bounds = [true]} : memref<?xf32>, vector<64xf32>
    %old_m = vector.transfer_read %m[%c0], %zero {in_bounds = [true]} : memref<64xf32>, vector<64xf32>
    %diff = arith.subf %x, %old_m : vector<64xf32>
    %count_i32 = arith.addi %i, %one_i32 : i32
    %count_f32 = arith.sitofp %count_i32 : i32 to f32
    %count = vector.broadcast %count_f32 : f32 to vector<64xf32>
    %delta = arith.divf %diff, %count : vector<64xf32>
    %new_m = arith.addf %old_m, %delta : vector<64xf32>
    %next_m = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
    vector.transfer_write %new_m, %next_m[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32>
    %old_s = vector.transfer_read %s[%c0], %zero {in_bounds = [true]} : memref<64xf32>, vector<64xf32>
    %new_diff = arith.subf %x, %new_m : vector<64xf32>
    %increment = arith.mulf %new_diff, %diff : vector<64xf32>
    %new_s = arith.addf %old_s, %increment : vector<64xf32>
    %next_s = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
    vector.transfer_write %new_s, %next_s[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32>
    %old_cnt = vector.transfer_read %cnt[%c0], %zero_i32 {in_bounds = [true]} : memref<64xi32>, vector<64xi32>
    %new_cnt = arith.addi %old_cnt, %ones : vector<64xi32>
    %next_cnt = memref.alloc() {alignment = 64 : i64} : memref<64xi32>
    vector.transfer_write %new_cnt, %next_cnt[%c0] {in_bounds = [true]} : vector<64xi32>, memref<64xi32>
    scf.yield %next_cnt, %next_m, %next_s : memref<64xi32>, memref<64xf32>, memref<64xf32>
  }
  return %result#0, %result#1, %result#2 : memref<64xi32>, memref<64xf32>, memref<64xf32>
}

// -----

// A copy needs distinct source and destination buffers. Nonzero lower bound
// and non-unit step check that parity is based on iteration number, not IV.
// CHECK-LABEL: func.func @double_buffer(
// CHECK-SAME: %[[INIT:.*]]: memref<4xf32>, %[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index
// CHECK: %[[A:.*]] = memref.alloc()
// CHECK-NEXT: %[[B:.*]] = memref.alloc()
// CHECK-NEXT: {{.*}}scf.for %[[IV:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%[[OLD:.*]] = %[[INIT]])
// CHECK-NEXT: %[[DIST:.*]] = arith.subi %[[IV]], %[[LB]]
// CHECK-NEXT: %[[ITER:.*]] = arith.divui %[[DIST]], %[[STEP]]
// CHECK-NEXT: %[[TWO:.*]] = arith.constant 2 : index
// CHECK-NEXT: %[[PAR:.*]] = arith.remui %[[ITER]], %[[TWO]]
// CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[EVEN:.*]] = arith.cmpi eq, %[[PAR]], %[[ZERO]]
// CHECK-NEXT: %[[SEL:.*]] = arith.select %[[EVEN]], %[[A]], %[[B]]
// CHECK-NEXT: memref.copy %[[OLD]], %[[SEL]]
// CHECK-NEXT: scf.yield %[[SEL]]
// CHECK-NEXT: }
func.func @double_buffer(%init: memref<4xf32>, %lb: index, %ub: index, %step: index) -> memref<4xf32> {
  %r = scf.for %i = %lb to %ub step %step iter_args(%old = %init) -> memref<4xf32> {
    %next = memref.alloc() : memref<4xf32>
    memref.copy %old, %next : memref<4xf32> to memref<4xf32>
    scf.yield %next : memref<4xf32>
  }
  return %r : memref<4xf32>
}

// -----

// Both arms yield fresh allocations. Each arm gets its own pair of buffers.
// CHECK-LABEL: func.func @conditional_buffers(
// CHECK: memref.alloc()
// CHECK-NEXT: memref.alloc()
// CHECK-NEXT: memref.alloc()
// CHECK-NEXT: memref.alloc()
// CHECK-NEXT: {{.*}}scf.for
// CHECK-NEXT: {{.*}}scf.if
// CHECK-NOT: memref.alloc
// CHECK: %[[THEN:.*]] = arith.select
// CHECK-NEXT: memref.copy {{.*}}, %[[THEN]]
// CHECK-NEXT: scf.yield %[[THEN]]
// CHECK-NEXT: } else {
// CHECK-NOT: memref.alloc
// CHECK: %[[ELSE:.*]] = arith.select
// CHECK-NEXT: memref.copy {{.*}}, %[[ELSE]]
// CHECK-NEXT: scf.yield %[[ELSE]]
// CHECK-NEXT: }
func.func @conditional_buffers(%init: memref<4xf32>, %n: index, %cond: i1) -> memref<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%old = %init) -> memref<4xf32> {
    %next = scf.if %cond -> memref<4xf32> {
      %a = memref.alloc() : memref<4xf32>
      memref.copy %old, %a : memref<4xf32> to memref<4xf32>
      scf.yield %a : memref<4xf32>
    } else {
      %b = memref.alloc() : memref<4xf32>
      memref.copy %old, %b : memref<4xf32> to memref<4xf32>
      scf.yield %b : memref<4xf32>
    }
    scf.yield %next : memref<4xf32>
  }
  return %r : memref<4xf32>
}

// -----

// An old-state read after a new-state write must keep two distinct buffers.
// CHECK-LABEL: func.func @read_after_write(
// CHECK: %[[A:.*]] = memref.alloc()
// CHECK-NEXT: %[[B:.*]] = memref.alloc()
// CHECK-NEXT: {{.*}}scf.for
// CHECK-NOT: memref.alloc
// CHECK: %[[SEL:.*]] = arith.select {{.*}}, %[[A]], %[[B]]
// CHECK-NEXT: memref.store {{.*}}, %[[SEL]]
// CHECK-NEXT: {{.*}}memref.load
// CHECK-NEXT: memref.store {{.*}}, %[[SEL]]
// CHECK-NEXT: scf.yield %[[SEL]]
func.func @read_after_write(%init: memref<1xi32>, %n: index) -> memref<1xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%old = %init) -> memref<1xi32> {
    %next = memref.alloc() : memref<1xi32>
    memref.store %zero, %next[%c0] : memref<1xi32>
    %v = memref.load %old[%c0] : memref<1xi32>
    memref.store %v, %next[%c0] : memref<1xi32>
    scf.yield %next : memref<1xi32>
  }
  return %r : memref<1xi32>
}
