// RUN: triton-shared-opt %s --double-buffer-loop-allocations --split-input-file | FileCheck %s

// Keep allocations in the loop when reuse is unsupported or unsafe.

// Forwarding old state can extend its lifetime beyond two iterations.
// CHECK-LABEL: func.func @conditional_forwarding(
// CHECK-NOT: memref.alloc
// CHECK: scf.for
// CHECK-NEXT: {{.*}}scf.if
// CHECK-NEXT: %[[NEW:.*]] = memref.alloc()
// CHECK-NEXT: memref.copy {{.*}}, %[[NEW]]
// CHECK-NEXT: scf.yield %[[NEW]]
// CHECK-NEXT: } else {
// CHECK-NEXT: scf.yield
func.func @conditional_forwarding(%init: memref<4xf32>, %n: index, %cond: i1) -> memref<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%old = %init) -> memref<4xf32> {
    %next = scf.if %cond -> memref<4xf32> {
      %a = memref.alloc() : memref<4xf32>
      memref.copy %old, %a : memref<4xf32> to memref<4xf32>
      scf.yield %a : memref<4xf32>
    } else {
      scf.yield %old : memref<4xf32>
    }
    scf.yield %next : memref<4xf32>
  }
  return %r : memref<4xf32>
}

// -----

// An extra iter_arg retains the previous allocation across iterations.
// CHECK-LABEL: func.func @retained_alias(
// CHECK-NOT: memref.alloc
// CHECK: scf.for
// CHECK-NEXT: %[[NEW:.*]] = memref.alloc()
// CHECK-NEXT: memref.copy
// CHECK-NEXT: scf.yield %[[NEW]],
func.func @retained_alias(%init: memref<4xf32>, %n: index) -> (memref<4xf32>, memref<4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%old = %init, %saved = %init) -> (memref<4xf32>, memref<4xf32>) {
    %next = memref.alloc() : memref<4xf32>
    memref.copy %old, %next : memref<4xf32> to memref<4xf32>
    scf.yield %next, %old : memref<4xf32>, memref<4xf32>
  }
  return %r#0, %r#1 : memref<4xf32>, memref<4xf32>
}

// -----

// CHECK-LABEL: func.func @dynamic_shape(
// CHECK-NOT: memref.alloc
// CHECK: scf.for
// CHECK-NEXT: %[[NEW:.*]] = memref.alloc
// CHECK-NEXT: memref.copy {{.*}}, %[[NEW]]
// CHECK-NEXT: scf.yield %[[NEW]]
func.func @dynamic_shape(%init: memref<?xf32>, %n: index, %size: index) -> memref<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%old = %init) -> memref<?xf32> {
    %next = memref.alloc(%size) : memref<?xf32>
    memref.copy %old, %next : memref<?xf32> to memref<?xf32>
    scf.yield %next : memref<?xf32>
  }
  return %r : memref<?xf32>
}

// -----

// CHECK-LABEL: func.func @non_identity_layout(
// CHECK-NOT: memref.alloc
// CHECK: scf.for
// CHECK-NEXT: %[[NEW:.*]] = memref.alloc
// CHECK-NEXT: memref.copy {{.*}}, %[[NEW]]
// CHECK-NEXT: scf.yield %[[NEW]]
func.func @non_identity_layout(%init: memref<4xf32, strided<[2]>>, %n: index) -> memref<4xf32, strided<[2]>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%old = %init) -> memref<4xf32, strided<[2]>> {
    %next = memref.alloc() : memref<4xf32, strided<[2]>>
    memref.copy %old, %next : memref<4xf32, strided<[2]>> to memref<4xf32, strided<[2]>>
    scf.yield %next : memref<4xf32, strided<[2]>>
  }
  return %r : memref<4xf32, strided<[2]>>
}

// -----

// CHECK-LABEL: func.func @over_size_limit(
// CHECK-NOT: memref.alloc
// CHECK: scf.for
// CHECK-NEXT: %[[NEW:.*]] = memref.alloc
// CHECK-NEXT: memref.copy {{.*}}, %[[NEW]]
// CHECK-NEXT: scf.yield %[[NEW]]
func.func @over_size_limit(%init: memref<4097xf32>, %n: index) -> memref<4097xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%old = %init) -> memref<4097xf32> {
    %next = memref.alloc() : memref<4097xf32>
    memref.copy %old, %next : memref<4097xf32> to memref<4097xf32>
    scf.yield %next : memref<4097xf32>
  }
  return %r : memref<4097xf32>
}

// -----

func.func private @escape(memref<4xf32>)

// CHECK-LABEL: func.func @escaping_buffer(
// CHECK-NOT: memref.alloc
// CHECK: scf.for
// CHECK-NEXT: %[[NEW:.*]] = memref.alloc
// CHECK-NEXT: memref.copy {{.*}}, %[[NEW]]
// CHECK-NEXT: func.call @escape(%[[NEW]])
// CHECK-NEXT: scf.yield %[[NEW]]
func.func @escaping_buffer(%init: memref<4xf32>, %n: index) -> memref<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%old = %init) -> memref<4xf32> {
    %next = memref.alloc() : memref<4xf32>
    memref.copy %old, %next : memref<4xf32> to memref<4xf32>
    func.call @escape(%next) : (memref<4xf32>) -> ()
    scf.yield %next : memref<4xf32>
  }
  return %r : memref<4xf32>
}
