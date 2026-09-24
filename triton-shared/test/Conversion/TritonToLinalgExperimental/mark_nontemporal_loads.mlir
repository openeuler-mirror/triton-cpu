// RUN: triton-shared-opt --split-input-file --mark-nontemporal-loads %s | FileCheck %s
// RUN: triton-shared-opt --split-input-file --mark-nontemporal-loads --verify-diagnostics %s

module {
  llvm.func @descriptor_chain(%arg0: i64, %arg1: !llvm.ptr, %arg2: i64, %arg3: !llvm.ptr, %arg4: i64)
      attributes {tts.nontemporal_load_args = array<i32: 1>, tts.ptr_arg_count = 2 : i32} {
    %d0 = llvm.getelementptr %arg1[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %p0 = llvm.load %d0 : !llvm.ptr -> !llvm.ptr
    %e0 = llvm.getelementptr %p0[%arg4] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %v0 = llvm.load %e0 {alignment = 4 : i64} : !llvm.ptr -> vector<16xf32>
    %d1 = llvm.getelementptr %arg3[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %p1 = llvm.load %d1 : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.getelementptr %p1[%arg4] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %e2 = llvm.getelementptr %e1[112] : (!llvm.ptr) -> !llvm.ptr, f32
    %v1 = llvm.load %e2 {alignment = 4 : i64} : !llvm.ptr -> vector<16xf32>
    llvm.return
  }
}
// CHECK-LABEL: llvm.func @descriptor_chain
// CHECK-NOT:   tts.
// CHECK:       %[[P0:.*]] = llvm.load %{{[0-9a-z]+}} : !llvm.ptr -> !llvm.ptr
// CHECK:       llvm.load %{{.*}} {alignment = 4 : i64} : !llvm.ptr -> vector<16xf32>
// CHECK:       %[[P1:.*]] = llvm.load %{{[0-9a-z]+}} : !llvm.ptr -> !llvm.ptr
// CHECK:       llvm.load %{{.*}} {alignment = 4 : i64, nontemporal} : !llvm.ptr -> vector<16xf32>
// CHECK-NOT:   tts.

// -----

module {
  llvm.func @phi_and_select(%arg0: i64, %arg1: !llvm.ptr, %arg2: i64, %arg3: !llvm.ptr, %c: i1)
      attributes {tts.nontemporal_load_args = array<i32: 1>, tts.ptr_arg_count = 2 : i32} {
    %p0 = llvm.load %arg1 : !llvm.ptr -> !llvm.ptr
    %p1 = llvm.load %arg3 : !llvm.ptr -> !llvm.ptr
    %q1 = llvm.getelementptr %p1[8] : (!llvm.ptr) -> !llvm.ptr, f32
    llvm.cond_br %c, ^bb1(%p1 : !llvm.ptr), ^bb1(%q1 : !llvm.ptr)
  ^bb1(%a: !llvm.ptr):
    %va = llvm.load %a : !llvm.ptr -> f32
    %s_mixed = llvm.select %c, %p0, %p1 : i1, !llvm.ptr
    %vm = llvm.load %s_mixed : !llvm.ptr -> f32
    %s_same = llvm.select %c, %p1, %q1 : i1, !llvm.ptr
    %vs = llvm.load %s_same : !llvm.ptr -> f32
    llvm.br ^bb2(%p0 : !llvm.ptr)
  ^bb2(%b: !llvm.ptr):
    %vb = llvm.load %b : !llvm.ptr -> f32
    llvm.return
  }
}
// CHECK-LABEL: llvm.func @phi_and_select
// CHECK:       ^bb1(%[[A:.*]]: !llvm.ptr):
// CHECK-NEXT:    llvm.load %[[A]] {nontemporal} : !llvm.ptr -> f32
// CHECK:         %[[SM:.*]] = llvm.select
// CHECK-NEXT:    llvm.load %[[SM]] : !llvm.ptr -> f32
// CHECK:         %[[SS:.*]] = llvm.select
// CHECK-NEXT:    llvm.load %[[SS]] {nontemporal} : !llvm.ptr -> f32
// CHECK:       ^bb2(%[[B:.*]]: !llvm.ptr):
// CHECK-NEXT:    llvm.load %[[B]] : !llvm.ptr -> f32

// -----

module {
  // expected-warning @below {{nontemporal argument annotations ignored: expected 2 pointer parameters, found 1}}
  llvm.func @count_mismatch(%arg0: i64, %arg1: !llvm.ptr, %arg2: i64)
      attributes {tts.nontemporal_load_args = array<i32: 0>, tts.ptr_arg_count = 2 : i32} {
    %p = llvm.load %arg1 : !llvm.ptr -> !llvm.ptr
    %v = llvm.load %p : !llvm.ptr -> f32
    llvm.return
  }
}
// CHECK-LABEL: llvm.func @count_mismatch
// CHECK-NOT:   tts.
// CHECK-NOT:   nontemporal

// -----

// No attribute, nothing happens.
module {
  llvm.func @untouched(%arg0: i64, %arg1: !llvm.ptr) {
    %p = llvm.load %arg1 : !llvm.ptr -> !llvm.ptr
    %v = llvm.load %p : !llvm.ptr -> f32
    llvm.return
  }
}
// CHECK-LABEL: llvm.func @untouched
// CHECK-NOT:   nontemporal

// -----

module {
  llvm.func @stores(%arg0: i64, %arg1: !llvm.ptr, %arg2: i64, %arg3: !llvm.ptr, %v: vector<16xf32>)
      attributes {tts.nontemporal_load_args = array<i32: 0>, tts.nontemporal_store_args = array<i32: 1>, tts.ptr_arg_count = 2 : i32} {
    %p0 = llvm.load %arg1 : !llvm.ptr -> !llvm.ptr
    %p1 = llvm.load %arg3 : !llvm.ptr -> !llvm.ptr
    %l0 = llvm.load %p0 {alignment = 4 : i64} : !llvm.ptr -> vector<16xf32>
    llvm.store %v, %p0 {alignment = 4 : i64} : vector<16xf32>, !llvm.ptr
    %l1 = llvm.load %p1 {alignment = 4 : i64} : !llvm.ptr -> vector<16xf32>
    %e1 = llvm.getelementptr %p1[16] : (!llvm.ptr) -> !llvm.ptr, f32
    llvm.store %v, %e1 {alignment = 4 : i64} : vector<16xf32>, !llvm.ptr
    llvm.return
  }
}
// CHECK-LABEL: llvm.func @stores
// CHECK-NOT:   tts.
// CHECK:       llvm.load %{{.*}} {alignment = 4 : i64, nontemporal} : !llvm.ptr -> vector<16xf32>
// CHECK-NEXT:  llvm.store %{{.*}}, %{{.*}} {alignment = 4 : i64} : vector<16xf32>, !llvm.ptr
// CHECK-NEXT:  llvm.load %{{.*}} {alignment = 4 : i64} : !llvm.ptr -> vector<16xf32>
// CHECK:       llvm.store %{{.*}}, %{{.*}} {alignment = 4 : i64, nontemporal} : vector<16xf32>, !llvm.ptr
