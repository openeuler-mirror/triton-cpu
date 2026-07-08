// RUN: triton-shared-opt --triton-to-structured="skip-prepass=true" --canonicalize %s | FileCheck %s

module {
  tt.func @fold_div_rem_pointer_offset(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %cst = arith.constant dense<64> : tensor<128xi32>
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %q = arith.divsi %range, %cst : tensor<128xi32>
    %qc = arith.muli %q, %cst : tensor<128xi32>
    %r = arith.remsi %range, %cst : tensor<128xi32>
    %offset = arith.addi %qc, %r : tensor<128xi32>
    %src = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src, %offset : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %value = tt.load %src_ptr : tensor<128x!tt.ptr<f32>>
    %dst = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst, %range : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %dst_ptr, %value : tensor<128x!tt.ptr<f32>>
    tt.return
  }

  tt.func @fold_extended_div_rem_pointer_offset(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %cst = arith.constant dense<64> : tensor<128xi32>
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %q = arith.divsi %range, %cst : tensor<128xi32>
    %qc = arith.muli %q, %cst : tensor<128xi32>
    %r = arith.remsi %range, %cst : tensor<128xi32>
    %qc64 = arith.extsi %qc : tensor<128xi32> to tensor<128xi64>
    %r64 = arith.extsi %r : tensor<128xi32> to tensor<128xi64>
    %offset = arith.addi %qc64, %r64 : tensor<128xi64>
    %src = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src, %offset : tensor<128x!tt.ptr<f32>>, tensor<128xi64>
    %value = tt.load %src_ptr : tensor<128x!tt.ptr<f32>>
    %dst = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst, %offset : tensor<128x!tt.ptr<f32>>, tensor<128xi64>
    tt.store %dst_ptr, %value : tensor<128x!tt.ptr<f32>>
    tt.return
  }

  tt.func @fold_unextended_shared_factor_pointer_offset(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %cst = arith.constant dense<64> : tensor<128xi32>
    %scale = arith.constant dense<4> : tensor<128xi32>
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %q = arith.divsi %range, %cst : tensor<128xi32>
    %qc = arith.muli %q, %cst : tensor<128xi32>
    %qcs = arith.muli %qc, %scale : tensor<128xi32>
    %r = arith.remsi %range, %cst : tensor<128xi32>
    %rs = arith.muli %r, %scale : tensor<128xi32>
    %offset = arith.addi %qcs, %rs : tensor<128xi32>
    %src = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src, %offset : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %value = tt.load %src_ptr : tensor<128x!tt.ptr<f32>>
    %dst = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst, %offset : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %dst_ptr, %value : tensor<128x!tt.ptr<f32>>
    tt.return
  }

  tt.func @do_not_reassociate_plain_integer_arith(%x: i32, %c: i32, %k: i32) -> i64 {
    %q = arith.divsi %x, %c : i32
    %qc = arith.muli %q, %c : i32
    %qck = arith.muli %qc, %k : i32
    %r = arith.remsi %x, %c : i32
    %rk = arith.muli %r, %k : i32
    %qck64 = arith.extsi %qck : i32 to i64
    %rk64 = arith.extsi %rk : i32 to i64
    %sum = arith.addi %qck64, %rk64 : i64
    tt.return %sum : i64
  }

  tt.func @do_not_reassociate_extended_shared_factor_offset(%arg0: !tt.ptr<f32>, %x: i32, %c: i32, %k: i32) -> tensor<1xf32> {
    %xv = tt.splat %x : i32 -> tensor<1xi32>
    %cv = tt.splat %c : i32 -> tensor<1xi32>
    %kv = tt.splat %k : i32 -> tensor<1xi32>
    %q = arith.divsi %xv, %cv : tensor<1xi32>
    %qc = arith.muli %q, %cv : tensor<1xi32>
    %qck = arith.muli %qc, %kv : tensor<1xi32>
    %r = arith.remsi %xv, %cv : tensor<1xi32>
    %rk = arith.muli %r, %kv : tensor<1xi32>
    %qck64 = arith.extsi %qck : tensor<1xi32> to tensor<1xi64>
    %rk64 = arith.extsi %rk : tensor<1xi32> to tensor<1xi64>
    %offset = arith.addi %qck64, %rk64 : tensor<1xi64>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %offset : tensor<1x!tt.ptr<f32>>, tensor<1xi64>
    %value = tt.load %ptr : tensor<1x!tt.ptr<f32>>
    tt.return %value : tensor<1xf32>
  }
}

// CHECK-LABEL: tt.func @fold_div_rem_pointer_offset
// CHECK-SAME: ([[ARG0:%.+]]: !tt.ptr<f32>, [[ARG1:%.+]]: !tt.ptr<f32>)
// CHECK-DAG: [[SRC:%.+]] = tts.make_tptr [[ARG0]] to sizes: [128], strides: [1], offsets: [0]
// CHECK-DAG: [[VALUE:%.+]] = "tts.load"([[SRC]])
// CHECK-DAG: [[DST:%.+]] = tts.make_tptr [[ARG1]] to sizes: [128], strides: [1], offsets: [0]
// CHECK: "tts.store"([[DST]], [[VALUE]])

// CHECK-LABEL: tt.func @fold_extended_div_rem_pointer_offset
// CHECK-SAME: ([[ARG0:%.+]]: !tt.ptr<f32>, [[ARG1:%.+]]: !tt.ptr<f32>)
// CHECK-DAG: [[SRC:%.+]] = tts.make_tptr [[ARG0]] to sizes: [128], strides: [1], offsets: [0]
// CHECK-DAG: [[VALUE:%.+]] = "tts.load"([[SRC]])
// CHECK-DAG: [[DST:%.+]] = tts.make_tptr [[ARG1]] to sizes: [128], strides: [1], offsets: [0]
// CHECK: "tts.store"([[DST]], [[VALUE]])

// CHECK-LABEL: tt.func @fold_unextended_shared_factor_pointer_offset
// CHECK-SAME: ([[ARG0:%.+]]: !tt.ptr<f32>, [[ARG1:%.+]]: !tt.ptr<f32>)
// CHECK: [[C4:%.+]] = arith.constant 4 : index
// CHECK-DAG: [[SRC:%.+]] = tts.make_tptr [[ARG0]] to sizes: [128], strides: [[[C4]]], offsets: [0]
// CHECK-DAG: [[VALUE:%.+]] = "tts.load"([[SRC]])
// CHECK-DAG: [[DST:%.+]] = tts.make_tptr [[ARG1]] to sizes: [128], strides: [[[C4]]], offsets: [0]
// CHECK: "tts.store"([[DST]], [[VALUE]])

// CHECK-LABEL: tt.func @do_not_reassociate_plain_integer_arith
// CHECK: arith.divsi
// CHECK: arith.remsi
// CHECK: arith.addi

// CHECK-LABEL: tt.func @do_not_reassociate_extended_shared_factor_offset
// CHECK: arith.divsi
// CHECK: arith.remsi
// CHECK: arith.addi
