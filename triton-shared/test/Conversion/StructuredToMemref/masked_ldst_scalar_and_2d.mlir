// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
module {
  tt.func @kernel(
  %arg0 : !tt.ptr<f16>,
  %arg1 : !tt.ptr<f16>,
  %arg2 : i32,
  %arg3 : i32
  )
  {
    %c197 = arith.constant 197 : i32
    %pid = tt.get_program_id x : i32
    %xmask = arith.cmpi slt, %pid, %arg2 : i32
    %xmask_t = tt.splat %xmask : i1 -> tensor<1x256xi1>

    %r = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %r2 = tt.expand_dims %r {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %rnum = tt.splat %arg3 : i32 -> tensor<1x256xi32>
    %rmask = arith.cmpi slt, %r2, %rnum : tensor<1x256xi32>

    %offset = arith.muli %pid, %c197 : i32
    %offset_t = tt.splat %offset : i32 -> tensor<1x256xi32>
    %idx = arith.addi %r2, %offset_t : tensor<1x256xi32>

    %ptr0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x256x!tt.ptr<f16>>
    %ptr1 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1x256x!tt.ptr<f16>>
    %ldptr = tt.addptr %ptr0, %idx : tensor<1x256x!tt.ptr<f16>>, tensor<1x256xi32>
    %stptr = tt.addptr %ptr1, %idx : tensor<1x256x!tt.ptr<f16>>, tensor<1x256xi32>

    %mask = arith.andi %rmask, %xmask_t : tensor<1x256xi1>
    %val = tt.load %ldptr, %mask : tensor<1x256x!tt.ptr<f16>>
    tt.store %stptr, %val, %mask : tensor<1x256x!tt.ptr<f16>>
    tt.return
  }
}

// CHECK-LABEL:  func.func @kernel
// CHECK:        [[CST_256:%.+]] = arith.constant 256 : index
// CHECK:        [[CST_0:%.+]] = arith.constant 0 : index
// CHECK:        [[CST_1:%.+]] = arith.constant 1 : index
// CHECK:        [[ARG3_IDX:%.+]] = arith.index_cast %arg3 : i32 to index
// CHECK:        [[MIN_256:%.+]] = arith.minsi [[ARG3_IDX]], [[CST_256]] : index
// CHECK:        [[CLAMPED:%.+]] = arith.maxsi [[MIN_256]], [[CST_0]] : index
// CHECK:        [[XMASK_IDX:%.+]] = arith.index_cast {{%.+}} : i1 to index
// CHECK:        [[XMASK_NE:%.+]] = arith.cmpi ne, [[XMASK_IDX]], [[CST_0]] : index
// CHECK:        [[ROW:%.+]] = arith.select [[XMASK_NE]], [[CST_1]], [[CST_0]] : index
// CHECK:        [[COL:%.+]] = arith.select [[XMASK_NE]], [[CLAMPED]], [[CST_0]] : index
// CHECK:        memref.subview %reinterpret_cast
