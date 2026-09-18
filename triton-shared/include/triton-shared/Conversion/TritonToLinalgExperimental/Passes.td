//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TO_LINALG_EXPERIMENTAL_CONVERSION_PASSES
#define TRITON_TO_LINALG_EXPERIMENTAL_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToLinalgExperimental : Pass<"triton-to-linalg-experimental", "mlir::ModuleOp"> {
  let summary = "Convert Triton to Linalg dialect";
  let constructor = "triton::createTritonToLinalgExperimentalPass()";
  let options = [
      Option<"enableMakeGatherScatterTensorPtr", "enable-make-gather-scatter", "bool", /*default*/"true",
             "Enable make_gather_scatter_tptr support">,
      Option<"enableCollapseShape", "enable-collapse-shape", "bool", /*default*/"false",
             "Enable collapse shape pass">,
      Option<"enableArmPL", "enable-armpl", "bool", /*default*/"false",
             "Enable ArmPL BLAS path for tt.dot operations">,
  ];
}

def ReconcilePtrCasts : Pass<"reconcile-ptr-casts", "mlir::ModuleOp"> {
  let summary = "Convert unrealized_cast op between tt.ptr or ptr.ptr to memref to to_memref or from_memref";
  let constructor = "triton::createReconcilePtrCastsPass()";
}

def TritonToPtr : Pass<"triton-to-ptr", "mlir::ModuleOp"> {
  let summary = "Convert Triton ops on pointers to the Ptr dialect";
  let constructor = "triton::createTritonToPtrPass()";
}

def CollapseShape : Pass</*cli-arg*/"collapse-shape", /*Op*/"mlir::ModuleOp"> {
  let summary = "Compress tensor dimensions to improve linalg{broadcast,transpose,fill,reduce} efficiency";
  let constructor = "triton::createCollapseShapePass()";
}

def HoistStaticAllocs : Pass</*cli-arg*/"hoist-static-allocs", /*Op*/"mlir::ModuleOp"> {
  let summary = "hoist statically sized allocs to the entry block, controlled by TRITON_SHARED_HOIST_STATIC_ALLOCS";
  let constructor = "triton::createHoistStaticAllocsPass()";
}

def TritonAnnotateNontemporalArgs : Pass</*cli-arg*/"triton-annotate-nontemporal-args", /*Op*/"mlir::ModuleOp"> {
  let summary = "record, per tt.func, the pointer arguments accessed only through evict_first loads / stores";
  let description = [{
    Sets `tts.nontemporal_load_args` and `tts.nontemporal_store_args`
    (ordinals among the function's pointer arguments) plus
    `tts.ptr_arg_count` on every tt.func that has a pointer argument whose
    tt.load ops or tt.store ops all carry evictionPolicy = evict_first.
  }];
  let constructor = "triton::createTritonAnnotateNontemporalArgsPass()";
}

def MarkNontemporalLoads : Pass</*cli-arg*/"mark-nontemporal-loads", /*Op*/"mlir::ModuleOp"> {
  let summary = "set nontemporal on llvm.load / llvm.store ops whose address derives from an annotated argument";
  let description = [{
    For every llvm.func carrying `tts.nontemporal_load_args` and/or
    `tts.nontemporal_store_args`, follows each llvm.load's / llvm.store's
    address and sets `nontemporal` when every path ends at one of the pointer parameters listed for that access kind.
  }];
  let constructor = "triton::createMarkNontemporalLoadsPass()";
}

#endif
