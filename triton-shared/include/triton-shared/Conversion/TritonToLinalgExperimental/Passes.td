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

def DoubleBufferLoopAllocations : Pass<"double-buffer-loop-allocations", "mlir::ModuleOp"> {
  let summary = "hoist small allocations from reusable loops into two buffers";
  let constructor = "triton::createDoubleBufferLoopAllocationsPass()";
}

def LowerOuterReductionSVE : Pass<"lower-outer-reduction-sve"> {
  let summary = "lower outer reductions and transposes in SVE-width chunks";
  let constructor = "triton::createLowerOuterReductionSVEPass()";
  let options = [
      Option<"vscale", "vscale", "int64_t", /*default*/"1",
             "runtime SVE vector length divided by 128 bits">,
  ];
}

def ExpandExp2Fexpa : Pass<"expand-exp2-fexpa", "mlir::ModuleOp"> {
  let summary = "Expand math.exp2 on fixed-length f32 vectors into SVE FEXPA sequences";
  let description = [{
    Rewrites math.exp2 on fixed-length f32 vectors whose size is a multiple of
    the chunk width into a 1-ULP-accurate FEXPA-based sequence per chunk of
    `chunk-width` lanes:

      clamp to [-126, 127], add 196735.0 (1.5 * 2^17 + 127), bitcast to i32,
      a call to the upstream llvm.aarch64.sve.fexpa.x intrinsic (declared in
      the module), then a correction polynomial built from fma.

    Each chunk is moved in and out of the SVE domain with the standard
    vector.scalable.insert / vector.scalable.extract operations (lowered to
    llvm.vector.insert/extract), which reinterpret a fixed-length vector as
    its `vector<[4]xf32>` SVE container register. The chunk width must
    therefore fit into one SVE register, i.e.
    it must be at most vscale_min * 4 lanes; the caller (e.g. the triton-shared
    pipeline) picks the chunk width from the same vscale value that is used to
    attach the vscale_range to the function.
  }];
  let constructor = "triton::createExpandExp2FexpaPass()";
  let options = [
      Option<"chunkWidth", "chunk-width", "int64_t", /*default*/"16",
             "Number of f32 lanes per SVE container chunk (= vscale_min * 4)">,
  ];
}

#endif
