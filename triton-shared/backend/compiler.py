from triton.backends.compiler import BaseBackend, GPUTarget, CPUFallbackException
from triton._C.libtriton import ir, passes, cpu, llvm, triton_shared
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional
from types import ModuleType
import hashlib
import tempfile
import os
import re
import shutil
import subprocess
import functools
import textwrap
from pathlib import Path
from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects import pdl
from mlir.dialects.transform import pdl as transform_pdl
from mlir.dialects.transform import structured, loop, vector, bufferization, tensor

## Use SME/SVE even if the CPU does not support it. Will result in a crash but will generate the binary
FORCE_SME = False
FORCE_SVE = False
ENABLE_FALLBACK = False

def _get_triton_shared_opt_path() -> str:
    path = os.getenv("TRITON_SHARED_OPT_PATH", "")
    if path == "":
        raise Exception("TRITON_SHARED_OPT_PATH is not set.")
    return path


def _get_llvm_bin_path(bin_name: str) -> str:
    path = os.getenv("LLVM_BINARY_DIR", "")
    if path == "":
        raise Exception("LLVM_BINARY_DIR is not set.")
    return os.path.join(path, bin_name)


def _dump_ir_if_needed(files):
    path = os.getenv("TRITON_SHARED_DUMP_PATH", "")
    if not path:
        return
    for f in files:
        shutil.copy(f, os.path.join(path, os.path.basename(f)))




@dataclass(frozen=True)
class CPUOptions:
    debug: bool = False
    arch: str = None
    num_warps: int = 0
    num_threads: int = 1
    num_ctas: int = 0
    num_stages: int = 1
    enable_warp_specialization: bool = False
    extern_libs = None
    cluster_dims: tuple = (1, 1, 1)
    shared: bool = False
    # Disable FP8 here since this is a sample CPU backend.
    # Target specific backends can eanble it with supported types.
    allowed_dot_input_precisions: Tuple[str] = ("ieee", "tf32", "tf32x3")
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", "fp8e5b16", "fp8e4nv")
    allow_fp8e4nv: bool = True
    allow_fp8e4b15: bool = True
    enable_fp_fusion: bool = True
    max_num_imprecise_acc_default: int = 0
    enable_fast_math: bool = True
    sanitize_overflow: bool = True
    vec_lib: Optional[str] = 'libsleef'

    def __post_init__(self):
        pass

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()


class CPUBackend(BaseBackend):
    binary_ext = 'obj'

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'cpu'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.cpu_features = llvm.get_cpu_features()
        self.cpu_arch = llvm.get_cpu_tripple().split("-")[0]
        # Only detect vscale on aarch64 with SVE; None means SVE unused
        if self.cpu_arch == "aarch64" and "sve" in self.cpu_features:
            self.sve_vscale = self._detect_sve_vscale()
        else:
            self.sve_vscale = None

    def _detect_sve_vscale(self):
        """Detect the SVE vector scale factor from hardware.

        vscale = SVE_vector_length_in_bytes / 16.
        Returns 1 for 128-bit SVE, 2 for 256-bit, etc.

        Must only be called on aarch64 with SVE support.
        Raises RuntimeError if detection fails — a silent fallback
        would hide configuration errors that crash at runtime."""
        try:
            import ctypes
            import tempfile
            import subprocess
            # Use a small C program to read RDVL
            with tempfile.TemporaryDirectory() as tmpdir:
                src = os.path.join(tmpdir, "rdvl.c")
                exe = os.path.join(tmpdir, "rdvl")
                Path(src).write_text(
                    '#include <stdio.h>\n#include <stdint.h>\n'
                    'int main(){uint64_t v;asm volatile("rdvl %0, #1":"=r"(v));'
                    'printf("%lu",v/16);return 0;}\n'
                )
                subprocess.check_call(["gcc", "-march=armv8-a+sve", src, "-o", exe],
                                       stderr=subprocess.DEVNULL)
                result = subprocess.check_output([exe]).decode().strip()
                vscale = int(result)
                return max(vscale, 1)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to detect SVE vscale via rdvl: {exc}"
            ) from exc

    def parse_options(self, opts) -> Any:
        args = {'arch': self.target.arch}
        args.update({k: opts[k] for k in CPUOptions.__dataclass_fields__.keys() if k in opts})
        return CPUOptions(**args)

    def get_codegen_implementation(self):
        codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
        return codegen_fns

    def pack_metadata(self, metadata):
        # Note: We actually don't need any of these except for the name which is
        # used in the launch function in driver.py. Putting these in so we're
        # consistent with other backends
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
            metadata.name
        )

    # Our compilation pipeline isn't in python like nvidia or amd, no need to load
    # dialects. See `triton_shared.cc`
    def load_dialects(self, ctx):
        return

    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        cpu.passes.ttcpuir.add_convert_unsupported_ops(pm, False, True, True)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod)
        return mod

    def _ttir_to_ttsharedir(self, mod):
        # Get Triton-MLIR as string
        ttir_code = str(mod)
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "tt.mlir")
            dst_path = os.path.join(tmpdir, "ttshared.mlir")
            Path(src_path).write_text(ttir_code)
            _dump_ir_if_needed([src_path])
            triton_shared_opt_path = _get_triton_shared_opt_path()
            try:
                subprocess.check_call([triton_shared_opt_path, src_path, "--triton-to-linalg-experimental", "-o", dst_path])
                return Path(dst_path).read_text()
            except subprocess.CalledProcessError as e:
                if ENABLE_FALLBACK:
                    print("TritonShared-MLIR optimization failed, falling back to CPU backend")
                    os.environ["TRITON_USE_SHARED_BACKEND"] = "0"
                    raise CPUFallbackException
            
            return Path(dst_path).read_text()



    def _sve_transform(self, src: str) -> str:

        def main_type1(include, name):
            sequence = transform.NamedSequenceOp(
                "main_type1_" + name,
                [transform.AnyOpType.get()],
                [],
                arg_attrs = [{"transform.readonly": UnitAttr.get()}],
            )

            with InsertionPoint(sequence.body):
                ## get all funcs
                funcs = structured.MatchOp.match_op_names(
                    transform.AnyOpType.get(),
                    sequence.bodyTarget,
                    ["func.func"]
                )
                ## get parent op
                p = transform.get_parent_op(
                    transform.AnyOpType.get(),
                    funcs.result, 
                    deduplicate=True,
                )
                ## include
                sme = transform.IncludeOp(
                    [transform.AnyOpType.get()],
                    include,
                    transform.FailurePropagationMode.Propagate,
                    [p],
                )
                
                ## cse
                cse = transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(),
                    sme.result,
                    "cse",
                )

                with InsertionPoint(transform.ApplyPatternsOp(cse).patterns):
                    structured.apply_patterns_linalg_tiling_canonicalization()
                    loop.apply_patterns_scf_for_loop_canonicalization()
                
                ## match looplike
                looplike = structured.MatchOp.__base__(
                    transform.AnyOpType.get(),
                    cse.result,
                    interface=structured.MatchInterfaceEnum.LoopLikeInterface
                )
                ## apply licm
                transform.apply_licm(
                    looplike.result,
                )
                ## match func from cse
                funcs = structured.MatchOp.match_op_names(
                    transform.AnyOpType.get(),
                    cse.result,
                    ["func.func"]
                )
                ## hoist redudant vector transfers
                a = transform.structured.HoistRedundantVectorTransfersOp(
                    transform.AnyOpType.get(),
                    funcs.result,
                )

                b = transform.structured.HoistRedundantVectorCastsOp(
                    transform.AnyOpType.get(),
                    a.result,
                )

                ## hoist redundant vector broadcasts
                c = transform.structured.HoistRedundantVectorBroadcastsOp(
                    transform.AnyOpType.get(),
                    b.result,
                )
                ## canonicalize
                transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(),
                    c.result,
                    "canonicalize",
                )

                transform.YieldOp([])
  
        def main_type2(include, name):
            sequence = transform.NamedSequenceOp(
                "main_type2_" + name,
                [transform.AnyOpType.get()],
                [],
                arg_attrs = [{"transform.readonly": UnitAttr.get()}],
            )

            with InsertionPoint(sequence.body):
                ## get all funcs
                funcs = structured.MatchOp.match_op_names(
                    transform.AnyOpType.get(),
                    sequence.bodyTarget,
                    ["func.func"]
                )
                ## get parent op
                p = transform.get_parent_op(
                    transform.AnyOpType.get(),
                    funcs.result, 
                    deduplicate=True,
                )
                ## cse
                cse = transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(),
                    p,
                    "cse",
                )

                can = transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(),
                    cse.result,
                    "canonicalize",
                )

                ## include
                sme = transform.IncludeOp(
                    [transform.AnyOpType.get()],
                    include, 
                    transform.FailurePropagationMode.Propagate,
                    [can],
                )
                
                cse = transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(),
                    sme.result,
                    "cse",
                )

                with InsertionPoint(transform.ApplyPatternsOp(cse).patterns):
                    structured.apply_patterns_linalg_tiling_canonicalization()
                    loop.apply_patterns_scf_for_loop_canonicalization()
                
                ## match looplike
                looplike = structured.MatchOp.__base__(
                    transform.AnyOpType.get(),
                    cse.result,
                    interface=structured.MatchInterfaceEnum.LoopLikeInterface
                )
                ## apply licm
                transform.apply_licm(
                    looplike.result,
                )
                ## match func from cse
                funcs = structured.MatchOp.match_op_names(
                    transform.AnyOpType.get(),
                    cse.result,
                    ["func.func"]
                )
                ## hoist redudant vector transfers
                a = transform.structured.HoistRedundantVectorTransfersOp(
                    transform.AnyOpType.get(),
                    funcs.result,
                )

                b = transform.structured.HoistRedundantVectorCastsOp(
                    transform.AnyOpType.get(),
                    a.result,
                )

                ## hoist redundant vector broadcasts
                c = transform.structured.HoistRedundantVectorBroadcastsOp(
                    transform.AnyOpType.get(),
                    b.result,
                )
                ## canonicalize
                transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(),
                    c.result,
                    "canonicalize",
                )

                transform.YieldOp([])
 
 

        def contraction_schedule():
            sequence = transform.NamedSequenceOp(
                "contraction_schedule",
                [transform.AnyOpType.get()],
                [transform.AnyOpType.get()],
                arg_attrs=[{"transform.readonly": UnitAttr.get()}],
            )

            with InsertionPoint(sequence.body):
                
                matmuls = structured.MatchOp.match_op_names(
                    sequence.bodyTarget,
                    ["linalg.matmul"]
                )
                
                tiled1 = structured.TileUsingForOp(
                    matmuls.result,
                    sizes=[2048, 256, 256],
                    interchange=Attribute.parse("[0, 2, 1]"),
                )

                _tile = 4 * self.sve_vscale
                tiled2 = structured.TileUsingForOp(
                    tiled1.results[0],
                    sizes=[_tile, _tile, 1],
                    interchange=Attribute.parse("[0, 1, 2]"),
                )

                padded_tuple = structured.PadOp(
                    tiled2.results[0],
                    copy_back_op="none",
                    pad_to_multiple_of=[1, 1, 1],
                    pack_paddings = Attribute.parse("[0, 1, 1, 0]"),
                    padding_dimensions=Attribute.parse("[0, 1, 2]"),
                    padding_values=[StringAttr.get("0x0"), StringAttr.get("0x0"), StringAttr.get("0x0"), StringAttr.get("0x0")],
                )

                fors = structured.MatchOp.match_op_names(
                    transform.AnyOpType.get(),
                    sequence.bodyTarget,
                    ["scf.for"]
                )

                transform.apply_licm(
                    fors.result,
                )

                funcs = structured.MatchOp.match_op_names(
                    transform.AnyOpType.get(),
                    sequence.bodyTarget,
                    ["func.func"]
                )

                alt = transform.AlternativesOp(
                    [],  # results_
                    2,                            # number of alternatives
                    scope=funcs                   # %funcs
                )

                block0 = alt.regions[0].blocks.append(transform.AnyOpType.get())
                f = block0.arguments[0]
                with InsertionPoint(block0):
                    mm = structured.MatchOp.match_op_names(
                        f,
                        ["linalg.matmul"] 
                    )
                    producer0 = transform.GetProducerOfOperand(transform.AnyOpType.get(), mm.results, 1)
                    producer1 = transform.GetProducerOfOperand(transform.AnyOpType.get(), mm.results, 2)
                    hoisted0 = structured.HoistPadOp(transform.AnyOpType.get(), producer0, 4, transpose=[1, 0])
                    hoisted1 = structured.HoistPadOp(transform.AnyOpType.get(), producer1, 3, transpose=[0, 1])
                    transform.YieldOp([])

                block1 = alt.regions[1].blocks.append(transform.AnyOpType.get())
                f = block1.arguments[0]
                with InsertionPoint(block1):
                    mm = structured.MatchOp.match_op_names(
                        f,
                        ["linalg.matmul"] 
                    )
                    transform.YieldOp([])

                transform.YieldOp([sequence.bodyTarget])

        def vectorize_schedule():
            sequence = transform.NamedSequenceOp(
                "vectorize_schedule",
                [transform.AnyOpType.get()],
                [transform.AnyOpType.get()],
                arg_attrs = [{"transform.consumed": UnitAttr.get()}],
            )
            with InsertionPoint(sequence.body):
                vec = structured.VectorizeChildrenAndApplyPatternsOp(sequence.bodyTarget, vectorize_padding=True, vectorize_nd_extract=True)
                isInnerReduction = triton_shared.get_cpu_options_from_env().enable_warp_specialization
                selectedLoweringStrategy = vector.VectorMultiReductionLowering.InnerReduction if isInnerReduction else vector.VectorMultiReductionLowering.InnerParallel
                # Apply patterns to the target
                with InsertionPoint(transform.ApplyPatternsOp(vec).patterns):
                    # Add the lower_multi_reduction pattern
                    vector.ApplyLowerMultiReductionPatternsOp(
                        lowering_strategy=selectedLoweringStrategy
                    )

                # The transform modifies the IR in-place, so we yield the original handle to the modified IR
                transform.YieldOp([vec])

        def bufferize_schedule():
            sequence = transform.NamedSequenceOp(
                "bufferize_schedule",
                [transform.AnyOpType.get()],
                [transform.AnyOpType.get()],
                arg_attrs = [{"transform.consumed": UnitAttr.get()}],
            )
            with InsertionPoint(sequence.body):
                matched = structured.MatchOp.match_op_names(
                    sequence.bodyTarget,
                    ["tensor.empty"]
                )
                
                cast = transform.CastOp(transform.OperationType.get("tensor.empty"), matched.result)               
                alloc = bufferization.EmptyTensorToAllocTensorOp(cast.result)
                oneshot = bufferization.OneShotBufferizeOp(sequence.bodyTarget, bufferize_function_boundaries=True, memcpy_op="linalg.copy")
                funcs = structured.MatchOp.match_op_names(
                    transform.AnyOpType.get(),
                    oneshot.result,
                    ["func.func"]
                )
                transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(),
                    funcs.result,
                    "buffer-deallocation"
                )

                transform.YieldOp([oneshot])

        def main_bufferize():
            sequence = transform.NamedSequenceOp(
                "main_bufferize",
                [transform.AnyOpType.get()],
                [],
                arg_attrs = [{"transform.readonly": UnitAttr.get()}],
            )

            with InsertionPoint(sequence.body):
                ## get all funcs
                funcs = structured.MatchOp.match_op_names(
                    transform.AnyOpType.get(),
                    sequence.bodyTarget,
                    ["func.func"]
                )
                ## get parent op
                p = transform.get_parent_op(
                    transform.AnyOpType.get(),
                    funcs.result, 
                    deduplicate=True,
                )
                ## include
                sme = transform.IncludeOp(
                    [transform.AnyOpType.get()],
                    "bufferize_schedule",
                    transform.FailurePropagationMode.Propagate,
                    [p],
                )
                
                ## cse
                cse = transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(),
                    sme.result,
                    "cse",
                )

                can = transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(),
                    cse.result,
                    "canonicalize",
                )

                funcs = structured.MatchOp.match_op_names(
                    transform.AnyOpType.get(),
                    can.result,
                    ["func.func"]
                )

                a = transform.structured.HoistRedundantVectorTransfersOp(
                    transform.AnyOpType.get(),
                    funcs.result,
                )

                cse = transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(),
                    a.result,
                    "cse",
                )

                with InsertionPoint(transform.ApplyPatternsOp(cse).patterns):
                    structured.apply_patterns_linalg_tiling_canonicalization()
                    loop.apply_patterns_scf_for_loop_canonicalization()
                
                ## match looplike
                looplike = structured.MatchOp.__base__(
                    transform.AnyOpType.get(),
                    cse.result,
                    interface=structured.MatchInterfaceEnum.LoopLikeInterface
                )
                ## apply licm
                transform.apply_licm(
                    looplike.result,
                )
                ## match func from cse
                funcs = structured.MatchOp.match_op_names(
                    transform.AnyOpType.get(),
                    cse.result,
                    ["func.func"]
                )
                ## hoist redudant vector transfers
                a = transform.structured.HoistRedundantVectorTransfersOp(
                    transform.AnyOpType.get(),
                    funcs.result,
                )

                b = transform.structured.HoistRedundantVectorCastsOp(
                    transform.AnyOpType.get(),
                    a.result,
                )

                ## hoist redundant vector broadcasts
                c = transform.structured.HoistRedundantVectorBroadcastsOp(
                    transform.AnyOpType.get(),
                    b.result,
                )
                ## canonicalize
                transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(),
                    c.result,
                    "canonicalize",
                )

                transform.YieldOp([])
 

        def legalize_schedule():
            sequence = transform.NamedSequenceOp(
                "legalize_schedule",
                [transform.AnyOpType.get()],
                [transform.AnyOpType.get()],
                arg_attrs=[{"transform.readonly": UnitAttr.get()}],
            )

            with InsertionPoint(sequence.body):
                result = transform.legalize(
                    vscale=self.sve_vscale,
                )
                transform.YieldOp([sequence.bodyTarget])
 
           
        
        def pipeline_schedule():
            sequence = transform.NamedSequenceOp(
                "pipeline_schedule",
                [transform.AnyOpType.get()],
                [transform.AnyOpType.get()],
                arg_attrs=[{"transform.consumed": UnitAttr.get()}],
            )

            with InsertionPoint(sequence.body):
                withPdl3 = transform_pdl.WithPDLPatternsOp(pdl.OperationType.get())
                with InsertionPoint(withPdl3.body):
                    pattern3 = pdl.PatternOp(1, "isMicroKernel")
                    with InsertionPoint(pattern3.body):
                        operands = pdl.OperandsOp()
                        ty = pdl.TypesOp()
                        loop_op = pdl.OperationOp(name="scf.for", args=[operands], types=[ty])
                        pdl.ApplyNativeConstraintOp([], "isMicroKernel", args=[loop_op])
                        pdl.RewriteOp(loop_op, name="transform.dialect")

                    pdl_seq3 = transform.SequenceOp(transform.FailurePropagationMode.Propagate, [], withPdl3.bodyTarget)
                    with InsertionPoint(pdl_seq3.body):
                        matched3 = transform_pdl.PDLMatchOp(pdl.OperationType.get(), pdl_seq3.bodyTarget, "isMicroKernel")
                        loop.LoopUnrollOp(matched3.result, factor=18)
                        transform.YieldOp([])

                res3 = transform.ApplyRegisteredPassOp(transform.AnyOpType.get(), sequence.bodyTarget, "cse")
                res4 = transform.ApplyRegisteredPassOp(transform.AnyOpType.get(), res3, "canonicalize")
                res5 = transform.ApplyRegisteredPassOp(transform.AnyOpType.get(), res4, "cse")

                withPdl4 = transform_pdl.WithPDLPatternsOp(pdl.OperationType.get())
                with InsertionPoint(withPdl4.body):
                    pattern4 = pdl.PatternOp(1, "isMicroKernel")
                    with InsertionPoint(pattern4.body):
                        operands = pdl.OperandsOp()
                        ty = pdl.TypesOp()
                        loop_op = pdl.OperationOp(name="scf.for", args=[operands], types=[ty])
                        pdl.ApplyNativeConstraintOp([], "isMicroKernel", args=[loop_op])
                        pdl.RewriteOp(loop_op, name="transform.dialect")

                    pdl_seq4 = transform.SequenceOp(transform.FailurePropagationMode.Propagate, [], withPdl4.bodyTarget)
                    with InsertionPoint(pdl_seq4.body):
                        matched4 = transform_pdl.PDLMatchOp(pdl.OperationType.get(), pdl_seq4.bodyTarget, "isMicroKernel")
                        split0, split1 = transform.SplitHandleOp([pdl.OperationType.get(), pdl.OperationType.get()], matched4.result).results  # yields two handles
                        cast_micro_for = transform.CastOp(transform.OperationType.get("scf.for"), matched4)
                        loop.LoopPipelineOp(
                            transform.OperationType.get("scf.for"),
                            cast_micro_for,
                            iteration_interval=18,
                            read_latency=1,
                        )
                        transform.YieldOp([])

                transform.YieldOp([res5])


        def loops_schedule():
            sequence = transform.NamedSequenceOp(
                "loops_schedule",
                [transform.AnyOpType.get()],
                [transform.AnyOpType.get()],
                arg_attrs=[{"transform.readonly": UnitAttr.get()}],
            )

            with InsertionPoint(sequence.body):
                # match all func.func in the input
                funcs = structured.MatchOp.match_op_names(
                    sequence.bodyTarget,
                    ["func.func"],
                )

                # 1) lower_contraction + transfer_permutation_patterns
                with InsertionPoint(transform.ApplyPatternsOp(funcs.result).patterns):
                    vector.ApplyLowerContractionPatternsOp()
                    vector.ApplyTransferPermutationPatternsOp()

                # 2) add lower_multi_reduction
                with InsertionPoint(transform.ApplyPatternsOp(funcs.result).patterns):
                    vector.ApplyLowerContractionPatternsOp()
                    vector.ApplyTransferPermutationPatternsOp()
                    vector.ApplyVectorReductionToContractPatternsOp()

                # 3) add split_transfer_full_partial (strategy = "vector-transfer")
                with InsertionPoint(transform.ApplyPatternsOp(funcs.result).patterns):
                    vector.ApplyLowerContractionPatternsOp()
                    vector.ApplyTransferPermutationPatternsOp()
                    vector.ApplyVectorReductionToContractPatternsOp()
                    # pass the split_transfer_strategy as a string attr
                    vector.ApplySplitTransferFullPartialPatternsOp(split_transfer_strategy=vector.VectorTransferSplit.VectorTransfer)

                # 4) add lower_transfer
                with InsertionPoint(transform.ApplyPatternsOp(funcs.result).patterns):
                    vector.ApplyLowerContractionPatternsOp()
                    vector.ApplyTransferPermutationPatternsOp()
                    vector.ApplyVectorReductionToContractPatternsOp()
                    vector.ApplySplitTransferFullPartialPatternsOp(split_transfer_strategy=vector.VectorTransferSplit.VectorTransfer)
                    vector.ApplyLowerTransferPatternsOp()

                # 5) add transfer_to_scf full_unroll = true
                with InsertionPoint(transform.ApplyPatternsOp(funcs.result).patterns):
                    vector.ApplyLowerContractionPatternsOp()
                    vector.ApplyTransferPermutationPatternsOp()
                    vector.ApplyVectorReductionToContractPatternsOp()
                    vector.ApplySplitTransferFullPartialPatternsOp(split_transfer_strategy=vector.VectorTransferSplit.VectorTransfer)
                    vector.ApplyLowerTransferPatternsOp()
                    # full_unroll is a boolean attribute
                    vector.ApplyTransferToScfPatternsOp(full_unroll=True)

                # 6) add lower_shape_cast
                with InsertionPoint(transform.ApplyPatternsOp(funcs.result).patterns):
                    vector.ApplyLowerContractionPatternsOp()
                    vector.ApplyTransferPermutationPatternsOp()
                    vector.ApplyVectorReductionToContractPatternsOp()
                    vector.ApplySplitTransferFullPartialPatternsOp(split_transfer_strategy=vector.VectorTransferSplit.VectorTransfer)
                    vector.ApplyLowerTransferPatternsOp()
                    vector.ApplyTransferToScfPatternsOp(full_unroll=True)
                    vector.ApplyLowerShapeCastPatternsOp()

                # 7) add lower_transpose
                with InsertionPoint(transform.ApplyPatternsOp(funcs.result).patterns):
                    vector.ApplyLowerContractionPatternsOp()
                    vector.ApplyTransferPermutationPatternsOp()
                    vector.ApplyVectorReductionToContractPatternsOp()
                    vector.ApplySplitTransferFullPartialPatternsOp(split_transfer_strategy=vector.VectorTransferSplit.VectorTransfer)
                    vector.ApplyLowerTransferPatternsOp()
                    vector.ApplyTransferToScfPatternsOp(full_unroll=True)
                    vector.ApplyLowerShapeCastPatternsOp()
                    vector.ApplyLowerTransposePatternsOp()

                # match again (equivalent to %1 in MLIR) and run registered pass
                funcs2 = structured.MatchOp.match_op_names(
                    sequence.bodyTarget,
                    ["func.func"],
                )
                res_pass = transform.ApplyRegisteredPassOp(transform.AnyOpType.get(), funcs2.result, "scf-for-to-while")

                # yield the original input handle (same as `transform.yield %arg0`)
                transform.YieldOp([sequence.bodyTarget])


        def lower_to_llvm_schedule():
            sequence = transform.NamedSequenceOp(
                "lower_to_llvm_schedule",
                [transform.AnyOpType.get()],
                [transform.AnyOpType.get()],
                arg_attrs=[{"transform.readonly": UnitAttr.get()}],
            )

            with InsertionPoint(sequence.body):
                result = transform.lower_to_llvm_new(
                    sequence.bodyTarget,
                    enable_arm_sve=True,
                    enable_index_optimizations=True,
                    vscale_range=self.sve_vscale,
                )
                transform.YieldOp([sequence.bodyTarget])
 

        ## instead of using mlir-opt we embed the optimization passes in the transform dialect 
        def opt():
            sequence = transform.NamedSequenceOp(
                "opt",
                [transform.OperationType.get("func.func")],
                [],
                arg_attrs = [{"transform.consumed": UnitAttr.get()}],
            )

            with InsertionPoint(sequence.body):

                   
                fp = transform.ApplyRegisteredPassOp(
                    transform.OperationType.get("func.func"),
                    sequence.bodyTarget,
                    "arith-emulate-unsupported-floats",
                    options='source-types=f8E5M2,f8E4M3FN,bf16 target-type=f32'
                )

                fp2 = transform.ApplyRegisteredPassOp(
                    transform.OperationType.get("func.func"),
                    fp.result,
                    "arith-expand",
                    options='include-f8e5m2=true include-bf16=true include-f8e4m3fn=true'
                )
 
                poly = transform.ApplyRegisteredPassOp(
                    transform.OperationType.get("func.func"),
                    fp2.result,
                    "test-math-polynomial-approximation",
                )

                p = transform.get_parent_op(
                    transform.AnyOpType.get(),
                    poly.result, 
                    deduplicate=True,
                )

                tptr = transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(),
                    p,
                    "tptr-to-llvm",
                )

                
                cann = transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(),
                    tptr.result,
                    "canonicalize",
                )


                fin = transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(),
                    cann.result,
                    "finalize-memref-to-llvm",
                )

                casts = transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(),
                    fin.result,
                    "reconcile-unrealized-casts",
                )

                transform.YieldOp([])

        def transform_main():
            sequence = transform.NamedSequenceOp(
                "__transform_main",
                [transform.AnyOpType.get()],
                [],
                arg_attrs = [{"transform.readonly": UnitAttr.get()}],
            )
                
            with InsertionPoint(sequence.body):
                    
                include2 = transform.IncludeOp(
                    [],
                    FlatSymbolRefAttr.get("main_type1_contraction"),
                    transform.FailurePropagationMode.Propagate,
                    [sequence.bodyTarget],
                )

                include3 = transform.IncludeOp(
                    [],
                    FlatSymbolRefAttr.get("main_type1_vectorize"),
                    transform.FailurePropagationMode.Propagate,
                    [sequence.bodyTarget],
                )

                include4 = transform.IncludeOp(
                    [],
                    FlatSymbolRefAttr.get("main_bufferize"),
                    transform.FailurePropagationMode.Propagate,
                    [sequence.bodyTarget],
                )
                
                include_legalize = transform.IncludeOp(
                    [],
                    FlatSymbolRefAttr.get("main_type1_legalize"),
                    transform.FailurePropagationMode.Propagate,
                    [sequence.bodyTarget],
                )
                

                # NOTE: pipeline_schedule skipped - requires unimplemented
                # isMicroKernel PDL native constraint
                
                include6 = transform.IncludeOp(
                    [],
                    FlatSymbolRefAttr.get("main_type2_loops"),
                    transform.FailurePropagationMode.Propagate,
                    [sequence.bodyTarget],
                )
                
                funcs = structured.MatchOp.match_op_names(
                    transform.OperationType.get("func.func"),
                    sequence.bodyTarget,
                    ["func.func"]
                )
 
                ## for each
                foreach = transform.ForeachOp(
                    [],
                    funcs,
                )
                    
                foreachBody = foreach.body.blocks.append(transform.OperationType.get("func.func"))
                    
                with InsertionPoint(foreachBody):
                    # passes for fp8
                    transform.IncludeOp(
                        [],
                        FlatSymbolRefAttr.get("opt"),
                        transform.FailurePropagationMode.Propagate,
                        [foreachBody.arguments[0]],
                    )
                    transform.YieldOp([])
                
                include7 = transform.IncludeOp(
                    [],
                    FlatSymbolRefAttr.get("main_type2_lower_to_llvm"),
                    transform.FailurePropagationMode.Propagate,
                    [sequence.bodyTarget],
                )

                transform.YieldOp([])
                     
        with Context() as ctx, Location.unknown():
            mod = Module.create()
            ## add attributes to the module
            mod.operation.attributes["transform.with_named_sequence"] = UnitAttr.get()
            
            with InsertionPoint(mod.body):
                contraction_schedule()
                main_type1("contraction_schedule", "contraction")
                vectorize_schedule()
                main_type1("vectorize_schedule", "vectorize")
                bufferize_schedule()
                main_bufferize()
                legalize_schedule()
                main_type1("legalize_schedule", "legalize")
                # pipeline_schedule skipped - isMicroKernel not implemented
                loops_schedule()
                opt()
                main_type2("loops_schedule", "loops")
                lower_to_llvm_schedule()
                main_type2("lower_to_llvm_schedule", "lower_to_llvm")
                transform_main()

            ## Append our transform to the original source
            return src + "\n" + str(mod)


    def _sme_transform(self, src: str) -> str:

        def step1():
            sequence = transform.NamedSequenceOp(
                "__step1",
                [transform.AnyOpType.get()],
                [transform.AnyOpType.get()],
                arg_attrs = [{"transform.consumed": UnitAttr.get()}],
            )
                
            with InsertionPoint(sequence.body):

                ## get matmul
                matmuls = structured.MatchOp.match_op_names(
                    sequence.bodyTarget,
                    ["linalg.matmul"]
                )
                
                # Step 1: Tile for size [4] x [4], which corresponds to SVLs x SVLs, where
                # SVLs is the number of 32-bit elements in a vector of SVL bits.
                tiled = structured.TileUsingForOp(matmuls.result, sizes=[[4],[4],1])
                # Step 2: Vectorize.
                structured.VectorizeOp(tiled.results[0], [[4], [4], 1])
                # Step 3: Bufferize ahead of TransferReadDropUnitDimsPattern, which
                # currently only supports memrefs.
                buff = bufferization.OneShotBufferizeOp(
                    sequence.bodyTarget, bufferize_function_boundaries=True)
                ## get the funcs
                funcs = structured.MatchOp.match_op_names(
                    transform.AnyOpType.get(),
                    buff.result,
                    ["func.func"]
                )

                l = transform.ApplyRegisteredPassOp(
                    transform.OperationType.get("func.func"),
                    funcs.result,
                    "convert-linalg-to-loops",
                )

                # Step 4: Lower vector.multi_reduction to vector.contract (+ some helpful patterns).
                with InsertionPoint(transform.ApplyPatternsOp(l).patterns):
                    vector.ApplyLowerMaskedTransfersPatternsOp()
                    vector.ApplyTransferPermutationPatternsOp()
                    vector.ApplyVectorReductionToContractPatternsOp()

                # Step 5: Lower vector.contract to vector.outerproduct. Also drop unit
                # dims, specifically to prevent vector.transfer_read of vector<[4]x1xf32>,
                # which can't be lowered in generic path.
                with InsertionPoint(transform.ApplyPatternsOp(l).patterns):
                    vector.ApplyCastAwayVectorLeadingOneDimPatternsOp()
                    tensor.ApplyFoldTensorSubsetOpsIntoVectorTransfersPatternsOp()
                    vector.ApplyLowerContractionPatternsOp(lowering_strategy=vector.VectorContractLowering.OuterProduct)
                    vector.ApplyLowerMasksPatternsOp()
                    transform.ApplyCanonicalizationPatternsOp()

                all_loops = structured.MatchOp.__base__(
                    transform.AnyOpType.get(),
                    buff.result,
                    interface=structured.MatchInterfaceEnum.LoopLikeInterface
                )

                transform.apply_licm(
                    all_loops.result,
                )

                loop.loop_hoist_loop_invariant_subsets(
                    all_loops.result,
                )

                transform.YieldOp([buff.result])
 
        def arm_sme_lowering_schedule():
            sequence = transform.NamedSequenceOp(
                "__arm_sme_lowering_schedule",
                [transform.AnyOpType.get()],
                [transform.AnyOpType.get()],
                arg_attrs = [{"transform.readonly": UnitAttr.get()}],
            )
            with InsertionPoint(sequence.body):
                result = transform.lower_to_arm_sme(
                    sequence.bodyTarget,
                )
                transform.YieldOp([sequence.bodyTarget])

        def lower_to_llvm():
            sequence = transform.NamedSequenceOp(
                "__lower_to_llvm",
                [transform.AnyOpType.get()],
                [transform.AnyOpType.get()],
                arg_attrs = [{"transform.readonly": UnitAttr.get()}],
            )
            with InsertionPoint(sequence.body):
                result = transform.lower_to_llvm_new(
                    sequence.bodyTarget,
                    enable_arm_sve=True,
                    enable_index_optimizations=True,
                    vscale_range=0,
                )
                transform.YieldOp([sequence.bodyTarget])
                    
        def step2(include, name):
            sequence = transform.NamedSequenceOp(
                "__step2_" + name,
                [transform.AnyOpType.get()],
                [],
                arg_attrs = [{"transform.readonly": UnitAttr.get()}],
            )
            with InsertionPoint(sequence.body):
                ## get all funcs
                funcs = structured.MatchOp.match_op_names(
                    transform.AnyOpType.get(),
                    sequence.bodyTarget,
                    ["func.func"]
                )
                ## get parent op
                p = transform.get_parent_op(
                    transform.AnyOpType.get(),
                    funcs.result, 
                    deduplicate=True,
                )
                ## include
                sme = transform.IncludeOp(
                    [transform.AnyOpType.get()],
                    include,
                    transform.FailurePropagationMode.Propagate,
                    [p],
                )
                
                ## cse
                cse = transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(),
                    sme.result,
                    "cse",
                )

                with InsertionPoint(transform.ApplyPatternsOp(cse).patterns):
                    structured.apply_patterns_linalg_tiling_canonicalization()
                    loop.apply_patterns_scf_for_loop_canonicalization()
                
                ## match looplike
                looplike = structured.MatchOp.__base__(
                    transform.AnyOpType.get(),
                    cse.result,
                    interface=structured.MatchInterfaceEnum.LoopLikeInterface
                )
                ## apply licm
                transform.apply_licm(
                    looplike.result,
                )
                ## match func from cse
                funcs = structured.MatchOp.match_op_names(
                    transform.AnyOpType.get(),
                    cse.result,
                    ["func.func"]
                )
                ## hoist redudant vector transfers
                a = transform.structured.HoistRedundantVectorTransfersOp(
                    transform.AnyOpType.get(),
                    funcs.result,
                )
                ## hoist redundant vector broadcasts
                b = transform.structured.HoistRedundantVectorBroadcastsOp(
                    transform.AnyOpType.get(),
                    a.result,
                )
                ## canonicalize
                transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(),
                    b.result,
                    "canonicalize",
                )

                transform.YieldOp([])

        def opt():
            sequence = transform.NamedSequenceOp(
                "opt",
                [transform.OperationType.get("func.func")],
                [],
                arg_attrs = [{"transform.consumed": UnitAttr.get()}],
            )

            with InsertionPoint(sequence.body):

                   
                fp = transform.ApplyRegisteredPassOp(
                    transform.OperationType.get("func.func"),
                    sequence.bodyTarget,
                    "arith-emulate-unsupported-floats",
                    options='source-types=f8E5M2,f8E4M3FN,bf16 target-type=f32'
                )

                fp2 = transform.ApplyRegisteredPassOp(
                    transform.OperationType.get("func.func"),
                    fp.result,
                    "arith-expand",
                    options='include-f8e5m2=true include-bf16=true include-f8e4m3fn=true'
                )
 
                poly = transform.ApplyRegisteredPassOp(
                    transform.OperationType.get("func.func"),
                    fp2.result,
                    "test-math-polynomial-approximation",
                )

                p = transform.get_parent_op(
                    transform.AnyOpType.get(),
                    poly.result, 
                    deduplicate=True,
                )

                tptr = transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(),
                    p,
                    "tptr-to-llvm",
                )

                
                cann = transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(),
                    tptr.result,
                    "canonicalize",
                )


                fin = transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(),
                    cann.result,
                    "finalize-memref-to-llvm",
                )

                casts = transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(),
                    fin.result,
                    "reconcile-unrealized-casts",
                )

                transform.YieldOp([])
        
        def transform_main():
            sequence = transform.NamedSequenceOp(
                "__transform_main",
                [transform.AnyOpType.get()],
                [],
                arg_attrs = [{"transform.consumed": UnitAttr.get()}],
            )
            with InsertionPoint(sequence.body):
                ## include step 1
                first = transform.IncludeOp(
                    [transform.AnyOpType.get()],
                    FlatSymbolRefAttr.get("__step1"),
                    transform.FailurePropagationMode.Propagate,
                    [sequence.bodyTarget],
                )
                ## include step 2 sme
                transform.IncludeOp(
                    [],
                    FlatSymbolRefAttr.get("__step2_sme"),
                    transform.FailurePropagationMode.Propagate,
                    [first],
                )

                funcs = structured.MatchOp.match_op_names(
                    transform.OperationType.get("func.func"),
                    first,
                    ["func.func"]
                )
 
                ## for each
                foreach = transform.ForeachOp(
                    [],
                    funcs,
                )
                    
                foreachBody = foreach.body.blocks.append(transform.OperationType.get("func.func"))
                    
                with InsertionPoint(foreachBody):
                    # passes for fp8
                    transform.IncludeOp(
                        [],
                        FlatSymbolRefAttr.get("opt"),
                        transform.FailurePropagationMode.Propagate,
                        [foreachBody.arguments[0]],
                    )
                    transform.YieldOp([])
 
                ## include step 2 llvm
                transform.IncludeOp(
                    [],
                    FlatSymbolRefAttr.get("__step2_llvm"),
                    transform.FailurePropagationMode.Propagate,
                    [first],
                )

                transform.YieldOp([])
        
        
        with Context() as ctx, Location.unknown():
            mod = Module.create()
            ## add attributes to the module
            mod.operation.attributes["transform.with_named_sequence"] = UnitAttr.get()
            
            with InsertionPoint(mod.body):
                step1()
                arm_sme_lowering_schedule()
                lower_to_llvm()
                step2(FlatSymbolRefAttr.get("__arm_sme_lowering_schedule"), "sme")
                opt()
                step2(FlatSymbolRefAttr.get("__lower_to_llvm"), "llvm")
                transform_main()

            ## Append our transform to the original source
            return src + "\n" + str(mod)

       

    def _optimize_ttsharedir(self, src: str):
        if(FORCE_SME or (self.cpu_arch == "aarch64" and "sme" in self.cpu_features)):
            return self._sme_transform(src)
        elif (FORCE_SVE or (self.cpu_arch == "aarch64" and "sve" in self.cpu_features)):
            return self._sve_transform(src)


        return src

    def _extract_mlir_function(self, filepath: str) -> None:
        """
        Reads an MLIR source file, retains external llvm.func declarations and
        extracts the first llvm.func definition with a body, then overwrites the file
        with these declarations followed by the function body.

        Args:
            filepath: Path to the MLIR source file to process.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        decl_pattern = re.compile(r"^\s*llvm\.func\b.*\)(?:\s*->.*)?\s*$")
        body_start_pattern = re.compile(r"^\s*llvm\.func\b.*\{")

        decl_lines = []
        body_lines = []
        in_body = False
        brace_balance = 0
        output = "#loc = loc(unknown)\n"

        for line in lines:
            if not in_body:
                # Collect external declarations (no body)
                if decl_pattern.match(line):
                    decl_lines.append(line)
                    continue
                # Detect start of first function with body
                if body_start_pattern.search(line):
                    in_body = True
                    brace_balance = line.count('{') - line.count('}')
                    body_lines.append(line)
                    continue
            else:
                body_lines.append(line)
                brace_balance += line.count('{') - line.count('}')
                if brace_balance == 0:
                    break

        if not body_lines:
            return

        # Dedent the function body block
        dedented_body = textwrap.dedent(''.join(body_lines)).strip() + '\n'

        # Combine declarations and body
        output += ''.join(decl_lines).strip()
        if decl_lines:
            output += '\n\n'
        output += dedented_body

        # Overwrite file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(output)


    def _ttsharedir_to_llir(self, ttsharedir: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            ttshared_path = os.path.join(tmpdir, "ttshared.mlir")
            llmlir_path = os.path.join(tmpdir, "ll.mlir")
            llir_path = os.path.join(tmpdir, "ll.ir")
            Path(ttshared_path).write_text(ttsharedir)
            _dump_ir_if_needed([ttshared_path])
            context = ir.context()
            triton_shared.ir.load_dialects(context)
            mod = ir.parse_mlir_module(ttshared_path, context)
            pm = ir.pass_manager(context)
            pm.enable_debug()
            
            
            if FORCE_SME or FORCE_SVE or (self.cpu_arch == "aarch64" and {"sme", "sve"} & set(self.cpu_features)):
                # DEBUG: run passes one at a time and dump IR between each
                _debug_dump = os.getenv("TRITON_DEBUG_SVE_PASSES", "")
                if _debug_dump:
                    _debug_dir = _debug_dump
                    os.makedirs(_debug_dir, exist_ok=True)
                    Path(os.path.join(_debug_dir, "00_input.mlir")).write_text(str(mod))
                    
                    pm1 = ir.pass_manager(context)
                    triton_shared.to_llir.add_transform_interpreter(pm1)
                    pm1.run(mod)
                    Path(os.path.join(_debug_dir, "01_after_transform_interpreter.mlir")).write_text(str(mod))
                    
                    pm2 = ir.pass_manager(context)
                    triton_shared.to_llir.add_test_transform_dialect_erase_schedule(pm2)
                    pm2.run(mod)
                    Path(os.path.join(_debug_dir, "02_after_erase_schedule.mlir")).write_text(str(mod))
                    
                    pm3 = ir.pass_manager(context)
                    triton_shared.to_llir.add_convert_to_llvm(pm3)
                    pm3.run(mod)
                    Path(os.path.join(_debug_dir, "03_after_convert_to_llvm.mlir")).write_text(str(mod))
                    
                    pm4 = ir.pass_manager(context)
                    triton_shared.to_llir.add_canonicalizer(pm4)
                    pm4.run(mod)
                    Path(os.path.join(_debug_dir, "04_after_canonicalize.mlir")).write_text(str(mod))
                    
                    pm5 = ir.pass_manager(context)
                    triton_shared.to_llir.add_strip_debug_info(pm5)
                    pm5.run(mod)
                    Path(os.path.join(_debug_dir, "05_after_strip_debug.mlir")).write_text(str(mod))
                else:
                    triton_shared.to_llir.add_transform_interpreter(pm)
                    triton_shared.to_llir.add_test_transform_dialect_erase_schedule(pm)
                    triton_shared.to_llir.add_convert_to_llvm(pm)
                    triton_shared.to_llir.add_canonicalizer(pm)
                    triton_shared.to_llir.add_strip_debug_info(pm)
            else:
                triton_shared.to_llir.add_convert_linalg_to_affine_loops(pm)
                triton_shared.to_llir.add_empty_tensor_to_alloc_tensor(pm)
                triton_shared.to_llir.add_one_shot_bufferize(pm)
                triton_shared.to_llir.add_lower_affine(pm)
                triton_shared.to_llir.add_convert_linalg_to_loops(pm)
                triton_shared.to_llir.add_expand_strided_metadata(pm)
                triton_shared.to_llir.add_convert_scf_to_cf(pm)
                triton_shared.to_llir.add_convert_to_llvm(pm)
                triton_shared.to_llir.add_strip_debug_info(pm)

            pm.run(mod)
            Path(llmlir_path).write_text(str(mod))

            # TritonShared-MLIR to LLVM-MLIR
            self._extract_mlir_function(llmlir_path)
            _dump_ir_if_needed([llmlir_path])
            # LLVM-MLIR to LLVM-IR
            mlir_translate_path = _get_llvm_bin_path("mlir-translate")
            subprocess.check_call([mlir_translate_path, llmlir_path,
                "--mlir-to-llvmir",
                "-o",
                llir_path])
            _dump_ir_if_needed([llir_path])
            return Path(llir_path).read_text()


    def _optimize_llir(self, llir: str):
        # We don't apply any optimizations now, but we can add passes if needed.
        return llir


    def _llir_to_bin(self, llir: str, metadata):
        pattern = r"define void @(\w+)\(.+"
        matches = re.findall(pattern, llir)
        assert len(matches) == 1
        metadata["name"] = matches[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "kernel.ll")
            dst_path = os.path.join(tmpdir, "kernel.o")
            Path(src_path).write_text(llir)
            llc_path = _get_llvm_bin_path("llc")
            flags = ""
            if FORCE_SME or (self.cpu_arch == "aarch64" and "sme" in self.cpu_features):
                flags = (
                    "-mtriple=aarch64-linux-gnu",
                    "-mattr=+sme,+dotprod",
                )
            elif FORCE_SVE or (self.cpu_arch == "aarch64" and "sve" in self.cpu_features):
                flags = (
                    "-mtriple=aarch64-linux-gnu",
                    "-mattr=+sve,+dotprod",
                )
            
            subprocess.check_call([llc_path, src_path, "-filetype=obj", "-o", dst_path] + list(flags))
            ## dump binary
            _dump_ir_if_needed([dst_path])
            return Path(dst_path).read_bytes()



    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttsharedir"] = lambda src, metadata: self._optimize_ttsharedir(self._ttir_to_ttsharedir(src))
        stages["llir"] = lambda src, metadata: self._optimize_llir(self._ttsharedir_to_llir(src))
        stages["obj"] = lambda src, metadata: self._llir_to_bin(src, metadata)


    @functools.lru_cache()
    def hash(self):
        return self.target

    # The CPU backend does not use any extra python modules, return an empty dictionary
    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}
