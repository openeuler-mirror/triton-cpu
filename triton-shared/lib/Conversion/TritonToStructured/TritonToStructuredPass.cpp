//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/AnalysisStructured/PtrAnalysis.h"
#include "triton-shared/Conversion/TritonToStructured/TritonToStructured.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <cassert>
#include <optional>

#define DEBUG_TYPE "triton-to-structured"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonToStructured/Passes.h.inc"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TRITONTOSTRUCTURED
#include "triton-shared/Conversion/TritonToStructured/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

template <typename OpTy>
static bool hasDefaultOverflowFlags(OpTy op) {
  return op.getOverflowFlags() == arith::IntegerOverflowFlags::none;
}

// Collect the leaf operands of a (possibly nested) integer add tree.
static bool collectAddends(Value v, SmallVectorImpl<Value> &addends) {
  if (auto add = v.getDefiningOp<arith::AddIOp>()) {
    if (!hasDefaultOverflowFlags(add))
      return false;
    return collectAddends(add.getLhs(), addends) &&
           collectAddends(add.getRhs(), addends);
  } else {
    addends.push_back(v);
  }
  return true;
}

// Collect the leaf operands of a (possibly nested) integer mul tree.
static bool collectMulFactors(Value v, SmallVectorImpl<Value> &factors) {
  if (auto mul = v.getDefiningOp<arith::MulIOp>()) {
    if (!hasDefaultOverflowFlags(mul))
      return false;
    return collectMulFactors(mul.getLhs(), factors) &&
           collectMulFactors(mul.getRhs(), factors);
  } else {
    factors.push_back(v);
  }
  return true;
}

// One addend of the identity `(x / c) * c + (x % c) == x`, possibly scaled by a
// shared factor `k`: the "div leg" `k * (x / c) * c` or the "rem leg"
// `k * (x % c)`.
struct DivRemLeg {
  bool isDiv;                 // div leg vs rem leg
  Value x;                    // dividend
  Value c;                    // divisor
  SmallVector<Value> factors; // remaining shared factors `k`, sorted
  Type innerType;             // type before an optional sign extension
  Type addendType;            // type of the addend in the root add tree
  bool isExtended;            // whether the addend was wrapped by arith.extsi
};

static void sortByPointer(SmallVectorImpl<Value> &vals) {
  llvm::sort(vals, [](Value a, Value b) {
    return a.getAsOpaquePointer() < b.getAsOpaquePointer();
  });
}

// Try to classify `addend` as a div/rem leg. The addend may be wrapped in an
// arith.extsi; the narrow operand is analyzed while `addendType` records the
// type that must be produced for the root add tree.
static std::optional<DivRemLeg> classifyLeg(Value addend) {
  Type addendType = addend.getType();
  Value inner = addend;
  bool isExtended = false;
  if (auto ext = addend.getDefiningOp<arith::ExtSIOp>()) {
    inner = ext.getIn();
    isExtended = true;
  }
  Type innerType = inner.getType();

  SmallVector<Value> factors;
  if (!collectMulFactors(inner, factors))
    return std::nullopt;

  // Require exactly one divsi/remsi among the multiplicative factors.
  int coreIdx = -1;
  bool isDiv = false;
  for (size_t i = 0; i < factors.size(); ++i) {
    bool d = factors[i].getDefiningOp<arith::DivSIOp>() != nullptr;
    bool r = factors[i].getDefiningOp<arith::RemSIOp>() != nullptr;
    if (d || r) {
      if (coreIdx != -1)
        return std::nullopt;
      coreIdx = i;
      isDiv = d;
    }
  }
  if (coreIdx == -1)
    return std::nullopt;

  Value core = factors[coreIdx];
  Value x, c;
  if (isDiv) {
    auto divOp = core.getDefiningOp<arith::DivSIOp>();
    x = divOp.getLhs();
    c = divOp.getRhs();
  } else {
    auto remOp = core.getDefiningOp<arith::RemSIOp>();
    x = remOp.getLhs();
    c = remOp.getRhs();
  }

  SmallVector<Value> rest;
  for (size_t i = 0; i < factors.size(); ++i)
    if ((int)i != coreIdx)
      rest.push_back(factors[i]);

  // The div leg must carry an explicit `* c`; remove one occurrence of it.
  if (isDiv) {
    auto *it = llvm::find(rest, c);
    if (it == rest.end())
      return std::nullopt;
    rest.erase(it);
  }

  sortByPointer(rest);
  return DivRemLeg{isDiv, x, c, std::move(rest), innerType, addendType,
                   isExtended};
}

static bool legsMatch(const DivRemLeg &a, const DivRemLeg &b) {
  return a.x == b.x && a.c == b.c && a.innerType == b.innerType &&
         a.addendType == b.addendType && a.isExtended == b.isExtended &&
         a.factors.size() == b.factors.size() &&
         std::equal(a.factors.begin(), a.factors.end(), b.factors.begin());
}

static Value rebuildDivRemLeg(const DivRemLeg &leg, Location loc,
                              OpBuilder &builder) {
  // The unextended shared-factor form is safe in the original integer type:
  // `(k * (x / c) * c + k * (x % c)) mod 2^n == (k * x) mod 2^n`.
  //
  // If each leg has already been sign-extended, however, the source computes
  // two separate narrow products and then adds them in the wider type. Rebuilding
  // `k * x` in the wider type (or as one narrow product followed by one extsi)
  // is not equivalent when the narrow products wrap, so require an unscaled
  // extended pair unless a stronger no-overflow proof is added.
  if (leg.isExtended && !leg.factors.empty())
    return {};

  Value rebuilt = leg.x;
  if (rebuilt.getType() != leg.innerType)
    return {};

  for (Value factor : leg.factors) {
    if (factor.getType() != leg.innerType)
      return {};
    rebuilt = builder.create<arith::MulIOp>(loc, rebuilt, factor);
  }

  if (rebuilt.getType() != leg.addendType)
    rebuilt = builder.create<arith::ExtSIOp>(loc, leg.addendType, rebuilt);
  return rebuilt;
}

// Reassociate an add-root offset of the form `(x / c) * c + (x % c)` into `x`,
// including the unextended shared-factor form `k*(x / c)*c + k*(x % c)` into
// `k*x`. This is intentionally a pointer-address canonicalization helper rather
// than a module-wide arith combine.
static Value foldDivRemReconstruct(Value root, Location loc, OpBuilder &builder) {
  SmallVector<Value> addends;
  if (!collectAddends(root, addends))
    return {};
  if (addends.size() < 2)
    return {};

  SmallVector<std::optional<DivRemLeg>> legs;
  legs.reserve(addends.size());
  for (Value addend : addends)
    legs.push_back(classifyLeg(addend));

  for (size_t i = 0; i < addends.size(); ++i) {
    if (!legs[i] || !legs[i]->isDiv)
      continue;
    for (size_t j = 0; j < addends.size(); ++j) {
      if (i == j || !legs[j] || legs[j]->isDiv)
        continue;
      if (!legsMatch(*legs[i], *legs[j]))
        continue;

      Value rebuilt = rebuildDivRemLeg(*legs[i], loc, builder);
      if (!rebuilt)
        continue;

      Value sum = rebuilt;
      for (size_t k = 0; k < addends.size(); ++k) {
        if (k == i || k == j)
          continue;
        sum = builder.create<arith::AddIOp>(loc, sum, addends[k]);
      }

      return sum;
    }
  }
  return {};
}

class TritonToStructuredPass
    : public triton::impl::TritonToStructuredBase<TritonToStructuredPass> {
  using TritonToStructuredBase<TritonToStructuredPass>::TritonToStructuredBase;

  static TupleType getStructuredStateTupleType(MLIRContext *context, Type t) {
    SmallVector<Type> tupleTypes{t};
    auto [offsetTypes, strideTypes] =
        *tts::GetStructuredStateOp::getOffsetAndStrideTypes(context, t);
    tupleTypes.append(offsetTypes);
    tupleTypes.append(strideTypes);
    return TupleType::get(context, tupleTypes);
  }

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, math::MathDialect, affine::AffineDialect,
                scf::SCFDialect, tensor::TensorDialect, triton::TritonDialect,
                tts::TritonStructuredDialect>();
  }

  LogicalResult convertToPointerTupleWithOffsetsAndStrides() {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());

    auto context = &getContext();
    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });

    // We are doing a 1->1 type conversion here, where a triton pointer type
    // maps to a tuple of {pointer, offset_0, offset_1,..., stride_0,
    // stride_1,...} type.
    //
    // Case 1: Unstructured pointers (tensor<!tt.ptr<type>>)
    converter.addConversion([context](RankedTensorType tensorType,
                                      SmallVectorImpl<Type> &types)
                                -> std::optional<LogicalResult> {
      // Important note:
      // We only care about tensor of index / int (in addition to pointer type)
      // because only values of int and index type can potentially be part of a
      // pointer arithmetic sequence.
      // Specifically, Triton pointer offsets are always i32 or i64; narrower
      // integer types (e.g. i16, i8) are data values and must not be treated
      // as structured pointer offsets. This also matches the TT_IndexTensorLike
      // constraint (I32Tensor | I64Tensor) in TritonStructuredDialect.td.
      auto elementType = tensorType.getElementType();
      bool isStructuredIntTensor =
          elementType.isIndex() || elementType.isInteger(32) ||
          elementType.isInteger(64);
      if (isa<triton::PointerType>(elementType) || isStructuredIntTensor) {
        types =
            SmallVector<Type>{getStructuredStateTupleType(context, tensorType)};
        return success();
      }
      // There's a subtle difference between returning failure() and
      // std::nullopt. From the documentation:
      //
      // If std::nullopt is returned, the converter is allowed to try another
      // conversion function to perform the conversion.
      //
      // Say we have type tensor<4x256xbf16> which is a RankedTensorType. Even
      // though this RankedTensorType matches the converter that handles the
      // tuple conversion, we want to keep this type as is because the inner
      // type isn't a pointer.
      //
      // By returning failure(), the TypeConverters will stop trying the
      // remaining converters. In our case, the last type converter which
      // simply returns the same type is skipped. And because the conversion
      // for this type has failed, the whole conversion process is also
      // skipped.
      //
      // Relevant links to the implementation:
      //
      // https://github.com/llvm/llvm-project/blob/cb5dc1faa8b3702e0d03426ee5dfc5e1b903ec47/mlir/lib/Transforms/Utils/DialectConversion.cpp#L2958
      // https://github.com/llvm/llvm-project/blob/cb5dc1faa8b3702e0d03426ee5dfc5e1b903ec47/mlir/lib/Transforms/Utils/DialectConversion.cpp#L3033
      return std::nullopt;
    });

    // Case 2: Block pointers (!tt.ptr<tensor<type>> or !tt.ptr<type>)
    converter.addConversion([context](triton::PointerType ptrType,
                                      SmallVectorImpl<Type> &types)
                                -> std::optional<LogicalResult> {
      types = SmallVector<Type>{getStructuredStateTupleType(context, ptrType)};
      return success();
    });

    // Hooks to compute the correct materialization, "argument" and "source"
    // materialization are used when we need to convert the tuple type back to
    // the original triton pointer type. These are used when there are ops that
    // still need to use the original pointer type. For instance, we convert the
    // result of tt.addptr from tt.ptr type to a tuple, but the original ptr
    // result is still being used by another tt.load or tt.store.
    auto materialize = [](OpBuilder &builder, Type resultType,
                          ValueRange inputs, Location loc) {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    };

    converter.addArgumentMaterialization(materialize);
    converter.addSourceMaterialization(materialize);

    // Compute the target materialization, given a value with the pointer type,
    // convert that value to a tuple type.
    converter.addTargetMaterialization(
        [](OpBuilder &builder, TypeRange resultTypes, ValueRange input,
           Location loc) -> SmallVector<Value> {
          return builder
              .create<UnrealizedConversionCastOp>(loc, resultTypes, input.front())
              ->getResults();
        });

    scf::populateSCFStructuralOneToNTypeConversions(converter, patterns);

    if (failed(applyPartialOneToNConversion(getOperation(), converter,
                                            std::move(patterns)))) {
      return failure();
    }

    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(createCanonicalizerPass());
    if (failed(runPipeline(pm, getOperation()))) {
      return failure();
    }

    return success();
  }

  LogicalResult decomposePointerTuple() {
    auto moduleOp = getOperation();

    auto context = &getContext();
    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });

    // We are doing a 1->N type conversion here, where a pointer tuple type
    // maps to a sequence of {pointer, offset_0, offset_1,..., stride_0,
    // stride_1,...}
    converter.addConversion(
        [context](TupleType tupleType, SmallVectorImpl<Type> &types)
            -> std::optional<LogicalResult> {
          tupleType.getFlattenedTypes(types);
          return success();
        });

    // Hooks to compute the correct materialization, "argument" and "source"
    // materialization are used when we need to convert a series of {pointer,
    // offset_0, offset_1,..., stride_0, stride_1,...} type back to the "pointer
    // tuple type".
    //
    // Because we actually want to get rid of the tuple type, return `inputs[0]`
    // which corresponds to a "triton pointer type". This approach will work as
    // intended because the ops that currently take "pointer tuple type" are
    // `unrealized_conversion_cast` ops which will get removed below during
    // reconcile-unrealized-conversion-casts.
    auto materialize = [](OpBuilder &builder, Type resultType,
                          ValueRange inputs,
                          Location loc) { return inputs[0]; };
    converter.addArgumentMaterialization(materialize);
    converter.addSourceMaterialization(materialize);

    // For each value of "pointer tuple type" that gets decomposed into a
    // sequence of {pointer, offset_0, offset_1,..., stride_0, stride_1,...},
    // create a `tts.get_structured_state` op that serves as a placeholder.
    // The return values for this op will be used as the init-args for scf.for.
    // At the end of pointer analysis, we will use the PtrState to create the
    // correct offsets, strides, and remove these ops.
    converter.addTargetMaterialization([](OpBuilder &builder,
                                          TypeRange resultTypes, ValueRange input,
                                          Location loc) {
      auto placeholder = builder.create<tts::GetStructuredStateOp>(
          loc, input.front().getDefiningOp()->getOperand(0));
      assert(llvm::equal(placeholder.getResultTypes(), resultTypes));
      return placeholder.getResults();
    });

    RewritePatternSet patterns(&getContext());
    scf::populateSCFStructuralOneToNTypeConversions(converter, patterns);
    if (failed(applyPartialOneToNConversion(getOperation(), converter,
                                            std::move(patterns)))) {
      return failure();
    }

    // Note:
    // Be careful not to run canonicalization here, because the
    // tts.get_structured_state ops created above are just placeholders and
    // don't have any effects. Canonicalization will remove them altogether.
    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }

    return success();
  }

  // Prepass that inserts `tts.get_structured_state` ops. These ops are used as
  // placeholders to make passing structured pointer state into scf.for loop's
  // init args easier, especially with multiple levels of loops.
  //
  // Background:
  //
  // PtrAnalysis computes a PtrState for every operand (or triton value)
  // involved in a sequence of pointer arithmetic; some examples include: triton
  // pointer, offsets (which could be a tensor of indices or just a simple index
  // value).
  //
  // If a triton value is updated and returned in a scf.for op, it means
  // that we have to carry its offsets and strides in the scf.for's iterargs.
  //
  // Previously, we have to manually rewrite the loops to include the
  // relevant information from a PtrState which was rather involved and
  // error-prone; this was also hard to scale up to multiple level of loops
  // because there are several book-keeping data structures that we have to
  // maintain.
  //
  // With the introduction of the prepass that inserts
  // `tts.get_structured_state`. The return values of these ops, which include a
  // triton value with its original result type and its corresponding offsets
  // and strides, will be used as "placeholders" into the scf.for's init-args.
  // We leverage standard MLIR infrastructure 1->N conversion to perform this
  // rewrite, which helps simplify the logic significantly.
  //
  // After PtrAnalysis finishes, the return values of these
  // `tts.get_structured_state` ops will be remapped to the correct
  // initialization of the value's offsets and strides through the value's
  // computed PtrState.
  //
  // Implementation details:
  // In essence, what we really want to do in the prepass is, for every value
  // of triton-pointer-like type (tt.ptr or tensor<tt.ptr<>>) and tensor of
  // indices (tensor<i32>) which might be used in a sequence of pointer
  // arithmetic, we want to create an op `tts.get_structured_state` that takes
  // in the original triton value and returns a series of values:
  //
  // {triton_value, offset_0, offset_1, ..., stride_0, stride_1,...}
  //
  // Applying the above conversion will also mean that any structural ops such
  // as scf.for and scf.yield that originally takes the triton pointer will
  // then take {triton_value, offset_0, offset_1, ..., stride_0, stride_1,...}.
  //
  // The 1->N type conversion is a perfect fit for this transformation.
  // Unfortunately, we cannot do this is one pass, because the current 1->N
  // type conversion implementation for scf.for ops doesn't provide us with a
  // way to detect that a type conversion is recursive. So a triton_value type
  // that gets converted to a {triton_value, offset_0, offset_1, ..., stride_0,
  // stride_1,...} will recursively trigger other conversions.
  //
  // To fix this issue, we have to first convert triton_value to
  // tuple<triton_value, offset_0, offset_1, ..., stride_0, stride_1,...>.
  // Finally, we decompose these tuples into the desired sequence.
  //
  // Note that even though the type conversion happens for every integer tensor
  // appearing in loops' iter-args, this conversion is reversible. If the
  // integer tensor isn't used in a pointer arithmetic sequence,
  // canonicalization will remove all the `tts.get_structured_state` ops and
  // revert the IR back to its original form.
  LogicalResult runTritonToStructuredPrepass() {
    if (failed(convertToPointerTupleWithOffsetsAndStrides())) {
      return failure();
    }

    return decomposePointerTuple();
  }

  // Canonicalize reshape-style pointer offsets immediately before PtrAnalysis.
  // The transform is scoped to `tt.addptr` offsets: it builds a folded value for
  // the pointer operation without replacing the original arithmetic tree, so
  // ordinary integer users keep their exact source semantics.
  bool canonicalizePointerArithmeticBeforePtrAnalysis(ModuleOp moduleOp) {
    bool changed = false;
    OpBuilder builder(&getContext());

    moduleOp.walk([&](triton::AddPtrOp addPtrOp) {
      Value offset = addPtrOp.getOffset();
      bool localChanged = false;

      while (offset.getDefiningOp<arith::AddIOp>()) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(addPtrOp);
        Value folded =
            foldDivRemReconstruct(offset, addPtrOp.getLoc(), builder);
        if (!folded)
          break;
        offset = folded;
        localChanged = true;
      }

      if (!localChanged)
        return;

      addPtrOp->setOperand(1, offset);
      changed = true;
    });

    return changed;
  }

  void runOnOperation() override {
    if (!skipPrepass && failed(runTritonToStructuredPrepass())) {
      signalPassFailure();
      return;
    }

    if (runPrepassOnly) {
      return;
    }

    auto moduleOp = getOperation();
    (void)canonicalizePointerArithmeticBeforePtrAnalysis(moduleOp);

    mlir::tts::PtrAnalysis ptrAnalysis(enableMakeGatherScatterTensorPtr);
    ptrAnalysis.initializeMaybeStructuredArgs(moduleOp);

    if (failed(ptrAnalysis.rewriteOp(moduleOp, useUnsafeMask))) {
      moduleOp->emitWarning("PtrAnalysis failed");
    }

    // Now that all the PtrStates have been populated, we can wire up the states
    // with the tts.get_structured_state ops inserted in the prepass.
    moduleOp.walk([&ptrAnalysis](tts::GetStructuredStateOp op) {
      if (failed(ptrAnalysis.rewriteGetStructuredStateOp(op))) {
        op.emitWarning("Rewriting GetStructuredStateOp failed.");
      }
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonToStructuredPass(bool enableMakeGatherScatterTensorPtr) {
  TritonToStructuredOptions options;
  options.enableMakeGatherScatterTensorPtr = enableMakeGatherScatterTensorPtr;
  return std::make_unique<TritonToStructuredPass>(options);
}
