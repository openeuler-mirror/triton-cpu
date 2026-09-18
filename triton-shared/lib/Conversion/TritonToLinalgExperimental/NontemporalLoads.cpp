//===----------------------------------------------------------------------===//
//
// Copyright (c) OpenEuler.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//
//
// tl.load / tl.store (..., eviction_policy="evict_first") -> non-temporal
// loads and stores.
//
// Because the hint is set at the triton ir level of the pipeline and
// can only be applied at the llvm ir level, we need two passes:
//
//   triton-annotate-nontemporal-args   (Triton dialect)
//     For every tt.func, find the pointer arguments that are read *only*
//     through tt.load carrying evict_first, and separately those that
//     are written *only* through tt.store ops carrying it, and record their
//     ordinals in two function attributes.
//
//   mark-nontemporal-loads              (LLVM dialect)
//     For every llvm.func carrying those attributes, set `nontemporal` on
//     each llvm.load / llvm.store whose address derives from one of the
//     listed arguments.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/TritonToLinalgExperimental/NontemporalLoads.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallPtrSet.h"

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Pass 1: Triton dialect -> function attribute
//===----------------------------------------------------------------------===//

struct TritonPtrRoots {
  Block *entry;
  SmallPtrSet<Value, 8> visiting;

  bool collect(Value v, SmallPtrSetImpl<BlockArgument> &roots) {
    if (!visiting.insert(v).second)
      return true;

    if (auto ba = dyn_cast<BlockArgument>(v)) {
      if (ba.getOwner() == entry) {
        roots.insert(ba);
        return true;
      }
      // A loop-carried pointer: it derives from the init operand and from
      // whatever the body yields for it.
      if (auto loop = dyn_cast<LoopLikeOpInterface>(ba.getOwner()->getParentOp())) {
        OpOperand *init = loop.getTiedLoopInit(ba);
        OpOperand *yielded = loop.getTiedLoopYieldedValue(ba);
        if (!init || !yielded)
          return false;
        return collect(init->get(), roots) && collect(yielded->get(), roots);
      }
      return false;
    }

    Operation *op = v.getDefiningOp();
    if (!op)
      return false;

    // Ops that pass a pointer through unchanged, or offset it.
    if (auto o = dyn_cast<triton::AddPtrOp>(op))
      return collect(o.getPtr(), roots);
    if (auto o = dyn_cast<triton::AdvanceOp>(op))
      return collect(o.getPtr(), roots);
    if (auto o = dyn_cast<triton::MakeTensorPtrOp>(op))
      return collect(o.getBase(), roots);
    if (auto o = dyn_cast<triton::SplatOp>(op))
      return collect(o.getSrc(), roots);
    if (auto o = dyn_cast<triton::BroadcastOp>(op))
      return collect(o.getSrc(), roots);
    if (auto o = dyn_cast<triton::ExpandDimsOp>(op))
      return collect(o.getSrc(), roots);
    if (auto o = dyn_cast<triton::ReshapeOp>(op))
      return collect(o.getSrc(), roots);
    if (auto o = dyn_cast<triton::TransOp>(op))
      return collect(o.getSrc(), roots);
    if (auto o = dyn_cast<triton::BitcastOp>(op))
      return collect(o.getSrc(), roots);
    if (auto o = dyn_cast<arith::SelectOp>(op))
      return collect(o.getTrueValue(), roots) &&
             collect(o.getFalseValue(), roots);

    // Control flow that yields pointers.
    auto result = cast<OpResult>(v);
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      unsigned i = result.getResultNumber();
      return collect(ifOp.thenYield().getOperand(i), roots) &&
             collect(ifOp.elseYield().getOperand(i), roots);
    }
    if (auto loop = dyn_cast<LoopLikeOpInterface>(op)) {
      OpOperand *init = loop.getTiedLoopInit(result);
      BlockArgument iterArg = loop.getTiedLoopRegionIterArg(result);
      if (!init || !iterArg)
        return false;
      OpOperand *yielded = loop.getTiedLoopYieldedValue(iterArg);
      if (!yielded)
        return false;
      return collect(init->get(), roots) && collect(yielded->get(), roots);
    }
    return false;
  }
};

struct TritonAnnotateNontemporalArgsPass
    : public TritonAnnotateNontemporalArgsBase<
          TritonAnnotateNontemporalArgsPass> {
  void runOnOperation() override {
    getOperation().walk([&](triton::FuncOp func) { annotate(func); });
  }

  static void annotate(triton::FuncOp func) {
    if (func.getBody().empty())
      return;
    Block &entry = func.getBody().front();

    DenseMap<BlockArgument, int32_t> ordinal;
    for (BlockArgument arg : entry.getArguments()) {
      if (isa<triton::PointerType>(arg.getType())) {
        int32_t next = ordinal.size();
        ordinal[arg] = next;
      }
    }
    if (ordinal.empty())
      return;

    // Per argument and per access kind: touched at all / touched with
    // anything but evict_first.  Loads and stores are judged independently.
    struct Kind {
      DenseSet<BlockArgument> any, temporal;
    } loads, stores;
    bool traced = true;
    auto note = [&](Value ptr, triton::EvictionPolicy evict, bool masked,
                    Kind &kind) {
      TritonPtrRoots tracer{&entry, {}};
      SmallPtrSet<BlockArgument, 4> roots;
      if (!tracer.collect(ptr, roots)) {
        traced = false;
        return;
      }
      bool evictFirst = evict == triton::EvictionPolicy::EVICT_FIRST && !masked;
      for (BlockArgument r : roots) {
        if (!ordinal.count(r)) {
          traced = false;
          return;
        }
        kind.any.insert(r);
        if (!evictFirst)
          kind.temporal.insert(r);
      }
    };
    func.walk([&](Operation *op) {
      if (!traced)
        return;
      if (auto ld = dyn_cast<triton::LoadOp>(op))
        note(ld.getPtr(), ld.getEvict(), ld.getMask() != nullptr, loads);
      else if (auto st = dyn_cast<triton::StoreOp>(op))
        note(st.getPtr(), st.getEvict(),  st.getMask() != nullptr, stores);
    });
    if (!traced)
      return;

    auto streaming = [&](const Kind &kind) {
      SmallVector<int32_t> out;
      for (auto [arg, i] : ordinal)
        if (kind.any.contains(arg) && !kind.temporal.contains(arg))
          out.push_back(i);
      llvm::sort(out);
      return out;
    };
    SmallVector<int32_t> loadArgs = streaming(loads);
    SmallVector<int32_t> storeArgs = streaming(stores);
    if (loadArgs.empty() && storeArgs.empty())
      return;

    OpBuilder b(func.getContext());
    if (!loadArgs.empty())
      func->setAttr(triton::kNontemporalLoadArgsAttr,
                    b.getDenseI32ArrayAttr(loadArgs));
    if (!storeArgs.empty())
      func->setAttr(triton::kNontemporalStoreArgsAttr,
                    b.getDenseI32ArrayAttr(storeArgs));
    func->setAttr(triton::kPtrArgCountAttr,
                  b.getI32IntegerAttr(ordinal.size()));
  }
};

//===----------------------------------------------------------------------===//
// Pass 2: function attributes -> llvm.load / llvm.store nontemporal
//===----------------------------------------------------------------------===//

struct LLVMPtrRoots {
  Block *entry;
  const SmallPtrSetImpl<BlockArgument> &targets;
  SmallPtrSet<Value, 8> visiting;

  bool allIn(Value v) {
    if (!visiting.insert(v).second)
      return true;

    if (auto ba = dyn_cast<BlockArgument>(v)) {
      if (ba.getOwner() == entry)
        return targets.contains(ba);
      Block *block = ba.getOwner();
      bool any = false;
      for (Block *pred : block->getPredecessors()) {
        auto br = dyn_cast<BranchOpInterface>(pred->getTerminator());
        if (!br)
          return false;
        for (unsigned s = 0, e = pred->getNumSuccessors(); s < e; ++s) {
          if (pred->getSuccessor(s) != block)
            continue;
          Value in = br.getSuccessorOperands(s)[ba.getArgNumber()];
          if (!in || !allIn(in))
            return false;
          any = true;
        }
      }
      return any;
    }

    Operation *op = v.getDefiningOp();
    if (!op)
      return false;
    if (auto o = dyn_cast<LLVM::GEPOp>(op))
      return allIn(o.getBase());
    if (auto o = dyn_cast<LLVM::BitcastOp>(op))
      return allIn(o.getArg());
    if (auto o = dyn_cast<LLVM::AddrSpaceCastOp>(op))
      return allIn(o.getArg());
    if (auto o = dyn_cast<LLVM::IntToPtrOp>(op)) {
      if (auto p = o.getArg().getDefiningOp<LLVM::PtrToIntOp>())
        return allIn(p.getArg());
      return false;
    }
    if (auto o = dyn_cast<LLVM::ExtractValueOp>(op))
      return allIn(o.getContainer());
    if (auto o = dyn_cast<LLVM::SelectOp>(op))
      return allIn(o.getTrueValue()) && allIn(o.getFalseValue());
    if (auto o = dyn_cast<LLVM::LoadOp>(op))
      return isa<LLVM::LLVMPointerType>(o.getType()) && allIn(o.getAddr());
    return false;
  }
};

struct MarkNontemporalLoadsPass
    : public MarkNontemporalLoadsBase<MarkNontemporalLoadsPass> {
  void runOnOperation() override {
    getOperation().walk([&](LLVM::LLVMFuncOp func) { mark(func); });
  }

  static void mark(LLVM::LLVMFuncOp func) {
    auto loadArgs =
        func->getAttrOfType<DenseI32ArrayAttr>(triton::kNontemporalLoadArgsAttr);
    auto storeArgs = func->getAttrOfType<DenseI32ArrayAttr>(
        triton::kNontemporalStoreArgsAttr);
    auto count = func->getAttrOfType<IntegerAttr>(triton::kPtrArgCountAttr);
    func->removeAttr(triton::kNontemporalLoadArgsAttr);
    func->removeAttr(triton::kNontemporalStoreArgsAttr);
    func->removeAttr(triton::kPtrArgCountAttr);
    if ((!loadArgs && !storeArgs) || func.getBody().empty())
      return;
    Block &entry = func.getBody().front();

    SmallVector<BlockArgument> ptrParams;
    for (BlockArgument arg : entry.getArguments())
      if (isa<LLVM::LLVMPointerType>(arg.getType()))
        ptrParams.push_back(arg);

    if (count && count.getInt() != static_cast<int64_t>(ptrParams.size())) {
      func.emitWarning() << "nontemporal argument annotations ignored: expected "
                         << count.getInt() << " pointer parameters, found "
                         << ptrParams.size();
      return;
    }

    auto targetsOf = [&](DenseI32ArrayAttr ordinals) {
      SmallPtrSet<BlockArgument, 4> targets;
      if (ordinals)
        for (int32_t i : ordinals.asArrayRef())
          if (i >= 0 && static_cast<size_t>(i) < ptrParams.size())
            targets.insert(ptrParams[i]);
      return targets;
    };
    SmallPtrSet<BlockArgument, 4> loadTargets = targetsOf(loadArgs);
    SmallPtrSet<BlockArgument, 4> storeTargets = targetsOf(storeArgs);

    func.walk([&](Operation *op) {
      if (auto load = dyn_cast<LLVM::LoadOp>(op)) {
        if (loadTargets.empty() || isa<LLVM::LLVMPointerType>(load.getType()))
          return;
        LLVMPtrRoots tracer{&entry, loadTargets, {}};
        if (tracer.allIn(load.getAddr()))
          load.setNontemporal(true);
      } else if (auto store = dyn_cast<LLVM::StoreOp>(op)) {
        if (storeTargets.empty())
          return;
        LLVMPtrRoots tracer{&entry, storeTargets, {}};
        if (tracer.allIn(store.getAddr()))
          store.setNontemporal(true);
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createTritonAnnotateNontemporalArgsPass() {
  return std::make_unique<TritonAnnotateNontemporalArgsPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createMarkNontemporalLoadsPass() {
  return std::make_unique<MarkNontemporalLoadsPass>();
}
