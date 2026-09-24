//===----------------------------------------------------------------------===//
// Double-buffer loop-carried allocations
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/TritonToLinalgExperimental/DoubleBufferLoopAllocations.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"

using namespace mlir;
using namespace triton;

namespace {

static bool isSmallStaticMemRef(MemRefType type) {
  constexpr int64_t maxBytes = 16 * 1024;
  if (!type.hasStaticShape() || !type.getLayout().isIdentity())
    return false;
  Type elementType = type.getElementType();
  if (!elementType.isIntOrFloat())
    return false;
  // Round sub-byte elements up and avoid overflowing the size calculation.
  int64_t bytes = (elementType.getIntOrFloatBitWidth() + 7) / 8;
  for (int64_t dim : type.getShape()) {
    if (dim && bytes > maxBytes / dim)
      return false;
    bytes *= dim;
  }
  return bytes <= maxBytes;
}

// Keep accesses in the loop body itself. Nested regions may execute
// asynchronously or otherwise outlive the current iteration.
static bool hasOnlyLocalAccesses(Value buffer, Block *body,
                                 scf::YieldOp allowedYield = {}) {
  for (OpOperand &use : buffer.getUses()) {
    Operation *user = use.getOwner();
    if (user->getBlock() != body)
      return false;
    if (allowedYield && user == allowedYield.getOperation())
      continue;
    if (!isa<memref::CopyOp, memref::LoadOp, memref::StoreOp,
             vector::TransferReadOp, vector::TransferWriteOp>(user))
      return false;
  }
  return true;
}

static bool isReusableAllocation(memref::AllocOp alloc, scf::ForOp loop,
                                 scf::YieldOp yield) {
  if (alloc->getBlock() != loop.getBody() ||
      !isSmallStaticMemRef(alloc.getType()) ||
      !hasOnlyLocalAccesses(alloc.getResult(), loop.getBody(), yield))
    return false;

  // The next-iteration state must not retain aliases or forward the buffer
  // into another iter_arg. Otherwise two iterations may not span its lifetime.
  unsigned occurrences = 0;
  for (unsigned i = 0; i < yield.getNumOperands(); ++i) {
    if (yield.getOperand(i) != alloc.getResult())
      continue;
    ++occurrences;
    if (!hasOnlyLocalAccesses(loop.getRegionIterArgs()[i], loop.getBody()))
      return false;
  }
  return occurrences == 1;
}

// scf.if is synchronous. Do not extend this to arbitrary region-bearing ops:
// an async region or a nested loop can extend a buffer's lifetime.
static bool hasOnlyConditionalAccesses(Value state, Block *loopBody) {
  for (Operation *user : state.getUsers()) {
    if (!isa<memref::CopyOp, memref::LoadOp, memref::StoreOp,
             vector::TransferReadOp, vector::TransferWriteOp>(user))
      return false;
    Operation *ancestor = user;
    while (ancestor->getBlock() != loopBody) {
      ancestor = ancestor->getParentOp();
      if (!ancestor || !isa<scf::IfOp>(ancestor))
        return false;
    }
  }
  return true;
}

// Every branch must yield a fresh allocation. In particular, forwarding the
// previous state on one branch is unsafe: that buffer can then survive more
// than one iteration, invalidating parity-based reuse.
static bool
collectConditionalAllocations(Value value, scf::YieldOp yield,
                              SmallVectorImpl<memref::AllocOp> &out) {
  Operation *def = value.getDefiningOp();
  if (!def || def->getBlock() != yield->getBlock())
    return false;
  if (auto alloc = dyn_cast<memref::AllocOp>(def)) {
    if (!isSmallStaticMemRef(alloc.getType()) ||
        !hasOnlyLocalAccesses(value, yield->getBlock(), yield))
      return false;
    unsigned count = 0;
    for (Value operand : yield.getOperands())
      count += operand == value;
    if (count != 1)
      return false;
    out.push_back(alloc);
    return true;
  }
  auto ifOp = dyn_cast<scf::IfOp>(def);
  if (!ifOp || !value.hasOneUse() || ifOp.getElseRegion().empty())
    return false;
  unsigned index = cast<OpResult>(value).getResultNumber();
  auto thenYield =
      cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
  auto elseYield =
      cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());
  return collectConditionalAllocations(thenYield.getOperand(index), thenYield,
                                       out) &&
         collectConditionalAllocations(elseYield.getOperand(index), elseYield,
                                       out);
}

static Value createIntegerLikeConstant(OpBuilder &builder, Location loc,
                                       Type type, int64_t value) {
  if (auto integerType = dyn_cast<IntegerType>(type))
    return builder.create<arith::ConstantIntOp>(loc, value, integerType);
  return builder.create<arith::ConstantIndexOp>(loc, value);
}

// A single scratch buffer suffices if all reads of the previous iteration's
// state finish before any access to the newly allocated buffer. Keep copies
// and writes to the old state on the double-buffer path: they may overlap.
static bool canUseSingleBuffer(memref::AllocOp alloc, scf::ForOp loop,
                               scf::YieldOp yield) {
  unsigned position = 0;
  while (yield.getOperand(position) != alloc.getResult())
    ++position;
  Value state = loop.getRegionIterArgs()[position];
  for (Operation *read : state.getUsers()) {
    if (!isa<memref::LoadOp, vector::TransferReadOp>(read))
      return false;
    for (Operation *access : alloc->getUsers()) {
      if (access == yield.getOperation())
        continue;
      if (!read->isBeforeInBlock(access))
        return false;
    }
  }
  return true;
}

struct DoubleBufferLoopAllocationsPass
    : public DoubleBufferLoopAllocationsBase<DoubleBufferLoopAllocationsPass> {
  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

    IRRewriter rewriter(&getContext());
    for (scf::ForOp forOp : llvm::reverse(loops)) {
      Type inductionType = forOp.getInductionVar().getType();
      if (!isa<IntegerType, IndexType>(inductionType))
        continue;

      SmallVector<memref::AllocOp> allocs;
      auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      for (Value yielded : yieldOp.getResults()) {
        auto alloc = yielded.getDefiningOp<memref::AllocOp>();
        if (!alloc || !isReusableAllocation(alloc, forOp, yieldOp) ||
            llvm::is_contained(allocs, alloc))
          continue;
        allocs.push_back(alloc);
      }
      for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
        Value yielded = yieldOp.getOperand(i);
        if (!yielded.getDefiningOp<scf::IfOp>() ||
            !hasOnlyConditionalAccesses(forOp.getRegionIterArgs()[i],
                                        forOp.getBody()))
          continue;
        SmallVector<memref::AllocOp> conditionalAllocs;
        if (collectConditionalAllocations(yielded, yieldOp, conditionalAllocs))
          allocs.append(conditionalAllocs.begin(), conditionalAllocs.end());
      }

      for (memref::AllocOp alloc : allocs) {
        rewriter.setInsertionPoint(forOp);
        auto first = rewriter.create<memref::AllocOp>(
            alloc.getLoc(), alloc.getType(), ValueRange{}, ValueRange{},
            alloc.getAlignmentAttr());
        if (alloc->getBlock() == forOp.getBody() &&
            canUseSingleBuffer(alloc, forOp, yieldOp)) {
          rewriter.replaceOp(alloc, first.getResult());
          continue;
        }
        auto second = rewriter.create<memref::AllocOp>(
            alloc.getLoc(), alloc.getType(), ValueRange{}, ValueRange{},
            alloc.getAlignmentAttr());

        rewriter.setInsertionPoint(alloc);
        Value distance = rewriter.create<arith::SubIOp>(
            alloc.getLoc(), forOp.getInductionVar(), forOp.getLowerBound());
        Value iteration = rewriter.create<arith::DivUIOp>(
            alloc.getLoc(), distance, forOp.getStep());
        Value two = createIntegerLikeConstant(rewriter, alloc.getLoc(),
                                              inductionType, 2);
        Value parity =
            rewriter.create<arith::RemUIOp>(alloc.getLoc(), iteration, two);
        Value zero = createIntegerLikeConstant(rewriter, alloc.getLoc(),
                                               inductionType, 0);
        Value useFirst = rewriter.create<arith::CmpIOp>(
            alloc.getLoc(), arith::CmpIPredicate::eq, parity, zero);
        Value selected = rewriter.create<arith::SelectOp>(
            alloc.getLoc(), useFirst, first, second);
        rewriter.replaceOp(alloc, selected);
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createDoubleBufferLoopAllocationsPass() {
  return std::make_unique<DoubleBufferLoopAllocationsPass>();
}
