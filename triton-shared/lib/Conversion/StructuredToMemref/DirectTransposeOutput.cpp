// Rewrites eligible buffered transposes to write directly to their outputs.
#include "triton-shared/Conversion/StructuredToMemref/DirectTransposeOutput.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::triton {
#define GEN_PASS_DEF_DIRECTTRANSPOSEOUTPUT
#include "triton-shared/Conversion/StructuredToMemref/Passes.h.inc"

namespace {
// Both bufferization copy spellings are accepted; unmatched fallbacks stay intact.
class BufferCopy {
public:
  BufferCopy() = default;
  static BufferCopy get(Operation *operation) {
    if (isa<memref::CopyOp>(operation))
      return BufferCopy(operation);
    auto copy = dyn_cast<linalg::CopyOp>(operation);
    if (!copy || copy->getNumResults() != 0 ||
        copy.getInputs().size() != 1 || copy.getOutputs().size() != 1)
      return {};
    auto source = dyn_cast<MemRefType>(copy.getInputs().front().getType());
    auto target = dyn_cast<MemRefType>(copy.getOutputs().front().getType());
    if (!source || !target || source.getRank() != target.getRank() ||
        source.getElementType() != target.getElementType())
      return {};
    return BufferCopy(operation);
  }
  explicit operator bool() const { return operation != nullptr; }
  operator Operation *() const { return operation; }
  Operation *operator->() const { return operation; }
  Operation *getOperation() const { return operation; }
  Location getLoc() const { return operation->getLoc(); }
  Value getSource() const {
    if (auto copy = dyn_cast<memref::CopyOp>(operation))
      return copy.getSource();
    return cast<linalg::CopyOp>(operation).getInputs().front();
  }
  Value getTarget() const {
    if (auto copy = dyn_cast<memref::CopyOp>(operation))
      return copy.getTarget();
    return cast<linalg::CopyOp>(operation).getOutputs().front();
  }

private:
  explicit BufferCopy(Operation *operation) : operation(operation) {}
  Operation *operation = nullptr;
};

struct Chain {
  memref::AllocOp inputAlloc, outputAlloc;
  memref::SubViewOp inputView, stagedView, outputView, destinationView;
  BufferCopy stageCopy, finalCopy;
  bufferization::ToTensorOp toTensor;
  tensor::EmptyOp empty;
  linalg::TransposeOp transpose;
  bufferization::MaterializeInDestinationOp materialize;
  linalg::FillOp inputFill;
  memref::DeallocOp inputDealloc, outputDealloc;
};

static bool staticRowMajor(MemRefType type) {
  // Require a static, contiguous two-dimensional scratch layout.

  if (!type || type.getRank() != 2 || llvm::any_of(type.getShape(), ShapedType::isDynamic))
    return false;
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset)) || offset != 0 || strides.size() != 2)
    return false;
  return strides[0] == type.getShape()[1] && strides[1] == 1;

}

static bool zeroUnitSubview(memref::SubViewOp view) {
  // Accept only zero-offset, unit-stride views of a complete tile.

  auto type = dyn_cast<MemRefType>(view.getResult().getType());
  if (!type || type.getRank() != 2 || !view.getOffsets().empty() || !view.getStrides().empty())
    return false;
  return llvm::all_of(view.getStaticOffsets(), [](int64_t value) { return value == 0; }) &&
         llvm::all_of(view.getStaticStrides(), [](int64_t value) { return value == 1; });

}

static bool sameSubviewExtent(memref::SubViewOp first,
                              memref::SubViewOp second) {
  // Require matching staged view extents before bypassing materialization.

  return first.getStaticSizes() == second.getStaticSizes() &&
         llvm::equal(first.getSizes(), second.getSizes());

}

static bool sole(Value value, Operation *user) {
  // Reject observable aliases by requiring the expected unique consumer.

  return value.hasOneUse() && *value.user_begin() == user;

}

static bool sameBlockInOrder(ArrayRef<Operation *> operations) {
  // Keep the rewrite within one block and preserve producer ordering.

  if (operations.empty())
    return false;
  Block *block = operations.front()->getBlock();
  for (Operation *operation : operations)
    if (!operation || operation->getBlock() != block)
      return false;
  for (auto [first, second] :
       llvm::zip(operations.drop_back(), operations.drop_front()))
    if (!first->isBeforeInBlock(second))
      return false;
  return true;

}

// The buffered rewrite writes the destination before the original final copy.
// Unknown effects, including reads through a base alias, must not be crossed.
static bool canMoveOutputBefore(Operation *first, Operation *last,
                                ArrayRef<Operation *> movedChain) {
  if (first->getBlock() != last->getBlock() || !first->isBeforeInBlock(last))
    return false;
  for (Operation *op = first; op != last; op = op->getNextNode())
    if (!llvm::is_contained(movedChain, op) && !isMemoryEffectFree(op))
      return false;
  return true;
}

static Value appendFullTileChecks(IRRewriter &rewriter, Location loc,
                                  Value view, ArrayRef<int64_t> shape,
                                  Value condition) {
  // Guard the direct path with runtime checks for complete static tiles.

  for (auto [index, size] : llvm::enumerate(shape)) {
    Value actual = rewriter.create<memref::DimOp>(loc, view, index);
    Value expected = rewriter.create<arith::ConstantIndexOp>(loc, size);
    Value equal = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, actual, expected);
    condition = rewriter.create<arith::AndIOp>(loc, condition, equal);
  }
  return condition;

}

static bool isFullVectorRead(vector::TransferReadOp read,
                             MemRefType sourceType) {
  // Recognize only unmasked identity reads of the whole scratch tile.

  VectorType vectorType = read.getVectorType();
  if (vectorType.getShape() != sourceType.getShape() ||
      vectorType.getElementType() != sourceType.getElementType() ||
      !read.getPermutationMap().isIdentity() || read.getMask())
    return false;
  return llvm::all_of(read.getIndices(),
                      [](Value index) { return matchPattern(index, m_Zero()); });

}

static bool isFullVectorWrite(vector::TransferWriteOp write,
                              MemRefType targetType) {
  // Recognize only unmasked identity writes of the whole scratch tile.

  VectorType vectorType = write.getVectorType();
  if (vectorType.getShape() != targetType.getShape() ||
      vectorType.getElementType() != targetType.getElementType() ||
      !write.getPermutationMap().isIdentity() || write.getMask())
    return false;
  return llvm::all_of(write.getIndices(),
                      [](Value index) { return matchPattern(index, m_Zero()); });

}

struct CurrentBufferizedChain {
  scf::IfOp inputIf, outputIf;
  Operation *inputAlloc = nullptr;
  Operation *transposeAlloc = nullptr;
  vector::TransferReadOp inputRead, transposeRead;
  vector::TransferWriteOp inputWrite, paddingWrite, transposeWrite;
  memref::SubViewOp inputView, stagedView, outputView, destinationView;
  BufferCopy stageCopy, fullCopy, tailCopy;
  memref::ReinterpretCastOp destinationParent;
  memref::DeallocOp inputDealloc, transposeDealloc;
  vector::TransposeOp transpose;
};

static bool collectRegionOps(Region &region,
                             SmallVectorImpl<Operation *> &operations) {
  if (region.empty() || !llvm::hasSingleElement(region))
    return false;
  Block &block = region.front();
  Operation *terminator = block.getTerminator();
  if (!terminator || !isa<scf::YieldOp>(terminator))
    return false;
  for (Operation &operation : block)
    if (&operation != terminator)
      operations.push_back(&operation);
  return true;
}

static Operation *scratchOp(Value value) {
  if (!value)
    return nullptr;
  if (auto alloc = value.getDefiningOp<memref::AllocOp>())
    return alloc.getOperation();
  if (auto allocaOp = value.getDefiningOp<memref::AllocaOp>())
    return allocaOp.getOperation();
  return nullptr;
}

static MemRefType scratchMemrefType(Operation *op) {
  if (!op || op->getNumResults() != 1)
    return {};
  return dyn_cast<MemRefType>(op->getResult(0).getType());
}

static bool sameBlockInOrderPresent(ArrayRef<Operation *> operations) {
  SmallVector<Operation *> present;
  for (Operation *operation : operations)
    if (operation)
      present.push_back(operation);
  return sameBlockInOrder(present);
}

static bool resultlessIfWithThen(scf::IfOp op) {
  return op && op.getNumResults() == 0 && !op.getThenRegion().empty() &&
         llvm::hasSingleElement(op.getThenRegion());
}

static bool resultlessTwoRegionIf(scf::IfOp op) {
  return resultlessIfWithThen(op) && !op.getElseRegion().empty() &&
         llvm::hasSingleElement(op.getElseRegion());
}

static bool staticShapeUnitInnerStride(MemRefType type,
                                       ArrayRef<int64_t> expectedShape,
                                       Type elementType) {
  if (!type || type.getRank() != 2 || type.getShape() != expectedShape ||
      type.getElementType() != elementType ||
      llvm::any_of(type.getShape(), ShapedType::isDynamic))
    return false;
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset)) || strides.size() != 2)
    return false;
  return strides[1] == 1;
}

struct VersionedFullTileChain {
  scf::IfOp branch;
  Operation *inputAlloc = nullptr;
  Operation *transposeAlloc = nullptr;
  vector::TransferReadOp transposeRead;
  vector::TransferWriteOp inputWrite, transposeWrite;
  vector::TransposeOp transpose;
  BufferCopy finalCopy;
  memref::ReinterpretCastOp destination;
  memref::DeallocOp inputDealloc, transposeDealloc;
};

static FailureOr<VersionedFullTileChain>
matchVersionedFullTileCopy(BufferCopy finalCopy) {
  VersionedFullTileChain chain;
  chain.finalCopy = finalCopy;
  chain.branch = dyn_cast<scf::IfOp>(finalCopy->getParentOp());
  if (!resultlessTwoRegionIf(chain.branch) ||
      !chain.branch->hasAttr("tts.full_tile_versioned") ||
      finalCopy->getBlock() != &chain.branch.getThenRegion().front())
    return failure();

  chain.transposeAlloc = scratchOp(finalCopy.getSource());
  chain.destination =
      finalCopy.getTarget().getDefiningOp<memref::ReinterpretCastOp>();
  if (!chain.transposeAlloc || !chain.destination ||
      finalCopy.getSource() != chain.transposeAlloc->getResult(0) ||
      finalCopy.getTarget() != chain.destination.getResult() ||
      !sole(chain.destination.getResult(), finalCopy))
    return failure();

  for (Operation *user : chain.transposeAlloc->getUsers()) {
    if (user == finalCopy.getOperation())
      continue;
    if (auto write = dyn_cast<vector::TransferWriteOp>(user)) {
      if (chain.transposeWrite ||
          write.getSource() != chain.transposeAlloc->getResult(0))
        return failure();
      chain.transposeWrite = write;
      continue;
    }
    if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
      if (chain.transposeDealloc)
        return failure();
      chain.transposeDealloc = dealloc;
      continue;
    }
    return failure();
  }
  auto transposeType = scratchMemrefType(chain.transposeAlloc);
  if (!chain.transposeWrite || !staticRowMajor(transposeType) ||
      !isFullVectorWrite(chain.transposeWrite, transposeType))
    return failure();

  chain.transpose = chain.transposeWrite.getVector()
                        .getDefiningOp<vector::TransposeOp>();
  if (!chain.transpose ||
      chain.transpose.getPermutation() != ArrayRef<int64_t>{1, 0} ||
      !sole(chain.transpose.getResult(), chain.transposeWrite.getOperation()))
    return failure();
  chain.transposeRead = chain.transpose.getVector()
                            .getDefiningOp<vector::TransferReadOp>();
  if (!chain.transposeRead ||
      !sole(chain.transposeRead.getResult(), chain.transpose.getOperation()))
    return failure();

  chain.inputAlloc = scratchOp(chain.transposeRead.getSource());
  auto inputType = scratchMemrefType(chain.inputAlloc);
  if (!chain.inputAlloc || !staticRowMajor(inputType) ||
      chain.transposeRead.getSource() != chain.inputAlloc->getResult(0) ||
      !isFullVectorRead(chain.transposeRead, inputType) ||
      inputType.getElementType() != transposeType.getElementType() ||
      inputType.getShape()[0] != transposeType.getShape()[1] ||
      inputType.getShape()[1] != transposeType.getShape()[0])
    return failure();

  auto destinationType =
      dyn_cast<MemRefType>(chain.destination.getResult().getType());
  if (!staticShapeUnitInnerStride(destinationType, transposeType.getShape(),
                                  transposeType.getElementType()))
    return failure();

  for (Operation *user : chain.inputAlloc->getUsers()) {
    if (user == chain.transposeRead.getOperation())
      continue;
    if (auto write = dyn_cast<vector::TransferWriteOp>(user)) {
      if (chain.inputWrite ||
          write.getSource() != chain.inputAlloc->getResult(0))
        return failure();
      chain.inputWrite = write;
      continue;
    }
    if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
      if (chain.inputDealloc)
        return failure();
      chain.inputDealloc = dealloc;
      continue;
    }
    return failure();
  }
  if (!chain.inputWrite || !isFullVectorWrite(chain.inputWrite, inputType) ||
      !sameBlockInOrderPresent(SmallVector<Operation *>{
          chain.inputAlloc, chain.inputWrite.getOperation(),
          chain.transposeRead.getOperation(),
          chain.transpose.getOperation(), chain.transposeWrite.getOperation(),
          chain.destination.getOperation(), chain.finalCopy.getOperation(),
          chain.inputDealloc ? chain.inputDealloc.getOperation() : nullptr,
          chain.transposeDealloc ? chain.transposeDealloc.getOperation()
                                 : nullptr}))
    return failure();
  // The independent transpose allocation may follow the vector snapshot.
  // Keep its initialization ordered without constraining unrelated reads.
  if (!sameBlockInOrder({chain.inputWrite.getOperation(), chain.transposeAlloc,
                         chain.transposeWrite.getOperation()}))
    return failure();
  return chain;
}

static void rewriteVersionedFullTileCopy(VersionedFullTileChain chain,
                                         IRRewriter &rewriter) {
  Location loc = chain.finalCopy.getLoc();
  auto outputShape = scratchMemrefType(chain.transposeAlloc).getShape();
  rewriter.setInsertionPoint(chain.finalCopy);
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value rows = rewriter.create<arith::ConstantIndexOp>(loc, outputShape[0]);
  Value cols = rewriter.create<arith::ConstantIndexOp>(loc, outputShape[1]);
  auto outer = rewriter.create<scf::ForOp>(loc, zero, rows, one);
  rewriter.setInsertionPointToStart(outer.getBody());
  auto inner = rewriter.create<scf::ForOp>(loc, zero, cols, one);
  rewriter.setInsertionPointToStart(inner.getBody());
  Value value = rewriter.create<memref::LoadOp>(
      loc, chain.inputAlloc->getResult(0),
      ValueRange{inner.getInductionVar(), outer.getInductionVar()});
  rewriter.create<memref::StoreOp>(
      loc, value, chain.destination,
      ValueRange{outer.getInductionVar(), inner.getInductionVar()});

  rewriter.eraseOp(chain.finalCopy);
  rewriter.eraseOp(chain.transposeWrite);
  rewriter.eraseOp(chain.transpose);
  rewriter.eraseOp(chain.transposeRead);
  if (chain.transposeDealloc)
    rewriter.eraseOp(chain.transposeDealloc);
  rewriter.eraseOp(chain.transposeAlloc);
}

static FailureOr<CurrentBufferizedChain>
matchCurrentBufferizedCopy(BufferCopy fullCopy) {
  CurrentBufferizedChain chain;
  chain.fullCopy = fullCopy;

  chain.outputIf = dyn_cast<scf::IfOp>(fullCopy->getParentOp());
  if (!resultlessTwoRegionIf(chain.outputIf) ||
      fullCopy->getBlock() != &chain.outputIf.getThenRegion().front())
    return failure();

  SmallVector<Operation *> outputThen, outputElse;
  if (!collectRegionOps(chain.outputIf.getThenRegion(), outputThen) ||
      !collectRegionOps(chain.outputIf.getElseRegion(), outputElse) ||
      outputThen.size() != 1 || outputThen.front() != fullCopy.getOperation() ||
      outputElse.size() != 3)
    return failure();

  chain.outputView = dyn_cast<memref::SubViewOp>(outputElse[0]);
  chain.destinationView = dyn_cast<memref::SubViewOp>(outputElse[1]);
  chain.tailCopy = BufferCopy::get(outputElse[2]);
  if (!chain.outputView || !chain.destinationView || !chain.tailCopy ||
      !zeroUnitSubview(chain.outputView) ||
      !zeroUnitSubview(chain.destinationView) ||
      !sole(chain.outputView.getResult(), chain.tailCopy) ||
      !sole(chain.destinationView.getResult(), chain.tailCopy) ||
      chain.tailCopy.getSource() != chain.outputView.getResult() ||
      chain.tailCopy.getTarget() != chain.destinationView.getResult())
    return failure();

  chain.transposeAlloc = scratchOp(fullCopy.getSource());
  chain.destinationParent =
      fullCopy.getTarget().getDefiningOp<memref::ReinterpretCastOp>();
  if (!chain.transposeAlloc || !chain.destinationParent ||
      fullCopy.getSource() != chain.transposeAlloc->getResult(0) ||
      fullCopy.getTarget() != chain.destinationParent.getResult())
    return failure();

  unsigned destinationUsers = 0;
  for (Operation *user : chain.destinationParent->getUsers()) {
    if (user == fullCopy.getOperation() ||
        user == chain.destinationView.getOperation()) {
      ++destinationUsers;
      continue;
    }
    return failure();
  }
  if (destinationUsers != 2)
    return failure();

  for (Operation *user : chain.transposeAlloc->getUsers()) {
    if (user == fullCopy.getOperation())
      continue;
    if (auto write = dyn_cast<vector::TransferWriteOp>(user)) {
      if (chain.transposeWrite ||
          write.getSource() != chain.transposeAlloc->getResult(0))
        return failure();
      chain.transposeWrite = write;
      continue;
    }
    if (user == chain.outputView.getOperation())
      continue;
    if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
      if (chain.transposeDealloc)
        return failure();
      chain.transposeDealloc = dealloc;
      continue;
    }
    return failure();
  }
  if (!chain.transposeWrite ||
      !isFullVectorWrite(chain.transposeWrite,
                         scratchMemrefType(chain.transposeAlloc)))
    return failure();

  chain.transpose = chain.transposeWrite.getVector()
                        .getDefiningOp<vector::TransposeOp>();
  if (!chain.transpose ||
      chain.transpose.getPermutation() != ArrayRef<int64_t>{1, 0} ||
      !sole(chain.transpose.getResult(), chain.transposeWrite.getOperation()))
    return failure();
  chain.transposeRead = chain.transpose.getVector()
                            .getDefiningOp<vector::TransferReadOp>();
  if (!chain.transposeRead ||
      !sole(chain.transposeRead.getResult(), chain.transpose.getOperation()))
    return failure();
  chain.inputAlloc = scratchOp(chain.transposeRead.getSource());
  if (!chain.inputAlloc || !staticRowMajor(scratchMemrefType(chain.inputAlloc)) ||
      chain.transposeRead.getSource() != chain.inputAlloc->getResult(0) ||
      !isFullVectorRead(chain.transposeRead, scratchMemrefType(chain.inputAlloc)))
    return failure();

  auto inputShape = scratchMemrefType(chain.inputAlloc).getShape();
  auto outputShape = scratchMemrefType(chain.transposeAlloc).getShape();
  if (inputShape.size() != 2 || outputShape.size() != 2 ||
      inputShape[0] != outputShape[1] || inputShape[1] != outputShape[0])
    return failure();
  auto transposeType = scratchMemrefType(chain.transposeAlloc);
  if (!staticRowMajor(transposeType))
    return failure();
  auto destinationType =
      dyn_cast<MemRefType>(chain.destinationParent.getResult().getType());
  if (!staticShapeUnitInnerStride(destinationType, outputShape,
                                  transposeType.getElementType()))
    return failure();
  if (transposeType.getElementType() !=
          scratchMemrefType(chain.inputAlloc).getElementType() ||
      !sameSubviewExtent(chain.outputView, chain.destinationView))
    return failure();

  for (Operation *user : chain.inputAlloc->getUsers()) {
    if (user == chain.transposeRead.getOperation())
      continue;
    if (auto write = dyn_cast<vector::TransferWriteOp>(user)) {
      if (write.getSource() != chain.inputAlloc->getResult(0))
        return failure();
      auto parentIf = dyn_cast<scf::IfOp>(write->getParentOp());
      if (parentIf &&
          parentIf->getBlock() == chain.outputIf->getBlock() &&
          write->getBlock() == &parentIf.getThenRegion().front()) {
        if (chain.inputWrite)
          return failure();
        chain.inputWrite = write;
        chain.inputIf = parentIf;
      } else {
        if (chain.paddingWrite)
          return failure();
        chain.paddingWrite = write;
      }
      continue;
    }
    if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
      if (chain.inputDealloc)
        return failure();
      chain.inputDealloc = dealloc;
      continue;
    }
    if (auto view = dyn_cast<memref::SubViewOp>(user)) {
      if (chain.stagedView || !view.getResult().hasOneUse())
        return failure();
      auto stage = BufferCopy::get(*view.getResult().user_begin());
      if (!stage || stage.getTarget() != view.getResult() || chain.stageCopy)
        return failure();
      chain.stagedView = view;
      chain.stageCopy = stage;
      chain.inputView = stage.getSource().getDefiningOp<memref::SubViewOp>();
      continue;
    }
    return failure();
  }
  if (!chain.inputIf || !chain.inputWrite || !chain.stagedView ||
      !chain.inputView || !chain.stageCopy ||
      !resultlessTwoRegionIf(chain.inputIf) ||
      chain.inputIf.getCondition() != chain.outputIf.getCondition() ||
      chain.inputIf->getBlock() != chain.outputIf->getBlock() ||
      chain.inputIf->getBlock() != chain.transposeAlloc->getBlock() ||
      !chain.inputIf->isBeforeInBlock(chain.transposeAlloc) ||
      !zeroUnitSubview(chain.inputView) || !zeroUnitSubview(chain.stagedView) ||
      !sole(chain.inputView.getResult(), chain.stageCopy) ||
      !sole(chain.stagedView.getResult(), chain.stageCopy))
    return failure();

  SmallVector<Operation *> inputThen, inputElse;
  if (!collectRegionOps(chain.inputIf.getThenRegion(), inputThen) ||
      !collectRegionOps(chain.inputIf.getElseRegion(), inputElse) ||
      inputThen.size() != 2)
    return failure();
  chain.inputRead = dyn_cast<vector::TransferReadOp>(inputThen[0]);
  if (!chain.inputRead || (inputThen[1] != chain.inputWrite.getOperation()) ||
      !isFullVectorRead(chain.inputRead, scratchMemrefType(chain.inputAlloc)) ||
      !isFullVectorWrite(chain.inputWrite, scratchMemrefType(chain.inputAlloc)) ||
      !sole(chain.inputRead.getResult(), chain.inputWrite.getOperation()) ||
      chain.inputWrite.getSource() != chain.inputAlloc->getResult(0) ||
      inputElse.size() < 3)
    return failure();

  unsigned tailIndex = inputElse.size() - 3;
  if (chain.paddingWrite) {
    if (tailIndex == 0)
      return failure();
    auto paddingIf = dyn_cast<scf::IfOp>(inputElse[tailIndex - 1]);
    if (!paddingIf || paddingIf.getNumResults() != 0 ||
        !resultlessIfWithThen(paddingIf) ||
        !paddingIf.getElseRegion().empty() ||
        chain.paddingWrite->getParentOp() != paddingIf.getOperation())
      return failure();
    SmallVector<Operation *> paddingOps;
    if (!collectRegionOps(paddingIf.getThenRegion(), paddingOps) ||
        paddingOps.size() != 1 ||
        paddingOps.front() != chain.paddingWrite.getOperation() ||
        !isFullVectorWrite(chain.paddingWrite,
                           scratchMemrefType(chain.inputAlloc)) ||
        chain.paddingWrite.getSource() != chain.inputAlloc->getResult(0))
      return failure();

    SmallVector<Operation *> conditionSetup;
    auto collectConditionSetup = [&](auto &&self, Value value) -> bool {
      Operation *op = value.getDefiningOp();
      if (!op || op->getBlock() != paddingIf->getBlock())
        return true;
      if (!op->isBeforeInBlock(paddingIf) ||
          op->getName().getDialectNamespace() != "arith")
        return false;
      if (llvm::find(conditionSetup, op) != conditionSetup.end())
        return true;
      conditionSetup.push_back(op);
      return llvm::all_of(op->getOperands(),
                          [&](Value operand) { return self(self, operand); });
    };
    ArrayRef<Operation *> inputConditionOps(inputElse.data(), tailIndex - 1);
    if (!collectConditionSetup(collectConditionSetup,
                               paddingIf.getCondition()) ||
        conditionSetup.size() != inputConditionOps.size() ||
        llvm::any_of(inputConditionOps, [&](Operation *op) {
          return llvm::find(conditionSetup, op) == conditionSetup.end();
        }))
      return failure();
  } else if (tailIndex != 0) {
    return failure();
  }
  if (inputElse[tailIndex] != chain.inputView.getOperation() ||
      inputElse[tailIndex + 1] != chain.stagedView.getOperation() ||
      inputElse[tailIndex + 2] != chain.stageCopy.getOperation() ||
      chain.inputView.getSource() != chain.inputRead.getSource() ||
      chain.stagedView.getSource() != chain.inputAlloc->getResult(0) ||
      !sameSubviewExtent(chain.inputView, chain.stagedView))
    return failure();

  for (Operation *user : chain.destinationParent->getUsers())
    if (user != fullCopy.getOperation() &&
        user != chain.destinationView.getOperation())
      return failure();
  if (!sameBlockInOrderPresent(
          SmallVector<Operation *>{chain.inputAlloc, chain.inputIf.getOperation(),
                                   chain.transposeRead.getOperation(),
                                   chain.transpose.getOperation(),
                                   chain.transposeWrite.getOperation(),
                                   chain.destinationParent.getOperation(),
                                   chain.outputIf.getOperation(),
                                   chain.inputDealloc
                                       ? chain.inputDealloc.getOperation()
                                       : nullptr,
                                   chain.transposeDealloc
                                       ? chain.transposeDealloc.getOperation()
                                       : nullptr}))
    return failure();
  // The independent output allocation may precede or follow the vector read.
  if (!sameBlockInOrder({chain.inputIf.getOperation(), chain.transposeAlloc,
                         chain.transposeWrite.getOperation()}))
    return failure();
  return chain;
}

static void rewriteCurrentBufferizedCopy(CurrentBufferizedChain chain,
                                         IRRewriter &rewriter) {
  Location loc = chain.fullCopy.getLoc();
  auto outputShape = scratchMemrefType(chain.transposeAlloc).getShape();
  rewriter.setInsertionPoint(chain.fullCopy);
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value rows = rewriter.create<arith::ConstantIndexOp>(loc, outputShape[0]);
  Value cols = rewriter.create<arith::ConstantIndexOp>(loc, outputShape[1]);
  auto outer = rewriter.create<scf::ForOp>(loc, zero, rows, one);
  rewriter.setInsertionPointToStart(outer.getBody());
  auto inner = rewriter.create<scf::ForOp>(loc, zero, cols, one);
  rewriter.setInsertionPointToStart(inner.getBody());
  Value value = rewriter.create<memref::LoadOp>(
      loc, chain.inputAlloc->getResult(0),
      ValueRange{inner.getInductionVar(), outer.getInductionVar()});
  rewriter.create<memref::StoreOp>(
      loc, value, chain.destinationParent,
      ValueRange{outer.getInductionVar(), inner.getInductionVar()});

  Operation *elseAnchor = chain.outputView.getOperation();
  rewriter.setInsertionPoint(chain.fullCopy);
  rewriter.eraseOp(chain.fullCopy);
  for (Operation *op : {chain.transposeAlloc, chain.transposeRead.getOperation(),
                        chain.transpose.getOperation(),
                        chain.transposeWrite.getOperation()})
    rewriter.moveOpBefore(op, elseAnchor);
  if (chain.transposeDealloc)
    rewriter.moveOpBefore(
        chain.transposeDealloc.getOperation(),
        chain.outputIf.getElseRegion().front().getTerminator());
}

static bool rewriteVectorizedBufferizedCopy(BufferCopy finalCopy,
                                            IRRewriter &rewriter) {
  // Rewrite the vectorized full-tile chain only after all ownership checks pass.

  auto outputView = finalCopy.getSource().getDefiningOp<memref::SubViewOp>();
  auto destinationView =
      finalCopy.getTarget().getDefiningOp<memref::SubViewOp>();
  if (!outputView || !destinationView || !zeroUnitSubview(outputView) ||
      !zeroUnitSubview(destinationView) ||
      !sole(outputView.getResult(), finalCopy) ||
      !sole(destinationView.getResult(), finalCopy))
    return false;
  auto outputAlloc = outputView.getSource().getDefiningOp<memref::AllocOp>();
  auto destinationParent =
      destinationView.getSource().getDefiningOp<memref::ReinterpretCastOp>();
  if (!outputAlloc || !destinationParent ||
      !staticRowMajor(outputAlloc.getType()) ||
      !sole(destinationParent.getResult(), destinationView.getOperation()))
    return false;

  linalg::CopyOp materialize;
  memref::DeallocOp outputDealloc;
  for (Operation *user : outputAlloc->getUsers()) {
    if (user == outputView.getOperation())
      continue;
    if (auto op = dyn_cast<linalg::CopyOp>(user)) {
      if (materialize)
        return false;
      materialize = op;
    } else if (auto op = dyn_cast<memref::DeallocOp>(user)) {
      if (outputDealloc)
        return false;
      outputDealloc = op;
    } else {
      return false;
    }
  }
  // Direct loads and stores cannot replace a converting materialization.
  if (!materialize || !BufferCopy::get(materialize.getOperation()) ||
      materialize.getOutputs().front() != outputAlloc.getResult())
    return false;
  auto transposeAlloc =
      materialize.getInputs().front().getDefiningOp<memref::AllocOp>();
  if (!transposeAlloc || !staticRowMajor(transposeAlloc.getType()))
    return false;

  vector::TransferWriteOp write;
  memref::DeallocOp transposeDealloc;
  for (Operation *user : transposeAlloc->getUsers()) {
    if (user == materialize.getOperation())
      continue;
    if (auto op = dyn_cast<vector::TransferWriteOp>(user)) {
      if (write || op.getSource() != transposeAlloc.getResult())
        return false;
      write = op;
      continue;
    }
    auto op = dyn_cast<memref::DeallocOp>(user);
    if (!op || transposeDealloc)
      return false;
    transposeDealloc = op;
  }
  auto transpose =
      write ? write.getVector().getDefiningOp<vector::TransposeOp>() : nullptr;
  if (!transpose || transpose.getPermutation() != ArrayRef<int64_t>{1, 0} ||
      !sole(transpose->getResult(0), write.getOperation()))
    return false;
  auto read =
      transpose->getOperand(0).getDefiningOp<vector::TransferReadOp>();
  if (!read || !sole(read->getResult(0), transpose.getOperation()))
    return false;
  auto inputAlloc = read.getSource().getDefiningOp<memref::AllocOp>();
  if (!inputAlloc || !staticRowMajor(inputAlloc.getType()) ||
      !isFullVectorRead(read, inputAlloc.getType()) ||
      !isFullVectorWrite(write, transposeAlloc.getType()))
    return false;

  memref::SubViewOp stagedView, inputView;
  BufferCopy stageCopy;
  memref::DeallocOp inputDealloc;
  vector::TransferWriteOp tailPaddingWrite;
  for (Operation *user : inputAlloc->getUsers()) {
    if (user == read.getOperation())
      continue;
    if (isa<linalg::FillOp>(user)) {
      // Moving the scratch read must not cross a later initialization.
      if (!sameBlockInOrder({user, read.getOperation()}))
        return false;
      continue;
    }
    if (auto op = dyn_cast<vector::TransferWriteOp>(user)) {
      if (tailPaddingWrite || op.getSource() != inputAlloc.getResult())
        return false;
      tailPaddingWrite = op;
      continue;
    }
    if (auto op = dyn_cast<memref::DeallocOp>(user)) {
      if (inputDealloc)
        return false;
      inputDealloc = op;
      continue;
    }
    auto view = dyn_cast<memref::SubViewOp>(user);
    if (!view || stagedView || !view.getResult().hasOneUse())
      return false;
    stageCopy = BufferCopy::get(*view.getResult().user_begin());
    if (!stageCopy || stageCopy.getTarget() != view.getResult())
      return false;
    stagedView = view;
    inputView = stageCopy.getSource().getDefiningOp<memref::SubViewOp>();
  }
  if (!inputView || !stageCopy || !inputDealloc || !transposeDealloc ||
      !outputDealloc || !zeroUnitSubview(inputView) ||
      !zeroUnitSubview(stagedView) || !sole(inputView.getResult(), stageCopy) ||
      !sameSubviewExtent(inputView, stagedView) ||
      !sameSubviewExtent(outputView, destinationView) ||
      inputAlloc.getType().getShape()[0] != outputAlloc.getType().getShape()[1] ||
      inputAlloc.getType().getShape()[1] != outputAlloc.getType().getShape()[0] ||
      transposeAlloc.getType().getShape() != outputAlloc.getType().getShape())
    return false;
  if (tailPaddingWrite) {
    auto paddingIf = dyn_cast<scf::IfOp>(tailPaddingWrite->getParentOp());
    if (!paddingIf || paddingIf->getBlock() != stagedView->getBlock() ||
        !paddingIf->isBeforeInBlock(stagedView) ||
        tailPaddingWrite->getBlock() != &paddingIf.getThenRegion().front() ||
        !isFullVectorWrite(tailPaddingWrite, inputAlloc.getType()))
      return false;
  }
  if (!sameBlockInOrder({inputAlloc.getOperation(), inputView.getOperation(),
                         stagedView.getOperation(), stageCopy.getOperation(),
                         read.getOperation(), transpose.getOperation(),
                         transposeAlloc.getOperation(), write.getOperation(),
                         destinationParent.getOperation(), outputAlloc.getOperation(),
                         materialize.getOperation(), outputView.getOperation(),
                         destinationView.getOperation(), finalCopy.getOperation(),
                         inputDealloc.getOperation(), transposeDealloc.getOperation(),
                         outputDealloc.getOperation()}))
    return false;

  if (!canMoveOutputBefore(
          transposeAlloc, finalCopy,
          {transposeAlloc.getOperation(), write.getOperation(),
           outputAlloc.getOperation(), materialize.getOperation(),
           outputView.getOperation()}))
    return false;

  SmallVector<Operation *> setup;
  auto collect = [&](auto &&self, Value value) -> bool {
    Operation *op = value.getDefiningOp();
    if (!op)
      return true;
    // isBeforeInBlock requires both operations to belong to the same block.
    if (op->getBlock() != transposeAlloc->getBlock())
      return false;
    if (op->isBeforeInBlock(transposeAlloc))
      return true;
    if (op->getName().getDialectNamespace() != "arith")
      return false;
    for (Value operand : op->getOperands())
      if (!self(self, operand))
        return false;
    if (llvm::find(setup, op) == setup.end())
      setup.push_back(op);
    return true;
  };
  for (Value operand : destinationParent->getOperands())
    if (!collect(collect, operand))
      return false;
  for (Value operand : destinationView->getOperands().drop_front())
    if (!collect(collect, operand))
      return false;
  for (Operation *op : setup)
    rewriter.moveOpBefore(op, transposeAlloc);
  rewriter.moveOpBefore(destinationParent, transposeAlloc);
  rewriter.moveOpBefore(destinationView, transposeAlloc);

  Location loc = finalCopy.getLoc();
  auto inputShape = inputAlloc.getType().getShape();
  auto outputShape = outputAlloc.getType().getShape();
  rewriter.setInsertionPoint(transposeAlloc);
  Value condition =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
  condition = appendFullTileChecks(rewriter, loc, inputView.getResult(),
                                   inputShape, condition);
  condition = appendFullTileChecks(rewriter, loc, stagedView.getResult(),
                                   inputShape, condition);
  condition = appendFullTileChecks(rewriter, loc, destinationView.getResult(),
                                   outputShape, condition);
  auto branch = rewriter.create<scf::IfOp>(loc, condition, true);
  OpBuilder builder(&branch.getThenRegion().front(),
                    branch.getThenRegion().front().begin());
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  auto outer = builder.create<scf::ForOp>(
      loc, zero, builder.create<arith::ConstantIndexOp>(loc, outputShape[0]), one);
  builder.setInsertionPointToStart(outer.getBody());
  auto inner = builder.create<scf::ForOp>(
      loc, zero, builder.create<arith::ConstantIndexOp>(loc, outputShape[1]), one);
  builder.setInsertionPointToStart(inner.getBody());
  Value value = builder.create<memref::LoadOp>(
      loc, inputAlloc,
      ValueRange{inner.getInductionVar(), outer.getInductionVar()});
  builder.create<memref::StoreOp>(
      loc, value, destinationView,
      ValueRange{outer.getInductionVar(), inner.getInductionVar()});

  Operation *terminator = branch.getElseRegion().front().getTerminator();
  for (Operation *op : {read.getOperation(), transpose.getOperation(),
                        transposeAlloc.getOperation(), write.getOperation(),
                        outputAlloc.getOperation(), materialize.getOperation(),
                        outputView.getOperation(), finalCopy.getOperation(),
                        transposeDealloc.getOperation(), outputDealloc.getOperation()})
    rewriter.moveOpBefore(op, terminator);
  return true;

}

static bool rewriteBufferizedCopy(BufferCopy finalCopy,
                                  IRRewriter &rewriter) {
  // Rewrite the scalar bufferized full-tile chain only after all ownership checks pass.

  auto outputView = finalCopy.getSource().getDefiningOp<memref::SubViewOp>();
  auto destinationView = finalCopy.getTarget().getDefiningOp<memref::SubViewOp>();
  if (!outputView || !destinationView || !zeroUnitSubview(outputView) ||
      !zeroUnitSubview(destinationView) || !sole(outputView.getResult(), finalCopy) ||
      !sole(destinationView.getResult(), finalCopy))
    return false;
  auto outputAlloc = outputView.getSource().getDefiningOp<memref::AllocOp>();
  auto destinationParent =
      destinationView.getSource().getDefiningOp<memref::ReinterpretCastOp>();
  if (!outputAlloc || !destinationParent || !staticRowMajor(outputAlloc.getType()) ||
      !sole(destinationParent.getResult(), destinationView.getOperation()))
    return false;
  linalg::CopyOp materialize;
  memref::DeallocOp outputDealloc;
  for (Operation *user : outputAlloc->getUsers()) {
    if (user == outputView.getOperation())
      continue;
    if (auto op = dyn_cast<linalg::CopyOp>(user)) {
      if (materialize)
        return false;
      materialize = op;
    } else if (auto op = dyn_cast<memref::DeallocOp>(user)) {
      if (outputDealloc)
        return false;
      outputDealloc = op;
    } else {
      return false;
    }
  }
  // Direct loads and stores cannot replace a converting materialization.
  if (!materialize || !BufferCopy::get(materialize.getOperation()) ||
      materialize.getOutputs().front() != outputAlloc.getResult())
    return false;
  auto transposeAlloc =
      materialize.getInputs().front().getDefiningOp<memref::AllocOp>();
  if (!transposeAlloc || !staticRowMajor(transposeAlloc.getType()))
    return false;

  memref::SubViewOp stagedView, inputView;
  BufferCopy stageCopy;
  memref::DeallocOp inputDealloc, transposeDealloc;
  linalg::TransposeOp transpose;
  for (Operation *user : transposeAlloc->getUsers()) {
    if (user == materialize.getOperation())
      continue;
    if (auto op = dyn_cast<linalg::TransposeOp>(user)) {
      if (transpose || op.getInit() != transposeAlloc.getResult() ||
          op.getPermutation() != ArrayRef<int64_t>{1, 0})
        return false;
      transpose = op;
      continue;
    }
    auto op = dyn_cast<memref::DeallocOp>(user);
    if (!op || transposeDealloc)
      return false;
    transposeDealloc = op;
  }
  auto inputAlloc =
      transpose ? transpose.getInput().getDefiningOp<memref::AllocOp>() : nullptr;
  if (!transpose || !inputAlloc)
    return false;
  for (Operation *user : inputAlloc->getUsers()) {
    if (user == transpose.getOperation())
      continue;
    if (isa<linalg::FillOp>(user)) {
      // The guarded branch is inserted before transposeAlloc, so reads must
      // not cross an initialization between that point and the transpose.
      if (!sameBlockInOrder({user, transposeAlloc.getOperation()}))
        return false;
      continue;
    }
    if (auto op = dyn_cast<memref::DeallocOp>(user)) {
      if (inputDealloc)
        return false;
      inputDealloc = op;
      continue;
    }
    auto view = dyn_cast<memref::SubViewOp>(user);
    if (!view || stagedView || !view.getResult().hasOneUse())
      return false;
    stageCopy = BufferCopy::get(*view.getResult().user_begin());
    if (!stageCopy || stageCopy.getTarget() != view.getResult())
      return false;
    stagedView = view;
    inputView = stageCopy.getSource().getDefiningOp<memref::SubViewOp>();
  }
  if (!inputView || !stageCopy || !inputDealloc || !transposeDealloc ||
      !outputDealloc ||
      !staticRowMajor(inputAlloc.getType()) || !zeroUnitSubview(inputView) ||
      !zeroUnitSubview(stagedView) || !sole(inputView.getResult(), stageCopy) ||
      !sameSubviewExtent(inputView, stagedView) ||
      !sameSubviewExtent(outputView, destinationView) ||
      inputAlloc.getType().getShape()[0] != outputAlloc.getType().getShape()[1] ||
      inputAlloc.getType().getShape()[1] != outputAlloc.getType().getShape()[0] ||
      transposeAlloc.getType().getShape() != outputAlloc.getType().getShape())
    return false;
  if (!sameBlockInOrder({inputAlloc.getOperation(), inputView.getOperation(),
                         stagedView.getOperation(), stageCopy.getOperation(),
                         transposeAlloc.getOperation(), transpose.getOperation(),
                         destinationParent.getOperation(), outputAlloc.getOperation(),
                         materialize.getOperation(), outputView.getOperation(),
                         destinationView.getOperation(), finalCopy.getOperation(),
                         inputDealloc.getOperation(), transposeDealloc.getOperation(),
                         outputDealloc.getOperation()}))
    return false;
  if (!canMoveOutputBefore(
          transposeAlloc, finalCopy,
          {transposeAlloc.getOperation(), transpose.getOperation(),
           outputAlloc.getOperation(), materialize.getOperation(),
           outputView.getOperation()}))
    return false;

  SmallVector<Operation *> setup;
  auto collect = [&](auto &&self, Value value) -> bool {
    Operation *op = value.getDefiningOp();
    if (!op)
      return true;
    // isBeforeInBlock requires both operations to belong to the same block.
    if (op->getBlock() != transposeAlloc->getBlock())
      return false;
    if (op->isBeforeInBlock(transposeAlloc))
      return true;
    if (op->getName().getDialectNamespace() != "arith")
      return false;
    for (Value operand : op->getOperands())
      if (!self(self, operand))
        return false;
    if (llvm::find(setup, op) == setup.end())
      setup.push_back(op);
    return true;
  };
  for (Value operand : destinationParent->getOperands())
    if (!collect(collect, operand))
      return false;
  for (Value operand : destinationView->getOperands().drop_front())
    if (!collect(collect, operand))
      return false;
  for (Operation *op : setup)
    rewriter.moveOpBefore(op, transposeAlloc);
  rewriter.moveOpBefore(destinationParent, transposeAlloc);
  rewriter.moveOpBefore(destinationView, transposeAlloc);
  rewriter.setInsertionPoint(transposeAlloc);
  Location loc = finalCopy.getLoc();
  auto inputShape = inputAlloc.getType().getShape();
  auto outputShape = outputAlloc.getType().getShape();
  Value condition = rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
  condition = appendFullTileChecks(rewriter, loc, inputView.getResult(), inputShape, condition);
  condition = appendFullTileChecks(rewriter, loc, stagedView.getResult(), inputShape, condition);
  condition = appendFullTileChecks(rewriter, loc, destinationView.getResult(), outputShape, condition);
  auto branch = rewriter.create<scf::IfOp>(loc, condition, true);
  OpBuilder builder(&branch.getThenRegion().front(), branch.getThenRegion().front().begin());
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  auto outer = builder.create<scf::ForOp>(loc, zero, builder.create<arith::ConstantIndexOp>(loc, outputShape[0]), one);
  builder.setInsertionPointToStart(outer.getBody());
  auto inner = builder.create<scf::ForOp>(loc, zero, builder.create<arith::ConstantIndexOp>(loc, outputShape[1]), one);
  builder.setInsertionPointToStart(inner.getBody());
  Value value = builder.create<memref::LoadOp>(loc, inputAlloc,
      ValueRange{inner.getInductionVar(), outer.getInductionVar()});
  builder.create<memref::StoreOp>(loc, value, destinationView,
      ValueRange{outer.getInductionVar(), inner.getInductionVar()});

  Operation *terminator = branch.getElseRegion().front().getTerminator();
  for (Operation *op : {transposeAlloc.getOperation(), transpose.getOperation(),
                        outputAlloc.getOperation(), materialize.getOperation(),
                        outputView.getOperation(), finalCopy.getOperation(),
                        transposeDealloc.getOperation(), outputDealloc.getOperation()})
    rewriter.moveOpBefore(op, terminator);
  return true;

}

static FailureOr<Chain> matchCopy(BufferCopy copy) {
  // Match the unique staged transpose materialization chain conservatively.

  Chain c;
  c.finalCopy = copy;
  c.outputView = copy.getSource().getDefiningOp<memref::SubViewOp>();
  c.destinationView = copy.getTarget().getDefiningOp<memref::SubViewOp>();
  if (!c.outputView || !c.destinationView || !zeroUnitSubview(c.outputView) ||
      !zeroUnitSubview(c.destinationView) || !sole(c.outputView.getResult(), copy) ||
      !sole(c.destinationView.getResult(), copy))
    return failure();
  c.outputAlloc = c.outputView.getSource().getDefiningOp<memref::AllocOp>();
  if (!c.outputAlloc || !staticRowMajor(c.outputAlloc.getType())) return failure();
  for (Operation *user : c.outputAlloc->getUsers()) {
    if (user == c.outputView.getOperation()) continue;
    if (auto materialize = dyn_cast<bufferization::MaterializeInDestinationOp>(user)) {
      if (c.materialize) return failure();
      c.materialize = materialize;
    } else if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
      if (c.outputDealloc) return failure();
      c.outputDealloc = dealloc;
    } else return failure();
  }
  if (!c.materialize || c.materialize.getDest() != c.outputAlloc.getResult())
    return failure();
  c.transpose = c.materialize.getSource().getDefiningOp<linalg::TransposeOp>();
  if (!c.transpose || !sole(c.transpose.getResult().front(), c.materialize) ||
      c.transpose.getPermutation() != ArrayRef<int64_t>{1, 0}) return failure();
  c.toTensor = c.transpose.getInput().getDefiningOp<bufferization::ToTensorOp>();
  c.empty = c.transpose.getInit().getDefiningOp<tensor::EmptyOp>();
  if (!c.toTensor || !c.empty || !sole(c.toTensor.getResult(), c.transpose) ||
      !sole(c.empty.getResult(), c.transpose)) return failure();
  c.inputAlloc = c.toTensor.getMemref().getDefiningOp<memref::AllocOp>();
  if (!c.inputAlloc || !staticRowMajor(c.inputAlloc.getType())) return failure();
  for (Operation *user : c.inputAlloc->getUsers()) {
    if (user == c.toTensor.getOperation())
      continue;
    if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
      if (c.inputDealloc) return failure();
      c.inputDealloc = dealloc;
      continue;
    }
    if (auto fill = dyn_cast<linalg::FillOp>(user)) {
      // Preserve the input snapshot when moving reads to the output loop.
      if (c.inputFill || fill.getOutputs().size() != 1 ||
          fill.getOutputs().front() != c.inputAlloc.getResult() ||
          !sameBlockInOrder({fill.getOperation(), c.toTensor.getOperation()}))
        return failure();
      c.inputFill = fill;
      continue;
    }
    auto view = dyn_cast<memref::SubViewOp>(user);
    if (!view || c.stagedView)
      return failure();
    for (Operation *viewUser : view->getUsers()) {
      auto stage = BufferCopy::get(viewUser);
      if (!stage || stage.getTarget() != view.getResult() || c.stageCopy)
        return failure();
      c.stagedView = view;
      c.stageCopy = stage;
      c.inputView = stage.getSource().getDefiningOp<memref::SubViewOp>();
    }
    if (!c.stageCopy)
      return failure();
  }
  if (!c.stageCopy || !c.inputView || !zeroUnitSubview(c.inputView) ||
      !zeroUnitSubview(c.stagedView) || !sole(c.inputView.getResult(), c.stageCopy) ||
      !sole(c.stagedView.getResult(), c.stageCopy))
    return failure();
  auto inputTensor = dyn_cast<RankedTensorType>(c.toTensor.getType());
  auto outputTensor = dyn_cast<RankedTensorType>(c.transpose.getResult().front().getType());
  if (!inputTensor || !outputTensor ||
      inputTensor.getElementType() != outputTensor.getElementType() ||
      inputTensor.getShape() != c.inputAlloc.getType().getShape() ||
      outputTensor.getShape() != c.outputAlloc.getType().getShape() ||
      c.inputAlloc.getType().getShape()[0] != c.outputAlloc.getType().getShape()[1] ||
      c.inputAlloc.getType().getShape()[1] != c.outputAlloc.getType().getShape()[0]) return failure();
  if (!sameBlockInOrder({c.inputView.getOperation(), c.stageCopy.getOperation(),
                         c.toTensor.getOperation(), c.empty.getOperation(),
                         c.transpose.getOperation(), c.outputAlloc.getOperation(),
                         c.materialize.getOperation(), c.outputView.getOperation(),
                         c.destinationView.getOperation(), c.finalCopy.getOperation()}))
    return failure();
  if ((c.inputDealloc &&
       !sameBlockInOrder({c.finalCopy.getOperation(), c.inputDealloc.getOperation()})) ||
      (c.outputDealloc &&
       !sameBlockInOrder({c.finalCopy.getOperation(), c.outputDealloc.getOperation()})))
    return failure();
  return c;

}

static void rewrite(Chain c, IRRewriter &rewriter) {
  // Emit direct transposed stores and retain the original materialization in the fallback branch.

  Location loc = c.finalCopy.getLoc();
  auto inputShape = c.inputAlloc.getType().getShape();
  auto outputShape = c.outputAlloc.getType().getShape();
  rewriter.setInsertionPoint(c.finalCopy);
  Value condition = rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
  condition = appendFullTileChecks(rewriter, loc, c.inputView.getResult(),
                                   inputShape, condition);
  condition = appendFullTileChecks(rewriter, loc, c.stagedView.getResult(),
                                   inputShape, condition);
  condition = appendFullTileChecks(rewriter, loc,
                                   c.destinationView.getResult(), outputShape,
                                   condition);
  auto branch = rewriter.create<scf::IfOp>(loc, condition, true);
  OpBuilder builder(&branch.getThenRegion().front(), branch.getThenRegion().front().begin());
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  auto outer = builder.create<scf::ForOp>(loc, zero, builder.create<arith::ConstantIndexOp>(loc, outputShape[0]), one);
  builder.setInsertionPointToStart(outer.getBody());
  auto inner = builder.create<scf::ForOp>(loc, zero, builder.create<arith::ConstantIndexOp>(loc, outputShape[1]), one);
  builder.setInsertionPointToStart(inner.getBody());
  Value value = builder.create<memref::LoadOp>(loc, c.inputAlloc,
      ValueRange{inner.getInductionVar(), outer.getInductionVar()});
  builder.create<memref::StoreOp>(loc, value, c.destinationView,
      ValueRange{outer.getInductionVar(), inner.getInductionVar()});
  Operation *elseTerminator = branch.getElseRegion().front().getTerminator();
  for (Operation *op : {c.toTensor.getOperation(), c.empty.getOperation(), c.transpose.getOperation(),
                        c.outputAlloc.getOperation(), c.materialize.getOperation(), c.outputView.getOperation(),
                        c.finalCopy.getOperation()})
    rewriter.moveOpBefore(op, elseTerminator);
  if (c.outputDealloc)
    rewriter.moveOpBefore(c.outputDealloc, elseTerminator);

}

struct DirectTransposeOutputPass : public impl::DirectTransposeOutputBase<DirectTransposeOutputPass> {
  void runOnOperation() override {
    // Attempt only the guarded direct-output rewrites; all other copies fall through.

    SmallVector<BufferCopy> copies;
    getOperation().walk([&](Operation *op) {
      if (auto copy = BufferCopy::get(op))
        copies.push_back(copy);
    });
    IRRewriter rewriter(&getContext());

    for (BufferCopy copy : llvm::reverse(copies))
      if (FailureOr<VersionedFullTileChain> chain =
              matchVersionedFullTileCopy(copy);
          succeeded(chain))
        rewriteVersionedFullTileCopy(*chain, rewriter);

    copies.clear();
    getOperation().walk([&](Operation *op) {
      if (auto copy = BufferCopy::get(op))
        copies.push_back(copy);
    });

    // The current lowering uses matching full/tail branches. Rewrite in
    // reverse order because erasing a full copy invalidates its operation.
    for (BufferCopy copy : llvm::reverse(copies))
      if (FailureOr<CurrentBufferizedChain> chain =
              matchCurrentBufferizedCopy(copy);
          succeeded(chain))
        rewriteCurrentBufferizedCopy(*chain, rewriter);

    copies.clear();
    getOperation().walk([&](Operation *op) {
      if (auto copy = BufferCopy::get(op))
        copies.push_back(copy);
    });
    for (BufferCopy copy : copies)
      if (!rewriteVectorizedBufferizedCopy(copy, rewriter) &&
          !rewriteBufferizedCopy(copy, rewriter))
        if (FailureOr<Chain> chain = matchCopy(copy); succeeded(chain))
          rewrite(*chain, rewriter);

  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDirectTransposeOutput() {
  // Register the conservative direct-output transpose pass.

  return std::make_unique<DirectTransposeOutputPass>();

}
} // namespace mlir::triton
