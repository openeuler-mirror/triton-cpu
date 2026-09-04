//===----------------------------------------------------------------------===//
// Full-tile region versioning
//===----------------------------------------------------------------------===//
//
// StructuredToMemref lowers a masked tts.load/tts.store to a scratch tile plus
// a runtime `dim == tileSize` guard, taking a whole-tile copy when the guard
// holds.  That leaves the copy *conditional*, and both branches of the guard
// write the same scratch slot -- one with a static full-tile copy, one with a
// dynamically sized partial copy.  LLVM's SROA cannot split a slot written that
// way, so the scratch round-trip survives all the way to the assembly.
//
// This pass moves the decision up one level.  Instead of one guard per access,
// the conjunction of every access's full-tile condition is computed once and
// the block is cloned:
//
//   scf.if (all tiles full) { <block with the masks dropped> }
//   else                    { <block exactly as before> }
//
// In the if-branch each access gets its own scratch written by a single
// unconditional, statically sized copy, which is the shape SROA can promote. 
// One thing to note is SROA can only handle alloc inside the entry block, so we
// need another pass to hoist the scratch alloc to the entry block.
//
#include "triton-shared/Conversion/TritonToLinalgExperimental/FullTileRegionVersioning.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"

using namespace mlir;
using namespace triton;

namespace {

static SmallVector<OpFoldResult> getMaskDims(Operation *op) {
  if (auto load = dyn_cast<tts::LoadOp>(op))
    return load.getMixedMaskDims();
  return cast<tts::StoreOp>(op).getMixedMaskDims();
}

static Value getAccessPtr(Operation *op) {
  if (auto load = dyn_cast<tts::LoadOp>(op))
    return load.getPtr();
  return cast<tts::StoreOp>(op).getPtr();
}

/// Bytes of the tile an access moves.
static int64_t tileBytes(Operation *op) {
  Type t = isa<tts::LoadOp>(op) ? op->getResult(0).getType()
                                : cast<tts::StoreOp>(op).getValue().getType();
  auto tensor = dyn_cast<RankedTensorType>(t);
  if (!tensor || !tensor.hasStaticShape())
    return -1;
  auto elem = tensor.getElementType();
  if (!elem.isIntOrFloat())
    return -1;
  return tensor.getNumElements() * elem.getIntOrFloatBitWidth() / 8;
}

/// Versioning only pays when the fast branch's scratch tile can become
/// register-resident: what it buys is a single unconditional constant-size copy
/// that SROA can forward into the consuming loads.  A tile too large for the
/// vector register file is copied either way, and all the duplicated region
/// does is double the pressure on the register allocator.
///
/// So cap the tile, and cap the total across one versioned region so a block
/// full of small tiles cannot add up to the same problem.  Both are tunable;
/// set TRITON_SHARED_VERSION_TILE_BYTES=0 to turn versioning off entirely.
static int64_t maxTileBytes() {
  if (const char *e = getenv("TRITON_SHARED_VERSION_TILE_BYTES"))
    return atoll(e);
  return 512;
}
static int64_t maxRegionTileBytes() {
  if (const char *e = getenv("TRITON_SHARED_VERSION_REGION_BYTES"))
    return atoll(e);
  return 4096;
}

/// An access this pass can unmask: a masked load or store straight off a block
/// pointer.  `mask_dims` is the number of valid elements per dimension, so
/// `dim_i == size_i` for all i is exactly "this tile is full", whatever shape
/// the mask expression itself has.
static bool isVersionableAccess(Operation *op) {
  if (!isa<tts::LoadOp, tts::StoreOp>(op))
    return false;
  if (getMaskDims(op).empty())
    return false;
  auto blockPtr = getAccessPtr(op).getDefiningOp<tts::MakeTensorPtrOp>();
  if (!blockPtr)
    return false;
  return blockPtr.getMixedSizes().size() == getMaskDims(op).size();
}

struct VersionableBlock {
  SmallVector<Operation *> maskedAccesses;
  /// Pure index computations the conditions read, in block order.  They are
  /// cloned ahead of the guard rather than moved: the mask dims of a later
  /// access can be produced after an earlier access has already been issued.
  SmallVector<Operation *> condSlice;
};

static bool analyzeBlock(Block &block, VersionableBlock &result) {
  for (Operation &op : block) {
    if (isVersionableAccess(&op))
      result.maskedAccesses.push_back(&op);
    else if (isa<tts::LoadOp, tts::StoreOp>(&op) && !getMaskDims(&op).empty())
      // A masked access this pass cannot reason about would stay masked in the
      // fast branch, so versioning around it buys nothing.
      return false;
  }
  if (result.maskedAccesses.empty())
    return false;

  const int64_t tileCap = maxTileBytes();
  if (tileCap <= 0)
    return false;
  int64_t total = 0;
  for (Operation *access : result.maskedAccesses) {
    int64_t bytes = tileBytes(access);
    if (bytes < 0 || bytes > tileCap)
      return false;
    total += bytes;
  }
  if (total > maxRegionTileBytes())
    return false;

  DenseSet<Operation *> condDefs;
  for (Operation *access : result.maskedAccesses) {
    for (OpFoldResult dim : getMaskDims(access)) {
      auto v = dyn_cast<Value>(dim);
      if (!v)
        continue;
      SetVector<Operation *> slice;
      BackwardSliceOptions opts;
      opts.omitBlockArguments = true;
      opts.filter = [&](Operation *o) { return o->getBlock() == &block; };
      getBackwardSlice(v, &slice, opts);
      condDefs.insert(slice.begin(), slice.end());
      if (Operation *def = v.getDefiningOp(); def && def->getBlock() == &block)
        condDefs.insert(def);
    }
  }

  // The slice is cloned above the guard, so every op in it must be safe to
  // duplicate.  Mask dims are index arithmetic, so this normally holds.
  for (Operation &op : block) {
    if (!condDefs.contains(&op))
      continue;
    if (!isMemoryEffectFree(&op))
      return false;
    result.condSlice.push_back(&op);
  }
  return true;
}

static void cloneUnmasked(OpBuilder &builder, Operation *op, IRMapping &map) {
  auto loc = op->getLoc();
  SmallVector<OpFoldResult> noMask;
  if (auto load = dyn_cast<tts::LoadOp>(op)) {
    // No mask means no out-of-bounds element, so `other` is dead too.
    auto newLoad = builder.create<tts::LoadOp>(
        loc, map.lookupOrDefault(load.getPtr()), noMask, /*other=*/Value());
    map.map(load.getResult(), newLoad.getResult());
    return;
  }
  auto store = cast<tts::StoreOp>(op);
  builder.create<tts::StoreOp>(loc, map.lookupOrDefault(store.getPtr()),
                               map.lookupOrDefault(store.getValue()), noMask);
}

static LogicalResult versionBlock(Block &block, const VersionableBlock &info) {
  Operation *terminator = block.getTerminator();
  Operation *first = info.maskedAccesses.front();

  OpBuilder builder(first);
  Location loc = first->getLoc();

  // Recompute the mask dims ahead of the guard; CSE folds away the copies of
  // the ones that were already available here.
  IRMapping condMap;
  for (Operation *op : info.condSlice)
    builder.clone(*op, condMap);

  Value cond;
  for (Operation *access : info.maskedAccesses) {
    auto blockPtr = getAccessPtr(access).getDefiningOp<tts::MakeTensorPtrOp>();
    auto sizes = blockPtr.getMixedSizes();
    auto dims = getMaskDims(access);
    for (size_t i = 0; i < dims.size(); ++i) {
      auto dimAttr = getConstantIntValue(dims[i]);
      auto sizeAttr = getConstantIntValue(sizes[i]);
      if (dimAttr && sizeAttr && *dimAttr == *sizeAttr)
        continue; // statically whole
      Value dimVal = getValueOrCreateConstantIndexOp(
          builder, loc, dims[i].is<Value>()
                            ? OpFoldResult(condMap.lookupOrDefault(
                                  cast<Value>(dims[i])))
                            : dims[i]);
      Value sizeVal = getValueOrCreateConstantIndexOp(builder, loc, sizes[i]);
      Value eq = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               dimVal, sizeVal);
      cond = cond ? builder.create<arith::AndIOp>(loc, cond, eq) : eq;
    }
  }
  if (!cond)
    return failure(); // every tile statically whole: nothing to version

  SmallVector<Type> resultTypes(terminator->getOperandTypes());
  auto ifOp = builder.create<scf::IfOp>(loc, resultTypes, cond,
                                        /*addThenBlock=*/true,
                                        /*addElseBlock=*/true);

  SmallVector<Operation *> versioned;
  for (Operation *op = first; op != terminator; op = op->getNextNode())
    versioned.push_back(op);

  {
    OpBuilder thenBuilder(ifOp.thenBlock(), ifOp.thenBlock()->begin());
    IRMapping map;
    for (Operation *op : versioned) {
      if (isVersionableAccess(op))
        cloneUnmasked(thenBuilder, op, map);
      else
        thenBuilder.clone(*op, map);
    }
    SmallVector<Value> yielded;
    for (Value v : terminator->getOperands())
      yielded.push_back(map.lookupOrDefault(v));
    thenBuilder.create<scf::YieldOp>(loc, yielded);
  }
  {
    OpBuilder elseBuilder(ifOp.elseBlock(), ifOp.elseBlock()->begin());
    auto yield =
        elseBuilder.create<scf::YieldOp>(loc, terminator->getOperands());
    for (Operation *op : versioned)
      op->moveBefore(yield);
  }

  terminator->setOperands(ifOp.getResults());
  ifOp->setAttr("tts.full_tile_versioned", builder.getUnitAttr());
  return success();
}

struct FullTileVersioningPass
    : public PassWrapper<FullTileVersioningPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FullTileVersioningPass)

  void runOnOperation() override {
    SmallVector<std::pair<Block *, VersionableBlock>> work;
    getOperation()->walk([&](Block *block) {
      VersionableBlock info;
      if (analyzeBlock(*block, info))
        work.emplace_back(block, std::move(info));
    });
    for (auto &[block, info] : work)
      (void)versionBlock(*block, info);
  }
};
}

std::unique_ptr<Pass> triton::createFullTileVersioningPass() {
  return std::make_unique<FullTileVersioningPass>();
}
