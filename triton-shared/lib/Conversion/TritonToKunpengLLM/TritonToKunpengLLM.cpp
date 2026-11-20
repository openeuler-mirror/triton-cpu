//===----------------------------------------------------------------------===//
//
// TritonFlashAttentionToKunpengLLM: recognise a flash-attention forward
// kernel (one program = one block of query rows of one head) and replace its
// body by a call into libkunpeng_llm.so:
//
//   kunpeng_flash_attention_forward_{bf16,f16}(
//       q, q_rs, k, k_rs, v, v_rs, o, o_rs, lse, row_start, num_rows,
//       seqlen_q, seqlen_k, window_flags, left_shift, right_shift, head_dim,
//       scale_log2)
//
// (see third_party/kunpeng/llm/kunpeng_llm.h).
//
// Addresses, indices and loop bounds are compared as integer polynomials over
// SSA values ("atoms") and the two tile index variables I (dim 0) and J
// (dim 1), so the match does not depend on how the frontend spelled the
// arithmetic.  The pass only fires when it can show that the kernel computes
// exactly what the library computes; otherwise the kernel is left alone.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/TritonToKunpengLLM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/Support/Debug.h"

#include <cmath>
#include <map>
#include <optional>
#include <vector>

#define DEBUG_TYPE "triton-to-kunpeng-llm"

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_TRITONFLASHATTENTIONTOKUNPENGLLM
#define GEN_PASS_DEF_TRITONFLASHMLATOKUNPENGLLM
#include "triton-shared/Conversion/TritonToKunpengLLM/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Integer polynomials over atoms
//===----------------------------------------------------------------------===//

// Tile index variables: element (i, j) of a 2-D tensor, element i of a 1-D
// tensor (whose lane maps to I or J depending on how it is expanded).
// KROW stands for the cache row a paged K/V tile reads key J from.
const char kDimI = 0, kDimJ = 0, kDimKRow = 0;
const void *const DIM_I = &kDimI;
const void *const DIM_J = &kDimJ;
const void *const DIM_KROW = &kDimKRow;

struct Poly {
  using Mono = std::vector<const void *>; // sorted atoms, with repetition
  std::map<Mono, int64_t> terms;

  static Poly cst(int64_t c) {
    Poly p;
    if (c != 0)
      p.terms[{}] = c;
    return p;
  }
  static Poly atom(const void *a) {
    Poly p;
    p.terms[{a}] = 1;
    return p;
  }
  static Poly atom(Value v) { return atom(v.getAsOpaquePointer()); }

  Poly operator+(const Poly &o) const {
    Poly r = *this;
    for (auto &[m, c] : o.terms)
      if ((r.terms[m] += c) == 0)
        r.terms.erase(m);
    return r;
  }
  Poly operator-() const {
    Poly r = *this;
    for (auto &t : r.terms)
      t.second = -t.second;
    return r;
  }
  Poly operator-(const Poly &o) const { return *this + (-o); }
  Poly operator*(const Poly &o) const {
    Poly r;
    for (auto &[ma, ca] : terms)
      for (auto &[mb, cb] : o.terms) {
        Mono m = ma;
        m.insert(m.end(), mb.begin(), mb.end());
        std::sort(m.begin(), m.end());
        if ((r.terms[m] += ca * cb) == 0)
          r.terms.erase(m);
      }
    return r;
  }
  bool operator==(const Poly &o) const { return (*this - o).terms.empty(); }
  bool operator!=(const Poly &o) const { return !(*this == o); }

  std::optional<int64_t> getConst() const {
    if (terms.empty())
      return 0;
    if (terms.size() == 1 && terms.begin()->first.empty())
      return terms.begin()->second;
    return std::nullopt;
  }
  bool mentions(const void *a) const {
    for (auto &t : terms)
      if (llvm::is_contained(t.first, a))
        return true;
    return false;
  }
  // p = coeff(a) * a + rest, with `a` of degree <= 1 everywhere.
  std::optional<std::pair<Poly, Poly>> split(const void *a) const {
    Poly coeff, rest;
    for (auto &[m, c] : terms) {
      auto n = llvm::count(m, a);
      if (n > 1)
        return std::nullopt;
      if (n == 0) {
        rest.terms[m] = c;
        continue;
      }
      Mono mm = m;
      mm.erase(std::find(mm.begin(), mm.end(), a));
      coeff.terms[mm] = c;
    }
    return std::make_pair(coeff, rest);
  }
  SmallVector<const void *> atoms() const {
    SmallVector<const void *> r;
    for (auto &t : terms)
      for (auto *a : t.first)
        if (!llvm::is_contained(r, a))
          r.push_back(a);
    return r;
  }
};

static std::optional<int64_t> getIntSplat(Value v) {
  APInt i;
  if (matchPattern(v, m_ConstantInt(&i)))
    return i.getSExtValue();
  DenseIntElementsAttr d;
  if (matchPattern(v, m_Constant(&d)) && d.isSplat())
    return d.getSplatValue<APInt>().getSExtValue();
  return std::nullopt;
}

static std::optional<double> getFloatSplat(Value v) {
  FloatAttr f;
  if (matchPattern(v, m_Constant(&f)))
    return f.getValueAsDouble();
  DenseFPElementsAttr d;
  if (matchPattern(v, m_Constant(&d)) && d.isSplat())
    return d.getSplatValue<APFloat>().convertToDouble();
  return std::nullopt;
}

static bool isFloatSplat(Value v, double x) {
  auto f = getFloatSplat(v);
  return f && (*f == x || (std::isinf(x) && *f == x));
}

// Polynomial value of an integer scalar or integer tensor.  Scalars that are
// not + - * of other scalars are atoms.  Tensors must be built from
// make_range, splat, expand_dims, broadcast, constants and + - *; `lane`
// names the index variable of a 1-D tensor.
class PolyEval {
public:
  // Values standing for a polynomial that cannot be read off the IR, e.g. the
  // cache row a paged K/V load gathers (bound to DIM_KROW once its structure
  // has been checked), or a loop-carried affine counter.
  llvm::DenseMap<Value, Poly> bound;
  // When set, integer tensors outside the understood op set (loads, loop
  // iter args, ...) become opaque atoms instead of failing the evaluation.
  // Used by the flash-MLA matcher, whose gather addresses are built from
  // loaded page indices; identity of those values is still checked (the same
  // SSA value gives the same atom), so this stays sound.
  bool atomizeUnknownTensors = false;
  // Optional hook consulted before atomizing; lets the flash-MLA matcher
  // resolve ops it has verified structurally (e.g. offs_n % BLOCK_N -> the
  // key lane).
  std::function<std::optional<Poly>(Operation *, const void *)> tensorHook;

  std::optional<Poly> scalar(Value v) {
    if (auto it = bound.find(v); it != bound.end())
      return it->second;
    if (!v.getType().isIntOrIndex())
      return std::nullopt;
    if (auto c = getIntSplat(v))
      return Poly::cst(*c);
    Operation *op = v.getDefiningOp();
    if (op && isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op)) {
      auto a = scalar(op->getOperand(0)), b = scalar(op->getOperand(1));
      if (!a || !b)
        return std::nullopt;
      if (isa<arith::AddIOp>(op))
        return *a + *b;
      if (isa<arith::SubIOp>(op))
        return *a - *b;
      return *a * *b;
    }
    if (op && isa<arith::ExtSIOp, arith::IndexCastOp>(op))
      return scalar(op->getOperand(0));
    return Poly::atom(v);
  }

  std::optional<Poly> tensor(Value v, const void *lane) {
    if (auto it = bound.find(v); it != bound.end())
      return it->second;
    auto ty = dyn_cast<RankedTensorType>(v.getType());
    if (!ty)
      return scalar(v);
    if (!ty.getElementType().isInteger() || ty.getRank() > 2)
      return std::nullopt;
    if (auto c = getIntSplat(v))
      return Poly::cst(*c);
    Operation *op = v.getDefiningOp();
    if (!op)
      return std::nullopt;
    if (auto r = dyn_cast<triton::MakeRangeOp>(op))
      return Poly::cst(r.getStart()) + Poly::atom(lane);
    if (auto s = dyn_cast<triton::SplatOp>(op))
      return scalar(s.getSrc());
    if (auto b = dyn_cast<triton::BroadcastOp>(op))
      return tensor(b.getSrc(), lane);
    if (auto e = dyn_cast<triton::ExpandDimsOp>(op)) {
      if (ty.getRank() != 2)
        return std::nullopt;
      return tensor(e.getSrc(), e.getAxis() == 1 ? DIM_I : DIM_J);
    }
    // A boundary-checked block pointer computes its offsets in i64.
    if (isa<arith::ExtSIOp, arith::IndexCastOp>(op))
      return tensor(op->getOperand(0), lane);
    if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op)) {
      auto a = tensor(op->getOperand(0), lane);
      auto b = tensor(op->getOperand(1), lane);
      if (!a || !b)
        return std::nullopt;
      if (isa<arith::AddIOp>(op))
        return *a + *b;
      if (isa<arith::SubIOp>(op))
        return *a - *b;
      return *a * *b;
    }
    if (tensorHook)
      if (auto h = tensorHook(op, lane))
        return *h;
    if (atomizeUnknownTensors)
      return Poly::atom(v);
    return std::nullopt;
  }
};

// Address of a load/store: `base` (a scalar pointer) + offset polynomial,
// where the offset is what is added at tensor level (after the splat).
struct Address {
  Value base;
  Poly offset;
};

template <typename Evaluator>
static std::optional<Address> evalAddress(Evaluator &pe, Value ptr,
                                          const void *lane) {
  Poly off;
  while (true) {
    if (!isa<RankedTensorType>(ptr.getType()))
      return std::nullopt;
    Operation *op = ptr.getDefiningOp();
    if (auto add = dyn_cast_or_null<triton::AddPtrOp>(op)) {
      auto o = pe.tensor(add.getOffset(), lane);
      if (!o)
        return std::nullopt;
      off = off + *o;
      ptr = add.getPtr();
      continue;
    }
    if (auto b = dyn_cast_or_null<triton::BroadcastOp>(op)) {
      ptr = b.getSrc();
      continue;
    }
    if (auto s = dyn_cast_or_null<triton::SplatOp>(op))
      return Address{s.getSrc(), off};
    return std::nullopt;
  }
}

// Row-major [rows, cols] tile: offset = rowPoly * rs + colPoly (column
// stride 1).  Returns rs and the remainder (offset of the tile element
// (0, 0) with the row part rowPoly * rs taken out is checked by the caller).
struct TileLayout {
  Poly rowStride;
  Poly rest; // offset with the I and J terms removed
};

static std::optional<TileLayout> splitTile(const Poly &off, const void *rowDim,
                                           const void *colDim) {
  auto r = off.split(rowDim);
  if (!r)
    return std::nullopt;
  auto c = r->second.split(colDim);
  if (!c || c->first != Poly::cst(1) || r->first.mentions(colDim) ||
      r->first.mentions(rowDim))
    return std::nullopt;
  return TileLayout{r->first, c->second};
}

//===----------------------------------------------------------------------===//
// Masks
//===----------------------------------------------------------------------===//

// One comparison `lhs pred rhs` of a mask conjunction/disjunction.
struct Cmp {
  arith::CmpIPredicate pred;
  Value lhs, rhs;
  const void *lane;
};

// Splits `cond` at `Combine` (arith.andi or arith.ori) into comparisons,
// looking through broadcast/expand_dims.
template <typename Combine>
static bool collectCmps(Value cond, const void *lane, SmallVector<Cmp> &out) {
  Operation *op = cond.getDefiningOp();
  if (!op)
    return false;
  if (isa<Combine>(op))
    return collectCmps<Combine>(op->getOperand(0), lane, out) &&
           collectCmps<Combine>(op->getOperand(1), lane, out);
  if (auto b = dyn_cast<triton::BroadcastOp>(op))
    return collectCmps<Combine>(b.getSrc(), lane, out);
  if (auto e = dyn_cast<triton::ExpandDimsOp>(op))
    return collectCmps<Combine>(e.getSrc(),
                                e.getAxis() == 1 ? DIM_I : DIM_J, out);
  if (auto c = dyn_cast<arith::CmpIOp>(op)) {
    out.push_back({c.getPredicate(), c.getLhs(), c.getRhs(), lane});
    return true;
  }
  return false;
}

// `idx < bound` terms of a load/store mask: maps the index polynomial to the
// bound polynomial.  A redundant `idx >= 0` is dropped: a boundary-checked
// block pointer tests both ends of every dimension, and the indices here are
// a tile offset (rowStart + I, or J) with rowStart a multiple of the block
// size taken from a program id, so the lower test is always true.
static std::optional<SmallVector<std::pair<Poly, Poly>>>
evalBoundMask(PolyEval &pe, Value mask) {
  SmallVector<std::pair<Poly, Poly>> r;
  if (!mask)
    return r;
  SmallVector<Cmp> cmps;
  if (!collectCmps<arith::AndIOp>(mask, DIM_I, cmps))
    return std::nullopt;
  for (auto &c : cmps) {
    auto l = pe.tensor(c.lhs, c.lane), h = pe.tensor(c.rhs, c.lane);
    if (!l || !h)
      return std::nullopt;
    if (c.pred == arith::CmpIPredicate::slt)
      r.push_back({*l, *h});
    else if (c.pred == arith::CmpIPredicate::sgt)
      r.push_back({*h, *l});
    else if (c.pred == arith::CmpIPredicate::sge && *h == Poly::cst(0))
      continue; // idx >= 0
    else if (c.pred == arith::CmpIPredicate::sle && *l == Poly::cst(0))
      continue; // 0 <= idx
    else
      return std::nullopt;
  }
  return r;
}

//===----------------------------------------------------------------------===//
// The match
//===----------------------------------------------------------------------===//

struct KeyLoop {
  scf::ForOp loop;
  triton::LoadOp kLoad, vLoad;
  Value s;      // S = dot(Q, K)
  Value masked; // S after the mask select (== s when unmasked)
  bool isMasked = false;
};

struct Window {
  std::optional<Poly> left, right; // shifts; nullopt = unbounded
  std::optional<Poly> seqK;        // from the mask, if it bounds keys by Sk
  bool boundsSk = false;           // mask excludes keys >= Sk
};

class FlashAttentionMatcher {
public:
  explicit FlashAttentionMatcher(triton::FuncOp f) : func(f) {}

  bool match();
  void rewrite();

private:
  bool fail(const Twine &why) {
    LLVM_DEBUG(llvm::dbgs() << "[kunpeng-llm] " << func.getName()
                            << ": no match: " << why << "\n");
    return false;
  }

  bool isTopLevel(const void *atom) const;
  bool topLevelPoly(const Poly &p) const;
  // Reads a row-index mask term `idx < bound`, where idx must be X + I for a
  // loop-invariant X, as a limit on rowStart + I: it returns the bound that
  // rowStart + I is below.  Stores and loads may index the same rows from
  // different origins -- a varlen kernel keeps lse for the whole batch but
  // folds the sequence's first token into the Q and O base pointers -- and
  // then state the same row limit in their own terms.
  std::optional<Poly> rowBound(const Poly &idx, const Poly &bound) const;
  bool matchEpilogue();
  bool matchLoopChain();
  bool matchLoopBody(KeyLoop &kl);
  bool matchMask(KeyLoop &kl, const Poly &colStart, Window &w);
  bool matchQ();
  bool matchKV(KeyLoop &kl, const Poly &colStart);
  bool matchFrame();
  bool matchElementKeyLoops();
  bool matchBlockKeyLoops();
  // The key index of element (.., 0) of a key loop's tile, as a polynomial in
  // the loop's induction variable (counters already bound).
  std::optional<Poly> colStartOf(scf::ForOp loop);
  // Binds the loop's iter_args other than acc/m/l, which must be affine
  // counters, to polynomials in its induction variable.
  bool bindCounters(scf::ForOp loop);
  // Recognises a paged key-row index in the loop body and binds it to
  // DIM_KROW; sets `paged` on the first loop it is found in.
  bool bindPagedRow(scf::ForOp loop, const Poly &colStart);
  Value materialize(OpBuilder &b, Location loc, const Poly &p);

  mutable triton::FuncOp func;
  Block *body = nullptr; // the block the stores and key loops live in
  PolyEval pe;

  triton::StoreOp oStore, lseStore;
  SmallVector<KeyLoop> loops; // in execution order
  Value accFinal, mFinal, lFinal;
  unsigned iAcc = 0, iM = 1, iL = 2; // positions in the loops' iter_args
  Value qLoad;
  Type elemTy;
  int64_t BM = 0, BN = 0, DK = 0, D = 0;
  DenseMap<const void *, Value> widened; // atoms sign-extended to i64
  double lseScale = 0;  // natural-log scale of the lse
  double scaleLog2 = 0; // exp2 scale of the scores

  // Results of the match, all polynomials over top-level atoms.
  Poly rowStart;
  Address q, k, v, o, lse;
  Poly qRs, kRs, vRs, oRs;
  std::optional<Poly> seqQ, seqK;
  Window win;
  // Paged K/V (flash_attn_varlen_func): key j of this sequence is row
  // (j % pageSize) of page pageTable[j / pageSize], so the K/V row it reads
  // is pageTable[j / pageSize] * pageSize + j % pageSize.
  bool paged = false;
  Address pageTable;
  int64_t pageSize = 0;
};

bool FlashAttentionMatcher::isTopLevel(const void *atom) const {
  if (atom == DIM_I || atom == DIM_J || atom == DIM_KROW)
    return false;
  Value v = Value::getFromOpaquePointer(atom);
  Block *entry = &func.getBody().front();
  if (auto ba = dyn_cast<BlockArgument>(v))
    return ba.getOwner() == entry || ba.getOwner() == body;
  Block *b = v.getDefiningOp()->getBlock();
  return b == entry || b == body;
}

bool FlashAttentionMatcher::topLevelPoly(const Poly &p) const {
  return llvm::all_of(p.atoms(), [&](const void *a) { return isTopLevel(a); });
}

std::optional<Poly> FlashAttentionMatcher::rowBound(const Poly &idx,
                                                    const Poly &bound) const {
  auto s = idx.split(DIM_I);
  if (!s || s->first != Poly::cst(1) || s->second.mentions(DIM_J) ||
      !topLevelPoly(s->second) || !topLevelPoly(bound))
    return std::nullopt;
  // X + I < bound  <=>  rowStart + I < rowStart + bound - X
  return rowStart + bound - s->second;
}

// Return the operand of commutative binary `op` that is not `x`.
template <typename OpT> static Value otherOperand(Value v, Value x) {
  auto op = v.getDefiningOp<OpT>();
  if (!op)
    return {};
  if (op->getOperand(0) == x)
    return op->getOperand(1);
  if (op->getOperand(1) == x)
    return op->getOperand(0);
  return {};
}

// Look through broadcast(expand_dims(x, axis)) of a 1-D row vector.
static Value rowVector(Value v) {
  if (auto b = v.getDefiningOp<triton::BroadcastOp>())
    v = b.getSrc();
  auto e = v.getDefiningOp<triton::ExpandDimsOp>();
  if (!e || e.getAxis() != 1)
    return {};
  return e.getSrc();
}

// Look through splat/broadcast/expand_dims down to the defining scalar.
static Value scalarSource(Value v) {
  Operation *op = v.getDefiningOp();
  if (auto s = dyn_cast_or_null<triton::SplatOp>(op))
    return s.getSrc();
  if (auto b = dyn_cast_or_null<triton::BroadcastOp>(op))
    return scalarSource(b.getSrc());
  if (auto e = dyn_cast_or_null<triton::ExpandDimsOp>(op))
    return scalarSource(e.getSrc());
  return {};
}

// Peel broadcast/expand_dims off a broadcast mask down to the 1-D condition.
static Value maskCondition(Value v) {
  Operation *op = v.getDefiningOp();
  while (op && (isa<triton::BroadcastOp>(op) || isa<triton::ExpandDimsOp>(op))) {
    v = op->getOperand(0);
    op = v.getDefiningOp();
  }
  return v;
}

// cond (1-D, possibly broadcast/expanded) is `offs_n < splat(seqlen)`.
static bool isOffsLtSeqLen(Value cond, Value offsN, Value seqLoad) {
  auto cmp = dyn_cast_or_null<arith::CmpIOp>(cond.getDefiningOp());
  return cmp && cmp.getPredicate() == arith::CmpIPredicate::slt &&
         cmp.getLhs() == offsN &&
         scalarSource(cmp.getRhs()) == seqLoad;
}

static bool isReduce(Value v, Value src, bool isMax) {
  auto r = v.getDefiningOp<triton::ReduceOp>();
  if (!r || r.getAxis() != 1 || r.getSrcs().size() != 1 ||
      r.getSrcs()[0] != src)
    return false;
  Block &body = r.getCombineOp().front();
  if (body.getOperations().size() != 2)
    return false;
  Operation &comb = body.front();
  if (isMax ? !isa<arith::MaxNumFOp, arith::MaximumFOp>(comb)
            : !isa<arith::AddFOp>(comb))
    return false;
  auto ret = dyn_cast<triton::ReduceReturnOp>(body.back());
  return ret && ret->getNumOperands() == 1 &&
         ret->getOperand(0) == comb.getResult(0) &&
         ((comb.getOperand(0) == body.getArgument(0) &&
           comb.getOperand(1) == body.getArgument(1)) ||
          (comb.getOperand(0) == body.getArgument(1) &&
           comb.getOperand(1) == body.getArgument(0)));
}

// x * c with c a splat float constant: returns c.
static std::optional<double> scaledBy(Value v, Value x) {
  Value c = otherOperand<arith::MulFOp>(v, x);
  return c ? getFloatSplat(c) : std::nullopt;
}

// select(x == -inf, 0, y): returns y.
static Value zeroIfNegInf(Value v, Value x) {
  auto sel = v.getDefiningOp<arith::SelectOp>();
  if (!sel || !isFloatSplat(sel.getTrueValue(), 0.0))
    return {};
  auto cmp = sel.getCondition().getDefiningOp<arith::CmpFOp>();
  if (!cmp || cmp.getPredicate() != arith::CmpFPredicate::OEQ)
    return {};
  Value other = cmp.getLhs() == x ? cmp.getRhs()
                : cmp.getRhs() == x ? cmp.getLhs()
                                    : Value();
  if (!other || !isFloatSplat(other, -INFINITY))
    return {};
  return sel.getFalseValue();
}

// (l == 0) | (l != l)
static bool isZeroOrNaN(Value cond, Value l) {
  auto ori = cond.getDefiningOp<arith::OrIOp>();
  if (!ori)
    return false;
  bool zero = false, nan = false;
  for (Value c : ori->getOperands()) {
    auto cmp = c.getDefiningOp<arith::CmpFOp>();
    if (!cmp)
      return false;
    if (cmp.getPredicate() == arith::CmpFPredicate::UNE &&
        cmp.getLhs() == l && cmp.getRhs() == l)
      nan = true;
    else if (cmp.getPredicate() == arith::CmpFPredicate::OEQ &&
             ((cmp.getLhs() == l && isFloatSplat(cmp.getRhs(), 0.0)) ||
              (cmp.getRhs() == l && isFloatSplat(cmp.getLhs(), 0.0))))
      zero = true;
  }
  return zero && nan;
}

bool FlashAttentionMatcher::match() {
  if (func.isExternal())
    return fail("not a kernel");
  // Either a single block, or an entry block that branches on a guard (the
  // "no query row in this program" early return of a varlen kernel) to a
  // block that only returns and to the real body.
  Region &region = func.getBody();
  if (region.hasOneBlock()) {
    body = &region.front();
  } else {
    Block *entry = &region.front();
    auto br = dyn_cast<cf::CondBranchOp>(entry->getTerminator());
    if (!br || region.getBlocks().size() != 3)
      return fail("not a single block or a guarded body");
    for (Block *succ : {br.getTrueDest(), br.getFalseDest()}) {
      if (succ->getNumArguments())
        return fail("guard successor takes arguments");
      // One successor returns at once, the other is the body.
      if (isa<triton::ReturnOp>(&succ->front()))
        continue;
      if (body)
        return fail("neither guard successor returns at once");
      body = succ;
    }
    if (!body)
      return fail("guard has no body successor");
  }

  // The kernel's only effects must be the output and lse stores.
  SmallVector<triton::StoreOp> stores;
  bool otherEffects = false;
  func.walk([&](Operation *op) {
    if (auto st = dyn_cast<triton::StoreOp>(op)) {
      stores.push_back(st);
      return;
    }
    if (isa<triton::LoadOp>(op) || op == func.getOperation() ||
        op->hasTrait<OpTrait::IsTerminator>())
      return;
    if (isa<CallOpInterface>(op))
      otherEffects = true;
    if (auto me = dyn_cast<MemoryEffectOpInterface>(op)) {
      if (!me.hasNoEffect() && !me.onlyHasEffect<MemoryEffects::Read>())
        otherEffects = true;
    } else if (!op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
      otherEffects = true;
    }
  });
  if (otherEffects)
    return fail("side effects other than stores");
  if (stores.size() != 2)
    return fail("expected exactly two stores");
  for (auto st : stores) {
    auto ty = dyn_cast<RankedTensorType>(st.getValue().getType());
    if (!ty || st.getOperation()->getBlock() != body)
      return fail("store not a top-level tensor store");
    if (ty.getRank() == 2)
      oStore = st;
    else if (ty.getRank() == 1 && ty.getElementType().isF32())
      lseStore = st;
  }
  if (!oStore || !lseStore)
    return fail("no output/lse store pair");

  return matchEpilogue() && matchLoopChain() && matchFrame();
}

// o = trunc(acc * where(l == 0 | l != l, 1, 1 / l))
// lse = where(l == 0 | l != l, -inf, m * scale + log(l))
bool FlashAttentionMatcher::matchEpilogue() {
  auto oTy = cast<RankedTensorType>(oStore.getValue().getType());
  elemTy = oTy.getElementType();
  if (!elemTy.isBF16() && !elemTy.isF16())
    return fail("output is not bf16/fp16");
  BM = oTy.getShape()[0];
  DK = oTy.getShape()[1];

  auto trunc = oStore.getValue().getDefiningOp<arith::TruncFOp>();
  if (!trunc)
    return fail("output not truncated from fp32");
  auto mul = trunc.getIn().getDefiningOp<arith::MulFOp>();
  if (!mul)
    return fail("output not acc * inv");
  Value inv;
  for (int i = 0; i < 2 && !inv; ++i) {
    if (Value r = rowVector(mul->getOperand(i))) {
      inv = r;
      accFinal = mul->getOperand(1 - i);
    }
  }
  if (!inv)
    return fail("output not scaled by a row vector");
  auto invSel = inv.getDefiningOp<arith::SelectOp>();
  if (!invSel || !isFloatSplat(invSel.getTrueValue(), 1.0))
    return fail("inv not where(..., 1, 1 / l)");
  auto div = invSel.getFalseValue().getDefiningOp<arith::DivFOp>();
  if (!div || !isFloatSplat(div.getLhs(), 1.0))
    return fail("inv not 1 / l");
  lFinal = div.getRhs();
  if (!isZeroOrNaN(invSel.getCondition(), lFinal))
    return fail("inv guard is not l == 0 | isnan(l)");

  auto lseSel = lseStore.getValue().getDefiningOp<arith::SelectOp>();
  if (!lseSel || !isFloatSplat(lseSel.getTrueValue(), -INFINITY) ||
      !isZeroOrNaN(lseSel.getCondition(), lFinal))
    return fail("lse not where(l == 0 | isnan(l), -inf, ...)");
  auto add = lseSel.getFalseValue().getDefiningOp<arith::AddFOp>();
  if (!add)
    return fail("lse not m * scale + log(l)");
  double scale = 0;
  for (int i = 0; i < 2 && !mFinal; ++i) {
    auto log = add->getOperand(i).getDefiningOp<math::LogOp>();
    auto m = add->getOperand(1 - i).getDefiningOp<arith::MulFOp>();
    if (!log || log.getOperand() != lFinal || !m)
      continue;
    for (int j = 0; j < 2 && !mFinal; ++j)
      if (auto c = getFloatSplat(m->getOperand(j))) {
        scale = *c;
        mFinal = m->getOperand(1 - j);
      }
  }
  if (!mFinal)
    return fail("lse not m * scale + log(l)");

  // acc, m and l are the results of the last key loop (in any order).  A loop
  // may carry further values (a varlen kernel carries its block counter).
  auto loop = accFinal.getDefiningOp<scf::ForOp>();
  if (!loop || loop.getNumResults() < 3 ||
      mFinal.getDefiningOp() != loop || lFinal.getDefiningOp() != loop)
    return fail("acc/m/l are not the results of a key loop");
  iAcc = cast<OpResult>(accFinal).getResultNumber();
  iM = cast<OpResult>(mFinal).getResultNumber();
  iL = cast<OpResult>(lFinal).getResultNumber();
  if (iAcc == iM || iAcc == iL || iM == iL)
    return fail("acc/m/l are not distinct loop results");
  loops.push_back({loop});
  lseScale = scale;
  return true;
}

bool FlashAttentionMatcher::matchLoopChain() {
  // Walk back through loops whose acc/m/l inits are the previous loop's
  // results.  The three must sit at the same positions in every loop of the
  // chain; any further iter_args are checked as counters by the frame.
  while (true) {
    scf::ForOp first = loops.front().loop;
    auto inits = first.getInitArgs();
    auto prev = inits[iAcc].getDefiningOp<scf::ForOp>();
    if (prev && prev.getNumResults() >= 3 &&
        inits[iAcc] == prev.getResult(iAcc) &&
        inits[iM] == prev.getResult(iM) && inits[iL] == prev.getResult(iL)) {
      loops.insert(loops.begin(), {prev});
      if (loops.size() > 2)
        return fail("more than two key loops");
      continue;
    }
    if (!isFloatSplat(inits[iAcc], 0.0) ||
        !isFloatSplat(inits[iM], -INFINITY) || !isFloatSplat(inits[iL], 0.0))
      return fail("loop inits are not acc = 0, m = -inf, l = 0");
    break;
  }
  for (auto &kl : loops) {
    if (kl.loop->getBlock() != body)
      return fail("key loop not at top level");
    if (!matchLoopBody(kl))
      return false;
  }
  return true;
}

// One online-softmax step; yields (acc', m', l').
bool FlashAttentionMatcher::matchLoopBody(KeyLoop &kl) {
  scf::ForOp loop = kl.loop;
  Value acc = loop.getRegionIterArg(iAcc), m = loop.getRegionIterArg(iM),
        l = loop.getRegionIterArg(iL);
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  Value accN = yield.getOperand(iAcc), mN = yield.getOperand(iM),
        lN = yield.getOperand(iL);

  // m' = max(m, rowmax(S'))
  Value rmax = otherOperand<arith::MaxNumFOp>(mN, m);
  if (!rmax)
    rmax = otherOperand<arith::MaximumFOp>(mN, m);
  auto red = rmax ? rmax.getDefiningOp<triton::ReduceOp>() : nullptr;
  if (!red || red.getSrcs().size() != 1 ||
      !isReduce(rmax, red.getSrcs()[0], /*isMax=*/true))
    return fail("m' is not max(m, rowmax(S))");
  kl.masked = red.getSrcs()[0];

  // acc' = dot(trunc(P), V, acc * alpha)
  auto pv = accN.getDefiningOp<triton::DotOp>();
  if (!pv)
    return fail("acc' is not a dot");
  auto pTrunc = pv.getA().getDefiningOp<arith::TruncFOp>();
  if (!pTrunc || getElementTypeOrSelf(pTrunc.getType()) != elemTy)
    return fail("P is not truncated to the input type");
  Value p = pTrunc.getIn();
  Value alpha;
  if (Value o = otherOperand<arith::MulFOp>(pv.getC(), acc))
    alpha = rowVector(o);
  if (!alpha)
    return fail("acc not rescaled by a row vector");

  // alpha = exp2((m - cur) * c), cur = m' or where(m' == -inf, 0, m')
  auto aExp = alpha.getDefiningOp<math::Exp2Op>();
  if (!aExp)
    return fail("alpha is not exp2");
  auto aMul = aExp.getOperand().getDefiningOp<arith::MulFOp>();
  auto aSub = aMul ? aMul.getLhs().getDefiningOp<arith::SubFOp>() : nullptr;
  Value aC = aMul ? aMul.getRhs() : Value();
  if (aMul && !aSub) {
    aSub = aMul.getRhs().getDefiningOp<arith::SubFOp>();
    aC = aMul.getLhs();
  }
  auto c = aC ? getFloatSplat(aC) : std::nullopt;
  if (!aSub || aSub.getLhs() != m || !c)
    return fail("alpha is not exp2((m - m') * c)");
  Value cur = aSub.getRhs();
  if (cur != mN && zeroIfNegInf(cur, mN) != mN)
    return fail("alpha uses an unexpected running max");
  if (std::abs(*c - lseScale * M_LOG2E) > 1e-6 * std::abs(*c))
    return fail("exp2 scale does not match the lse scale");
  if (scaleLog2 != 0 && *c != scaleLog2)
    return fail("key loops use different scales");
  scaleLog2 = *c;

  // l' = l * alpha + rowsum(P)
  Value lsc;
  for (int i = 0; i < 2 && !lsc; ++i) {
    auto add = lN.getDefiningOp<arith::AddFOp>();
    if (!add)
      break;
    Value a = add->getOperand(i), b = add->getOperand(1 - i);
    if (otherOperand<arith::MulFOp>(a, l) == alpha && isReduce(b, p, false))
      lsc = a;
  }
  if (!lsc)
    return fail("l' is not l * alpha + rowsum(P)");

  // P = exp2(S' * c - where(m' == -inf, 0, m' * c))
  auto pExp = p.getDefiningOp<math::Exp2Op>();
  auto pSub = pExp ? pExp.getOperand().getDefiningOp<arith::SubFOp>() : nullptr;
  if (!pSub)
    return fail("P is not exp2(S * c - max)");
  auto sc = scaledBy(pSub.getLhs(), kl.masked);
  if (!sc || *sc != *c)
    return fail("P does not scale the masked scores by c");
  Value ms = rowVector(pSub.getRhs());
  if (!ms)
    return fail("P subtracts no row vector");
  if (Value y = zeroIfNegInf(ms, mN))
    ms = y;
  auto msc = scaledBy(ms, mN);
  if (!msc || *msc != *c)
    return fail("P does not subtract m' * c");

  // S' = where(mask, -inf, S) or S;  S = dot(Q, K, 0)
  Value s = kl.masked;
  if (auto sel = s.getDefiningOp<arith::SelectOp>()) {
    if (!isFloatSplat(sel.getTrueValue(), -INFINITY))
      return fail("mask does not select -inf");
    s = sel.getFalseValue();
    kl.isMasked = true;
  }
  auto qk = s.getDefiningOp<triton::DotOp>();
  if (!qk || !isFloatSplat(qk.getC(), 0.0))
    return fail("S is not dot(Q, K)");
  kl.s = s;
  auto sTy = cast<RankedTensorType>(s.getType());
  if (sTy.getShape()[0] != BM)
    return fail("S rows != output rows");
  int64_t bn = sTy.getShape()[1];
  if (BN && bn != BN)
    return fail("key loops use different block sizes");
  BN = bn;

  if (qLoad && qLoad != qk.getA())
    return fail("key loops use different Q");
  qLoad = qk.getA();
  kl.kLoad = qk.getB().getDefiningOp<triton::LoadOp>();
  kl.vLoad = pv.getB().getDefiningOp<triton::LoadOp>();
  if (!kl.kLoad || !kl.vLoad || !loop->isAncestor(kl.kLoad) ||
      !loop->isAncestor(kl.vLoad))
    return fail("K/V are not loaded in the key loop");
  return true;
}

// Q: [BM, DK] tile at rows rowStart + I, loop invariant.
bool FlashAttentionMatcher::matchQ() {
  auto ld = qLoad.getDefiningOp<triton::LoadOp>();
  if (!ld || ld->getBlock() != body)
    return fail("Q is not a top-level load");
  if (getElementTypeOrSelf(ld.getType()) != elemTy)
    return fail("Q type != output type");
  auto addr = evalAddress(pe, ld.getPtr(), DIM_I);
  if (!addr)
    return fail("Q address is not affine");
  auto t = splitTile(addr->offset, DIM_I, DIM_J);
  if (!t)
    return fail("Q is not a row-major tile");
  q = {addr->base, t->rest - rowStart * t->rowStride};
  qRs = t->rowStride;

  auto terms = evalBoundMask(pe, ld.getMask());
  if (!terms)
    return fail("Q mask is not a conjunction of bounds");
  bool dMasked = false;
  for (auto &[idx, bound] : *terms) {
    if (idx == Poly::atom(DIM_J) && bound == Poly::cst(D))
      dMasked = true;
    else if (auto sq = rowBound(idx, bound)) {
      if (seqQ && *seqQ != *sq)
        return fail("inconsistent Q row bound");
      seqQ = *sq;
    } else
      return fail("unexpected Q mask term");
  }
  if (D != DK && (!dMasked || !ld.getOther() ||
                  !isFloatSplat(ld.getOther(), 0.0)))
    return fail("padded head_dim of Q is not zero-filled");
  return true;
}

// K: [DK, BN] tile, element (d, j) at key colStart + j;
// V: [BN, DK] tile, element (j, d) at key colStart + j.
bool FlashAttentionMatcher::matchKV(KeyLoop &kl, const Poly &colStart) {
  auto kTy = cast<RankedTensorType>(kl.kLoad.getType());
  auto vTy = cast<RankedTensorType>(kl.vLoad.getType());
  if (kTy.getElementType() != elemTy || vTy.getElementType() != elemTy ||
      kTy.getShape()[0] != DK || kTy.getShape()[1] != BN ||
      vTy.getShape()[0] != BN || vTy.getShape()[1] != DK)
    return fail("K/V tiles have unexpected shapes");

  // A paged cache gathers the K/V row of each key through a page table.
  const bool isPaged = bindPagedRow(kl.loop, colStart);
  if (paged && !isPaged)
    return fail("only some key loops read a paged K/V cache");

  auto ka = evalAddress(pe, kl.kLoad.getPtr(), DIM_I);
  auto va = evalAddress(pe, kl.vLoad.getPtr(), DIM_I);
  if (!ka || !va)
    return fail("K/V address is not affine");
  // With a paged cache the tiles are indexed by the gathered cache row
  // rather than by the key, and the key's whole contribution to the address
  // is that row, so there is no colStart term to take out.  K is [DK, BN],
  // so its key runs along J; V is [BN, DK], so its key runs along I.
  const void *kRow = isPaged ? DIM_KROW : DIM_J;
  const void *vRow = isPaged ? DIM_KROW : DIM_I;
  Poly keyOff = isPaged ? Poly::cst(0) : colStart;
  auto kt = splitTile(ka->offset, kRow, DIM_I);
  auto vt = splitTile(va->offset, vRow, DIM_J);
  if (!kt || !vt)
    return fail("K/V are not row-major tiles");
  Address kh{ka->base, kt->rest - keyOff * kt->rowStride};
  Address vh{va->base, vt->rest - keyOff * vt->rowStride};
  if (!topLevelPoly(kh.offset) || !topLevelPoly(vh.offset) ||
      !topLevelPoly(kt->rowStride) || !topLevelPoly(vt->rowStride) ||
      kh.base.getParentRegion() != &func.getBody() ||
      vh.base.getParentRegion() != &func.getBody())
    return fail("K/V head base depends on the key loop");
  if (k.base) {
    if (k.base != kh.base || k.offset != kh.offset || v.base != vh.base ||
        v.offset != vh.offset || kRs != kt->rowStride || vRs != vt->rowStride)
      return fail("key loops read different K/V");
  }
  k = kh;
  v = vh;
  kRs = kt->rowStride;
  vRs = vt->rowStride;

  // Load masks may only cut the padded head_dim (to zero) and keys >= Sk.
  auto check = [&](triton::LoadOp ld, const void *keyDim, const void *dDim,
                   bool needD) {
    auto terms = evalBoundMask(pe, ld.getMask());
    if (!terms)
      return false;
    bool dMasked = false;
    for (auto &[idx, bound] : *terms) {
      if (idx == Poly::atom(dDim) && bound == Poly::cst(D))
        dMasked = true;
      else if (!(idx == colStart + Poly::atom(keyDim) && seqK &&
                 bound == *seqK))
        return false;
    }
    if (needD && D != DK &&
        (!dMasked || !ld.getOther() || !isFloatSplat(ld.getOther(), 0.0)))
      return false;
    return true;
  };
  if (!check(kl.kLoad, DIM_J, DIM_I, /*needD=*/true) ||
      !check(kl.vLoad, DIM_I, DIM_J, /*needD=*/false))
    return fail("unexpected K/V load mask");
  return true;
}

// a < b  <=>  b > a
static arith::CmpIPredicate swapPredicate(arith::CmpIPredicate p) {
  using P = arith::CmpIPredicate;
  switch (p) {
  case P::slt:
    return P::sgt;
  case P::sgt:
    return P::slt;
  case P::sle:
    return P::sge;
  case P::sge:
    return P::sle;
  default:
    return p;
  }
}

// The score mask of a key loop: a disjunction of
//   key > min(Sk - 1, row + right)    (right-bounded)
//   key < max(row + left, 0)          (left-bounded)
//   key >= Sk                         (key range)
// with key = colStart + J and row = rowStart + I.
bool FlashAttentionMatcher::matchMask(KeyLoop &kl, const Poly &colStart,
                                      Window &w) {
  auto sel = kl.masked.getDefiningOp<arith::SelectOp>();
  SmallVector<Cmp> cmps;
  if (!collectCmps<arith::OrIOp>(sel.getCondition(), DIM_I, cmps))
    return fail("mask is not a disjunction of comparisons");
  Poly key = colStart + Poly::atom(DIM_J);
  Poly row = rowStart + Poly::atom(DIM_I);
  for (auto c : cmps) {
    auto lhs = pe.tensor(c.lhs, c.lane);
    if (!lhs || *lhs != key) {
      auto rhs = pe.tensor(c.rhs, c.lane);
      if (!rhs || *rhs != key)
        return fail("mask compares something other than the key index");
      std::swap(c.lhs, c.rhs);
      c.pred = swapPredicate(c.pred);
    }
    // Bound: a plain polynomial, or min/max of two (behind expand/broadcast).
    Value b = c.rhs;
    const void *lane = c.lane;
    while (true) {
      if (auto bc = b.getDefiningOp<triton::BroadcastOp>()) {
        b = bc.getSrc();
        continue;
      }
      if (auto e = b.getDefiningOp<triton::ExpandDimsOp>()) {
        lane = e.getAxis() == 1 ? DIM_I : DIM_J;
        b = e.getSrc();
        continue;
      }
      break;
    }
    auto rowShift = [&](const Poly &p) -> std::optional<Poly> {
      auto s = p.split(DIM_I);
      if (!s || s->first != Poly::cst(1))
        return std::nullopt;
      return s->second - rowStart;
    };
    if (c.pred == arith::CmpIPredicate::sgt) {
      auto mn = b.getDefiningOp<arith::MinSIOp>();
      if (!mn) {
        auto p = pe.tensor(b, lane);
        if (!p || p->mentions(DIM_I) || p->mentions(DIM_J))
          return fail("unsupported right bound");
        w.seqK = *p + Poly::cst(1);
        w.boundsSk = true;
        continue;
      }
      auto a0 = pe.tensor(mn.getLhs(), lane), a1 = pe.tensor(mn.getRhs(), lane);
      if (!a0 || !a1)
        return fail("right bound not polynomial");
      if (a0->mentions(DIM_I))
        std::swap(a0, a1);
      auto sh = rowShift(*a1);
      if (!sh || a0->mentions(DIM_I) || a0->mentions(DIM_J))
        return fail("right bound not min(Sk - 1, row + shift)");
      w.right = *sh;
      w.seqK = *a0 + Poly::cst(1);
      w.boundsSk = true;
    } else if (c.pred == arith::CmpIPredicate::slt) {
      std::optional<Poly> p;
      if (auto mx = b.getDefiningOp<arith::MaxSIOp>()) {
        auto a0 = pe.tensor(mx.getLhs(), lane);
        auto a1 = pe.tensor(mx.getRhs(), lane);
        if (!a0 || !a1)
          return fail("left bound not polynomial");
        if (*a0 == Poly::cst(0))
          std::swap(a0, a1);
        if (*a1 != Poly::cst(0))
          return fail("left bound not max(row + shift, 0)");
        p = a0;
      } else {
        p = pe.tensor(b, lane);
      }
      auto sh = p ? rowShift(*p) : std::nullopt;
      if (!sh)
        return fail("left bound not row + shift");
      w.left = *sh;
    } else if (c.pred == arith::CmpIPredicate::sge) {
      auto p = pe.tensor(b, lane);
      if (!p || p->mentions(DIM_I) || p->mentions(DIM_J))
        return fail("unsupported key bound");
      w.seqK = *p;
      w.boundsSk = true;
    } else {
      return fail("unsupported mask predicate");
    }
  }
  return true;
}

// Key-loop frame.  Two chained loops are the flash-attention split into a
// masked tail walked backwards from E and an interior walked forwards:
//   L1: for i in [0, M) step BN: key0 = E - i - BN    (M = min(E - Lo, K))
//   L2: for i in [Lo, E - M) step BN: key0 = i
// one loop is for i in [Lo, E) step BN: key0 = i.  The keys visited are
// [Lo, E); with
//   Lo = 0 or round_down(max(rowStart + left, 0))
//   E  = max(E', Lo),  E' = [round_up](Sk or min(Sk, rowStart + BM + right))
// they cover every key the mask admits for rows [rowStart, rowStart + BM).
bool FlashAttentionMatcher::matchElementKeyLoops() {
  auto stepOf = [&](scf::ForOp l) { return getIntSplat(l.getStep()); };
  for (auto &kl : loops)
    if (stepOf(kl.loop) != BN)
      return fail("key loop step != BLOCK_N");

  // Lo / E from the (last) forward loop.
  scf::ForOp fwd = loops.back().loop;
  Value lo = fwd.getLowerBound();
  Value e;
  Value mCount; // L1 trip range M
  if (loops.size() == 2) {
    scf::ForOp tail = loops.front().loop;
    if (getIntSplat(tail.getLowerBound()) != 0)
      return fail("masked tail loop does not start at 0");
    mCount = tail.getUpperBound();
    auto ub = pe.scalar(fwd.getUpperBound());
    auto mPoly = pe.scalar(mCount);
    // fwd ub = E - M: find E as the atom with ub + M == E.
    Poly eP = *ub + *mPoly;
    auto atoms = eP.atoms();
    if (atoms.size() != 1 || eP != Poly::atom(atoms[0]))
      return fail("forward loop does not end where the tail loop starts");
    e = Value::getFromOpaquePointer(atoms[0]);
  } else {
    e = fwd.getUpperBound();
  }
  if (!isTopLevel(lo.getAsOpaquePointer()) && !getIntSplat(lo))
    return fail("loop bounds not top level");

  auto loP = *pe.scalar(lo);
  auto eP = *pe.scalar(e);

  // Key index per loop, then masks and K/V.
  SmallVector<Poly> colStarts;
  for (auto &kl : loops) {
    Poly iv = Poly::atom(kl.loop.getInductionVar());
    bool isTail = loops.size() == 2 && kl.loop == loops.front().loop;
    colStarts.push_back(isTail ? eP - iv - Poly::cst(BN) : iv);
  }
  // Masks first: they provide Sk/Sq for the K/V mask checks.
  bool first = true;
  for (auto [kl, cs] : llvm::zip(loops, colStarts)) {
    if (!kl.isMasked)
      continue;
    Window w;
    if (!matchMask(kl, cs, w))
      return false;
    if (!first && (w.left.has_value() != win.left.has_value() ||
                   w.right.has_value() != win.right.has_value() ||
                   (w.left && *w.left != *win.left) ||
                   (w.right && *w.right != *win.right) ||
                   w.boundsSk != win.boundsSk ||
                   (w.seqK && win.seqK && *w.seqK != *win.seqK)))
      return fail("key loops use different masks");
    if (first)
      win = w;
    first = false;
  }
  if (win.seqK)
    seqK = win.seqK;

  // Lo.
  bool loRounded = false;
  std::optional<Poly> loArg; // Lo = [round_down](max(loArg, 0))
  if (loP != Poly::cst(0)) {
    Value x = lo;
    if (auto mul = x.getDefiningOp<arith::MulIOp>()) {
      auto div = mul.getLhs().getDefiningOp<arith::DivSIOp>();
      if (div && getIntSplat(mul.getRhs()) == BN &&
          getIntSplat(div.getRhs()) == BN) {
        x = div.getLhs();
        loRounded = true;
      }
    }
    auto mx = x.getDefiningOp<arith::MaxSIOp>();
    if (!mx)
      return fail("Lo is not max(row + shift, 0)");
    auto a0 = pe.scalar(mx.getLhs()), a1 = pe.scalar(mx.getRhs());
    if (*a0 == Poly::cst(0))
      std::swap(a0, a1);
    if (*a1 != Poly::cst(0))
      return fail("Lo is not max(row + shift, 0)");
    loArg = a0;
  }

  // E = max(E', Lo) or E'; E' = [round_up](Sk | min(Sk, x)).
  Value ex = e;
  if (auto mx = ex.getDefiningOp<arith::MaxSIOp>()) {
    if (mx.getRhs() == lo || *pe.scalar(mx.getRhs()) == loP)
      ex = mx.getLhs();
    else if (mx.getLhs() == lo || *pe.scalar(mx.getLhs()) == loP)
      ex = mx.getRhs();
  }
  bool eRounded = false;
  if (auto mul = ex.getDefiningOp<arith::MulIOp>()) {
    auto div = mul.getLhs().getDefiningOp<arith::DivSIOp>();
    if (div && getIntSplat(mul.getRhs()) == BN &&
        getIntSplat(div.getRhs()) == BN) {
      auto num = pe.scalar(div.getLhs());
      // round_up(x) = (x + BN - 1) / BN * BN; strip the BN - 1.
      Poly x = *num - Poly::cst(BN - 1);
      auto add = div.getLhs().getDefiningOp<arith::AddIOp>();
      if (!add)
        return fail("E rounding is not (x + BN - 1) / BN * BN");
      Value xv = getIntSplat(add.getRhs()) == BN - 1   ? add.getLhs()
                 : getIntSplat(add.getLhs()) == BN - 1 ? add.getRhs()
                                                       : Value();
      if (!xv || *pe.scalar(xv) != x)
        return fail("E rounding is not (x + BN - 1) / BN * BN");
      ex = xv;
      eRounded = true;
    }
  }
  std::optional<Poly> eArg; // E' = min(Sk, eArg)
  if (auto mn = ex.getDefiningOp<arith::MinSIOp>()) {
    auto a0 = pe.scalar(mn.getLhs()), a1 = pe.scalar(mn.getRhs());
    if (!seqK)
      return fail("E = min(...) without a known Sk");
    if (*a1 == *seqK)
      std::swap(a0, a1);
    if (*a0 != *seqK)
      return fail("E is not min(Sk, x)");
    eArg = a1;
  } else {
    auto sk = *pe.scalar(ex);
    if (seqK && *seqK != sk)
      return fail("E is not Sk");
    seqK = sk;
  }
  if (!seqK || !topLevelPoly(*seqK))
    return fail("Sk unknown");

  // Coverage: Lo <= lb(rowStart), E >= rb(rowStart + BM - 1) + 1.
  if (win.left) {
    if (!loArg || *loArg != rowStart + *win.left)
      return fail("Lo does not match the left window bound");
  } else if (loArg) {
    return fail("Lo > 0 without a left window bound");
  }
  if (win.right) {
    if (!eArg || *eArg != rowStart + Poly::cst(BM) + *win.right)
      return fail("E does not match the right window bound");
  } else if (eArg) {
    return fail("E < Sk without a right window bound");
  }
  // Keys >= Sk visited (E rounded up) must be masked.
  for (auto &kl : loops)
    if (eRounded && kl.isMasked && !win.boundsSk)
      return fail("rounded-up key range is not masked by Sk");

  // The tail loop is the masked one; its trip range must be M = min(E - Lo,
  // K) so that it never walks below Lo.
  if (loops.size() == 2) {
    auto &tail = loops.front();
    if (!tail.isMasked)
      return fail("tail key loop is not masked");
    auto mn = mCount.getDefiningOp<arith::MinSIOp>();
    if (!mn)
      return fail("tail trip range is not min(E - Lo, K)");
    auto a0 = pe.scalar(mn.getLhs()), a1 = pe.scalar(mn.getRhs());
    if (!a0->getConst())
      std::swap(a0, a1);
    if (!a0->getConst() || *a1 != eP - loP)
      return fail("tail trip range is not min(E - Lo, K)");
    int64_t kCols = *a0->getConst();
    if (!loops.back().isMasked) {
      // The interior keys j < E - K are not masked, so they must be visible
      // to every row.  With E <= min(Sk, rowStart + BM + right) (+ BN - 1
      // if rounded up), K >= BM (if right-bounded) + BN - 1 (if rounded)
      // gives j < Sk and j <= rowStart + right; there must be no left bound.
      int64_t need = (win.right ? BM : 0) + (eRounded ? BN - 1 : 0);
      if (win.left || kCols < need)
        return fail("unmasked interior loop may see invisible keys");
    }
  } else if (!loops.front().isMasked) {
    if (win.left || win.right || loArg || eRounded || loP != Poly::cst(0))
      return fail("unmasked loop over a partial key range");
  }

  for (auto [kl, cs] : llvm::zip(loops, colStarts))
    if (!matchKV(kl, cs))
      return false;

  return true;
}

// Binds every iter_arg other than acc/m/l to a polynomial in the induction
// variable.  Such an arg must be an affine counter: an integer starting at a
// loop-invariant value and advancing by a constant every iteration, which is
// how a varlen kernel carries the index of the key block it is on.
bool FlashAttentionMatcher::bindCounters(scf::ForOp loop) {
  auto step = getIntSplat(loop.getStep());
  auto lb = pe.scalar(loop.getLowerBound());
  if (!step || *step == 0 || !lb)
    return fail("key loop step/lower bound is not affine");
  Poly iv = Poly::atom(loop.getInductionVar());
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  for (unsigned i = 0, e = loop.getNumRegionIterArgs(); i < e; ++i) {
    if (i == iAcc || i == iM || i == iL)
      continue;
    Value arg = loop.getRegionIterArg(i);
    if (!arg.getType().isIntOrIndex())
      return fail("extra iter_arg is not an integer counter");
    auto init = pe.scalar(loop.getInitArgs()[i]);
    if (!init || !topLevelPoly(*init))
      return fail("counter init is not loop invariant");
    // yield = arg + c, with c constant and a multiple of the step.
    Value next = yield.getOperand(i);
    std::optional<int64_t> c;
    if (auto add = next.getDefiningOp<arith::AddIOp>()) {
      if (Value o = otherOperand<arith::AddIOp>(next, arg))
        c = getIntSplat(o);
    } else if (auto sub = next.getDefiningOp<arith::SubIOp>()) {
      if (sub.getLhs() == arg)
        if (auto o = getIntSplat(sub.getRhs()))
          c = -*o;
    }
    if (!c || *c % *step != 0)
      return fail("extra iter_arg is not an affine counter");
    // After n steps the induction variable is lb + n * step, so the counter
    // is init + (c / step) * (iv - lb).
    Poly k = Poly::cst(*c / *step);
    pe.bound[arg] = *init + k * (iv - *lb);
  }
  return true;
}

// The key index tensor of a key loop is splat(colStart) + arange(0, BLOCK_N);
// its polynomial over DIM_J is colStart + J.  Looking it up in the body
// rather than deriving it from the loop bounds covers both an element-space
// loop (colStart = induction variable) and a block-space one (colStart =
// block index * BLOCK_N).  A wrong pick makes the mask and K/V checks fail
// rather than miscompile, so requiring a unique candidate is enough.
std::optional<Poly> FlashAttentionMatcher::colStartOf(scf::ForOp loop) {
  std::optional<Poly> found;
  for (Operation &op : loop.getBody()->without_terminator()) {
    auto add = dyn_cast<arith::AddIOp>(&op);
    if (!add)
      continue;
    auto ty = dyn_cast<RankedTensorType>(add.getType());
    if (!ty || ty.getRank() != 1 || ty.getShape()[0] != BN ||
        !ty.getElementType().isInteger())
      continue;
    auto p = pe.tensor(add.getResult(), DIM_J);
    if (!p)
      continue;
    auto s = p->split(DIM_J);
    if (!s || s->first != Poly::cst(1) || s->second.mentions(DIM_I))
      continue;
    if (found && *found != s->second)
      return std::nullopt; // ambiguous
    found = s->second;
  }
  return found;
}

// Key loops that walk key *blocks*, one per iteration, which is what a
// varlen kernel generates:
//
//   for step in range(0, M):                 # diagonal blocks, masked
//     n = n_block_max - 1 - step             # carried in an iter_arg
//   for n in range(n_block_max - M - 1, n_block_min - 1, -1):
//
// with keys n * BLOCK_N .. n * BLOCK_N + BLOCK_N - 1, n_block_max =
// min(cdiv(Sk, BN), cdiv(row + BM + right, BN)) and n_block_min =
// max((row + left) / BN, 0).  (Triton normalises the descending loop to an
// ascending one, so the direction is read off colStart, not the bounds.)
// Both ends of the key range are therefore rounded out to whole blocks.
bool FlashAttentionMatcher::matchBlockKeyLoops() {
  // x from cdiv(x, BN) = (x + BN - 1) / BN, or from x / BN.
  auto divByBN = [&](Value v, bool ceil) -> std::optional<Poly> {
    auto div = v.getDefiningOp<arith::DivSIOp>();
    if (!div || getIntSplat(div.getRhs()) != BN)
      return std::nullopt;
    if (!ceil)
      return pe.scalar(div.getLhs());
    auto add = div.getLhs().getDefiningOp<arith::AddIOp>();
    if (!add)
      return std::nullopt;
    Value x = getIntSplat(add.getRhs()) == BN - 1   ? add.getLhs()
              : getIntSplat(add.getLhs()) == BN - 1 ? add.getRhs()
                                                    : Value();
    return x ? pe.scalar(x) : std::nullopt;
  };
  // A key bound that is `blocks * BLOCK_N` for one SSA block count.
  auto blockCount = [&](const Poly &p) -> Value {
    if (p.terms.size() != 1)
      return {};
    auto &t = *p.terms.begin();
    if (t.first.size() != 1 || t.second != BN)
      return {};
    return Value::getFromOpaquePointer(t.first[0]);
  };

  // Each loop's key index and the contiguous key interval it visits.
  SmallVector<Poly> colStarts;
  SmallVector<std::pair<Poly, Poly>> span;
  for (auto &kl : loops) {
    if (!bindCounters(kl.loop))
      return false;
    auto cs = colStartOf(kl.loop);
    if (!cs)
      return fail("no unique key index in the key loop");
    auto step = getIntSplat(kl.loop.getStep());
    auto lb = pe.scalar(kl.loop.getLowerBound());
    auto ub = pe.scalar(kl.loop.getUpperBound());
    auto sp = cs->split(kl.loop.getInductionVar().getAsOpaquePointer());
    if (!step || !lb || !ub || !sp)
      return fail("key loop bounds are not affine");
    auto b = sp->first.getConst();
    if (!b || (*b * *step != BN && *b * *step != -BN))
      return fail("key loop does not advance by one key block");
    // colStart = A + B * iv over iv in [lb, ub) step `step`; the step moves
    // the window by exactly BLOCK_N, so the visited keys are contiguous and
    // the last iteration is at ub - step.
    const Poly &A = sp->second;
    Poly B = Poly::cst(*b);
    Poly lo = *b * *step > 0 ? A + B * *lb : A + B * *ub + Poly::cst(BN);
    Poly hi = *b * *step > 0 ? A + B * *ub : A + B * *lb + Poly::cst(BN);
    if (!topLevelPoly(lo) || !topLevelPoly(hi))
      return fail("key range is not loop invariant");
    colStarts.push_back(*cs);
    span.push_back({lo, hi});
  }

  // Masks first: they provide Sk/Sq for the K/V mask checks.
  bool first = true;
  for (auto [kl, cs] : llvm::zip(loops, colStarts)) {
    if (!kl.isMasked)
      continue;
    Window w;
    if (!matchMask(kl, cs, w))
      return false;
    if (!first && (w.left.has_value() != win.left.has_value() ||
                   w.right.has_value() != win.right.has_value() ||
                   (w.left && *w.left != *win.left) ||
                   (w.right && *w.right != *win.right) ||
                   w.boundsSk != win.boundsSk ||
                   (w.seqK && win.seqK && *w.seqK != *win.seqK)))
      return fail("key loops use different masks");
    if (first)
      win = w;
    first = false;
  }
  if (win.seqK)
    seqK = win.seqK;

  // The loops must cover one contiguous range [Lo, E).
  Poly Lo = span[0].first, E = span[0].second;
  unsigned iInterior = 0;
  if (loops.size() == 2) {
    if (span[0].first == span[1].second) {
      Lo = span[1].first;
      E = span[0].second;
      iInterior = 1;
    } else if (span[1].first == span[0].second) {
      Lo = span[0].first;
      E = span[1].second;
      iInterior = 0;
    } else {
      return fail("key loops do not cover adjacent key ranges");
    }
  }

  // E = cdiv(Sk, BN) * BN, or min(cdiv(Sk, BN), cdiv(eArg, BN)) * BN.
  std::optional<Poly> eArg;
  Value nbm = blockCount(E);
  if (!nbm)
    return fail("key range end is not a block count");
  if (auto mn = nbm.getDefiningOp<arith::MinSIOp>()) {
    auto a0 = divByBN(mn.getLhs(), /*ceil=*/true);
    auto a1 = divByBN(mn.getRhs(), /*ceil=*/true);
    if (!a0 || !a1)
      return fail("E is not min(cdiv(Sk), cdiv(x))");
    if (!seqK)
      return fail("E = min(...) without a known Sk");
    if (*a1 == *seqK)
      std::swap(a0, a1);
    if (*a0 != *seqK)
      return fail("E is not min(Sk, x)");
    eArg = a1;
  } else {
    auto sk = divByBN(nbm, /*ceil=*/true);
    if (!sk)
      return fail("E is not cdiv(Sk, BLOCK_N)");
    if (seqK && *seqK != *sk)
      return fail("E is not Sk");
    seqK = sk;
  }
  if (!seqK || !topLevelPoly(*seqK))
    return fail("Sk unknown");

  // Lo = 0, or max(loArg / BN, 0) * BN.
  std::optional<Poly> loArg;
  if (!Lo.terms.empty()) {
    Value nbmin = blockCount(Lo);
    auto mx = nbmin ? nbmin.getDefiningOp<arith::MaxSIOp>() : nullptr;
    if (!mx)
      return fail("Lo is not max((row + shift) / BN, 0)");
    auto a0 = pe.scalar(mx.getLhs()), a1 = pe.scalar(mx.getRhs());
    Value x = a0 && *a0 == Poly::cst(0)   ? mx.getRhs()
              : a1 && *a1 == Poly::cst(0) ? mx.getLhs()
                                          : Value();
    if (!x)
      return fail("Lo is not max(x, 0)");
    loArg = divByBN(x, /*ceil=*/false);
    if (!loArg)
      return fail("Lo is not max(x / BN, 0)");
  }

  // Coverage: Lo <= lb(rowStart) and E >= rb(rowStart + BM - 1) + 1.
  if (win.left) {
    if (!loArg || *loArg != rowStart + *win.left)
      return fail("Lo does not match the left window bound");
  } else if (loArg) {
    return fail("Lo > 0 without a left window bound");
  }
  if (win.right) {
    if (!eArg || *eArg != rowStart + Poly::cst(BM) + *win.right)
      return fail("E does not match the right window bound");
  } else if (eArg) {
    return fail("E < Sk without a right window bound");
  }
  // E is rounded up to a whole block, so keys >= Sk are visited and every
  // loop that can reach them has to mask them.
  for (auto &kl : loops)
    if (kl.isMasked && !win.boundsSk)
      return fail("rounded-up key range is not masked by Sk");

  if (loops.size() == 2 && !loops[iInterior].isMasked) {
    // The keys the unmasked loop visits are not tested against the window,
    // so they must be visible to every row of the block.  Its range ends
    // where the masked loop starts, i.e. at E - K * BN with the masked loop's
    // trip range min(E - Lo, K) blocks: the first branch of that min leaves
    // the unmasked loop empty, and the second has to leave only keys j with
    // j < Sk and j <= rowStart + right.  Rounding E up costs BN - 1 of that
    // margin, and a right bound another BM, since E follows the *last* row.
    KeyLoop &masked = loops[1 - iInterior];
    if (!masked.isMasked)
      return fail("neither key loop is masked");
    auto lbm = pe.scalar(masked.loop.getLowerBound());
    auto mn = masked.loop.getUpperBound().getDefiningOp<arith::MinSIOp>();
    if (!mn || !lbm || *lbm != Poly::cst(0))
      return fail("masked key loop trip range is not min(E - Lo, K)");
    auto a0 = pe.scalar(mn.getLhs()), a1 = pe.scalar(mn.getRhs());
    if (!a0 || !a1)
      return fail("masked key loop trip range is not affine");
    if (!a0->getConst())
      std::swap(a0, a1);
    auto K = a0->getConst();
    if (!K || *a1 * Poly::cst(BN) != E - Lo)
      return fail("masked key loop trip range is not min(E - Lo, K)");
    int64_t need = (win.right ? BM : 0) + BN - 1;
    if (win.left || *K * BN < need)
      return fail("unmasked interior loop may see invisible keys");
  }

  for (auto [kl, cs] : llvm::zip(loops, colStarts))
    if (!matchKV(kl, cs))
      return false;
  return true;
}

// A paged K/V cache addresses key j of the sequence through a page table:
//
//   row(j) = page_table[j / page_size] * page_size + j % page_size
//
// and then reads row(j) of the cache.  Recognising that whole expression and
// binding it to DIM_KROW makes a paged K/V load a row-major tile indexed by
// the cache row, which is what matchKV already knows how to check.  Returns
// true if the loop reads K/V through such an index.
bool FlashAttentionMatcher::bindPagedRow(scf::ForOp loop,
                                         const Poly &colStart) {
  Poly key = colStart + Poly::atom(DIM_J);
  for (Operation &op : loop.getBody()->without_terminator()) {
    auto add = dyn_cast<arith::AddIOp>(&op);
    if (!add)
      continue;
    auto ty = dyn_cast<RankedTensorType>(add.getType());
    if (!ty || ty.getRank() != 1 || ty.getShape()[0] != BN ||
        !ty.getElementType().isInteger())
      continue;
    for (int i = 0; i < 2; ++i) {
      auto mul = add->getOperand(i).getDefiningOp<arith::MulIOp>();
      auto rem = add->getOperand(1 - i).getDefiningOp<arith::RemSIOp>();
      if (!mul || !rem)
        continue;
      // page_block * page_size, either operand order
      Value pageBlock = mul.getLhs();
      auto ps = getIntSplat(mul.getRhs());
      if (!ps) {
        pageBlock = mul.getRhs();
        ps = getIntSplat(mul.getLhs());
      }
      if (!ps || *ps <= 0 || getIntSplat(rem.getRhs()) != *ps)
        continue;
      auto remKey = pe.tensor(rem.getLhs(), DIM_J);
      if (!remKey || *remKey != key)
        continue;
      // page_block = page_table[key / page_size]
      auto load = pageBlock.getDefiningOp<triton::LoadOp>();
      auto ap = load ? load.getPtr().getDefiningOp<triton::AddPtrOp>()
                     : nullptr;
      auto sp = ap ? ap.getPtr().getDefiningOp<triton::SplatOp>() : nullptr;
      auto dv = ap ? ap.getOffset().getDefiningOp<arith::DivSIOp>() : nullptr;
      if (!sp || !dv || getIntSplat(dv.getRhs()) != *ps)
        continue;
      auto divKey = pe.tensor(dv.getLhs(), DIM_J);
      if (!divKey || *divKey != key)
        continue;
      // The page-table load may only be masked by the key range; keys past
      // the end then read page 0, and are masked out of the scores anyway.
      // Its mask is 1-D, so evalBoundMask names the lane DIM_I.
      Poly keyLane = colStart + Poly::atom(DIM_I);
      auto terms = evalBoundMask(pe, load.getMask());
      if (!terms)
        continue;
      bool maskOk = true;
      for (auto &[idx, bound] : *terms)
        if (!(idx == keyLane && seqK && bound == *seqK))
          maskOk = false;
      if (!maskOk)
        continue;
      Address pt{sp.getSrc(), Poly::cst(0)};
      if (paged && (pt.base != pageTable.base || pageSize != *ps))
        return fail("key loops use different page tables");
      paged = true;
      pageTable = pt;
      pageSize = *ps;
      pe.bound[add.getResult()] = Poly::atom(DIM_KROW);
      return true;
    }
  }
  return false;
}

bool FlashAttentionMatcher::matchFrame() {
  // Rows are rowStart + I as seen by the lse store; everything else is
  // checked against that.
  auto la = evalAddress(pe, lseStore.getPtr(), DIM_I);
  if (!la)
    return fail("lse address is not affine");
  auto ls = la->offset.split(DIM_I);
  if (!ls || ls->first != Poly::cst(1))
    return fail("lse is not contiguous over rows");
  rowStart = ls->second;
  lse = {la->base, Poly::cst(0)};
  if (!topLevelPoly(rowStart))
    return fail("row start is not loop invariant");

  // Output tile and row/head_dim bounds from the store masks.
  auto oa = evalAddress(pe, oStore.getPtr(), DIM_I);
  auto ot = oa ? splitTile(oa->offset, DIM_I, DIM_J) : std::nullopt;
  if (!ot)
    return fail("output is not a row-major tile");
  o = {oa->base, ot->rest - rowStart * ot->rowStride};
  oRs = ot->rowStride;
  D = DK;
  for (auto st : {oStore, lseStore}) {
    auto terms = evalBoundMask(pe, st.getMask());
    if (!terms)
      return fail("store mask is not a conjunction of bounds");
    for (auto &[idx, bound] : *terms) {
      if (st == oStore && idx == Poly::atom(DIM_J) && bound.getConst()) {
        D = *bound.getConst();
      } else if (auto sq = rowBound(idx, bound)) {
        if (seqQ && *seqQ != *sq)
          return fail("inconsistent row bounds");
        seqQ = *sq;
      } else {
        return fail("unexpected store mask term");
      }
    }
  }
  if (D <= 0 || D > DK || D % 16 != 0 || D > 256)
    return fail("unsupported head_dim");
  if (D != DK) {
    // The padded columns must not be stored.
    auto terms = evalBoundMask(pe, oStore.getMask());
    if (!terms || !llvm::any_of(*terms, [&](auto &t) {
          return t.first == Poly::atom(DIM_J);
        }))
      return fail("padded head_dim is stored");
  }

  if (!matchQ())
    return false;

  // The key loops either step through keys (BLOCK_N at a time) or through
  // key blocks (one at a time, the varlen form); the two are told apart by
  // their step and checked separately.
  auto unitStep = [&](KeyLoop &kl) {
    auto s = getIntSplat(kl.loop.getStep());
    return s && (*s == 1 || *s == -1);
  };
  if (llvm::all_of(loops, unitStep) ? !matchBlockKeyLoops()
                                   : !matchElementKeyLoops())
    return false;

  // Head bases and strides must be computable at the end of the kernel.
  SmallVector<Address *> bases = {&q, &k, &v, &o, &lse};
  if (paged)
    bases.push_back(&pageTable);
  for (Address *a : bases)
    if (a->base.getParentRegion() != &func.getBody() ||
        !topLevelPoly(a->offset))
      return fail("head base is not loop invariant");
  for (Poly *p : {&qRs, &kRs, &vRs, &oRs})
    if (!topLevelPoly(*p))
      return fail("row stride is not loop invariant");
  for (auto *p : {&win.left, &win.right, &seqQ})
    if (*p && !topLevelPoly(**p))
      return fail("window/seqlen is not loop invariant");
  return true;
}

Value FlashAttentionMatcher::materialize(OpBuilder &b, Location loc,
                                         const Poly &p) {
  auto i64 = b.getI64Type();
  auto atom64 = [&](const void *a) -> Value {
    Value &v64 = widened[a];
    if (!v64) {
      Value v = Value::getFromOpaquePointer(a);
      if (v.getType().isIndex())
        v64 = b.create<arith::IndexCastOp>(loc, i64, v);
      else if (v.getType().getIntOrFloatBitWidth() < 64)
        v64 = b.create<arith::ExtSIOp>(loc, i64, v);
      else
        v64 = v;
    }
    return v64;
  };
  Value sum;
  int64_t constant = 0;
  for (auto &[mono, c] : p.terms) {
    if (mono.empty()) {
      constant = c;
      continue;
    }
    Value t;
    for (auto *a : mono)
      t = t ? b.create<arith::MulIOp>(loc, t, atom64(a)) : atom64(a);
    if (c == -1 && sum) {
      sum = b.create<arith::SubIOp>(loc, sum, t);
      continue;
    }
    if (c != 1)
      t = b.create<arith::MulIOp>(
          loc, t, b.create<arith::ConstantIntOp>(loc, c, 64));
    sum = sum ? b.create<arith::AddIOp>(loc, sum, t) : t;
  }
  Value cst = b.create<arith::ConstantIntOp>(loc, constant, 64);
  if (!sum)
    return cst;
  return constant ? b.create<arith::AddIOp>(loc, sum, cst) : sum;
}

void FlashAttentionMatcher::rewrite() {
  auto ret = cast<triton::ReturnOp>(body->getTerminator());
  OpBuilder b(ret);
  Location loc = oStore.getLoc();
  auto i64 = b.getI64Type();

  auto addr = [&](const Address &a) -> Value {
    Value p = a.base;
    if (!a.offset.terms.empty())
      p = b.create<triton::AddPtrOp>(loc, p.getType(), p,
                                     materialize(b, loc, a.offset));
    return b.create<triton::PtrToIntOp>(loc, i64, p);
  };
  auto num = [&](const Poly &p) { return materialize(b, loc, p); };
  auto c64 = [&](int64_t c) -> Value {
    return b.create<arith::ConstantIntOp>(loc, c, 64);
  };

  Poly sq = seqQ ? *seqQ : rowStart + Poly::cst(BM);
  int64_t flags = (win.left ? 1 : 0) | (win.right ? 2 : 0);
  SmallVector<Value> args;
  std::string name;
  if (paged) {
    // The cache row of key j is page_table[j / page_size] * page_size +
    // j % page_size, so one page is page_size rows apart.
    args = {addr(q),          num(qRs),
            addr(k),          num(Poly::cst(pageSize) * kRs),
            num(kRs),         addr(v),
            num(Poly::cst(pageSize) * vRs),
            num(vRs),         addr(o),
            num(oRs),         addr(lse),
            addr(pageTable),  c64(pageSize),
            num(rowStart),    c64(BM),
            num(sq),          num(*seqK),
            c64(flags),
            win.left ? num(*win.left) : c64(0),
            win.right ? num(*win.right) : c64(0),
            c64(D)};
    name = elemTy.isBF16() ? "kunpeng_flash_attention_varlen_forward_bf16"
                           : "kunpeng_flash_attention_varlen_forward_f16";
  } else {
    args = {addr(q),   num(qRs), addr(k),   num(kRs),      addr(v),    num(vRs),
            addr(o),   num(oRs), addr(lse), num(rowStart), c64(BM),    num(sq),
            num(*seqK), c64(flags),
            win.left ? num(*win.left) : c64(0),
            win.right ? num(*win.right) : c64(0),
            c64(D)};
    name = elemTy.isBF16() ? "kunpeng_flash_attention_forward_bf16"
                           : "kunpeng_flash_attention_forward_f16";
  }
  args.push_back(b.create<arith::ConstantOp>(
      loc, b.getF32FloatAttr(static_cast<float>(scaleLog2))));

  auto module = func->getParentOfType<ModuleOp>();
  auto fnTy = b.getFunctionType(ValueRange(args).getTypes(), {});
  auto callee = module.lookupSymbol<func::FuncOp>(name);
  if (!callee) {
    OpBuilder mb(module.getBodyRegion());
    mb.setInsertionPointToStart(module.getBody());
    callee = mb.create<func::FuncOp>(module.getLoc(), name, fnTy);
    callee.setPrivate();
  }
  b.create<func::CallOp>(loc, callee, args);

  LLVM_DEBUG(llvm::dbgs() << "[kunpeng-llm] " << func.getName()
                          << ": replaced by " << name << " (BLOCK_M=" << BM
                          << ", BLOCK_N=" << BN << ", head_dim=" << D
                          << ")\n");

  // Drop the Triton implementation: the stores, then everything dead.
  oStore.erase();
  lseStore.erase();
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> dead;
    body->walk<WalkOrder::PostOrder>([&](Operation *op) {
      if (op->getParentOp() == func.getOperation() && isOpTriviallyDead(op))
        dead.push_back(op);
    });
    for (Operation *op : llvm::reverse(dead)) {
      op->erase();
      changed = true;
    }
  }
}

//===----------------------------------------------------------------------===//
// Flash-MLA decode matcher
//===----------------------------------------------------------------------===//

// Recognise the FlagGems flash-MLA decode kernel (one program = BLOCK_H
// query heads of one batch at a single query position, s_q == 1) and replace
// its body by a call to kunpeng_flash_mla_bf16:
//
//   kunpeng_flash_mla_bf16(q, q_hs, kv, kv_rs, block_table, seqlen, o,
//                          o_hs, num_heads, page_size, dv, d, scale)
//
// (see third_party/kunpeng/llm/kunpeng_llm.h).  The kernel has the shape
//
//   q_nope, q_pe = split loads of Q [BLOCK_H, d]      (no mask: EVEN_H)
//   acc = 0, l = 0, m = -inf, offs = arange(BLOCK_N)
//   for i in [0, seqlen // BLOCK_N):                  (one page per iter)
//     page  = load(block_table + i)                   (offs // PAGE)
//     kvloc = page * PAGE + offs % PAGE
//     ckv   = load(kv + kvloc * kv_rs + arange(DV))   (gather, no mask)
//     kpe   = load(kv + kvloc * kv_rs + DV + I)       (gather, no mask)
//     s     = dot(q_pe, kpe, dot(q_nope, trans(ckv), 0)) * scale
//     m' = max(m, rowmax(s)); alpha = exp(m - m')
//     p = exp(s - m'); acc = acc * alpha + dot(trunc(p), ckv)
//     l = l * alpha + rowsum(p); offs += BLOCK_N
//   if seqlen % BLOCK_N != 0:                         (masked tail page)
//     ... one more step with masked loads and select(mask, s, -inf) ...
//   o = trunc(acc / l)
//
// All addresses are checked as integer polynomials over top-level atoms, the
// page-table indirection and the softmax data flow structurally; a kernel
// that computes anything else is left untouched and keeps running as generic
// Triton code.
class FlashMLAMatcher {
public:
  explicit FlashMLAMatcher(triton::FuncOp f) : func(f) {}

  bool match();
  void rewrite();

private:
  bool fail(const Twine &why) {
    LLVM_DEBUG(llvm::dbgs() << "[kunpeng-llm] " << func.getName()
                            << ": no flash-mla match: " << why << "\n");
    return false;
  }

  bool matchEpilogue();
  bool matchLoop();
  bool matchTail();
  bool matchFrame();
  // One online-softmax step (loop body or tail region).  oAcc/oL/oM are the
  // yield operand positions of the step's outputs; oM may be -1 when the
  // running max is not yielded (the tail keeps it local).  Sets
  // stepCkvLoad / stepKpeLoad / stepPageLoad / btRow.
  bool matchStep(Block &block, Value inAcc, Value inL, Value inM, Value offsN,
                 bool masked, unsigned oAcc, unsigned oL, unsigned oM);
  bool checkPageLoad(triton::LoadOp pageLoad, Value offsN, bool masked);
  bool checkGatherLoads(bool masked);

  triton::FuncOp func;
  PolyEval pe;

  triton::StoreOp oStore;
  Type elemTy;
  int64_t BH = 0, DV = 0, DKPE = 0, BN = 0, D = 0;

  scf::ForOp loop;
  scf::IfOp tailIf;
  unsigned iAcc, iL, iM, iOffs;
  Value offsIter; // the loop's offs_n iter arg
  Value accOut, lOut;

  triton::LoadOp qNopeLoad, qPeLoad, seqLoad;
  Value scaleVal;

  // Per-step state
  triton::LoadOp stepCkvLoad, stepKpeLoad, stepPageLoad;
  Value btRow;
  Value offsLane; // offs_n value of the step being matched (hook target)

  // Frame results
  Value qBase, kvBase, oBase, btBase;
  Poly qOff, oOff, qRs, oRs, kvRs, btOff;
  bool kvSet = false;
};

// The only store is o = trunc(acc / l[:, None]); acc and l come from the key
// loop, optionally through the tail scf.if.
bool FlashMLAMatcher::matchEpilogue() {
  auto oTy = dyn_cast<RankedTensorType>(oStore.getValue().getType());
  if (!oTy || oTy.getRank() != 2 || !oTy.getElementType().isBF16())
    return fail("output is not a 2-D bf16 tensor");
  elemTy = oTy.getElementType();
  BH = oTy.getShape()[0];
  DV = oTy.getShape()[1];
  if (BH % 16 != 0 || BH < 16 || BH > 128 || DV % 16 != 0 || DV <= 0 ||
      DV > 1024)
    return fail("unsupported BLOCK_H / head_dim_v");

  auto trunc = oStore.getValue().getDefiningOp<arith::TruncFOp>();
  if (!trunc)
    return fail("output not truncated from fp32");
  auto div = trunc.getIn().getDefiningOp<arith::DivFOp>();
  if (!div)
    return fail("output is not acc / l");
  lOut = rowVector(div.getRhs());
  if (!lOut)
    return fail("output is not divided by a row vector");
  accOut = div.getLhs();

  tailIf = accOut.getDefiningOp<scf::IfOp>();
  if (tailIf) {
    if (!tailIf.thenBlock() || !tailIf.elseBlock() ||
        tailIf.getNumResults() != 2)
      return fail("tail is not a two-result if");
    if (lOut.getDefiningOp() != tailIf)
      return fail("acc/l do not both come from the tail if");
    if (tailIf->getBlock() != &func.getBody().front())
      return fail("tail if is not at top level");
    // The key loop feeds the tail; it is the only top-level scf.for.
    loop = nullptr;
    bool multi = false;
    for (Operation &op : func.getBody().front()) {
      if (auto f = dyn_cast<scf::ForOp>(&op)) {
        if (loop)
          multi = true;
        loop = f;
      }
    }
    if (!loop || multi)
      return fail("cannot identify the single top-level key loop");
    return true;
  }
  loop = accOut.getDefiningOp<scf::ForOp>();
  if (!loop || lOut.getDefiningOp() != loop)
    return fail("acc/l are not results of a key loop");
  return true;
}

// The key loop: for i in [0, seqlen // BLOCK_N) step 1, iterating
// (acc, l, m, offs_n) with the canonical inits.
bool FlashMLAMatcher::matchLoop() {
  if (loop->getBlock() != &func.getBody().front())
    return fail("key loop not at top level");
  auto lb = getIntSplat(loop.getLowerBound());
  auto step = getIntSplat(loop.getStep());
  if (!lb || *lb != 0 || !step || *step != 1)
    return fail("key loop bounds are not for i in [0, n) step 1");

  // Upper bound: seqlen // BLOCK_N.
  auto ubDiv = loop.getUpperBound().getDefiningOp<arith::DivSIOp>();
  if (!ubDiv)
    return fail("key loop upper bound is not seqlen // BLOCK_N");
  seqLoad = ubDiv.getLhs().getDefiningOp<triton::LoadOp>();
  if (!seqLoad || !seqLoad->getBlock() ||
      seqLoad->getBlock() != &func.getBody().front() || seqLoad.getMask())
    return fail("seqlen is not an unmasked top-level scalar load");
  if (!seqLoad.getType().isInteger())
    return fail("seqlen is not an integer");
  auto bn = getIntSplat(ubDiv.getRhs());
  if (!bn)
    return fail("key loop upper bound divisor is not constant");
  BN = *bn;

  // Iter args: acc = 0, l = 0, m = -inf, offs_n = arange(0, BLOCK_N).
  if (loop.getNumRegionIterArgs() != 4)
    return fail("key loop does not have exactly four iter args");
  bool hasAcc = false, hasL = false, hasM = false, hasOffs = false;
  for (unsigned i = 0; i < 4; ++i) {
    Value arg = loop.getRegionIterArg(i);
    Value init = loop.getInitArgs()[i];
    auto ty = dyn_cast<RankedTensorType>(arg.getType());
    if (!ty)
      return fail("key loop iter arg is not a tensor");
    if (ty.getRank() == 2 && ty.getShape()[0] == BH &&
        ty.getShape()[1] == DV && ty.getElementType().isF32() &&
        isFloatSplat(init, 0.0) && !hasAcc) {
      iAcc = i;
      hasAcc = true;
    } else if (ty.getRank() == 1 && ty.getShape()[0] == BH &&
               ty.getElementType().isF32() && isFloatSplat(init, 0.0) &&
               !hasL) {
      iL = i;
      hasL = true;
    } else if (ty.getRank() == 1 && ty.getShape()[0] == BH &&
               ty.getElementType().isF32() && isFloatSplat(init, -INFINITY) &&
               !hasM) {
      iM = i;
      hasM = true;
    } else if (ty.getRank() == 1 && ty.getShape()[0] == BN &&
               ty.getElementType().isInteger() && !hasOffs) {
      auto mr = init.getDefiningOp<triton::MakeRangeOp>();
      if (!mr || mr.getStart() != 0 || mr.getEnd() != BN)
        return fail("offs_n init is not arange(0, BLOCK_N)");
      iOffs = i;
      hasOffs = true;
    } else {
      return fail("unexpected key loop iter arg");
    }
  }
  offsIter = loop.getRegionIterArg(iOffs);

  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  auto offsAdd = yield->getOperand(iOffs).getDefiningOp<arith::AddIOp>();
  if (!offsAdd || offsAdd.getLhs() != offsIter ||
      getIntSplat(offsAdd.getRhs()) != BN)
    return fail("offs_n is not incremented by BLOCK_N");

  if (!matchStep(*loop.getBody(), loop.getRegionIterArg(iAcc),
                 loop.getRegionIterArg(iL), loop.getRegionIterArg(iM),
                 offsIter, /*masked=*/false, iAcc, iL, iM))
    return false;

  if (tailIf) {
    if (!matchTail())
      return false;
  } else {
    // Without a tail region the kernel cannot handle seqlen % BLOCK_N keys,
    // which the library would happily process: require the tail to exist.
    return fail("kernel has no masked tail chunk");
  }
  return true;
}

// One online-softmax step over one chunk of BN keys.
bool FlashMLAMatcher::matchStep(Block &block, Value inAcc, Value inL,
                                Value inM, Value offsN, bool masked,
                                unsigned oAcc, unsigned oL, unsigned oM) {
  offsLane = offsN;
  auto yield = cast<scf::YieldOp>(block.getTerminator());
  Value accN = yield->getOperand(oAcc), lN = yield->getOperand(oL);
  Value mN;
  if (oM != (unsigned)-1) {
    mN = yield->getOperand(oM);
    Value rmax = otherOperand<arith::MaxNumFOp>(mN, inM);
    if (!rmax)
      rmax = otherOperand<arith::MaximumFOp>(mN, inM);
    if (!rmax)
      return fail("m' is not max(m, rowmax(S))");
  }

  // acc' = dot(trunc(p), ckv, acc * alpha)
  auto pv = accN.getDefiningOp<triton::DotOp>();
  if (!pv)
    return fail("acc' is not a dot");
  auto pTrunc = pv.getA().getDefiningOp<arith::TruncFOp>();
  if (!pTrunc || getElementTypeOrSelf(pTrunc.getType()) != elemTy)
    return fail("P is not truncated to the input type");
  Value p = pTrunc.getIn();

  stepCkvLoad = pv.getB().getDefiningOp<triton::LoadOp>();
  if (!stepCkvLoad || stepCkvLoad->getBlock() != &block)
    return fail("V is not loaded in the step");
  auto ckvTy = dyn_cast<RankedTensorType>(stepCkvLoad.getType());
  if (!ckvTy || ckvTy.getShape()[0] != BN || ckvTy.getShape()[1] != DV ||
      ckvTy.getElementType() != elemTy)
    return fail("V tile has an unexpected shape");

  Value alpha;
  if (auto accMul = pv.getC().getDefiningOp<arith::MulFOp>()) {
    for (int i = 0; i < 2 && !alpha; ++i) {
      if (accMul->getOperand(i) == inAcc)
        alpha = rowVector(accMul->getOperand(1 - i));
    }
  }
  if (!alpha)
    return fail("acc is not rescaled by acc * row vector");
  auto aExp = alpha.getDefiningOp<math::ExpOp>();
  auto aSub =
      aExp ? aExp.getOperand().getDefiningOp<arith::SubFOp>() : nullptr;
  if (!aSub)
    return fail("alpha is not exp(m - m')");
  if (oM != (unsigned)-1) {
    if (!((aSub.getLhs() == inM && aSub.getRhs() == mN) ||
          (aSub.getRhs() == inM && aSub.getLhs() == mN)))
      return fail("alpha does not use the yielded running max");
  } else {
    mN = aSub.getLhs() == inM ? aSub.getRhs() : aSub.getLhs();
    if (aSub.getLhs() != inM && aSub.getRhs() != inM)
      return fail("alpha does not use the running max");
    if (!mN.getDefiningOp<arith::MaxNumFOp>() &&
        !mN.getDefiningOp<arith::MaximumFOp>())
      return fail("tail running max is not a maximum");
    Value rmax = otherOperand<arith::MaxNumFOp>(mN, inM);
    if (!rmax)
      rmax = otherOperand<arith::MaximumFOp>(mN, inM);
    if (!rmax)
      return fail("tail running max is not max(m, rowmax(S))");
  }

  // l' = l * alpha + rowsum(P)
  bool lok = false;
  if (auto lAdd = lN.getDefiningOp<arith::AddFOp>()) {
    for (int i = 0; i < 2 && !lok; ++i) {
      Value a = lAdd->getOperand(i), b = lAdd->getOperand(1 - i);
      auto lmul = a.getDefiningOp<arith::MulFOp>();
      if (lmul && isReduce(b, p, false) &&
          ((lmul->getOperand(0) == inL && lmul->getOperand(1) == alpha) ||
           (lmul->getOperand(0) == alpha && lmul->getOperand(1) == inL)))
        lok = true;
    }
  }
  if (!lok)
    return fail("l' is not l * alpha + rowsum(P)");

  // P = exp(S - m'); S = select(mask, qk, -inf) or qk;
  // qk = dot(q_pe, k_pe, dot(q_nope, trans(ckv), 0)) * scale
  auto pExp = p.getDefiningOp<math::ExpOp>();
  auto pSub =
      pExp ? pExp.getOperand().getDefiningOp<arith::SubFOp>() : nullptr;
  if (!pSub || rowVector(pSub.getRhs()) != mN)
    return fail("P is not exp(S - m')");
  Value sSel = pSub.getLhs();
  Value s = sSel;
  if (masked) {
    auto sel = sSel.getDefiningOp<arith::SelectOp>();
    if (!sel || !isFloatSplat(sel.getFalseValue(), -INFINITY))
      return fail("tail does not select -inf for masked keys");
    s = sel.getTrueValue();
    if (!isOffsLtSeqLen(maskCondition(sel.getCondition()), offsN, seqLoad))
      return fail("tail mask is not offs_n < seqlen");
  }
  auto qkMul = s.getDefiningOp<arith::MulFOp>();
  if (!qkMul)
    return fail("S is not scaled");
  Value dot2;
  for (int i = 0; i < 2 && !dot2; ++i) {
    if (qkMul->getOperand(i).getDefiningOp<triton::DotOp>()) {
      dot2 = qkMul->getOperand(i);
      scaleVal = scalarSource(qkMul->getOperand(1 - i));
    }
  }
  if (!dot2)
    return fail("S is not a scaled dot");
  if (!scaleVal || !scaleVal.getType().isF32() ||
      (!scaleVal.isa<BlockArgument>() &&
       scaleVal.getDefiningOp()->getBlock() != &func.getBody().front()))
    return fail("scale is not an entry-block f32 scalar");

  // m' source: rowmax over the (masked) scores.
  Value rmax;
  if (oM != (unsigned)-1) {
    rmax = otherOperand<arith::MaxNumFOp>(mN, inM);
    if (!rmax)
      rmax = otherOperand<arith::MaximumFOp>(mN, inM);
  } else {
    if (auto mx = mN.getDefiningOp<arith::MaxNumFOp>()) {
      if (mx->getOperand(0) == inM)
        rmax = mx->getOperand(1);
      else if (mx->getOperand(1) == inM)
        rmax = mx->getOperand(0);
    }
    if (!rmax)
      if (auto mx = mN.getDefiningOp<arith::MaximumFOp>()) {
        if (mx->getOperand(0) == inM)
          rmax = mx->getOperand(1);
        else if (mx->getOperand(1) == inM)
          rmax = mx->getOperand(0);
      }
  }
  if (!rmax || !isReduce(rmax, sSel, /*isMax=*/true))
    return fail("m' is not max(m, rowmax(S))");

  auto d2 = dot2.getDefiningOp<triton::DotOp>();
  stepKpeLoad = d2.getB().getDefiningOp<triton::LoadOp>();
  qPeLoad = d2.getA().getDefiningOp<triton::LoadOp>();
  if (!stepKpeLoad || stepKpeLoad->getBlock() != &block || !qPeLoad)
    return fail("K_pe is not loaded in the step / q_pe is not a load");
  auto kpeTy = dyn_cast<RankedTensorType>(stepKpeLoad.getType());
  DKPE = kpeTy ? kpeTy.getShape()[0] : 0;
  if (!kpeTy || kpeTy.getRank() != 2 || kpeTy.getShape()[1] != BN ||
      kpeTy.getElementType() != elemTy || DKPE <= 0)
    return fail("K_pe tile has an unexpected shape");
  auto d1 = d2.getC().getDefiningOp<triton::DotOp>();
  if (!d1 || !isFloatSplat(d1.getC(), 0.0))
    return fail("S is not dot(q_pe, k_pe, dot(q_nope, K^T))");
  qNopeLoad = d1.getA().getDefiningOp<triton::LoadOp>();
  auto trans = d1.getB().getDefiningOp<triton::TransOp>();
  if (!qNopeLoad || !trans ||
      trans.getSrc() != Value(stepCkvLoad.getResult()))
    return fail("S is not dot(q_nope, trans(ckv)) + dot(q_pe, k_pe)");

  // The page-table load: the only 1-D [BLOCK_N] integer load of the step.
  triton::LoadOp page = nullptr;
  block.walk([&](triton::LoadOp ld) {
    auto ty = dyn_cast<RankedTensorType>(ld.getType());
    if (ty && ty.getRank() == 1 && ty.getShape()[0] == BN &&
        ty.getElementType().isInteger())
      page = ld;
  });
  if (!page)
    return fail("no page-table load in the step");
  stepPageLoad = page;
  if (!checkPageLoad(page, offsN, masked))
    return false;
  if (!checkGatherLoads(masked))
    return false;
  return true;
}

// page = load(block_table_row + offs_n // BLOCK_N); the tail loads it with
// the offs_n < seqlen mask.
bool FlashMLAMatcher::checkPageLoad(triton::LoadOp pageLoad, Value offsN,
                                    bool masked) {
  auto add = pageLoad.getPtr().getDefiningOp<triton::AddPtrOp>();
  if (!add)
    return fail("page address is not affine");
  Value base = scalarSource(add.getPtr());
  if (!base || !base.getType().isa<triton::PointerType>())
    return fail("page address base is not a scalar pointer");
  if (btRow) {
    if (btRow != base)
      return fail("tail reads a different page table row");
  }
  btRow = base;
  auto div = add.getOffset().getDefiningOp<arith::DivSIOp>();
  if (!div || div.getLhs() != offsN ||
      getIntSplat(div.getRhs()) != BN)
    return fail("page index is not offs_n // BLOCK_N");
  if (!masked) {
    if (pageLoad.getMask() || pageLoad.getOther())
      return fail("interior page load is masked");
    return true;
  }
  if (!pageLoad.getMask())
    return fail("tail page load has no mask");
  if (!isOffsLtSeqLen(pageLoad.getMask(), offsN, seqLoad))
    return fail("tail page mask is not offs_n < seqlen");
  auto other = getIntSplat(pageLoad.getOther());
  if (!other || *other != 0)
    return fail("tail page load other is not 0");
  return true;
}

// ckv = load(kv + kvloc * kv_rs + J) and kpe = load(kv + kvloc * kv_rs +
// DV + I), with kvloc = page * BLOCK_N + offs_n % BLOCK_N; the tail loads
// zero beyond seqlen.
bool FlashMLAMatcher::checkGatherLoads(bool masked) {
  Value pageVal = stepPageLoad.getResult();
  auto va = evalAddress(pe, stepCkvLoad.getPtr(), DIM_I);
  if (!va)
    return fail("V address is not affine");
  auto vt = splitTile(va->offset, DIM_I, DIM_J);
  if (!vt)
    return fail("V is not a row-major tile");
  // With offs_n % BLOCK_N resolved to the key lane, the row stride of the
  // gather is exactly kv_rs and the tile offset remainder is
  // kv_rs * page * BLOCK_N.
  Poly pageTerm = vt->rest;
  Poly kvRsCand = vt->rowStride;
  if (pageTerm != kvRsCand * Poly::cst(BN) * Poly::atom(pageVal))
    return fail("V row stride does not match the gathered key rows");
  if (!kvSet) {
    kvSet = true;
    kvBase = va->base;
    kvRs = kvRsCand;
  } else {
    if (kvBase != va->base || kvRs != kvRsCand)
      return fail("tail gathers from a different kv cache");
  }

  auto ka = evalAddress(pe, stepKpeLoad.getPtr(), DIM_I);
  if (!ka || ka->base != kvBase)
    return fail("K_pe address is not affine over the kv cache");
  auto kr = ka->offset.split(DIM_I);
  if (!kr || kr->first != Poly::cst(1))
    return fail("K_pe row part is not DV + I");
  auto kc = kr->second.split(DIM_J);
  if (!kc || kc->first != kvRsCand ||
      kc->second != Poly::cst(DV) + pageTerm)
    return fail("K_pe is not the rope part of the same key rows");

  auto checkMask = [&](triton::LoadOp ld, int axis) {
    if (!masked) {
      if (ld.getMask() || ld.getOther())
        return fail("interior gather load is masked");
      return true;
    }
    if (!ld.getMask())
      return fail("tail gather load is unmasked");
    if (!isOffsLtSeqLen(maskCondition(ld.getMask()), loop.getResult(iOffs),
                        seqLoad))
      return fail("tail gather mask is not offs_n < seqlen");
    auto other = getFloatSplat(ld.getOther());
    if (!other || *other != 0.0)
      return fail("tail gather load other is not 0");
    return true;
  };
  if (!checkMask(stepCkvLoad, /*axis=*/1) || !checkMask(stepKpeLoad, /*axis=*/0))
    return false;
  return true;
}

// The masked tail: one more softmax step over the final partial page.
bool FlashMLAMatcher::matchTail() {
  auto cond = tailIf.getCondition().getDefiningOp<arith::CmpIOp>();
  auto rem =
      cond ? cond.getLhs().getDefiningOp<arith::RemSIOp>() : nullptr;
  if (!cond || cond.getPredicate() != arith::CmpIPredicate::ne || !rem ||
      rem.getLhs() != Value(seqLoad) || getIntSplat(rem.getRhs()) != BN)
    return fail("tail condition is not seqlen % BLOCK_N != 0");

  // Identify the acc/l result positions by type.
  unsigned oAcc = 2, oL = 2;
  for (unsigned i = 0; i < 2; ++i) {
    Type ty = tailIf.getResult(i).getType();
    if (ty == accOut.getType())
      oAcc = i;
    else if (ty == lOut.getType())
      oL = i;
  }
  if (oAcc == 2 || oL == 2 || oAcc == oL)
    return fail("tail results are not acc/l");

  triton::LoadOp qNopePrev = qNopeLoad, qPePrev = qPeLoad;
  if (!matchStep(*tailIf.thenBlock(), loop.getResult(iAcc),
                 loop.getResult(iL), loop.getResult(iM),
                 loop.getResult(iOffs), /*masked=*/true, oAcc, oL,
                 /*oM=*/-1))
    return false;
  // The tail must consume the same query tiles as the interior loop.
  if (qNopeLoad != qNopePrev || qPeLoad != qPePrev)
    return fail("tail uses different Q tiles");

  auto elseYield =
      cast<scf::YieldOp>(tailIf.elseBlock()->getTerminator());
  if (elseYield->getOperand(oAcc) != loop.getResult(iAcc) ||
      elseYield->getOperand(oL) != loop.getResult(iL))
    return fail("tail else does not yield the loop values unchanged");
  return true;
}

// Q tiles, seqlen load and the output tile: addresses as top-level
// polynomials, shapes consistent, no masks (EVEN_H).
bool FlashMLAMatcher::matchFrame() {
  // Q = [q_nope | q_pe], one [BH, D] row-major tile per program.
  auto qa = evalAddress(pe, qNopeLoad.getPtr(), DIM_I);
  if (!qa)
    return fail("Q address is not affine");
  auto qt = splitTile(qa->offset, DIM_I, DIM_J);
  if (!qt)
    return fail("Q is not a row-major tile");
  auto pa = evalAddress(pe, qPeLoad.getPtr(), DIM_I);
  if (!pa)
    return fail("Q_pe address is not affine");
  auto pt = splitTile(pa->offset, DIM_I, DIM_J);
  if (!pt)
    return fail("Q_pe is not a row-major tile");
  qBase = qa->base;
  qRs = qt->rowStride;
  qOff = qt->rest;
  if (pa->base != qBase || pt->rowStride != qRs ||
      pt->rest != qOff + Poly::cst(DV))
    return fail("q_nope / q_pe are not consecutive tiles of one Q tile");
  auto qNopeTy = dyn_cast<RankedTensorType>(qNopeLoad.getType());
  auto qPeTy = dyn_cast<RankedTensorType>(qPeLoad.getType());
  if (!qNopeTy || !qPeTy || qNopeTy.getShape()[0] != BH ||
      qNopeTy.getShape()[1] != DV || qPeTy.getShape()[0] != BH ||
      qPeTy.getShape()[1] != DKPE)
    return fail("Q tiles have unexpected shapes");
  D = DV + DKPE;
  if (qNopeLoad.getMask() || qPeLoad.getMask())
    return fail("Q loads are masked (EVEN_H required)");
  if (qNopeLoad->getBlock() != &func.getBody().front() ||
      qPeLoad->getBlock() != &func.getBody().front())
    return fail("Q is not loaded at top level");

  // Output tile.
  auto oa = evalAddress(pe, oStore.getPtr(), DIM_I);
  if (!oa)
    return fail("output address is not affine");
  auto ot = splitTile(oa->offset, DIM_I, DIM_J);
  if (!ot)
    return fail("output is not a row-major tile");
  oBase = oa->base;
  oRs = ot->rowStride;
  oOff = ot->rest;
  if (oStore.getMask())
    return fail("output store is masked (EVEN_H required)");

  // All bases must be plain function arguments and all offsets/strides
  // loop-invariant (top-level) polynomials.
  for (Value v : {qBase, kvBase, oBase, btBase}) {
    if (!v || !v.isa<BlockArgument>())
      return fail("head base is not a function argument");
  }
  for (Poly *p : {&qOff, &oOff, &qRs, &oRs, &kvRs, &btOff}) {
    for (auto *a : p->atoms()) {
      Value v = Value::getFromOpaquePointer(a);
      Block *entry = &func.getBody().front();
      bool top = isa<BlockArgument>(v)
                     ? cast<BlockArgument>(v).getOwner() == entry
                     : v.getDefiningOp()->getBlock() == entry;
      if (!top)
        return fail("head offset or stride is not loop invariant");
    }
  }

  // Library contract on the block sizes.
  if (BN % 16 != 0 || BN < 16 || BN > 512)
    return fail("unsupported page size / BLOCK_N");
  if (D % 16 != 0 || D <= DV || D > 1024)
    return fail("unsupported head_dim");
  return true;
}

bool FlashMLAMatcher::match() {
  if (func.isExternal() || !func.getBody().hasOneBlock())
    return fail("not a single-block kernel");
  pe.atomizeUnknownTensors = true;
  // offs_n % BLOCK_N is the key lane (verified structurally in matchStep);
  // resolve it to the lane index so the gather addresses evaluate cleanly.
  // Installed up front: BN and offsLane are filled in while matching.
  pe.tensorHook = [&](Operation *op, const void *lane) -> std::optional<Poly> {
    if (auto r = dyn_cast<arith::RemSIOp>(op))
      if (r.getLhs() == offsLane && getIntSplat(r.getRhs()) == BN)
        return Poly::atom(lane);
    return std::nullopt;
  };

  // The kernel's only effect must be the output store (no lse in MLA).
  SmallVector<triton::StoreOp> stores;
  bool otherEffects = false;
  func.walk([&](Operation *op) {
    if (auto st = dyn_cast<triton::StoreOp>(op)) {
      stores.push_back(st);
      return;
    }
    if (isa<triton::LoadOp>(op) || op == func.getOperation() ||
        op->hasTrait<OpTrait::IsTerminator>())
      return;
    if (isa<CallOpInterface>(op))
      otherEffects = true;
    if (auto me = dyn_cast<MemoryEffectOpInterface>(op)) {
      if (!me.hasNoEffect() && !me.onlyHasEffect<MemoryEffects::Read>())
        otherEffects = true;
    } else if (!op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
      otherEffects = true;
    }
  });
  if (otherEffects)
    return fail("side effects other than the output store");
  if (stores.size() != 1)
    return fail("expected exactly one store");
  oStore = stores[0];

  if (!matchEpilogue())
    return false;
  if (!matchLoop())
    return false;

  // btRow: the batch's page-table row; walk the addptr chain for the offset.
  Value b = btRow;
  btOff = Poly::cst(0);
  while (auto ap = b.getDefiningOp<triton::AddPtrOp>()) {
    auto o = pe.scalar(ap.getOffset());
    if (!o)
      return fail("page table row offset is not affine");
    btOff = *o + btOff;
    b = ap.getPtr();
  }
  btBase = b;
  if (!matchFrame())
    return false;
  LLVM_DEBUG(llvm::dbgs() << "[kunpeng-llm] " << func.getName()
                          << ": matched flash-mla (BLOCK_H=" << BH
                          << ", BLOCK_N=" << BN << ", d=" << D
                          << ", dv=" << DV << ")\n");
  return true;
}

void FlashMLAMatcher::rewrite() {
  auto ret = cast<triton::ReturnOp>(func.getBody().front().getTerminator());
  OpBuilder b(ret);
  Location loc = oStore.getLoc();
  auto i64 = b.getI64Type();
  DenseMap<const void *, Value> widened;

  // Same widening/materialization as the flash-attention rewrite.
  auto atom64 = [&](const void *a) -> Value {
    Value &v64 = widened[a];
    if (!v64) {
      Value v = Value::getFromOpaquePointer(a);
      if (v.getType().isIndex())
        v64 = b.create<arith::IndexCastOp>(loc, i64, v);
      else if (v.getType().getIntOrFloatBitWidth() < 64)
        v64 = b.create<arith::ExtSIOp>(loc, i64, v);
      else
        v64 = v;
    }
    return v64;
  };
  auto emit = [&](const Poly &p) -> Value {
    Value sum;
    int64_t constant = 0;
    for (auto &[mono, c] : p.terms) {
      if (mono.empty()) {
        constant = c;
        continue;
      }
      Value t;
      for (auto *a : mono)
        t = t ? b.create<arith::MulIOp>(loc, t, atom64(a)) : atom64(a);
      if (c == -1 && sum) {
        sum = b.create<arith::SubIOp>(loc, sum, t);
        continue;
      }
      if (c != 1)
        t = b.create<arith::MulIOp>(
            loc, t, b.create<arith::ConstantIntOp>(loc, c, 64));
      sum = sum ? b.create<arith::AddIOp>(loc, sum, t) : t;
    }
    Value cst = b.create<arith::ConstantIntOp>(loc, constant, 64);
    if (!sum)
      return cst;
    return constant ? b.create<arith::AddIOp>(loc, sum, cst) : sum;
  };
  auto addr = [&](Value base, const Poly &off) -> Value {
    Value p = base;
    if (!off.terms.empty())
      p = b.create<triton::AddPtrOp>(loc, p.getType(), p, emit(off));
    return b.create<triton::PtrToIntOp>(loc, i64, p);
  };
  auto c64 = [&](int64_t c) {
    return b.create<arith::ConstantIntOp>(loc, c, 64);
  };

  SmallVector<Value> args = {addr(qBase, qOff), emit(qRs),
                             addr(kvBase, Poly::cst(0)), emit(kvRs),
                             addr(btBase, btOff), emit(Poly::atom(Value(seqLoad))),
                             addr(oBase, oOff),   emit(oRs),
                             c64(BH),             c64(BN),
                             c64(DV),             c64(D),
                             scaleVal};

  std::string name = "kunpeng_flash_mla_bf16";
  auto module = func->getParentOfType<ModuleOp>();
  auto fnTy = b.getFunctionType(ValueRange(args).getTypes(), {});
  auto callee = module.lookupSymbol<func::FuncOp>(name);
  if (!callee) {
    OpBuilder mb(module.getBodyRegion());
    mb.setInsertionPointToStart(module.getBody());
    callee = mb.create<func::FuncOp>(module.getLoc(), name, fnTy);
    callee.setPrivate();
  }
  b.create<func::CallOp>(loc, callee, args);

  LLVM_DEBUG(llvm::dbgs() << "[kunpeng-llm] " << func.getName()
                          << ": replaced by " << name << " (BLOCK_H=" << BH
                          << ", BLOCK_N=" << BN << ", d=" << D
                          << ", dv=" << DV << ")\n");

  // Drop the Triton implementation: the store, then everything dead.
  oStore.erase();
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> dead;
    func.getBody().front().walk<WalkOrder::PostOrder>([&](Operation *op) {
      if (op->getParentOp() == func.getOperation() && isOpTriviallyDead(op))
        dead.push_back(op);
    });
    for (Operation *op : llvm::reverse(dead)) {
      op->erase();
      changed = true;
    }
  }
}

// Recognize the shared-KV BF16 forward graph before structured lowering.
// Validate its addresses, masking, online-softmax recurrence and epilogue;
// preserve unsupported programs and outline a fallback for runtime rejection.
static bool lowerSparseMLA(triton::FuncOp f) {
  auto a = f.getArguments();
  if (a.size() != 19 || !llvm::hasSingleElement(f.getBody()))
    return false;
  Builder types(f.getContext());
  for (unsigned i = 0; i < a.size(); ++i) {
    if (i < 5) {
      auto p = dyn_cast<triton::PointerType>(a[i].getType());
      if (!p || p.getPointeeType() != (i == 2 ? Type(types.getI32Type())
                                              : Type(types.getBF16Type())))
        return false;
    } else if (!a[i].getType().isInteger(32))
      return false;
  }
  SmallVector<triton::DotOp> dot;
  SmallVector<triton::LoadOp> load;
  SmallVector<triton::StoreOp> store;
  SmallVector<scf::ForOp> loops;
  SmallVector<scf::IfOp> guards;
  SmallVector<math::Exp2Op> exp;
  SmallVector<math::Log2Op> log;
  Value pid[3];
  bool effects = false;
  f.walk([&](Operation *op) {
    if (auto x = dyn_cast<triton::DotOp>(op))
      dot.push_back(x);
    if (auto x = dyn_cast<triton::LoadOp>(op))
      load.push_back(x);
    if (auto x = dyn_cast<triton::StoreOp>(op))
      store.push_back(x);
    if (auto x = dyn_cast<scf::ForOp>(op))
      loops.push_back(x);
    if (auto x = dyn_cast<scf::IfOp>(op))
      guards.push_back(x);
    if (auto x = dyn_cast<math::Exp2Op>(op))
      exp.push_back(x);
    if (auto x = dyn_cast<math::Log2Op>(op))
      log.push_back(x);
    if (auto x = dyn_cast<triton::GetProgramIdOp>(op))
      pid[unsigned(x.getAxis())] = x;
    if (isa<CallOpInterface>(op))
      effects = true;
    if (op != f.getOperation() && !isa<triton::LoadOp, triton::StoreOp>(op) &&
        !op->hasTrait<OpTrait::IsTerminator>() &&
        !op->hasTrait<OpTrait::HasRecursiveMemoryEffects>() &&
        !isMemoryEffectFree(op))
      effects = true;
  });
  if (effects || dot.size() != 3 || load.size() != 5 || store.size() != 2 ||
      loops.size() != 1 || guards.size() != 1 || exp.size() != 2 ||
      log.size() != 1 || !pid[0] || !pid[1] || !pid[2])
    return false;
  auto shape = [](Value v, ArrayRef<int64_t> s) {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    return t && t.getShape() == s;
  };
  if (!shape(dot[0].getA(), {64, 512}) || !shape(dot[0].getB(), {512, 32}) ||
      !shape(dot[1].getA(), {64, 64}) || !shape(dot[1].getB(), {64, 32}) ||
      dot[1].getC() != dot[0] || dot[2].getA() != exp[0])
    return false;
  auto trans = dot[2].getB().getDefiningOp<triton::TransOp>();
  if (!trans || trans.getSrc() != dot[0].getB())
    return false;
  auto unwrap = [](Value v) -> triton::LoadOp {
    auto e = v.getDefiningOp<arith::ExtFOp>();
    return e ? e.getIn().getDefiningOp<triton::LoadOp>() : nullptr;
  };
  auto q = unwrap(dot[0].getA()), k = unwrap(dot[0].getB()),
       tq = unwrap(dot[1].getA()), tk = unwrap(dot[1].getB());
  if (!q || !k || !tq || !tk)
    return false;
  for (auto input : {q, k, tq, tk})
    if (!input.getOther() || !isFloatSplat(input.getOther(), 0))
      return false;
  triton::LoadOp ids;
  for (auto x : load)
    if (x != q && x != k && x != tq && x != tk)
      ids = x;
  if (!ids)
    return false;
  auto loop = loops[0];
  auto guard = guards[0];
  auto integer = [](Value v) -> std::optional<int64_t> {
    Attribute at;
    if (!matchPattern(v, m_Constant(&at)))
      return {};
    if (auto x = dyn_cast<IntegerAttr>(at))
      return x.getInt();
    if (auto x = dyn_cast<DenseIntElementsAttr>(at); x && x.isSplat())
      return x.getSplatValue<APInt>().getSExtValue();
    return {};
  };
  auto mask = ids.getMask().getDefiningOp<arith::CmpIOp>();
  if (!mask || mask.getPredicate() != arith::CmpIPredicate::slt)
    return false;
  int64_t topk = integer(mask.getRhs()).value_or(0);
  if (topk <= 0 || topk % 32 || topk > 4096 ||
      integer(loop.getLowerBound()) != 0 || integer(loop.getStep()) != 1 ||
      integer(loop.getUpperBound()) != topk / 32 ||
      !isFloatSplat(loop.getInitArgs()[0], -INFINITY) ||
      !isFloatSplat(loop.getInitArgs()[1], 1) ||
      !isFloatSplat(loop.getInitArgs()[2], 0))
    return false;
  class GatherEval : public PolyEval {
    Value ids;

  public:
    GatherEval(Value v) : ids(v) {}
    std::optional<Poly> tensor(Value v, const void *lane) {
      if (v == ids)
        return Poly::atom(v);
      if (!isa<RankedTensorType>(v.getType()))
        return scalar(v);
      if (auto c = getIntSplat(v))
        return Poly::cst(*c);
      auto *op = v.getDefiningOp();
      if (!op)
        return {};
      if (auto x = dyn_cast<triton::MakeRangeOp>(op))
        return Poly::cst(x.getStart()) + Poly::atom(lane);
      if (auto x = dyn_cast<triton::SplatOp>(op))
        return scalar(x.getSrc());
      if (auto x = dyn_cast<triton::ExpandDimsOp>(op))
        return tensor(x.getSrc(), x.getAxis() == 1 ? DIM_I : DIM_J);
      if (auto x = dyn_cast<triton::BroadcastOp>(op))
        return tensor(x.getSrc(), lane);
      if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op)) {
        auto l = tensor(op->getOperand(0), lane),
             r = tensor(op->getOperand(1), lane);
        if (!l || !r)
          return {};
        if (isa<arith::AddIOp>(op))
          return *l + *r;
        if (isa<arith::SubIOp>(op))
          return *l - *r;
        return *l * *r;
      }
      return {};
    }
  } pe(ids);
  auto atom = [](Value v) { return Poly::atom(v); };
  Poly batch = atom(pid[0]), row = atom(pid[1]), head = atom(pid[2]),
       i = Poly::atom(DIM_I), j = Poly::atom(DIM_J);
  Poly group;
  bool found = false;
  f.walk([&](arith::DivSIOp x) {
    if (x.getLhs() == pid[2] && integer(x.getRhs()) == 128) {
      group = atom(x);
      found = true;
    }
  });
  if (!found)
    return false;
  auto ptr = [&](Value v, unsigned base, Poly expected) {
    auto addr = evalAddress(pe, v, DIM_I);
    if (!addr)
      return false;
    while (auto add = addr->base.getDefiningOp<triton::AddPtrOp>()) {
      auto n = pe.scalar(add.getOffset());
      if (!n)
        return false;
      addr->offset = addr->offset + *n;
      addr->base = add.getPtr();
    }
    return addr->base == a[base] && addr->offset == expected;
  };
  Poly qb =
      batch * atom(a[5]) + row * atom(a[7]) + head * Poly::cst(64) * atom(a[6]);
  Poly kb = batch * atom(a[8]) + group * atom(a[9]);
  if (!ptr(q.getPtr(), 0, qb + i * atom(a[6]) + j) ||
      !ptr(tq.getPtr(), 0, qb + i * atom(a[6]) + j + Poly::cst(512)) ||
      !ptr(k.getPtr(), 1, kb + i + atom(ids) * atom(a[10])) ||
      !ptr(tk.getPtr(), 1, kb + i + Poly::cst(512) + atom(ids) * atom(a[10])) ||
      !ptr(ids.getPtr(), 2,
           batch * atom(a[11]) + row * atom(a[13]) + group * atom(a[12]) +
               atom(loop.getInductionVar()) * Poly::cst(32) + i) ||
      !ptr(store[0].getPtr(), 3,
           batch * atom(a[14]) + row * atom(a[16]) +
               (head * Poly::cst(64) + i) * atom(a[15]) + j) ||
      !ptr(store[1].getPtr(), 4,
           batch * atom(a[17]) + row * atom(a[18]) + head * Poly::cst(64) + i))
    return false;
  auto mul = dot[1]->hasOneUse()
                 ? dyn_cast<arith::MulFOp>(*dot[1]->user_begin())
                 : nullptr;
  auto select = mul && mul->hasOneUse()
                    ? dyn_cast<arith::SelectOp>(*mul->user_begin())
                    : nullptr;
  auto delta = exp[0].getOperand().getDefiningOp<arith::SubFOp>();
  auto alpha = exp[1].getOperand().getDefiningOp<arith::SubFOp>();
  if (!mul || !select || select.getTrueValue() != mul ||
      !isFloatSplat(select.getFalseValue(), -INFINITY) || !delta ||
      delta.getLhs() != select || !alpha ||
      alpha.getLhs() != loop.getRegionIterArgs()[0] ||
      rowVector(delta.getRhs()) != alpha.getRhs())
    return false;
  auto scaleValue = otherOperand<arith::MulFOp>(mul, dot[1]);
  auto scale = getFloatSplat(scaleValue);
  if (!scale)
    return false;
  auto gc = guard.getCondition().getDefiningOp<arith::CmpIOp>();
  auto guardStart = gc ? pe.scalar(gc.getLhs()) : std::nullopt;
  auto indexStart = pe.tensor(mask.getLhs(), DIM_I);
  if (!gc || gc.getPredicate() != arith::CmpIPredicate::sle ||
      gc.getRhs() != pid[1] || !guardStart ||
      *guardStart != atom(loop.getInductionVar()) * Poly::cst(32) ||
      !indexStart ||
      *indexStart != atom(loop.getInductionVar()) * Poly::cst(32) + i ||
      ids->getParentOp() != loop || guard->getParentOp() != loop ||
      dot[0]->getParentOp() != guard || dot[1]->getParentOp() != guard ||
      dot[2]->getParentOp() != guard || !guard.elseBlock())
    return false;
  auto iter = loop.getRegionIterArgs();
  auto yes = cast<scf::YieldOp>(guard.thenBlock()->getTerminator());
  auto no = cast<scf::YieldOp>(guard.elseBlock()->getTerminator());
  auto end = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  Value maximum = otherOperand<arith::MaximumFOp>(alpha.getRhs(), iter[0]);
  if (!maximum)
    maximum = otherOperand<arith::MaxNumFOp>(alpha.getRhs(), iter[0]);
  auto den = yes.getOperand(1).getDefiningOp<arith::AddFOp>();
  Value scaledDen;
  if (den)
    for (Value v : den->getOperands())
      if (otherOperand<arith::MulFOp>(v, iter[1]) == exp[1])
        scaledDen = v;
  Value accScale = otherOperand<arith::MulFOp>(dot[2].getC(), iter[2]);
  if (!isFloatSplat(dot[0].getC(), 0) || !maximum ||
      !isReduce(maximum, select, true) || !scaledDen ||
      !isReduce(otherOperand<arith::AddFOp>(yes.getOperand(1), scaledDen),
                exp[0], false) ||
      !accScale || rowVector(accScale) != exp[1] ||
      yes.getOperand(0) != alpha.getRhs() || yes.getOperand(2) != dot[2] ||
      !llvm::equal(no.getOperands(), iter) ||
      !llvm::equal(end.getOperands(), guard.getResults()) ||
      log[0].getOperand() != loop.getResult(1) || integer(ids.getOther()) != -1)
    return false;
  Value remainder;
  f.walk([&](arith::RemSIOp x) {
    if (x.getLhs() == pid[2] && integer(x.getRhs()) == 128)
      remainder = x;
  });
  if (!remainder)
    return false;
  struct Bound {
    arith::CmpIPredicate pred;
    Poly lhs, rhs;
  };
  auto checkMask = [&](Value v, std::initializer_list<Bound> bounds) {
    SmallVector<Cmp> cs;
    if (!v || !collectCmps<arith::AndIOp>(v, DIM_I, cs) ||
        cs.size() != bounds.size())
      return false;
    SmallVector<bool> used(bounds.size(), false);
    for (auto &cmp : cs) {
      auto l = pe.tensor(cmp.lhs, cmp.lane), r = pe.tensor(cmp.rhs, cmp.lane);
      bool found = false;
      unsigned n = 0;
      for (auto &bound : bounds) {
        if (!used[n] && l && r && cmp.pred == bound.pred && *l == bound.lhs &&
            *r == bound.rhs) {
          used[n] = true;
          found = true;
          break;
        }
        ++n;
      }
      if (!found)
        return false;
    }
    return true;
  };
  auto slt = arith::CmpIPredicate::slt;
  Bound headBound{slt, atom(remainder) * Poly::cst(64) + i, Poly::cst(128)};
  Bound visible{arith::CmpIPredicate::sle, atom(ids), row};
  Bound positive{arith::CmpIPredicate::sge, atom(ids), Poly::cst(0)};
  if (!checkMask(q.getMask(), {headBound, {slt, j, Poly::cst(512)}}) ||
      !checkMask(tq.getMask(), {headBound, {slt, j, Poly::cst(64)}}) ||
      !checkMask(k.getMask(), {{slt, i, Poly::cst(512)}, visible, positive}) ||
      !checkMask(tk.getMask(), {{slt, i, Poly::cst(64)}, visible, positive}) ||
      !checkMask(select.getCondition(), {visible, positive}) ||
      !checkMask(store[0].getMask(), {headBound, {slt, j, Poly::cst(512)}}) ||
      !checkMask(store[1].getMask(), {headBound}))
    return false;
  auto output = store[0].getValue().getDefiningOp<arith::TruncFOp>();
  auto division =
      output ? output.getIn().getDefiningOp<arith::DivFOp>() : nullptr;
  auto lse = store[1].getValue().getDefiningOp<arith::TruncFOp>();
  if (!division || division.getLhs() != loop.getResult(2) ||
      rowVector(division.getRhs()) != loop.getResult(1) || !lse ||
      otherOperand<arith::AddFOp>(lse.getIn(), loop.getResult(0)) != log[0])
    return false;

  auto module = f->getParentOfType<ModuleOp>();
  auto loc = f.getLoc();
  OpBuilder b(f);
  SmallVector<Operation *> original;
  for (auto &op : f.getBody().front().without_terminator())
    original.push_back(&op);
  b.setInsertionPointToStart(&f.getBody().front());
  auto c = [&](int64_t v) -> Value {
    return b.create<arith::ConstantIntOp>(loc, v, 32);
  };
  Value id[3];
  for (int d = 0; d < 3; ++d)
    id[d] = b.create<triton::GetProgramIdOp>(
        loc, static_cast<triton::ProgramIDDim>(d));
  Value ok =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, id[2], c(2));
  for (auto [slot, value] : {std::pair<int, int>{6, 576},
                             {9, 576},
                             {10, 576},
                             {12, int(topk)},
                             {13, int(topk)},
                             {15, 512},
                             {16, 128 * 512},
                             {18, 128},
                             {7, 128 * 576}}) {
    Value eq = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, a[slot],
                                       c(value));
    ok = b.create<arith::AndIOp>(loc, ok, eq);
  }
  Value minKV = b.create<arith::DivSIOp>(loc, a[5], c(128));
  Value enoughKV =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, a[8], minKV);
  ok = b.create<arith::AndIOp>(loc, ok, enoughKV);
  auto attempt = b.create<scf::IfOp>(loc, TypeRange{b.getI32Type()}, ok, true);
  b.setInsertionPointToStart(&attempt.getThenRegion().front());
  auto address = [&](int base, int bs, int rs, int hs) -> Value {
    Value off = b.create<arith::MulIOp>(loc, id[0], a[bs]);
    if (rs >= 0) {
      Value r = b.create<arith::MulIOp>(loc, id[1], a[rs]);
      off = b.create<arith::AddIOp>(loc, off, r);
    }
    if (hs) {
      Value h = b.create<arith::MulIOp>(loc, id[2], c(hs));
      off = b.create<arith::AddIOp>(loc, off, h);
    }
    Value p = b.create<triton::AddPtrOp>(loc, a[base].getType(), a[base], off);
    return b.create<triton::PtrToIntOp>(loc, b.getI64Type(), p);
  };
  SmallVector<Value> args{
      address(0, 5, 7, 64 * 576),
      address(1, 8, -1, 0),
      address(2, 11, 13, 0),
      address(3, 14, 16, 64 * 512),
      address(4, 17, 18, 64),
      id[1],
      c(topk),
      b.create<arith::ConstantFloatOp>(
          loc, APFloat(float(*scale / 1.4426950408889634)), b.getF32Type()),
      b.create<arith::DivSIOp>(loc, a[5], c(128 * 576))};
  auto name = "kunpeng_sparse_mla_forward_bf16";
  auto callee = module.lookupSymbol<func::FuncOp>(name);
  if (!callee) {
    OpBuilder mb(module.getBody(), module.getBody()->begin());
    callee = mb.create<func::FuncOp>(
        loc, name,
        b.getFunctionType(ValueRange(args).getTypes(), {b.getI32Type()}));
    callee.setPrivate();
  }
  auto call = b.create<func::CallOp>(loc, callee, args);
  b.create<scf::YieldOp>(loc, call.getResults());
  b.setInsertionPointToStart(&attempt.getElseRegion().front());
  b.create<scf::YieldOp>(loc, c(1));
  b.setInsertionPointAfter(attempt);
  Value failed = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                         attempt.getResult(0), c(0));
  auto fallback = b.create<scf::IfOp>(loc, failed, false);
  SmallVector<Type> ts;
  for (Value v : a)
    ts.push_back(v.getType());
  ts.append(3, b.getI32Type());
  OpBuilder mb(f);
  mb.setInsertionPointAfter(f);
  auto helper = mb.create<func::FuncOp>(loc, f.getName().str() + "_fallback",
                                        b.getFunctionType(ts, {}));
  helper.setPrivate();
  helper->setAttr("llvm.linkage", LLVM::LinkageAttr::get(
                                      f.getContext(), LLVM::Linkage::Internal));
  helper->setAttr("passthrough", b.getStrArrayAttr({"noinline"}));
  auto entry = helper.addEntryBlock();
  OpBuilder hb(entry, entry->begin());
  IRMapping mapping;
  for (unsigned n = 0; n < a.size(); ++n)
    mapping.map(a[n], entry->getArgument(n));
  for (unsigned n = 0; n < 3; ++n)
    mapping.map(pid[n], entry->getArgument(a.size() + n));
  for (auto *op : original)
    if (!isa<triton::GetProgramIdOp>(op))
      hb.clone(*op, mapping);
  hb.create<func::ReturnOp>(loc);
  b.setInsertionPoint(fallback.thenBlock()->getTerminator());
  SmallVector<Value> inputs(a.begin(), a.end());
  inputs.append(id, id + 3);
  b.create<func::CallOp>(loc, helper, inputs);
  for (auto *op : llvm::reverse(original))
    op->erase();
  return true;
}

struct TritonFlashAttentionToKunpengLLMPass
    : public impl::TritonFlashAttentionToKunpengLLMBase<
          TritonFlashAttentionToKunpengLLMPass> {
  void runOnOperation() override {
    SmallVector<triton::FuncOp> functions;
    getOperation().walk([&](triton::FuncOp f) { functions.push_back(f); });
    for (auto f : functions) {
      if (lowerSparseMLA(f))
        continue;
      FlashAttentionMatcher m(f);
      if (m.match())
        m.rewrite();
    }
  }
};

struct TritonFlashMLAToKunpengLLMPass
    : public impl::TritonFlashMLAToKunpengLLMBase<
          TritonFlashMLAToKunpengLLMPass> {
  void runOnOperation() override {
    getOperation().walk([](triton::FuncOp f) {
      FlashMLAMatcher m(f);
      if (m.match())
        m.rewrite();
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createTritonFlashAttentionToKunpengLLMPass() {
  return std::make_unique<TritonFlashAttentionToKunpengLLMPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createTritonFlashMLAToKunpengLLMPass() {
  return std::make_unique<TritonFlashMLAToKunpengLLMPass>();
}

} // namespace triton
} // namespace mlir
