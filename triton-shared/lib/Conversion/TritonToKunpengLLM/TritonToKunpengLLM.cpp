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
    return std::nullopt;
  }
};

// Address of a load/store: `base` (a scalar pointer) + offset polynomial,
// where the offset is what is added at tensor level (after the splat).
struct Address {
  Value base;
  Poly offset;
};

static std::optional<Address> evalAddress(PolyEval &pe, Value ptr,
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

struct TritonFlashAttentionToKunpengLLMPass
    : public impl::TritonFlashAttentionToKunpengLLMBase<
          TritonFlashAttentionToKunpengLLMPass> {
  void runOnOperation() override {
    getOperation().walk([](triton::FuncOp f) {
      FlashAttentionMatcher m(f);
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

} // namespace triton
} // namespace mlir
