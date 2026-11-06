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
#include "triton-shared/Conversion/TritonToKunpengLLM/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Integer polynomials over atoms
//===----------------------------------------------------------------------===//

// Tile index variables: element (i, j) of a 2-D tensor, element i of a 1-D
// tensor (whose lane maps to I or J depending on how it is expanded).
const char kDimI = 0, kDimJ = 0;
const void *const DIM_I = &kDimI;
const void *const DIM_J = &kDimJ;

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
  std::optional<Poly> scalar(Value v) {
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
// bound polynomial.
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
  bool matchEpilogue();
  bool matchLoopChain();
  bool matchLoopBody(KeyLoop &kl);
  bool matchMask(KeyLoop &kl, const Poly &colStart, Window &w);
  bool matchQ();
  bool matchKV(KeyLoop &kl, const Poly &colStart);
  bool matchFrame();
  Value materialize(OpBuilder &b, Location loc, const Poly &p);

  mutable triton::FuncOp func;
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
};

bool FlashAttentionMatcher::isTopLevel(const void *atom) const {
  if (atom == DIM_I || atom == DIM_J)
    return false;
  Value v = Value::getFromOpaquePointer(atom);
  Block *entry = &func.getBody().front();
  if (auto ba = dyn_cast<BlockArgument>(v))
    return ba.getOwner() == entry;
  return v.getDefiningOp()->getBlock() == entry;
}

bool FlashAttentionMatcher::topLevelPoly(const Poly &p) const {
  return llvm::all_of(p.atoms(), [&](const void *a) { return isTopLevel(a); });
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
  if (func.isExternal() || !func.getBody().hasOneBlock())
    return fail("not a single-block kernel");

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
    if (!ty || st.getOperation()->getBlock() != &func.getBody().front())
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

  // acc, m and l are the results of the last key loop (in any order).
  auto loop = accFinal.getDefiningOp<scf::ForOp>();
  if (!loop || loop.getNumResults() != 3 ||
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
  // Walk back through loops whose inits are the previous loop's results.
  while (true) {
    scf::ForOp first = loops.front().loop;
    auto inits = first.getInitArgs();
    auto prev = inits[0].getDefiningOp<scf::ForOp>();
    if (prev && prev.getNumResults() == 3 && inits[0] == prev.getResult(0) &&
        inits[1] == prev.getResult(1) && inits[2] == prev.getResult(2)) {
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
    if (kl.loop->getBlock() != &func.getBody().front())
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
  if (!ld || ld->getBlock() != &func.getBody().front())
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
    else if (idx == rowStart + Poly::atom(DIM_I) && (!seqQ || bound == *seqQ))
      seqQ = bound;
    else
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

  auto ka = evalAddress(pe, kl.kLoad.getPtr(), DIM_I);
  auto va = evalAddress(pe, kl.vLoad.getPtr(), DIM_I);
  if (!ka || !va)
    return fail("K/V address is not affine");
  auto kt = splitTile(ka->offset, DIM_J, DIM_I);
  auto vt = splitTile(va->offset, DIM_I, DIM_J);
  if (!kt || !vt)
    return fail("K/V are not row-major tiles");
  Address kh{ka->base, kt->rest - colStart * kt->rowStride};
  Address vh{va->base, vt->rest - colStart * vt->rowStride};
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
      if (idx == rowStart + Poly::atom(DIM_I)) {
        if (seqQ && *seqQ != bound)
          return fail("inconsistent row bounds");
        seqQ = bound;
      } else if (st == oStore && idx == Poly::atom(DIM_J) && bound.getConst()) {
        D = *bound.getConst();
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

  // Head bases and strides must be computable at the end of the kernel.
  for (Address *a : {&q, &k, &v, &o, &lse})
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
  auto ret = cast<triton::ReturnOp>(func.getBody().front().getTerminator());
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
  SmallVector<Value> args = {
      addr(q),   num(qRs), addr(k),   num(kRs),      addr(v),    num(vRs),
      addr(o),   num(oRs), addr(lse), num(rowStart), c64(BM),    num(sq),
      num(*seqK), c64(flags),
      win.left ? num(*win.left) : c64(0), win.right ? num(*win.right) : c64(0),
      c64(D)};
  args.push_back(b.create<arith::ConstantOp>(
      loc, b.getF32FloatAttr(static_cast<float>(scaleLog2))));

  std::string name = elemTy.isBF16() ? "kunpeng_flash_attention_forward_bf16"
                                     : "kunpeng_flash_attention_forward_f16";
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

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createTritonFlashAttentionToKunpengLLMPass() {
  return std::make_unique<TritonFlashAttentionToKunpengLLMPass>();
}

} // namespace triton
} // namespace mlir
