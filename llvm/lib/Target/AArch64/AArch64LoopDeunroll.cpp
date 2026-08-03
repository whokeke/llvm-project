//===-- AArch64LoopDeunroll.cpp --- Reroll manual N-x unrolled loops -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Reroll manual N-x unrolled loops back to single-iteration form so the
/// loop vectorizer can pick the arch-native VF.
///
/// Motivation
/// ==========
/// SEAL NTT butterfly (dwthandler.h) is hand-unrolled 4x to match NEON
/// ld4/st4 (4-way interleave, 128-bit). On 256-bit SVE this pins the
/// vectorized loop to VF=2 (only the low 128 bits of z registers are used;
/// `umulh z.d` wastes half the SVE bandwidth). De-unrolling to a single
/// body lets LoopVectorize pick VF=4 SVE with `ld1d`/`st1d`, halving the
/// `umulh z` count and scaling to wider SVE.
///
/// Two access patterns are recognised
/// =================================
///  Pattern A (index-based, e.g. `x[j+k]`):
///    The GEP index is `(or disjoint | add) IV, k` for k=1..N-1.
///    The IV is a `phi i64` with `add IV, N` step.
///
///  Pattern B (pointer-based, e.g. `*x++` unrolled, as in SEAL dwthandler):
///    The body uses `phi ptr` (pointer phi) as the base, and the 4 sub-blocks
///    use `gep i8, ptr %ptr_phi, i64 k*elem_size` for k=1..N-1.
///    The pointer phi back-edge is `gep ..., i64 N*elem_size`.
///    A separate `phi i64` IV with `add IV, N` step drives the guard.
///
/// Algorithm
/// =========
///   1. Find header phis: i64 IVs (add step) and ptr phis (GEP back-edge).
///   2. Determine N (must agree across IV step and ptr-phi back-edge).
///   3. Collect "deletable" GEPs:
///      - Pattern A: index is offset root (or disjoint|add IV, k), k=1..N-1.
///      - Pattern B: base is a ptr phi, constant index = k*elem_size, k=1..N-1.
///   4. Forward-walk from deletable GEPs to collect all dependent instructions.
///   5. Update guard if it uses `(or disjoint|add) step, (N-1)`.
///   6. Verify no deletable inst has an external use (no cross-body dep).
///   7. Update: i64 IV step N->1, ptr phi back-edge offset N*es->1*es.
///   8. Erase deletable instructions (reverse order for def-use safety).
///
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64LoopDeunroll.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "aarch64-loop-deunroll"

static cl::opt<bool> EnableLoopDeunroll(
    "aarch64-loop-deunroll",
    cl::init(false), cl::Hidden,
    cl::desc("Reroll manual N-x unrolled loops to single-iteration form "
             "so the loop vectorizer can pick arch-native VF"));

namespace {

class AArch64LoopDeunrollImpl {
public:
  PreservedAnalyses run(Function &F, LoopInfo &LI);

private:
  bool tryRerollLoopAndSubloops(Loop *L);
  bool tryRerollLoop(Loop *L);
};

struct PtrPhiInfo {
  PHINode *Phi;
  GetElementPtrInst *BackEdgeGEP;
  uint64_t OffsetBytes;
};

} // end anonymous namespace

// Match `or disjoint IV, k` or `add IV, k` and return k, or None.
static std::optional<unsigned> matchOffset(Value *V, Value *IV) {
  auto *BO = dyn_cast<BinaryOperator>(V);
  if (!BO)
    return std::nullopt;
  if (BO->getOpcode() != Instruction::Or && BO->getOpcode() != Instruction::Add)
    return std::nullopt;
  if (BO->getOpcode() == Instruction::Or) {
    auto *PDI = dyn_cast<PossiblyDisjointInst>(BO);
    if (!PDI || !PDI->isDisjoint())
      return std::nullopt;
  }
  Value *Other = nullptr;
  if (BO->getOperand(0) == IV) Other = BO->getOperand(1);
  else if (BO->getOperand(1) == IV) Other = BO->getOperand(0);
  else return std::nullopt;
  auto *C = dyn_cast<ConstantInt>(Other);
  if (!C) return std::nullopt;
  return C->getZExtValue();
}

// Compute the byte offset of a single-index GEP `gep T, ptr base, i64 K`.
// Returns 0 if V is not a GEP or has a non-constant single index.
static std::optional<uint64_t> getGEPConstByteOffset(Value *V) {
  auto *GEP = dyn_cast<GetElementPtrInst>(V);
  if (!GEP || GEP->getNumOperands() != 2)  // base + 1 index only
    return std::nullopt;
  auto *C = dyn_cast<ConstantInt>(GEP->getOperand(1));
  if (!C)
    return std::nullopt;
  Type *ElemTy = GEP->getSourceElementType();
  uint64_t ElemBytes = ElemTy->getScalarSizeInBits() / 8;
  if (ElemBytes == 0)
    return std::nullopt;
  return C->getZExtValue() * ElemBytes;
}

PreservedAnalyses
AArch64LoopDeunrollImpl::run(Function &F, LoopInfo &LI) {
  if (!EnableLoopDeunroll)
    return PreservedAnalyses::all();

  // Gate on SVE2/umulh availability (same as MulI128Lowering).
  if (F.hasFnAttribute("target-features")) {
    StringRef TF = F.getFnAttribute("target-features").getValueAsString();
    if (!TF.contains("+sve2") && !TF.contains("+v8.6a") &&
        !TF.contains("armv8.6-a") && !TF.contains("+sve2-aes"))
      return PreservedAnalyses::all();
  }

  bool Debug = getenv("LOOP_DEUNROLL_DEBUG") != nullptr;
  if (Debug)
    errs() << "=== AArch64LoopDeunroll on " << F.getName() << " ===\n";

  bool Changed = false;
  for (Loop *L : LI.getTopLevelLoops())
    Changed |= tryRerollLoopAndSubloops(L);

  if (Debug && Changed)
    errs() << "  -> rerolled at least one loop\n";

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

bool AArch64LoopDeunrollImpl::tryRerollLoopAndSubloops(Loop *L) {
  bool Changed = false;
  // Innermost first (rerolling an inner loop changes its body, but the
  // outer loop's structure is preserved).
  for (Loop *SL : L->getSubLoops())
    Changed |= tryRerollLoopAndSubloops(SL);
  Changed |= tryRerollLoop(L);
  return Changed;
}

bool AArch64LoopDeunrollImpl::tryRerollLoop(Loop *L) {
  // 1. Single-block loop (header == latch) with preheader.
  BasicBlock *Header = L->getHeader();
  BasicBlock *Latch = L->getLoopLatch();
  BasicBlock *Preheader = L->getLoopPreheader();
  if (!Latch || Latch != Header || !Preheader)
    return false;

  // 2. Back-branch + guard icmp.
  auto *Br = dyn_cast<BranchInst>(Header->getTerminator());
  if (!Br || !Br->isConditional())
    return false;
  if (Br->getSuccessor(0) != Header && Br->getSuccessor(1) != Header)
    return false;
  auto *Cmp = dyn_cast<ICmpInst>(Br->getCondition());
  if (!Cmp)
    return false;

  // 3. Find header phis:
  //    - i64 IV with `add IV, N` step (for guard / Pattern A).
  //    - ptr phis with `gep base, const` back-edge (Pattern B).
  PHINode *IV = nullptr;
  BinaryOperator *StepBO = nullptr;
  unsigned NFromIV = 0;
  SmallVector<PtrPhiInfo, 4> PtrPhis;

  for (PHINode &PN : Header->phis()) {
    if (PN.getNumIncomingValues() != 2)
      continue;
    int PrehIdx = PN.getBasicBlockIndex(Preheader);
    if (PrehIdx < 0)
      continue;
    int LatchIdx = 1 - PrehIdx;
    if (PN.getIncomingBlock(LatchIdx) != Latch)
      continue;
    Value *BackEdge = PN.getIncomingValue(LatchIdx);

    if (PN.getType()->isIntegerTy(64)) {
      auto *BO = dyn_cast<BinaryOperator>(BackEdge);
      if (!BO || BO->getOpcode() != Instruction::Add)
        continue;
      auto Off = matchOffset(BO, &PN);
      if (!Off)
        continue;
      if (!IV) {  // take the first i64 IV
        IV = &PN;
        StepBO = BO;
        NFromIV = *Off;
      }
    } else if (PN.getType()->isPointerTy()) {
      auto *GEP = dyn_cast<GetElementPtrInst>(BackEdge);
      if (!GEP)
        continue;
      if (GEP->getPointerOperand() != &PN)
        continue;
      auto ByteOff = getGEPConstByteOffset(GEP);
      if (!ByteOff)
        continue;
      PtrPhis.push_back({&PN, GEP, *ByteOff});
    }
  }

  // 4. Determine N and elem size.
  //    N can come from the i64 IV step, or from the ptr-phi back-edge offset
  //    divided by the element size (derived from loads in the body).
  unsigned N = 0;
  uint64_t ElemBytes = 0;

  // Find element size from a load whose pointer is a ptr phi or a GEP of one.
  if (!PtrPhis.empty()) {
    SmallPtrSet<Value *, 4> PhiSet;
    for (auto &P : PtrPhis)
      PhiSet.insert(P.Phi);
    for (Instruction &I : *Header) {
      auto *LI = dyn_cast<LoadInst>(&I);
      if (!LI)
        continue;
      Value *Ptr = LI->getPointerOperand();
      // direct load from ptr phi
      if (PhiSet.count(Ptr)) {
        ElemBytes = LI->getType()->getScalarSizeInBits() / 8;
        break;
      }
      // load from GEP whose base is a ptr phi
      if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
        if (PhiSet.count(GEP->getPointerOperand())) {
          ElemBytes = LI->getType()->getScalarSizeInBits() / 8;
          break;
        }
      }
    }
    if (ElemBytes == 0)
      ElemBytes = 8;  // fallback
  }

  unsigned NFromPtr = 0;
  if (!PtrPhis.empty() && ElemBytes > 0)
    NFromPtr = PtrPhis[0].OffsetBytes / ElemBytes;

  // Validate N: must have at least one source; if both, they must agree.
  if (NFromIV == 0 && NFromPtr == 0)
    return false;
  if (NFromIV != 0 && NFromPtr != 0 && NFromIV != NFromPtr)
    return false;
  N = NFromIV ? NFromIV : NFromPtr;
  if (N < 2 || N > 8)
    return false;

  // 5. Collect deletable GEPs from both patterns.
  SmallPtrSet<Instruction *, 32> Deletable;
  SmallVector<Instruction *, 32> Worklist;

  // Pattern A: GEPs whose index is (or disjoint | add) IV, k for k=1..N-1.
  if (IV) {
    for (Instruction &I : *Header) {
      auto *GEP = dyn_cast<GetElementPtrInst>(&I);
      if (!GEP || GEP->getNumOperands() < 2)
        continue;
      Value *Idx = GEP->getOperand(1);
      auto Off = matchOffset(Idx, IV);
      if (!Off)
        continue;
      unsigned K = *Off;
      if (K == 0 || K >= N)
        continue;
      Deletable.insert(GEP);
      Worklist.push_back(GEP);
    }
  }

  // Pattern B: GEPs whose base is a ptr phi, with const offset k*elem_bytes,
  //             k=1..N-1. Skip the back-edge GEPs (k=N).
  SmallPtrSet<const Value *, 4> BackEdgeGEPs;
  for (auto &P : PtrPhis)
    BackEdgeGEPs.insert(P.BackEdgeGEP);

  for (Instruction &I : *Header) {
    auto *GEP = dyn_cast<GetElementPtrInst>(&I);
    if (!GEP || Deletable.count(GEP) || BackEdgeGEPs.count(GEP))
      continue;
    auto ByteOff = getGEPConstByteOffset(GEP);
    if (!ByteOff)
      continue;
    // Must use one of the ptr phis as base.
    bool IsFromPtrPhi = false;
    for (auto &P : PtrPhis) {
      if (GEP->getPointerOperand() == P.Phi) {
        IsFromPtrPhi = true;
        break;
      }
    }
    if (!IsFromPtrPhi)
      continue;
    if (ElemBytes == 0 || *ByteOff % ElemBytes != 0)
      continue;
    unsigned K = *ByteOff / ElemBytes;
    if (K == 0 || K >= N)
      continue;
    Deletable.insert(GEP);
    Worklist.push_back(GEP);
  }

  if (Worklist.empty())
    return false;

  // 6. Forward walk: collect all instructions transitively dependent on the
  //    deletable GEPs. Exclude loop-control instructions (phis, step, guard,
  //    branch) and the ptr-phi back-edge GEPs.
  SmallPtrSet<Instruction *, 8> ControlInsts;
  if (IV) ControlInsts.insert(IV);
  if (StepBO) ControlInsts.insert(StepBO);
  ControlInsts.insert(Cmp);
  ControlInsts.insert(Br);
  for (auto &P : PtrPhis)
    ControlInsts.insert(P.BackEdgeGEP);

  while (!Worklist.empty()) {
    Instruction *I = Worklist.pop_back_val();
    for (User *U : I->users()) {
      auto *UI = dyn_cast<Instruction>(U);
      if (!UI || UI->getParent() != Header)
        continue;
      if (Deletable.count(UI) || ControlInsts.count(UI))
        continue;
      // Don't pull in the ptr phis themselves.
      if (isa<PHINode>(UI))
        continue;
      Deletable.insert(UI);
      Worklist.push_back(UI);
    }
  }

  // 7. Update the guard: an icmp operand may be `(or disjoint | add) StepBO,
  //    (N-1)`. Replace with StepBO so the guard becomes `step < bound`.
  Instruction *GuardOffsetInst = nullptr;
  if (StepBO) {
    for (int OpIdx = 0; OpIdx < 2; OpIdx++) {
      Value *Op = Cmp->getOperand(OpIdx);
      auto Off = matchOffset(Op, StepBO);
      if (!Off || *Off != N - 1)
        continue;
      GuardOffsetInst = dyn_cast<Instruction>(Op);
      Cmp->setOperand(OpIdx, StepBO);
      break;
    }
    if (!GuardOffsetInst) {
      // Guard must be directly on StepBO or we abort.
      if (Cmp->getOperand(0) != StepBO && Cmp->getOperand(1) != StepBO)
        return false;
    }
  }

  // 8. Add pattern-A offset roots and the guard offset inst to Deletable.
  if (IV) {
    for (Instruction &I : *Header) {
      if (&I == IV || &I == StepBO)
        continue;
      auto Off = matchOffset(&I, IV);
      if (!Off)
        continue;
      unsigned K = *Off;
      if (K == 0 || K >= N)
        continue;
      Deletable.insert(&I);
    }
  }
  if (GuardOffsetInst)
    Deletable.insert(GuardOffsetInst);

  // 9. Verify: no deletable instruction has an external use.
  for (Instruction *I : Deletable) {
    for (User *U : I->users()) {
      auto *UI = dyn_cast<Instruction>(U);
      if (!UI)
        continue;
      if (Deletable.count(UI))
        continue;
      return false;  // cross-body dependency; abort
    }
  }

  // 10. Update step constants.
  //     i64 IV:  add IV, N  ->  add IV, 1
  //     ptr phi back-edge GEP:  ..., N*elem_bytes  ->  ..., 1*elem_bytes
  if (StepBO) {
    if (StepBO->getOperand(0) == IV)
      StepBO->setOperand(1, ConstantInt::get(IV->getType(), 1));
    else
      StepBO->setOperand(0, ConstantInt::get(IV->getType(), 1));
  }
  for (auto &P : PtrPhis) {
    auto *GEP = P.BackEdgeGEP;
    auto *C = dyn_cast<ConstantInt>(GEP->getOperand(1));
    if (!C)
      continue;
    GEP->setOperand(1, ConstantInt::get(C->getType(), ElemBytes));
  }

  // 11. Erase deletable instructions (reverse order for def-use safety).
  SmallVector<Instruction *, 32> ToErase;
  for (Instruction &I : *Header)
    if (Deletable.count(&I))
      ToErase.push_back(&I);
  for (Instruction *I : reverse(ToErase))
    I->eraseFromParent();

  return true;
}

PreservedAnalyses
AArch64LoopDeunrollPass::run(Function &F, FunctionAnalysisManager &AM) {
  auto &LI = AM.getResult<LoopAnalysis>(F);
  return AArch64LoopDeunrollImpl{}.run(F, LI);
}

// Legacy pass wrapper so the transform also runs in the legacy backend
// pipeline (llc -O1, opt -enable-new-pm=0).
namespace {
class AArch64LoopDeunroll : public FunctionPass {
public:
  static char ID;
  AArch64LoopDeunroll() : FunctionPass(ID) {
    initializeAArch64LoopDeunrollPass(*PassRegistry::getPassRegistry());
  }
  bool runOnFunction(Function &F) override {
    auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    auto PA = AArch64LoopDeunrollImpl{}.run(F, LI);
    return !PA.areAllPreserved();
  }
  StringRef getPassName() const override {
    return "AArch64 loop de-unroll (reroll manual N-x unroll)";
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LoopInfoWrapperPass>();
  }
};
} // end anonymous namespace

char AArch64LoopDeunroll::ID = 0;
INITIALIZE_PASS_BEGIN(AArch64LoopDeunroll, DEBUG_TYPE,
                      "AArch64 loop de-unroll", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(AArch64LoopDeunroll, DEBUG_TYPE,
                    "AArch64 loop de-unroll", false, false)

FunctionPass *llvm::createAArch64LoopDeunrollPass() {
  return new AArch64LoopDeunroll();
}
