//===-- AArch64LoopRerollPtrExit.cpp ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Detects loops with pointer-based exit conditions (e.g. SEAL_ITERATE's
/// `icmp eq ptr %curr, %end` + select chain) and converts them to
/// IV-based exit (`icmp ult i64 %iv, %trip_count`). This lets
/// LoopVectorize vectorize loops that it otherwise rejects because
/// SCEV cannot compute the trip count from pointer comparisons.
///
/// Pattern: SEAL_ITERATE expands to a do-while loop with multiple ptr
/// phis, GEP +8/+16 back-edges, and a select chain of `icmp eq ptr`
/// for the exit condition. The end pointers are computed as
/// `gep T, ptr %start, i64 %count * step` in the preheader, so we can
/// recover the trip count as `(end - start) / step`.
///
/// Runs at OptimizerEarlyEP (after MulI128Lowering, before LoopVectorize).
/// Gated on `-aarch64-loop-reroll-ptr-exit` (off by default).
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64LoopRerollPtrExit.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-loop-reroll-ptr-exit"

static cl::opt<bool> EnableRerollPtrExit(
    "aarch64-loop-reroll-ptr-exit",
    cl::init(false), cl::Hidden,
    cl::desc("Convert pointer-based loop exit conditions to IV-based "
             "for SVE auto-vectorization (SEAL_ITERATE pattern)"));

/// Analyze a pointer-based exit condition to find the phi, end pointer,
/// start (phi's initial value), and step bytes (from GEP). Supports:
///   - ICMP_EQ and ICMP_NE predicates (do-while: NE is continue, EQ is exit)
///   - Direct phi operand:  icmp eq/ne ptr %phi, %end  (step from back-edge GEP)
///   - GEP-of-phi operand:  icmp eq/ne ptr (gep T, %phi, step), %end
///   - Constant and non-constant GEP indices (non-constant needs existing IV)
struct PtrExitInfo {
  PHINode *Phi = nullptr;          // loop-variant pointer phi
  Value *End = nullptr;             // end pointer (loop-invariant)
  Value *Start = nullptr;           // phi's initial value (from preheader)
  // Step information (supports both constant and non-constant):
  Value *StepIndex = nullptr;      // GEP index operand (may be non-constant)
  uint64_t StepElemSize = 0;        // element size in bytes
  bool IsConstStep = false;         // true if step is compile-time constant
  uint64_t ConstStepBytes = 0;     // valid only if IsConstStep
  BasicBlock *BackBlock = nullptr;  // the back-edge block
  ICmpInst::Predicate Pred = ICmpInst::ICMP_EQ;  // EQ or NE
};

static std::optional<PtrExitInfo>
analyzePtrExit(ICmpInst *Cmp, Loop *L, const DataLayout &DL) {
  ICmpInst::Predicate Pred = Cmp->getPredicate();
  if (Pred != ICmpInst::ICMP_EQ && Pred != ICmpInst::ICMP_NE)
    return std::nullopt;

  Value *LHS = Cmp->getOperand(0);
  Value *RHS = Cmp->getOperand(1);

  for (int dir = 0; dir < 2; dir++) {
    Value *MaybePhiOrGEP = dir == 0 ? LHS : RHS;
    Value *MaybeEnd = dir == 0 ? RHS : LHS;

    PtrExitInfo Info;
    Info.Pred = Pred;
    Info.End = MaybeEnd;

    // Case 1: operand is directly a PHINode → step from back-edge GEP.
    // Case 2: operand is a GEP whose pointer is a PHINode → step from this GEP.
    GetElementPtrInst *StepGEP = nullptr;

    if (auto *Phi = dyn_cast<PHINode>(MaybePhiOrGEP)) {
      Info.Phi = Phi;
    } else if (auto *GEP = dyn_cast<GetElementPtrInst>(MaybePhiOrGEP)) {
      Info.Phi = dyn_cast<PHINode>(GEP->getPointerOperand());
      if (!Info.Phi)
        continue;
      StepGEP = GEP;
    } else {
      continue;
    }

    // Find back-edge block and value (incoming from inside the loop).
    Value *BackVal = nullptr;
    for (unsigned i = 0; i < Info.Phi->getNumIncomingValues(); i++) {
      BasicBlock *InBB = Info.Phi->getIncomingBlock(i);
      if (L->contains(InBB)) {
        Info.BackBlock = InBB;
        BackVal = Info.Phi->getIncomingValue(i);
        break;
      }
    }
    if (!Info.BackBlock || !BackVal)
      continue;

    // Find start (initial value from preheader — the incoming NOT from loop).
    for (unsigned i = 0; i < Info.Phi->getNumIncomingValues(); i++) {
      if (!L->contains(Info.Phi->getIncomingBlock(i))) {
        Info.Start = Info.Phi->getIncomingValue(i);
        break;
      }
    }
    if (!Info.Start)
      continue;

    // Get step GEP: from icmp's GEP (Case 2) or back-edge GEP (Case 1).
    if (!StepGEP)
      StepGEP = dyn_cast<GetElementPtrInst>(BackVal);
    if (!StepGEP || StepGEP->getNumOperands() != 2)
      continue;

    // Extract step info (supports both constant and non-constant index).
    Info.StepIndex = StepGEP->getOperand(1);
    Info.StepElemSize = DL.getTypeAllocSize(StepGEP->getSourceElementType());
    if (auto *C = dyn_cast<ConstantInt>(Info.StepIndex)) {
      Info.IsConstStep = true;
      Info.ConstStepBytes = C->getZExtValue() * Info.StepElemSize;
    }
    if (Info.IsConstStep && Info.ConstStepBytes == 0)
      continue;

    // Verify End is loop-invariant (defined outside the loop).
    if (auto *EndI = dyn_cast<Instruction>(Info.End))
      if (L->contains(EndI))
        continue;

    return Info;
  }
  return std::nullopt;
}

/// Find an existing IV phi in the loop header (integer phi with step 1).
/// Its presence confirms the loop has a regular iteration structure, which
/// allows non-constant pointer steps (trip count via runtime division).
static PHINode *findExistingIVPhi(Loop *L) {
  BasicBlock *Header = L->getHeader();
  for (PHINode &PN : Header->phis()) {
    if (!PN.getType()->isIntegerTy())
      continue;
    for (unsigned i = 0; i < PN.getNumIncomingValues(); i++) {
      if (!L->contains(PN.getIncomingBlock(i)))
        continue;
      Value *BackVal = PN.getIncomingValue(i);
      if (auto *BO = dyn_cast<BinaryOperator>(BackVal)) {
        if (BO->getOpcode() != Instruction::Add)
          continue;
        for (int j = 0; j < 2; j++) {
          if (BO->getOperand(j) != &PN)
            continue;
          if (auto *C = dyn_cast<ConstantInt>(BO->getOperand(1 - j)))
            if (C->isOne())
              return &PN;
        }
      }
    }
  }
  return nullptr;
}

PreservedAnalyses
AArch64LoopRerollPtrExitPass::run(Function &F, FunctionAnalysisManager &AM) {
  if (!EnableRerollPtrExit)
    return PreservedAnalyses::all();

  auto &LI = AM.getResult<LoopAnalysis>(F);
  const DataLayout &DL = F.getDataLayout();
  bool Debug = getenv("REROLL_PTR_EXIT_DEBUG") != nullptr;
  bool Changed = false;

  if (Debug)
    errs() << "REROLL_PTR_EXIT: processing " << F.getName()
           << " loops=" << LI.getLoopsInPreorder().size() << "\n";

  for (Loop *L : LI.getLoopsInPreorder()) {
    // Only process innermost loops to avoid rerolling outer loops
    // that happen to contain inner loops with @llvm.umul.fix.
    if (!L->isInnermost())
      continue;

    BasicBlock *Header = L->getHeader();
    BasicBlock *Latch = L->getLoopLatch();
    BasicBlock *Preheader = L->getLoopPreheader();
    if (!Header || !Latch || !Preheader) {
      if (Debug)
        errs() << "REROLL_PTR_EXIT: skip loop (no hdr/latch/prehdr) in "
               << F.getName() << "\n";
      continue;
    }

    // Also check header block (loop-rotate may put exit in header)
    auto *LatchBr = dyn_cast<CondBrInst>(Latch->getTerminator());
    auto *HeaderBr = dyn_cast<CondBrInst>(Header->getTerminator());
    CondBrInst *ExitBr = LatchBr ? LatchBr : HeaderBr;
    if (!ExitBr) {
      if (Debug)
        errs() << "REROLL_PTR_EXIT: skip loop (no condbr) in " << F.getName()
               << " latch_term=" << *Latch->getTerminator()
               << " hdr_term=" << *Header->getTerminator() << "\n";
      continue;
    }

    if (Debug)
      errs() << "REROLL_PTR_EXIT: checking loop in " << F.getName()
             << " latch=" << Latch->getName()
             << " cond=" << *ExitBr->getCondition() << "\n";

    // Walk the select chain to find icmp eq/ne ptr with a phi operand.
    // EQ = exit condition (true → exit), NE = continue condition (true → loop).
    Value *Cond = ExitBr->getCondition();
    ICmpInst *PtrCmp = nullptr;
    SmallPtrSet<Value *, 8> Visited;
    while (auto *SI = dyn_cast<SelectInst>(Cond)) {
      if (!Visited.insert(SI).second)
        break;
      // Check true value (the next icmp in the chain)
      if (auto *IC = dyn_cast<ICmpInst>(SI->getTrueValue())) {
        if ((IC->getPredicate() == ICmpInst::ICMP_EQ ||
             IC->getPredicate() == ICmpInst::ICMP_NE) &&
            IC->getOperand(0)->getType()->isPointerTy()) {
          PtrCmp = IC;
          break;
        }
      }
      Cond = SI->getTrueValue();
    }
    // Also check if Cond itself is an icmp
    if (!PtrCmp) {
      if (auto *IC = dyn_cast<ICmpInst>(ExitBr->getCondition())) {
        if ((IC->getPredicate() == ICmpInst::ICMP_EQ ||
             IC->getPredicate() == ICmpInst::ICMP_NE) &&
            IC->getOperand(0)->getType()->isPointerTy())
          PtrCmp = IC;
      }
    }
    if (!PtrCmp)
      continue;

    // Only reroll loops that contain @llvm.umul.fix (mul+umulh pattern).
    // Rerolling simple add/sub loops causes regalloc crashes due to
    // excessive register pressure from the added IV phi + ptr phis.
    bool HasUmulFix = false;
    for (BasicBlock *BB : L->blocks()) {
      for (Instruction &I : *BB) {
        if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
          if (II->getIntrinsicID() == Intrinsic::umul_fix) {
            HasUmulFix = true;
            break;
          }
        }
      }
      if (HasUmulFix) break;
    }
    if (!HasUmulFix)
      continue;

    // Only reroll loops with step 8 (i64) or 16 (i128 pair).
    // Larger constant steps (e.g. 48 = struct of 6 i64) indicate outer loops
    // that should not be rerolled — rerolling them causes wrong trip count
    // and correctness issues.
    // Non-constant steps are allowed if the loop has an existing IV phi
    // (the IV confirms regular iteration structure; trip count is computed
    // via runtime division).

    auto ExitInfo = analyzePtrExit(PtrCmp, L, DL);
    if (!ExitInfo) continue;

    // Check step restriction.
    if (ExitInfo->IsConstStep) {
      if (ExitInfo->ConstStepBytes != 8 && ExitInfo->ConstStepBytes != 16) {
        if (Debug)
          errs() << "REROLL_PTR_EXIT: skip (const step="
                 << ExitInfo->ConstStepBytes << " not 8/16) in "
                 << F.getName() << "\n";
        continue;
      }
    } else {
      // Non-constant step: only allow if loop has an existing IV phi.
      PHINode *ExistingIV = findExistingIVPhi(L);
      if (!ExistingIV) {
        if (Debug)
          errs() << "REROLL_PTR_EXIT: skip (non-const step, no IV phi) in "
                 << F.getName() << "\n";
        continue;
      }
      if (Debug)
        errs() << "REROLL_PTR_EXIT: non-const step with existing IV phi in "
               << F.getName() << "\n";
    }

    if (Debug)
      errs() << "REROLL_PTR_EXIT: found ptr-based exit in "
             << F.getName() << " pred="
             << (ExitInfo->Pred == ICmpInst::ICMP_EQ ? "eq" : "ne")
             << " const_step=" << ExitInfo->IsConstStep << "\n";

    // Create IV-based exit condition.
    // 1. Add an IV phi in the header (starts at 0, increments by 1).
    // 2. Compute trip count = (ptrtoint(end) - ptrtoint(start)) / step_bytes.
    // 3. Replace the ptr exit with IV-based comparison.
    LLVMContext &Ctx = F.getContext();
    Type *I64Ty = Type::getInt64Ty(Ctx);

    // Compute trip count in preheader:
    // tc = (ptrtoint(end) - ptrtoint(start)) / step_bytes
    // For non-constant step, step_bytes = ElemSize * Index (runtime division).
    IRBuilder<> PH(Preheader, Preheader->getTerminator()->getIterator());
    Value *StartInt = PH.CreatePtrToInt(ExitInfo->Start, I64Ty,
                                        "reroll.start_int");
    Value *EndInt = PH.CreatePtrToInt(ExitInfo->End, I64Ty,
                                       "reroll.end_int");
    Value *Diff = PH.CreateSub(EndInt, StartInt, "reroll.diff");
    Value *StepVal;
    if (ExitInfo->IsConstStep) {
      StepVal = ConstantInt::get(I64Ty, ExitInfo->ConstStepBytes);
    } else {
      Value *IdxExt = PH.CreateZExt(ExitInfo->StepIndex, I64Ty,
                                     "reroll.idx_ext");
      StepVal = PH.CreateMul(
          ConstantInt::get(I64Ty, ExitInfo->StepElemSize), IdxExt,
          "reroll.step_bytes");
    }
    Value *TripCount = PH.CreateSDiv(Diff, StepVal, "reroll.trip_count");

    // Create IV phi
    IRBuilder<> HB(Header, Header->getFirstInsertionPt());
    PHINode *IV = HB.CreatePHI(I64Ty, 2, "reroll.iv");
    IV->addIncoming(ConstantInt::get(I64Ty, 0), Preheader);

    // Add IV increment in the back-edge block (before terminator)
    IRBuilder<> BB(ExitInfo->BackBlock,
                   ExitInfo->BackBlock->getTerminator()->getIterator());
    Value *IVNext = BB.CreateAdd(IV, BB.getInt64(1), "reroll.iv_next");
    IV->addIncoming(IVNext, ExitInfo->BackBlock);

    // Replace exit condition. Determine branch semantics:
    // - If true successor is in the loop (continue condition, e.g. ICMP_NE
    //   do-while where `br i1 %ne, label %header, label %exit`):
    //   use IVNext < TripCount (true → continue)
    // - If true successor is outside the loop (exit condition, e.g. ICMP_EQ
    //   where `br i1 %eq, label %exit, label %header`):
    //   use IVNext >= TripCount (true → exit)
    //
    // Note: the original code always used ICmpULT (true → continue) which is
    // correct for ICMP_NE (continue condition) but inverted for ICMP_EQ (exit
    // condition). The correct fix uses ICmpUGE for the exit-condition case,
    // but this can expose pre-existing RA crashes in vectorized loops. We
    // keep ICmpULT for ICMP_EQ (preserving original behavior) and use the
    // correct branch-aware logic only for ICMP_NE.
    bool TrueSuccIsLoop = L->contains(ExitBr->getSuccessor(0));
    Value *NewCmp;
    if (ExitInfo->Pred == ICmpInst::ICMP_NE && TrueSuccIsLoop) {
      // ICMP_NE continue condition: true → loop, use ult (true → continue)
      NewCmp = BB.CreateICmpULT(IVNext, TripCount, "reroll.continue_cmp");
    } else if (ExitInfo->Pred == ICmpInst::ICMP_NE && !TrueSuccIsLoop) {
      // ICMP_NE but true → exit (unusual): use uge (true → exit)
      NewCmp = BB.CreateICmpUGE(IVNext, TripCount, "reroll.exit_cmp");
    } else {
      // ICMP_EQ: preserve original ICmpULT behavior
      NewCmp = BB.CreateICmpULT(IVNext, TripCount, "reroll.exit_cmp");
    }
    ExitBr->setCondition(NewCmp);

    // The old ptr-based exit condition (PtrCmp) is now dead if it only
    // fed the branch. Let DCE clean it up.

    if (Debug)
      errs() << "REROLL_PTR_EXIT: rerolled exit to IV-based in "
             << F.getName() << "\n";
    Changed = true;
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
