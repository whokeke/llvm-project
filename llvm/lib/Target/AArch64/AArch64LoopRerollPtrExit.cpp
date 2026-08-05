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

/// Check if a Value is a GEP with a constant offset (e.g. gep i8, ptr %x, i64 8).
/// Returns the base pointer and the byte offset, or nullopt if not a simple GEP.
static std::optional<std::pair<Value *, uint64_t>>
getConstGEPBaseAndOffset(Value *V, const DataLayout &DL) {
  auto *GEP = dyn_cast<GetElementPtrInst>(V);
  if (!GEP || GEP->getNumOperands() != 2)
    return std::nullopt;
  auto *C = dyn_cast<ConstantInt>(GEP->getOperand(1));
  if (!C)
    return std::nullopt;
  uint64_t ElemSize = DL.getTypeAllocSize(GEP->getSourceElementType());
  return std::make_pair(GEP->getPointerOperand(), C->getZExtValue() * ElemSize);
}

/// Analyze a pointer-based exit condition to find the phi, end pointer,
/// start (phi's initial value), and step bytes (from back-edge GEP).
struct PtrExitInfo {
  PHINode *Phi = nullptr;     // loop-variant pointer phi
  Value *End = nullptr;       // end pointer (loop-invariant)
  Value *Start = nullptr;     // phi's initial value (from preheader)
  uint64_t StepBytes = 0;     // back-edge GEP step in bytes
  BasicBlock *BackBlock = nullptr;  // the back-edge block
  Value *BackVal = nullptr;   // the back-edge value (GEP that advances phi)
};

static std::optional<PtrExitInfo>
analyzePtrExit(ICmpInst *Cmp, Loop *L, const DataLayout &DL) {
  if (Cmp->getPredicate() != ICmpInst::ICMP_EQ)
    return std::nullopt;
  Value *LHS = Cmp->getOperand(0);
  Value *RHS = Cmp->getOperand(1);

  for (int dir = 0; dir < 2; dir++) {
    Value *MaybePhi = dir == 0 ? LHS : RHS;
    Value *MaybeEnd = dir == 0 ? RHS : LHS;
    auto *Phi = dyn_cast<PHINode>(MaybePhi);
    if (!Phi)
      continue;

    PtrExitInfo Info;
    Info.Phi = Phi;
    Info.End = MaybeEnd;

    // Find back-edge value (incoming from a block inside the loop).
    for (unsigned i = 0; i < Phi->getNumIncomingValues(); i++) {
      BasicBlock *InBB = Phi->getIncomingBlock(i);
      if (L->contains(InBB)) {
        Info.BackBlock = InBB;
        Info.BackVal = Phi->getIncomingValue(i);
        break;
      }
    }
    if (!Info.BackVal)
      continue;

    // Find start (initial value from preheader — the incoming that's NOT
    // from inside the loop).
    for (unsigned i = 0; i < Phi->getNumIncomingValues(); i++) {
      if (!L->contains(Phi->getIncomingBlock(i))) {
        Info.Start = Phi->getIncomingValue(i);
        break;
      }
    }
    if (!Info.Start)
      continue;

    // Get step bytes from back-edge GEP.
    auto BackGEP = getConstGEPBaseAndOffset(Info.BackVal, DL);
    if (!BackGEP)
      continue;
    Info.StepBytes = BackGEP->second;
    if (Info.StepBytes == 0)
      continue;

    // Verify End is loop-invariant (defined outside the loop).
    if (auto *EndI = dyn_cast<Instruction>(Info.End))
      if (L->contains(EndI))
        continue;

    return Info;
  }
  return std::nullopt;
}

PreservedAnalyses
AArch64LoopRerollPtrExitPass::run(Function &F, FunctionAnalysisManager &AM) {
  if (!EnableRerollPtrExit)
    return PreservedAnalyses::all();

  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
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

    // Walk the select chain to find icmp eq ptr with a phi operand.
    Value *Cond = ExitBr->getCondition();
    ICmpInst *PtrCmp = nullptr;
    SmallPtrSet<Value *, 8> Visited;
    while (auto *SI = dyn_cast<SelectInst>(Cond)) {
      if (!Visited.insert(SI).second)
        break;
      // Check true value (the next icmp in the chain)
      if (auto *IC = dyn_cast<ICmpInst>(SI->getTrueValue())) {
        if (IC->getPredicate() == ICmpInst::ICMP_EQ &&
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
        if (IC->getPredicate() == ICmpInst::ICMP_EQ &&
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
    // Larger steps (e.g. 48 = struct of 6 i64) indicate outer loops
    // that should not be rerolled — rerolling them causes wrong trip
    // count and correctness issues.
    // We'll check this after analyzePtrExit returns the step.

    auto ExitInfo = analyzePtrExit(PtrCmp, L, DL);
    if (!ExitInfo) continue;

    // Only reroll loops with step 8 (i64 element) or 16 (i128 pair).
    // Larger steps indicate outer loops (e.g. step=48 = struct of 6 i64)
    // that should not be rerolled.
    if (ExitInfo->StepBytes != 8 && ExitInfo->StepBytes != 16) {
      if (Debug)
        errs() << "REROLL_PTR_EXIT: skip (step=" << ExitInfo->StepBytes
               << " not 8/16) in " << F.getName() << "\n";
      continue;
    }

    if (Debug)
      errs() << "REROLL_PTR_EXIT: found ptr-based exit in "
             << F.getName() << " step=" << ExitInfo->StepBytes << "\n";

    // Create IV-based exit condition.
    // 1. Add an IV phi in the header (starts at 0, increments by 1).
    // 2. Compute trip count = (ptrtoint(end) - ptrtoint(start)) / step_bytes.
    // 3. Replace the ptr exit with `icmp ult i64 %iv, %trip_count`.
    LLVMContext &Ctx = F.getContext();
    Type *I64Ty = Type::getInt64Ty(Ctx);

    // Compute trip count in preheader:
    // tc = (ptrtoint(end) - ptrtoint(start)) / step_bytes
    IRBuilder<> PH(Preheader, Preheader->getTerminator()->getIterator());
    Value *StartInt = PH.CreatePtrToInt(ExitInfo->Start, I64Ty,
                                       "reroll.start_int");
    Value *EndInt = PH.CreatePtrToInt(ExitInfo->End, I64Ty,
                                      "reroll.end_int");
    Value *Diff = PH.CreateSub(EndInt, StartInt, "reroll.diff");
    Value *TripCount = PH.CreateSDiv(Diff,
        ConstantInt::get(I64Ty, ExitInfo->StepBytes),
        "reroll.trip_count");

    // Create IV phi
    IRBuilder<> HB(Header, Header->getFirstInsertionPt());
    PHINode *IV = HB.CreatePHI(I64Ty, 2, "reroll.iv");
    IV->addIncoming(ConstantInt::get(I64Ty, 0), Preheader);

    // Add IV increment in the back-edge block (before terminator)
    IRBuilder<> BB(ExitInfo->BackBlock,
                   ExitInfo->BackBlock->getTerminator()->getIterator());
    Value *IVNext = BB.CreateAdd(IV, BB.getInt64(1), "reroll.iv_next");
    IV->addIncoming(IVNext, ExitInfo->BackBlock);

    // Replace exit condition
    Value *NewCmp = BB.CreateICmpULT(IVNext, TripCount,
                                     "reroll.exit_cmp");
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
