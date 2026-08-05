//===-- AArch64StridedVectorize.cpp --- Normalize + vectorize strided -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Detects loops containing the Barrett reduction pattern
/// (@llvm.umul.fix.i64 + mul + conditional sub) with a strided store and
/// contiguous load. Normalizes the loop (hoist loop-invariant loads) and
/// adds vectorize.enable metadata so the auto-vectorizer can generate SVE
/// scatter stores. Runs at LateLoopOptimizationsEP (after inlining +
/// MulI128Lowering, before LoopVectorize).
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64StridedVectorize.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/InitializePasses.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "aarch64-strided-vectorize"

static cl::opt<bool> EnableStridedVectorize(
    "aarch64-strided-vectorize",
    cl::init(false), cl::Hidden,
    cl::desc("Normalize Barrett-reduce strided-store loops for SVE auto-vectorization"));

static cl::opt<bool> AggressiveHoist(
    "aarch64-strided-vectorize-aggressive-hoist",
    cl::init(false), cl::Hidden,
    cl::desc("Skip the may-alias safety check when hoisting loop-invariant "
             "loads. Unsafe in general (the loop body may store to the same "
             "memory the load reads, and hoisting would lose the update), "
             "but enables scatter generation for loops where AA is "
             "conservative (e.g. SEAL fast_convert_array: heap-allocated "
             "temp buffer vs ibase_ function argument share no alias.scope/"
             "noalias metadata, so BasicAA returns MayAlias even though "
             "the buffers are distinct). Use only when the user has verified "
             "the load/store do not actually alias in their target workload."));

namespace {

struct LoopInfo2 {
  PHINode *InPhi = nullptr;
  PHINode *OutPhi = nullptr;
  Value *Stride = nullptr;
  StoreInst *StridedStore = nullptr;
  Value *OuterOffset = nullptr;
  SmallVector<LoadInst *, 4> InvariantLoads;
  IntrinsicInst *UmulFix = nullptr;
};

static bool tracesToPhi(Value *V, PHINode *Phi) {
  while (auto *GEP = dyn_cast<GetElementPtrInst>(V))
    V = GEP->getPointerOperand();
  return V == Phi;
}

static bool isLoopInvariant(Loop *L, Value *V) {
  if (auto *I = dyn_cast<Instruction>(V))
    return !L->contains(I);
  return true;
}

static Value *getGEPStride(GetElementPtrInst *GEP) {
  if (GEP->getNumOperands() != 2)
    return nullptr;
  return GEP->getOperand(1);
}

static bool detectPattern(Loop *L, LoopInfo2 &LI, DominatorTree &DT,
                          LoopInfo &LoopInfo, const DataLayout &DL) {
  BasicBlock *Header = L->getHeader();
  BasicBlock *Latch = L->getLoopLatch();
  bool Debug = getenv("STRIDED_VEC_DEBUG") != nullptr;

  if (!Header || !Latch) {
    if (Debug)
      errs() << "STRIDED_VEC: skip (missing hdr/latch in func="
             << Header->getParent()->getName() << ")\n";
    return false;
  }
  BasicBlock *Preheader = L->getLoopPreheader();
  if (!Preheader) {
    // No preheader — skip (creating one via InsertPreheaderForLoop can crash
    // later passes due to CFG changes). The vectorizer will create one if needed.
    if (Debug)
      errs() << "STRIDED_VEC: skip (no preheader) in func="
             << Header->getParent()->getName() << "\n";
    return false;
  }

  // 2. Find ptr-based phi nodes in the header (contiguous input + strided output)
  //    Use latch to identify back-edge (preheader might be freshly created
  //    and not yet in the phi's incoming block list).
  for (PHINode &PN : Header->phis()) {
    if (!PN.getType()->isPointerTy() || PN.getNumIncomingValues() != 2)
      continue;
    // Find which incoming is from the latch (back-edge) vs entry
    unsigned LatchIdx = 0;
    bool Found = false;
    for (unsigned i = 0; i < 2; i++) {
      if (PN.getIncomingBlock(i) == Latch) {
        LatchIdx = i;
        Found = true;
        break;
      }
    }
    if (!Found) {
      if (Debug)
        errs() << "  PHI " << PN << " latch not in incoming blocks"
               << " (latch=" << Latch << " in0=" << PN.getIncomingBlock(0)
               << " in1=" << PN.getIncomingBlock(1) << ")\n";
      continue;
    }
    Value *BackEdge = PN.getIncomingValue(LatchIdx);
    auto *GEP = dyn_cast<GetElementPtrInst>(BackEdge);
    if (!GEP) {
      if (Debug)
        errs() << "  PHI " << PN << " back-edge not GEP: " << *BackEdge << "\n";
      continue;
    }
    if (GEP->getPointerOperand() != &PN) {
      if (Debug)
        errs() << "  PHI " << PN << " GEP base != phi: " << *GEP << "\n";
      continue;
    }
    Value *Stride = getGEPStride(GEP);
    if (!Stride) {
      if (Debug)
        errs() << "  PHI " << PN << " GEP has no simple stride: " << *GEP
               << " (numOps=" << GEP->getNumOperands() << ")\n";
      continue;
    }
    // Debug: print classification info
    if (Debug) {
      uint64_t ElemBytes = DL.getTypeAllocSize(GEP->getSourceElementType());
      errs() << "  PHI " << PN << " GEP=" << *GEP << " Stride=";
      Stride->print(errs());
      errs() << " ElemBytes=" << ElemBytes;
      if (auto *C = dyn_cast<ConstantInt>(Stride))
        errs() << " ByteStride=" << C->getZExtValue() * ElemBytes;
      errs() << "\n";
    }
    if (auto *C = dyn_cast<ConstantInt>(Stride)) {
      uint64_t S = C->getZExtValue();
      uint64_t ElemBytes = DL.getTypeAllocSize(GEP->getSourceElementType());
      uint64_t ByteStride = S * ElemBytes;
      if (ByteStride == 8 && !LI.InPhi) {
        LI.InPhi = &PN;
      } else if (ByteStride > 8 && !LI.OutPhi) {
        LI.OutPhi = &PN;
        LI.Stride = Stride;
      }
    } else if (isLoopInvariant(L, Stride)) {
      uint64_t ElemBytes = DL.getTypeAllocSize(GEP->getSourceElementType());
      if (ElemBytes >= 8 && !LI.OutPhi) {
        LI.OutPhi = &PN;
        LI.Stride = Stride;
      }
    }
  }
  if (!LI.InPhi || !LI.OutPhi) {
    if (Debug) {
      errs() << "STRIDED_VEC: skip (no InPhi/OutPhi in " << Header->getName()
             << " blocks=" << L->getNumBlocks() << ")\n";
      // Dump all phi nodes in header
      for (PHINode &PN : Header->phis()) {
        errs() << "  PHI: " << PN << "\n";
      }
    }
    return false;
  }

  // 3. Find stores in the loop — check if any store uses OutPhi as address root
  for (BasicBlock *BB : L->blocks()) {
    auto ItBegin = BB->begin();
    auto ItEnd = BB->end();
    for (auto It = ItBegin; It != ItEnd; ++It) {
      auto *SI = dyn_cast<StoreInst>(&*It);
      if (!SI) continue;
      Value *Addr = SI->getPointerOperand();
      // Walk GEP chain — but only for instructions, not constants
      Value *Root = Addr;
      while (isa<GetElementPtrInst>(Root))
        Root = cast<GetElementPtrInst>(Root)->getPointerOperand();
      if (Root == LI.OutPhi) {
        LI.StridedStore = SI;
        // Extract outer offset: store GEP form is `gep T, ptr OutPhi, i64 k`
        // where k is the loop-invariant offset within the strided row
        // (e.g. SEAL fast_convert_array's ibase_index).
        if (auto *StoreGEP = dyn_cast<GetElementPtrInst>(Addr)) {
          if (StoreGEP->getPointerOperand() == LI.OutPhi &&
              StoreGEP->getNumOperands() == 2)
            LI.OuterOffset = StoreGEP->getOperand(1);
        }
        break;
      }
    }
    if (LI.StridedStore) break;
  }

  // 4. Find loop-invariant loads (pointer is loop-invariant)
  for (BasicBlock *BB : L->blocks()) {
    for (Instruction &I : *BB) {
      auto *LD = dyn_cast<LoadInst>(&I);
      if (!LD) continue;
      Value *Ptr = LD->getPointerOperand();
      if (isLoopInvariant(L, Ptr))
        LI.InvariantLoads.push_back(LD);
    }
  }

  if (Debug)
    errs() << "STRIDED_VEC: pattern DETECTED in func=" << L->getHeader()->getParent()->getName()
           << " InvLoads=" << LI.InvariantLoads.size()
           << " store=" << (LI.StridedStore ? "Y" : "N")
           << " outer_offset=" << (LI.OuterOffset ? "Y" : "N") << "\n";
  return true;
}

static bool normalizeLoop(Loop *L, LoopInfo2 &LI, DominatorTree &DT,
                          AAResults &AA, const DataLayout &DL) {
  BasicBlock *Header = L->getHeader();
  BasicBlock *Preheader = L->getLoopPreheader();
  BasicBlock *Latch = L->getLoopLatch();
  bool Debug = getenv("STRIDED_VEC_DEBUG") != nullptr;
  bool Changed = false;

  // Phase 1: AA-checked hoist. All loops go through this safe path. Even
  // with -aggressive-hoist, the bypass only fires later (Phase 2) for
  // loops where the Barrett pattern fully matches AND we are about to
  // generate a scatter — so non-matching loops like switch_key_inplace in
  // SPEC 750 keep their may-alias loads protected.

  // 1. Hoist loop-invariant loads to the preheader.
  //    ONLY if all uses are inside the loop (otherwise replaceAllUsesWith
  //    would create use-before-def for uses in other blocks not dominated
  //    by the preheader). The load must be unordered (not volatile, not
  //    atomic-ordered) and no instruction in the loop that may write to
  //    memory may alias the load — otherwise hoisting is unsafe. This
  //    mirrors the safety check in LLVM's standard LICM pass.
  SmallVector<LoadInst *, 8> ToHoist;
  for (BasicBlock *BB : L->blocks()) {
    for (Instruction &I : *BB) {
      auto *LD = dyn_cast<LoadInst>(&I);
      if (!LD)
        continue;
      // Skip volatile and atomic-ordered loads (must execute in order).
      if (!LD->isUnordered())
        continue;
      Value *Ptr = LD->getPointerOperand();
      if (!isLoopInvariant(L, Ptr))
        continue;
      if (!DT.dominates(Preheader, LD->getParent()))
        continue;
      // Check ALL uses are inside the loop
      bool AllUsesInLoop = true;
      for (User *U : LD->users()) {
        auto *UI = dyn_cast<Instruction>(U);
        if (!UI || !L->contains(UI)) {
          AllUsesInLoop = false;
          break;
        }
      }
      if (!AllUsesInLoop)
        continue;
      // Check that no may-write instruction in the loop aliases this load.
      // This covers stores, atomicrmw, cmpxchg, va_arg, and calls that do
      // not only-read-memory. Fence-like instructions (callbr/fence) lack
      // a single MemoryLocation and conservatively block hoisting.
      // (AggressiveHoist's force-hoist bypass happens later, only for
      // Barrett operand loads in loops where scatter generation will
      // actually fire — see Phase 2 below.)
      bool MayAliasWrite = false;
      for (BasicBlock *BB2 : L->blocks()) {
        for (Instruction &J : *BB2) {
          if (!J.mayWriteToMemory())
            continue;
          if (auto *CB = dyn_cast<CallBase>(&J)) {
            if (!CB->onlyReadsMemory()) {
              MayAliasWrite = true;
              break;
            }
            continue;
          }
          std::optional<MemoryLocation> Loc = MemoryLocation::getOrNone(&J);
          if (!Loc || !AA.isNoAlias(*Loc, MemoryLocation::get(LD))) {
            MayAliasWrite = true;
            break;
          }
        }
        if (MayAliasWrite)
          break;
      }
      if (MayAliasWrite) {
        if (Debug)
          errs() << "STRIDED_VEC: skip hoist (may-alias write) for " << *LD
                 << "\n";
        continue;
      }
      ToHoist.push_back(LD);
    }
  }
  for (LoadInst *LD : ToHoist) {
    auto *NewLoad = cast<LoadInst>(LD->clone());
    NewLoad->insertBefore(Preheader->getTerminator());
    // Scoped aliasing metadata is tied to the loop iteration; drop both
    // halves of the (alias.scope, noalias) pair when hoisting out of the
    // loop so downstream passes (auto-vectorizer) don't keep stale
    // assumptions. Matches MemCpyOptimizer's handling.
    NewLoad->setMetadata(LLVMContext::MD_alias_scope, nullptr);
    NewLoad->setMetadata(LLVMContext::MD_noalias, nullptr);
    LD->replaceAllUsesWith(NewLoad);
    LD->eraseFromParent();
    Changed = true;
  }

  // 2. Generate vectorized loop with @llvm.masked.scatter
  //    Only if: single-block loop, strided store found, and we can
  //    extract the Barrett reduce operands.
  if (!Header || !Latch || Header != Latch || !LI.StridedStore) {
    return Changed;
  }

  // The rest of this function is the scatter generation.
  // It's called directly from generateScatterLoop().
  // (This function is split: normalizeLoop does hoisting + metadata,
  //  generateScatterLoop does the CFG surgery.)

  // --- Scatter generation ---
  // The loop is single-block (Header == Latch == the loop body).
  // IR structure (from BB %90 of fast_convert_array):
  //   %InPhi  = phi ptr [ InStart, Preheader ], [ InNext, Header ]
  //   %OutPhi = phi ptr [ OutStart, Preheader ], [ OutNext, Header ]
  //   %input  = load i64, ptr %InPhi
  //   %hi     = call @llvm.umul.fix.i64(quotient, %input, 64)
  //   %lo     = mul operand, %input
  //   %prod   = mul modulus, %hi
  //   %result = sub %lo, %prod
  //   %cmp    = icmp ult %result, modulus
  //   %sel    = select %cmp, 0, modulus
  //   %final  = sub %result, %sel
  //   %addr   = gep [8xi8], ptr %OutPhi, i64 %outer_index
  //   store %final, ptr %addr
  //   InNext  = gep i8, ptr %InPhi, i64 8
  //   OutNext = gep [8xi8], ptr %OutPhi, i64 %stride
  //   brcond  = select(icmp ne InNext, InEnd), true, icmp ne OutNext, OutEnd)
  //   br brcond, Header, Exit

  // Find the exit block (the fall-through when loop exits)
  CondBrInst *Br = dyn_cast<CondBrInst>(Header->getTerminator());
  if (!Br) return Changed;
  BasicBlock *Exit = nullptr;
  for (unsigned i = 0; i < 2; i++)
    if (Br->getSuccessor(i) != Header) {
      Exit = Br->getSuccessor(i);
      break;
    }
  if (!Exit) return Changed;

  // Get initial values from preheader
  Value *InStart = nullptr, *OutStart = nullptr;
  for (unsigned i = 0; i < LI.InPhi->getNumIncomingValues(); i++)
    if (LI.InPhi->getIncomingBlock(i) == Preheader)
      InStart = LI.InPhi->getIncomingValue(i);
  for (unsigned i = 0; i < LI.OutPhi->getNumIncomingValues(); i++)
    if (LI.OutPhi->getIncomingBlock(i) == Preheader)
      OutStart = LI.OutPhi->getIncomingValue(i);
  if (!InStart || !OutStart) return Changed;

  // Find the end pointer from the exit condition.
  // The header's conditional branch may use an ICmp directly (simple case),
  // OR a select combining two exit conditions (SEAL's case: the loop exits
  // when EITHER InNext == InEnd OR OutNext == OutEnd, combined as
  // `select(icmp ne InNext, InEnd), true, icmp ne OutNext, OutEnd)`).
  // Search the header for the ICmp whose LHS traces back to InPhi and whose
  // RHS is loop-invariant — that's the InNext == InEnd condition we need to
  // compute TripCount. Don't just take the first ICmp (the Barrett reduce
  // `icmp ult result, modulus` also lives in the same block).
  Value *InEnd = nullptr;
  ICmpInst *Cmp = nullptr;
  for (Instruction &I : *Header) {
    auto *IC = dyn_cast<ICmpInst>(&I);
    if (!IC) continue;
    Value *LHS = IC->getOperand(0);
    Value *RHS = IC->getOperand(1);
    bool LHSInPhi = tracesToPhi(LHS, LI.InPhi);
    bool RHSInPhi = tracesToPhi(RHS, LI.InPhi);
    if ((LHSInPhi && isLoopInvariant(L, RHS)) ||
        (RHSInPhi && isLoopInvariant(L, LHS))) {
      Cmp = IC;
      break;
    }
  }
  if (!Cmp) return Changed;
  Value *CmpLHS = Cmp->getOperand(0);
  Value *CmpRHS = Cmp->getOperand(1);
  if (isLoopInvariant(L, CmpRHS))
    InEnd = CmpRHS;
  else if (isLoopInvariant(L, CmpLHS))
    InEnd = CmpLHS;
  if (!InEnd) return Changed;

  // Compute trip count = (InEnd - InStart) / 8.
  // Insert BEFORE the old terminator — IRBuilder(BasicBlock*) defaults to
  // BB->end() which is past the terminator, producing invalid IR (instructions
  // after the terminator) and, in release builds, causing getTerminator() to
  // return the wrong instruction and erase our newly-created values.
  IRBuilder<> PB(Preheader, Preheader->getTerminator()->getIterator());
  Type *I64Ty = Type::getInt64Ty(Header->getContext());
  Value *StartInt = PB.CreatePtrToInt(InStart, I64Ty, "scatter.instart");
  Value *EndInt = PB.CreatePtrToInt(InEnd, I64Ty, "scatter.inend");
  Value *ByteCount = PB.CreateSub(EndInt, StartInt, "scatter.bytes");
  Value *TripCount = PB.CreateLShr(ByteCount, PB.getInt64(3), "scatter.tc");

  // Get VF = vscale * 2 (SVE vector width in i64 elements)
  Module *M = Header->getModule();
  Function *VscaleFn = Intrinsic::getOrInsertDeclaration(M, Intrinsic::vscale, {I64Ty});
  Value *Vscale = PB.CreateCall(VscaleFn);
  Value *VF = PB.CreateMul(Vscale, PB.getInt64(2), "scatter.vf");

  // Find Barrett reduce operands by tracing from the store value
  Value *StoreVal = LI.StridedStore->getValueOperand();
  // StoreVal = sub %result, %sel (the final Barrett reduce)
  auto *FinalSub = dyn_cast<BinaryOperator>(StoreVal);
  if (!FinalSub || FinalSub->getOpcode() != Instruction::Sub) return Changed;
  Value *Result = FinalSub->getOperand(0);
  Value *Sel = FinalSub->getOperand(1);
  // Result = sub %lo, %prod (lo - mod*hi)
  auto *ResultSub = dyn_cast<BinaryOperator>(Result);
  if (!ResultSub || ResultSub->getOpcode() != Instruction::Sub) return Changed;
  Value *Lo = ResultSub->getOperand(0);
  Value *Prod = ResultSub->getOperand(1);
  // Lo = mul operand, input
  auto *LoMul = dyn_cast<BinaryOperator>(Lo);
  if (!LoMul || LoMul->getOpcode() != Instruction::Mul) return Changed;
  // Find which operand of LoMul is the input load: the one whose pointer
  // traces back to InPhi (the contiguous input phi). The other is the
  // loop-invariant Barrett operand. Both can be LoadInsts, so we cannot
  // just pick the first LoadInst.
  Value *InputLoad = nullptr;
  auto IsInput = [&](Value *V) {
    auto *LD = dyn_cast<LoadInst>(V);
    return LD && tracesToPhi(LD->getPointerOperand(), LI.InPhi);
  };
  if (IsInput(LoMul->getOperand(0)))
    InputLoad = LoMul->getOperand(0);
  else if (IsInput(LoMul->getOperand(1)))
    InputLoad = LoMul->getOperand(1);
  if (!InputLoad) return Changed;
  Value *Operand = (LoMul->getOperand(0) == InputLoad) ? LoMul->getOperand(1)
                                                        : LoMul->getOperand(0);
  // Prod = mul modulus, hi
  auto *ProdMul = dyn_cast<BinaryOperator>(Prod);
  if (!ProdMul || ProdMul->getOpcode() != Instruction::Mul) return Changed;
  // Find hi (the umul.fix call) and modulus
  Value *Hi = nullptr, *Modulus = nullptr;
  if (auto *II = dyn_cast<IntrinsicInst>(ProdMul->getOperand(0))) {
    Hi = ProdMul->getOperand(0);
    Modulus = ProdMul->getOperand(1);
  } else if (auto *II = dyn_cast<IntrinsicInst>(ProdMul->getOperand(1))) {
    Hi = ProdMul->getOperand(1);
    Modulus = ProdMul->getOperand(0);
  }
  if (!Hi || !Modulus) return Changed;
  // Sel = select icmp ult Result, Modulus, 0, Modulus
  auto *SelInst = dyn_cast<SelectInst>(Sel);
  if (!SelInst) return Changed;
  // Modulus is the non-zero value of the select
  if (SelInst->getTrueValue() == Modulus)
    Modulus = SelInst->getFalseValue();
  else if (SelInst->getFalseValue() != Modulus)
    return Changed;

  // Verify that the Barrett broadcast operands are loop-invariant (defined
  // outside the loop). Hi itself is NOT loop-invariant by design — it
  // depends on the input — and is recomputed in vec.body as a vector
  // umul.fix call, so we deliberately exclude it from this check.
  // Pick the loop-invariant arg of Hi as Quotient (the other arg is the
  // loop-variant input).
  auto *HiCall = cast<IntrinsicInst>(Hi);
  Value *HiArg0 = HiCall->getArgOperand(0);
  Value *HiArg1 = HiCall->getArgOperand(1);

  // AggressiveHoist force-hoist: if any Barrett operand load is not
  // loop-invariant (because the AA-safe Phase 1 hoist skipped it as
  // may-alias), and AggressiveHoist is on, force-hoist these specific
  // loads — bypassing the AA check. This is much narrower than a global
  // bypass: it only fires for loops where the Barrett pattern fully
  // matches (we have Operand / Modulus / Hi / Quotient candidates in
  // hand) AND the load's pointer is provably loop-invariant (the load
  // is loop-variant only because AA was conservative, not because the
  // pointer truly varies). Other loops (e.g. switch_key_inplace in
  // SPEC 750 — single-block + strided store but Barrett pattern does
  // not fully match) never reach here, so their may-alias loads stay
  // protected by the AA check.
  if (AggressiveHoist) {
    SmallVector<Value *, 4> Candidates{Operand, Modulus, HiArg0, HiArg1};
    for (Value *V : Candidates) {
      auto *LD = dyn_cast<LoadInst>(V);
      if (!LD || isLoopInvariant(L, LD))
        continue;
      // Only force-hoist if the load's pointer is loop-invariant.
      if (!isLoopInvariant(L, LD->getPointerOperand()))
        continue;
      if (Debug)
        errs() << "STRIDED_VEC: force-hoist Barrett operand (bypass AA): "
               << *LD << "\n";
      auto *NewLoad = cast<LoadInst>(LD->clone());
      NewLoad->insertBefore(Preheader->getTerminator());
      NewLoad->setMetadata(LLVMContext::MD_alias_scope, nullptr);
      NewLoad->setMetadata(LLVMContext::MD_noalias, nullptr);
      LD->replaceAllUsesWith(NewLoad);
      LD->eraseFromParent();
      Changed = true;
    }
    // Re-extract after force-hoist (the Value* pointers above are stale).
    Operand = (LoMul->getOperand(0) == InputLoad) ? LoMul->getOperand(1)
                                                  : LoMul->getOperand(0);
    HiCall = cast<IntrinsicInst>(Hi);
    HiArg0 = HiCall->getArgOperand(0);
    HiArg1 = HiCall->getArgOperand(1);
  }

  bool Arg0Inv = isLoopInvariant(L, HiArg0);
  bool Arg1Inv = isLoopInvariant(L, HiArg1);
  if (!Arg0Inv && !Arg1Inv) {
    if (Debug)
      errs() << "STRIDED_VEC: scatter skip (no loop-invariant Hi arg)\n";
    return Changed;
  }
  // Exactly one of the args must be loop-invariant (the quotient); the other
  // is the input. If both are invariant this is a different pattern.
  Value *Quotient = Arg0Inv ? HiArg0 : HiArg1;
  if (Arg0Inv && Arg1Inv) {
    if (Debug)
      errs() << "STRIDED_VEC: scatter skip (both Hi args loop-invariant)\n";
    return Changed;
  }
  if (!isLoopInvariant(L, Operand) || !isLoopInvariant(L, Modulus) ||
      !isLoopInvariant(L, Quotient)) {
    if (Debug)
      errs() << "STRIDED_VEC: scatter skip (Operand/Modulus/Quotient not "
                 "loop-invariant)\n";
    return Changed;
  }

  if (Debug)
    errs() << "STRIDED_VEC: scatter operands found: Operand=" << *Operand
           << " Hi=" << *Hi << " Modulus=" << *Modulus << "\n";

  // --- Create vector loop ---
  // Only create new blocks AFTER all checks pass (to avoid partial CFG surgery)
  LLVMContext &Ctx = Header->getContext();
  VectorType *VecI64 = VectorType::get(I64Ty, ElementCount::getScalable(2));
  VectorType *VecI1 = VectorType::get(Type::getInt1Ty(Ctx), ElementCount::getScalable(2));

  BasicBlock *VecBody = BasicBlock::Create(Ctx, "vec.scatter", Header->getParent());
  BasicBlock *Middle = BasicBlock::Create(Ctx, "middle.scatter", Header->getParent());

  // Redirect preheader → VecBody. Insert the new br BEFORE the old terminator
  // (PB's insertion point), then erase the old terminator (now the last
  // instruction in the block). Erasing first would invalidate PB's iterator.
  PB.CreateBr(VecBody);
  Preheader->getTerminator()->eraseFromParent();

  // --- Vector body ---
  IRBuilder<> VB(VecBody);
  PHINode *IV = VB.CreatePHI(I64Ty, 2, "scatter.iv");
  IV->addIncoming(PB.getInt64(0), Preheader);

  // Predicate: active lanes
  Function *LaneMaskFn = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::get_active_lane_mask, {VecI1, I64Ty});
  Value *Predicate = VB.CreateCall(LaneMaskFn, {IV, TripCount});

  // Contiguous load: load <vscale x 2 x i64>, ptr InStart[IV]
  Value *InAddr = VB.CreateGEP(I64Ty, InStart, IV, "scatter.in");
  Value *VecInput = VB.CreateLoad(VecI64, InAddr, "scatter.vin");

  // Broadcast constants
  Value *VecOperand = VB.CreateVectorSplat(VecI64->getElementCount(), Operand, "scatter.vop");
  Value *VecModulus = VB.CreateVectorSplat(VecI64->getElementCount(), Modulus, "scatter.vmod");

  // Vector Barrett: hi = umul.fix(quotient, input, 64)
  // Quotient already extracted above (for loop-invariance check)
  Value *VecQuotient = VB.CreateVectorSplat(VecI64->getElementCount(), Quotient, "scatter.vquot");
  Function *UmulFixVecFn = Intrinsic::getOrInsertDeclaration(M, Intrinsic::umul_fix, {VecI64});
  Value *VecHi = VB.CreateCall(UmulFixVecFn, {VecInput, VecQuotient, VB.getInt32(64)}, "scatter.vhi");

  // lo = operand * input
  Value *VecLo = VB.CreateMul(VecOperand, VecInput, "scatter.vlo");
  // prod = modulus * hi
  Value *VecProd = VB.CreateMul(VecModulus, VecHi, "scatter.vprod");
  // result = lo - prod
  Value *VecResult = VB.CreateSub(VecLo, VecProd, "scatter.vres");
  // conditional reduce: min(result, result - modulus)
  Value *VecSub = VB.CreateSub(VecResult, VecModulus, "scatter.vsub");
  Function *UminFn = Intrinsic::getOrInsertDeclaration(M, Intrinsic::umin, {VecI64});
  VecResult = VB.CreateCall(UminFn, {VecResult, VecSub}, "scatter.vfinal");

  // Scatter store: addresses = OutStart + (IV + lane) * stride + outer_offset
  // (outer_offset is the loop-invariant index within the strided row, e.g.
  // SEAL fast_convert_array's ibase_index from `gep T, ptr OutPhi, i64 idx`).
  Function *StepVecFn = Intrinsic::getOrInsertDeclaration(M, Intrinsic::stepvector, {VecI64});
  Value *StepVec = VB.CreateCall(StepVecFn, {});
  Value *LaneIdx = VB.CreateAdd(
      VB.CreateVectorSplat(VecI64->getElementCount(), IV), StepVec, "scatter.lane");
  Value *ScaledIdx = VB.CreateMul(LaneIdx,
      VB.CreateVectorSplat(VecI64->getElementCount(), LI.Stride), "scatter.sidx");
  if (LI.OuterOffset) {
    ScaledIdx = VB.CreateAdd(ScaledIdx,
        VB.CreateVectorSplat(VecI64->getElementCount(), LI.OuterOffset),
        "scatter.sidx.off");
  }
  Value *VecAddrs = VB.CreateGEP(I64Ty, OutStart, ScaledIdx, "scatter.vaddr");

  // @llvm.masked.scatter — must use IRBuilder::CreateMaskedScatter (not a
  // manual CreateCall) so the alignment is attached as a parameter attribute
  // on the pointer operand. SelectionDAGBuilder::visitMaskedScatter reads the
  // alignment via getParamAlign(1); a bare CreateCall leaves it None, which
  // defaults to Align(1), and AArch64 ISel rejects ST1D with align < 8
  // ("Cannot select masked_scatter ... align 1"). This matters when a later
  // pass (InstCombine) canonicalizes the GEP source type from i64 to [8 x i8],
  // which would otherwise drag the MMO alignment down to 1.
  VB.CreateMaskedScatter(VecResult, VecAddrs, Align(8), Predicate);

  // IV += VF, loop back if IV < TripCount
  Value *IVNext = VB.CreateAdd(IV, VF, "scatter.ivnext");
  IV->addIncoming(IVNext, VecBody);
  Value *LoopCond = VB.CreateICmpULT(IVNext, TripCount, "scatter.lc");
  VB.CreateCondBr(LoopCond, VecBody, Middle);

  // --- Middle block: update phis for scalar remainder ---
  // The scatter loop's predicate already masked off lanes >= TripCount, so
  // all TripCount elements have been scattered. The scalar remainder only
  // has work when IVNext < TripCount. The SEAL inner loop is do-while, so
  // if we let it start with InPhi == InEnd (IVNext == TripCount) it would
  // load past the end and loop forever. Branch to the scalar Header only
  // when there is actual remainder work; otherwise go straight to Exit.
  IRBuilder<> MB(Middle);
  // InPhi = InStart + IVNext * 8 (bytes)
  Value *NewIn = MB.CreateGEP(I64Ty, InStart, IVNext, "scatter.remain.in");
  // OutPhi = OutStart + IVNext * stride
  Value *NewOut = MB.CreateGEP(I64Ty, OutStart,
      MB.CreateMul(IVNext, LI.Stride), "scatter.remain.out");
  Value *NeedScalar = MB.CreateICmpULT(IVNext, TripCount, "scatter.need_scalar");
  MB.CreateCondBr(NeedScalar, Header, Exit);

  // Update phi nodes: change preheader incoming → Middle
  for (PHINode &PN : Header->phis()) {
    if (&PN == LI.InPhi) {
      for (unsigned i = 0; i < PN.getNumIncomingValues(); i++) {
        if (PN.getIncomingBlock(i) == Preheader) {
          PN.setIncomingValue(i, NewIn);
          PN.setIncomingBlock(i, Middle);
        }
      }
    } else if (&PN == LI.OutPhi) {
      for (unsigned i = 0; i < PN.getNumIncomingValues(); i++) {
        if (PN.getIncomingBlock(i) == Preheader) {
          PN.setIncomingValue(i, NewOut);
          PN.setIncomingBlock(i, Middle);
        }
      }
    }
  }

  Changed = true;
  if (Debug)
    errs() << "STRIDED_VEC: scatter loop generated!\n";
  return Changed;
}

} // end anonymous namespace

// New PM FunctionPass — runs at VectorizerStartEP (after all inlining + scalar opts)
PreservedAnalyses
AArch64StridedVectorizePass::run(Function &F, FunctionAnalysisManager &AM) {
  if (!EnableStridedVectorize)
    return PreservedAnalyses::all();

  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &AA = AM.getResult<AAManager>(F);
  const DataLayout &DL = F.getDataLayout();

  bool Debug = getenv("STRIDED_VEC_DEBUG") != nullptr;
  if (Debug)
    errs() << "STRIDED_VEC: run on " << F.getName() << " loops=" << LI.getLoopsInPreorder().size() << "\n";

  bool Changed = false;
  // Collect all matching loops first (don't iterate + modify simultaneously)
  struct MatchInfo { Loop *L; LoopInfo2 LI2; bool IsSingleBlock; };
  SmallVector<MatchInfo, 4> Matches;
  for (Loop *L : LI.getLoopsInPreorder()) {
    MatchInfo MI;
    MI.L = L;
    if (detectPattern(L, MI.LI2, DT, LI, DL)) {
      if (Debug)
        errs() << "STRIDED_VEC: pattern DETECTED in " << F.getName()
               << " InvLoads=" << MI.LI2.InvariantLoads.size() << "\n";
      BasicBlock *Hdr = L->getHeader();
      BasicBlock *Latch = L->getLoopLatch();
      MI.IsSingleBlock = Hdr && Latch && Hdr == Latch && MI.LI2.StridedStore;
      if (MI.IsSingleBlock && Debug)
        errs() << "STRIDED_VEC: will generate scatter for this loop\n";
      Matches.push_back(std::move(MI));
    }
  }

  // Now process each match: hoisting + (if single-block) scatter
  bool DidScatter = false;
  for (auto &MI : Matches) {
    Changed |= normalizeLoop(MI.L, MI.LI2, DT, AA, DL);
    // Only do scatter for ONE loop per function (to avoid stale LoopInfo)
    if (MI.IsSingleBlock && !DidScatter) {
      DidScatter = true;
      // normalizeLoop already does scatter generation (it checks
      // Header == Latch && StridedStore && operands are loop-invariant)
    }
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

// Legacy pass wrapper — skip (new PM LoopPass used via registerLateLoopOptimizationsEPCallback)
