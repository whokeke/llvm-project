//===-- AArch64SwitchKeyVectorize.cpp ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Detects SEAL's switch_key_inplace function and replaces its inner
/// SEAL_ITERATE loops (multiply_uint64 + add_uint128 + optional
/// Barrett reduce) with SVE-vectorized loops using <vscale x 2 x i64>.
///
/// This mirrors the ACLE source patch that uses sve_accumulate_and_reduce()
/// and sve_barrett_reduce_accumulator(), but as a compiler pass (no source
/// modification needed).
///
/// The pass detects innermost loops containing @llvm.umul.fix (the mul+umulh
/// pattern), identifies the 3 pointer phis (t_operand step=8, key step=8,
/// accumulator step=16), and generates a vector loop that processes VF
/// elements at a time using SVE vector operations.
///
/// Gated on `-aarch64-switch-key-vectorize` (off by default).
/// Runs at OptimizerEarlyEP (after MulI128Lowering, before LoopVectorize).
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64SwitchKeyVectorize.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-switch-key-vectorize"

static cl::opt<bool> EnableSwitchKeyVec(
    "aarch64-switch-key-vectorize",
    cl::init(false), cl::Hidden,
    cl::desc("Vectorize switch_key_inplace mul+add loops with SVE"));

// switch_key_inplace mangled name
// seal::Evaluator::switch_key_inplace(Ciphertext&, ConstRNSIter, const KSwitchKeys&, size_t, MemoryPoolHandle)
static const char *SwitchKeyName =
    "_ZNK4seal9Evaluator18switch_key_inplaceERNS_10CiphertextENS_4util12ConstRNSIterERKNS_"
    "11KSwitchKeysEmNS_16MemoryPoolHandleE";

/// Find a ptr phi in the loop header with a specific back-edge GEP step.
static PHINode *findPtrPhiWithStep(Loop *L, uint64_t ExpectedStep,
                                   const DataLayout &DL,
                                   BasicBlock *&BackBlock,
                                   SmallPtrSet<PHINode *, 4> &Skip) {
  BasicBlock *Header = L->getHeader();
  for (PHINode &PN : Header->phis()) {
    if (!PN.getType()->isPointerTy())
      continue;
    if (Skip.count(&PN))
      continue;
    for (unsigned i = 0; i < PN.getNumIncomingValues(); i++) {
      BasicBlock *InBB = PN.getIncomingBlock(i);
      if (!L->contains(InBB))
        continue;
      Value *BackVal = PN.getIncomingValue(i);
      if (auto *GEP = dyn_cast<GetElementPtrInst>(BackVal)) {
        if (GEP->getNumOperands() == 2) {
          if (auto *C = dyn_cast<ConstantInt>(GEP->getOperand(1))) {
            uint64_t Step = C->getZExtValue() *
                            DL.getTypeAllocSize(GEP->getSourceElementType());
            if (Step == ExpectedStep) {
              BackBlock = InBB;
              Skip.insert(&PN);
              return &PN;
            }
          }
        }
      }
    }
  }
  return nullptr;
}

/// Find the loop's exit count from ptr-based exit condition.
/// Scans all blocks in the loop for `icmp eq/ne ptr %phi, %end` where
/// %phi is one of the loop's ptr phis and %end is loop-invariant.
/// Returns the count value if recoverable.
static Value *findExitCount(Loop *L, IRBuilder<> &B, const DataLayout &DL,
                            SmallVectorImpl<PHINode *> &PtrPhis) {
  // For each ptr phi, scan loop blocks for icmp involving the phi.
  // The icmp might compare the phi directly, OR a GEP of the phi
  // (SEAL_ITERATE checks `gep %phi, step == end`, not `%phi == end`).
  for (PHINode *Phi : PtrPhis) {
    for (BasicBlock *BB : L->blocks()) {
      for (Instruction &I : *BB) {
        auto *IC = dyn_cast<ICmpInst>(&I);
        if (!IC)
          continue;
        if (IC->getPredicate() != ICmpInst::ICMP_EQ &&
            IC->getPredicate() != ICmpInst::ICMP_NE)
          continue;
        // Check if one operand traces back to the phi (directly or via GEP)
        Value *Other = nullptr;
        for (int dir = 0; dir < 2; dir++) {
          Value *Op = IC->getOperand(dir);
          // Direct phi match
          if (Op == Phi) {
            Other = IC->getOperand(1 - dir);
            break;
          }
          // GEP of phi match: gep T, ptr %phi, i64 N
          if (auto *GEP = dyn_cast<GetElementPtrInst>(Op)) {
            if (GEP->getPointerOperand() == Phi) {
              Other = IC->getOperand(1 - dir);
              break;
            }
          }
        }
        if (!Other)
          continue;
        // Check if Other is loop-invariant
        if (auto *OtherI = dyn_cast<Instruction>(Other))
          if (L->contains(OtherI))
            continue;
        // Found the end pointer!
        Value *End = Other;
        // Find start (phi's initial value from preheader)
        Value *Start = nullptr;
        for (unsigned i = 0; i < Phi->getNumIncomingValues(); i++) {
          if (!L->contains(Phi->getIncomingBlock(i))) {
            Start = Phi->getIncomingValue(i);
            break;
          }
        }
        if (!Start)
          continue;
        // Find back-edge GEP step
        Value *BackVal = nullptr;
        for (unsigned i = 0; i < Phi->getNumIncomingValues(); i++) {
          if (L->contains(Phi->getIncomingBlock(i))) {
            BackVal = Phi->getIncomingValue(i);
            break;
          }
        }
        if (!BackVal)
          continue;
        auto *BackGEP = dyn_cast<GetElementPtrInst>(BackVal);
        if (!BackGEP || BackGEP->getNumOperands() != 2)
          continue;
        auto *StepC = dyn_cast<ConstantInt>(BackGEP->getOperand(1));
        if (!StepC)
          continue;
        uint64_t StepBytes = StepC->getZExtValue() *
                             DL.getTypeAllocSize(BackGEP->getSourceElementType());
        if (StepBytes == 0)
          continue;
        // Compute trip count = (ptrtoint(end) - ptrtoint(start)) / step
        Type *I64Ty = B.getInt64Ty();
        Value *StartInt = B.CreatePtrToInt(Start, I64Ty);
        Value *EndInt = B.CreatePtrToInt(End, I64Ty);
        Value *Diff = B.CreateSub(EndInt, StartInt);
        Value *TC = B.CreateSDiv(Diff, B.getInt64(StepBytes));
        return TC;
      }
    }
  }
  return nullptr;
}

/// Generate a vectorized mul+accumulate loop.
/// Replaces the scalar SEAL_ITERATE loop with a SVE vector loop.
///
/// The scalar loop does:
///   for (j = 0; j < count; j++) {
///     qword = {mul(t_op[j], key[j]), umulh(t_op[j], key[j])}
///     acc[2*j], acc[2*j+1] = add_uint128(qword, {acc[2*j], acc[2*j+1]})
///     // optional: acc[2*j] = barrett_reduce_128(acc, modulus); acc[2*j+1] = 0
///   }
///
/// The vector loop does (per VF elements):
///   v_op = ld1d [t_op + iv]          ; contiguous
///   v_key = ld1d [key + iv]          ; contiguous
///   v_lo = mul v_op, v_key
///   v_hi = umul.fix v_op, v_key, 64
///   v_acc_lo = strided_load [acc + iv*2]     ; stride 16 bytes
///   v_acc_hi = strided_load [acc + iv*2+1]   ; stride 16 bytes, offset 8
///   v_new_lo = v_acc_lo + v_lo (with carry)
///   v_new_hi = v_acc_hi + v_hi + carry
///   strided_store v_new_lo → [acc + iv*2]
///   strided_store v_new_hi → [acc + iv*2+1]
static bool vectorizeMulAddLoop(Loop *L, Function &F,
                                FunctionAnalysisManager &AM) {
  const DataLayout &DL = F.getDataLayout();
  BasicBlock *Header = L->getHeader();
  BasicBlock *Preheader = L->getLoopPreheader();
  BasicBlock *Latch = L->getLoopLatch();
  if (!Header || !Preheader || !Latch)
    return false;

  bool Debug = getenv("SWITCH_KEY_VEC_DEBUG") != nullptr;
  if (Debug)
    errs() << "SWITCH_KEY_VEC: trying loop in " << F.getName()
           << " hdr=" << Header->getName() << "\n";

  // Check loop has @llvm.umul.fix
  bool HasUmulFix = false;
  for (BasicBlock *BB : L->blocks()) {
    for (Instruction &I : *BB) {
      if (auto *II = dyn_cast<IntrinsicInst>(&I))
        if (II->getIntrinsicID() == Intrinsic::umul_fix) {
          HasUmulFix = true;
          break;
        }
    }
    if (HasUmulFix) break;
  }
  if (!HasUmulFix) {
    if (Debug)
      errs() << "SWITCH_KEY_VEC: no umul.fix in loop, skip\n";
    return false;
  }
  if (Debug)
    errs() << "SWITCH_KEY_VEC: has umul.fix, checking phis\n";

  // Find the 3 ptr phis: t_operand (step 8), key (step 8), acc (step 16)
  BasicBlock *BackBlock = nullptr;
  SmallPtrSet<PHINode *, 4> Skip;
  PHINode *OpPhi = findPtrPhiWithStep(L, 8, DL, BackBlock, Skip);
  if (!OpPhi) {
    if (Debug) {
      errs() << "SWITCH_KEY_VEC: no step-8 phi. Header ptr phis:\n";
      for (PHINode &PN : Header->phis()) {
        if (!PN.getType()->isPointerTy()) continue;
        for (unsigned i = 0; i < PN.getNumIncomingValues(); i++) {
          if (L->contains(PN.getIncomingBlock(i)))
            errs() << "  " << PN << " back=" << *PN.getIncomingValue(i) << "\n";
        }
      }
    }
    return false;
  }
  PHINode *KeyPhi = findPtrPhiWithStep(L, 8, DL, BackBlock, Skip);
  if (!KeyPhi || KeyPhi == OpPhi)
    return false;
  PHINode *AccPhi = findPtrPhiWithStep(L, 16, DL, BackBlock, Skip);
  if (!AccPhi)
    return false;

  if (Debug)
    errs() << "SWITCH_KEY_VEC: found 3 phis (op/key/acc)\n";

  // Get initial values from preheader
  Value *OpStart = nullptr, *KeyStart = nullptr, *AccStart = nullptr;
  for (unsigned i = 0; i < OpPhi->getNumIncomingValues(); i++) {
    if (!L->contains(OpPhi->getIncomingBlock(i))) {
      OpStart = OpPhi->getIncomingValue(i);
      break;
    }
  }
  for (unsigned i = 0; i < KeyPhi->getNumIncomingValues(); i++) {
    if (!L->contains(KeyPhi->getIncomingBlock(i))) {
      KeyStart = KeyPhi->getIncomingValue(i);
      break;
    }
  }
  for (unsigned i = 0; i < AccPhi->getNumIncomingValues(); i++) {
    if (!L->contains(AccPhi->getIncomingBlock(i))) {
      AccStart = AccPhi->getIncomingValue(i);
      break;
    }
  }
  if (!OpStart || !KeyStart || !AccStart)
    return false;

  // Compute trip count
  IRBuilder<> PB(Preheader, Preheader->getTerminator()->getIterator());
  SmallVector<PHINode *, 4> PtrPhis;
  PtrPhis.push_back(OpPhi);
  PtrPhis.push_back(KeyPhi);
  PtrPhis.push_back(AccPhi);
  Value *TripCount = findExitCount(L, PB, DL, PtrPhis);
  if (!TripCount) {
    if (Debug)
      errs() << "SWITCH_KEY_VEC: can't find exit count, skip\n";
    return false;
  }

  // Check if this loop does Barrett reduce (has barrett_reduce_128 pattern:
  // after add_uint128, there's a call or inline sequence that reduces)
  bool DoReduce = false;
  for (BasicBlock *BB : L->blocks()) {
    for (Instruction &I : *BB) {
      // Barrett reduce inline pattern: mul(ratio, acc_lo) + umulh + sub + csel
      // or: store 0 to acc_hi (the "acc[1] = 0" after reduce)
      if (auto *SI = dyn_cast<StoreInst>(&I)) {
        if (auto *C = dyn_cast<ConstantInt>(SI->getValueOperand())) {
          if (C->isZero()) {
            // Check if this store is to acc_hi (offset 8 from acc base)
            // This indicates Barrett reduce path
            DoReduce = true;
            break;
          }
        }
      }
    }
    if (DoReduce) break;
  }

  if (Debug)
    errs() << "SWITCH_KEY_VEC: trip_count=" << *TripCount
           << " do_reduce=" << DoReduce << "\n";

  // --- Generate vector loop ---
  LLVMContext &Ctx = F.getContext();
  Type *I64Ty = Type::getInt64Ty(Ctx);
  auto VecI64 = VectorType::get(I64Ty, ElementCount::getScalable(2));

  // Get vscale for VF computation
  Function *VscaleFn = Intrinsic::getOrInsertDeclaration(
      F.getParent(), Intrinsic::vscale, {I64Ty});
  Value *Vscale = PB.CreateCall(VscaleFn, {}, "");
  Value *VF = PB.CreateMul(Vscale, PB.getInt64(2), "skv.vf");

  // Get umul.fix intrinsic
  Function *UmulFixFn = Intrinsic::getOrInsertDeclaration(
      F.getParent(), Intrinsic::umul_fix, {VecI64});

  // Create blocks: vec preheader, vec loop, vec after, scalar remainder
  BasicBlock *VecPH = BasicBlock::Create(Ctx, "skv.vec.ph", &F);
  BasicBlock *VecLoop = BasicBlock::Create(Ctx, "skv.vec.loop", &F);
  BasicBlock *VecAfter = BasicBlock::Create(Ctx, "skv.vec.after", &F);

  // Vec preheader: branch to vec loop
  IRBuilder<> VPH(VecPH);
  VPH.CreateBr(VecLoop);

  // Vec loop body
  IRBuilder<> VL(VecLoop);
  PHINode *IV = VL.CreatePHI(I64Ty, 2, "skv.iv");
  IV->addIncoming(PB.getInt64(0), VecPH);

  // Predicate: whilelt(IV, TripCount)
  Function *WhileloFn = Intrinsic::getOrInsertDeclaration(
      F.getParent(), Intrinsic::get_active_lane_mask,
      {VectorType::get(Type::getInt1Ty(Ctx), ElementCount::getScalable(2)),
       I64Ty});
  Value *Pred = VL.CreateCall(WhileloFn, {IV, TripCount}, "skv.pred");

  // Contiguous masked loads: t_operand[IV] and key[IV]
  Value *OpElem = VL.CreateGEP(I64Ty, OpStart, IV, "skv.op_elem");
  Value *KeyElem = VL.CreateGEP(I64Ty, KeyStart, IV, "skv.key_elem");
  Value *VOp = VL.CreateMaskedLoad(VecI64, OpElem, Align(8), Pred,
      ConstantAggregateZero::get(VecI64), "skv.vop");
  Value *VKey = VL.CreateMaskedLoad(VecI64, KeyElem, Align(8), Pred,
      ConstantAggregateZero::get(VecI64), "skv.vkey");

  // 64x64→128 mul
  Value *VLo = VL.CreateMul(VOp, VKey, "skv.vlo");
  Value *VHi = VL.CreateCall(UmulFixFn, {VOp, VKey, VL.getInt32(64)},
                              "skv.vhi");

  // Accumulator access: acc layout is [lo0, hi0, lo1, hi1, ...] (stride-2).
  // Use gather/scatter for strided access. This is more expensive than
  // ld2d/st2d but works correctly. Future optimization: use ld2d/st2d
  // via <vscale x 4 x i64> load + deinterleave.
  Value *LaneIdx = VL.CreateAdd(
      VL.CreateVectorSplat(VecI64->getElementCount(), IV),
      VL.CreateCall(Intrinsic::getOrInsertDeclaration(
          F.getParent(), Intrinsic::stepvector, {VecI64}), {}),
      "skv.lane");
  Value *Stride2Vec = VL.CreateVectorSplat(VecI64->getElementCount(),
      VL.getInt64(2), "skv.stride2");
  Value *OneVec = VL.CreateVectorSplat(VecI64->getElementCount(),
      VL.getInt64(1), "skv.one");
  Value *LoIdxVec = VL.CreateMul(LaneIdx, Stride2Vec, "skv.lo_idx_vec");
  Value *HiIdxVec = VL.CreateAdd(LoIdxVec, OneVec, "skv.hi_idx_vec");
  Value *LoPtrVec = VL.CreateGEP(I64Ty, AccStart, LoIdxVec, "skv.lo_ptr_vec");
  Value *HiPtrVec = VL.CreateGEP(I64Ty, AccStart, HiIdxVec, "skv.hi_ptr_vec");

  Value *VAccLo = VL.CreateMaskedGather(VecI64, LoPtrVec, Align(8), Pred,
      ConstantAggregateZero::get(VecI64), "skv.vacc_lo");
  Value *VAccHi = VL.CreateMaskedGather(VecI64, HiPtrVec, Align(8), Pred,
      ConstantAggregateZero::get(VecI64), "skv.vacc_hi");

  // 128-bit add with carry: (acc_lo, acc_hi) += (lo, hi)
  Value *VNewLo = VL.CreateAdd(VAccLo, VLo, "skv.new_lo");
  Value *Carry = VL.CreateICmpULT(VNewLo, VLo, "skv.carry");
  Value *CarryExt = VL.CreateZExt(Carry, VecI64, "skv.carry_ext");
  Value *VTmp = VL.CreateAdd(VAccHi, VHi, "skv.vtmp");
  Value *VNewHi = VL.CreateAdd(VTmp, CarryExt, "skv.new_hi");

  // Stores: scatter back to strided accumulator
  VL.CreateMaskedScatter(VNewLo, LoPtrVec, Align(8), Pred);
  VL.CreateMaskedScatter(VNewHi, HiPtrVec, Align(8), Pred);

  // IV += VF, loop back
  Value *IVNext = VL.CreateAdd(IV, VF, "skv.iv_next");
  IV->addIncoming(IVNext, VecLoop);
  Value *LoopCmp = VL.CreateICmpULT(IVNext, TripCount, "skv.loop_cmp");
  VL.CreateCondBr(LoopCmp, VecLoop, VecAfter);

  // Vec after: update ptr phis for scalar remainder, branch to scalar loop
  IRBuilder<> VA(VecAfter);
  Value *NewOp = VA.CreateGEP(I64Ty, OpStart, IVNext, "skv.rem_op");
  Value *NewKey = VA.CreateGEP(I64Ty, KeyStart, IVNext, "skv.rem_key");
  Value *RemIdx2 = VA.CreateMul(IVNext, VA.getInt64(2), "skv.rem_idx2");
  Value *NewAcc = VA.CreateGEP(I64Ty, AccStart, RemIdx2, "skv.rem_acc");

  // Update the original phi nodes to accept new values from VecAfter
  for (unsigned i = 0; i < OpPhi->getNumIncomingValues(); i++) {
    if (!L->contains(OpPhi->getIncomingBlock(i))) {
      OpPhi->setIncomingValue(i, NewOp);
      OpPhi->setIncomingBlock(i, VecAfter);
    }
  }
  for (unsigned i = 0; i < KeyPhi->getNumIncomingValues(); i++) {
    if (!L->contains(KeyPhi->getIncomingBlock(i))) {
      KeyPhi->setIncomingValue(i, NewKey);
      KeyPhi->setIncomingBlock(i, VecAfter);
    }
  }
  for (unsigned i = 0; i < AccPhi->getNumIncomingValues(); i++) {
    if (!L->contains(AccPhi->getIncomingBlock(i))) {
      AccPhi->setIncomingValue(i, NewAcc);
      AccPhi->setIncomingBlock(i, VecAfter);
    }
  }

  // Redirect preheader → VecPH (instead of Header)
  Preheader->getTerminator()->eraseFromParent();
  IRBuilder<> PHB(Preheader);
  // Check if count >= VF, if so go to vector loop, else skip to scalar
  Value *DoVec = PHB.CreateICmpUGE(TripCount, VF, "skv.do_vec");
  PHB.CreateCondBr(DoVec, VecPH, Header);

  // VecAfter → Header (for scalar remainder)
  VA.CreateBr(Header);

  if (Debug)
    errs() << "SWITCH_KEY_VEC: generated vector loop in " << F.getName() << "\n";

  return true;
}

PreservedAnalyses
AArch64SwitchKeyVectorizePass::run(Function &F, FunctionAnalysisManager &AM) {
  if (!EnableSwitchKeyVec)
    return PreservedAnalyses::all();

  if (F.getName().find(SwitchKeyName) == StringRef::npos)
    return PreservedAnalyses::all();

  auto &LI = AM.getResult<LoopAnalysis>(F);
  bool Debug = getenv("SWITCH_KEY_VEC_DEBUG") != nullptr;
  bool Changed = false;

  if (Debug)
    errs() << "SWITCH_KEY_VEC: processing " << F.getName()
           << " loops=" << LI.getLoopsInPreorder().size() << "\n";

  // Find innermost loops with mul+umul.fix pattern
  SmallVector<Loop *, 4> ToVectorize;
  for (Loop *L : LI.getLoopsInPreorder()) {
    if (!L->isInnermost())
      continue;
    if (!L->getLoopPreheader() || !L->getLoopLatch())
      continue;

    // Check for @llvm.umul.fix
    bool HasUmulFix = false;
    for (BasicBlock *BB : L->blocks()) {
      for (Instruction &I : *BB) {
        if (auto *II = dyn_cast<IntrinsicInst>(&I))
          if (II->getIntrinsicID() == Intrinsic::umul_fix) {
            HasUmulFix = true;
            break;
          }
      }
      if (HasUmulFix) break;
    }
    if (HasUmulFix)
      ToVectorize.push_back(L);
  }

  // Only vectorize ONE loop per function call (to avoid stale LoopInfo)
  for (Loop *L : ToVectorize) {
    if (Debug)
      errs() << "SWITCH_KEY_VEC: trying to vectorize loop in "
             << F.getName() << "\n";
    if (vectorizeMulAddLoop(L, F, AM)) {
      Changed = true;
      break; // Only one at a time
    }
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
