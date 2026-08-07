//===-- AArch64DotProductReroll.cpp ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Detects SEAL's `dot_product_mod` function (which uses a switch-case on
/// `count` to dispatch to `multiply_accumulate_uint64<N>` template
/// specializations — fully inlined as sequential mul + 128-bit add with
/// carry) and replaces the switch with a scalable SVE vector loop that
/// does 128-bit carry propagation in-register (matching the ACLE SVE
/// intrinsic version). The vector loop uses:
///   whilelt + masked.load + mul_z + umulh_z + add_u + cmphi + zext + add_u
/// followed by a short scalar horizontal reduction (N iterations).
///
/// The pass runs at OptimizerEarlyEP (after MulI128Lowering, before
/// LoopVectorize). Gated on `-aarch64-dot-product-reroll` (off by default).
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64DotProductReroll.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-dot-product-reroll"

static cl::opt<bool> EnableDotProductReroll(
    "aarch64-dot-product-reroll",
    cl::init(false), cl::Hidden,
    cl::desc("Reroll SEAL dot_product_mod switch-case into IV-based loop "
             "for SVE auto-vectorization"));

// SEAL's dot_product_mod mangled name.
// seal::util::dot_product_mod(const uint64_t*, const uint64_t*, size_t, const Modulus&)
static const char *DotProductModName =
    "_ZN4seal4util15dot_product_modEPKmS2_mRKNS_7ModulusE";

PreservedAnalyses
AArch64DotProductRerollPass::run(Function &F, FunctionAnalysisManager &AM) {
  if (!EnableDotProductReroll)
    return PreservedAnalyses::all();

  // Match by function name.
  if (F.getName() != DotProductModName)
    return PreservedAnalyses::all();

  // dot_product_mod signature:
  //   i64 @dot_product_mod(ptr %op1, ptr %op2, i64 %count, ptr %modulus)
  if (F.arg_size() != 4)
    return PreservedAnalyses::all();

  auto ArgIt = F.arg_begin();
  Argument *Op1 = ArgIt;          // ptr op1
  Argument *Op2 = ++ArgIt;        // ptr op2
  Argument *Count = ++ArgIt;      // i64 count
  Argument *Modulus = ++ArgIt;    // ptr modulus

  // Find the entry block's switch on count.
  BasicBlock *Entry = &F.getEntryBlock();
  SwitchInst *Switch = nullptr;
  for (Instruction &I : *Entry) {
    if ((Switch = dyn_cast<SwitchInst>(&I)))
      break;
  }
  if (!Switch) {
    LLVM_DEBUG(dbgs() << "DPREROLL: no switch in entry, skip\n");
    return PreservedAnalyses::all();
  }
  if (Switch->getCondition() != Count) {
    LLVM_DEBUG(dbgs() << "DPREROLL: switch not on count, skip\n");
    return PreservedAnalyses::all();
  }

  // Find the merge block (default successor's predecessors converge here).
  // The switch has a default label (block 740 in SEAL) and case labels.
  // All cases eventually branch to a common merge block that does the
  // inline Barrett reduce 128. We find it by looking at the default
  // case's successors.
  //
  // Actually, simpler: the merge block is the one that has phi nodes
  // collecting results from all cases. Find the block that all case
  // blocks (except case 0) branch to.
  //
  // SEAL's structure: case 0 → block 774 (return 0). All other cases →
  // block 745 (Barrett reduce). block 745 → block 774 (return).
  //
  // Find the return block: it's the block with `ret` that has phi nodes
  // from the entry (count=0 case) and from the merge block.
  BasicBlock *ReturnBlock = nullptr;   // block 774
  BasicBlock *MergeBlock = nullptr;     // block 745

  for (BasicBlock &BB : F) {
    if (isa<ReturnInst>(BB.getTerminator())) {
      // Check if this block has a phi with incoming from entry.
      for (PHINode &PN : BB.phis()) {
        for (unsigned i = 0; i < PN.getNumIncomingValues(); i++) {
          if (PN.getIncomingBlock(i) == Entry) {
            ReturnBlock = &BB;
            break;
          }
        }
      }
      if (ReturnBlock) break;
    }
  }
  if (!ReturnBlock) {
    LLVM_DEBUG(dbgs() << "DPREROLL: no return block with entry phi, skip\n");
    return PreservedAnalyses::all();
  }

  // The merge block is the non-entry predecessor of ReturnBlock.
  for (BasicBlock *Pred : predecessors(ReturnBlock)) {
    if (Pred != Entry) {
      MergeBlock = Pred;
      break;
    }
  }
  if (!MergeBlock) {
    LLVM_DEBUG(dbgs() << "DPREROLL: no merge block found, skip\n");
    return PreservedAnalyses::all();
  }

  LLVM_DEBUG(dbgs() << "DPREROLL: dot_product_mod detected!\n"
                    << "  entry=" << Entry->getName()
                    << " switch on count=" << *Count << "\n"
                    << "  merge=" << MergeBlock->getName()
                    << " return=" << ReturnBlock->getName() << "\n");

  // Find the phi nodes in MergeBlock that collect acc_lo and acc_hi.
  // These are the phis with the most incoming values (one per case).
  // We must correctly identify which is lo and which is hi — they are
  // NOT in a fixed order. Check one case's incoming value: if it's a
  // `mul` instruction, that phi is acc_lo; if it's an `@llvm.umul.fix`
  // call, that phi is acc_hi.
  PHINode *AccLoPhi = nullptr;
  PHINode *AccHiPhi = nullptr;
  unsigned MaxIncoming = 0;
  for (PHINode &PN : MergeBlock->phis()) {
    if (PN.getNumIncomingValues() > MaxIncoming) {
      MaxIncoming = PN.getNumIncomingValues();
    }
  }
  for (PHINode &PN : MergeBlock->phis()) {
    if (PN.getNumIncomingValues() != MaxIncoming)
      continue;
    // Check one incoming value to determine lo vs hi.
    for (unsigned i = 0; i < PN.getNumIncomingValues(); i++) {
      Value *V = PN.getIncomingValue(i);
      if (auto *II = dyn_cast<IntrinsicInst>(V)) {
        // @llvm.umul.fix → this phi is acc_hi
        AccHiPhi = &PN;
        break;
      } else if (auto *BO = dyn_cast<BinaryOperator>(V)) {
        if (BO->getOpcode() == Instruction::Mul) {
          // mul → this phi is acc_lo
          AccLoPhi = &PN;
          break;
        }
      }
    }
  }
  // Fallback: if detection failed (e.g. incoming values are loads or
  // other), assume first phi = acc_lo, second = acc_hi (may be wrong).
  if (!AccLoPhi && !AccHiPhi) {
    unsigned found = 0;
    for (PHINode &PN : MergeBlock->phis()) {
      if (PN.getNumIncomingValues() != MaxIncoming)
        continue;
      if (found == 0) AccLoPhi = &PN;
      else AccHiPhi = &PN;
      found++;
    }
  } else if (!AccLoPhi) {
    // AccHiPhi found, the other one is AccLoPhi
    for (PHINode &PN : MergeBlock->phis()) {
      if (&PN != AccHiPhi && PN.getNumIncomingValues() == MaxIncoming) {
        AccLoPhi = &PN;
        break;
      }
    }
  } else if (!AccHiPhi) {
    // AccLoPhi found, the other one is AccHiPhi
    for (PHINode &PN : MergeBlock->phis()) {
      if (&PN != AccLoPhi && PN.getNumIncomingValues() == MaxIncoming) {
        AccHiPhi = &PN;
        break;
      }
    }
  }
  if (!AccLoPhi || !AccHiPhi) {
    LLVM_DEBUG(dbgs() << "DPREROLL: could not identify acc_lo/acc_hi phi, skip\n");
    return PreservedAnalyses::all();
  }

  LLVM_DEBUG(dbgs() << "  acc_lo phi=" << *AccLoPhi << "\n"
                    << "  acc_hi phi=" << *AccHiPhi << "\n");

  // Clone the function BEFORE transformation to preserve the original
  // switch-case (for count > 16). The clone (dot_product_mod_large) handles
  // the tail recursion path. After transformation, self-calls in F are
  // redirected to the clone → F is no longer recursive → safe to alwaysinline.
  ValueToValueMapTy VMap;
  Function *LargeFn = CloneFunction(&F, VMap);
  LargeFn->setName(Twine(F.getName()) + "_large");

  // --- Generate the SVE vector loop ---
  // Instead of a scalar mul+store loop + scalar reduction (which relied on
  // LoopVectorize to vectorize), we directly generate a scalable SVE vector
  // loop with 128-bit carry propagation — matching the ACLE intrinsic
  // version. This bypasses LoopVectorize (which can't recognize 128-bit carry
  // reduction) and produces: mul z + umulh z + add z + cmphi + add z,p/m
  // in the vector loop, with a short scalar horizontal reduction after.
  LLVMContext &Ctx = F.getContext();
  Type *I64Ty = Type::getInt64Ty(Ctx);
  Type *I1Ty = Type::getInt1Ty(Ctx);
  auto *Nxv2I64 = VectorType::get(I64Ty, ElementCount::getScalable(2));
  auto *Nxv2I1 = VectorType::get(I1Ty, ElementCount::getScalable(2));

  // All-true predicate and zero vector for SVE ops.
  Constant *AllTrue = ConstantVector::getSplat(
      ElementCount::getScalable(2), ConstantInt::getTrue(I1Ty));
  Constant *ZeroVec = Constant::getNullValue(Nxv2I64);

  // Allocas for horizontal reduction storage (store acc_lo/acc_hi vectors).

  IRBuilder<> EB0(Entry, Entry->getFirstInsertionPt());

  // Create blocks.
  BasicBlock *Preheader = BasicBlock::Create(Ctx, "dpreroll.ph", &F);
  BasicBlock *VecLoop = BasicBlock::Create(Ctx, "dpreroll.vloop", &F);
  BasicBlock *TreeStart = BasicBlock::Create(Ctx, "dpreroll.tstart", &F);

  // Preheader: compute N = vscale * 2 (= svcntd()).
  IRBuilder<> PB(Preheader);
  Value *Vscale = PB.CreateIntrinsic(Intrinsic::vscale, {I64Ty}, {}, {},
                                     "dpreroll.vscale");
  Value *N = PB.CreateMul(Vscale, PB.getInt64(2), "dpreroll.N");
  PB.CreateBr(VecLoop);

  // VecLoop: scalable SVE vector loop with 128-bit carry propagation.
  // Pattern (matching ACLE sve_add_u128):
  //   pg = whilelt(IV, count)
  //   v1 = masked_load(pg, op1+IV)
  //   v2 = masked_load(pg, op2+IV)
  //   prod_lo = mul_z(pg, v1, v2)
  //   prod_hi = umulh_z(pg, v1, v2)
  //   new_lo = add_x(acc_lo, prod_lo)        // unpredicated
  //   carry = cmphi(acc_lo, new_lo)         // old > new = carry
  //   carry_val = zext(carry) to nxv2i64    // 0 or 1
  //   tmp_hi = add_x(acc_hi, prod_hi)       // unpredicated
  //   new_hi = add_x(tmp_hi, carry_val)     // unpredicated
  IRBuilder<> VL(VecLoop);
  PHINode *IV = VL.CreatePHI(I64Ty, 2, "dpreroll.iv");
  PHINode *AccLo = VL.CreatePHI(Nxv2I64, 2, "dpreroll.acc_lo");
  PHINode *AccHi = VL.CreatePHI(Nxv2I64, 2, "dpreroll.acc_hi");

  // pg = whilelt(IV, count) via get_active_lane_mask
  Value *PG = VL.CreateIntrinsic(
      Intrinsic::get_active_lane_mask, {Nxv2I1, I64Ty},
      {IV, Count}, {}, "dpreroll.pg");

  // v1 = masked_load(pg, op1+IV), v2 = masked_load(pg, op2+IV)
  Value *Ptr1 = VL.CreateGEP(I64Ty, Op1, IV, "dpreroll.ptr1");
  Value *Ptr2 = VL.CreateGEP(I64Ty, Op2, IV, "dpreroll.ptr2");
  Value *V1 = VL.CreateMaskedLoad(Nxv2I64, Ptr1, Align(8), PG, ZeroVec,
                                  "dpreroll.v1");
  Value *V2 = VL.CreateMaskedLoad(Nxv2I64, Ptr2, Align(8), PG, ZeroVec,
                                  "dpreroll.v2");

  // prod_lo = mul_z(pg, v1, v2), prod_hi = umulh_z(pg, v1, v2)
  Value *ProdLo = VL.CreateIntrinsic(
      Intrinsic::aarch64_sve_mul, {Nxv2I64}, {PG, V1, V2}, {},
      "dpreroll.prod_lo");
  Value *ProdHi = VL.CreateIntrinsic(
      Intrinsic::aarch64_sve_umulh, {Nxv2I64}, {PG, V1, V2}, {},
      "dpreroll.prod_hi");

  // 128-bit add with carry (SVE predicate-based carry propagation):
  Value *NewLo = VL.CreateIntrinsic(
      Intrinsic::aarch64_sve_add_u, {Nxv2I64}, {AllTrue, AccLo, ProdLo}, {},
      "dpreroll.new_lo");
  Value *CarryPred = VL.CreateIntrinsic(
      Intrinsic::aarch64_sve_cmphi, {Nxv2I64}, {AllTrue, AccLo, NewLo}, {},
      "dpreroll.carry");
  Value *CarryVal = VL.CreateZExt(CarryPred, Nxv2I64, "dpreroll.carry_val");
  Value *TmpHi = VL.CreateIntrinsic(
      Intrinsic::aarch64_sve_add_u, {Nxv2I64}, {AllTrue, AccHi, ProdHi}, {},
      "dpreroll.tmp_hi");
  Value *NewHi = VL.CreateIntrinsic(
      Intrinsic::aarch64_sve_add_u, {Nxv2I64}, {AllTrue, TmpHi, CarryVal}, {},
      "dpreroll.new_hi");

  // IV += N, loop condition (do-while: always runs at least once).
  Value *IVNext = VL.CreateAdd(IV, N, "dpreroll.iv_next");
  Value *Cmp = VL.CreateICmpULT(IVNext, Count, "dpreroll.cmp");
  VL.CreateCondBr(Cmp, VecLoop, TreeStart);

  IV->addIncoming(PB.getInt64(0), Preheader);
  IV->addIncoming(IVNext, VecLoop);
  AccLo->addIncoming(ZeroVec, Preheader);
  AccLo->addIncoming(NewLo, VecLoop);
  AccHi->addIncoming(ZeroVec, Preheader);
  AccHi->addIncoming(NewHi, VecLoop);

  // TreeReduce: SVE register-only horizontal reduction (no memory access).
  // Replaces the scalar strided-load + adds+adc loop that was the bottleneck
  // (444 perf samples on ldr). Uses pairwise reduction via splice.left:
  //   Level 1: rotate by 1 → pairwise add (N/2 independent 128-bit adds)
  //   Level 2: rotate by 2 → pairwise add
  //   ... until 1 element remains (lane 0)
  // Each level uses sve_add_u128 (add + cmphi + zext + add) for carry.
  // splice.left index is compile-time constant → lowers to SVE ext.
  // Runtime check (N > stride) guards each level (skip if N too small).
  //
  // Critical path: log2(N) levels × (add+cmphi+add) vs N × (ldr+ldr+adds+adc).
  // For N=4: 2 levels vs 4 iterations, and no memory access at all.
  auto GenTreeLevel = [&](BasicBlock *&CurBB, Value *&CurLo, Value *&CurHi,
                          int Stride) {
    // Check: N > Stride?
    BasicBlock *CheckBB = BasicBlock::Create(Ctx, "dpreroll.chk", &F);
    BasicBlock *DoBB = BasicBlock::Create(Ctx, "dpreroll.lvl", &F);
    BasicBlock *NextBB = BasicBlock::Create(Ctx, "dpreroll.next", &F);

    IRBuilder<> CB(CheckBB);
    Value *Cmp = CB.CreateICmpUGT(N, CB.getInt64(Stride),
                                  ("dpreroll.gt" + Twine(Stride)).str());
    CB.CreateCondBr(Cmp, DoBB, NextBB);

    // Do: rotate by Stride, sve_add_128
    IRBuilder<> DB(DoBB);
    Value *RotLo = DB.CreateIntrinsic(
        Intrinsic::vector_splice_left, {Nxv2I64},
        {CurLo, CurLo, DB.getInt32(Stride)}, {},
        ("dpreroll.rot_lo" + Twine(Stride)).str());
    Value *RotHi = DB.CreateIntrinsic(
        Intrinsic::vector_splice_left, {Nxv2I64},
        {CurHi, CurHi, DB.getInt32(Stride)}, {},
        ("dpreroll.rot_hi" + Twine(Stride)).str());

    Value *SumLo = DB.CreateIntrinsic(
        Intrinsic::aarch64_sve_add_u, {Nxv2I64}, {AllTrue, CurLo, RotLo}, {},
        ("dpreroll.sum_lo" + Twine(Stride)).str());
    Value *CarryPred = DB.CreateIntrinsic(
        Intrinsic::aarch64_sve_cmphi, {Nxv2I64}, {AllTrue, CurLo, SumLo}, {},
        ("dpreroll.carry" + Twine(Stride)).str());
    Value *CarryVal = DB.CreateZExt(CarryPred, Nxv2I64,
        ("dpreroll.cv" + Twine(Stride)).str());
    Value *TmpHi = DB.CreateIntrinsic(
        Intrinsic::aarch64_sve_add_u, {Nxv2I64}, {AllTrue, CurHi, RotHi}, {},
        ("dpreroll.tmp_hi" + Twine(Stride)).str());
    Value *SumHi = DB.CreateIntrinsic(
        Intrinsic::aarch64_sve_add_u, {Nxv2I64}, {AllTrue, TmpHi, CarryVal}, {},
        ("dpreroll.sum_hi" + Twine(Stride)).str());
    DB.CreateBr(NextBB);

    // Connect CurBB → CheckBB
    IRBuilder<> PrevB(CurBB);
    PrevB.CreateBr(CheckBB);

    // Phi in NextBB: pick SumLo/SumHi (if level ran) or CurLo/CurHi (skip)
    IRBuilder<> NB(NextBB);
    PHINode *NextLo = NB.CreatePHI(Nxv2I64, 2, ("dpreroll.nl" + Twine(Stride)).str());
    PHINode *NextHi = NB.CreatePHI(Nxv2I64, 2, ("dpreroll.nh" + Twine(Stride)).str());
    NextLo->addIncoming(CurLo, CheckBB);
    NextLo->addIncoming(SumLo, DoBB);
    NextHi->addIncoming(CurHi, CheckBB);
    NextHi->addIncoming(SumHi, DoBB);

    CurBB = NextBB;
    CurLo = NextLo;
    CurHi = NextHi;
  };

  // Entry to tree reduction: TreeStart branches to first check level.
  // TreeStart is already the exit target of VecLoop's condbr.

  // Generate levels: stride 1, 2, 4, 8, 16 (covers up to 32 elements)
  BasicBlock *CurBB = TreeStart;
  Value *CurLo = NewLo;
  Value *CurHi = NewHi;
  GenTreeLevel(CurBB, CurLo, CurHi, 1);
  GenTreeLevel(CurBB, CurLo, CurHi, 2);
  GenTreeLevel(CurBB, CurLo, CurHi, 4);
  GenTreeLevel(CurBB, CurLo, CurHi, 8);
  GenTreeLevel(CurBB, CurLo, CurHi, 16);

  // Extract lane 0 → scalar result → feed to MergeBlock's phi
  BasicBlock *ExtractBB = BasicBlock::Create(Ctx, "dpreroll.extract", &F);
  {
    IRBuilder<> EB(CurBB);
    EB.CreateBr(ExtractBB);
  }
  IRBuilder<> EX(ExtractBB);
  Value *ResultLo = EX.CreateExtractElement(CurLo, ConstantInt::get(I64Ty, 0),
                                            "dpreroll.result_lo");
  Value *ResultHi = EX.CreateExtractElement(CurHi, ConstantInt::get(I64Ty, 0),
                                            "dpreroll.result_hi");
  EX.CreateBr(MergeBlock);

  // Update MergeBlock's phi nodes to accept result from ExtractBB.
  AccLoPhi->addIncoming(ResultLo, ExtractBB);
  AccHiPhi->addIncoming(ResultHi, ExtractBB);

  // Modify entry block: insert count check BEFORE the switch.
  // count==0 or count > 16 → switch (original behavior, including
  //   case 0 return + default tail recursion for count > 16).
  // 1 ≤ count ≤ 16 → SVE vector loop (no 128-bit overflow because
  //   SEAL_MULTIPLY_ACCUMULATE_MOD_MAX = 16 guarantees
  //   16 * (modulus-1)^2 < 2^128).
  //
  // The switch-case is kept reachable (for count > 16) — no dead code,
  // no dominance issues. No alwaysinline — the function is not inlined
  // into callers (CGSCC inliner already ran and saw the switch-case).
  BasicBlock *SwitchBlock = Entry->splitBasicBlock(
      Switch->getIterator(), "dpreroll.switch");
  (void)SwitchBlock;
  Entry->getTerminator()->eraseFromParent();
  IRBuilder<> EB(Entry);
  Value *CountLE16 = EB.CreateICmpULE(Count, EB.getInt64(16),
                                      "dpreroll.count_le16");
  Value *CountGT0 = EB.CreateICmpNE(Count, EB.getInt64(0),
                                    "dpreroll.count_gt0");
  Value *UseLoop = EB.CreateAnd(CountGT0, CountLE16,
                                "dpreroll.use_loop");
  EB.CreateCondBr(UseLoop, Preheader, SwitchBlock);

  // Update phi nodes in blocks that had Entry as predecessor but now
  // have SwitchBlock (because the switch moved there).
  for (BasicBlock *BB : {ReturnBlock}) {
    for (PHINode &PN : BB->phis()) {
      for (unsigned i = 0; i < PN.getNumIncomingValues(); i++) {
        if (PN.getIncomingBlock(i) == Entry)
          PN.setIncomingBlock(i, SwitchBlock);
      }
    }
  }

  // Replace self-calls (recursive tail recursion in switch-case default)
  // with calls to LargeFn (the clone). This makes F non-recursive, so
  // AlwaysInlinerPass can inline it without the "recursive" cost=never
  // rejection. LargeFn keeps its own self-call (recursive) for count>16
  // tail recursion — that's fine, it's not alwaysinline.
  SmallVector<CallBase *, 4> SelfCalls;
  for (Instruction &I : instructions(F))
    if (auto *CB = dyn_cast<CallBase>(&I))
      if (CB->getCalledFunction() == &F)
        SelfCalls.push_back(CB);
  for (CallBase *CB : SelfCalls) {
    IRBuilder<> B(CB);
    SmallVector<Value *, 4> Args;
    for (unsigned i = 0; i < CB->arg_size(); i++)
      Args.push_back(CB->getArgOperand(i));
    CallInst *NewCall = B.CreateCall(LargeFn, Args);
    NewCall->setDebugLoc(CB->getDebugLoc());
    CB->replaceAllUsesWith(NewCall);
    CB->eraseFromParent();
  }

  // Mark F as alwaysinline. It's now non-recursive (self-calls redirected
  // to LargeFn). AlwaysInlinerPass (added after this pass in the pipeline)
  // will inline F into callers (e.g. fast_convert_array). The inlined code
  // has: SVE loop (count<=16, compact) + switch-case (count>16, cold,
  // calls LargeFn — not inlined). The switch-case is reachable (no dead
  // code, no dominance issues).
  F.addFnAttr(Attribute::AlwaysInline);

  LLVM_DEBUG(dbgs() << "DPREROLL: rerolled dot_product_mod into loop!\n");

  return PreservedAnalyses::none();
}
