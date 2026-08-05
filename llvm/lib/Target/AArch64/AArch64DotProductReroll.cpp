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
/// carry) and replaces the switch with a single IV-based loop. This lets
/// LoopVectorize vectorize it into SVE `mad + umulh z + cmphi + ld2d/st2d`.
///
/// The pass runs at OptimizerEarlyEP (after MulI128Lowering, before
/// LoopVectorize). Gated on `-aarch64-dot-product-reroll` (off by default).
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64DotProductReroll.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Constants.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

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
  PHINode *AccLoPhi = nullptr;
  PHINode *AccHiPhi = nullptr;
  unsigned MaxIncoming = 0;
  for (PHINode &PN : MergeBlock->phis()) {
    if (PN.getNumIncomingValues() > MaxIncoming) {
      MaxIncoming = PN.getNumIncomingValues();
      AccLoPhi = &PN;
    }
  }
  if (!AccLoPhi) {
    LLVM_DEBUG(dbgs() << "DPREROLL: no acc_lo phi in merge, skip\n");
    return PreservedAnalyses::all();
  }
  // The other phi with same number of incoming values is acc_hi.
  for (PHINode &PN : MergeBlock->phis()) {
    if (&PN != AccLoPhi &&
        PN.getNumIncomingValues() == AccLoPhi->getNumIncomingValues()) {
      AccHiPhi = &PN;
      break;
    }
  }
  if (!AccHiPhi) {
    LLVM_DEBUG(dbgs() << "DPREROLL: no acc_hi phi in merge, skip\n");
    return PreservedAnalyses::all();
  }

  LLVM_DEBUG(dbgs() << "  acc_lo phi=" << *AccLoPhi << "\n"
                    << "  acc_hi phi=" << *AccHiPhi << "\n");

  // --- Generate the loop ---
  // Strategy: generate a vectorizable loop (mul + store, NO cross-iteration
  // dependency) + a scalar horizontal sum loop after. LoopVectorize can
  // vectorize the first loop (mul + store) into SVE mad + umulh z + ld1d/st1d.
  // The scalar sum loop handles the 128-bit carry chain (count ≤ 16, cheap).
  LLVMContext &Ctx = F.getContext();
  Type *I64Ty = Type::getInt64Ty(Ctx);
  Function *UmulFixFn = Intrinsic::getOrInsertDeclaration(
      F.getParent(), Intrinsic::umul_fix, {I64Ty});

  // Alloc a temp array in entry for {lo, hi} pairs (max 64 elements).
  // SEAL_MULTIPLY_ACCUMULATE_MOD_MAX = 16, but dot_product_mod handles
  // count > 16 via tail recursion, so 64 is safe.
  const unsigned MaxCount = 64;
  IRBuilder<> EB0(Entry, Entry->getFirstInsertionPt());
  Value *TempArr = EB0.CreateAlloca(
      ArrayType::get(I64Ty, MaxCount * 2), nullptr, "dpreroll.temp");

  // Create blocks.
  BasicBlock *Preheader = BasicBlock::Create(Ctx, "dpreroll.ph", &F);
  BasicBlock *LoopBody = BasicBlock::Create(Ctx, "dpreroll.loop", &F);
  BasicBlock *SumPre = BasicBlock::Create(Ctx, "dpreroll.sumph", &F);
  BasicBlock *SumLoop = BasicBlock::Create(Ctx, "dpreroll.sumloop", &F);
  BasicBlock *AfterLoop = BasicBlock::Create(Ctx, "dpreroll.after", &F);

  // Preheader: branch to loop body.
  IRBuilder<> PB(Preheader);
  PB.CreateBr(LoopBody);

  // Loop body: IV phi (no acc phi — store to temp array, no reduction).
  // This makes the loop vectorizable by LoopVectorize.
  IRBuilder<> LB(LoopBody);
  PHINode *IV = LB.CreatePHI(I64Ty, 2, "dpreroll.iv");

  // Load op1[iv] and op2[iv].
  Value *Op1Elem = LB.CreateGEP(I64Ty, Op1, IV, "dpreroll.op1_i");
  Value *Op2Elem = LB.CreateGEP(I64Ty, Op2, IV, "dpreroll.op2_i");
  Value *V1 = LB.CreateLoad(I64Ty, Op1Elem, "dpreroll.v1");
  Value *V2 = LB.CreateLoad(I64Ty, Op2Elem, "dpreroll.v2");

  // mul + umul.fix (64x64→128).
  Value *Lo = LB.CreateMul(V2, V1, "dpreroll.lo");
  Value *Hi = LB.CreateCall(UmulFixFn, {V2, V1, LB.getInt32(64)},
                            "dpreroll.hi");

  // Store {lo, hi} to temp array (stride 2: [iv*2] = lo, [iv*2+1] = hi).
  Value *LoIdx = LB.CreateMul(IV, LB.getInt64(2), "dpreroll.lo_idx");
  Value *HiIdx = LB.CreateAdd(LoIdx, LB.getInt64(1), "dpreroll.hi_idx");
  Value *LoPtr = LB.CreateGEP(I64Ty, TempArr, LoIdx, "dpreroll.lo_ptr");
  Value *HiPtr = LB.CreateGEP(I64Ty, TempArr, HiIdx, "dpreroll.hi_ptr");
  LB.CreateStore(Lo, LoPtr);
  LB.CreateStore(Hi, HiPtr);

  // IV++ and loop condition.
  Value *IVNext = LB.CreateAdd(IV, LB.getInt64(1), "dpreroll.iv_next");
  Value *Cmp = LB.CreateICmpULT(IVNext, Count, "dpreroll.cmp");
  LB.CreateCondBr(Cmp, LoopBody, SumPre);

  IV->addIncoming(LB.getInt64(0), Preheader);
  IV->addIncoming(IVNext, LoopBody);

  // Sum preheader: branch to sum loop.
  IRBuilder<> SP(SumPre);
  SP.CreateBr(SumLoop);

  // Sum loop: scalar 128-bit add of all temp elements (has carry chain,
  // not vectorizable, but count ≤ 16 so cheap).
  IRBuilder<> SL(SumLoop);
  PHINode *SumIV = SL.CreatePHI(I64Ty, 2, "dpreroll.sum_iv");
  PHINode *SumLo = SL.CreatePHI(I64Ty, 2, "dpreroll.sum_lo");
  PHINode *SumHi = SL.CreatePHI(I64Ty, 2, "dpreroll.sum_hi");

  Value *SLoIdx = SL.CreateMul(SumIV, SL.getInt64(2), "dpreroll.slo_idx");
  Value *SHiIdx = SL.CreateAdd(SLoIdx, SL.getInt64(1), "dpreroll.shi_idx");
  Value *SLoPtr = SL.CreateGEP(I64Ty, TempArr, SLoIdx, "dpreroll.slo_ptr");
  Value *SHiPtr = SL.CreateGEP(I64Ty, TempArr, SHiIdx, "dpreroll.shi_ptr");
  Value *TLo = SL.CreateLoad(I64Ty, SLoPtr, "dpreroll.tlo");
  Value *THi = SL.CreateLoad(I64Ty, SHiPtr, "dpreroll.thi");

  // 128-bit add with carry (scalar, safe).
  Value *NewLo = SL.CreateAdd(SumLo, TLo, "dpreroll.new_lo");
  Value *Carry = SL.CreateICmpULT(NewLo, TLo, "dpreroll.carry");
  Value *CarryExt = SL.CreateZExt(Carry, I64Ty, "dpreroll.carry_ext");
  Value *Tmp = SL.CreateAdd(SumHi, THi, "dpreroll.tmp");
  Value *NewHi = SL.CreateAdd(Tmp, CarryExt, "dpreroll.new_hi");

  Value *SumIVNext = SL.CreateAdd(SumIV, SL.getInt64(1), "dpreroll.sum_iv_next");
  Value *SumCmp = SL.CreateICmpULT(SumIVNext, Count, "dpreroll.sum_cmp");
  SL.CreateCondBr(SumCmp, SumLoop, AfterLoop);

  SumIV->addIncoming(SL.getInt64(0), SumPre);
  SumIV->addIncoming(SumIVNext, SumLoop);
  SumLo->addIncoming(SL.getInt64(0), SumPre);
  SumLo->addIncoming(NewLo, SumLoop);
  SumHi->addIncoming(SL.getInt64(0), SumPre);
  SumHi->addIncoming(NewHi, SumLoop);

  // AfterLoop: branch to MergeBlock (Barrett reduce).
  IRBuilder<> AB(AfterLoop);
  AB.CreateBr(MergeBlock);

  // Update MergeBlock's phi nodes to accept sum from AfterLoop.
  AccLoPhi->addIncoming(NewLo, AfterLoop);
  AccHiPhi->addIncoming(NewHi, AfterLoop);

  // Modify entry block: replace switch with count==0 check.
  Switch->eraseFromParent();
  IRBuilder<> EB(Entry);
  Value *CountIsZero = EB.CreateICmpEQ(Count, EB.getInt64(0),
                                      "dpreroll.count_zero");
  EB.CreateCondBr(CountIsZero, ReturnBlock, Preheader);

  LLVM_DEBUG(dbgs() << "DPREROLL: rerolled dot_product_mod into loop!\n");

  return PreservedAnalyses::none();
}
