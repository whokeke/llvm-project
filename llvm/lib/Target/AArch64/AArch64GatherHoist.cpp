//===-- AArch64GatherHoist.cpp ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Detects @llvm.masked.gather calls where the pointer vector is a splat
/// (all lanes point to the same address). Replaces them with a scalar
/// load + vector splat, which LICM can hoist to the loop preheader.
///
/// The IR-level approach of hoisting the gather intrinsic directly doesn't
/// work because instcombine folds `select(ptrue, gather(ptrue), zero)` back
/// to `gather(ptrue)`. By converting the splat-ptr gather to a regular
/// scalar load + splat, LICM can hoist the load naturally.
///
/// Gated on `-aarch64-gather-hoist` (off by default).
/// Runs at OptimizerLastEP (after LoopVectorize, before ISel).
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64GatherHoist.h"
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
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-gather-hoist"

static cl::opt<bool> EnableGatherHoist(
    "aarch64-gather-hoist",
    cl::init(false), cl::Hidden,
    cl::desc("Convert splat-ptr masked gathers to scalar load + splat "
             "for LICM hoisting"));

/// Check if a Value is a splat of a single scalar. Returns the scalar
/// if yes, nullptr if not.
static Value *getSplatScalar(Value *V) {
  // shufflevector zeroinitializer → splat of insertelement's element
  if (auto *SV = dyn_cast<ShuffleVectorInst>(V)) {
    if (SV->getShuffleMask().size() > 0 &&
        SV->getShuffleMask()[0] == 0 &&
        std::all_of(SV->getShuffleMask().begin() + 1,
                    SV->getShuffleMask().end(),
                    [&](int M) { return M == 0 || M == -1; })) {
      Value *Op0 = SV->getOperand(0);
      if (auto *IE = dyn_cast<InsertElementInst>(Op0))
        return IE->getOperand(1);
      // poison insertelement with undef element
      if (isa<UndefValue>(Op0))
        return nullptr;
    }
  }
  // Direct splat via vectorSplat
  if (auto *IE = dyn_cast<InsertElementInst>(V))
    return IE->getOperand(1);
  return nullptr;
}

PreservedAnalyses
AArch64GatherHoistPass::run(Function &F, FunctionAnalysisManager &AM) {
  if (!EnableGatherHoist)
    return PreservedAnalyses::all();

  auto &LI = AM.getResult<LoopAnalysis>(F);
  bool Debug = getenv("GATHER_HOIST_DEBUG") != nullptr;
  bool Changed = false;

  if (Debug)
    errs() << "GATHER_HOIST: processing " << F.getName()
           << " loops=" << LI.getLoopsInPreorder().size() << "\n";

  // Scan ALL instructions (not just in loops — we replace the pattern
  // everywhere; LICM will handle hoisting if it's in a loop).
  for (BasicBlock &BB : F) {
    SmallVector<IntrinsicInst *, 4> ToTransform;
    for (Instruction &I : BB) {
      auto *II = dyn_cast<IntrinsicInst>(&I);
      if (!II || II->getIntrinsicID() != Intrinsic::masked_gather)
        continue;

      // Check if pointer vector is a splat.
      Value *PtrVec = II->getArgOperand(0);
      Value *ScalarPtr = getSplatScalar(PtrVec);
      if (!ScalarPtr)
        continue;

      ToTransform.push_back(II);
    }

    for (IntrinsicInst *Gather : ToTransform) {
      Value *PtrVec = Gather->getArgOperand(0);
      Value *Mask = Gather->getArgOperand(1);
      Value *Passthru = Gather->getArgOperand(2);
      Type *VecTy = Gather->getType();
      Type *EltTy = VecTy->getScalarType();

      Value *ScalarPtr = getSplatScalar(PtrVec);

      if (Debug)
        errs() << "GATHER_HOIST: replacing splat-ptr gather in "
               << F.getName() << " scalar_ptr=" << *ScalarPtr << "\n";

      IRBuilder<> B(Gather);

      // Create scalar load from the splat base address.
      Value *ScalarLoad = B.CreateLoad(EltTy, ScalarPtr,
                                        "gh.scalar_load");

      // Splat the scalar to all lanes.
      Value *Splat = B.CreateVectorSplat(
          cast<VectorType>(VecTy)->getElementCount(),
          ScalarLoad, "gh.splat");

      // Apply mask: for inactive lanes, use passthru.
      Value *Result;
      if (isa<PoisonValue>(Passthru)) {
        // With poison passthru, the gather returns poison for inactive
        // lanes. But our splat returns the loaded value for ALL lanes.
        // Use select to match the original semantics.
        Result = B.CreateSelect(Mask, Splat,
                                ConstantAggregateZero::get(VecTy),
                                "gh.result");
      } else {
        Result = B.CreateSelect(Mask, Splat, Passthru, "gh.result");
      }

      Gather->replaceAllUsesWith(Result);
      Gather->eraseFromParent();
      Changed = true;

      if (Debug)
        errs() << "GATHER_HOIST: replaced with scalar_load + splat + select\n";
    }
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
