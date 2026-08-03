//===-- AArch64MulI128Lowering.cpp ---- Lower mul i128 (zext,zext)+extract -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This pass lowers the SEAL/FHE idiom
///   %m = mul nuw i128 (zext i64 X), (zext i64 Y)
///   %lo = trunc i128 %m to i64              (or: and %m, low_64_mask)
///   %hi = trunc (lshr i128 %m, 64) to i64    (or: and (lshr %m, 64), low_64_mask)
/// to
///   %lo = mul i64 X, Y
///   %hi = call i64 @llvm.umul.fix.i64(X, Y, 64)
///   %m_lo_zext = zext i64 %lo to i128  (replaces `and %m, low_mask`)
///   %m_hi_zext = zext i64 %hi to i128  (replaces `lshr %m, 64`)
///
/// The transform removes the illegal `mul i128` (which the loop vectorizer
/// cannot pack into a scalable <N x i128> on AArch64 -- only `<N x i64>` is
/// legal). After lowering, the loop vectorizer can pack the resulting
/// `mul i64` + `@llvm.umul.fix.i64` to scalable `<N x i64>` SVE forms, which
/// the backend lowers to `mul z.d` + `umulh z.d`. This is the key enabling
/// step for vectorizing SEAL's dyadic_product_coeffmod / multiply_uint64 /
/// dot_product_mod inner loops.
///
/// The transform is only applied when both operands of the `mul i128` are
/// `zext i64 -> i128` (the "clean" widening form). A mul i128 whose operands
/// are not zexts is left untouched (the backend cannot lower a full 128-bit
/// product to a single instruction on AArch64).
///
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64MulI128Lowering.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-mul-i128-lowering"

namespace {

// The actual implementation is in this anonymous-namespace class; the public
// `AArch64MulI128LoweringPass` in AArch64MulI128Lowering.h just delegates.
class AArch64MulI128LoweringImpl {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // end anonymous namespace

// Match `zext i64 -> i128` (scalar) or `zext <N x i64> -> <N x i128>` (vector)
// and return the i64 source. Returns nullptr if V is not a zext from a 64-bit
// integer type to a 128-bit integer type.
static Value *matchZextI64ToI128(Value *V) {
  auto *ZI = dyn_cast<ZExtInst>(V);
  if (!ZI)
    return nullptr;
  Type *SrcTy = ZI->getSrcTy();
  Type *DstTy = ZI->getDestTy();
  if (!SrcTy->isIntOrIntVectorTy() || SrcTy->getScalarSizeInBits() != 64)
    return nullptr;
  if (!DstTy->isIntOrIntVectorTy() || DstTy->getScalarSizeInBits() != 128)
    return nullptr;
  return ZI->getOperand(0);
}

// Return true if V is a ConstantInt whose value masks out the high half of a
// 128-bit integer (low-64-bits set, high-64-bits clear).
static bool isLow64Mask(Value *V) {
  auto *CI = dyn_cast<ConstantInt>(V);
  if (!CI)
    return false;
  const APInt &M = CI->getValue();
  if (M.getBitWidth() != 128)
    return false;
  // getLoBits/getHiBits return 128-bit APInts; truncate to 64 bits before
  // checking isAllOnes/isZero so the check is over the half we care about.
  return M.trunc(64).isAllOnes() && M.lshr(64).trunc(64).isZero();
}

// Return true if V masks out the low half of a 128-bit integer.
static bool isHigh64Mask(Value *V) {
  auto *CI = dyn_cast<ConstantInt>(V);
  if (!CI)
    return false;
  const APInt &M = CI->getValue();
  if (M.getBitWidth() != 128)
    return false;
  return M.trunc(64).isZero() && M.lshr(64).trunc(64).isAllOnes();
}

// Process a single candidate `mul i128 (zext i64 X, zext i64 Y)`. Returns true
// if all uses were rewritten and MulI can be erased.
static bool processCandidate(Function &F, Instruction *MulI) {
  Value *Op0 = MulI->getOperand(0);
  Value *Op1 = MulI->getOperand(1);
  Value *X = matchZextI64ToI128(Op0);
  Value *Y = matchZextI64ToI128(Op1);
  if (!X || !Y)
    return false;

  Type *I128Ty = MulI->getType();
  Type *I64Ty = X->getType();

  IRBuilder<> B(MulI);
  // Always needed (trunc mul_i128 -> low half).
  // Do NOT add 'nuw' to the i64 mul: while the i128 mul of two zext i64
  // operands is nuw (the 128-bit product never wraps), the *low 64 bits*
  // extracted via trunc CAN overflow i64 (e.g. 2^63 * 3 = 1.5 * 2^64, low
  // half wraps). Adding nuw would be UB on such overflow, causing miscompiles
  // (infinite loops in regex/library code that relies on modular arithmetic).
  Value *LowHalf = B.CreateMul(X, Y, /*Name=*/"", /*HasNUW=*/false,
                               /*HasNSW=*/false);
  Function *UMulFix =
      Intrinsic::getOrInsertDeclaration(F.getParent(), Intrinsic::umul_fix, {I64Ty});
  // @llvm.umul.fix.i64 has signature (i64, i64, i32 scale). The scale is the
  // fixed-point shift applied to the 128-bit product; scale=64 returns the
  // high 64 bits (umulh). Use i32 for the scale argument.
  Value *Scale = ConstantInt::get(Type::getInt32Ty(F.getContext()), 64);
  Value *HighHalf = B.CreateCall(UMulFix, {X, Y, Scale});

  // Lazily create `zext (low/high) to i128` only when a user actually wants a
  // 128-bit value (e.g. `and mul, low_mask`, or `lshr mul, 64` whose user is
  // another mul i128 taking the high half as a 128-bit operand). The trunc
  // case is rewritten in place without ever materializing a zext.
  Value *ZExtLo = nullptr;
  Value *ZExtHi = nullptr;
  auto getZExtLo = [&]() -> Value * {
    if (!ZExtLo)
      ZExtLo = B.CreateZExt(LowHalf, I128Ty);
    return ZExtLo;
  };
  auto getZExtHi = [&]() -> Value * {
    if (!ZExtHi)
      ZExtHi = B.CreateZExt(HighHalf, I128Ty);
    return ZExtHi;
  };

  SmallVector<User *, 16> Users(MulI->users());
  SmallVector<Instruction *, 16> Dead;

  if (getenv("MULI128_TRACE"))
    errs() << "  processCandidate: MulI=" << *MulI
           << "  users=" << Users.size() << "\n";

  for (User *U : Users) {
    Instruction *UI = dyn_cast<Instruction>(U);
    if (!UI) {
      if (getenv("MULI128_TRACE"))
        errs() << "    user not Instruction: " << *U << " -- BAIL\n";
      return false;
    }

    if (getenv("MULI128_TRACE"))
      errs() << "    user: " << *UI << "\n";

    // Case 1: `trunc i128 %m to i64`  (low half, scalar)
    if (auto *TI = dyn_cast<TruncInst>(UI)) {
      if (TI->getType()->getScalarSizeInBits() == 64) {
        TI->replaceAllUsesWith(LowHalf);
        Dead.push_back(TI);
        continue;
      }
    }
    // Case 2: `and i128 %m, mask` -- low or high half (low or high 64 bits).
    if (auto *AI = dyn_cast<BinaryOperator>(UI)) {
      if (AI->getOpcode() == Instruction::And) {
        Value *Other = AI->getOperand(0) == MulI ? AI->getOperand(1)
                                                  : AI->getOperand(0);
        if (getenv("MULI128_TRACE")) {
          if (auto *CI = dyn_cast<ConstantInt>(Other))
            errs() << "      and-mask: bitwidth=" << CI->getValue().getBitWidth()
                   << " value=" << CI->getValue() << " isLow=" << isLow64Mask(Other)
                   << " isHigh=" << isHigh64Mask(Other) << "\n";
          else
            errs() << "      and-mask: non-ConstantInt\n";
        }
        if (isLow64Mask(Other)) {
          AI->replaceAllUsesWith(getZExtLo());
          Dead.push_back(AI);
          continue;
        }
        if (isHigh64Mask(Other)) {
          AI->replaceAllUsesWith(getZExtHi());
          Dead.push_back(AI);
          continue;
        }
      }
      // Case 3: `lshr i128 %m, 64` -- replace per-user (trunc->high_half,
      // and-mask->zext high, downstream mul operand->zext high).
      if (AI->getOpcode() == Instruction::LShr) {
        if (auto *ShiftC = dyn_cast<ConstantInt>(AI->getOperand(1))) {
          if (ShiftC->equalsInt(64)) {
            // Process the lshr's own users; only the lshr itself is replaced
            // by `zext(high)` if it has any non-trunc/non-and-mask user
            // (e.g. it feeds another mul i128 in zext form).
            SmallVector<User *, 8> LShrUsers(AI->users());
            bool LShrHasUnhandledUser = false;
            for (User *LU : LShrUsers) {
              Instruction *LUI = dyn_cast<Instruction>(LU);
              if (!LUI) {
                LShrHasUnhandledUser = true;
                continue;
              }
              if (auto *TI = dyn_cast<TruncInst>(LUI)) {
                if (TI->getType()->getScalarSizeInBits() == 64) {
                  TI->replaceAllUsesWith(HighHalf);
                  Dead.push_back(TI);
                  continue;
                }
              }
              if (auto *MAI = dyn_cast<BinaryOperator>(LUI)) {
                if (MAI->getOpcode() == Instruction::And &&
                    isLow64Mask(MAI->getOperand(0) == AI ? MAI->getOperand(1)
                                                          : MAI->getOperand(0))) {
                  MAI->replaceAllUsesWith(getZExtHi());
                  Dead.push_back(MAI);
                  continue;
                }
              }
              // The lshr's user is another mul i128 taking it directly as
              // a 128-bit operand (e.g. SEAL Barrett `mul (lshr mul, 64),
              // zext ratio`). Replace the lshr itself with zext(high) so the
              // downstream mul sees a clean `zext i64 -> i128` form.
              LShrHasUnhandledUser = true;
            }
            if (LShrHasUnhandledUser) {
              AI->replaceAllUsesWith(getZExtHi());
              Dead.push_back(AI);
            } else {
              // All lshr users were rewritten (trunc/and-mask cases). The lshr
              // itself is now a dead intermediate -- erase it unconditionally
              // (its operands will be dropped by dropAllReferences below).
              Dead.push_back(AI);
            }
            continue;
          }
        }
      }
    }
    // Direct use of MulI as a 128-bit value (e.g. another `mul i128 %m, z`
    // without an intervening lshr/and/trunc) is unsafe to rewrite without
    // knowing the intended half; bail.
    if (getenv("MULI128_TRACE"))
      errs() << "    UNHANDLED user -- BAIL\n";
    return false;
  }

  // All uses were rewritten. Schedule the dead intermediate instructions and
  // the original MulI for erasure.
  for (Instruction *I : Dead)
    I->dropAllReferences();
  MulI->dropAllReferences();
  for (Instruction *I : Dead)
    if (I->use_empty())
      I->eraseFromParent();
  if (MulI->use_empty())
    MulI->eraseFromParent();
  return true;
}



PreservedAnalyses
AArch64MulI128LoweringImpl::run(Function &F, FunctionAnalysisManager &AM) {
  // The transform is only useful when SVE2 (umulh) is available; otherwise
  // @llvm.umul.fix.i64 gets expanded to a `__udivti3` libcall, which is
  // strictly worse than the original `mul i128 + lshr + trunc` (the latter
  // at least lets the backend emit mul + umulh on AArch64 with +v8.6a).
  // Gate on the function's target-features attribute (set by clang -march=...).
  if (F.hasFnAttribute("target-features")) {
    StringRef TF = F.getFnAttribute("target-features").getValueAsString();
    if (!TF.contains("+sve2") && !TF.contains("+v8.6a") &&
        !TF.contains("armv8.6-a") && !TF.contains("+sve2-aes"))
      return PreservedAnalyses::all();
  }

  if (getenv("MULI128_DEBUG"))
    errs() << "\n=== BEFORE " << F.getName() << " ===\n" << F;

  bool Changed = false;
  bool Progress = true;
  // Iterate to a fixed point. Each round may rewrite a `mul i128`'s operands
  // (turning an `and %m, low_mask` operand into a `zext (mul i64) to i128`),
  // which can turn a previously non-candidate mul into a clean candidate for
  // the next round.
  while (Progress) {
    Progress = false;
    SmallVector<Instruction *, 32> MulI128s;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (I.getOpcode() == Instruction::Mul &&
            I.getType()->isIntOrIntVectorTy() &&
            I.getType()->getScalarSizeInBits() == 128)
          MulI128s.push_back(&I);

    for (Instruction *MulI : MulI128s) {
      // MulI may have been erased by an earlier iteration of the inner loop.
      if (!MulI->getParent())
        continue;
      if (processCandidate(F, MulI)) {
        Progress = true;
        Changed = true;
      }
    }
  }

  if (getenv("MULI128_DEBUG"))
    errs() << "\n=== AFTER " << F.getName() << " ===\n" << F;

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

PreservedAnalyses
AArch64MulI128LoweringPass::run(Function &F, FunctionAnalysisManager &AM) {
  return AArch64MulI128LoweringImpl{}.run(F, AM);
}

// Provide a legacy pass wrapper so this transform also runs in -O0/fast-isel
// path (legacy backend pipeline).
namespace {
class AArch64MulI128Lowering : public FunctionPass {
public:
  static char ID;
  AArch64MulI128Lowering() : FunctionPass(ID) {
    initializeAArch64MulI128LoweringPass(*PassRegistry::getPassRegistry());
  }
  bool runOnFunction(Function &F) override {
    FunctionAnalysisManager DummyFAM;
    auto PA = AArch64MulI128LoweringImpl{}.run(F, DummyFAM);
    return !PA.areAllPreserved();
  }
  StringRef getPassName() const override {
    return "AArch64 mul i128 (zext,zext)+extract lowering";
  }
};
} // end anonymous namespace

char AArch64MulI128Lowering::ID = 0;
INITIALIZE_PASS_BEGIN(AArch64MulI128Lowering, DEBUG_TYPE,
                      "AArch64 mul i128 lowering", false, false)
INITIALIZE_PASS_END(AArch64MulI128Lowering, DEBUG_TYPE,
                    "AArch64 mul i128 lowering", false, false)

FunctionPass *llvm::createAArch64MulI128LoweringPass() {
  return new AArch64MulI128Lowering();
}
