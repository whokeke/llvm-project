//===-- AArch64LoopDeunroll.h --- AArch64 IR pass ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Declaration of the new-PM IR pass `AArch64LoopDeunrollPass` that rerolls
/// manual N-x unrolled loops (e.g. SEAL NTT butterfly
///   for (j=0; j+(N-1)<bound; j+=N) { ... N isomorphic bodies ... }
/// ) back to single-iteration form:
///   for (j=0; j<bound; j+=1) { ... 1 body ... }
///
/// After reroll, the loop vectorizer is free to pick the arch-native VF
/// (e.g. SVE VF=4) instead of being pinned to a narrower VF dictated by
/// the manual unroll (e.g. NEON ld4/st4 forces VF=2 on a 256-bit SVE).
///
/// Gated by `-mllvm -aarch64-loop-deunroll`. See AArch64LoopDeunroll.cpp
/// for the pattern-matching algorithm and safety checks.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64LOOPDEUNROLL_H
#define LLVM_LIB_TARGET_AARCH64_AARCH64LOOPDEUNROLL_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class AArch64LoopDeunrollPass
    : public PassInfoMixin<AArch64LoopDeunrollPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AARCH64_AARCH64LOOPDEUNROLL_H
