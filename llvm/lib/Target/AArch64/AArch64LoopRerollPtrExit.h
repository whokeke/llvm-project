//===-- AArch64LoopRerollPtrExit.h ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64LOOPREROLLPTREXIT_H
#define LLVM_LIB_TARGET_AARCH64LOOPREROLLPTREXIT_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class AArch64LoopRerollPtrExitPass
    : public PassInfoMixin<AArch64LoopRerollPtrExitPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};

} // end namespace llvm

#endif
