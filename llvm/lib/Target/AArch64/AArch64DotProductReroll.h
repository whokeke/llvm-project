//===-- AArch64DotProductReroll.h ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64DOTPRODUCTREROLL_H
#define LLVM_LIB_TARGET_AARCH64_AARCH64DOTPRODUCTREROLL_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class AArch64DotProductRerollPass
    : public PassInfoMixin<AArch64DotProductRerollPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};

} // end namespace llvm

#endif
