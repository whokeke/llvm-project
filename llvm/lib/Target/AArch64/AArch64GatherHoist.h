//===-- AArch64GatherHoist.h ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64GATHERHOIST_H
#define LLVM_LIB_TARGET_AARCH64_AARCH64GATHERHOIST_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class AArch64GatherHoistPass
    : public PassInfoMixin<AArch64GatherHoistPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};

} // end namespace llvm

#endif
