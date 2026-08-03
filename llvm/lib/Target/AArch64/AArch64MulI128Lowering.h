//===-- AArch64MulI128Lowering.h --- AArch64 IR pass ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Declaration of the new-PM IR pass `AArch64MulI128LoweringPass` that lowers
/// the `mul i128 (zext i64, zext i64) + lshr + trunc` idiom to `mul i64 +
/// @llvm.umul.fix.i64`, enabling SVE vectorization of SEAL/FHE high-multiply
/// loops. See AArch64MulI128Lowering.cpp for full documentation.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64MULI128LOWERING_H
#define LLVM_LIB_TARGET_AARCH64_AARCH64MULI128LOWERING_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class AArch64MulI128LoweringPass
    : public PassInfoMixin<AArch64MulI128LoweringPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AARCH64_AARCH64MULI128LOWERING_H
