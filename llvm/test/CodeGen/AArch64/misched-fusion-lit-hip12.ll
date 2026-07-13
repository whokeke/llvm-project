; RUN: llc %s -o - -mtriple=aarch64 -mcpu=hip12 \
; RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=HIP12
; RUN: llc %s -o - -mtriple=aarch64 -mcpu=hip12 -mattr=-fuse-literals \
; RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=NOLIT

; Verify that -mcpu=hip12 enables FeatureFuseLiterals.
;
; For 64-bit constants that need four half-words the scheduler picks:
;   - with FuseLiterals OFF:  adrp + ldr  (load from constant pool)
;   - with FuseLiterals ON:   movz + movk * 3 + fmov  (in-register
;                            materialisation, since the hardware fuses
;                            adjacent MOVZ + MOVK pairs into one MOP)
;
; The 32-bit integer case is always materialised in-register (only two
; half-words, cheaper than a constant-pool load) but CHECK-NEXT locks in
; the fusion-friendly adjacency.
;
; Background: upstream TuneHIP12 (commit d87350127163) did not enable
; FeatureFuseLiterals.  Empirical evaluation on a real HIP12 system
; (Part 0xD06, 2.3 GHz, kernel 6.6) shows MOVZ + MOVK pairs are decoded
; as a single MOP, giving a 25-28% wall-clock speed-up vs. the NOP-split
; variant over 50M iterations.  See HIP12 fusion-benefit bench T1.

; -----------------------------------------------------------------------------
; 32-bit integer constant: MOVZ + MOVK (lsl #16), always in-register.
; CHECK-NEXT guarantees the fusion-friendly adjacency.
; -----------------------------------------------------------------------------
define i32 @mat32() nounwind readnone {
; CHECK-LABEL: mat32:
; HIP12:      mov  [[R:w[0-9]+]], #{{[0-9]+}}
; HIP12-NEXT: movk [[R]], #{{[0-9]+}}, lsl #16
; NOLIT:      mov  [[R:w[0-9]+]], #{{[0-9]+}}
; NOLIT-NEXT: movk [[R]], #{{[0-9]+}}, lsl #16
entry:
  ret i32 -559038737
}

; -----------------------------------------------------------------------------
; 64-bit double constant that needs four half-words.
;   - HIP12  (FuseLiterals ON):  movz + movk*3 + fmov
;   - NOLIT  (FuseLiterals OFF): adrp + ldr  (constant pool)
; -----------------------------------------------------------------------------
define double @litf() nounwind readnone {
; CHECK-LABEL: litf:
; HIP12:      mov  [[R:x[0-9]+]], #11544
; HIP12-NEXT: movk [[R]], #21572, lsl #16
; HIP12-NEXT: movk [[R]], #8699, lsl #32
; HIP12-NEXT: movk [[R]], #16393, lsl #48
; HIP12-NEXT: fmov {{d[0-9]+}}, [[R]]
; NOLIT:      adrp [[A:x[0-9]+]], [[POOL:.LCPI[0-9]+_[0-9]+]]
; NOLIT-NEXT: ldr  {{d[0-9]+}}, {{[[]}}[[A]], :lo12:[[POOL]]{{[]]}}
entry:
  ret double 0x400921FB54442D18
}
