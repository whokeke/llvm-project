; RUN: llc -mtriple=aarch64 -verify-machineinstrs -O2 < %s \
; RUN:   | FileCheck %s --check-prefix=CHECK
; RUN: llc -mtriple=aarch64 -verify-machineinstrs -O2 \
; RUN:   -aarch64-atomic-relaxed-pairing=0 < %s \
; RUN:   | FileCheck %s --check-prefix=DISABLED
; RUN: llc -mtriple=aarch64 -global-isel -mattr=+lse2 -verify-machineinstrs -O2 < %s \
; RUN:   | FileCheck %s --check-prefix=GISEL

; A relaxed (monotonic) atomic load/store is a plain ldr/str on AArch64 (no
; barrier), identical to a non-atomic access at the instruction level. LDP/STP
; is two independent 64-bit accesses and preserves per-access atomicity and
; ordering semantics, so monotonic accesses may be paired. Ordered atomics
; (acquire/release/seqcst) and volatile accesses are never paired.
;
; See also the GISel precedent: AArch64LegalizerInfo already lowers 128-bit
; monotonic atomics to LDPXi/STPXi under +lse2.

define void @load_relaxed_then_plain(ptr %p) {
; CHECK-LABEL: load_relaxed_then_plain:
; CHECK:         ldp x8, x9, [x0]
;
; DISABLED-LABEL: load_relaxed_then_plain:
; DISABLED:         ldr x8, [x0]
; DISABLED:         ldr x9, [x0, #8]
  %gep1 = getelementptr i64, ptr %p, i64 1
  %v0 = load atomic i64, ptr %p monotonic, align 8
  %v1 = load i64, ptr %gep1, align 8
  store volatile i64 %v0, ptr undef
  store volatile i64 %v1, ptr undef
  ret void
}

define void @load_two_relaxed(ptr %p) {
; CHECK-LABEL: load_two_relaxed:
; CHECK:         ldp x8, x9, [x0]
;
; DISABLED-LABEL: load_two_relaxed:
; DISABLED:         ldr x8, [x0]
; DISABLED:         ldr x9, [x0, #8]
  %gep1 = getelementptr i64, ptr %p, i64 1
  %v0 = load atomic i64, ptr %p monotonic, align 8
  %v1 = load atomic i64, ptr %gep1 monotonic, align 8
  store volatile i64 %v0, ptr undef
  store volatile i64 %v1, ptr undef
  ret void
}

; Unordered atomics are unconditionally allowed (matching hasOrderedMemoryRef),
; so they pair into LDP regardless of -aarch64-atomic-relaxed-pairing.
define void @load_unordered_then_plain(ptr %p) {
; CHECK-LABEL: load_unordered_then_plain:
; CHECK:         ldp x8, x9, [x0]
;
; DISABLED-LABEL: load_unordered_then_plain:
; DISABLED:         ldp x8, x9, [x0]
  %gep1 = getelementptr i64, ptr %p, i64 1
  %v0 = load atomic i64, ptr %p unordered, align 8
  %v1 = load i64, ptr %gep1, align 8
  store volatile i64 %v0, ptr undef
  store volatile i64 %v1, ptr undef
  ret void
}

define void @load_acquire_then_plain(ptr %p) {
; CHECK-LABEL: load_acquire_then_plain:
; CHECK:         ldar x8, [x0]
; CHECK:         ldr x9, [x0, #8]
; CHECK-NOT:     ldp
;
; DISABLED-LABEL: load_acquire_then_plain:
; DISABLED:         ldar x8, [x0]
; DISABLED:         ldr x9, [x0, #8]
; DISABLED-NOT:    ldp
  %gep1 = getelementptr i64, ptr %p, i64 1
  %v0 = load atomic i64, ptr %p acquire, align 8
  %v1 = load i64, ptr %gep1, align 8
  store volatile i64 %v0, ptr undef
  store volatile i64 %v1, ptr undef
  ret void
}

define void @load_seqcst_then_plain(ptr %p) {
; CHECK-LABEL: load_seqcst_then_plain:
; CHECK:         ldar x8, [x0]
; CHECK:         ldr x9, [x0, #8]
; CHECK-NOT:     ldp
;
; DISABLED-LABEL: load_seqcst_then_plain:
; DISABLED:         ldar x8, [x0]
; DISABLED:         ldr x9, [x0, #8]
; DISABLED-NOT:    ldp
  %gep1 = getelementptr i64, ptr %p, i64 1
  %v0 = load atomic i64, ptr %p seq_cst, align 8
  %v1 = load i64, ptr %gep1, align 8
  store volatile i64 %v0, ptr undef
  store volatile i64 %v1, ptr undef
  ret void
}

define void @load_volatile_then_plain(ptr %p) {
; CHECK-LABEL: load_volatile_then_plain:
; CHECK:         ldr x8, [x0]
; CHECK:         ldr x9, [x0, #8]
; CHECK-NOT:     ldp
;
; DISABLED-LABEL: load_volatile_then_plain:
; DISABLED:         ldr x8, [x0]
; DISABLED:         ldr x9, [x0, #8]
; DISABLED-NOT:    ldp
  %gep1 = getelementptr i64, ptr %p, i64 1
  %v0 = load volatile i64, ptr %p, align 8
  %v1 = load i64, ptr %gep1, align 8
  store volatile i64 %v0, ptr undef
  store volatile i64 %v1, ptr undef
  ret void
}

define void @store_relaxed_then_plain(ptr %p, i64 %a, i64 %b) {
; CHECK-LABEL: store_relaxed_then_plain:
; CHECK:         stp x1, x2, [x0]
;
; DISABLED-LABEL: store_relaxed_then_plain:
; DISABLED:         str x1, [x0]
; DISABLED:         str x2, [x0, #8]
  %gep1 = getelementptr i64, ptr %p, i64 1
  store atomic i64 %a, ptr %p monotonic, align 8
  store i64 %b, ptr %gep1, align 8
  ret void
}

define void @store_two_relaxed(ptr %p, i64 %a, i64 %b) {
; CHECK-LABEL: store_two_relaxed:
; CHECK:         stp x1, x2, [x0]
;
; DISABLED-LABEL: store_two_relaxed:
; DISABLED:         str x1, [x0]
; DISABLED:         str x2, [x0, #8]
  %gep1 = getelementptr i64, ptr %p, i64 1
  store atomic i64 %a, ptr %p monotonic, align 8
  store atomic i64 %b, ptr %gep1 monotonic, align 8
  ret void
}

define void @store_release_then_plain(ptr %p, i64 %a, i64 %b) {
; CHECK-LABEL: store_release_then_plain:
; CHECK:         stlr x1, [x0]
; CHECK:         str x2, [x0, #8]
; CHECK-NOT:     stp
;
; DISABLED-LABEL: store_release_then_plain:
; DISABLED:         stlr x1, [x0]
; DISABLED:         str x2, [x0, #8]
; DISABLED-NOT:    stp
  %gep1 = getelementptr i64, ptr %p, i64 1
  store atomic i64 %a, ptr %p release, align 8
  store i64 %b, ptr %gep1, align 8
  ret void
}

define void @store_volatile_then_plain(ptr %p, i64 %a, i64 %b) {
; CHECK-LABEL: store_volatile_then_plain:
; CHECK:         str x1, [x0]
; CHECK:         str x2, [x0, #8]
; CHECK-NOT:     stp
;
; DISABLED-LABEL: store_volatile_then_plain:
; DISABLED:         str x1, [x0]
; DISABLED:         str x2, [x0, #8]
; DISABLED-NOT:    stp
  %gep1 = getelementptr i64, ptr %p, i64 1
  store volatile i64 %a, ptr %p, align 8
  store i64 %b, ptr %gep1, align 8
  ret void
}

; Non-regression: the GISel legalizer still lowers a 128-bit monotonic atomic
; load to LDPXi under +lse2. This path is independent of the LoadStoreOptimizer
; change (isSafeToPairMemRef is not consulted by the GISel legalizer).
define i128 @load_i128_monotonic_gisel(ptr %p) {
; GISEL-LABEL: load_i128_monotonic_gisel:
; GISEL:         ldp x0, x1, [x0]
  %v = load atomic i128, ptr %p monotonic, align 16
  ret i128 %v
}
