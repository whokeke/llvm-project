; RUN: llc -mtriple=aarch64 -verify-machineinstrs -O2 < %s \
; RUN:   | FileCheck %s --check-prefix=CHECK
; RUN: llc -mtriple=aarch64 -verify-machineinstrs -O2 \
; RUN:   -aarch64-base-address-cse=0 < %s \
; RUN:   | FileCheck %s --check-prefix=NOBASECSE

; Coalesce sibling SUBXri base-address materializations so that adjacent
; negative far-offset loads/stores (which ISel materializes with independent
; bases, defeating LSO pairing) can pair into LDP/STP.

define void @cse_i32_load(ptr %p) {
; CHECK-LABEL: cse_i32_load:
; CHECK:         sub x8, x0, #400
; CHECK-NOT:     sub
; CHECK:         ldp
;
; NOBASECSE-LABEL: cse_i32_load:
; NOBASECSE:        sub x8, x0, #400
; NOBASECSE:        sub x9, x0, #396
; NOBASECSE:        ldr
; NOBASECSE:        ldr
; NOBASECSE-NOT:    ldp
  %g0 = getelementptr i32, ptr %p, i64 -100
  %g1 = getelementptr i32, ptr %p, i64 -99
  %v0 = load i32, ptr %g0
  %v1 = load i32, ptr %g1
  store volatile i32 %v0, ptr undef
  store volatile i32 %v1, ptr undef
  ret void
}

define void @cse_i64_load(ptr %p) {
; CHECK-LABEL: cse_i64_load:
; CHECK:         sub x8, x0, #800
; CHECK-NOT:     sub
; CHECK:         ldp
;
; NOBASECSE-LABEL: cse_i64_load:
; NOBASECSE:        sub x8, x0, #800
; NOBASECSE:        sub x9, x0, #792
; NOBASECSE:        ldr
; NOBASECSE:        ldr
; NOBASECSE-NOT:    ldp
  %g0 = getelementptr i64, ptr %p, i64 -100
  %g1 = getelementptr i64, ptr %p, i64 -99
  %v0 = load i64, ptr %g0
  %v1 = load i64, ptr %g1
  store volatile i64 %v0, ptr undef
  store volatile i64 %v1, ptr undef
  ret void
}

define void @cse_q_load(ptr %p) {
; CHECK-LABEL: cse_q_load:
; CHECK:         sub x8, x0, #1600
; CHECK-NOT:     sub
; CHECK:         ldp
;
; NOBASECSE-LABEL: cse_q_load:
; NOBASECSE:        sub x8, x0, #1600
; NOBASECSE:        sub x9, x0, #1584
; NOBASECSE:        ldr
; NOBASECSE:        ldr
; NOBASECSE-NOT:    ldp
  %g0 = getelementptr <2 x i64>, ptr %p, i64 -100
  %g1 = getelementptr <2 x i64>, ptr %p, i64 -99
  %v0 = load <2 x i64>, ptr %g0
  %v1 = load <2 x i64>, ptr %g1
  store volatile <2 x i64> %v0, ptr undef
  store volatile <2 x i64> %v1, ptr undef
  ret void
}

define void @cse_i32_store(ptr %p, i32 %a, i32 %b) {
; CHECK-LABEL: cse_i32_store:
; CHECK:         sub x8, x0, #400
; CHECK-NOT:     sub
; CHECK:         stp
;
; NOBASECSE-LABEL: cse_i32_store:
; NOBASECSE:        sub x8, x0, #400
; NOBASECSE:        sub x9, x0, #396
; NOBASECSE:        str
; NOBASECSE:        str
; NOBASECSE-NOT:    stp
  %g0 = getelementptr i32, ptr %p, i64 -100
  %g1 = getelementptr i32, ptr %p, i64 -99
  store i32 %a, ptr %g0
  store i32 %b, ptr %g1
  ret void
}
