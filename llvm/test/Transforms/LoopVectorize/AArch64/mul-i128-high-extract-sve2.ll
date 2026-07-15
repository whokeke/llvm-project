; RUN: opt -mtriple=aarch64-linux-gnu -mattr=+sve2 -passes='default<O3>' -S < %s | FileCheck %s --check-prefix=ON
; RUN: opt -mtriple=aarch64-linux-gnu -mattr=+sve2 -passes='default<O3>' -aarch64-sve-mul-i128-high-extract-vec=0 -S < %s | FileCheck %s --check-prefix=OFF

; The high-multiply-extract loop `out[i] = (a[i]*b[i]) >> 64` should be
; vectorized on SVE2 (cost model treats the high-mul idiom as cheap), forming
; scalable <N x i128> mul + lshr, which the backend then lowers to `umulh z.d`.
; With the flag off (-aarch64-sve-mul-i128-high-extract-vec=0), the vectorizer
; bails (original scalar behavior).

define void @high_mul_loop(ptr nocapture readonly %a, ptr nocapture readonly %b, ptr nocapture writeonly %out, i32 %n) #0 {
; ON-LABEL: @high_mul_loop
; ON: mul {{.*}}<vscale x 2 x i128>
; ON: lshr <vscale x 2 x i128>
; OFF-LABEL: @high_mul_loop
; OFF-NOT: <vscale x 2 x i128>
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop, label %exit

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %ai = getelementptr inbounds i64, ptr %a, i32 %i
  %bi = getelementptr inbounds i64, ptr %b, i32 %i
  %oi = getelementptr inbounds i64, ptr %out, i32 %i
  %av = load i64, ptr %ai, align 8
  %bv = load i64, ptr %bi, align 8
  %za = zext i64 %av to i128
  %zb = zext i64 %bv to i128
  %m = mul i128 %za, %zb
  %s = lshr i128 %m, 64
  %t = trunc i128 %s to i64
  store i64 %t, ptr %oi, align 8
  %i.next = add i32 %i, 1
  %cond = icmp slt i32 %i.next, %n
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}

attributes #0 = { nounwind }
