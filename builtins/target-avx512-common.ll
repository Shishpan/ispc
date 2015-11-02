;;  Copyright (c) 2015, Intel Corporation
;;  All rights reserved.
;;
;;  Redistribution and use in source and binary forms, with or without
;;  modification, are permitted provided that the following conditions are
;;  met:
;;
;;    * Redistributions of source code must retain the above copyright
;;      notice, this list of conditions and the following disclaimer.
;;
;;    * Redistributions in binary form must reproduce the above copyright
;;      notice, this list of conditions and the following disclaimer in the
;;      documentation and/or other materials provided with the distribution.
;;
;;    * Neither the name of Intel Corporation nor the names of its
;;      contributors may be used to endorse or promote products derived from
;;      this software without specific prior written permission.
;;
;;
;;   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
;;   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
;;   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
;;   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
;;   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
;;   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
;;   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
;;   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
;;   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
;;   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
;;   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  

define(`MASK',`i1')
define(`HAVE_GATHER',`1')
define(`HAVE_SCATTER',`1')

include(`util.m4')

stdlib_core()
scans()
reduce_equal(WIDTH)
rdrand_definition()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; broadcast/rotate/shuffle

define_shuffles()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; aos/soa

aossoa()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half conversion routines

declare <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16>) nounwind readnone
declare <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float>, i32) nounwind readnone

define <16 x float> @__half_to_float_varying(<16 x i16> %v) nounwind readnone {
  %r_0 = shufflevector <16 x i16> %v, <16 x i16> undef,
             <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vr_0 = call <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16> %r_0)
  %r_1 = shufflevector <16 x i16> %v, <16 x i16> undef,
             <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vr_1 = call <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16> %r_1)
  %r = shufflevector <8 x float> %vr_0, <8 x float> %vr_1,
           <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                       i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x float> %r
}

define <16 x i16> @__float_to_half_varying(<16 x float> %v) nounwind readnone {
  %r_0 = shufflevector <16 x float> %v, <16 x float> undef,
             <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vr_0 = call <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float> %r_0, i32 0)
  %r_1 = shufflevector <16 x float> %v, <16 x float> undef,
             <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vr_1 = call <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float> %r_1, i32 0)
  %r = shufflevector <8 x i16> %vr_0, <8 x i16> %vr_1,
           <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                       i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i16> %r
}

define float @__half_to_float_uniform(i16 %v) nounwind readnone {
  %v1 = bitcast i16 %v to <1 x i16>
  %vv = shufflevector <1 x i16> %v1, <1 x i16> undef,
           <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef,
                      i32 undef, i32 undef, i32 undef, i32 undef>
  %rv = call <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16> %vv)
  %r = extractelement <8 x float> %rv, i32 0
  ret float %r
}

define i16 @__float_to_half_uniform(float %v) nounwind readnone {
  %v1 = bitcast float %v to <1 x float>
  %vv = shufflevector <1 x float> %v1, <1 x float> undef,
           <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef,
                      i32 undef, i32 undef, i32 undef, i32 undef>
  ; round to nearest even
  %rv = call <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float> %vv, i32 0)
  %r = extractelement <8 x i16> %rv, i32 0
  ret i16 %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; fast math mode

declare void @llvm.x86.sse.stmxcsr(i8 *) nounwind
declare void @llvm.x86.sse.ldmxcsr(i8 *) nounwind

define void @__fastmath() nounwind alwaysinline {
  %ptr = alloca i32
  %ptr8 = bitcast i32 * %ptr to i8 *
  call void @llvm.x86.sse.stmxcsr(i8 * %ptr8)
  %oldval = load PTR_OP_ARGS(`i32 ') %ptr

  ; turn on DAZ (64)/FTZ (32768) -> 32832
  %update = or i32 %oldval, 32832
  store i32 %update, i32 *%ptr
  call void @llvm.x86.sse.ldmxcsr(i8 * %ptr8)
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; round/floor/ceil

declare <4 x float> @llvm.x86.sse41.round.ss(<4 x float>, <4 x float>, i32) nounwind readnone

define float @__round_uniform_float(float) nounwind readonly alwaysinline {
  ; roundss, round mode nearest 0b00 | don't signal precision exceptions 0b1000 = 8
  ; the roundss intrinsic is a total mess--docs say:
  ;
  ;  __m128 _mm_round_ss (__m128 a, __m128 b, const int c)
  ;       
  ;  b is a 128-bit parameter. The lowest 32 bits are the result of the rounding function
  ;  on b0. The higher order 96 bits are copied directly from input parameter a. The
  ;  return value is described by the following equations:
  ;
  ;  r0 = RND(b0)
  ;  r1 = a1
  ;  r2 = a2
  ;  r3 = a3
  ;
  ;  It doesn't matter what we pass as a, since we only need the r0 value
  ;  here.  So we pass the same register for both.  Further, only the 0th
  ;  element of the b parameter matters
  %xi = insertelement <4 x float> undef, float %0, i32 0
  %xr = call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %xi, <4 x float> %xi, i32 8)
  %rs = extractelement <4 x float> %xr, i32 0
  ret float %rs
}

define float @__floor_uniform_float(float) nounwind readonly alwaysinline {
  ; see above for round_ss instrinsic discussion...
  %xi = insertelement <4 x float> undef, float %0, i32 0
  ; roundps, round down 0b01 | don't signal precision exceptions 0b1001 = 9
  %xr = call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %xi, <4 x float> %xi, i32 9)
  %rs = extractelement <4 x float> %xr, i32 0
  ret float %rs
}

define float @__ceil_uniform_float(float) nounwind readonly alwaysinline {
  ; see above for round_ss instrinsic discussion...
  %xi = insertelement <4 x float> undef, float %0, i32 0
  ; roundps, round up 0b10 | don't signal precision exceptions 0b1010 = 10
  %xr = call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %xi, <4 x float> %xi, i32 10)
  %rs = extractelement <4 x float> %xr, i32 0
  ret float %rs
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding doubles

declare <2 x double> @llvm.x86.sse41.round.sd(<2 x double>, <2 x double>, i32) nounwind readnone

define double @__round_uniform_double(double) nounwind readonly alwaysinline {
  %xi = insertelement <2 x double> undef, double %0, i32 0
  %xr = call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %xi, <2 x double> %xi, i32 8)
  %rs = extractelement <2 x double> %xr, i32 0
  ret double %rs
}

define double @__floor_uniform_double(double) nounwind readonly alwaysinline {
  ; see above for round_ss instrinsic discussion...
  %xi = insertelement <2 x double> undef, double %0, i32 0
  ; roundsd, round down 0b01 | don't signal precision exceptions 0b1001 = 9
  %xr = call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %xi, <2 x double> %xi, i32 9)
  %rs = extractelement <2 x double> %xr, i32 0
  ret double %rs
}

define double @__ceil_uniform_double(double) nounwind readonly alwaysinline {
  ; see above for round_ss instrinsic discussion...
  %xi = insertelement <2 x double> undef, double %0, i32 0
  ; roundsd, round up 0b10 | don't signal precision exceptions 0b1010 = 10
  %xr = call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %xi, <2 x double> %xi, i32 10)
  %rs = extractelement <2 x double> %xr, i32 0
  ret double %rs
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding floats

declare <16 x float> @llvm.nearbyint.v16f32(<16 x float> %p)
declare <16 x float> @llvm.floor.v16f32(<16 x float> %p)
declare <16 x float> @llvm.ceil.v16f32(<16 x float> %p)

define <16 x float> @__round_varying_float(<16 x float>) nounwind readonly alwaysinline {
  %res = call <16 x float> @llvm.nearbyint.v16f32(<16 x float> %0)
  ret <16 x float> %res
}

define <16 x float> @__floor_varying_float(<16 x float>) nounwind readonly alwaysinline {
  %res = call <16 x float> @llvm.floor.v16f32(<16 x float> %0)
  ret <16 x float> %res
}

define <16 x float> @__ceil_varying_float(<16 x float>) nounwind readonly alwaysinline {
  %res = call <16 x float> @llvm.ceil.v16f32(<16 x float> %0)
  ret <16 x float> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding doubles

declare <8 x double> @llvm.nearbyint.v8f64(<8 x double> %p)
declare <8 x double> @llvm.floor.v8f64(<8 x double> %p)
declare <8 x double> @llvm.ceil.v8f64(<8 x double> %p)

define <16 x double> @__round_varying_double(<16 x double>) nounwind readonly alwaysinline {
  %v0 = shufflevector <16 x double> %0, <16 x double> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v1 = shufflevector <16 x double> %0, <16 x double> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %r0 = call <8 x double> @llvm.nearbyint.v8f64(<8 x double> %v0)
  %r1 = call <8 x double> @llvm.nearbyint.v8f64(<8 x double> %v1)
  %res = shufflevector <8 x double> %r0, <8 x double> %r1, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                       i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x double> %res
}

define <16 x double> @__floor_varying_double(<16 x double>) nounwind readonly alwaysinline {
  %v0 = shufflevector <16 x double> %0, <16 x double> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v1 = shufflevector <16 x double> %0, <16 x double> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %r0 = call <8 x double> @llvm.floor.v8f64(<8 x double> %v0)
  %r1 = call <8 x double> @llvm.floor.v8f64(<8 x double> %v1)
  %res = shufflevector <8 x double> %r0, <8 x double> %r1, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                       i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x double> %res
}

define <16 x double> @__ceil_varying_double(<16 x double>) nounwind readonly alwaysinline {
  %v0 = shufflevector <16 x double> %0, <16 x double> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v1 = shufflevector <16 x double> %0, <16 x double> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %r0 = call <8 x double> @llvm.ceil.v8f64(<8 x double> %v0)
  %r1 = call <8 x double> @llvm.ceil.v8f64(<8 x double> %v1)
  %res = shufflevector <8 x double> %r0, <8 x double> %r1, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                       i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x double> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; min/max

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int64/uint64 min/max
define i64 @__max_uniform_int64(i64, i64) nounwind readonly alwaysinline {
  %c = icmp sgt i64 %0, %1
  %r = select i1 %c, i64 %0, i64 %1
  ret i64 %r
}

define i64 @__max_uniform_uint64(i64, i64) nounwind readonly alwaysinline {
  %c = icmp ugt i64 %0, %1
  %r = select i1 %c, i64 %0, i64 %1
  ret i64 %r
}

define i64 @__min_uniform_int64(i64, i64) nounwind readonly alwaysinline {
  %c = icmp slt i64 %0, %1
  %r = select i1 %c, i64 %0, i64 %1
  ret i64 %r
}

define i64 @__min_uniform_uint64(i64, i64) nounwind readonly alwaysinline {
  %c = icmp ult i64 %0, %1
  %r = select i1 %c, i64 %0, i64 %1
  ret i64 %r
}

declare <8 x i64> @llvm.x86.avx512.mask.pmaxs.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)
declare <8 x i64> @llvm.x86.avx512.mask.pmaxu.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)
declare <8 x i64> @llvm.x86.avx512.mask.pmins.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)
declare <8 x i64> @llvm.x86.avx512.mask.pminu.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)

define <16 x i64> @__max_varying_int64(<16 x i64>, <16 x i64>) nounwind readonly alwaysinline {
  %v0_lo = shufflevector <16 x i64> %0, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v0_hi = shufflevector <16 x i64> %0, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v1_lo = shufflevector <16 x i64> %1, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v1_hi = shufflevector <16 x i64> %1, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %r0 = call <8 x i64> @llvm.x86.avx512.mask.pmaxs.q.512(<8 x i64> %v0_lo, <8 x i64> %v1_lo, <8 x i64>zeroinitializer, i8 -1)
  %r1 = call <8 x i64> @llvm.x86.avx512.mask.pmaxs.q.512(<8 x i64> %v0_hi, <8 x i64> %v1_hi, <8 x i64>zeroinitializer, i8 -1)
  %res = shufflevector <8 x i64> %r0, <8 x i64> %r1, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                 i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i64> %res
}

define <16 x i64> @__max_varying_uint64(<16 x i64>, <16 x i64>) nounwind readonly alwaysinline {
  %v0_lo = shufflevector <16 x i64> %0, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v0_hi = shufflevector <16 x i64> %0, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v1_lo = shufflevector <16 x i64> %1, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v1_hi = shufflevector <16 x i64> %1, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

  %r0 = call <8 x i64> @llvm.x86.avx512.mask.pmaxu.q.512(<8 x i64> %v0_lo, <8 x i64> %v1_lo, <8 x i64>zeroinitializer, i8 -1)
  %r1 = call <8 x i64> @llvm.x86.avx512.mask.pmaxu.q.512(<8 x i64> %v0_hi, <8 x i64> %v1_hi, <8 x i64>zeroinitializer, i8 -1)
  %res = shufflevector <8 x i64> %r0, <8 x i64> %r1, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                 i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i64> %res
}

define <16 x i64> @__min_varying_int64(<16 x i64>, <16 x i64>) nounwind readonly alwaysinline {
  %v0_lo = shufflevector <16 x i64> %0, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v0_hi = shufflevector <16 x i64> %0, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v1_lo = shufflevector <16 x i64> %1, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v1_hi = shufflevector <16 x i64> %1, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

  %r0 = call <8 x i64> @llvm.x86.avx512.mask.pmins.q.512(<8 x i64> %v0_lo, <8 x i64> %v1_lo, <8 x i64>zeroinitializer, i8 -1)
  %r1 = call <8 x i64> @llvm.x86.avx512.mask.pmins.q.512(<8 x i64> %v0_hi, <8 x i64> %v1_hi, <8 x i64>zeroinitializer, i8 -1)
  %res = shufflevector <8 x i64> %r0, <8 x i64> %r1, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                 i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i64> %res
}

define <16 x i64> @__min_varying_uint64(<16 x i64>, <16 x i64>) nounwind readonly alwaysinline {
  %v0_lo = shufflevector <16 x i64> %0, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v0_hi = shufflevector <16 x i64> %0, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v1_lo = shufflevector <16 x i64> %1, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v1_hi = shufflevector <16 x i64> %1, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

  %r0 = call <8 x i64> @llvm.x86.avx512.mask.pminu.q.512(<8 x i64> %v0_lo, <8 x i64> %v1_lo, <8 x i64>zeroinitializer, i8 -1)
  %r1 = call <8 x i64> @llvm.x86.avx512.mask.pminu.q.512(<8 x i64> %v0_hi, <8 x i64> %v1_hi, <8 x i64>zeroinitializer, i8 -1)
  %res = shufflevector <8 x i64> %r0, <8 x i64> %r1, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                 i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i64> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float min/max

define float @__max_uniform_float(float, float) nounwind readonly alwaysinline {
  %cmp = fcmp ogt float %1, %0
  %ret = select i1 %cmp, float %1, float %0
  ret float %ret
}

define float @__min_uniform_float(float, float) nounwind readonly alwaysinline {
  %cmp = fcmp ogt float %1, %0
  %ret = select i1 %cmp, float %0, float %1
  ret float %ret
}

declare <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float>, <16 x float>, <16 x float>, i16, i32)
declare <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float>, <16 x float>, <16 x float>, i16, i32)

define <16 x float> @__max_varying_float(<16 x float>, <16 x float>) nounwind readonly alwaysinline {
  %res = call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> %0, <16 x float> %1, <16 x float>zeroinitializer, i16 -1, i32 4)
  ret <16 x float> %res
}

define <16 x float> @__min_varying_float(<16 x float>, <16 x float>) nounwind readonly alwaysinline {
  %res = call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> %0, <16 x float> %1, <16 x float>zeroinitializer, i16 -1, i32 4)
  ret <16 x float> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int min/max

define i32 @__min_uniform_int32(i32, i32) nounwind readonly alwaysinline {
  %cmp = icmp sgt i32 %1, %0
  %ret = select i1 %cmp, i32 %0, i32 %1
  ret i32 %ret
}

define i32 @__max_uniform_int32(i32, i32) nounwind readonly alwaysinline {
  %cmp = icmp sgt i32 %1, %0
  %ret = select i1 %cmp, i32 %1, i32 %0
  ret i32 %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unsigned int min/max

define i32 @__min_uniform_uint32(i32, i32) nounwind readonly alwaysinline {
  %cmp = icmp ugt i32 %1, %0
  %ret = select i1 %cmp, i32 %0, i32 %1
  ret i32 %ret
}

define i32 @__max_uniform_uint32(i32, i32) nounwind readonly alwaysinline {
  %cmp = icmp ugt i32 %1, %0
  %ret = select i1 %cmp, i32 %1, i32 %0
  ret i32 %ret
}

declare <16 x i32> @llvm.x86.avx512.mask.pmins.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)
declare <16 x i32> @llvm.x86.avx512.mask.pmaxs.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)

define <16 x i32> @__min_varying_int32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
  %ret = call <16 x i32> @llvm.x86.avx512.mask.pmins.d.512(<16 x i32> %0, <16 x i32> %1, 
                                                           <16 x i32> zeroinitializer, i16 -1)
  ret <16 x i32> %ret
}

define <16 x i32> @__max_varying_int32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
  %ret = call <16 x i32> @llvm.x86.avx512.mask.pmaxs.d.512(<16 x i32> %0, <16 x i32> %1,
                                                           <16 x i32> zeroinitializer, i16 -1)
  ret <16 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx512.mask.pminu.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)
declare <16 x i32> @llvm.x86.avx512.mask.pmaxu.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)

define <16 x i32> @__min_varying_uint32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
  %ret = call <16 x i32> @llvm.x86.avx512.mask.pminu.d.512(<16 x i32> %0, <16 x i32> %1,
                                                           <16 x i32> zeroinitializer, i16 -1)
  ret <16 x i32> %ret
}

define <16 x i32> @__max_varying_uint32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
  %ret = call <16 x i32> @llvm.x86.avx512.mask.pmaxu.d.512(<16 x i32> %0, <16 x i32> %1,
                                                           <16 x i32> zeroinitializer, i16 -1)
  ret <16 x i32> %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision min/max

define double @__min_uniform_double(double, double) nounwind readnone alwaysinline {
  %cmp = fcmp ogt double %1, %0
  %ret = select i1 %cmp, double %0, double %1
  ret double %ret
}

define double @__max_uniform_double(double, double) nounwind readnone alwaysinline {
  %cmp = fcmp ogt double %1, %0
  %ret = select i1 %cmp, double %1, double %0
  ret double %ret
}

declare <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double>, <8 x double>,
                    <8 x double>, i8, i32)
declare <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double>, <8 x double>,
                    <8 x double>, i8, i32)

define <16 x double> @__min_varying_double(<16 x double>, <16 x double>) nounwind readnone alwaysinline {
  %a_0 = shufflevector <16 x double> %0, <16 x double> undef,
                       <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %a_1 = shufflevector <16 x double> %1, <16 x double> undef,
                       <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %res_a = call <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double> %a_0, <8 x double> %a_1,
                <8 x double> zeroinitializer, i8 -1, i32 4)
  %b_0 = shufflevector <16 x double> %0, <16 x double> undef,
                       <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %b_1 = shufflevector <16 x double> %1, <16 x double> undef,
                       <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %res_b = call <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double> %b_0, <8 x double> %b_1,
                <8 x double> zeroinitializer, i8 -1, i32 4)
  %res = shufflevector <8 x double> %res_a, <8 x double> %res_b,
                       <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                   i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x double> %res                       
}

define <16 x double> @__max_varying_double(<16 x double>, <16 x double>) nounwind readnone alwaysinline {
  %a_0 = shufflevector <16 x double> %0, <16 x double> undef,
                       <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %a_1 = shufflevector <16 x double> %1, <16 x double> undef,
                       <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %res_a = call <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double> %a_0, <8 x double> %a_1,
                <8 x double> zeroinitializer, i8 -1, i32 4)
  %b_0 = shufflevector <16 x double> %0, <16 x double> undef,
                       <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %b_1 = shufflevector <16 x double> %1, <16 x double> undef,
                       <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %res_b = call <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double> %b_0, <8 x double> %b_1,
                <8 x double> zeroinitializer, i8 -1, i32 4)
  %res = shufflevector <8 x double> %res_a, <8 x double> %res_b,
                       <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                   i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x double> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rsqrt

declare <4 x float> @llvm.x86.sse.rsqrt.ss(<4 x float>) nounwind readnone

define float @__rsqrt_uniform_float(float) nounwind readonly alwaysinline {
  ;  uniform float is = extract(__rsqrt_u(v), 0);
  %v = insertelement <4 x float> undef, float %0, i32 0
  %vis = call <4 x float> @llvm.x86.sse.rsqrt.ss(<4 x float> %v)
  %is = extractelement <4 x float> %vis, i32 0

  ; Newton-Raphson iteration to improve precision
  ;  return 0.5 * is * (3. - (v * is) * is);
  %v_is = fmul float %0, %is
  %v_is_is = fmul float %v_is, %is
  %three_sub = fsub float 3., %v_is_is
  %is_mul = fmul float %is, %three_sub
  %half_scale = fmul float 0.5, %is_mul
  ret float %half_scale
}

declare <16 x float> @llvm.x86.avx512.rsqrt28.ps(<16 x float>, <16 x float>, i16, i32) nounwind readnone

define <16 x float> @__rsqrt_varying_float(<16 x float> %v) nounwind readonly alwaysinline {
  %res = call <16 x float> @llvm.x86.avx512.rsqrt28.ps(<16 x float> %v, <16 x float> undef, i16 -1, i32 8)
  ret <16 x float> %res  
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp

declare <4 x float> @llvm.x86.sse.rcp.ss(<4 x float>) nounwind readnone

define float @__rcp_uniform_float(float) nounwind readonly alwaysinline {
  ; do the rcpss call
  ;    uniform float iv = extract(__rcp_u(v), 0);
  ;    return iv * (2. - v * iv);
  %vecval = insertelement <4 x float> undef, float %0, i32 0
  %call = call <4 x float> @llvm.x86.sse.rcp.ss(<4 x float> %vecval)
  %scall = extractelement <4 x float> %call, i32 0

  ; do one N-R iteration to improve precision, as above
  %v_iv = fmul float %0, %scall
  %two_minus = fsub float 2., %v_iv
  %iv_mul = fmul float %scall, %two_minus
  ret float %iv_mul
}

declare <16 x float> @llvm.x86.avx512.rcp28.ps(<16 x float>, <16 x float>, i16, i32) nounwind readnone

define <16 x float> @__rcp_varying_float(<16 x float>) nounwind readonly alwaysinline {
  %res = call <16 x float> @llvm.x86.avx512.rcp28.ps(<16 x float> %0, <16 x float> undef, i16 -1, i32 8)
  ret <16 x float> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; sqrt

declare <4 x float> @llvm.x86.sse.sqrt.ss(<4 x float>) nounwind readnone

define float @__sqrt_uniform_float(float) nounwind readonly alwaysinline {
  sse_unary_scalar(ret, 4, float, @llvm.x86.sse.sqrt.ss, %0)
  ret float %ret
}

declare <16 x float> @llvm.x86.avx512.mask.sqrt.ps.512(<16 x float>, <16 x float>, i16, i32) nounwind readnone

define <16 x float> @__sqrt_varying_float(<16 x float>) nounwind readonly alwaysinline {
  %res = call <16 x float> @llvm.x86.avx512.mask.sqrt.ps.512(<16 x float> %0, <16 x float> zeroinitializer, i16 -1, i32 4)
  ret <16 x float> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision sqrt

declare <2 x double> @llvm.x86.sse2.sqrt.sd(<2 x double>) nounwind readnone

define double @__sqrt_uniform_double(double) nounwind alwaysinline {
  sse_unary_scalar(ret, 2, double, @llvm.x86.sse2.sqrt.sd, %0)
  ret double %ret
}

declare <8 x double> @llvm.x86.avx512.mask.sqrt.pd.512(<8 x double>, <8 x double>, i8, i32) nounwind readnone

define <16 x double> @__sqrt_varying_double(<16 x double>) nounwind alwaysinline {
  %v0 = shufflevector <16 x double> %0, <16 x double> undef, 
                      <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v1 = shufflevector <16 x double> %0, <16 x double> undef, 
                      <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %r0 = call <8 x double> @llvm.x86.avx512.mask.sqrt.pd.512(<8 x double> %v0,  <8 x double> zeroinitializer, i8 -1, i32 4)
  %r1 = call <8 x double> @llvm.x86.avx512.mask.sqrt.pd.512(<8 x double> %v1,  <8 x double> zeroinitializer, i8 -1, i32 4)
  %res = shufflevector <8 x double> %r0, <8 x double> %r1, 
                       <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                   i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x double> %res
}
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; bit ops

declare i32 @llvm.ctpop.i32(i32) nounwind readnone

define i32 @__popcnt_int32(i32) nounwind readonly alwaysinline {
  %call = call i32 @llvm.ctpop.i32(i32 %0)
  ret i32 %call
}

declare i64 @llvm.ctpop.i64(i64) nounwind readnone

define i64 @__popcnt_int64(i64) nounwind readonly alwaysinline {
  %call = call i64 @llvm.ctpop.i64(i64 %0)
  ret i64 %call
}
ctlztz()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; FIXME: need either to wire these up to the 8-wide SVML entrypoints,
; or, use the macro to call the 4-wide ones twice with our 8-wide
; vectors...

;; svml

include(`svml.m4')
svml_stubs(float,f,WIDTH)
svml_stubs(double,d,WIDTH)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reductions

define i64 @__movmsk(<WIDTH x i1>) nounwind readnone alwaysinline {
  %intmask = bitcast <WIDTH x i1> %0 to i16
  %res = zext i16 %intmask to i64
  ret i64 %res
}

define i1 @__any(<WIDTH x i1>) nounwind readnone alwaysinline {
  %intmask = bitcast <WIDTH x i1> %0 to i16
  %res = icmp ne i16 %intmask, 0
  ret i1 %res
}

define i1 @__all(<WIDTH x i1>) nounwind readnone alwaysinline {
  %intmask = bitcast <WIDTH x i1> %0 to i16
  %res = icmp eq i16 %intmask, 65535
  ret i1 %res
}

define i1 @__none(<WIDTH x i1>) nounwind readnone alwaysinline {
  %intmask = bitcast <WIDTH x i1> %0 to i16
  %res = icmp eq i16 %intmask, 0
  ret i1 %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal int8/16 ops

declare <2 x i64> @llvm.x86.sse2.psad.bw(<16 x i8>, <16 x i8>) nounwind readnone

define i16 @__reduce_add_int8(<16 x i8>) nounwind readnone alwaysinline {
  %rv = call <2 x i64> @llvm.x86.sse2.psad.bw(<16 x i8> %0,
                                              <16 x i8> zeroinitializer)
  %r0 = extractelement <2 x i64> %rv, i32 0
  %r1 = extractelement <2 x i64> %rv, i32 1
  %r = add i64 %r0, %r1
  %r16 = trunc i64 %r to i16
  ret i16 %r16
}

define internal <16 x i16> @__add_varying_i16(<16 x i16>,
                                  <16 x i16>) nounwind readnone alwaysinline {
  %r = add <16 x i16> %0, %1
  ret <16 x i16> %r
}

define internal i16 @__add_uniform_i16(i16, i16) nounwind readnone alwaysinline {
  %r = add i16 %0, %1
  ret i16 %r
}

define i16 @__reduce_add_int16(<16 x i16>) nounwind readnone alwaysinline {
  reduce16(i16, @__add_varying_i16, @__add_uniform_i16)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal float ops

declare <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float>, <8 x float>) nounwind readnone

define float @__reduce_add_float(<16 x float>) nounwind readonly alwaysinline {
  %va = shufflevector <16 x float> %0, <16 x float> undef,
          <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vb = shufflevector <16 x float> %0, <16 x float> undef,
          <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v1 = call <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float> %va, <8 x float> %vb)
  %v2 = call <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float> %v1, <8 x float> %v1)
  %v3 = call <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float> %v2, <8 x float> %v2)
  %scalar1 = extractelement <8 x float> %v3, i32 0
  %scalar2 = extractelement <8 x float> %v3, i32 4
  %sum = fadd float %scalar1, %scalar2
  ret float %sum
}

define float @__reduce_min_float(<16 x float>) nounwind readnone alwaysinline {
  reduce16(float, @__min_varying_float, @__min_uniform_float)
}

define float @__reduce_max_float(<16 x float>) nounwind readnone alwaysinline {
  reduce16(float, @__max_varying_float, @__max_uniform_float)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal int32 ops

define internal <16 x i32> @__add_varying_int32(<16 x i32>,
                                       <16 x i32>) nounwind readnone alwaysinline {
  %s = add <16 x i32> %0, %1
  ret <16 x i32> %s
}

define internal i32 @__add_uniform_int32(i32, i32) nounwind readnone alwaysinline {
  %s = add i32 %0, %1
  ret i32 %s
}

define i32 @__reduce_add_int32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__add_varying_int32, @__add_uniform_int32)
}

define i32 @__reduce_min_int32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__min_varying_int32, @__min_uniform_int32)
}

define i32 @__reduce_max_int32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__max_varying_int32, @__max_uniform_int32)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; horizontal uint32 ops

define i32 @__reduce_min_uint32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__min_varying_uint32, @__min_uniform_uint32)
}

define i32 @__reduce_max_uint32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__max_varying_uint32, @__max_uniform_uint32)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal double ops

declare <4 x double> @llvm.x86.avx.hadd.pd.256(<4 x double>, <4 x double>) nounwind readnone

define double @__reduce_add_double(<16 x double>) nounwind readonly alwaysinline {
  %va = shufflevector <16 x double> %0, <16 x double> undef,
         <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %vb = shufflevector <16 x double> %0, <16 x double> undef,
         <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vc = shufflevector <16 x double> %0, <16 x double> undef,
         <4 x i32> <i32 8, i32 9, i32 10, i32 11>
  %vd = shufflevector <16 x double> %0, <16 x double> undef,
         <4 x i32> <i32 12, i32 13, i32 14, i32 15>
  %vab = fadd <4 x double> %va, %vb
  %vcd = fadd <4 x double> %vc, %vd

  %sum0 = call <4 x double> @llvm.x86.avx.hadd.pd.256(<4 x double> %vab, <4 x double> %vcd)
  %sum1 = call <4 x double> @llvm.x86.avx.hadd.pd.256(<4 x double> %sum0, <4 x double> %sum0)
  %final0 = extractelement <4 x double> %sum1, i32 0
  %final1 = extractelement <4 x double> %sum1, i32 2
  %sum = fadd double %final0, %final1
  ret double %sum
}

define double @__reduce_min_double(<16 x double>) nounwind readnone alwaysinline {
  reduce16(double, @__min_varying_double, @__min_uniform_double)
}

define double @__reduce_max_double(<16 x double>) nounwind readnone alwaysinline {
  reduce16(double, @__max_varying_double, @__max_uniform_double)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal int64 ops

define internal <16 x i64> @__add_varying_int64(<16 x i64>,
                                                <16 x i64>) nounwind readnone alwaysinline {
  %s = add <16 x i64> %0, %1
  ret <16 x i64> %s
}

define internal i64 @__add_uniform_int64(i64, i64) nounwind readnone alwaysinline {
  %s = add i64 %0, %1
  ret i64 %s
}

define i64 @__reduce_add_int64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__add_varying_int64, @__add_uniform_int64)
}

define i64 @__reduce_min_int64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__min_varying_int64, @__min_uniform_int64)
}

define i64 @__reduce_max_int64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__max_varying_int64, @__max_uniform_int64)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; horizontal uint64 ops

define i64 @__reduce_min_uint64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__min_varying_uint64, @__min_uniform_uint64)
}

define i64 @__reduce_max_uint64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__max_varying_uint64, @__max_uniform_uint64)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unaligned loads/loads+broadcasts

masked_load(i8,  1)
masked_load(i16, 2)

declare <16 x i32> @llvm.x86.avx512.mask.loadu.d.512(i8*, <16 x i32>, i16)
define <16 x i32> @__masked_load_i32(i8 * %ptr, <16 x i1> %mask) nounwind alwaysinline {
  %mask_i16 = bitcast <16 x i1> %mask to i16
  %res = call <16 x i32> @llvm.x86.avx512.mask.loadu.d.512(i8* %ptr, <16 x i32> zeroinitializer, i16 %mask_i16)
  ret <16 x i32> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.loadu.q.512(i8*, <8 x i64>, i8)
define <16 x i64> @__masked_load_i64(i8 * %ptr, <16 x i1> %mask) nounwind alwaysinline {
  %mask_i16 = bitcast <16 x i1> %mask to i16
  %mask_lo_i8 = trunc i16 %mask_i16 to i8
  %mask_hi = shufflevector <16 x i1> %mask, <16 x i1> undef,
                           <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %mask_hi_i8 = bitcast <8 x i1> %mask_hi to i8
  
  %ptr_d = bitcast i8* %ptr to <16 x i64>*
  %ptr_hi = getelementptr PTR_OP_ARGS(`<16 x i64>') %ptr_d, i32 0, i32 8
  %ptr_hi_i8 = bitcast i64* %ptr_hi to i8*

  %r0 = call <8 x i64> @llvm.x86.avx512.mask.loadu.q.512(i8* %ptr, <8 x i64> zeroinitializer, i8 %mask_lo_i8)
  %r1 = call <8 x i64> @llvm.x86.avx512.mask.loadu.q.512(i8* %ptr_hi_i8, <8 x i64> zeroinitializer, i8 %mask_hi_i8)
  
  %res = shufflevector <8 x i64> %r0, <8 x i64> %r1,
                       <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                   i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i64> %res
}


declare <16 x float> @llvm.x86.avx512.mask.loadu.ps.512(i8*, <16 x float>, i16)
define <16 x float> @__masked_load_float(i8 * %ptr, <16 x i1> %mask) readonly alwaysinline {
  %mask_i16 = bitcast <16 x i1> %mask to i16
  %res = call <16 x float> @llvm.x86.avx512.mask.loadu.ps.512(i8* %ptr, <16 x float> zeroinitializer, i16 %mask_i16)
  ret <16 x float> %res
}

declare <8 x double> @llvm.x86.avx512.mask.loadu.pd.512(i8*, <8 x double>, i8)
define <16 x double> @__masked_load_double(i8 * %ptr, <16 x i1> %mask) readonly alwaysinline {
  %mask_i16 = bitcast <16 x i1> %mask to i16
  %mask_lo_i8 = trunc i16 %mask_i16 to i8
  %mask_hi = shufflevector <16 x i1> %mask, <16 x i1> undef,
                           <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %mask_hi_i8 = bitcast <8 x i1> %mask_hi to i8

  %ptr_d = bitcast i8* %ptr to <16 x double>*
  %ptr_hi = getelementptr PTR_OP_ARGS(`<16 x double>') %ptr_d, i32 0, i32 8
  %ptr_hi_i8 = bitcast double* %ptr_hi to i8*

  %r0 = call <8 x double> @llvm.x86.avx512.mask.loadu.pd.512(i8* %ptr, <8 x double> zeroinitializer, i8 %mask_lo_i8)
  %r1 = call <8 x double> @llvm.x86.avx512.mask.loadu.pd.512(i8* %ptr_hi_i8, <8 x double> zeroinitializer, i8 %mask_hi_i8)
  
  %res = shufflevector <8 x double> %r0, <8 x double> %r1, 
                       <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                   i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x double> %res
}


gen_masked_store(i8) ; llvm.x86.sse2.storeu.dq
gen_masked_store(i16)

declare void @llvm.x86.avx512.mask.storeu.d.512(i8*, <16 x i32>, i16)
define void @__masked_store_i32(<16 x i32>* nocapture, <16 x i32> %v, <16 x i1> %mask) nounwind alwaysinline {
  %mask_i16 = bitcast <16 x i1> %mask to i16
  %ptr_i8 = bitcast <16 x i32>* %0 to i8*
  call void @llvm.x86.avx512.mask.storeu.d.512(i8* %ptr_i8, <16 x i32> %v, i16 %mask_i16)
  ret void
}

declare void @llvm.x86.avx512.mask.storeu.q.512(i8*, <8 x i64>, i8)
define void @__masked_store_i64(<16 x i64>* nocapture, <16 x i64> %v, <16 x i1> %mask) nounwind alwaysinline {
  %mask_i16 = bitcast <16 x i1> %mask to i16
  %mask_lo_i8 = trunc i16 %mask_i16 to i8
  %mask_hi = shufflevector <16 x i1> %mask, <16 x i1> undef,
                           <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %mask_hi_i8 = bitcast <8 x i1> %mask_hi to i8

  %ptr_i8 = bitcast <16 x i64>* %0 to i8*
  %ptr_lo = getelementptr PTR_OP_ARGS(`<16 x i64>') %0, i32 0, i32 8
  %ptr_lo_i8 = bitcast i64* %ptr_lo to i8*

  %v_lo = shufflevector <16 x i64> %v, <16 x i64> undef,
                        <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v_hi = shufflevector <16 x i64> %v, <16 x i64> undef,
                        <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

  call void @llvm.x86.avx512.mask.storeu.q.512(i8* %ptr_i8, <8 x i64> %v_lo, i8 %mask_lo_i8)
  call void @llvm.x86.avx512.mask.storeu.q.512(i8* %ptr_lo_i8, <8 x i64> %v_hi, i8 %mask_hi_i8)
  ret void
}

declare void @llvm.x86.avx512.mask.storeu.ps.512(i8*, <16 x float>, i16 )
define void @__masked_store_float(<16 x float>* nocapture, <16 x float> %v, <16 x i1> %mask) nounwind alwaysinline {
  %mask_i16 = bitcast <16 x i1> %mask to i16
  %ptr_i8 = bitcast <16 x float>* %0 to i8*
  call void @llvm.x86.avx512.mask.storeu.ps.512(i8* %ptr_i8, <16 x float> %v, i16 %mask_i16)
  ret void
}

declare void @llvm.x86.avx512.mask.storeu.pd.512(i8*, <8 x double>, i8)
define void @__masked_store_double(<16 x double>* nocapture, <16 x double> %v, <16 x i1> %mask) nounwind alwaysinline {
  %mask_i16 = bitcast <16 x i1> %mask to i16
  %mask_lo_i8 = trunc i16 %mask_i16 to i8
  %mask_hi = shufflevector <16 x i1> %mask, <16 x i1> undef,
                           <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %mask_hi_i8 = bitcast <8 x i1> %mask_hi to i8

  %ptr_i8 = bitcast <16 x double>* %0 to i8*
  %ptr_lo = getelementptr PTR_OP_ARGS(`<16 x double>') %0, i32 0, i32 8
  %ptr_lo_i8 = bitcast double* %ptr_lo to i8*

  %v_lo = shufflevector <16 x double> %v, <16 x double> undef,
                        <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v_hi = shufflevector <16 x double> %v, <16 x double> undef,
                        <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

  call void @llvm.x86.avx512.mask.storeu.pd.512(i8* %ptr_i8, <8 x double> %v_lo, i8 %mask_lo_i8)
  call void @llvm.x86.avx512.mask.storeu.pd.512(i8* %ptr_lo_i8, <8 x double> %v_hi, i8 %mask_hi_i8)
  ret void
}

define void @__masked_store_blend_i8(<WIDTH x i8>* nocapture, <WIDTH x i8>, 
                                     <WIDTH x i1>) nounwind alwaysinline {
  %v = load PTR_OP_ARGS(`<WIDTH x i8> ')  %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x i8> %1, <WIDTH x i8> %v
  store <WIDTH x i8> %v1, <WIDTH x i8> * %0
  ret void
}

define void @__masked_store_blend_i16(<WIDTH x i16>* nocapture, <WIDTH x i16>, 
                                      <WIDTH x i1>) nounwind alwaysinline {
  %v = load PTR_OP_ARGS(`<WIDTH x i16> ')  %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x i16> %1, <WIDTH x i16> %v
  store <WIDTH x i16> %v1, <WIDTH x i16> * %0
  ret void
}

define void @__masked_store_blend_i32(<WIDTH x i32>* nocapture, <WIDTH x i32>, 
                                      <WIDTH x i1>) nounwind alwaysinline {
  call void @__masked_store_i32(<16 x i32>* %0, <16 x i32> %1, <16 x i1> %2)
  ret void
}

define void @__masked_store_blend_float(<WIDTH x float>* nocapture, <WIDTH x float>, 
                                        <WIDTH x i1>) nounwind alwaysinline {
  call void @__masked_store_float(<16 x float>* %0, <16 x float> %1, <16 x i1> %2)
  ret void
}

define void @__masked_store_blend_i64(<WIDTH x i64>* nocapture,
                            <WIDTH x i64>, <WIDTH x i1>) nounwind alwaysinline {
  call void @__masked_store_i64(<16 x i64>* %0, <16 x i64> %1, <16 x i1> %2)
  ret void
}

define void @__masked_store_blend_double(<WIDTH x double>* nocapture,
                            <WIDTH x double>, <WIDTH x i1>) nounwind alwaysinline {
  call void @__masked_store_double(<16 x double>* %0, <16 x double> %1, <16 x i1> %2)
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; gather/scatter

;; gather - i8
gen_gather(i8)

;; gather - i16
gen_gather(i16)

;; gather - i32
declare <16 x i32> @llvm.x86.avx512.gather.dpi.512(<16 x i32>, i8*, <16 x i32>, i16, i32)
define <16 x i32> 
@__gather_base_offsets32_i32(i8 * %ptr, i32 %offset_scale, <16 x i32> %offsets, <16 x i1> %vecmask) nounwind readonly alwaysinline {
  %mask = bitcast <16 x i1> %vecmask to i16
  %res =  call <16 x i32> @llvm.x86.avx512.gather.dpi.512 (<16 x i32> undef, i8* %ptr, <16 x i32> %offsets, i16 %mask, i32 %offset_scale)
  ret <16 x i32> %res
}

declare <8 x i32> @llvm.x86.avx512.gather.qpi.512 (<8 x i32>, i8*, <8 x i64>, i8, i32)
define <16 x i32> 
@__gather_base_offsets64_i32(i8 * %ptr, i32 %offset_scale, <16 x i64> %offsets, <16 x i1> %vecmask) nounwind readonly alwaysinline {
  %scalarMask = bitcast <16 x i1> %vecmask to i16 
  %scalarMask1 = trunc i16 %scalarMask to i8 
  %scalarMask2Tmp = lshr i16 %scalarMask, 8
  %scalarMask2 = trunc i16  %scalarMask2Tmp to i8 
  %offsets_lo = shufflevector <16 x i64> %offsets, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %offsets_hi = shufflevector <16 x i64> %offsets, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15> 
  %res1 = call <8 x i32> @llvm.x86.avx512.gather.qpi.512 (<8 x i32> undef, i8* %ptr, <8 x i64> %offsets_lo, i8 %scalarMask1, i32 %offset_scale)
  %res2 = call <8 x i32> @llvm.x86.avx512.gather.qpi.512 (<8 x i32> undef, i8* %ptr, <8 x i64> %offsets_hi, i8 %scalarMask2, i32 %offset_scale)
  %res = shufflevector <8 x i32> %res1, <8 x i32> %res2 , <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15> 
  ret <16 x i32> %res
}

define <16 x i32> 
@__gather32_i32(<16 x i32> %ptrs, <16 x i1> %vecmask) nounwind readonly alwaysinline {
  %res = call <16 x i32> @__gather_base_offsets32_i32(i8 * zeroinitializer, i32 1, <16 x i32> %ptrs, <16 x i1> %vecmask)
  ret <16 x i32> %res
}

define <16 x i32> 
@__gather64_i32(<16x i64> %ptrs, <16 x i1> %vecmask) nounwind readonly alwaysinline {
  %res = call <16 x i32> @__gather_base_offsets64_i32(i8 * zeroinitializer, i32 1, <16 x i64> %ptrs, <16 x i1> %vecmask)
  ret <16 x i32> %res
}

;; gather - i64
gen_gather(i64)

;; gather - float
declare <16 x float> @llvm.x86.avx512.gather.dps.512 (<16 x float>, i8*, <16 x i32>, i16, i32)
define <16 x float>
@__gather_base_offsets32_float(i8 * %ptr, i32 %offset_scale, <16 x i32> %offsets, <16 x i1> %vecmask) nounwind readonly alwaysinline {
  %mask = bitcast <16 x i1> %vecmask to i16
  %res = call <16 x float> @llvm.x86.avx512.gather.dps.512 (<16 x float> undef, i8* %ptr, <16 x i32>%offsets, i16 %mask, i32 %offset_scale)
  ret <16 x float> %res
}

declare <8 x float> @llvm.x86.avx512.gather.qps.512 (<8 x float>, i8*, <8 x i64>, i8, i32)
define <16 x float>
@__gather_base_offsets64_float(i8 * %ptr, i32 %offset_scale, <16 x i64> %offsets, <16 x i1> %vecmask) nounwind readonly alwaysinline {
  %mask = bitcast <16 x i1> %vecmask to i16
  %mask_shifted = lshr i16 %mask, 8
  %mask_lo = trunc i16 %mask to i8 
  %mask_hi = trunc i16 %mask_shifted to i8 
  %offsets_lo = shufflevector <16 x i64> %offsets, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %offsets_hi = shufflevector <16 x i64> %offsets, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15> 
  %res_lo = call <8 x float> @llvm.x86.avx512.gather.qps.512 (<8 x float> undef, i8* %ptr, <8 x i64> %offsets_lo, i8 %mask_lo, i32 %offset_scale)
  %res_hi = call <8 x float> @llvm.x86.avx512.gather.qps.512 (<8 x float> undef, i8* %ptr, <8 x i64> %offsets_hi, i8 %mask_hi, i32 %offset_scale)
  %res = shufflevector <8 x float> %res_lo, <8 x float> %res_hi, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15> 
  ret <16 x float> %res
}

define <16 x float> 
@__gather32_float(<16 x i32> %ptrs, <16 x i1> %vecmask) nounwind readonly alwaysinline {
  %res = call <16 x float> @__gather_base_offsets32_float(i8 * zeroinitializer, i32 1, <16 x i32> %ptrs, <16 x i1> %vecmask)
  ret <16 x float> %res
}

define <16 x float> 
@__gather64_float(<16 x i64> %ptrs,  <16 x i1> %vecmask) nounwind readonly alwaysinline {
  %res = call <16 x float> @__gather_base_offsets64_float(i8 * zeroinitializer, i32 1, <16 x i64> %ptrs, <16 x i1> %vecmask)
  ret <16 x float> %res
}

;; gather - double
gen_gather(double)


define(`scatterbo32_64', `
define void @__scatter_base_offsets32_$1(i8* %ptr, i32 %scale, <WIDTH x i32> %offsets,
                                         <WIDTH x $1> %vals, <WIDTH x i1> %mask) nounwind {
  call void @__scatter_factored_base_offsets32_$1(i8* %ptr, <16 x i32> %offsets,
      i32 %scale, <16 x i32> zeroinitializer, <16 x $1> %vals, <WIDTH x i1> %mask)
  ret void
}

define void @__scatter_base_offsets64_$1(i8* %ptr, i32 %scale, <WIDTH x i64> %offsets,
                                         <WIDTH x $1> %vals, <WIDTH x i1> %mask) nounwind {
  call void @__scatter_factored_base_offsets64_$1(i8* %ptr, <16 x i64> %offsets,
      i32 %scale, <16 x i64> zeroinitializer, <16 x $1> %vals, <WIDTH x i1> %mask)
  ret void
} 
')

;; scatter - i8
scatterbo32_64(i8)
gen_scatter(i8)

;; scatter - i16
scatterbo32_64(i16)
gen_scatter(i16)

;; scatter - i32
declare void @llvm.x86.avx512.scatter.dpi.512 (i8*, i16, <16 x i32>, <16 x i32>, i32)
define void 
@__scatter_base_offsets32_i32(i8* %ptr, i32 %offset_scale, <16 x i32> %offsets, <16 x i32> %vals, <16 x i1> %vecmask) nounwind {
  %mask = bitcast <16 x i1> %vecmask to i16
  call void @llvm.x86.avx512.scatter.dpi.512 (i8* %ptr, i16 %mask, <16 x i32> %offsets, <16 x i32> %vals, i32 %offset_scale)
  ret void
}

declare void @llvm.x86.avx512.scatter.qpi.512 (i8*, i8, <8 x i64>, <8 x i32>, i32)
define void 
@__scatter_base_offsets64_i32(i8* %ptr, i32 %offset_scale, <16 x i64> %offsets, <16 x i32> %vals, <16 x i1> %vecmask) nounwind {
  %mask = bitcast <16 x i1> %vecmask to i16
  %mask_shifted = lshr i16 %mask, 8
  %mask_lo = trunc i16 %mask to i8 
  %mask_hi = trunc i16 %mask_shifted to i8 
  %offsets_lo = shufflevector <16 x i64> %offsets, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %offsets_hi = shufflevector <16 x i64> %offsets, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15> 
  %res_lo = shufflevector <16 x i32> %vals, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %res_hi = shufflevector <16 x i32> %vals, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15> 
  call void @llvm.x86.avx512.scatter.qpi.512 (i8* %ptr, i8 %mask_lo, <8 x i64> %offsets_lo, <8 x i32> %res_lo, i32 %offset_scale)
  call void @llvm.x86.avx512.scatter.qpi.512 (i8* %ptr, i8 %mask_hi, <8 x i64> %offsets_hi, <8 x i32> %res_hi, i32 %offset_scale)
  ret void
} 

define void 
@__scatter32_i32(<16 x i32> %ptrs, <16 x i32> %values, <16 x i1> %vecmask) nounwind alwaysinline {
  call void @__scatter_base_offsets32_i32(i8 * zeroinitializer, i32 1, <16 x i32> %ptrs, <16 x i32> %values, <16 x i1> %vecmask)
  ret void
}

define void 
@__scatter64_i32(<16 x i64> %ptrs, <16 x i32> %values, <16 x i1> %vecmask) nounwind alwaysinline {
  call void @__scatter_base_offsets64_i32(i8 * zeroinitializer, i32 1, <16 x i64> %ptrs, <16 x i32> %values, <16 x i1> %vecmask)
  ret void
}

;; scatter - i64
scatterbo32_64(i64)
gen_scatter(i64)

;; scatter - float
declare void @llvm.x86.avx512.scatter.dps.512 (i8*, i16, <16 x i32>, <16 x float>, i32)
define void 
@__scatter_base_offsets32_float(i8* %ptr, i32 %offset_scale, <16 x i32> %offsets, <16 x float> %vals, <16 x i1> %vecmask) nounwind {
  %mask = bitcast <16 x i1> %vecmask to i16
  call void @llvm.x86.avx512.scatter.dps.512 (i8* %ptr, i16 %mask, <16 x i32> %offsets, <16 x float> %vals, i32 %offset_scale)
  ret void
}

declare void @llvm.x86.avx512.scatter.qps.512 (i8*, i8, <8 x i64>, <8 x float>, i32)
define void 
@__scatter_base_offsets64_float(i8* %ptr, i32 %offset_scale, <16 x i64> %offsets, <16 x float> %vals, <16 x i1> %vecmask) nounwind {
  %mask = bitcast <16 x i1> %vecmask to i16
  %mask_shifted = lshr i16 %mask, 8
  %mask_lo = trunc i16 %mask to i8 
  %mask_hi = trunc i16 %mask_shifted to i8 
  %offsets_lo = shufflevector <16 x i64> %offsets, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %offsets_hi = shufflevector <16 x i64> %offsets, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15> 
  %res_lo = shufflevector <16 x float> %vals, <16 x float> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %res_hi = shufflevector <16 x float> %vals, <16 x float> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15> 
  call void @llvm.x86.avx512.scatter.qps.512 (i8* %ptr, i8 %mask_lo, <8 x i64> %offsets_lo, <8 x float> %res_lo, i32 %offset_scale)
  call void @llvm.x86.avx512.scatter.qps.512 (i8* %ptr, i8 %mask_hi, <8 x i64> %offsets_hi, <8 x float> %res_hi, i32 %offset_scale)
  ret void
} 

define void 
@__scatter32_float(<16 x i32> %ptrs, <16 x float> %values, <16 x i1> %vecmask) nounwind alwaysinline {
  call void @__scatter_base_offsets32_float(i8 * zeroinitializer, i32 1, <16 x i32> %ptrs, <16 x float> %values, <16 x i1> %vecmask)
  ret void
}

define void 
@__scatter64_float(<16 x i64> %ptrs, <16 x float> %values, <16 x i1> %vecmask) nounwind alwaysinline {
  call void @__scatter_base_offsets64_float(i8 * zeroinitializer, i32 1, <16 x i64> %ptrs, <16 x float> %values, <16 x i1> %vecmask)
  ret void
}

;; scatter - double
scatterbo32_64(double)
gen_scatter(double)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; packed_load/store

declare <16 x i32> @llvm.x86.avx512.mask.expand.load.d.512(i8* %addr, <16 x i32> %data, i16 %mask)

define i32 @__packed_load_active(i32 * %startptr, <16 x i32> * %val_ptr,
                                 <16 x i1> %full_mask) nounwind alwaysinline {
  %addr = bitcast i32* %startptr to i8*
  %data = load PTR_OP_ARGS(`<16 x i32> ') %val_ptr
  %mask = bitcast <16 x i1> %full_mask to i16
  %store_val = call <16 x i32> @llvm.x86.avx512.mask.expand.load.d.512(i8* %addr, <16 x i32> %data, i16 %mask)
  store <16 x i32> %store_val, <16 x i32> * %val_ptr
  %mask_i32 = zext i16 %mask to i32
  %res = call i32 @llvm.ctpop.i32(i32 %mask_i32)
  ret i32 %res
}

declare void @llvm.x86.avx512.mask.compress.store.d.512(i8* %addr, <16 x i32> %data, i16 %mask)

define i32 @__packed_store_active(i32 * %startptr, <16 x i32> %vals,
                                   <16 x i1> %full_mask) nounwind alwaysinline {
  %addr = bitcast i32* %startptr to i8*
  %mask = bitcast <16 x i1> %full_mask to i16
  call void @llvm.x86.avx512.mask.compress.store.d.512(i8* %addr, <16 x i32> %vals, i16 %mask)
  %mask_i32 = zext i16 %mask to i32
  %res = call i32 @llvm.ctpop.i32(i32 %mask_i32)
  ret i32 %res
}

define i32 @__packed_store_active2(i32 * %startptr, <16 x i32> %vals,
                                   <16 x i1> %full_mask) nounwind alwaysinline {
  %res = call i32 @__packed_store_active(i32 * %startptr, <16 x i32> %vals,
                                   <16 x i1> %full_mask)
  ret i32 %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; prefetch

define_prefetches()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int8/int16 builtins

define_avgs()
declare_nvptx()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reciprocals in double precision, if supported

rsqrtd_decl()
rcpd_decl()

declare float @__log_uniform_float(float) nounwind readnone
;;    declare <WIDTH x float> @__log_varying_float(<WIDTH x float>) nounwind readnone
    declare float @__exp_uniform_float(float) nounwind readnone
;;    declare <WIDTH x float> @__exp_varying_float(<WIDTH x float>) nounwind readnone
    declare float @__pow_uniform_float(float, float) nounwind readnone
    declare <WIDTH x float> @__pow_varying_float(<WIDTH x float>, <WIDTH x float>) nounwind readnone

    declare double @__log_uniform_double(double) nounwind readnone
;;    declare <WIDTH x double> @__log_varying_double(<WIDTH x double>) nounwind readnone
    declare double @__exp_uniform_double(double) nounwind readnone
;;    declare <WIDTH x double> @__exp_varying_double(<WIDTH x double>) nounwind readnone
    declare double @__pow_uniform_double(double, double) nounwind readnone
    declare <WIDTH x double> @__pow_varying_double(<WIDTH x double>, <WIDTH x double>) nounwind readnone

declare <16 x float> @llvm.x86.avx512.exp2.ps(<16 x float>, <16 x float>, i16, i32) nounwind readnone
declare <8 x double> @llvm.x86.avx512.exp2.pd(<8 x double>, <8 x double>, i8, i32) nounwind readnone
declare <16 x float> @llvm.x86.avx512.mask.getexp.ps.512(<16 x float>, <16 x float>, i16, i32) nounwind readnone
declare <8 x double> @llvm.x86.avx512.mask.getexp.pd.512(<8 x double>, <8 x double>, i8, i32) nounwind readnone
declare <16 x float> @llvm.x86.avx512.mask.getmant.ps.512(<16 x float>, i32, <16 x float>, i16, i32)
declare <8 x double> @llvm.x86.avx512.mask.getmant.pd.512(<8 x double>, i32, <8 x double>, i8, i32)
declare <16 x float> @llvm.x86.avx512.mask.vfmadd.ps.512(<16 x float>, <16 x float>, <16 x float>, i16, i32)
declare <8 x double> @llvm.x86.avx512.mask.mul.pd.512(<8 x double>, <8 x double>, <8 x double>, i8, i32)
declare <8 x i64> @llvm.x86.avx512.mask.psrli.q(<8 x i64>, i32, <8 x i64>, i8) nounwind readnone
declare <16 x float> @llvm.x86.avx512.mask.vpermilvar.ps.512(<16 x float>, <16 x i32>, <16 x float>, i16)
declare <8 x double> @llvm.x86.avx512.mask.rndscale.pd.512(<8 x double>, i32, <8 x double>, i8, i32)
declare <8 x double> @llvm.x86.avx512.mask.vfmadd.pd.512(<8 x double>, <8 x double>, <8 x double>, i8, i32)
declare <8 x double> @llvm.x86.avx512.mask3.vfmsub.pd.512(<8 x double>, <8 x double>, <8 x double>, i8, i32)
declare <8 x double> @llvm.x86.avx512.maskz.vpermt2var.pd.512(<8 x i64>, <8 x double>, <8 x double>, i8)
declare <8 x double> @llvm.x86.avx512.rcp14.pd.512(<8 x double>, <8 x double>, i8) nounwind readnone
define <16 x float> @__exp_varying_float(<16 x float> %x_full) nounwind readnone {
    %ptr_for_constants = alloca <16 x float>
    store <16 x float> <float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000>, <16 x float>* %ptr_for_constants
    %rcp_ln2 = load PTR_OP_ARGS(`<16 x float> ') %ptr_for_constants
    %x = fmul <16 x float> %x_full, %rcp_ln2
    %res = call <16 x float> @llvm.x86.avx512.exp2.ps(<16 x float> %x, <16 x float> zeroinitializer, i16 -1, i32 8)
    ret <16 x float> %res
}
define <16 x double> @__exp_varying_double(<16 x double> %x_full) nounwind readnone {
    %ptr_for_constants = alloca <16 x double>
    store <16 x double> <double 0x3FF71547652B82FE, double 0x3FF71547652B82FE, double 0x3FF71547652B82FE, double 0x3FF71547652B82FE, double 0x3FF71547652B82FE, double 0x3FF71547652B82FE, double 0x3FF71547652B82FE, double 0x3FF71547652B82FE, double 0x3FF71547652B82FE, double 0x3FF71547652B82FE, double 0x3FF71547652B82FE, double 0x3FF71547652B82FE, double 0x3FF71547652B82FE, double 0x3FF71547652B82FE, double 0x3FF71547652B82FE, double 0x3FF71547652B82FE>, <16 x double>* %ptr_for_constants
    %rcp_ln2 = load PTR_OP_ARGS(`<16 x double> ') %ptr_for_constants
    %x = fmul <16 x double> %x_full, %rcp_ln2
    %x_part1 = shufflevector <16 x double> %x, <16 x double> undef, <8 x i32> <i32 0,i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
    %x_part2 = shufflevector <16 x double> %x, <16 x double> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
    %res1 = call <8 x double> @llvm.x86.avx512.exp2.pd(<8 x double> %x_part1, <8 x double> zeroinitializer, i8 -1, i32 8)
    %res2 = call <8 x double> @llvm.x86.avx512.exp2.pd(<8 x double> %x_part2, <8 x double> zeroinitializer, i8 -1, i32 8)
    %res = shufflevector <8 x double> %res1, <8 x double> %res2, <16 x i32> <i32 0,i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
    ret <16 x double> %res
}
define <16 x float> @__log_varying_float(<16 x float> %x_full) nounwind readnone {
    %exp = call <16 x float> @llvm.x86.avx512.mask.getexp.ps.512(<16 x float> %x_full, <16 x float> zeroinitializer, i16 -1, i32 4)
    %mant = call <16 x float> @llvm.x86.avx512.mask.getmant.ps.512(<16 x float> %x_full, i32 11, <16 x float> %x_full, i16 -1, i32 4)
    %exp_of_mant = call <16 x float> @llvm.x86.avx512.mask.getexp.ps.512(<16 x float> %mant, <16 x float> zeroinitializer, i16 -1, i32 4)
    %exp_final = fsub <16 x float> %exp, %exp_of_mant
    %real_mant = fmul <16 x float> %mant, <float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0>
    %arg = fsub <16 x float> %mant, <float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0>
    %ptr_for_constants = alloca <16 x float>
    store <16 x float> <float 0x3FE62E4300000000, float 0x3FE62E4300000000, float 0x3FE62E4300000000, float 0x3FE62E4300000000, float 0x3FE62E4300000000, float 0x3FE62E4300000000, float 0x3FE62E4300000000, float 0x3FE62E4300000000, float 0x3FE62E4300000000, float 0x3FE62E4300000000, float 0x3FE62E4300000000, float 0x3FE62E4300000000, float 0x3FE62E4300000000, float 0x3FE62E4300000000, float 0x3FE62E4300000000, float 0x3FE62E4300000000>, <16 x float>* %ptr_for_constants
    %ln2 = load PTR_OP_ARGS(`<16 x float> ') %ptr_for_constants
   
    %mant_i32 = bitcast <16 x float> %mant to <16 x i32>
    %mant_16xi32 = lshr <16 x i32> %mant_i32, <i32 19, i32 19, i32 19, i32 19, i32 19, i32 19, i32 19, i32 19, i32 19, i32 19, i32 19, i32 19, i32 19, i32 19, i32 19, i32 19>    
    %mant_16xi32_zero = and <16 x i32> %mant_16xi32, <i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15>


    store <16 x float> <float 0xBFCD063DE0000000, float 0xBFC80811C0000000, float 0xBFC419C4A0000000, float 0xBFC0F768A0000000, float 0xBFBCE0AE20000000, float 0xBFB8C23E20000000, float 0xBFB55F10C0000000, float 0xBFB28FCF00000000, float 0xBFE2BDD1A0000000, float 0xBFE0817AE0000000, float 0xBFDD362E40000000, float 0xBFD9F67EE0000000, float 0xBFD72B7560000000, float 0xBFD4C18260000000, float 0xBFD2A8DF60000000, float 0xBFD0D4BC00000000>, <16 x float>* %ptr_for_constants
    %c1 = load PTR_OP_ARGS(`<16 x float> ') %ptr_for_constants
    store <16 x float> <float 0x3FD546D760000000, float 0x3FD4D16720000000, float 0x3FD4168060000000, float 0x3FD3362F20000000, float 0x3FD24490E0000000, float 0x3FD14E3C00000000, float 0x3FD05AF8C0000000, float 0x3FCEDEEEC0000000, float 0x3FC8D09FC0000000, float 0x3FCEB1E7A0000000, float 0x3FD17D52A0000000, float 0x3FD3042620000000, float 0x3FD4111880000000, float 0x3FD4BF91A0000000, float 0x3FD524C160000000, float 0x3FD5511F40000000>, <16 x float>* %ptr_for_constants
    %c2 = load PTR_OP_ARGS(`<16 x float> ') %ptr_for_constants
    store <16 x float> <float 0xBFDFFFD260000000, float 0xBFDFF85160000000, float 0xBFDFE0EE60000000, float 0xBFDFB6ED60000000, float 0xBFDF7A9F60000000, float 0xBFDF2DC400000000, float 0xBFDED2AC20000000, float 0xBFDE6BC520000000, float 0xBFE0C1E500000000, float 0xBFE06F58C0000000, float 0xBFE03BC440000000, float 0xBFE01D1B40000000, float 0xBFE00C3400000000, float 0xBFE003F480000000, float 0xBFE000BE20000000, float 0xBFE0000360000000>, <16 x float>* %ptr_for_constants
    %c3 = load PTR_OP_ARGS(`<16 x float> ') %ptr_for_constants
    store <16 x float> <float 0x3FF0000000000000, float 0x3FEFFFEAA0000000, float 0x3FEFFF6C80000000, float 0x3FEFFE1B60000000, float 0x3FEFFB9780000000, float 0x3FEFF796A0000000, float 0x3FEFF1E580000000, float 0x3FEFEA6580000000, float 0x3FEFF41120000000, float 0x3FEFFA1B80000000, float 0x3FEFFD5880000000, float 0x3FEFFEF3A0000000, float 0x3FEFFFA980000000, float 0x3FEFFFEC40000000, float 0x3FEFFFFDE0000000, float 0x3FF0000000000000>, <16 x float>* %ptr_for_constants
    %c4 = load PTR_OP_ARGS(`<16 x float> ') %ptr_for_constants

    forloop(i, 1, 16, `   %mant_16xi32_`'i = extractelement <16 x i32> %mant_16xi32_zero, i32 eval(i-1)
')
    forloop(J, 1, 4, `
    %c`'J`'_elem_1 = extractelement <16 x float> %c`'J, i32 %mant_16xi32_1
    %c`'J`'_final_1 = insertelement <16 x float> %c`'J, float %c`'J`'_elem_1, i32 0
    forloop(i, 2, 16, `
    %c`'J`'_elem_`'i = extractelement <16 x float> %c`'J, i32 %mant_16xi32_`'i
    %c`'J`'_final_`'i = insertelement <16 x float> %c`'J`'_final_`'eval(i-1), float %c`'J`'_elem_`'i, i32 eval(i-1)
')
')
    %x_0 = fmul <16 x float> %exp_final, %ln2
    %iter1 = call <16 x float> @llvm.x86.avx512.mask.vfmadd.ps.512(<16 x float> %arg, <16 x float> %c1_final_16, <16 x float> %c2_final_16, i16 -1, i32 4) nounwind
    %iter2 = call <16 x float> @llvm.x86.avx512.mask.vfmadd.ps.512(<16 x float> %arg, <16 x float> %iter1, <16 x float> %c3_final_16, i16 -1, i32 4) nounwind 
    %iter3 = call <16 x float> @llvm.x86.avx512.mask.vfmadd.ps.512(<16 x float> %arg, <16 x float> %iter2, <16 x float> %c4_final_16, i16 -1, i32 4) nounwind
    %res = call <16 x float> @llvm.x86.avx512.mask.vfmadd.ps.512(<16 x float> %arg, <16 x float> %iter3, <16 x float> %x_0, i16 -1, i32 4) nounwind
    ret <16 x float> %res
    }
define <16 x double> @__log_varying_double(<16 x double> %x_full) nounwind readnone {
    %x1 = shufflevector <16 x double> %x_full, <16 x double> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
    %x2 = shufflevector <16 x double> %x_full, <16 x double> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
    %ptr_for_constants = alloca <8 x double>

    %mant = call <8 x double> @llvm.x86.avx512.mask.getmant.pd.512(<8 x double> %x1, i32 8, <8 x double> %x1, i8 -1, i32 8)
    %exp = call <8 x double> @llvm.x86.avx512.mask.getexp.pd.512(<8 x double> %x1,  <8 x double> zeroinitializer, i8 -1, i32 4)
    %rcp_mant = call <8 x double> @llvm.x86.avx512.rcp14.pd.512(<8 x double> %mant, <8 x double> zeroinitializer, i8 -1) 
    store <8 x double> <double 0x3ff0000000000000, double 0x3ff0000000000000, double 0x3ff0000000000000, double 0x3ff0000000000000, double 0x3ff0000000000000, double 0x3ff0000000000000, double 0x3ff0000000000000, double 0x3ff0000000000000>, <8 x double>* %ptr_for_constants
    %c1 = load PTR_OP_ARGS(`<8 x double> ') %ptr_for_constants
    store <8 x double> <double 0x3fbc81cd309d7c70, double 0x3fbc81cd309d7c70, double 0x3fbc81cd309d7c70, double 0x3fbc81cd309d7c70, double 0x3fbc81cd309d7c70, double 0x3fbc81cd309d7c70, double 0x3fbc81cd309d7c70, double 0x3fbc81cd309d7c70>, <8 x double>* %ptr_for_constants
    %c2 = load PTR_OP_ARGS(`<8 x double> ') %ptr_for_constants
    store <8 x double> <double 0x3fc249229cee81ef, double 0x3fc249229cee81ef, double 0x3fc249229cee81ef, double 0x3fc249229cee81ef, double 0x3fc249229cee81ef, double 0x3fc249229cee81ef, double 0x3fc249229cee81ef, double 0x3fc249229cee81ef>, <8 x double>* %ptr_for_constants
    %c3 = load PTR_OP_ARGS(`<8 x double> ') %ptr_for_constants
    store <8 x double> <double 0x3fc9999999cc9f5c, double 0x3fc9999999cc9f5c, double 0x3fc9999999cc9f5c, double 0x3fc9999999cc9f5c, double 0x3fc9999999cc9f5c, double 0x3fc9999999cc9f5c, double 0x3fc9999999cc9f5c, double 0x3fc9999999cc9f5c>, <8 x double>* %ptr_for_constants
    %c4 = load PTR_OP_ARGS(`<8 x double> ') %ptr_for_constants
    store <8 x double> <double 0x3fd5555555555466, double 0x3fd5555555555466, double 0x3fd5555555555466, double 0x3fd5555555555466, double 0x3fd5555555555466, double 0x3fd5555555555466, double 0x3fd5555555555466, double 0x3fd5555555555466>, <8 x double>* %ptr_for_constants
    %c5 = load PTR_OP_ARGS(`<8 x double> ') %ptr_for_constants
    store <8 x double> <double 0x0000000000000000, double 0xbfaf0a30c01162a6, double 0xbfbe27076e2af2e6, double 0xbfc5ff3070a793d4, double 0xbfcc8ff7c79a9a22, double 0xbfd1675cababa60e, double 0xbfd4618bc21c5ec2, double 0xbfd739d7f6bbd007>, <8 x double>* %ptr_for_constants
    %c6 = load PTR_OP_ARGS(`<8 x double> ') %ptr_for_constants
    %after_scale = call <8 x double> @llvm.x86.avx512.mask.rndscale.pd.512(<8 x double> %rcp_mant, i32 88, <8 x double> %rcp_mant, i8 -1, i32 4)
    %after_sub =  call <8 x double> @llvm.x86.avx512.mask3.vfmsub.pd.512(<8 x double> %after_scale, <8 x double> %mant, <8 x double> %c1, i8 -1, i32 4)
    %after_scale_i64 = bitcast <8 x double> %after_scale to <8 x i64>
    %after_shift = call <8 x i64> @llvm.x86.avx512.mask.psrli.q(<8 x i64> %after_scale_i64, i32 48, <8 x i64> %after_scale_i64, i8 -1)
    %comp = fcmp olt <8 x double> %after_scale, <double 0x3fe8000000000000, double 0x3fe8000000000000, double 0x3fe8000000000000, double 0x3fe8000000000000, double 0x3fe8000000000000, double 0x3fe8000000000000, double 0x3fe8000000000000, double 0x3fe8000000000000>

    %after_perm = call <8 x double> @llvm.x86.avx512.maskz.vpermt2var.pd.512(<8 x i64> %after_shift, <8 x double> <double 0x3fd269621134db92, double 0x3fcf991c6cb3b379, double 0x3fca93ed3c8ad9e3, double 0x3fc5bf406b543db2, double 0x3fc1178e8227e47c, double 0x3fb9335e5d594989, double 0x3fb08598b59e3a07, double 0x3fa0415d89e74444>, <8 x double> %c6, i8 127)
    %vadd = fadd <8 x double> %exp, <double 0x3ff0000000000000, double 0x3ff0000000000000, double 0x3ff0000000000000, double 0x3ff0000000000000, double 0x3ff0000000000000, double 0x3ff0000000000000, double 0x3ff0000000000000, double 0x3ff0000000000000>
    %vadd_masked = select <8 x i1> %comp, <8 x double> %vadd, <8 x double> %exp
    %vmul = call <8 x double> @llvm.x86.avx512.mask.mul.pd.512(<8 x double> %after_sub, <8 x double> %after_sub, <8 x double> zeroinitializer, i8 -1, i32 0)
    %iter1 = call <8 x double> @llvm.x86.avx512.mask.vfmadd.pd.512(<8 x double> %after_sub, <8 x double> %c2, <8 x double> <double 0xbfc007357e93af62, double 0xbfc007357e93af62, double 0xbfc007357e93af62, double 0xbfc007357e93af62, double 0xbfc007357e93af62, double 0xbfc007357e93af62, double 0xbfc007357e93af62, double 0xbfc007357e93af62>, i8 -1, i32 4)
    %iter2 = call <8 x double> @llvm.x86.avx512.mask.vfmadd.pd.512(<8 x double> %after_sub, <8 x double> %c3, <8 x double> <double 0xbfc55553fb28db06, double 0xbfc55553fb28db06, double 0xbfc55553fb28db06, double 0xbfc55553fb28db06, double 0xbfc55553fb28db06, double 0xbfc55553fb28db06, double 0xbfc55553fb28db06, double 0xbfc55553fb28db06>, i8 -1, i32 4)
    %iter3 = call <8 x double> @llvm.x86.avx512.mask.vfmadd.pd.512(<8 x double> %after_sub, <8 x double> %c4, <8 x double> <double 0xbfd00000000c05bd, double 0xbfd00000000c05bd, double 0xbfd00000000c05bd, double 0xbfd00000000c05bd, double 0xbfd00000000c05bd, double 0xbfd00000000c05bd, double 0xbfd00000000c05bd, double 0xbfd00000000c05bd>, i8 -1, i32 4) 
    %iter4 = call <8 x double> @llvm.x86.avx512.mask.vfmadd.pd.512(<8 x double> %after_sub, <8 x double> %c5, <8 x double> <double 0xbfdfffffffffffc6, double 0xbfdfffffffffffc6, double 0xbfdfffffffffffc6, double 0xbfdfffffffffffc6, double 0xbfdfffffffffffc6, double 0xbfdfffffffffffc6, double 0xbfdfffffffffffc6, double 0xbfdfffffffffffc6>, i8 -1, i32 4)
    %iter5 = call <8 x double> @llvm.x86.avx512.mask.vfmadd.pd.512(<8 x double> %vmul, <8 x double> %iter1, <8 x double> %iter2, i8 -1, i32 4)
    %vmul_last = call <8 x double> @llvm.x86.avx512.mask.mul.pd.512(<8 x double> %vmul, <8 x double> %vmul, <8 x double> zeroinitializer, i8 -1, i32 0)
    %iter6 = call <8 x double> @llvm.x86.avx512.mask.vfmadd.pd.512(<8 x double> %vmul, <8 x double> %iter3, <8 x double> %iter4, i8 -1, i32 4)
    %iter7 = call <8 x double> @llvm.x86.avx512.mask.vfmadd.pd.512(<8 x double> %vmul_last, <8 x double> %iter5, <8 x double> %iter3, i8 -1, i32 4)    
    %iter8 = call <8 x double> @llvm.x86.avx512.mask.vfmadd.pd.512(<8 x double> %vmul, <8 x double> %iter7, <8 x double> %after_sub, i8 -1, i32 4)
    %vadd_last = fadd <8 x double> %iter8, %after_perm
    %res_1 = call <8 x double> @llvm.x86.avx512.mask.vfmadd.pd.512(<8 x double> %vadd_masked, <8 x double> <double 0x3fe62e42fefa39ef, double 0x3fe62e42fefa39ef, double 0x3fe62e42fefa39ef, double 0x3fe62e42fefa39ef, double 0x3fe62e42fefa39ef, double 0x3fe62e42fefa39ef, double 0x3fe62e42fefa39ef, double 0x3fe62e42fefa39ef>, <8 x double> %vadd_last, i8 -1, i32 4)

        %tmp = uitofp <8 x i64> %after_shift to <8 x double>
    %res = shufflevector <8 x double> %res_1, <8 x double> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
    ret <16 x double> %res
}
trigonometry_decl()
