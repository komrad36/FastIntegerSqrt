/*******************************************************************
*
*	Author: Kareem Omar
*	kareem.h.omar@gmail.com
*	https://github.com/komrad36
*
*	Last updated Feb 24, 2021
*******************************************************************/

// Fastest implementations of 32-bit and 64-bit integer square roots for x86,
// and querying for perfect squares, by a significant margin.
//
// The square roots truncate, i.e. round down (IntegerSqrt(8) -> 2).
//
// AVX recommended. SSE3 required. haddpd can be replaced with permilpd/addsd to drop the
// requirement down to SSE2.
//
// The I32/U32 versions work regardless of the FPU rounding mode.
// The I64/U64 versions require the FPU to be in round-to-nearest mode (which is the default).
//

#pragma once

#include <cstdint>
#include <immintrin.h>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

static inline __m128d _FastIntegerSqrtInternal_I32ToSd(int32_t x)
{
#ifdef __clang__
    __m128d v;
#ifdef __AVX__
    asm("vxorpd %[v], %[v], %[v]\n\
         vcvtsi2sd %[x], %[v], %[v]"
        : [v] "=x" (v)
        : [x] "r" (x)
    );
#else
    asm("xorpd %[v], %[v]\n\
         cvtsi2sd %[x], %[v]"
        : [v] "=x" (v)
        : [x] "r" (x)
    );
#endif
    return v;
#else
    return _mm_cvtsi32_sd(_mm_setzero_pd(), x);
#endif
}

static inline __m128d _FastIntegerSqrtInternal_I64ToSd(int64_t x)
{
#ifdef __clang__
    __m128d v;
#ifdef __AVX__
    asm("vxorpd %[v], %[v], %[v]\n\
         vcvtsi2sd %[x], %[v], %[v]"
        : [v] "=x" (v)
        : [x] "r" (x)
    );
#else
    asm("xorpd %[v], %[v]\n\
         cvtsi2sd %[x], %[v]"
        : [v] "=x" (v)
        : [x] "r" (x)
    );
#endif
    return v;
#else
    return _mm_cvtsi64_sd(_mm_setzero_pd(), x);
#endif
}

static inline __m128 _FastIntegerSqrtInternal_I32ToSs(int32_t x)
{
#ifdef __clang__
    __m128 v;
#ifdef __AVX__
    asm("vxorps %[v], %[v], %[v]\n\
         vcvtsi2ss %[x], %[v], %[v]"
        : [v] "=x" (v)
        : [x] "r" (x)
    );
#else
    asm("xorps %[v], %[v]\n\
         cvtsi2ss %[x], %[v]"
        : [v] "=x" (v)
        : [x] "r" (x)
    );
#endif
    return v;
#else
    return _mm_cvtsi32_ss(_mm_setzero_ps(), x);
#endif
}

static inline __m128 _FastIntegerSqrtInternal_I64ToSs(int64_t x)
{
#ifdef __clang__
    __m128 v;
#ifdef __AVX__
    asm("vxorps %[v], %[v], %[v]\n\
         vcvtsi2ss %[x], %[v], %[v]"
        : [v] "=x" (v)
        : [x] "r" (x)
    );
#else
    asm("xorps %[v], %[v]\n\
         cvtsi2ss %[x], %[v]"
        : [v] "=x" (v)
        : [x] "r" (x)
    );
#endif
    return v;
#else
    return _mm_cvtsi64_ss(_mm_setzero_ps(), x);
#endif
}

static inline uint64_t _FastIntegerSqrtInternal_DecIfLess(uint64_t r, uint64_t x, uint64_t m)
{
#if defined(__clang__) || defined(__GNUC__)
    if (x < m)
        --r;
#else
    uint64_t unused;
    _subborrow_u64(_subborrow_u64(0, x, m, &unused), r, 0, &r);
#endif
    return r;
}

// returns 0x80000000 for negative inputs
static inline int32_t IntegerSqrt(int32_t x)
{
    const __m128d v = _FastIntegerSqrtInternal_I32ToSd(x);
    return _mm_cvttsd_si32(_mm_sqrt_sd(v, v));
}

static inline uint32_t IntegerSqrt(uint32_t x)
{
    const __m128d v = _FastIntegerSqrtInternal_I64ToSd(x);
    return uint32_t(_mm_cvttsd_si32(_mm_sqrt_sd(v, v)));
}

// returns 0x8000000000000000 for negative inputs
static inline int64_t IntegerSqrt(int64_t x)
{
    const __m128d v = _FastIntegerSqrtInternal_I64ToSd(x);
    const uint64_t r = uint64_t(_mm_cvttsd_si64(_mm_sqrt_sd(v, v)));
    return int64_t(_FastIntegerSqrtInternal_DecIfLess(r, uint64_t(x), r * r));
}

static inline uint64_t IntegerSqrt(uint64_t x)
{
    __m128d v = _FastIntegerSqrtInternal_I64ToSd(int64_t(x));
    const uint64_t a = x >> 63 ? 0x43f0000000000001ULL : 0ULL;
    v = _mm_add_sd(v, _mm_castsi128_pd(_mm_cvtsi64_si128(int64_t(a))));
    const uint64_t r = uint64_t(_mm_cvttsd_si64(_mm_sqrt_sd(v, v)));
    return _FastIntegerSqrtInternal_DecIfLess(r, x, r * r);
}

// supports negative inputs (i.e. correctly returns false for all x < 0)
static inline bool IsPerfectSqr(int32_t x)
{
    const __m128 v = _FastIntegerSqrtInternal_I32ToSs(x);
    const int32_t r = _mm_cvttss_si32(_mm_sqrt_ss(v));
    return r * r == x;
}

static inline bool IsPerfectSqr(uint32_t x)
{
    const __m128 v = _FastIntegerSqrtInternal_I64ToSs(x);
    const uint32_t r = uint32_t(_mm_cvttss_si32(_mm_sqrt_ss(v)));
    return r * r == x;
}

// supports negative inputs (i.e. correctly returns false for all x < 0)
static inline bool IsPerfectSqr(int64_t x)
{
    const __m128d v = _FastIntegerSqrtInternal_I64ToSd(x);
    const int64_t r = _mm_cvttsd_si64(_mm_sqrt_sd(v, v));
    return r * r == x;
}

static inline bool IsPerfectSqr(uint64_t x)
{
    __m128d v = _FastIntegerSqrtInternal_I64ToSd(int64_t(x));
    const uint64_t a = x >> 63 ? 0x43f0000000000001ULL : 0ULL;
    v = _mm_add_sd(v, _mm_castsi128_pd(_mm_cvtsi64_si128(int64_t(a))));
    const uint64_t r = uint64_t(_mm_cvttsd_si64(_mm_sqrt_sd(v, v)));
    return r * r == x;
}
