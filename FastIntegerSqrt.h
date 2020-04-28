/*******************************************************************
*
*	Author: Kareem Omar
*	kareem.h.omar@gmail.com
*	https://github.com/komrad36
*
*	Last updated April 27, 2020
*******************************************************************/

// Fastest implementations of 32-bit and 64-bit integer square roots for x86,
// by a significant margin.
//
// Correct for any nonnegative inputs.
//
// The operation is truncating, i.e. rounds down (IntegerSqrt(8) -> 2).
// 
// Requires SSE3. haddpd can be replaced with permilpd/addsd to drop the
// requirement down to SSE2.
//
// The I32 and U32 versions work regardless of the FPU rounding mode. The I64
// and U64 versions require the FPU to be in round-to-nearest mode (which is the default).
//

#pragma once

#include <cstdint>
#include <immintrin.h>

using U32 = uint32_t;
using I32 = int32_t;
using U64 = uint64_t;
using I64 = int64_t;

// produces 0x80000000 for negative inputs
I32 IntegerSqrt(I32 x)
{
	__m128d v = _mm_cvtsi32_sd(_mm_setzero_pd(), x);
	v = _mm_sqrt_sd(v, v);
	return _mm_cvttsd_si32(v);
}

U32 IntegerSqrt(U32 x)
{
	__m128d v = _mm_cvtsi64_sd(_mm_setzero_pd(), x);
	v = _mm_sqrt_sd(v, v);
	return (U32)_mm_cvttsd_si64(v);
}

// produces 0x7fffffffffffffff for negative inputs
I64 IntegerSqrt(I64 x)
{
	__m128d v = _mm_cvtsi64_sd(_mm_setzero_pd(), x);
	v = _mm_sqrt_sd(v, v);
	const I64 g = _mm_cvttsd_si64(v);
	return g - (U64(x - g * g) >> 63);
}

U64 IntegerSqrt(U64 x)
{
#if 0
	// performs no memory accesses on MSVC
	const __m128i k1 = _mm_cvtsi64_si128(0x4530000043300000);
	const __m128d k2 = _mm_castsi128_pd(_mm_shuffle_epi32(k1, _MM_SHUFFLE(1, 3, 0, 2)));
#else
	// performs memory access; slightly faster than the above if memory is in cache,
	// e.g. for repeated calls to this function, but slower if it's not in cache,
	// e.g. for occasional calls.
	const __m128i k1 = _mm_set_epi64x(0x0000000000000000, 0x4530000043300000);
	const __m128d k2 = _mm_castsi128_pd(_mm_set_epi64x(0x4530000000000000, 0x4330000000000000));
#endif

	__m128d v = _mm_castsi128_pd(_mm_unpacklo_epi32(_mm_cvtsi64_si128(x), k1));
	v = _mm_sub_pd(v, k2);
	v = _mm_hadd_pd(v, v);
	v = _mm_sqrt_sd(v, v);
	const U64 g = (U64)_mm_cvttsd_si64(v);
	return g - ((x - g * g) >> 63);
}
