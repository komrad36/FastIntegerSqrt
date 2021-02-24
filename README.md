Fastest implementations of 32-bit and 64-bit integer square roots for x86,
and querying for perfect squares, by a significant margin.

The square roots truncate, i.e. round down (IntegerSqrt(8) -> 2).

AVX recommended. SSE3 required. haddpd can be replaced with permilpd/addsd to drop the
requirement down to SSE2.

The I32/U32 versions work regardless of the FPU rounding mode.
The I64/U64 versions require the FPU to be in round-to-nearest mode (which is the default).
