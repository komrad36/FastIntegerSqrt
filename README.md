Fastest implementations of 32-bit and 64-bit integer square roots for x86,
by a significant margin.

Correct for any nonnegative inputs.

The operation is truncating, i.e. rounds down (IntegerSqrt(8) -> 2).

Requires SSE3. haddpd can be replaced with permilpd/addsd to drop the
requirement down to SSE2.

The I32 and U32 versions work regardless of the FPU rounding mode. The I64
and U64 versions require the FPU to be in round-to-nearest mode (which is the default).
