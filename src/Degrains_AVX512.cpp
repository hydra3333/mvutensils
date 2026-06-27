// AVX-512 Degrain dispatch: the scalar Degrain_C8/C16 (Degrains.h) auto-vectorized at AVX-512 -- this TU is
// built with -march=x86-64-v4 -mprefer-vector-width=512. Benchmarks showed a hand-written AVX-512 intrinsic
// only ties the compiler's auto-vec at this ISA (and the auto-vec is ~2x the AVX2 path), so we dispatch the
// auto-vectorized C instead of an intrinsic kernel. v3 twin: Degrains_AVX2.cpp.
//
// Only the block sizes where AVX-512 actually beats AVX2 are listed (8-bit width>=32, 16-bit width>=16);
// smaller blocks fall through to the AVX2 auto-vec path (AVX-512 ties or slightly trails it there).
#include <unordered_map>

#include "Degrains.h"

#define KEY(width, height, bits) (unsigned)(width) << 24 | (height) << 16 | (bits) << 8

#ifdef MVTOOLS_X86

#define DEGRAIN_AV8(radius, width, height)  { KEY(width, height, 8),  Degrain_C8<radius, width, height> },
#define DEGRAIN_AV16(radius, width, height) { KEY(width, height, 16), Degrain_C16<radius, width, height> },

#define DEGRAIN_LEVEL_AV(radius) {\
        DEGRAIN_AV16(radius, 16, 1) DEGRAIN_AV16(radius, 16, 2) DEGRAIN_AV16(radius, 16, 4)\
        DEGRAIN_AV16(radius, 16, 8) DEGRAIN_AV16(radius, 16, 16) DEGRAIN_AV16(radius, 16, 32)\
        DEGRAIN_AV8(radius, 32, 8)   DEGRAIN_AV16(radius, 32, 8)\
        DEGRAIN_AV8(radius, 32, 16)  DEGRAIN_AV16(radius, 32, 16)\
        DEGRAIN_AV8(radius, 32, 32)  DEGRAIN_AV16(radius, 32, 32)\
        DEGRAIN_AV8(radius, 32, 64)  DEGRAIN_AV16(radius, 32, 64)\
        DEGRAIN_AV8(radius, 64, 16)  DEGRAIN_AV16(radius, 64, 16)\
        DEGRAIN_AV8(radius, 64, 32)  DEGRAIN_AV16(radius, 64, 32)\
        DEGRAIN_AV8(radius, 64, 64)  DEGRAIN_AV16(radius, 64, 64)\
        DEGRAIN_AV8(radius, 64, 128) DEGRAIN_AV16(radius, 64, 128)\
        DEGRAIN_AV8(radius, 128, 32)  DEGRAIN_AV16(radius, 128, 32)\
        DEGRAIN_AV8(radius, 128, 64)  DEGRAIN_AV16(radius, 128, 64)\
        DEGRAIN_AV8(radius, 128, 128) DEGRAIN_AV16(radius, 128, 128)\
    }

static const std::unordered_map<uint32_t, DenoiseFunction> degrain_functions[6] = {
    DEGRAIN_LEVEL_AV(1), DEGRAIN_LEVEL_AV(2), DEGRAIN_LEVEL_AV(3),
    DEGRAIN_LEVEL_AV(4), DEGRAIN_LEVEL_AV(5), DEGRAIN_LEVEL_AV(6),
};

void selectDegrainFunctionAVX512(unsigned radius, unsigned width, unsigned height, unsigned bits, DenoiseFunction &degrain) noexcept {
    const auto &m = degrain_functions[radius - 1];
    auto it = m.find(KEY(width, height, bits));
    if (it != m.end())
        degrain = it->second;
}

#else
void selectDegrainFunctionAVX512(unsigned, unsigned, unsigned, unsigned, DenoiseFunction &) noexcept {}
#endif
