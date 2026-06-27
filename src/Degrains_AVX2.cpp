// AVX2 Degrain dispatch: the scalar Degrain_C8/C16 (Degrains.h) auto-vectorized at AVX2 -- this TU is
// built with -march=x86-64-v3. Benchmarks showed a hand-written AVX2 intrinsic only ties the compiler's
// auto-vec at this ISA (the whole ~2x win is just "compile wider"), so we dispatch the auto-vectorized C
// instead of maintaining intrinsic kernels. selectDegrainFunctionAVX512 (Degrains_AVX512.cpp) is the v4 twin.
#include <unordered_map>

#include "Degrains.h"

#define KEY(width, height, bits) (unsigned)(width) << 24 | (height) << 16 | (bits) << 8

#ifdef MVTOOLS_X86

#define DEGRAIN_AV(radius, width, height) \
    { KEY(width, height, 8), Degrain_C8<radius, width, height> }, \
    { KEY(width, height, 16), Degrain_C16<radius, width, height> },

#define DEGRAIN_LEVEL_AV(radius) {\
        DEGRAIN_AV(radius, 2, 2)   DEGRAIN_AV(radius, 2, 4)\
        DEGRAIN_AV(radius, 4, 2)   DEGRAIN_AV(radius, 4, 4)   DEGRAIN_AV(radius, 4, 8)\
        DEGRAIN_AV(radius, 8, 1)   DEGRAIN_AV(radius, 8, 2)   DEGRAIN_AV(radius, 8, 4)   DEGRAIN_AV(radius, 8, 8)   DEGRAIN_AV(radius, 8, 16)\
        DEGRAIN_AV(radius, 16, 1)  DEGRAIN_AV(radius, 16, 2)  DEGRAIN_AV(radius, 16, 4)  DEGRAIN_AV(radius, 16, 8)  DEGRAIN_AV(radius, 16, 16) DEGRAIN_AV(radius, 16, 32)\
        DEGRAIN_AV(radius, 32, 8)  DEGRAIN_AV(radius, 32, 16) DEGRAIN_AV(radius, 32, 32) DEGRAIN_AV(radius, 32, 64)\
        DEGRAIN_AV(radius, 64, 16) DEGRAIN_AV(radius, 64, 32) DEGRAIN_AV(radius, 64, 64) DEGRAIN_AV(radius, 64, 128)\
        DEGRAIN_AV(radius, 128, 32) DEGRAIN_AV(radius, 128, 64) DEGRAIN_AV(radius, 128, 128)\
    }

static const std::unordered_map<uint32_t, DenoiseFunction> degrain_functions[6] = {
    DEGRAIN_LEVEL_AV(1), DEGRAIN_LEVEL_AV(2), DEGRAIN_LEVEL_AV(3),
    DEGRAIN_LEVEL_AV(4), DEGRAIN_LEVEL_AV(5), DEGRAIN_LEVEL_AV(6),
};

void selectDegrainFunctionAVX2(unsigned radius, unsigned width, unsigned height, unsigned bits, DenoiseFunction &degrain) noexcept {
    const auto &m = degrain_functions[radius - 1];
    auto it = m.find(KEY(width, height, bits));
    if (it != m.end())
        degrain = it->second;
}

#else
void selectDegrainFunctionAVX2(unsigned, unsigned, unsigned, unsigned, DenoiseFunction &) noexcept {}
#endif
