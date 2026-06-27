// AVX-512 float SATD override: the pure-scalar satd_f32_c auto-vectorized at AVX-512 (this TU is built
// with -march=x86-64-v4 -mprefer-vector-width=512 AND fast-math). Registered only for width >= 16,
// where 512-bit beats AVX2 (1.2-2.6x); for W <= 8 it ties AVX2, so those stay on the AVX2 override.
// See bench_degrain/satd_f32_bench.cpp.
#include <unordered_map>

#include "SADFunctions.h"
#include "SADFunctions_Float.h"

#if defined(MVTOOLS_X86)

#define KEY(width, height, bits) (unsigned)(width) << 24 | (height) << 16 | (bits) << 8
#define SATD_F32(width, height) { KEY(width, height, 32), satd_f32_c<width, height> },

static const std::unordered_map<uint32_t, SADFunction> satd_f32_functions = {
    SATD_F32(16, 8) SATD_F32(16, 16) SATD_F32(32, 16) SATD_F32(32, 32)
    SATD_F32(64, 32) SATD_F32(64, 64) SATD_F32(128, 64) SATD_F32(128, 128)
};

void selectSATDFunctionFloatAVX512(unsigned width, unsigned height, SADFunction &satd) {
    auto it = satd_f32_functions.find(KEY(width, height, 32));
    if (it != satd_f32_functions.end())
        satd = it->second;
}

#endif // MVTOOLS_X86
