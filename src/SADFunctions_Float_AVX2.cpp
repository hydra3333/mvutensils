// AVX2 float SATD override: the pure-scalar satd_f32_c auto-vectorized at AVX2 (this TU is built with
// -march=x86-64-v3 AND fast-math). ~1.3-1.4x the SSE2 baseline across all sizes -- see
// bench_degrain/satd_f32_bench.cpp. (Separate from SADFunctions_AVX2.cpp because that TU must NOT have
// fast-math: it would break the hand SAD bitmask abs.)
#include <unordered_map>

#include "SADFunctions.h"
#include "SADFunctions_Float.h"

#if defined(MVTOOLS_X86)

#define KEY(width, height, bits) (unsigned)(width) << 24 | (height) << 16 | (bits) << 8
#define SATD_F32(width, height) { KEY(width, height, 32), satd_f32_c<width, height> },

static const std::unordered_map<uint32_t, SADFunction> satd_f32_functions = {
    SATD_F32(4, 4) SATD_F32(8, 4) SATD_F32(8, 8) SATD_F32(16, 8) SATD_F32(16, 16)
    SATD_F32(32, 16) SATD_F32(32, 32) SATD_F32(64, 32) SATD_F32(64, 64) SATD_F32(128, 64) SATD_F32(128, 128)
};

void selectSATDFunctionFloatAVX2(unsigned width, unsigned height, SADFunction &satd) {
    auto it = satd_f32_functions.find(KEY(width, height, 32));
    if (it != satd_f32_functions.end())
        satd = it->second;
}

#endif // MVTOOLS_X86
