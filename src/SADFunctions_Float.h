#pragma once

// 32-bit FLOAT pixel block-SAD kernels (SAD = sum |src - ref|). abs = sign-bit clear; no FMA.
//   * sad_f32_c_raw    - scalar; auto-vectorizes when its TU is built with fast-math (/fp:fast or
//                        -ffast-math). The cross-platform baseline / non-x86 fallback, and the path
//                        for narrow widths (W < 8) and pre-AVX2 x86 -- there is intentionally NO SSE2
//                        hand kernel for float (it never beat the auto-vec C in benchmarks).
//   * sad_f32_hand_raw - hand intrinsics, AVX2 (YMM, W>=8) / AVX-512 (ZMM, W>=16). NOT fast-math
//                        (that breaks the bitmask abs). ~2-3x the auto-vec C on clang.
// Both return the RAW float sum; scale_f32_sad() maps it to the 16-bit integer pixel range so the
// result drops straight into the integer cost model without overflowing int/int64 math downstream.

#include <cstdint>
#include <cstddef>
#include <cmath>

// Map a normalized-[0,1] float SAD to the 16-bit pixel scale. Deliberately 16-bit (x65535), NOT the
// 32-bit sample width: a 2^32 scale would overflow the unsigned-int SAD and the int/int64 cost sums
// later. A float clip then behaves exactly like a 16-bit clip for verybigSAD / lambda / thSCD.
static inline unsigned int scale_f32_sad(float sad) noexcept {
    float v = sad * 65535.0f;
    if (!(v > 0.0f)) return 0u;                  // <= 0 or NaN
    if (v >= 4294967040.0f) return 0xFFFFFFFFu;  // clamp out-of-range (HDR / > 1.0) float
    return (unsigned int)(v + 0.5f);
}

// scalar reference / baseline / non-x86 fallback (RAW float). __restrict + fixed trip counts -> clean
// auto-vec. Also the float SAD for W < 8 and pre-AVX2 x86 (no SSE2 hand kernel exists by design).
template <unsigned W, unsigned H>
static float sad_f32_c_raw(const uint8_t *__restrict s8, intptr_t sp, const uint8_t *__restrict r8, intptr_t rp) noexcept {
    float acc = 0.0f;
    for (unsigned y = 0; y < H; y++) {
        const float *__restrict s = (const float *)(s8 + (intptr_t)y * sp);
        const float *__restrict r = (const float *)(r8 + (intptr_t)y * rp);
        for (unsigned x = 0; x < W; x++)
            acc += fabsf(s[x] - r[x]);
    }
    return acc;
}

// Pure-scalar float SATD: standard separable 4x4 Hadamard, tiled over the block (4x4 units; an 8x4 is
// just two 4x4, so this matches the integer Satd_C result). Auto-vectorizes when the TU has fast-math.
// The >>1 SATD normalization is the *0.5f; scale_f32_sad maps to the 16-bit scale (same as the SAD).
template <unsigned W, unsigned H>
static unsigned int satd_f32_c(const uint8_t *s8, intptr_t sp, const uint8_t *r8, intptr_t rp) noexcept {
    float total = 0.0f;
    for (unsigned by = 0; by < H; by += 4) {
        for (unsigned bx = 0; bx < W; bx += 4) {
            float t[4][4];
            for (int i = 0; i < 4; i++) { // horizontal 1D Hadamard of the diff rows
                const float *s = (const float *)(s8 + (intptr_t)(by + i) * sp) + bx;
                const float *r = (const float *)(r8 + (intptr_t)(by + i) * rp) + bx;
                float e0 = s[0] - r[0], e1 = s[1] - r[1], e2 = s[2] - r[2], e3 = s[3] - r[3];
                float a0 = e0 + e1, a1 = e0 - e1, a2 = e2 + e3, a3 = e2 - e3;
                t[i][0] = a0 + a2; t[i][1] = a1 + a3; t[i][2] = a0 - a2; t[i][3] = a1 - a3;
            }
            for (int j = 0; j < 4; j++) { // vertical 1D Hadamard + sum of |coeffs|
                float a0 = t[0][j] + t[1][j], a1 = t[0][j] - t[1][j], a2 = t[2][j] + t[3][j], a3 = t[2][j] - t[3][j];
                total += fabsf(a0 + a2) + fabsf(a1 + a3) + fabsf(a0 - a2) + fabsf(a1 - a3);
            }
        }
    }
    return scale_f32_sad(total * 0.5f);
}

#if defined(MVTOOLS_X86) && defined(__AVX__)
#include <immintrin.h>

// Hand intrinsics start at AVX2 (256-bit). No SSE2 (128-bit) float kernel: the auto-vec C baseline
// handles SSE2/non-x86 and W < 8. The only 128-bit ops here are the YMM horizontal-reduce tail.
static inline float sadf_reduce128(__m128 v) {
    v = _mm_add_ps(v, _mm_movehl_ps(v, v));
    v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 1));
    return _mm_cvtss_f32(v);
}
static inline __m256 sadf_absmask256() { return _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)); }
static inline float sadf_reduce256(__m256 a) {
    return sadf_reduce128(_mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1)));
}

// W >= 8, W % 8 == 0. 4-row unroll, 4 accumulators.
template <unsigned W, unsigned H>
static float sadf_ymm(const uint8_t *s8, intptr_t sp, const uint8_t *r8, intptr_t rp) noexcept {
    const __m256 m = sadf_absmask256();
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    unsigned y = 0;
    for (; y + 4 <= H; y += 4) {
        const float *s0 = (const float *)(s8 + (intptr_t)(y + 0) * sp), *r0 = (const float *)(r8 + (intptr_t)(y + 0) * rp);
        const float *s1 = (const float *)(s8 + (intptr_t)(y + 1) * sp), *r1 = (const float *)(r8 + (intptr_t)(y + 1) * rp);
        const float *s2 = (const float *)(s8 + (intptr_t)(y + 2) * sp), *r2 = (const float *)(r8 + (intptr_t)(y + 2) * rp);
        const float *s3 = (const float *)(s8 + (intptr_t)(y + 3) * sp), *r3 = (const float *)(r8 + (intptr_t)(y + 3) * rp);
        for (unsigned x = 0; x < W; x += 8) {
            a0 = _mm256_add_ps(a0, _mm256_and_ps(_mm256_sub_ps(_mm256_loadu_ps(s0 + x), _mm256_loadu_ps(r0 + x)), m));
            a1 = _mm256_add_ps(a1, _mm256_and_ps(_mm256_sub_ps(_mm256_loadu_ps(s1 + x), _mm256_loadu_ps(r1 + x)), m));
            a2 = _mm256_add_ps(a2, _mm256_and_ps(_mm256_sub_ps(_mm256_loadu_ps(s2 + x), _mm256_loadu_ps(r2 + x)), m));
            a3 = _mm256_add_ps(a3, _mm256_and_ps(_mm256_sub_ps(_mm256_loadu_ps(s3 + x), _mm256_loadu_ps(r3 + x)), m));
        }
    }
    for (; y < H; y++) {
        const float *s0 = (const float *)(s8 + (intptr_t)y * sp), *r0 = (const float *)(r8 + (intptr_t)y * rp);
        for (unsigned x = 0; x < W; x += 8)
            a0 = _mm256_add_ps(a0, _mm256_and_ps(_mm256_sub_ps(_mm256_loadu_ps(s0 + x), _mm256_loadu_ps(r0 + x)), m));
    }
    return sadf_reduce256(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
}

#if defined(__AVX512F__)
static inline __m512 sadf_absmask512() { return _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF)); }
// W >= 16, W % 16 == 0. 4-row unroll, 4 accumulators.
template <unsigned W, unsigned H>
static float sadf_zmm(const uint8_t *s8, intptr_t sp, const uint8_t *r8, intptr_t rp) noexcept {
    const __m512 m = sadf_absmask512();
    __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    unsigned y = 0;
    for (; y + 4 <= H; y += 4) {
        const float *s0 = (const float *)(s8 + (intptr_t)(y + 0) * sp), *r0 = (const float *)(r8 + (intptr_t)(y + 0) * rp);
        const float *s1 = (const float *)(s8 + (intptr_t)(y + 1) * sp), *r1 = (const float *)(r8 + (intptr_t)(y + 1) * rp);
        const float *s2 = (const float *)(s8 + (intptr_t)(y + 2) * sp), *r2 = (const float *)(r8 + (intptr_t)(y + 2) * rp);
        const float *s3 = (const float *)(s8 + (intptr_t)(y + 3) * sp), *r3 = (const float *)(r8 + (intptr_t)(y + 3) * rp);
        for (unsigned x = 0; x < W; x += 16) {
            a0 = _mm512_add_ps(a0, _mm512_and_ps(_mm512_sub_ps(_mm512_loadu_ps(s0 + x), _mm512_loadu_ps(r0 + x)), m));
            a1 = _mm512_add_ps(a1, _mm512_and_ps(_mm512_sub_ps(_mm512_loadu_ps(s1 + x), _mm512_loadu_ps(r1 + x)), m));
            a2 = _mm512_add_ps(a2, _mm512_and_ps(_mm512_sub_ps(_mm512_loadu_ps(s2 + x), _mm512_loadu_ps(r2 + x)), m));
            a3 = _mm512_add_ps(a3, _mm512_and_ps(_mm512_sub_ps(_mm512_loadu_ps(s3 + x), _mm512_loadu_ps(r3 + x)), m));
        }
    }
    for (; y < H; y++) {
        const float *s0 = (const float *)(s8 + (intptr_t)y * sp), *r0 = (const float *)(r8 + (intptr_t)y * rp);
        for (unsigned x = 0; x < W; x += 16)
            a0 = _mm512_add_ps(a0, _mm512_and_ps(_mm512_sub_ps(_mm512_loadu_ps(s0 + x), _mm512_loadu_ps(r0 + x)), m));
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
}
#endif

// AVX2: YMM (W>=8). AVX-512: ZMM (W>=16), YMM otherwise. Callers only register widths these handle.
template <unsigned W, unsigned H>
static float sad_f32_hand_raw(const uint8_t *s8, intptr_t sp, const uint8_t *r8, intptr_t rp) noexcept {
#if defined(__AVX512F__)
    if constexpr (W >= 16 && W % 16 == 0) return sadf_zmm<W, H>(s8, sp, r8, rp);
    else return sadf_ymm<W, H>(s8, sp, r8, rp);
#else
    return sadf_ymm<W, H>(s8, sp, r8, rp);
#endif
}

#endif // MVTOOLS_X86 && __AVX__
