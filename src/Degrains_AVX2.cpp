#include <stdexcept>
#include <unordered_map>

#include "Degrains.h"

#define KEY(width, height, bits) (unsigned)(width) << 24 | (height) << 16 | (bits) << 8

#ifdef MVTOOLS_X86
#define DEGRAIN_AVX2(radius, width, height) \
    { KEY(width, height, 8), Degrain_avx2<radius, width, height> }, \
    { KEY(width, height, 16), Degrain_uint16_avx2<radius, width, height> },

// 8-bit only -- no 16-bit 2xN kernel (width 2 stays scalar for 16-bit pixels).
#define DEGRAIN_W2_AVX2(radius) \
    { KEY(2, 2, 8), Degrain_avx2<radius, 2, 2> }, \
    { KEY(2, 4, 8), Degrain_avx2<radius, 2, 4> },

#define DEGRAIN_LEVEL_AVX2(radius) \
    {\
        DEGRAIN_W2_AVX2(radius)\
        DEGRAIN_AVX2(radius, 8, 2)\
        DEGRAIN_AVX2(radius, 8, 4)\
        DEGRAIN_AVX2(radius, 8, 8)\
        DEGRAIN_AVX2(radius, 8, 16)\
        DEGRAIN_AVX2(radius, 16, 1)\
        DEGRAIN_AVX2(radius, 16, 2)\
        DEGRAIN_AVX2(radius, 16, 4)\
        DEGRAIN_AVX2(radius, 16, 8)\
        DEGRAIN_AVX2(radius, 16, 16)\
        DEGRAIN_AVX2(radius, 16, 32)\
        DEGRAIN_AVX2(radius, 32, 8)\
        DEGRAIN_AVX2(radius, 32, 16)\
        DEGRAIN_AVX2(radius, 32, 32)\
        DEGRAIN_AVX2(radius, 32, 64)\
        DEGRAIN_AVX2(radius, 64, 16)\
        DEGRAIN_AVX2(radius, 64, 32)\
        DEGRAIN_AVX2(radius, 64, 64)\
        DEGRAIN_AVX2(radius, 64, 128)\
        DEGRAIN_AVX2(radius, 128, 32)\
        DEGRAIN_AVX2(radius, 128, 64)\
        DEGRAIN_AVX2(radius, 128, 128)\
    }
#else
#define DEGRAIN_AVX2(radius, width, height)
#define DEGRAIN_W2_AVX2(radius)
#define DEGRAIN_LEVEL_AVX2(radius)
#endif


#ifdef MVTOOLS_X86

#include <immintrin.h>

// Load 2 rows of 2 px -> 4x u16 [r0p0 r0p1 r1p0 r1p1] (2-byte reads, no over-read).
static inline __m128i degrain_w2_pack(const uint8_t *p, ptrdiff_t pitch) {
    __m128i a = _mm_cvtsi32_si128(*(const uint16_t *)p);
    __m128i b = _mm_cvtsi32_si128(*(const uint16_t *)(p + pitch));
    return _mm_unpacklo_epi8(_mm_unpacklo_epi16(a, b), _mm_setzero_si128());
}

// XXX Moves the pointers passed in pRefs. This is okay because they are not
// used after this function is done with them.
template <int radius, int blockWidth, int blockHeight>
static void Degrain_avx2(uint8_t * MVU_RESTRICT pDst, ptrdiff_t nDstPitch, const uint8_t * MVU_RESTRICT pSrc, ptrdiff_t nSrcPitch, const uint8_t ** MVU_RESTRICT pRefs, const ptrdiff_t * MVU_RESTRICT nRefPitches, uint16_t WSrc, const uint16_t * MVU_RESTRICT WRefs) noexcept {
    static_assert(blockWidth == 2 || blockWidth >= 16 || (blockWidth == 8 && blockHeight >= 2), "");

    if constexpr (blockWidth == 2) {
        static_assert(blockHeight % 2 == 0, "2xN needs even height");
        const __m128i zero128 = _mm_setzero_si128();
        const __m128i wsrc2 = _mm_set1_epi16(WSrc);
        __m128i wrefs2[12];
        for (int i = 0; i < radius * 2; i++)
            wrefs2[i] = _mm_set1_epi16(WRefs[i]);

        for (int y = 0; y < blockHeight; y += 2) {
            __m128i accum = _mm_add_epi16(_mm_set1_epi16(128), _mm_mullo_epi16(degrain_w2_pack(pSrc, nSrcPitch), wsrc2));
            for (int i = 0; i < radius * 2; i++)
                accum = _mm_add_epi16(accum, _mm_mullo_epi16(degrain_w2_pack(pRefs[i], nRefPitches[i]), wrefs2[i]));
            accum = _mm_srli_epi16(accum, 8);
            accum = _mm_packus_epi16(accum, zero128);
            *(uint16_t *)pDst = (uint16_t)_mm_cvtsi128_si32(accum);
            *(uint16_t *)(pDst + nDstPitch) = (uint16_t)_mm_extract_epi16(accum, 1);
            pDst += nDstPitch * 2;
            pSrc += nSrcPitch * 2;
            for (int i = 0; i < radius * 2; i++)
                pRefs[i] += nRefPitches[i] * 2;
        }
        return;
    }

    __m256i zero = _mm256_setzero_si256();
    __m256i wsrc = _mm256_set1_epi16(WSrc);

    __m256i wrefs[12];
    for(int i = 0; i < radius * 2; i += 2) {
        wrefs[i] = _mm256_set1_epi16(WRefs[i]);
        wrefs[i + 1] = _mm256_set1_epi16(WRefs[i + 1]);
    }
    __m256i src, accum, refs[12];

    int pitchMul = blockWidth == 8 ? 2 : 1;

    for (int y = 0; y < blockHeight; y += pitchMul) {
        for (int x = 0; x < blockWidth; x += 16 / pitchMul) {
            if (blockWidth == 8) {
                src = _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *)(pSrc + x)), _mm_loadl_epi64((const __m128i *)(pSrc + nSrcPitch + x))));
                for(int i = 0; i < radius * 2; i += 2) {
                    refs[i] = _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *)(pRefs[i] + x)), _mm_loadl_epi64((const __m128i *)(pRefs[i] + nRefPitches[i] + x))));
                    refs[i + 1] = _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *)(pRefs[i + 1] + x)), _mm_loadl_epi64((const __m128i *)(pRefs[i + 1] + nRefPitches[i + 1] + x))));
                }
            } else {
                src = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(pSrc + x)));
                for(int i = 0; i < radius * 2; i += 2) {
                    refs[i] = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(pRefs[i] + x)));
                    refs[i + 1] = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(pRefs[i + 1] + x)));
                }
            }

            src = _mm256_mullo_epi16(src, wsrc);
            for(int i = 0; i < radius * 2; i += 2) {
                refs[i] = _mm256_mullo_epi16(refs[i], wrefs[i]);
                refs[i + 1] = _mm256_mullo_epi16(refs[i + 1], wrefs[i + 1]);
            }

            accum = _mm256_set1_epi16(128);
            accum = _mm256_add_epi16(accum, src);

            for(int i = 0; i < radius * 2; i += 2) {
                accum = _mm256_add_epi16(accum, refs[i]);
                accum = _mm256_add_epi16(accum, refs[i + 1]);
            }
            accum = _mm256_srli_epi16(accum, 8);
            accum = _mm256_packus_epi16(accum, zero);

            if (blockWidth == 8) {
                _mm_storel_epi64((__m128i *)(pDst + x), _mm256_castsi256_si128(accum));
                _mm_storel_epi64((__m128i *)(pDst + nDstPitch + x), _mm256_extracti128_si256(accum, 1));
            } else {
                accum = _mm256_permute4x64_epi64(accum, _MM_SHUFFLE(0, 0, 2, 0));
                _mm_storeu_si128((__m128i *)(pDst + x), _mm256_castsi256_si128(accum));
            }
        }

        pDst += nDstPitch * pitchMul;
        pSrc += nSrcPitch * pitchMul;

        for(int i = 0; i < radius * 2; i += 2) {
            pRefs[i] += nRefPitches[i] * pitchMul;
            pRefs[i + 1] += nRefPitches[i + 1] * pitchMul;
        }
    }
}

template <int radius, int blockWidth, int blockHeight>
static void Degrain_uint16_avx2(uint8_t * MVU_RESTRICT pDst8, ptrdiff_t nDstPitch, const uint8_t * MVU_RESTRICT pSrc8, ptrdiff_t nSrcPitch, const uint8_t ** MVU_RESTRICT pRefs, const ptrdiff_t * MVU_RESTRICT nRefPitches, uint16_t WSrc, const uint16_t * MVU_RESTRICT WRefs) noexcept {
    static_assert(blockWidth >= 8, "");

    __m256i wsrc = _mm256_set1_epi32(WSrc);

    __m256i wrefs[12];
    for(int i = 0; i < radius * 2; i += 2) {
        wrefs[i] = _mm256_set1_epi32(WRefs[i]);
        wrefs[i + 1] = _mm256_set1_epi32(WRefs[i + 1]);
    }
    __m256i src, accum, refs[12];

    for (int y = 0; y < blockHeight; y++) {
        const uint16_t *pSrc = (const uint16_t *)pSrc8;
        uint16_t *pDst = (uint16_t *)pDst8;

        // 16-bit pixels (up to 65535) times weights (summing to 256) overflow
        // 16 bits, so widen to 32-bit lanes and accumulate there.
        for (int x = 0; x < blockWidth; x += 8) {
            src = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(pSrc + x)));
            for(int i = 0; i < radius * 2; i += 2) {
                refs[i] = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)((const uint16_t *)pRefs[i] + x)));
                refs[i + 1] = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)((const uint16_t *)pRefs[i + 1] + x)));
            }

            accum = _mm256_set1_epi32(128);
            accum = _mm256_add_epi32(accum, _mm256_mullo_epi32(src, wsrc));

            for(int i = 0; i < radius * 2; i += 2) {
                accum = _mm256_add_epi32(accum, _mm256_mullo_epi32(refs[i], wrefs[i]));
                accum = _mm256_add_epi32(accum, _mm256_mullo_epi32(refs[i + 1], wrefs[i + 1]));
            }
            accum = _mm256_srli_epi32(accum, 8);
            // Pack 8 uint32 -> 8 uint16; permute fixes the per-lane interleave.
            accum = _mm256_packus_epi32(accum, accum);
            accum = _mm256_permute4x64_epi64(accum, _MM_SHUFFLE(0, 0, 2, 0));
            _mm_storeu_si128((__m128i *)(pDst + x), _mm256_castsi256_si128(accum));
        }

        pDst8 += nDstPitch;
        pSrc8 += nSrcPitch;

        for(int i = 0; i < radius * 2; i += 2) {
            pRefs[i] += nRefPitches[i];
            pRefs[i + 1] += nRefPitches[i + 1];
        }
    }
}

#endif

static const std::unordered_map<uint32_t, DenoiseFunction> degrain_functions[6] = {
    DEGRAIN_LEVEL_AVX2(1),
    DEGRAIN_LEVEL_AVX2(2),
    DEGRAIN_LEVEL_AVX2(3),
    DEGRAIN_LEVEL_AVX2(4),
    DEGRAIN_LEVEL_AVX2(5),
    DEGRAIN_LEVEL_AVX2(6),
};

void selectDegrainFunctionAVX2(unsigned radius, unsigned width, unsigned height, unsigned bits, DenoiseFunction &degrain) noexcept {
    const auto &m = degrain_functions[radius - 1];
    auto it = m.find(KEY(width, height, bits));
    if (it != m.end())
        degrain = it->second;
}
