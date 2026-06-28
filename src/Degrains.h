#pragma once

#include <cstdint>
#include <cstring>
#include <optional>
#include <algorithm>

#include "Common.h"
#include "SuperPyramid.h"
#include "MotionBlockPyramid.h"

enum VectorOrder {
    Backward1 = 0,
    Forward1,
    Backward2,
    Forward2,
    Backward3,
    Forward3,
    Backward4,
    Forward4,
    Backward5,
    Forward5,
    Backward6,
    Forward6
};


typedef void (*DenoiseFunction)(uint8_t *pDst, ptrdiff_t nDstPitch, const uint8_t *pSrc, ptrdiff_t nSrcPitch, const uint8_t **_pRefs, const ptrdiff_t *nRefPitches, uint16_t WSrc, const uint16_t *WRefs) noexcept;


// XXX Both Degrain_C8/Degrain_C16 move the pointers passed in pRefs8. This is okay
// because they are not used after the function is done with them.

template <int radius, int blockWidth, int blockHeight>
static void Degrain_C8(uint8_t * MVU_RESTRICT pDst8, ptrdiff_t nDstPitch, const uint8_t * MVU_RESTRICT pSrc8, ptrdiff_t nSrcPitch, const uint8_t ** MVU_RESTRICT pRefs8, const ptrdiff_t * MVU_RESTRICT nRefPitches, uint16_t WSrc, const uint16_t * MVU_RESTRICT WRefs) noexcept {
    const uint16_t wsrc = WSrc;
    uint16_t wref[radius * 2];
    for (int r = 0; r < radius * 2; r++)
        wref[r] = WRefs[r];

    for (int y = 0; y < blockHeight; y++) {
        const uint8_t * MVU_RESTRICT pSrc = pSrc8;
        uint8_t * MVU_RESTRICT pDst = pDst8;

        for (int x = 0; x < blockWidth; x++) {
            uint16_t sum = (uint16_t)(128u + (uint16_t)((uint16_t)pSrc[x] * wsrc));
            for (int r = 0; r < radius * 2; r++) {
                const uint8_t * MVU_RESTRICT pRef = pRefs8[r];
                sum = (uint16_t)(sum + (uint16_t)((uint16_t)pRef[x] * wref[r]));
            }
            pDst[x] = (uint8_t)(sum >> 8);
        }

        pDst8 += nDstPitch;
        pSrc8 += nSrcPitch;
        for (int r = 0; r < radius * 2; r++)
            pRefs8[r] += nRefPitches[r];
    }
}

template <int radius, int blockWidth, int blockHeight>
static void Degrain_C16(uint8_t * MVU_RESTRICT pDst8, ptrdiff_t nDstPitch, const uint8_t * MVU_RESTRICT pSrc8, ptrdiff_t nSrcPitch, const uint8_t ** MVU_RESTRICT pRefs8, const ptrdiff_t * MVU_RESTRICT nRefPitches, uint16_t WSrc, const uint16_t * MVU_RESTRICT WRefs) noexcept {
    const int wsrc = WSrc;
    int wref[radius * 2];
    for (int r = 0; r < radius * 2; r++)
        wref[r] = WRefs[r];

    for (int y = 0; y < blockHeight; y++) {
        for (int x = 0; x < blockWidth; x++) {
            const uint16_t *pSrc = (const uint16_t * __restrict)pSrc8;
            uint16_t *pDst = (uint16_t * __restrict)pDst8;

            int sum = 128 + pSrc[x] * wsrc;

            for (int r = 0; r < radius * 2; r++) {
                const uint16_t *pRef = (const uint16_t * __restrict)pRefs8[r];
                sum += pRef[x] * wref[r];
            }

            pDst[x] = sum >> 8;
        }

        pDst8 += nDstPitch;
        pSrc8 += nSrcPitch;
        for (int r = 0; r < radius * 2; r++)
            pRefs8[r] += nRefPitches[r];
    }
}

template <int radius, int blockWidth, int blockHeight>
static void Degrain_F32(uint8_t *MVU_RESTRICT pDst8, ptrdiff_t nDstPitch, const uint8_t *MVU_RESTRICT pSrc8, ptrdiff_t nSrcPitch, const uint8_t **MVU_RESTRICT pRefs8, const ptrdiff_t *MVU_RESTRICT nRefPitches, uint16_t WSrc, const uint16_t *MVU_RESTRICT WRefs) noexcept {
    const float wsrc = WSrc;
    float wref[radius * 2];
    for (int r = 0; r < radius * 2; r++)
        wref[r] = WRefs[r];

    for (int y = 0; y < blockHeight; y++) {
        for (int x = 0; x < blockWidth; x++) {
            const float *pSrc = (const float * __restrict)pSrc8;
            float *pDst = (float * __restrict)pDst8;

            float sum = pSrc[x] * wsrc;

            for (int r = 0; r < radius * 2; r++) {
                const float *pRef = (const float * __restrict)pRefs8[r];
                sum += pRef[x] * wref[r];
            }

            pDst[x] = sum / 256.0f;
        }

        pDst8 += nDstPitch;
        pSrc8 += nSrcPitch;
        for (int r = 0; r < radius * 2; r++)
            pRefs8[r] += nRefPitches[r];
    }
}


#ifdef MVTOOLS_X86
#include <emmintrin.h>
void selectDegrainFunctionAVX2(unsigned radius, unsigned width, unsigned height, unsigned bits, DenoiseFunction &degrain) noexcept;
void selectDegrainFunctionAVX512(unsigned radius, unsigned width, unsigned height, unsigned bits, DenoiseFunction &degrain) noexcept;

template <int radius, int blockWidth, int blockHeight>
static void Degrain_sse2(uint8_t * MVU_RESTRICT pDst, ptrdiff_t nDstPitch, const uint8_t * MVU_RESTRICT pSrc, ptrdiff_t nSrcPitch, const uint8_t ** MVU_RESTRICT pRefs, const ptrdiff_t * MVU_RESTRICT nRefPitches, uint16_t WSrc, const uint16_t * MVU_RESTRICT WRefs) noexcept {
    static_assert(blockWidth >= 4, "");

    __m128i zero = _mm_setzero_si128();
    __m128i wsrc = _mm_set1_epi16(WSrc);
    __m128i wrefs[12];

    // We intentionally jump by 2 (here and below), as it delineates groups of
    // backward/forward and ALSO produces testably faster code.
    for(int i = 0; i < radius * 2; i += 2) {
        wrefs[i] = _mm_set1_epi16(WRefs[i]);
        wrefs[i + 1] = _mm_set1_epi16(WRefs[i + 1]);
    }

    __m128i src, accum, refs[12];

    for (int y = 0; y < blockHeight; y++) {
        for (int x = 0; x < blockWidth; x += 8) {
            if (blockWidth == 4) {
                src = _mm_cvtsi32_si128(*(const int *)pSrc);
                for(int i = 0; i < radius * 2; i += 2) {
                    refs[i] = _mm_cvtsi32_si128(*(const int *)pRefs[i]);
                    refs[i + 1] = _mm_cvtsi32_si128(*(const int *)pRefs[i + 1]);
                }
            } else {
                src = _mm_loadl_epi64((const __m128i *)(pSrc + x));
                for(int i = 0; i < radius * 2; i += 2) {
                    refs[i] = _mm_loadl_epi64((const __m128i *)(pRefs[i] + x));
                    refs[i + 1] = _mm_loadl_epi64((const __m128i *)(pRefs[i + 1] + x));
                }
            }

            src = _mm_unpacklo_epi8(src, zero);
            src = _mm_mullo_epi16(src, wsrc);

            for(int i = 0; i < radius * 2; i += 2) {
                refs[i] = _mm_unpacklo_epi8(refs[i], zero);
                refs[i + 1] = _mm_unpacklo_epi8(refs[i + 1], zero);

                refs[i] = _mm_mullo_epi16(refs[i], wrefs[i]);
                refs[i + 1] = _mm_mullo_epi16(refs[i + 1], wrefs[i + 1]);
            }

            accum = _mm_set1_epi16(128);
            accum = _mm_add_epi16(accum, src);

            for(int i = 0; i < radius * 2; i += 2) {
                accum = _mm_add_epi16(accum, refs[i]);
                accum = _mm_add_epi16(accum, refs[i + 1]);
            }

            accum = _mm_srli_epi16(accum, 8);
            accum = _mm_packus_epi16(accum, zero);

            if (blockWidth == 4)
                *(int *)pDst = _mm_cvtsi128_si32(accum);
            else
                _mm_storel_epi64((__m128i *)(pDst + x), accum);
        }
        pDst += nDstPitch;
        pSrc += nSrcPitch;
        for(int i = 0; i < radius * 2; i += 2) {
            pRefs[i] += nRefPitches[i];
            pRefs[i + 1] += nRefPitches[i + 1];
        }
    }
}

#endif // MVTOOLS_X86


// LimitChanges clamps each output pixel to [pSrc - nLimit, pSrc + nLimit]. nLimit is
// validated to [0, pixelMax]
// The idiom used here is very friendly to compiler optimizers and helps them to not
// widen the type used for calculations
template <typename PixelType>
static void LimitChanges_C(uint8_t * MVU_RESTRICT pDst8, ptrdiff_t nDstPitch, const uint8_t * MVU_RESTRICT pSrc8, ptrdiff_t nSrcPitch, int nWidth, int nHeight, PixelType nLimit) noexcept {
    if constexpr (std::is_integral_v<PixelType>) {
        const int lim = nLimit;
        const int maxValue = (PixelType)~(PixelType)0; // 255 or 65535
        for (int h = 0; h < nHeight; h++) {
            const PixelType *pSrc = (const PixelType *)pSrc8;
            PixelType *pDst = (PixelType *)pDst8;
            for (int i = 0; i < nWidth; i++) {
                const int s = pSrc[i];
                const int lo = std::max(s - lim, 0);
                const int hi = std::min(s + lim, maxValue);
                pDst[i] = (PixelType)std::clamp((int)pDst[i], lo, hi);
            }
            pDst8 += nDstPitch;
            pSrc8 += nSrcPitch;
        }
    } else {
        const float lim = nLimit;
        for (int h = 0; h < nHeight; h++) {
            const PixelType *pSrc = (const PixelType *)pSrc8;
            PixelType *pDst = (PixelType *)pDst8;
            for (int i = 0; i < nWidth; i++)
                pDst[i] = std::clamp(pDst[i], pSrc[i] - lim, pSrc[i] + lim);
            pDst8 += nDstPitch;
            pSrc8 += nSrcPitch;
        }
    }
}

static inline uint16_t DegrainWeight(int64_t thSAD, int64_t blockSAD) noexcept {
    if (blockSAD >= thSAD)
        return 0;

    const double r = (double)blockSAD / (double)thSAD; // r in [0, 1)
    return static_cast<uint16_t>(256.0 * (1.0 - r * r) / (1.0 + r * r)); // in [0, 256]
}

template<typename PixelType>
static inline void useBlock(const uint8_t *&p, ptrdiff_t &np, uint16_t &WRef, int isUsable, const std::optional<MotionBlockPyramid> &blocks, int i, const FramePyramidLevel *pPlane, const uint8_t **pSrcCur, int xx, const ptrdiff_t *nSrcPitch, int nLogPel, int plane, int xSubUV, int ySubUV, const int64_t *thSAD) noexcept {
    if (isUsable) {
        const BlockData block = blocks->GetBlock(i);
        int blx = (block.x << nLogPel) + block.vector.x;
        int bly = (block.y << nLogPel) + block.vector.y;
        p = pPlane->planes[plane].GetPointer<PixelType>(plane ? blx >> xSubUV : blx, plane ? bly >> ySubUV : bly);
        np = pPlane->planes[plane].nPitch;
        int64_t blockSAD = block.vector.sad;
        WRef = DegrainWeight(thSAD[plane], blockSAD);
    } else {
        p = pSrcCur[plane] + xx;
        np = nSrcPitch[plane];
        WRef = 0;
    }
}


template <int radius>
static inline void normaliseWeights(uint16_t &WSrc, uint16_t *WRefs) noexcept {
    int wsrc = 256;
    int WSum = wsrc + 1;
    for (int r = 0; r < radius * 2; r++)
        WSum += WRefs[r];

    double scale = 256.0 / WSum;

    for (int r = 0; r < radius * 2; r++) {
        int w = static_cast<int>(WRefs[r] * scale);
        WRefs[r] = static_cast<uint16_t>(w);
        wsrc -= w;
    }

    WSrc = static_cast<uint16_t>(wsrc < 0 ? 0 : wsrc);
}
