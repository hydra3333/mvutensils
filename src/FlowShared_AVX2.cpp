// AVX2 gather implementations of FlowInter / FlowInterExtra (declared in FlowShared.h). Same subpel
// base+offset gather as FlowShared_AVX512.cpp (see that file's header for the addressing, the centered
// base, the int32-offset limit and the <=3-byte tail over-read) -- here 8 pixels/iter via the 256-bit
// vpgatherdd. AVX2 has no k-masks and no cvtepi32->epi8/16 store, so:
//   - the width%8 tail is handled by reprocessing the last full group of 8 (overlapping, idempotent);
//   - width < 8 falls back to the scalar reference (FlowInter*_scalar);
//   - stores pack with packus (after masking to the pixel range, which gives the scalar's truncation).

#include "FlowShared.h"

#if defined(MVTOOLS_X86)

#include <immintrin.h>

static MVU_FORCE_INLINE __m256i ld_u16(const uint16_t *p) {
    return _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)p));
}
template <int S>
static MVU_FORCE_INLINE __m256i ld_pix(const uint8_t *p) {
    if constexpr (S == 1)
        return _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)p));
    else
        return _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)p));
}
// Always stores exactly 8 pixels (8 bytes for u8, 16 for u16). Callers guarantee that 8-pixel span is
// in bounds: the width<8 case falls back to scalar, and the width%8 tail reprocesses the last full group
// (do8(width-8) writes [width-8, width-1], ending exactly at width -- never past it).
template <int S>
static MVU_FORCE_INLINE void st_pix(uint8_t *p, __m256i v) {
    if constexpr (S == 1) {
        __m256i m = _mm256_and_si256(v, _mm256_set1_epi32(0xFF));
        __m128i p16 = _mm_packus_epi32(_mm256_castsi256_si128(m), _mm256_extracti128_si256(m, 1));
        _mm_storel_epi64((__m128i *)p, _mm_packus_epi16(p16, p16)); // low 8 bytes = the 8 pixels
    } else {
        __m256i m = _mm256_and_si256(v, _mm256_set1_epi32(0xFFFF));
        _mm_storeu_si128((__m128i *)p, _mm_packus_epi32(_mm256_castsi256_si128(m), _mm256_extracti128_si256(m, 1)));
    }
}

// (a*b + 256) >> 8 per 32-bit lane via 64-bit products (exact even when a*b overflows int32).
static MVU_FORCE_INLINE __m256i mul_round8(__m256i a, __m256i b) {
    const __m256i c256 = _mm256_set1_epi64x(256);
    __m256i ev = _mm256_srli_epi64(_mm256_add_epi64(_mm256_mul_epu32(a, b), c256), 8);
    __m256i od = _mm256_srli_epi64(_mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64(a, 32), _mm256_srli_epi64(b, 32)), c256), 8);
    return _mm256_blend_epi32(ev, _mm256_slli_epi64(od, 32), 0xAA); // even lanes <- ev, odd <- od
}

template <int S>
static MVU_FORCE_INLINE __m256i mc_gather(const PyramidPlane &p, __m256i nX, __m256i nY) {
    const int log = ilog2(p.nPel);
    const int mid = (p.nPel * p.nPel) / 2;
    const __m256i mask = _mm256_set1_epi32(p.nPel - 1);
    const __m256i X = _mm256_add_epi32(nX, _mm256_set1_epi32(p.nHPaddingPel));
    const __m256i Y = _mm256_add_epi32(nY, _mm256_set1_epi32(p.nVPaddingPel));
    __m256i idx = _mm256_or_si256(_mm256_and_si256(X, mask), _mm256_sllv_epi32(_mm256_and_si256(Y, mask), _mm256_set1_epi32(log)));
    __m256i off = _mm256_mullo_epi32(_mm256_sub_epi32(idx, _mm256_set1_epi32(mid)), _mm256_set1_epi32((int)p.subPelPlaneOffset));
    off = _mm256_add_epi32(off, _mm256_slli_epi32(_mm256_srai_epi32(X, log), S == 2 ? 1 : 0));
    off = _mm256_add_epi32(off, _mm256_mullo_epi32(_mm256_srai_epi32(Y, log), _mm256_set1_epi32((int)p.nPitch)));
    __m256i g = _mm256_i32gather_epi32((const int *)p.pPlane[mid], off, 1);
    return _mm256_and_si256(g, _mm256_set1_epi32(S == 1 ? 0xFF : 0xFFFF));
}

template <int S>
static void flowinter_impl(uint8_t *MVU_RESTRICT pdst, ptrdiff_t dst_pitch,
        const PyramidPlane &prefB, const PyramidPlane &prefF,
        const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
        const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch,
        int dstX, int dstY, int width, int height, int time256) noexcept {
    using PT = typename std::conditional<S == 1, uint8_t, uint16_t>::type;
    if (width < 8) {
        FlowInter_scalar<PT>(pdst, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256);
        return;
    }
    const ptrdiff_t tp = tilePitch / (ptrdiff_t)sizeof(uint16_t);
    const int nPelLog = ilog2(prefB.nPel);
    const int it = 256 - time256;
    const __m256i lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i v32768 = _mm256_set1_epi32(32768), v256 = _mm256_set1_epi32(256);
    const __m256i vtime = _mm256_set1_epi32(time256), vit = _mm256_set1_epi32(it);
    auto motion = [&](const uint16_t *VX, int scale) {
        return _mm256_srai_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(ld_u16(VX), v32768), _mm256_set1_epi32(scale)), 8);
    };

    for (int h = 0; h < height; h++) {
        const int yBase = (h + dstY) << nPelLog;
        const __m256i vyBase = _mm256_set1_epi32(yBase);
        const PT *F0 = reinterpret_cast<const PT *>(prefF.GetPointer<PT>(dstX << nPelLog, yBase));
        const PT *B0 = reinterpret_cast<const PT *>(prefB.GetPointer<PT>(dstX << nPelLog, yBase));

        auto do8 = [&](int w) {
            const __m256i xBase = _mm256_slli_epi32(_mm256_add_epi32(lanes, _mm256_set1_epi32(w + dstX)), nPelLog);
            __m256i F = mc_gather<S>(prefF, _mm256_add_epi32(motion(VXFullF + w, time256), xBase), _mm256_add_epi32(motion(VYFullF + w, time256), vyBase));
            __m256i B = mc_gather<S>(prefB, _mm256_add_epi32(motion(VXFullB + w, it), xBase), _mm256_add_epi32(motion(VYFullB + w, it), vyBase));
            __m256i F0v = ld_pix<S>((const uint8_t *)(F0 + w)), B0v = ld_pix<S>((const uint8_t *)(B0 + w));
            __m256i mF = ld_u16(MaskF + w), mB = ld_u16(MaskB + w);
            __m256i imF = _mm256_sub_epi32(v256, mF), imB = _mm256_sub_epi32(v256, mB);
            __m256i FmF = _mm256_mullo_epi32(F, imF), BmB = _mm256_mullo_epi32(B, imB);
            __m256i innerA = _mm256_add_epi32(BmB, _mm256_mullo_epi32(mB, F0v));
            __m256i innerB = _mm256_add_epi32(FmF, _mm256_mullo_epi32(mF, B0v));
            __m256i pA = mul_round8(mF, innerA), pB = mul_round8(mB, innerB);
            __m256i term1 = _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(FmF, pA), v256), 8);
            __m256i term2 = _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(BmB, pB), v256), 8);
            __m256i out = _mm256_srli_epi32(_mm256_add_epi32(_mm256_mullo_epi32(term1, vit), _mm256_mullo_epi32(term2, vtime)), 8);
            st_pix<S>(pdst + (size_t)w * S, _mm256_sub_epi32(out, _mm256_set1_epi32(1)));
        };
        int w = 0;
        for (; w + 8 <= width; w += 8) do8(w);
        if (w < width) do8(width - 8); // overlapping last group (idempotent)

        pdst += dst_pitch;
        VXFullB += tp; VYFullB += tp; VXFullF += tp; VYFullF += tp; MaskB += tp; MaskF += tp;
    }
}

template <int S>
static void flowinterextra_impl(uint8_t *MVU_RESTRICT pdst, ptrdiff_t dst_pitch,
        const PyramidPlane &prefB, const PyramidPlane &prefF,
        const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
        const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch,
        int dstX, int dstY, int width, int height, int time256,
        const uint16_t *VXFullBB, const uint16_t *VXFullFF, const uint16_t *VYFullBB, const uint16_t *VYFullFF) noexcept {
    using PT = typename std::conditional<S == 1, uint8_t, uint16_t>::type;
    if (width < 8) {
        FlowInterExtra_scalar<PT>(pdst, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256, VXFullBB, VXFullFF, VYFullBB, VYFullFF);
        return;
    }
    const ptrdiff_t tp = tilePitch / (ptrdiff_t)sizeof(int16_t);
    const int nPelLog = ilog2(prefB.nPel);
    const int it = 256 - time256;
    const __m256i lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i v32768 = _mm256_set1_epi32(32768), v256 = _mm256_set1_epi32(256);
    const __m256i vtime = _mm256_set1_epi32(time256), vit = _mm256_set1_epi32(it);
    auto motion = [&](const uint16_t *VX, int scale) {
        return _mm256_srai_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(ld_u16(VX), v32768), _mm256_set1_epi32(scale)), 8);
    };

    for (int h = 0; h < height; h++) {
        const int yBase = (h + dstY) << nPelLog;
        const __m256i vyBase = _mm256_set1_epi32(yBase);
        auto do8 = [&](int w) {
            const __m256i xBase = _mm256_slli_epi32(_mm256_add_epi32(lanes, _mm256_set1_epi32(w + dstX)), nPelLog);
            __m256i F  = mc_gather<S>(prefF, _mm256_add_epi32(motion(VXFullF + w, time256), xBase),  _mm256_add_epi32(motion(VYFullF + w, time256), vyBase));
            __m256i FF = mc_gather<S>(prefF, _mm256_add_epi32(motion(VXFullFF + w, time256), xBase), _mm256_add_epi32(motion(VYFullFF + w, time256), vyBase));
            __m256i B  = mc_gather<S>(prefB, _mm256_add_epi32(motion(VXFullB + w, it), xBase),  _mm256_add_epi32(motion(VYFullB + w, it), vyBase));
            __m256i BB = mc_gather<S>(prefB, _mm256_add_epi32(motion(VXFullBB + w, it), xBase), _mm256_add_epi32(motion(VYFullBB + w, it), vyBase));
            __m256i minfb = _mm256_min_epi32(B, F), maxfb = _mm256_max_epi32(B, F);
            __m256i medianBB = _mm256_max_epi32(minfb, _mm256_min_epi32(BB, maxfb));
            __m256i medianFF = _mm256_max_epi32(minfb, _mm256_min_epi32(FF, maxfb));
            __m256i mF = ld_u16(MaskF + w), mB = ld_u16(MaskB + w);
            __m256i t1 = _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(_mm256_mullo_epi32(medianBB, mF), _mm256_mullo_epi32(F, _mm256_sub_epi32(v256, mF))), v256), 8);
            __m256i t2 = _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(_mm256_mullo_epi32(medianFF, mB), _mm256_mullo_epi32(B, _mm256_sub_epi32(v256, mB))), v256), 8);
            __m256i out = _mm256_srli_epi32(_mm256_add_epi32(_mm256_mullo_epi32(t1, vit), _mm256_mullo_epi32(t2, vtime)), 8);
            st_pix<S>(pdst + (size_t)w * S, _mm256_sub_epi32(out, _mm256_set1_epi32(1)));
        };
        int w = 0;
        for (; w + 8 <= width; w += 8) do8(w);
        if (w < width) do8(width - 8);

        pdst += dst_pitch;
        VXFullB += tp; VYFullB += tp; VXFullF += tp; VYFullF += tp; MaskB += tp; MaskF += tp;
        VXFullBB += tp; VYFullBB += tp; VXFullFF += tp; VYFullFF += tp;
    }
}

// flowFetch: motion-compensated copy -- one gather + store per pixel, no blend (motion rounds +128).
template <int S>
static void flowfetch_impl(uint8_t *MVU_RESTRICT pdst, ptrdiff_t dst_pitch, const PyramidPlane &pref,
        const uint16_t *VXFull, const uint16_t *VYFull, ptrdiff_t tilePitch,
        int dstX, int dstY, int width, int height, int time256) noexcept {
    using PT = typename std::conditional<S == 1, uint8_t, uint16_t>::type;
    if (width < 8) {
        FlowFetch_scalar<PT>(pdst, dst_pitch, pref, VXFull, VYFull, tilePitch, dstX, dstY, width, height, time256);
        return;
    }
    const ptrdiff_t tp = tilePitch / (ptrdiff_t)sizeof(uint16_t);
    const int nPelLog = ilog2(pref.nPel);
    const __m256i lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i v32768 = _mm256_set1_epi32(32768), v128 = _mm256_set1_epi32(128), vtime = _mm256_set1_epi32(time256);
    auto motion = [&](const uint16_t *V) { // ((V - 32768)*time256 + 128) >> 8
        return _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(ld_u16(V), v32768), vtime), v128), 8);
    };

    for (int h = 0; h < height; h++) {
        const int yBase = (h + dstY) << nPelLog;
        const __m256i vyBase = _mm256_set1_epi32(yBase);
        auto do8 = [&](int w) {
            const __m256i xBase = _mm256_slli_epi32(_mm256_add_epi32(lanes, _mm256_set1_epi32(w + dstX)), nPelLog);
            __m256i g = mc_gather<S>(pref, _mm256_add_epi32(motion(VXFull + w), xBase), _mm256_add_epi32(motion(VYFull + w), vyBase));
            st_pix<S>(pdst + (size_t)w * S, g);
        };
        int w = 0;
        for (; w + 8 <= width; w += 8) do8(w);
        if (w < width) do8(width - 8); // overlapping last group (idempotent)

        pdst += dst_pitch;
        VXFull += tp; VYFull += tp;
    }
}

void FlowInter_avx2_u8(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &prefB, const PyramidPlane &prefF,
        const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
        const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256) noexcept {
    flowinter_impl<1>(pdst, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256);
}
void FlowInter_avx2_u16(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &prefB, const PyramidPlane &prefF,
        const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
        const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256) noexcept {
    flowinter_impl<2>(pdst, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256);
}
void FlowInterExtra_avx2_u8(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &prefB, const PyramidPlane &prefF,
        const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
        const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256,
        const uint16_t *VXFullBB, const uint16_t *VXFullFF, const uint16_t *VYFullBB, const uint16_t *VYFullFF) noexcept {
    flowinterextra_impl<1>(pdst, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256, VXFullBB, VXFullFF, VYFullBB, VYFullFF);
}
void FlowInterExtra_avx2_u16(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &prefB, const PyramidPlane &prefF,
        const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
        const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256,
        const uint16_t *VXFullBB, const uint16_t *VXFullFF, const uint16_t *VYFullBB, const uint16_t *VYFullFF) noexcept {
    flowinterextra_impl<2>(pdst, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256, VXFullBB, VXFullFF, VYFullBB, VYFullFF);
}

void FlowFetch_avx2_u8(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &pref,
        const uint16_t *VXFull, const uint16_t *VYFull, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256) noexcept {
    flowfetch_impl<1>(pdst, dst_pitch, pref, VXFull, VYFull, tilePitch, dstX, dstY, width, height, time256);
}
void FlowFetch_avx2_u16(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &pref,
        const uint16_t *VXFull, const uint16_t *VYFull, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256) noexcept {
    flowfetch_impl<2>(pdst, dst_pitch, pref, VXFull, VYFull, tilePitch, dstX, dstY, width, height, time256);
}

#endif // MVTOOLS_X86
