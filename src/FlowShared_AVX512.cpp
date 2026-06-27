// AVX-512 gather implementations of the motion-compensated FlowInter / FlowInterExtra kernels
// declared in FlowShared.h. The scalar versions issue one PyramidPlane::GetPointer() load per pixel;
// here we vectorise 16 output pixels at a time and replace the per-pixel pointer math + load with a
// single vpgatherdd.
//
// SUBPEL ADDRESSING (the reason a gather works): after the "rework subpel memory layout" change all
// nPel*nPel offset sub-planes live in ONE allocation reachable from pPlane[0]. GetPointer(nX,nY) is
// (unifying pel 1/2/4 with mask = nPel-1, log = ilog2(nPel)):
//     X = nX + nHPaddingPel;  Y = nY + nVPaddingPel
//     idx = (X & mask) | ((Y & mask) << log)                      // which sub-plane
//     byteOffset = idx*subPelPlaneOffset + (X>>log)*sizeof(PT) + (Y>>log)*nPitch
//     pixel = *(PT*)(pPlane[0] + byteOffset)
// The gather feeds vpgatherdd (scale 1) with base = the MIDDLE sub-plane pPlane[nPel*nPel/2] and signed
// index (byteOffset - mid*subPelPlaneOffset); base+index is the same address. Centering keeps the index
// in ~[-alloc/2, +alloc/2] so it fits int32 for twice the allocation a pPlane[0] base would allow.
//
// OVER-READ: gathers are 32-bit (vpgatherdd reads 4 bytes per lane) while a pixel is 1 or 2 bytes.
//   - End: at the highest in-bounds pixel the 4-byte load reads up to (4 - sizeof(PT)) bytes past the
//     end of the pPlane[0] allocation -> at most 3 bytes (8-bit) / 2 bytes (16-bit).
//   - Begin: no read precedes pPlane[0] beyond what the scalar path already does -- motion vectors are
//     bounded by the plane padding exactly as in the scalar code; the 4-byte granularity only extends
//     forward. Net additional out-of-bounds touch vs scalar: <= 3 bytes after the allocation.
//   The caller's allocation must tolerate a 3-byte tail over-read. (Tail lanes w >= width are excluded
//   from the gather via the active-lane mask, so their arbitrary offsets are never dereferenced.)
//
// LIMIT: gather indices are signed int32. With the centered base the binding quantity is the larger
//   half of the allocation, (nPel*nPel - nPel*nPel/2) * subPelPlaneOffset, which must be <= INT32_MAX
//   -- i.e. the whole allocation may approach ~4 GiB (double the pPlane[0]-base ceiling). The dispatcher
//   (FlowShared.h FlowAVX512Fits) enforces this and falls back to the scalar path above it.

#include "FlowShared.h"

#if defined(MVTOOLS_X86)

#include <immintrin.h>

static MVU_FORCE_INLINE __m512i ld_u16(const uint16_t *p, __mmask16 k) {
    return _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(k, p));
}

template <int S>
static MVU_FORCE_INLINE __m512i ld_pix(const uint8_t *p, __mmask16 k) {
    if constexpr (S == 1)
        return _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(k, p));
    else
        return _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(k, (const uint16_t *)p));
}

template <int S>
static MVU_FORCE_INLINE void st_pix(uint8_t *p, __mmask16 k, __m512i v) {
    if constexpr (S == 1)
        _mm512_mask_cvtepi32_storeu_epi8(p, k, v);
    else if constexpr (S == 2)
        _mm512_mask_cvtepi32_storeu_epi16(p, k, v);
    else
        _mm512_mask_storeu_epi32(p, k, v); // float: gathered 32-bit lanes ARE the pixels, store directly
}

// (a*b + 256) >> 8 per 32-bit lane, computed through 64-bit products so it is exact even when a*b
// overflows int32 (the FlowInter "mask * inner" term reaches ~2^33 for 16-bit pixels). a,b unsigned,
// result (<= ~2^17) fits 32-bit.
static MVU_FORCE_INLINE __m512i mul_round8(__m512i a, __m512i b) {
    const __m512i c256 = _mm512_set1_epi64(256);
    __m512i ev = _mm512_srli_epi64(_mm512_add_epi64(_mm512_mul_epu32(a, b), c256), 8);
    __m512i od = _mm512_srli_epi64(_mm512_add_epi64(_mm512_mul_epu32(_mm512_srli_epi64(a, 32), _mm512_srli_epi64(b, 32)), c256), 8);
    return _mm512_mask_blend_epi32(0xAAAA, ev, _mm512_slli_epi64(od, 32)); // even lanes <- ev, odd <- od
}

// 16 motion-compensated pixels from plane p at (nX[lane], nY[lane]), zero-extended to int32.
// The gather base is the MIDDLE sub-plane (pPlane[nPel*nPel/2]) rather than pPlane[0], so the signed
// int32 indices span ~[-alloc/2, +alloc/2] instead of [0, alloc]. That halves the largest |index| and
// thus doubles the allocation size the int32 gather can address. base + (idx - mid)*spo + within is the
// same absolute pixel address as the pPlane[0]-relative form.
template <int S>
static MVU_FORCE_INLINE __m512i mc_gather(const PyramidPlane &p, __m512i nX, __m512i nY, __mmask16 k) {
    const int log = ilog2(p.nPel);
    const int mid = (p.nPel * p.nPel) / 2; // middle sub-plane -> centered offsets
    const __m512i mask = _mm512_set1_epi32(p.nPel - 1);
    const __m512i X = _mm512_add_epi32(nX, _mm512_set1_epi32(p.nHPaddingPel));
    const __m512i Y = _mm512_add_epi32(nY, _mm512_set1_epi32(p.nVPaddingPel));
    __m512i idx = _mm512_or_epi32(_mm512_and_epi32(X, mask), _mm512_slli_epi32(_mm512_and_epi32(Y, mask), log));
    __m512i off = _mm512_mullo_epi32(_mm512_sub_epi32(idx, _mm512_set1_epi32(mid)), _mm512_set1_epi32((int)p.subPelPlaneOffset));
    off = _mm512_add_epi32(off, _mm512_slli_epi32(_mm512_srai_epi32(X, log), S == 4 ? 2 : (S == 2 ? 1 : 0)));  // (X>>log)*sizeof(PT)
    off = _mm512_add_epi32(off, _mm512_mullo_epi32(_mm512_srai_epi32(Y, log), _mm512_set1_epi32((int)p.nPitch))); // (Y>>log)*nPitch
    __m512i g = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), k, off, (const void *)p.pPlane[mid], 1);
    return _mm512_and_si512(g, _mm512_set1_epi32(S == 1 ? 0xFF : (S == 2 ? 0xFFFF : -1))); // float (S==4): keep all 32 bits
}

template <int S>
static void flowinter_impl(uint8_t *MVU_RESTRICT pdst, ptrdiff_t dst_pitch,
        const PyramidPlane &prefB, const PyramidPlane &prefF,
        const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
        const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch,
        int dstX, int dstY, int width, int height, int time256) noexcept {
    using PT = typename std::conditional<S == 1, uint8_t, uint16_t>::type;
    tilePitch /= sizeof(uint16_t);
    const int nPelLog = ilog2(prefB.nPel);
    const int it = 256 - time256;
    const __m512i lanes = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m512i v32768 = _mm512_set1_epi32(32768), v256 = _mm512_set1_epi32(256);
    const __m512i vtime = _mm512_set1_epi32(time256), vit = _mm512_set1_epi32(it);

    auto motion = [&](const uint16_t *VX, int scale, __mmask16 k) {
        return _mm512_srai_epi32(_mm512_mullo_epi32(_mm512_sub_epi32(ld_u16(VX, k), v32768), _mm512_set1_epi32(scale)), 8);
    };

    for (int h = 0; h < height; h++) {
        const int yBase = (h + dstY) << nPelLog;
        const __m512i vyBase = _mm512_set1_epi32(yBase);
        const PT *F0 = reinterpret_cast<const PT *>(prefF.GetPointer<PT>(dstX << nPelLog, yBase));
        const PT *B0 = reinterpret_cast<const PT *>(prefB.GetPointer<PT>(dstX << nPelLog, yBase));

        for (int w = 0; w < width; w += 16) {
            const __mmask16 k = (width - w >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << (width - w)) - 1);
            const __m512i xBase = _mm512_slli_epi32(_mm512_add_epi32(lanes, _mm512_set1_epi32(w + dstX)), nPelLog);

            __m512i F = mc_gather<S>(prefF, _mm512_add_epi32(motion(VXFullF + w, time256, k), xBase),
                                            _mm512_add_epi32(motion(VYFullF + w, time256, k), vyBase), k);
            __m512i B = mc_gather<S>(prefB, _mm512_add_epi32(motion(VXFullB + w, it, k), xBase),
                                            _mm512_add_epi32(motion(VYFullB + w, it, k), vyBase), k);
            __m512i F0v = ld_pix<S>((const uint8_t *)(F0 + w), k);
            __m512i B0v = ld_pix<S>((const uint8_t *)(B0 + w), k);
            __m512i mF = ld_u16(MaskF + w, k), mB = ld_u16(MaskB + w, k);
            __m512i imF = _mm512_sub_epi32(v256, mF), imB = _mm512_sub_epi32(v256, mB);

            __m512i FmF = _mm512_mullo_epi32(F, imF);
            __m512i BmB = _mm512_mullo_epi32(B, imB);
            __m512i innerA = _mm512_add_epi32(BmB, _mm512_mullo_epi32(mB, F0v)); // dstB*(256-mB) + mB*dstF0
            __m512i innerB = _mm512_add_epi32(FmF, _mm512_mullo_epi32(mF, B0v)); // dstF*(256-mF) + mF*dstB0
            __m512i pA = mul_round8(mF, innerA);                                 // (mF*innerA + 256) >> 8
            __m512i pB = mul_round8(mB, innerB);
            __m512i term1 = _mm512_srli_epi32(_mm512_add_epi32(_mm512_add_epi32(FmF, pA), v256), 8);
            __m512i term2 = _mm512_srli_epi32(_mm512_add_epi32(_mm512_add_epi32(BmB, pB), v256), 8);
            __m512i out = _mm512_srli_epi32(_mm512_add_epi32(_mm512_mullo_epi32(term1, vit), _mm512_mullo_epi32(term2, vtime)), 8);
            out = _mm512_sub_epi32(out, _mm512_set1_epi32(1));
            st_pix<S>(pdst + (size_t)w * S, k, out);
        }

        pdst += dst_pitch;
        VXFullB += tilePitch; VYFullB += tilePitch; VXFullF += tilePitch; VYFullF += tilePitch;
        MaskB += tilePitch;  MaskF += tilePitch;
    }
}

template <int S>
static void flowinterextra_impl(uint8_t *MVU_RESTRICT pdst, ptrdiff_t dst_pitch,
        const PyramidPlane &prefB, const PyramidPlane &prefF,
        const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
        const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch,
        int dstX, int dstY, int width, int height, int time256,
        const uint16_t *VXFullBB, const uint16_t *VXFullFF, const uint16_t *VYFullBB, const uint16_t *VYFullFF) noexcept {
    tilePitch /= sizeof(int16_t);
    const int nPelLog = ilog2(prefB.nPel);
    const int it = 256 - time256;
    const __m512i lanes = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m512i v32768 = _mm512_set1_epi32(32768), v256 = _mm512_set1_epi32(256);
    const __m512i vtime = _mm512_set1_epi32(time256), vit = _mm512_set1_epi32(it);

    auto motion = [&](const uint16_t *VX, int scale, __mmask16 k) {
        return _mm512_srai_epi32(_mm512_mullo_epi32(_mm512_sub_epi32(ld_u16(VX, k), v32768), _mm512_set1_epi32(scale)), 8);
    };

    for (int h = 0; h < height; h++) {
        const int yBase = (h + dstY) << nPelLog;
        const __m512i vyBase = _mm512_set1_epi32(yBase);
        for (int w = 0; w < width; w += 16) {
            const __mmask16 k = (width - w >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << (width - w)) - 1);
            const __m512i xBase = _mm512_slli_epi32(_mm512_add_epi32(lanes, _mm512_set1_epi32(w + dstX)), nPelLog);

            __m512i F  = mc_gather<S>(prefF, _mm512_add_epi32(motion(VXFullF + w, time256, k), xBase),  _mm512_add_epi32(motion(VYFullF + w, time256, k), vyBase), k);
            __m512i FF = mc_gather<S>(prefF, _mm512_add_epi32(motion(VXFullFF + w, time256, k), xBase), _mm512_add_epi32(motion(VYFullFF + w, time256, k), vyBase), k);
            __m512i B  = mc_gather<S>(prefB, _mm512_add_epi32(motion(VXFullB + w, it, k), xBase),  _mm512_add_epi32(motion(VYFullB + w, it, k), vyBase), k);
            __m512i BB = mc_gather<S>(prefB, _mm512_add_epi32(motion(VXFullBB + w, it, k), xBase), _mm512_add_epi32(motion(VYFullBB + w, it, k), vyBase), k);

            __m512i minfb = _mm512_min_epi32(B, F), maxfb = _mm512_max_epi32(B, F);
            __m512i medianBB = _mm512_max_epi32(minfb, _mm512_min_epi32(BB, maxfb));
            __m512i medianFF = _mm512_max_epi32(minfb, _mm512_min_epi32(FF, maxfb));

            __m512i mF = ld_u16(MaskF + w, k), mB = ld_u16(MaskB + w, k);
            __m512i t1 = _mm512_srli_epi32(_mm512_add_epi32(_mm512_add_epi32(_mm512_mullo_epi32(medianBB, mF), _mm512_mullo_epi32(F, _mm512_sub_epi32(v256, mF))), v256), 8);
            __m512i t2 = _mm512_srli_epi32(_mm512_add_epi32(_mm512_add_epi32(_mm512_mullo_epi32(medianFF, mB), _mm512_mullo_epi32(B, _mm512_sub_epi32(v256, mB))), v256), 8);
            __m512i out = _mm512_srli_epi32(_mm512_add_epi32(_mm512_mullo_epi32(t1, vit), _mm512_mullo_epi32(t2, vtime)), 8);
            out = _mm512_sub_epi32(out, _mm512_set1_epi32(1));
            st_pix<S>(pdst + (size_t)w * S, k, out);
        }
        pdst += dst_pitch;
        VXFullB += tilePitch; VYFullB += tilePitch; VXFullF += tilePitch; VYFullF += tilePitch;
        MaskB += tilePitch;  MaskF += tilePitch;
        VXFullBB += tilePitch; VYFullBB += tilePitch; VXFullFF += tilePitch; VYFullFF += tilePitch;
    }
}

// flowFetch: motion-compensated copy -- one gather + store per pixel, no blend. This is the purest
// gather kernel (the scalar has ~no arithmetic to amortise), so its speed is exactly vpgatherdd's
// throughput vs 16 scalar loads. NB the motion rounds (+128) before >>8, unlike FlowInter (truncates).
template <int S>
static void flowfetch_impl(uint8_t *MVU_RESTRICT pdst, ptrdiff_t dst_pitch, const PyramidPlane &pref,
        const uint16_t *VXFull, const uint16_t *VYFull, ptrdiff_t tilePitch,
        int dstX, int dstY, int width, int height, int time256) noexcept {
    tilePitch /= sizeof(uint16_t);
    const int nPelLog = ilog2(pref.nPel);
    const __m512i lanes = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m512i v32768 = _mm512_set1_epi32(32768), v128 = _mm512_set1_epi32(128), vtime = _mm512_set1_epi32(time256);

    auto motion = [&](const uint16_t *V, __mmask16 k) { // ((V - 32768)*time256 + 128) >> 8
        return _mm512_srai_epi32(_mm512_add_epi32(_mm512_mullo_epi32(_mm512_sub_epi32(ld_u16(V, k), v32768), vtime), v128), 8);
    };

    for (int h = 0; h < height; h++) {
        const int yBase = (h + dstY) << nPelLog;
        const __m512i vyBase = _mm512_set1_epi32(yBase);
        for (int w = 0; w < width; w += 16) {
            const __mmask16 k = (width - w >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << (width - w)) - 1);
            const __m512i xBase = _mm512_slli_epi32(_mm512_add_epi32(lanes, _mm512_set1_epi32(w + dstX)), nPelLog);
            __m512i g = mc_gather<S>(pref, _mm512_add_epi32(motion(VXFull + w, k), xBase),
                                           _mm512_add_epi32(motion(VYFull + w, k), vyBase), k);
            st_pix<S>(pdst + (size_t)w * S, k, g);
        }
        pdst += dst_pitch;
        VXFull += tilePitch; VYFull += tilePitch;
    }
}

void FlowInter_avx512_u8(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &prefB, const PyramidPlane &prefF,
        const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
        const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256) noexcept {
    flowinter_impl<1>(pdst, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256);
}
void FlowInter_avx512_u16(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &prefB, const PyramidPlane &prefF,
        const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
        const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256) noexcept {
    flowinter_impl<2>(pdst, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256);
}
void FlowInterExtra_avx512_u8(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &prefB, const PyramidPlane &prefF,
        const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
        const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256,
        const uint16_t *VXFullBB, const uint16_t *VXFullFF, const uint16_t *VYFullBB, const uint16_t *VYFullFF) noexcept {
    flowinterextra_impl<1>(pdst, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256, VXFullBB, VXFullFF, VYFullBB, VYFullFF);
}
void FlowInterExtra_avx512_u16(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &prefB, const PyramidPlane &prefF,
        const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
        const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256,
        const uint16_t *VXFullBB, const uint16_t *VXFullFF, const uint16_t *VYFullBB, const uint16_t *VYFullFF) noexcept {
    flowinterextra_impl<2>(pdst, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256, VXFullBB, VXFullFF, VYFullBB, VYFullFF);
}
void FlowInter_avx512_f32(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &prefB, const PyramidPlane &prefF,
        const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
        const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256) noexcept {
    // Hand gather-blend intrinsics (vpgatherdd or vgatherqps, ±FMA) all measured SLOWER than the compiler's
    // own auto-vectorization of the scalar at this ISA -- the win is its cross-iteration scheduling, not the
    // instruction selection. So the AVX-512 "intrinsic" IS the auto-vectorized scalar, instantiated in this
    // /arch:AVX512 TU. (AVX2 keeps its hand intrinsic, which DOES beat auto-vec scalar@avx2 -- see bench.)
    FlowInter_scalar<float>(pdst, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256);
}
void FlowInterExtra_avx512_f32(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &prefB, const PyramidPlane &prefF,
        const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
        const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256,
        const uint16_t *VXFullBB, const uint16_t *VXFullFF, const uint16_t *VYFullBB, const uint16_t *VYFullFF) noexcept {
    FlowInterExtra_scalar<float>(pdst, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256, VXFullBB, VXFullFF, VYFullBB, VYFullFF);
}

void FlowFetch_avx512_u8(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &pref,
        const uint16_t *VXFull, const uint16_t *VYFull, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256) noexcept {
    flowfetch_impl<1>(pdst, dst_pitch, pref, VXFull, VYFull, tilePitch, dstX, dstY, width, height, time256);
}
void FlowFetch_avx512_u16(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &pref,
        const uint16_t *VXFull, const uint16_t *VYFull, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256) noexcept {
    flowfetch_impl<2>(pdst, dst_pitch, pref, VXFull, VYFull, tilePitch, dstX, dstY, width, height, time256);
}
void FlowFetch_avx512_f32(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &pref,
        const uint16_t *VXFull, const uint16_t *VYFull, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256) noexcept {
    flowfetch_impl<4>(pdst, dst_pitch, pref, VXFull, VYFull, tilePitch, dstX, dstY, width, height, time256);
}

#endif // MVTOOLS_X86
