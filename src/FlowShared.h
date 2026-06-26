#pragma once

#include <cstdint>
#include <algorithm>
#include <type_traits>
#include "SuperPyramid.h"
#include "Common.h"
#include "CPU.h"

#if defined(MVTOOLS_X86)
// AVX-512 gather kernels (FlowShared_AVX512.cpp). See that file for the subpel-gather addressing and
// the documented <= 3-byte tail over-read of the pPlane[0] allocation.
void FlowInter_avx512_u8(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &prefB, const PyramidPlane &prefF,
    const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
    const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256) noexcept;
void FlowInter_avx512_u16(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &prefB, const PyramidPlane &prefF,
    const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
    const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256) noexcept;
void FlowInterExtra_avx512_u8(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &prefB, const PyramidPlane &prefF,
    const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
    const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256,
    const uint16_t *VXFullBB, const uint16_t *VXFullFF, const uint16_t *VYFullBB, const uint16_t *VYFullFF) noexcept;
void FlowInterExtra_avx512_u16(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &prefB, const PyramidPlane &prefF,
    const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
    const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256,
    const uint16_t *VXFullBB, const uint16_t *VXFullFF, const uint16_t *VYFullBB, const uint16_t *VYFullFF) noexcept;

// The gather kernels compute signed int32 indices relative to the MIDDLE sub-plane (pPlane[nPel*nPel/2]),
// so the largest |index| is the bigger half of the allocation: (nPel*nPel - nPel*nPel/2) sub-planes.
// That must fit int32 or the index would wrap; we fall back to scalar above it (the scalar path uses
// 64-bit pointer math and has no limit). Centering roughly doubles the ceiling vs a pPlane[0] base.
static MVU_FORCE_INLINE bool FlowAVX512Fits(const PyramidPlane &p) {
    const int n = p.nPel * p.nPel;
    return (int64_t)p.subPelPlaneOffset * (n - n / 2) <= INT32_MAX;
}
#endif

// time-weihted blend src with ref frames (used for interpolation for poor motion estimation)
template <typename PixelType>
static void Blend(uint8_t *MVU_RESTRICT pdst, const uint8_t *MVU_RESTRICT psrc, const uint8_t *MVU_RESTRICT pref, int height, int width, ptrdiff_t stride, int time256) noexcept {
    for (int h = 0; h < height; h++) {
        // Hoist the row casts out of the inner loop; recomputing them per pixel
        // defeats autovectorization (the result is the same unit-stride row).
        const PixelType *psrc_ = (const PixelType *)psrc;
        const PixelType *pref_ = (const PixelType *)pref;
        PixelType *pdst_ = (PixelType *)pdst;

        for (int w = 0; w < width; w++) {
            pdst_[w] = (psrc_[w] * (256 - time256) + pref_[w] * time256) >> 8;
        }
        pdst += stride;
        psrc += stride;
        pref += stride;
    }
}

// Loop-level hoisting accessor mirroring the AVX-512 gather. Copying the plane fields into locals once
// (rather than re-reading them through PyramidPlane::GetPointer every pixel) is what lets the compiler
// keep them in registers and auto-vectorize the inner loops -- a branchless GetPointer alone does NOT,
// because it can't prove the dst store doesn't alias the PyramidPlane object so it reloads each call.
// Offsets are 64-bit so this stays correct at any resolution (unlike the int32 gather).
template <typename PixelType>
struct PlaneGather {
    const uint8_t *base;
    ptrdiff_t spo, pitch;
    int hpad, vpad, log, mask;
    explicit PlaneGather(const PyramidPlane &p) noexcept
        : base(p.pPlane[0]), spo(p.subPelPlaneOffset), pitch(p.nPitch),
          hpad(p.nHPaddingPel), vpad(p.nVPaddingPel), log(ilog2(p.nPel)), mask(p.nPel - 1) {}
    MVU_FORCE_INLINE int operator()(int nx, int ny) const noexcept {
        int X = nx + hpad, Y = ny + vpad;
        ptrdiff_t off = (ptrdiff_t)((X & mask) | ((Y & mask) << log)) * spo
                      + (ptrdiff_t)(X >> log) * (ptrdiff_t)sizeof(PixelType)
                      + (ptrdiff_t)(Y >> log) * pitch;
        return *reinterpret_cast<const PixelType *>(base + off);
    }
};

template <typename PixelType>
static void FlowInter_scalar(
        uint8_t *MVU_RESTRICT pdst8, ptrdiff_t dst_pitch,
        const PyramidPlane &prefB, const PyramidPlane &prefF,
        const uint16_t *VXFullB, const uint16_t *VXFullF,
        const uint16_t *VYFullB, const uint16_t *VYFullF,
        const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch,
        int dstX, int dstY,
        int width, int height,
        int time256) noexcept {

    PixelType *pdst = (PixelType *)pdst8;

    tilePitch /= sizeof(uint16_t);
    dst_pitch /= sizeof(PixelType);

    int nPelLog = ilog2(prefB.nPel);
    PlaneGather<PixelType> gF(prefF), gB(prefB);

    for (int h = 0; h < height; h++) {
        int yBase = (h + dstY) << nPelLog;
        const PixelType *prefF0Ptr = reinterpret_cast<const PixelType *>(prefF.GetPointer<PixelType>(dstX << nPelLog, yBase));
        const PixelType *prefB0Ptr = reinterpret_cast<const PixelType *>(prefB.GetPointer<PixelType>(dstX << nPelLog, yBase));
        for (int w = 0; w < width; w++) {
            int xBase = (w + dstX) << nPelLog;
            int vxF = ((static_cast<int>(VXFullF[w]) - (1 << 15)) * time256) >> 8;
            int vyF = ((static_cast<int>(VYFullF[w]) - (1 << 15)) * time256) >> 8;
            int vxB = ((static_cast<int>(VXFullB[w]) - (1 << 15)) * (256 - time256)) >> 8;
            int vyB = ((static_cast<int>(VYFullB[w]) - (1 << 15)) * (256 - time256)) >> 8;
            // Hoisted branchless loads (see PlaneGather) so this loop auto-vectorizes.
            int64_t dstF = gF(vxF + xBase, vyF + yBase);
            int64_t dstB = gB(vxB + xBase, vyB + yBase);
            int dstF0 = prefF0Ptr[w];
            int dstB0 = prefB0Ptr[w];
            pdst[w] = (PixelType)((((dstF * (256 - MaskF[w]) + ((MaskF[w] * (dstB * (256 - MaskB[w]) + MaskB[w] * dstF0) + 256) >> 8) + 256) >> 8) * (256 - time256) +
                ((dstB * (256 - MaskB[w]) + ((MaskB[w] * (dstF * (256 - MaskF[w]) + MaskF[w] * dstB0) + 256) >> 8) + 256) >> 8) * time256) >> 8) - 1;
        }

        pdst += dst_pitch;
        VXFullB += tilePitch;
        VYFullB += tilePitch;
        VXFullF += tilePitch;
        VYFullF += tilePitch;
        MaskB += tilePitch;
        MaskF += tilePitch;
    }
}

template <typename PixelType>
static void FlowInterExtra_scalar(
        uint8_t *MVU_RESTRICT pdst8, ptrdiff_t dst_pitch,
        const PyramidPlane &prefB, const PyramidPlane &prefF,
        const uint16_t *VXFullB, const uint16_t *VXFullF,
        const uint16_t *VYFullB, const uint16_t *VYFullF,
        const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch,
        int dstX, int dstY,
        int width, int height,
        int time256,
        const uint16_t *VXFullBB, const uint16_t *VXFullFF,
        const uint16_t *VYFullBB, const uint16_t *VYFullFF) noexcept {

    PixelType *pdst = (PixelType *)pdst8;

    dst_pitch /= sizeof(PixelType);
    tilePitch /= sizeof(int16_t);

    int nPelLog = ilog2(prefB.nPel);
    PlaneGather<PixelType> gF(prefF), gB(prefB);

    for (int h = 0; h < height; h++) {
        int yBase = (h + dstY) << nPelLog;
        for (int w = 0; w < width; w++) {
            int xBase = (w + dstX) << nPelLog;
            int vxF = ((static_cast<int>(VXFullF[w]) - (1 << 15)) * time256) >> 8;
            int vyF = ((static_cast<int>(VYFullF[w]) - (1 << 15)) * time256) >> 8;
            int vxFF = ((static_cast<int>(VXFullFF[w]) - (1 << 15)) * time256) >> 8;
            int vyFF = ((static_cast<int>(VYFullFF[w]) - (1 << 15)) * time256) >> 8;
            int vxB = ((static_cast<int>(VXFullB[w]) - (1 << 15)) * (256 - time256)) >> 8;
            int vyB = ((static_cast<int>(VYFullB[w]) - (1 << 15)) * (256 - time256)) >> 8;
            int vxBB = ((static_cast<int>(VXFullBB[w]) - (1 << 15)) * (256 - time256)) >> 8;
            int vyBB = ((static_cast<int>(VYFullBB[w]) - (1 << 15)) * (256 - time256)) >> 8;
            // Hoisted branchless loads (see PlaneGather) so this loop auto-vectorizes.
            int dstF = gF(vxF + xBase, vyF + yBase);
            int dstFF = gF(vxFF + xBase, vyFF + yBase);
            int dstB = gB(vxB + xBase, vyB + yBase);
            int dstBB = gB(vxBB + xBase, vyBB + yBase);

            /* use median, firsly get min max of compensations */
            int minfb = std::min(dstB, dstF);
            int maxfb = std::max(dstB, dstF);

            int medianBB = std::max(minfb, std::min(dstBB, maxfb));
            int medianFF = std::max(minfb, std::min(dstFF, maxfb));

            pdst[w] = ((((medianBB * MaskF[w] + dstF * (256 - MaskF[w]) + 256) >> 8) * (256 - time256) +
                ((medianFF * MaskB[w] + dstB * (256 - MaskB[w]) + 256) >> 8) * time256) >> 8) - 1;
        }
        pdst += dst_pitch;
        VXFullB += tilePitch;
        VYFullB += tilePitch;
        VXFullF += tilePitch;
        VYFullF += tilePitch;
        MaskB += tilePitch;
        MaskF += tilePitch;
        VXFullBB += tilePitch;
        VYFullBB += tilePitch;
        VXFullFF += tilePitch;
        VYFullFF += tilePitch;
    }
}

// Dispatch wrappers: use the AVX-512 gather kernels when available, else the scalar reference above.
template <typename PixelType>
static MVU_FORCE_INLINE void FlowInter(
        uint8_t *MVU_RESTRICT pdst8, ptrdiff_t dst_pitch,
        const PyramidPlane &prefB, const PyramidPlane &prefF,
        const uint16_t *VXFullB, const uint16_t *VXFullF,
        const uint16_t *VYFullB, const uint16_t *VYFullF,
        const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch,
        int dstX, int dstY, int width, int height, int time256) noexcept {
#if defined(MVTOOLS_X86)
    if ((g_cpuinfo & MVU_CPU_AVX512_BASE) && FlowAVX512Fits(prefB) && FlowAVX512Fits(prefF)) {
        if constexpr (sizeof(PixelType) == 1)
            FlowInter_avx512_u8(pdst8, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256);
        else
            FlowInter_avx512_u16(pdst8, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256);
        return;
    }
#endif
    FlowInter_scalar<PixelType>(pdst8, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256);
}

template <typename PixelType>
static MVU_FORCE_INLINE void FlowInterExtra(
        uint8_t *MVU_RESTRICT pdst8, ptrdiff_t dst_pitch,
        const PyramidPlane &prefB, const PyramidPlane &prefF,
        const uint16_t *VXFullB, const uint16_t *VXFullF,
        const uint16_t *VYFullB, const uint16_t *VYFullF,
        const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch,
        int dstX, int dstY, int width, int height, int time256,
        const uint16_t *VXFullBB, const uint16_t *VXFullFF,
        const uint16_t *VYFullBB, const uint16_t *VYFullFF) noexcept {
#if defined(MVTOOLS_X86)
    if ((g_cpuinfo & MVU_CPU_AVX512_BASE) && FlowAVX512Fits(prefB) && FlowAVX512Fits(prefF)) {
        if constexpr (sizeof(PixelType) == 1)
            FlowInterExtra_avx512_u8(pdst8, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256, VXFullBB, VXFullFF, VYFullBB, VYFullFF);
        else
            FlowInterExtra_avx512_u16(pdst8, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256, VXFullBB, VXFullFF, VYFullBB, VYFullFF);
        return;
    }
#endif
    FlowInterExtra_scalar<PixelType>(pdst8, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256, VXFullBB, VXFullFF, VYFullBB, VYFullFF);
}