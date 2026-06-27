#pragma once

#include <vector>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#ifdef _WIN32
#include <malloc.h>
#else 
#include <cstdlib>
#endif
#include <VapourSynth4.h>

#define MVU_RESTRICT __restrict

#ifdef _MSC_VER
#define MVU_FORCE_INLINE __forceinline
#else
#define MVU_FORCE_INLINE inline __attribute__((always_inline))
#endif

static constexpr size_t MVU_MEMORY_ALIGN = 64;

class MVUtensilsError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

template<typename T, size_t U>
constexpr int ARRAY_SIZE(const T (&)[U]) {
    return static_cast<int>(U);
}

template<typename T>
static void VS_CC filterFree(void *instanceData, [[maybe_unused]] VSCore *core, [[maybe_unused]] const VSAPI *vsapi) {
    delete reinterpret_cast<T *>(instanceData);
}

constexpr int RoundUpToAlignment(int value, int alignment = MVU_MEMORY_ALIGN) {
    return ((value + alignment - 1) / alignment) * alignment;
}

constexpr int ERROR_SIZE = 1024;

static constexpr const int MV_DEFAULT_SCD1 = 400;
static constexpr const int MV_DEFAULT_SCD2 = 130;

static constexpr char DEFAULT_MVUTENSILS_PREFIX[] = "MVUtensils";

static inline void mvu_bitblt(void *dstp, ptrdiff_t dst_stride, const void *srcp, ptrdiff_t src_stride, size_t row_size, size_t height) {
    if (height) {
        if (src_stride == dst_stride && src_stride == (ptrdiff_t)row_size) {
            memcpy(dstp, srcp, row_size * height);
        } else {
            const uint8_t *srcp8 = (const uint8_t *)srcp;
            uint8_t *dstp8 = (uint8_t *)dstp;
            size_t i;
            for (i = 0; i < height; i++) {
                memcpy(dstp8, srcp8, row_size);
                srcp8 += src_stride;
                dstp8 += dst_stride;
            }
        }
    }
}

/* returns the biggest integer x such as 2^x <= i */
static constexpr inline int ilog2(int i) {
    int result = 0;
    while (i > 1) {
        i /= 2;
        result++;
    }
    return result;
}

template<typename T>
static inline T *mvu_aligned_malloc(size_t size, size_t alignment) {
#ifdef _WIN32
    return (T *)_aligned_malloc(size, alignment);
#else
    void *tmp = NULL;
    if (posix_memalign(&tmp, alignment, size))
        tmp = 0;
    return (T *)tmp;
#endif
}

static inline void mvu_aligned_free(void *ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else 
    free(ptr);
#endif
}

// Resolve the field order for frame n from its frame properties.
// Reads _Field from props. If the property is missing, throws std::runtime_error
// if requireField is true and tff_exists is false.
// When tff_exists is true, _Field is ignored entirely and tff XOR-flipped by frame
// parity is used instead.
inline bool GetTopField(const VSFrame *propsSrc, int n, bool tff_exists, bool tff, bool requireField, const VSAPI *vsapi) {
    int err;
    const VSMap *props = vsapi->getFramePropertiesRO(propsSrc);
    bool top_field = !!vsapi->mapGetInt(props, "_Field", 0, &err);
    if (err && requireField && !tff_exists)
        throw std::runtime_error("_Field property not found in input frame. Therefore, you must pass tff argument");
    if (tff_exists)
        top_field = tff ^ ((n % 2) != 0);
    return top_field;
}

// Compute the sub-pixel vertical field shift between src and ref frames.
// Returns nPel/2 if src is top-field and ref is bottom-field,
//        -nPel/2 if ref is top-field and src is bottom-field,
//         0      if both fields have the same parity.
inline int ComputeFieldShift(bool src_top_field, bool ref_top_field, int nPel) noexcept {
    return (src_top_field && !ref_top_field) ? nPel / 2 : ((ref_top_field && !src_top_field) ? -(nPel / 2) : 0);
}

template <typename PixelType>
static constexpr PixelType AveragePixels(PixelType p1, PixelType p2) noexcept {
    if constexpr (std::is_integral_v<PixelType>)
        return (p1 + p2 + 1) >> 1;
    else
        return (p1 + p2) * 0.5f;
}

template <typename PixelType>
static constexpr PixelType AveragePixels(PixelType p1, PixelType p2, PixelType p3, PixelType p4) noexcept {
    if constexpr (std::is_integral_v<PixelType>)
        return (p1 + p2 + p3 + p4 + 2) >> 2;
    else
        return (p1 + p2 + p3 + p4) * 0.25f;
}

template <typename T>
static constexpr T ClampIntToRange(T p, int maxVal) noexcept {
    if constexpr (std::is_integral_v<T>)
        return std::clamp(p, 0, maxVal);
    else
        return p;
}

template <int rShift, int roundBias, typename T>
static constexpr T ShiftDivide(T p) noexcept {
    if constexpr (std::is_integral_v<T>)
        return (p + roundBias) >> rShift;
    else
        return p / (1 << rShift);
}

template <typename PixelType>
static constexpr int PixelMaxValue(int bitsPerSample) noexcept {
    if constexpr (std::is_integral_v<PixelType>)
        return (1 << bitsPerSample) - 1;
    else
        return 0;
}

