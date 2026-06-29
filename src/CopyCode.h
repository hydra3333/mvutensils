#pragma once

#include <cstdint>
#include <cstddef>


using COPYFunction = void (*)(uint8_t *pDst, ptrdiff_t nDstPitch,
                              const uint8_t *pSrc, ptrdiff_t nSrcPitch);


COPYFunction selectCopyFunction(unsigned width, unsigned height, unsigned bits);
