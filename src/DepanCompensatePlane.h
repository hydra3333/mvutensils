#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include <VSHelper4.h>

#include "Common.h"
#include "DepanShared.h"


#define MIRROR_TOP 1
#define MIRROR_BOTTOM 2
#define MIRROR_LEFT 4
#define MIRROR_RIGHT 8


// Reflected/mirrored source columns can fall outside [0, row_size) for very
// large pans (the motion comes from the data clip and is never clamped to the
// frame size), so they are clamped via std::clamp(col, 0, row_size - 1) before
// indexing to avoid out-of-bounds reads.


//****************************************************************************
// move plane of nextp frame to dstp for motion compensation by trc, trm with NEAREST pixels
//
template <typename PixelType>
static void compensate_plane_nearest(uint8_t * MVU_RESTRICT dstp8, const uint8_t * MVU_RESTRICT srcp8, ptrdiff_t pitch, int row_size, int height, const transform *tr, int mirror, int border, int *work1row_size, int blurmax) {
    // if border >=0, then we fill empty edge (border) pixels by that value
    // work1row_size is work array, it must have size >= 1*row_size

    // if mirror > 0, than we fill empty edge (border) pixels by mirrored (reflected) pixels from border,
    // according to bit set of "mirror" parameter:                   (added in v.0.9)
    // mirror = 1 - only top
    // mirror = 2 - only bottom
    // mirror = 4 - only left
    // mirror = 8 - only right
    // any combination - sum of above

    const PixelType *srcp = (const PixelType *)srcp8;
    PixelType *dstp = (PixelType *)dstp8;

    pitch /= sizeof(PixelType);

    // for mirror

    bool mtop = (mirror & MIRROR_TOP) != 0;
    bool mbottom = (mirror & MIRROR_BOTTOM) != 0;
    bool mleft = (mirror & MIRROR_LEFT) != 0;
    bool mright = (mirror & MIRROR_RIGHT) != 0;

    //    select if rotation, zoom?

    if (tr->dxy == 0.0f && tr->dyx == 0.0f && tr->dxx == 1.0f && tr->dyy == 1.0f) { // only translation - fast

        const int inttr0 = (int)floorf(tr->dxc + 0.5f); // constant across the whole plane

        // out-of-frame columns: mirror or border fill (matches the per-pixel chain it replaces)
        auto edge_pixel = [&](ptrdiff_t w0, int row) {
            int rowleft = inttr0 + row;
            if (rowleft < 0 && mleft) {
                if (blurmax > 0) {
                    int blurlen = std::min(blurmax, -rowleft);
                    int64_t smoothed = 0;
                    for (int i = -rowleft - blurlen + 1; i <= -rowleft; i++)
                        smoothed += srcp[w0 + std::clamp(i, 0, row_size - 1)];
                    dstp[row] = static_cast<PixelType>(smoothed / blurlen);
                } else { // no blur
                    dstp[row] = srcp[w0 + std::clamp(-rowleft, 0, row_size - 1)]; // not very precise - may be bicubic?
                }
            } else if (rowleft >= row_size && mright) {
                if (blurmax > 0) {
                    int blurlen = std::min(blurmax, rowleft - row_size + 1);
                    int64_t smoothed = 0;
                    for (int i = row_size + row_size - rowleft - 2; i < row_size + row_size - rowleft - 2 + blurlen; i++)
                        smoothed += srcp[w0 + std::clamp(i, 0, row_size - 1)];
                    dstp[row] = static_cast<PixelType>(smoothed / blurlen);
                } else { // no blur
                    dstp[row] = srcp[w0 + std::clamp(row_size + row_size - rowleft - 2, 0, row_size - 1)]; // not very precise - may be bicubic?
                }
            } else if (border >= 0) { // if shifted point is out of frame, fill using border value
                dstp[row] = border;
            }
        };

        // central span where the source column is in-frame (0 <= inttr0+row < row_size):
        // a straight shifted copy with no per-pixel bounds test.
        const int goodstart = std::clamp(-inttr0, 0, row_size);
        const int goodend = std::clamp(row_size - inttr0, 0, row_size);

        for (int h = 0; h < height; h++) {

            float ysrc = tr->dyc + h;
            int hlow = (int)floorf(ysrc + 0.5f);

            if (hlow < 0 && mtop)
                hlow = -hlow; // mirror borders
            if (hlow >= height && mbottom)
                hlow = height + height - hlow - 2;

            ptrdiff_t w0 = hlow * pitch;
            if ((hlow >= 0) && (hlow < height)) { // middle lines
                for (int row = 0; row < goodstart; row++)
                    edge_pixel(w0, row);
                for (int row = goodstart; row < goodend; row++)
                    dstp[row] = srcp[w0 + inttr0 + row];
                for (int row = goodend; row < row_size; row++)
                    edge_pixel(w0, row);
            } else if (border >= 0) { // out lines
                for (int row = 0; row < row_size; row++) {
                    dstp[row] = border;
                }
            }

            dstp += pitch; // next line
        }
    }
    //-----------------------------------------------------------------------------
    else if (tr->dxy == 0.0f && tr->dyx == 0.0f) { // no rotation, only zoom and translation  - fast

        int *rowleftwork = work1row_size;

        // prepare positions   (they are not dependent from h) for fast processing
        for (int row = 0; row < row_size; row++) {
            float xsrc = tr->dxc + tr->dxx * row;
            rowleftwork[row] = (int)floorf(xsrc + 0.5f);
        }


        for (int h = 0; h < height; h++) {

            float ysrc = tr->dyc + tr->dyy * h;

            int hlow = (int)floorf(ysrc + 0.5f);

            if (hlow < 0 && mtop)
                hlow = -hlow; // mirror borders
            if (hlow >= height && mbottom)
                hlow = height + height - hlow - 2;

            ptrdiff_t w0 = hlow * pitch;
            if ((hlow >= 0) && (hlow < height)) { // incide


                for (int row = 0; row < row_size; row++) {

                    int rowleft = rowleftwork[row];

                    if ((rowleft >= 0) && (rowleft < row_size)) {
                        dstp[row] = srcp[w0 + rowleft];
                    } else if (rowleft < 0 && mleft) {
                        if (blurmax > 0) {
                            int blurlen = std::min(blurmax, -rowleft);
                            int64_t smoothed = 0;
                            for (int i = -rowleft - blurlen + 1; i <= -rowleft; i++)
                                smoothed += srcp[w0 + std::clamp(i, 0, row_size - 1)];
                            dstp[row] = static_cast<PixelType>(smoothed / blurlen);
                        } else { // no blur
                            dstp[row] = srcp[w0 + std::clamp(-rowleft, 0, row_size - 1)]; // not very precise - may be bicubic?
                        }
                    } else if (rowleft >= row_size && mright) {
                        if (blurmax > 0) {
                            int blurlen = std::min(blurmax, rowleft - row_size + 1);
                            int64_t smoothed = 0;
                            for (int i = row_size + row_size - rowleft - 2; i < row_size + row_size - rowleft - 2 + blurlen; i++)
                                smoothed += srcp[w0 + std::clamp(i, 0, row_size - 1)];
                            dstp[row] = static_cast<PixelType>(smoothed / blurlen);
                        } else { // no blur
                            dstp[row] = srcp[w0 + std::clamp(row_size + row_size - rowleft - 2, 0, row_size - 1)]; // not very precise - may be bicubic?
                        }
                    } else if (border >= 0) { // if shifted point is out of frame, fill using border value
                        dstp[row] = border;
                    }
                } // end for
            } else if (border >= 0) { // out lines
                for (int row = 0; row < row_size; row++) {
                    dstp[row] = border;
                }
            }

            dstp += pitch; // next line
        }
    }
    //-----------------------------------------------------------------------------
    else { // rotation, zoom and translation - slow

        for (int h = 0; h < height; h++) {

            float xsrc = tr->dxc + tr->dxy * h; // part not dependent from row
            float ysrc = tr->dyc + tr->dyy * h;

            for (int row = 0; row < row_size; row++) {

                int rowleft = (int)(xsrc + 0.5f); // use simply fast (int), not floor(), since followed check

                int hlow = (int)(ysrc + 0.5f); // use simply fast  (int), not floor(), since followed check


                if ((rowleft >= 0) && (rowleft < row_size) && (hlow >= 0) && (hlow < height)) {
                    dstp[row] = srcp[hlow * pitch + rowleft];
                } else { // try fill by mirror. Probability of these cases is small
                    if (hlow < 0 && mtop)
                        hlow = -hlow; // mirror borders
                    if (hlow >= height && mbottom)
                        hlow = height + height - hlow - 2;
                    if (rowleft < 0 && mleft)
                        rowleft = -rowleft;
                    if (rowleft >= row_size && mright)
                        rowleft = row_size + row_size - rowleft - 2;
                    // check mirrowed
                    if ((rowleft >= 0) && (rowleft < row_size) && (hlow >= 0) && (hlow < height)) {
                        dstp[row] = srcp[hlow * pitch + rowleft];
                    } else if (border >= 0) { // if shifted point is out of frame, fill using border value
                        dstp[row] = border;
                    }
                }
                xsrc += tr->dxx; // next
                ysrc += tr->dyx;
            } // end for row

            dstp += pitch; // next line
        } //end for h
    } // end if rotation
}


//****************************************************************************
// move plane of nextp frame to dstp for motion compensation by transform tr[]
// with BILINEAR interpolation of discrete neighbour source pixels
//   t[0] = dxc, t[1] = dxx, t[2] = dxy, t[3] = dyc, t[4] = dyx, t[5] = dyy
//
template <typename PixelType>
static void compensate_plane_bilinear(uint8_t * MVU_RESTRICT dstp8, const uint8_t * MVU_RESTRICT srcp8, ptrdiff_t pitch, int row_size, int height, const transform *tr, int mirror, int border, int *work2row_size, int blurmax) {
    // work2row_size is work array, it must have size >= 2*row_size

    const PixelType *srcp = (const PixelType *)srcp8;
    PixelType *dstp = (PixelType *)dstp8;

    pitch /= sizeof(PixelType);

    // for mirror
    bool mtop = (mirror & MIRROR_TOP) != 0;
    bool mbottom = (mirror & MIRROR_BOTTOM) != 0;
    bool mleft = (mirror & MIRROR_LEFT) != 0;
    bool mright = (mirror & MIRROR_RIGHT) != 0;

    // 1D bilinear interpolation weights for a sub-pixel position in the integer
    // grid (sx = i/32):  cx0 = (1-sx), cx1 = sx (and likewise cy). Scaled by 32.
    // Pure constant, so evaluate it at compile time (no runtime init, no guard).
    static constexpr std::array<int, 66> intcoef = [] {
        std::array<int, 66> t{};
        for (int i = 0; i <= 32; i++) {
            t[i * 2] = 32 - i;
            t[i * 2 + 1] = i;
        }
        return t;
    }();

    //    select if rotation, zoom?

    if (tr->dxy == 0.0f && tr->dyx == 0.0f && tr->dxx == 1.0f && tr->dyy == 1.0f) { // only translation - fast

        // x-direction quantities and the good/bad row spans are constant across the
        // whole plane (translation only), so compute them once.
        const int inttr0 = (int)floorf(tr->dxc);
        const int ix2 = 2 * ((int)floorf((tr->dxc - inttr0) * 32));

        int rowgoodstart, rowgoodend, rowbadstart, rowbadend;
        if (inttr0 >= 0) {
            rowgoodstart = 0;
            rowgoodend = std::max(0, row_size - 1 - inttr0); // clamp: empty good span for full-width pans
            rowbadstart = rowgoodend;
            rowbadend = row_size;
        } else {
            rowbadstart = 0;
            rowbadend = std::min(row_size, -inttr0); // clamp: bad span cannot exceed the row
            rowgoodstart = rowbadend;
            rowgoodend = row_size;
        }
        //                int rowgoodendpaired = (rowgoodend/2)*2; //even - but it was a little not optimal
        const int rowgoodendpaired = rowgoodstart + ((rowgoodend - rowgoodstart) / 2) * 2; //even length - small fix in v.1.8

        int prev_iy2 = -1;
        int intcoef2d[4];

        for (int h = 0; h < height; h++) {

            float ysrc = tr->dyc + h;
            int hlow = (int)floorf(ysrc);
            int iy2 = 2 * ((int)floorf((ysrc - hlow) * 32));

            if (iy2 != prev_iy2) { // sub-pixel y weight is constant in translation mode; rebuild only on change
                for (int j = 0; j < 2; j++) {
                    for (int i = 0; i < 2; i++) {
                        intcoef2d[j * 2 + i] = (intcoef[j + iy2] * intcoef[i + ix2]); // 4 coeff. for bilinear 2D
                    }
                }
                prev_iy2 = iy2;
            }

            if (hlow < 0 && mtop)
                hlow = -hlow; // mirror borders
            if (hlow >= height && mbottom)
                hlow = height + height - hlow - 2;

            ptrdiff_t w0 = hlow * pitch;

            if ((hlow >= 0) && (hlow < height - 1)) { // middle lines

                ptrdiff_t w = w0 + inttr0 + rowgoodstart;
                for (int row = rowgoodstart; row < rowgoodendpaired; row += 2) { // paired unroll for speed
                    //  x,y point is in square: (rowleft,hlow) to (rowleft+1,hlow+1)
                    dstp[row] = (intcoef2d[0] * srcp[w] + intcoef2d[1] * srcp[w + 1] + intcoef2d[2] * srcp[w + pitch] + intcoef2d[3] * srcp[w + pitch + 1]) >> 10; // i.e. divide by 32*32
                    dstp[row + 1] = (intcoef2d[0] * srcp[w + 1] + intcoef2d[1] * srcp[w + 2] + intcoef2d[2] * srcp[w + pitch + 1] + intcoef2d[3] * srcp[w + pitch + 2]) >> 10; // i.e. divide by 32*32
                    w += 2;
                }
                for (int row = std::max(rowgoodendpaired, rowgoodstart); row < rowgoodend; row++) { // process leftover odd pixel (if any); guarded against empty span
                    w = w0 + inttr0 + row;
                    dstp[row] = (intcoef2d[0] * srcp[w] + intcoef2d[1] * srcp[w + 1] +
                                 intcoef2d[2] * srcp[w + pitch] + intcoef2d[3] * srcp[w + pitch + 1]) >>
                                10; // i.e. divide by 32*32
                }
                for (int row = rowbadstart; row < rowbadend; row++) {
                    int rowleft = inttr0 + row;

                    if (rowleft < 0 && mleft) {
                        if (blurmax > 0) {
                            int blurlen = std::min(blurmax, -rowleft);
                            int64_t smoothed = 0;
                            for (int i = -rowleft - blurlen + 1; i <= -rowleft; i++)
                                smoothed += srcp[w0 + std::clamp(i, 0, row_size - 1)];
                            dstp[row] = static_cast<PixelType>(smoothed / blurlen);
                        } else { // no blur
                            dstp[row] = srcp[w0 + std::clamp(-rowleft, 0, row_size - 1)]; // not very precise - may be bicubic?
                        }
                    } else if (rowleft >= row_size - 1 && mright) {
                        if (blurmax > 0) {
                            int blurlen = std::min(blurmax, rowleft - row_size + 2);
                            int64_t smoothed = 0;
                            for (int i = row_size + row_size - rowleft - 2; i < row_size + row_size - rowleft - 2 + blurlen; i++)
                                smoothed += srcp[w0 + std::clamp(i, 0, row_size - 1)];
                            dstp[row] = static_cast<PixelType>(smoothed / blurlen);
                        } else { // no blur
                            dstp[row] = srcp[w0 + std::clamp(row_size + row_size - rowleft - 2, 0, row_size - 1)]; // not very precise - may be bicubic?
                        }
                    } else if (border >= 0) { // if shifted point is out of frame, fill using border value
                        dstp[row] = border;
                    }

                } // end for
            } else if (hlow == height - 1) { // edge (bottom) line
                for (int row = 0; row < row_size; row++) {
                    int rowleft = inttr0 + row;
                    if ((rowleft >= 0) && (rowleft < row_size)) {
                        dstp[row] = srcp[w0 + rowleft]; // nearest pixel, may be bilinear is better
                    } else if (rowleft < 0 && mleft) {
                        dstp[row] = srcp[w0 + std::clamp(-rowleft, 0, row_size - 1)];
                    } else if (rowleft >= row_size - 1 && mright) {
                        dstp[row] = srcp[w0 + std::clamp(row_size + row_size - rowleft - 2, 0, row_size - 1)];
                    } else if (border >= 0) { // left or right
                        dstp[row] = border;
                    }
                }
            } else if (border >= 0) { // out lines
                for (int row = 0; row < row_size; row++) {
                    dstp[row] = border;
                }
            }

            dstp += pitch; // next line
        }
    }
    //-----------------------------------------------------------------------------
    else if (tr->dxy == 0.0f && tr->dyx == 0.0f) { // no rotation, only zoom and translation  - fast

        int *rowleftwork = work2row_size;
        int *ix2work = rowleftwork + row_size;

        // prepare positions   (they are not dependent from h) for fast processing
        for (int row = 0; row < row_size; row++) {
            float xsrc = tr->dxc + tr->dxx * row;
            rowleftwork[row] = (int)floorf(xsrc);
            int rowleft = rowleftwork[row];
            ix2work[row] = 2 * ((int)floorf((xsrc - rowleft) * 32));
        }

        // 2D bilinear weight table (outer product of the 1D weights); a pure
        // constant, so evaluate it at compile time (no runtime init, no guard).
        static constexpr std::array<int, 66 * 66> intcoef2dzoom = [] {
            std::array<int, 66 * 66> t{};
            for (int j = 0; j < 66; j++)
                for (int i = 0; i < 66; i++)
                    t[j * 66 + i] = intcoef[j] * intcoef[i];
            return t;
        }();

        for (int h = 0; h < height; h++) {

            float ysrc = tr->dyc + tr->dyy * h;

            int hlow = (int)floorf(ysrc);
            int iy2 = 2 * ((int)floorf((ysrc - hlow) * 32));

            if (hlow < 0 && mtop)
                hlow = -hlow; // mirror borders
            if (hlow >= height && mbottom)
                hlow = height + height - hlow - 2;

            ptrdiff_t w0 = hlow * pitch;

            if ((hlow >= 0) && (hlow < height - 1)) { // incide

                const int *rowcoef = intcoef2dzoom.data() + iy2 * 66; // table row for this sub-pixel y


                for (int row = 0; row < row_size; row++) {

                    int rowleft = rowleftwork[row];

                    //  x,y point is in square: (rowleft,hlow) to (rowleft+1,hlow+1)

                    if ((rowleft >= 0) && (rowleft < row_size - 1)) {

                        int ix2 = ix2work[row];
                        ptrdiff_t w = w0 + rowleft;

                        int pixel = (rowcoef[ix2] * srcp[w] + rowcoef[ix2 + 1] * srcp[w + 1] +
                                 rowcoef[ix2 + 66] * srcp[w + pitch] + rowcoef[ix2 + 67] * srcp[w + pitch + 1]) >>
                                10;

                        dstp[row] = pixel; // maxmin disabled in v1.6
                    } else if (rowleft < 0 && mleft) {
                        if (blurmax > 0) {
                            int blurlen = std::min(blurmax, -rowleft);
                            int64_t smoothed = 0;
                            for (int i = -rowleft - blurlen + 1; i <= -rowleft; i++)
                                smoothed += srcp[w0 + std::clamp(i, 0, row_size - 1)];
                            dstp[row] = static_cast<PixelType>(smoothed / blurlen);
                        } else { // no blur
                            dstp[row] = srcp[w0 + std::clamp(-rowleft, 0, row_size - 1)];
                        }
                    } else if (rowleft >= row_size - 1 && mright) {
                        if (blurmax > 0) {
                            int blurlen = std::min(blurmax, rowleft - row_size + 2);
                            int64_t smoothed = 0;
                            for (int i = row_size + row_size - rowleft - 2; i < row_size + row_size - rowleft - 2 + blurlen; i++)
                                smoothed += srcp[w0 + std::clamp(i, 0, row_size - 1)];
                            dstp[row] = static_cast<PixelType>(smoothed / blurlen);
                        } else { // no blur
                            dstp[row] = srcp[w0 + std::clamp(row_size + row_size - rowleft - 2, 0, row_size - 1)]; // not very precise - may be bicubic?
                        }
                    } else if (border >= 0) { // if shifted point is out of frame, fill using border value
                        dstp[row] = border;
                    }
                } // end for
            } else if (hlow == height - 1) { // edge ( bottom) lines
                for (int row = 0; row < row_size; row++) {
                    int rowleft = rowleftwork[row];
                    if ((rowleft >= 0) && (rowleft < row_size)) {
                        dstp[row] = srcp[rowleft + hlow * pitch]; // nearest pixel, may be bilinear is better
                    } else if (border >= 0) { // left or right
                        dstp[row] = border;
                    }
                }
            } else if (border >= 0) { // out lines
                for (int row = 0; row < row_size; row++) {
                    dstp[row] = border;
                }
            }

            dstp += pitch; // next line
        }
    }
    //-----------------------------------------------------------------------------
    else { // rotation, zoom and translation - slow

        for (int h = 0; h < height; h++) {

            float xsrc = tr->dxc + tr->dxy * h; // part not dependent from row
            float ysrc = tr->dyc + tr->dyy * h;

            for (int row = 0; row < row_size; row++) {

                int rowleft = (int)(xsrc); // use simply fast (int), not floor(), since followed check >1
                float sx = xsrc - rowleft;
                if (sx < 0) {
                    sx += 1;
                    rowleft -= 1;
                }

                int hlow = (int)(ysrc); // use simply fast  (int), not floor(), since followed check >1
                float sy = ysrc - hlow;
                if (sy < 0) {
                    sy += 1;
                    hlow -= 1;
                }

                //  x,y point is in square: (rowleft,hlow) to (rowleft+1,hlow+1)


                if ((rowleft >= 0) && (rowleft < row_size - 1) && (hlow >= 0) && (hlow < height - 1)) {

                    int ix2 = ((int)(sx * 32)) << 1; // i.e. *2
                    int iy2 = ((int)(sy * 32)) << 1; // i.e. *2
                    ptrdiff_t w0 = rowleft + hlow * pitch;

                    int pixel = ((intcoef[ix2] * srcp[w0] + intcoef[ix2 + 1] * srcp[w0 + 1]) * intcoef[iy2] +
                             (intcoef[ix2] * srcp[w0 + pitch] + intcoef[ix2 + 1] * srcp[w0 + pitch + 1]) * intcoef[iy2 + 1]) >>
                            10;

                    dstp[row] = pixel; //maxmin disabled in v1.6
                } else {
                    if (hlow < 0 && mtop)
                        hlow = -hlow; // mirror borders
                    if (hlow >= height && mbottom)
                        hlow = height + height - hlow - 2;
                    if (rowleft < 0 && mleft)
                        rowleft = -rowleft;
                    if (rowleft >= row_size && mright)
                        rowleft = row_size + row_size - rowleft - 2;
                    // check mirrowed
                    if ((rowleft >= 0) && (rowleft < row_size) && (hlow >= 0) && (hlow < height)) {
                        dstp[row] = srcp[hlow * pitch + rowleft];
                    } else if (border >= 0) { // if shifted point is out of frame, fill using border value
                        dstp[row] = border;
                    }
                }
                xsrc += tr->dxx; // next
                ysrc += tr->dyx;
            } // end for row

            dstp += pitch; // next line
        } //end for h
    } // end if rotation
}


//****************************************************************************
// move plane of nextp frame to dstp for motion compensation by transform tr[]
// with BICUBIC interpolation of discrete neighbour source pixels
//
//   t[0] = dxc, t[1] = dxx, t[2] = dxy, t[3] = dyc, t[4] = dyx, t[5] = dyy
//
template <typename PixelType>
static void compensate_plane_bicubic(uint8_t * MVU_RESTRICT dstp8, const uint8_t * MVU_RESTRICT srcp8, ptrdiff_t pitch, int row_size, int height, const transform *tr, int mirror, int border, int *work2width1030, int blurmax, int pixel_max) {
    // work2width1030 is integer work array, it must have size >= 2*row_size

    const PixelType *srcp = (const PixelType *)srcp8;
    PixelType *dstp = (PixelType *)dstp8;

    pitch /= sizeof(PixelType);

    // for mirror
    bool mtop = (mirror & MIRROR_TOP) != 0;
    bool mbottom = (mirror & MIRROR_BOTTOM) != 0;
    bool mleft = (mirror & MIRROR_LEFT) != 0;
    bool mright = (mirror & MIRROR_RIGHT) != 0;

    // prepare interpolation coefficients tables
    // for position of xsrc in integer grid
    //        sx = (xsrc-rowleft);
    //        sy = (ysrc-hlow);
    //
    //            cx0 = -sx*(1-sx)*(1-sx);
    //            cx1 = (1-2*sx*sx+sx*sx*sx);
    //            cx2 =  sx*(1+sx-sx*sx);
    //            cx3 =  -sx*sx*(1-sx);
    //
    //            cy0 = -sy*(1-sy)*(1-sy);
    //            cy1 = (1-2*sy*sy+sy*sy*sy);
    //            cy2 =  sy*(1+sy-sy*sy);
    //            cy3 =  -sy*sy*(1-sy);
    //
    // now sx = i/256, sy = j/256  (discrete approximation)

    // float coeff. are changed by integer coeff. scaled by 256*256*256/8192 = 2048
    // Pure constant, so evaluate it at compile time (no runtime init, no guard).
    static constexpr std::array<int, 1028> intcoef = [] {
        std::array<int, 1028> t{};
        for (int i = 0; i <= 256; i++) { // 257 steps, 1028 numbers
            t[i * 4] = -((i * (256 - i) * (256 - i))) / 8192;
            t[i * 4 + 1] = (256 * 256 * 256 - 2 * 256 * i * i + i * i * i) / 8192;
            t[i * 4 + 2] = (i * (256 * 256 + 256 * i - i * i)) / 8192;
            t[i * 4 + 3] = -(i * i * (256 - i)) / 8192;
        }
        return t;
    }();

    //    select if rotation, zoom

    if (tr->dxy == 0.0f && tr->dyx == 0.0f && tr->dxx == 1.0f && tr->dyy == 1.0f) { // only translation - fast

        // x-direction quantities are constant across the whole plane
        const int inttr3 = (int)floorf(tr->dyc);
        const int inttr0 = (int)floorf(tr->dxc);
        const int ix4 = 4 * ((int)((tr->dxc - inttr0) * 256));

        // edge/border columns for a middle line (everything the central 16-tap
        // path cannot handle); identical to the per-pixel chain it replaces.
        auto middle_edge = [&](ptrdiff_t w0, int row) {
            int rowleft = inttr0 + row;
            if (rowleft < 0 && mleft) {
                if (blurmax > 0) {
                    int blurlen = std::min(blurmax, -rowleft);
                    int64_t smoothed = 0;
                    for (int i = -rowleft - blurlen + 1; i <= -rowleft; i++)
                        smoothed += srcp[w0 + std::clamp(i, 0, row_size - 1)];
                    dstp[row] = static_cast<PixelType>(smoothed / blurlen);
                } else { // no blur
                    dstp[row] = srcp[w0 + std::clamp(-rowleft, 0, row_size - 1)]; // not very precise - may be bicubic?
                }
            } else if (rowleft >= row_size && mright) {
                if (blurmax > 0) {
                    int blurlen = std::min(blurmax, rowleft - row_size + 1);
                    int64_t smoothed = 0;
                    for (int i = row_size + row_size - rowleft - 2; i < row_size + row_size - rowleft - 2 + blurlen; i++)
                        smoothed += srcp[w0 + std::clamp(i, 0, row_size - 1)];
                    dstp[row] = static_cast<PixelType>(smoothed / blurlen);
                } else { // no blur
                    dstp[row] = srcp[w0 + std::clamp(row_size + row_size - rowleft - 2, 0, row_size - 1)]; // not very precise - may be bicubic?
                }
            } else if (rowleft == 0 || rowleft == row_size - 1 || rowleft == row_size - 2) { // edges
                dstp[row] = srcp[w0 + rowleft];
            } else if (border >= 0) { // if shifted point is out of frame, fill using border value
                dstp[row] = border;
            }
        };

        // central span where the 4x4 bicubic stencil is fully in-frame (1 <= inttr0+row < row_size-2)
        const int fast_start = std::clamp(1 - inttr0, 0, row_size);
        const int fast_end = std::clamp(row_size - 2 - inttr0, fast_start, row_size);

        int prev_iy4 = -1;
        int intcoef2d[16];

        for (int h = 0; h < height; h++) {

            float ysrc = tr->dyc + h;
            int hlow = (int)floorf(ysrc);
            int iy4 = 4 * ((int)((ysrc - hlow) * 256));

            if (iy4 != prev_iy4) { // sub-pixel y weight is constant in translation mode; rebuild only on change
                for (int j = 0; j < 4; j++) {
                    for (int i = 0; i < 4; i++) {
                        intcoef2d[j * 4 + i] = ((intcoef[j + iy4] * intcoef[i + ix4])) / 2048; // 16 coeff. for bicubic 2D, scaled by 2048
                    }
                }
                prev_iy4 = iy4;
            }

            if (hlow < 0 && mtop)
                hlow = -hlow; // mirror borders
            if (hlow >= height && mbottom)
                hlow = height + height - hlow - 2;

            ptrdiff_t w0 = hlow * pitch;

            if ((hlow >= 1) && (hlow < height - 2)) { // middle lines

                for (int row = 0; row < fast_start; row++)
                    middle_edge(w0, row);
                for (int row = fast_start; row < fast_end; row++) {
                    ptrdiff_t w = w0 + inttr0 + row;

                    int pixel = (intcoef2d[0] * srcp[w - pitch - 1] + intcoef2d[1] * srcp[w - pitch] + intcoef2d[2] * srcp[w - pitch + 1] + intcoef2d[3] * srcp[w - pitch + 2] +
                             intcoef2d[4] * srcp[w - 1] + intcoef2d[5] * srcp[w] + intcoef2d[6] * srcp[w + 1] + intcoef2d[7] * srcp[w + 2] +
                             intcoef2d[8] * srcp[w + pitch - 1] + intcoef2d[9] * srcp[w + pitch] + intcoef2d[10] * srcp[w + pitch + 1] + intcoef2d[11] * srcp[w + pitch + 2] +
                             intcoef2d[12] * srcp[w + pitch * 2 - 1] + intcoef2d[13] * srcp[w + pitch * 2] + intcoef2d[14] * srcp[w + pitch * 2 + 1] + intcoef2d[15] * srcp[w + pitch * 2 + 2] + 1024) >>
                            11; // i.e. /2048

                    dstp[row] = std::clamp(pixel, 0, pixel_max);
                }
                for (int row = fast_end; row < row_size; row++)
                    middle_edge(w0, row);

            } else if (hlow == 0 || hlow == height - 2) { // near edge (top-1, bottom-1) lines
                for (int row = 0; row < row_size; row++) {
                    int rowleft = inttr0 + row;
                    float sx = tr->dxc - inttr0;
                    float sy = tr->dyc - inttr3;
                    if ((rowleft >= 0) && (rowleft < row_size - 1)) { // bug fixed for right edge in v.1.1.1
                        ptrdiff_t w = w0 + rowleft;
                        dstp[row] = (int)((1.0 - sy) * ((1.0 - sx) * srcp[w] + sx * srcp[w + 1]) +
                                          sy * ((1.0 - sx) * srcp[w + pitch] + sx * srcp[w + pitch + 1])); // bilinear
                    } else if (rowleft == row_size - 1) { // added in v.1.1.1
                        dstp[row] = srcp[rowleft + w0];
                    } else if (rowleft < 0 && mleft) {
                        dstp[row] = srcp[w0 + std::clamp(-rowleft, 0, row_size - 1)]; // not very precise - may be bicubic?
                    } else if (rowleft >= row_size && mright) {
                        dstp[row] = srcp[w0 + std::clamp(row_size + row_size - rowleft - 2, 0, row_size - 1)];
                    } else if (border >= 0) { // left or right
                        dstp[row] = border;
                    }
                }
            } else if (hlow == height - 1) { // bottom line
                for (int row = 0; row < row_size; row++) {
                    int rowleft = inttr0 + row;
                    if (rowleft >= 0 && rowleft < row_size) {
                        dstp[row] = srcp[w0 + rowleft];
                    } else if (rowleft < 0 && mleft) {
                        dstp[row] = srcp[w0 + std::clamp(-rowleft, 0, row_size - 1)]; // not very precise - may be bicubic?
                    } else if (rowleft >= row_size && mright) {
                        dstp[row] = srcp[w0 + std::clamp(row_size + row_size - rowleft - 2, 0, row_size - 1)];
                    } else if (border >= 0) { // left or right
                        dstp[row] = border;
                    }
                }
            } else if (border >= 0) { // out lines
                for (int row = 0; row < row_size; row++) {
                    dstp[row] = border;
                }
            }

            dstp += pitch; // next line
        }
    }
    //-----------------------------------------------------------------------------
    else if (tr->dxy == 0.0f && tr->dyx == 0.0f) { // no rotation, only zoom and translation  - fast

        int *rowleftwork = work2width1030;
        int *ix4work = work2width1030 + row_size;

        // prepare positions   (they are not dependent from h) for fast processing
        for (int row = 0; row < row_size; row++) {
            float xsrc = tr->dxc + tr->dxx * row;
            rowleftwork[row] = (int)floorf(xsrc);
            int rowleft = rowleftwork[row];
            ix4work[row] = 4 * ((int)((xsrc - rowleft) * 256));
        }


        for (int h = 0; h < height; h++) {

            float ysrc = tr->dyc + tr->dyy * h;

            int hlow = (int)floorf(ysrc);
            int iy4 = 4 * ((int)((ysrc - hlow) * 256));

            float sy = ysrc - hlow;

            if (hlow < 0 && mtop)
                hlow = -hlow; // mirror borders
            if (hlow >= height && mbottom)
                hlow = height + height - hlow - 2;

            ptrdiff_t w0 = hlow * pitch;
            if ((hlow >= 1) && (hlow < height - 2)) { // incide

                for (int row = 0; row < row_size; row++) {

                    int rowleft = rowleftwork[row];

                    //  x,y point is in square: (rowleft,hlow) to (rowleft+1,hlow+1)

                    if ((rowleft >= 1) && (rowleft < row_size - 2)) {

                        int ix4 = ix4work[row];
                        ptrdiff_t w = w0 + rowleft;

                        int64_t ts[4];
                        srcp -= pitch; // prev line
                        ts[0] = (intcoef[ix4] * srcp[w - 1] + intcoef[ix4 + 1] * srcp[w] + intcoef[ix4 + 2] * srcp[w + 1] + intcoef[ix4 + 3] * srcp[w + 2]);
                        srcp += pitch; // next line
                        ts[1] = (intcoef[ix4] * srcp[w - 1] + intcoef[ix4 + 1] * srcp[w] + intcoef[ix4 + 2] * srcp[w + 1] + intcoef[ix4 + 3] * srcp[w + 2]);
                        srcp += pitch; // next line
                        ts[2] = (intcoef[ix4] * srcp[w - 1] + intcoef[ix4 + 1] * srcp[w] + intcoef[ix4 + 2] * srcp[w + 1] + intcoef[ix4 + 3] * srcp[w + 2]);
                        srcp += pitch; // next line
                        ts[3] = (intcoef[ix4] * srcp[w - 1] + intcoef[ix4 + 1] * srcp[w] + intcoef[ix4 + 2] * srcp[w + 1] + intcoef[ix4 + 3] * srcp[w + 2]);

                        srcp -= (pitch << 1); // restore pointer, changed to shift in v 1.1.1

                        int64_t pixel = (intcoef[iy4] * ts[0] + intcoef[iy4 + 1] * ts[1] + intcoef[iy4 + 2] * ts[2] + intcoef[iy4 + 3] * ts[3]) >> 22;

                        dstp[row] = static_cast<PixelType>(std::clamp<int64_t>(pixel, 0, pixel_max));
                    } else if (rowleft < 0 && mleft) {
                        if (blurmax > 0) {
                            int blurlen = std::min(blurmax, -rowleft);
                            int64_t smoothed = 0;
                            for (int i = -rowleft - blurlen + 1; i <= -rowleft; i++)
                                smoothed += srcp[w0 + std::clamp(i, 0, row_size - 1)];
                            dstp[row] = static_cast<PixelType>(smoothed / blurlen);
                        } else {
                            dstp[row] = srcp[w0 + std::clamp(-rowleft, 0, row_size - 1)]; // not very precise - may be bicubic?
                        }
                    } else if (rowleft >= row_size && mright) {
                        if (blurmax > 0) {
                            int blurlen = std::min(blurmax, rowleft - row_size + 1);
                            int64_t smoothed = 0;
                            for (int i = row_size + row_size - rowleft - 2; i < row_size + row_size - rowleft - 2 + blurlen; i++)
                                smoothed += srcp[w0 + std::clamp(i, 0, row_size - 1)];
                            dstp[row] = static_cast<PixelType>(smoothed / blurlen);
                        } else { // no blur
                            dstp[row] = srcp[w0 + std::clamp(row_size + row_size - rowleft - 2, 0, row_size - 1)]; // not very precise - may be bicubic?
                        }
                    } else if (rowleft == 0 || rowleft == row_size - 1 || rowleft == row_size - 2) { // edges
                        dstp[row] = srcp[w0 + rowleft];
                    } else if (border >= 0) { // if shifted point is out of frame, fill using border value
                        dstp[row] = border;
                    }
                } // end for
            } else if (hlow == 0 || hlow == height - 2) { // near edge (top-1, bottom-1) lines
                for (int row = 0; row < row_size; row++) {
                    int rowleft = rowleftwork[row];
                    if ((rowleft >= 0) && (rowleft < row_size - 1)) { // bug fixed for right bound in v.1.10.0
                        float xsrc = tr->dxc + tr->dxx * row;
                        float sx = xsrc - rowleft;
                        ptrdiff_t w = w0 + rowleft;
                        int pixel = (int)((1.0 - sy) * ((1.0 - sx) * srcp[w] + sx * srcp[w + 1]) +
                                      sy * ((1.0 - sx) * srcp[w + pitch] + sx * srcp[w + pitch + 1])); // bilinear
                        dstp[row] = std::clamp(pixel, 0, pixel_max);
                    } else if (rowleft == row_size - 1) { // added in v.1.1.1
                        dstp[row] = srcp[rowleft + w0];
                    } else if (rowleft < 0 && mleft) {
                        dstp[row] = srcp[w0 + std::clamp(-rowleft, 0, row_size - 1)]; // not very precise - may be bicubic?
                    } else if (rowleft >= row_size && mright) {
                        dstp[row] = srcp[w0 + std::clamp(row_size + row_size - rowleft - 2, 0, row_size - 1)];
                    } else if (border >= 0) { // left or right
                        dstp[row] = border;
                    }
                }
            } else if (hlow == height - 1) { // bottom line
                for (int row = 0; row < row_size; row++) {
                    int rowleft = rowleftwork[row];
                    if (rowleft >= 0 && rowleft < row_size) {
                        dstp[row] = (srcp[w0 + rowleft] + srcp[w0 + rowleft - pitch]) / 2; // for some smoothing
                    } else if (rowleft < 0 && mleft) {
                        dstp[row] = srcp[w0 + std::clamp(-rowleft, 0, row_size - 1)];
                    } else if (rowleft >= row_size && mright) {
                        dstp[row] = srcp[w0 + std::clamp(row_size + row_size - rowleft - 2, 0, row_size - 1)];
                    } else if (border >= 0) { // left or right
                        dstp[row] = border;
                    }
                }
            } else if (border >= 0) { // out lines
                for (int row = 0; row < row_size; row++) {
                    // bug fixed here in v. 0.9.1 (access violation - bad w0)
                    /*                    rowleft = rowleftwork[row];
                    if (rowleft >=0 && rowleft < row_size) {
                        dstp[row] = srcp[w0+rowleft];
                    }
                    else if ( rowleft < 0 && mleft) {
                        dstp[row] = srcp[w0-rowleft];       // not very precise - may be bicubic?
                    }
                    else if ( rowleft >= row_size && mright) {
                        dstp[row] = srcp[w0+row_size + row_size - rowleft -2];
                    }
                    else  if (border >=0 ){ // left or right
*/
                    dstp[row] = border;
                    //                    }
                }
            }

            dstp += pitch; // next line
        }
    }
    //-----------------------------------------------------------------------------
    else { // rotation, zoom and translation - slow

        for (int h = 0; h < height; h++) {

            for (int row = 0; row < row_size; row++) {

                float xsrc = tr->dxc + tr->dxx * row + tr->dxy * h;
                float ysrc = tr->dyc + tr->dyx * row + tr->dyy * h;
                int rowleft = (int)(xsrc); // use simply fast (int), not floor(), since followed check >1
                if (xsrc < rowleft) {
                    rowleft -= 1;
                }

                int hlow = (int)(ysrc); // use simply fast  (int), not floor(), since followed check >1
                if (ysrc < hlow) {
                    hlow -= 1;
                }

                //  x,y point is in square: (rowleft,hlow) to (rowleft+1,hlow+1)

                if ((rowleft >= 1) && (rowleft < row_size - 2) && (hlow >= 1) && (hlow < height - 2)) {

                    int ix4 = 4 * ((int)((xsrc - rowleft) * 256));

                    ptrdiff_t w0 = rowleft + hlow * pitch;

                    int64_t ts[4];
                    srcp -= pitch; // prev line
                    ts[0] = (intcoef[ix4] * srcp[w0 - 1] + intcoef[ix4 + 1] * srcp[w0] + intcoef[ix4 + 2] * srcp[w0 + 1] + intcoef[ix4 + 3] * srcp[w0 + 2]);
                    srcp += pitch; // next line
                    ts[1] = (intcoef[ix4] * srcp[w0 - 1] + intcoef[ix4 + 1] * srcp[w0] + intcoef[ix4 + 2] * srcp[w0 + 1] + intcoef[ix4 + 3] * srcp[w0 + 2]);
                    srcp += pitch; // next line
                    ts[2] = (intcoef[ix4] * srcp[w0 - 1] + intcoef[ix4 + 1] * srcp[w0] + intcoef[ix4 + 2] * srcp[w0 + 1] + intcoef[ix4 + 3] * srcp[w0 + 2]);
                    srcp += pitch; // next line
                    ts[3] = (intcoef[ix4] * srcp[w0 - 1] + intcoef[ix4 + 1] * srcp[w0] + intcoef[ix4 + 2] * srcp[w0 + 1] + intcoef[ix4 + 3] * srcp[w0 + 2]);

                    srcp -= (pitch << 1); // restore pointer, changed to shift in v.1.1.1


                    int iy4 = ((int)((ysrc - hlow) * 256)) << 2; //changed to shift in v.1.1.1

                    int64_t pixel = (intcoef[iy4] * ts[0] + intcoef[iy4 + 1] * ts[1] + intcoef[iy4 + 2] * ts[2] + intcoef[iy4 + 3] * ts[3]) >> 22;
                    dstp[row] = static_cast<PixelType>(std::clamp<int64_t>(pixel, 0, pixel_max));
                } else {
                    if (hlow < 0 && mtop)
                        hlow = -hlow; // mirror borders
                    if (hlow >= height && mbottom)
                        hlow = height + height - hlow - 2;
                    if (rowleft < 0 && mleft)
                        rowleft = -rowleft;
                    if (rowleft >= row_size && mright)
                        rowleft = row_size + row_size - rowleft - 2;
                    // check mirrowed
                    if ((rowleft >= 0) && (rowleft < row_size) && (hlow >= 0) && (hlow < height)) {
                        dstp[row] = srcp[hlow * pitch + rowleft];
                    } else if (border >= 0) { // if shifted point is out of frame, fill using border value
                        dstp[row] = border;
                    }
                }
            } // end for row

            dstp += pitch; // next line
        } //end for h
    } // end if rotation
}
