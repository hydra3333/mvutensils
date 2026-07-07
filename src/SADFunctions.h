#pragma once

// Functions that computes distances between blocks

// See legal notice in Copying.txt for more information

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA, or visit
// http://www.gnu.org/copyleft/gpl.html .

#include <cstdint>


using SADFunction = unsigned int (*)(const uint8_t *pSrc, intptr_t nSrcPitch,
                                     const uint8_t *pRef, intptr_t nRefPitch) noexcept;


SADFunction selectSADFunction(unsigned width, unsigned height, unsigned bits);

SADFunction selectSATDFunction(unsigned width, unsigned height, unsigned bits);

// 32-bit float-pixel SAD baseline (auto-vec C, all platforms). bits is implicitly 32.
void selectSADFunctionFloat(unsigned width, unsigned height, SADFunction &sad);

// 32-bit float-pixel SATD (pure scalar C, all platforms). bits is implicitly 32.
void selectSATDFunctionFloat(unsigned width, unsigned height, SADFunction &satd);


#if defined(MVTOOLS_X86)
// Higher-ISA selectors overwrite sad/satd only when a kernel exists for the given size.
void selectSADFunctionAVX2(unsigned width, unsigned height, unsigned bits, SADFunction &sad);
void selectSATDFunctionAVX2(unsigned width, unsigned height, unsigned bits, SADFunction &satd);
void selectSADFunctionAVX512(unsigned width, unsigned height, unsigned bits, SADFunction &sad);
void selectSATDFunctionAVX512(unsigned width, unsigned height, unsigned bits, SADFunction &satd);
// AVX-512 VNNI 16-bit SAD (vpdpwssd bias trick). Full-16-bit correct; only the sizes where it beats
// the plain AVX-512 kernel (widths 8-64, not 128/4, height >= 8) are registered.
void selectSADFunctionAVX512VNNI(unsigned width, unsigned height, unsigned bits, SADFunction &sad);
#endif
