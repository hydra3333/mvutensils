#pragma once

#include <cstdint>


// CPU feature flags. We only care about AVX2 and the AVX-512 feature levels here; the detection
// (CPU.cpp) uses the CPUID/XGETBV compiler intrinsics, no external asm. Each AVX-512 bit also implies
// the OS saves the ZMM/opmask state. MVU_CPU_AVX512_BASE is the x86-64-v4 baseline (F+CD+BW+DQ+VL --
// what /arch:AVX512 and -march=x86-64-v4 target) collapsed into one flag; the rest are the optional
// extensions on top of that baseline.
enum {
    MVU_CPU_AVX2               = 1u << 0,
    MVU_CPU_AVX512_BASE        = 1u << 1,   // F + CD + BW + DQ + VL (x86-64-v4)
    MVU_CPU_AVX512IFMA         = 1u << 2,
    MVU_CPU_AVX512VBMI         = 1u << 3,
    MVU_CPU_AVX512VBMI2        = 1u << 4,
    MVU_CPU_AVX512VNNI         = 1u << 5,
    MVU_CPU_AVX512BITALG       = 1u << 6,
    MVU_CPU_AVX512VPOPCNTDQ    = 1u << 7,
    MVU_CPU_AVX512VP2INTERSECT = 1u << 8,
    MVU_CPU_AVX512FP16         = 1u << 9,
    MVU_CPU_AVX512BF16         = 1u << 10,
};

uint32_t cpu_detect();

extern uint32_t g_cpuinfo;
