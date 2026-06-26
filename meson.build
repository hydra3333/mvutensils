project('MVUtensils', 'cpp',
    default_options: [
        'buildtype=release', 'warning_level=2', 'b_lto=true', 'b_ndebug=if-release', 'cpp_std=c++17',
    ],
    license: 'GPL-2.0-or-later',
    license_files: 'LICENSE',
    meson_version: '>=1.2.3',
    version: '1',
)

py = import('python').find_installation(pure: false)

incdir = include_directories(
    run_command(py, '-c', 'import vapoursynth as vs; print(vs.get_include())', check: true).stdout().strip(),
)

deps = [
    dependency('zimg', version: '>=3.0.5'),
    dependency('fftw3f', version: '>=3.3.5')
]

libs = []

add_project_arguments('-DPACKAGE_VERSION="@0@"'.format(meson.project_version()), language: ['cpp'])

if host_machine.cpu_family() == 'x86_64'
    add_project_arguments('-DMVTOOLS_X86', language: ['cpp'])

    libs += static_library('avx2',
        files(
            'src/Degrains_AVX2.cpp',
            'src/Overlap_AVX2.cpp',
            'src/SADFunctions_AVX2.cpp',
        ),
        cpp_args: '-march=x86-64-v3',
        gnu_symbol_visibility: 'hidden',
        include_directories: incdir,
    )

    libs += static_library('avx512',
        files(
            'src/SADFunctions_AVX512.cpp',
            'src/FlowShared_AVX512.cpp',
        ),
        cpp_args: '-march=x86-64-v4 -mprefer-vector-width=512',
        gnu_symbol_visibility: 'hidden',
        include_directories: incdir,
    )
elif host_machine.cpu_family() == 'aarch64'
    add_project_arguments('-DMVTOOLS_ARM', language: ['cpp'])
endif

shared_module('mvutensils',
    files(
        'src/Analyse.cpp',
        'src/Compensate.cpp',
        'src/CopyCode.cpp',
        'src/CPU.cpp',
        'src/Degrains.cpp',
        'src/DepanAnalyse.cpp',
        'src/DepanCompensate.cpp',
        'src/DepanEstimate.cpp',
        'src/DepanShared.cpp',
        'src/DepanStabilise.cpp',
        'src/EntryPoint.cpp',
        'src/Flow.cpp',
        'src/FlowBlur.cpp',
        'src/FlowFPS.cpp',
        'src/FlowInter.cpp',
        'src/Mask.cpp',
        'src/MaskResize.cpp',
        'src/MotionBlockPyramid.cpp',
        'src/Recalculate.cpp',
        'src/SCDetection.cpp',
        'src/Super.cpp',
        'src/SuperPyramid.cpp',
        'src/Overlap.cpp',
        'src/SADFunctions.cpp',
    ),
    gnu_symbol_visibility: 'hidden',
    include_directories: incdir,
    dependencies: deps,
    install: true,
    install_dir: py.get_install_dir() / 'vapoursynth/plugins',
    link_with: libs,
    name_prefix: '',
)
