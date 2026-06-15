# Platform-specific compiler optimizations for Astral runtime
# See: docs/rules/CODING_STANDARDS.md § Compiler Optimization Flags

function(apply_compiler_flags target)
    # Release build optimizations
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        # Function alignment: 2-5% I-cache miss reduction
        # Trade-off: +5-10% binary size but better instruction cache utilization
        target_compile_options(${target} PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-falign-functions=32>
        )

        # Loop alignment: Improves branch prediction
        target_compile_options(${target} PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-falign-loops=32>
        )

        if(NOT ASTRAL_BUILD_UNREAL_PLUGIN)
            # Link-time optimization: Enables cross-translation-unit inlining
            # Critical for ASTRAL_FORCE_INLINE to work across compilation units
            set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        endif()
    endif()

    # Platform-specific SIMD optimizations
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        # AVX2 baseline for modern x86 (2013+)
        # Includes FMA for faster math and F16C for Float8 widening
        target_compile_options(${target} PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2 -mfma -mf16c>
            $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>
        )
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
        # ARM NEON is always available on ARMv8
        target_compile_options(${target} PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-march=armv8-a+fp+simd>
        )
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7")
        # ARMv7 with NEON (optional on ARMv7)
        target_compile_options(${target} PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-mfpu=neon -mfloat-abi=hard>
        )
    endif()

    # Debug build: Optional sanitizers for memory safety
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        # AddressSanitizer: Detects buffer overflows, use-after-free
        # UndefinedBehaviorSanitizer: Detects integer overflow, misaligned access, etc.
        # NOTE: Disabled by default for Valgrind compatibility
        # Enable with: cmake -DASTRAL_ENABLE_ASAN=ON
        if(ASTRAL_ENABLE_ASAN)
            target_compile_options(${target} PRIVATE
                $<$<CXX_COMPILER_ID:GNU,Clang>:-fsanitize=address -fsanitize=undefined>
            )
            target_link_options(${target} PRIVATE
                $<$<CXX_COMPILER_ID:GNU,Clang>:-fsanitize=address -fsanitize=undefined>
            )
        endif()

        # Stack protector for buffer overflow detection
        target_compile_options(${target} PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-fstack-protector-strong>
        )

        # Full debug symbols
        target_compile_options(${target} PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-g3>
            $<$<CXX_COMPILER_ID:MSVC>:/Zi>
        )
    endif()

    # Platform-specific linker flags are configured in the top-level `CMakeLists.txt`
    # so embedded profiles can avoid pulling in optional dependencies (e.g. `dl`).
endfunction()

# FORBIDDEN FLAGS (documented here as warnings)
# DO NOT USE:
# -ffast-math           # Breaks IEEE 754 semantics; causes numerical instability in softmax/logits
# -fomit-frame-pointer  # Makes debugging impossible; marginal performance gain not worth it
# -mavx512f             # Not universally available; use runtime dispatch instead
# /fp:fast (MSVC)       # Same issue as -ffast-math
