#!/usr/bin/env bash
# Builds the two engine libraries (mentalese-llm: llama.cpp; mentalese-voice: qwentts.cpp) for
# one or more GPU backends, into build/<engine>-<backend>/. Works on Linux, macOS and Windows
# (Git Bash).
#
#   tools/build-engines.sh cuda                 for this machine: its CPU's features, its GPU's arch
#   tools/build-engines.sh --dist vulkan cpu    portable builds for packaging, staged with the
#                                               runtime libraries they need in crates/app/engines/
#
# Backends: cuda (NVIDIA), vulkan (AMD, Intel, NVIDIA), metal (Apple), cpu.
# CUDA_ARCH overrides the CUDA architectures (default: native, or a range covering GTX 16xx to
# RTX 50 for --dist).
set -euo pipefail
cd "$(dirname "$0")/.."

DIST=0
if [ "${1:-}" = --dist ]; then DIST=1; shift; fi
[ $# -gt 0 ] || { sed -n '2,13p' "$0"; exit 1; }

case "$(uname -s)" in
    Linux) OS=linux; LIB=lib%s.so ;;
    Darwin) OS=macos; LIB=lib%s.dylib ;;
    MINGW*|MSYS*|CYGWIN*) OS=windows; LIB=%s.dll ;;
    *) echo "unsupported OS $(uname -s)" >&2; exit 1 ;;
esac
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu)

backend_flags() {
    case $1 in
        cuda)
            local arch=${CUDA_ARCH:-native}
            [ $DIST = 1 ] && arch=${CUDA_ARCH:-75-virtual;80-virtual;86-real;89-real;120a-real}
            echo "-DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=$arch" ;;
        vulkan)
            local f="-DGGML_VULKAN=ON"
            # without system Vulkan/SPIR-V headers, use Khronos' own (tools/setup.sh fetches them)
            [ -d vendor/Vulkan-Headers ] && f="$f -DVulkan_INCLUDE_DIR=$PWD/vendor/Vulkan-Headers/include"
            # (ggml finds SPIRV-Headers but relies on the system include path for it)
            [ -d build/spirv-headers-install ] && f="$f -DCMAKE_PREFIX_PATH=$PWD/build/spirv-headers-install -DCMAKE_CXX_FLAGS=-I$PWD/build/spirv-headers-install/include"
            echo "$f" ;;
        metal) echo "-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON" ;;
        cpu) echo "" ;;
        *) echo "unknown backend $1" >&2; exit 1 ;;
    esac
}

for backend in "$@"; do
    # portable: baseline x86-64-v3 (AVX2, 2013+) instead of this CPU, and no OpenMP runtime to ship
    common="-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=$([ $DIST = 1 ] && echo OFF || echo ON)"
    [ $DIST = 1 ] && common="$common -DGGML_OPENMP=OFF"
    for engine in llm voice; do
        dir=build/$engine-$backend
        echo "== $engine ($backend)"
        # shellcheck disable=SC2046
        cmake -S engines/$engine -B $dir $common $(backend_flags $backend) > $dir.configure.log 2>&1 \
            || { tail -30 $dir.configure.log; exit 1; }
        cmake --build $dir --config Release -j "$JOBS" --target mentalese-$engine > $dir.build.log 2>&1 \
            || { grep -m20 -iE 'error' $dir.build.log; exit 1; }
        # multi-config generators (Visual Studio) put it in Release/
        [ -f "$dir/Release/$(printf $LIB mentalese-$engine)" ] && cp "$dir/Release/$(printf $LIB mentalese-$engine)" $dir/
    done

    [ $DIST = 1 ] || continue
    stage=crates/app/engines/$backend
    rm -rf "$stage" && mkdir -p "$stage"
    cp "build/llm-$backend/$(printf $LIB mentalese-llm)" "build/voice-$backend/$(printf $LIB mentalese-voice)" "$stage/"
    if [ $backend = cuda ]; then
        # the CUDA runtime and cuBLAS (redistributable under NVIDIA's EULA); the driver brings the rest
        cuda=${CUDA_PATH:-$(dirname "$(dirname "$(command -v nvcc)")")}
        if [ $OS = windows ]; then
            cp "$cuda"/bin/{cudart64_,cublas64_,cublasLt64_}*.dll "$stage/" 2>/dev/null \
                || cp "$cuda"/bin/x64/{cudart64_,cublas64_,cublasLt64_}*.dll "$stage/"
        else
            for l in libcudart libcublas libcublasLt; do
                # by soname (libcudart.so.13), which is what the engines ask for
                soname=$(cd "$cuda/lib64" && ls $l.so.[0-9]* | awk '{print length, $0}' | sort -n | head -1 | cut -d' ' -f2)
                cp -L "$cuda/lib64/$soname" "$stage/"
            done
        fi
    fi
    echo "staged $stage: $(ls "$stage" | tr '\n' ' ')"
done
