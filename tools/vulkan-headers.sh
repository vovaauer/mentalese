#!/usr/bin/env bash
# Khronos' Vulkan and SPIR-V headers in vendor/, for building the Vulkan engines where the
# system doesn't have them (or has old ones). tools/build-engines.sh picks them up.
set -euo pipefail
cd "$(dirname "$0")/.."
for r in Vulkan-Headers SPIRV-Headers; do
    if [ -d vendor/$r ]; then git -C vendor/$r pull -q --ff-only
    else git clone -q --depth 1 https://github.com/KhronosGroup/$r vendor/$r; fi
done
cmake -S vendor/SPIRV-Headers -B build/spirv-headers -DCMAKE_INSTALL_PREFIX="$PWD/build/spirv-headers-install" \
    -DSPIRV_HEADERS_ENABLE_TESTS=OFF > /dev/null
cmake --install build/spirv-headers > /dev/null
