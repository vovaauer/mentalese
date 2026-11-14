#!/usr/bin/env bash
# Builds mentalese from source, or updates it: the newest engines, built for this machine, then
# the app and the command line. Safe to re-run: updating is just running it again.
#   tools/setup.sh                    picks the GPU backend: CUDA if nvcc is installed, Metal on a
#                                     Mac, otherwise Vulkan
#   BACKENDS="vulkan cpu" tools/setup.sh
# Needs git, cmake, a C++ compiler, Rust (rustup) and, for CUDA, the CUDA toolkit. The models
# download on first start (or: target/release/mentalese download).
set -euo pipefail
cd "$(dirname "$0")/.."

tools/fetch-engines.sh

if [ -z "${BACKENDS:-}" ]; then
    if [ "$(uname -s)" = Darwin ]; then BACKENDS=metal
    elif command -v nvcc > /dev/null; then BACKENDS=cuda
    else BACKENDS=vulkan; fi
fi

# Vulkan needs its headers to build: the system's if installed, else Khronos' own
if [[ " $BACKENDS " == *" vulkan "* ]] && [ ! -f /usr/include/vulkan/vulkan.h ] && [ -z "${VULKAN_SDK:-}" ]; then
    tools/vulkan-headers.sh
fi

# shellcheck disable=SC2086
tools/build-engines.sh $BACKENDS
cargo build --release -p mentalese -p mentalese-app
echo "ready: target/release/mentalese-app (the app) or target/release/mentalese (command line)"
