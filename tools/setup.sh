#!/usr/bin/env bash
# Build or update the native engines and fetch the models. Safe to re-run: updating is just
# running this again. Needs git, cmake, a CUDA toolkit and curl.
#   tools/setup.sh            build for this machine's GPU (default sm_120, RTX 50)
#   CUDA_ARCH=89 tools/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ARCH="${CUDA_ARCH:-120}"
JOBS="$(nproc)"

# llama.cpp: speech recognition and translation servers
if [ ! -d vendor/llama.cpp ]; then
    git clone https://github.com/ggml-org/llama.cpp vendor/llama.cpp
else
    git -C vendor/llama.cpp pull --ff-only
fi
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build vendor/llama.cpp/build -j "$JOBS" --target llama-server

# qwentts.cpp: the voice, loaded in-process through its C API (libqwen.so)
if [ ! -d vendor/qwentts.cpp ]; then
    git clone --recurse-submodules https://github.com/ServeurpersoCom/qwentts.cpp vendor/qwentts.cpp
else
    git -C vendor/qwentts.cpp checkout -- src/pipeline-tts.cpp   # drop our patch so the pull is clean
    git -C vendor/qwentts.cpp pull --ff-only
    git -C vendor/qwentts.cpp submodule update --init --recursive
fi
# Patch: the talker's KV cache holds 1024 positions instead of 4096 (F32: 896 -> 224 MB of VRAM).
# mentalese splits long text so a request never needs more (mentalese/tts.py, TTS.parts).
sed -i 's/pt->talker.head_dim, 4096, pt->max_batch/pt->talker.head_dim, 1024, pt->max_batch/' \
    vendor/qwentts.cpp/src/pipeline-tts.cpp
grep -q 'pt->talker.head_dim, 1024, pt->max_batch' vendor/qwentts.cpp/src/pipeline-tts.cpp \
    || { echo "qwentts.cpp changed: the KV cache patch no longer applies, see tools/setup.sh" >&2; exit 1; }
grep -q '#define QT_ABI_VERSION 6$' vendor/qwentts.cpp/src/qwen.h \
    || { echo "qwentts.cpp ABI changed: update the structs in mentalese/tts.py from src/qwen.h" >&2; exit 1; }
cmake -S vendor/qwentts.cpp -B vendor/qwentts.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${ARCH}a-real" \
    -DCMAKE_BUILD_TYPE=Release -DQWEN_SHARED=ON
cmake --build vendor/qwentts.cpp/build -j "$JOBS" --target qwen

# models
mkdir -p models
fetch() {  # fetch <url> <file>, unless it's already there
    [ -s "models/$2" ] && return
    curl -fL --progress-bar -o "models/$2.part" "$1"
    mv "models/$2.part" "models/$2"
}
HF=https://huggingface.co
fetch "$HF/IndexTeam/Index-Translate-2B-GGUF/resolve/main/Index-Translate-2B.Q4_K_M.gguf" Index-Translate-2B.Q4_K_M.gguf
fetch "$HF/ggml-org/Qwen3-ASR-0.6B-GGUF/resolve/main/Qwen3-ASR-0.6B-Q8_0.gguf" Qwen3-ASR-0.6B-Q8_0.gguf
fetch "$HF/ggml-org/Qwen3-ASR-0.6B-GGUF/resolve/main/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf" mmproj-Qwen3-ASR-0.6B-Q8_0.gguf
fetch "$HF/Serveurperso/Qwen3-TTS-GGUF/resolve/main/qwen-talker-0.6b-base-Q8_0.gguf" qwen-talker-0.6b-base-Q8_0.gguf
fetch "$HF/Serveurperso/Qwen3-TTS-GGUF/resolve/main/qwen-tokenizer-12hz-Q8_0.gguf" qwen-tokenizer-12hz-Q8_0.gguf
fetch "https://github.com/snakers4/silero-vad/raw/v6.2.3/src/silero_vad/data/silero_vad.onnx" silero_vad.onnx

uv sync
echo "ready: uv run mentalese record, then uv run mentalese run --to ru"
