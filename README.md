# mentalese

Speak one language; people on Discord hear another, in your own voice. Fully local, runs on your GPU.

```
mic → Silero VAD → Qwen3-ASR-0.6B → Hy-MT2-1.8B (llama.cpp) → Qwen3-TTS-0.6B-Base (faster-qwen3-tts) → Mentalese-Mic → Discord
```

About 0.85 s from the end of your sentence to the first translated sound. Uses about 6.5 GB of VRAM.

## Use

```bash
uv run mentalese record                 # once: record your voice sample
uv run mentalese say --to ru "hello"    # hear your clone in your headphones
uv run mentalese run --to ru            # go live; type zh / en / ru... + Enter to switch, m = mute, q = quit
```

In Discord → Voice & Video: set **Input Device** to **Mentalese-Mic**, and turn off Noise Suppression,
Echo Cancellation and Automatic Gain Control.

Output languages: en ru zh ja ko de fr es pt it (what Qwen3-TTS can speak). Input: any of ~30 languages, auto-detected.

## Setup notes

- `vendor/llama.cpp` is built with CUDA for sm_120 (RTX 50): `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 && cmake --build build -j --target llama-server`
- `models/Hy-MT2-1.8B-Q8_0.gguf` from `tencent/Hy-MT2-1.8B-GGUF`
- transformers is pinned to 5.15.1: faster-qwen3-tts breaks on 5.19.
- `bench.py` measures each stage.
