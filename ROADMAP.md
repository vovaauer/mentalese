# Roadmap: mentalese for everyone

Goal: anyone can download mentalese, click through a short setup, and talk in another language
in their own voice: on Windows, macOS or Linux, on any recent GPU, with the normal Discord app.

Today it is Linux only and needs a CUDA build from source, PipeWire, and Vesktop for subtitles.

## What's already settled

- **Models can ship freely.** Qwen3-ASR-0.6B, Index-Translate-2B and Qwen3-TTS-0.6B-Base are
  Apache-2.0, and Silero VAD is MIT. mentalese itself is GPL-3.0-or-later.
- **The engines run on every GPU type.** llama.cpp and qwentts.cpp both support CUDA (NVIDIA),
  Vulkan (AMD/Intel) and Metal (Mac).
- **It fits in about 5 GB of VRAM** (ASR 1.0, translator 1.5, voice 2.5), and reacts about 0.2 s
  after you stop talking.

## Phases

### 1. Rust core, one process (Linux)

Replace the ~1,150 lines of Python with Rust. It loads llama.cpp (libllama + mtmd) and
qwentts.cpp in-process, instead of two llama-server subprocesses plus Python.

- Why: shipping a Python runtime to everyone is the most fragile part of a consumer app. It's
  bundled per OS, slow to start, and flagged by antivirus. Three GPU contexts become one, which
  should free a few hundred MB of VRAM and drop the HTTP hop (to be measured).
- Pieces: lingua (Rust originally), Silero via onnxruntime's C API, audio I/O and resampling.
- Done when: same features and quality as today, with VRAM and latency equal or better on bench.

### 2. GUI (Tauri, system web view)

- Pick or record a voice, pick the language, start/stop and mute.
- Type to speak, and see live subtitles.
- First-run setup downloads the models and checks the GPU.
- Done when: someone who has never used a terminal can set it up and use it on Linux.

### 3. Windows

- **Virtual mic.** Windows can't create a microphone without a Microsoft-signed kernel driver.
  - Start with a guided install of VB-CABLE (free).
  - Our own signed driver is a later option; it needs a paid certificate.
- **Subtitles without Vesktop.** Capture Discord's audio with per-process loopback. It's one mixed
  stream, so it works best for 1:1 calls.
- **Engines.** Build for CUDA and Vulkan.
- Done when: works on a clean Windows 11 machine with stock Discord.

### 4. macOS (Apple Silicon)

- **Virtual mic.** Bundle a virtual audio device; BlackHole is GPL-3.0, the same license. It needs
  the admin password once.
- **Subtitles.** Capture Discord's audio with Core Audio process taps.
- **Engines.** Metal. Memory is shared with the CPU, so VRAM isn't the limit there.
- Done when: works on a stock Mac with stock Discord.

### 5. Release

- **Installers:** for all three platforms.
- **Auto-update:** through Tauri's updater.
- **Builds:** made by GitHub Actions.
- **Signing:**
  - Windows signing is free for open source through SignPath Foundation.
  - macOS needs notarization ($99/year Apple Developer account).
  - Without signing, both systems warn people off the app.
- **Fallbacks for weaker hardware:** pick settings by available VRAM. For example, speech
  recognition moves to the CPU on small GPUs; whether everything can run CPU-only is untested.

## Principles for a public release

- **Your own voice only.** Clone from a sample recorded live in the app while reading a prompt,
  not from an uploaded file.
- **Stock Discord first.** The Vencord plugin (per-person subtitles in Vesktop) stays an optional
  extra, because client mods break Discord's terms.
- **Light enough for mid-range GPUs.** VRAM stays the first constraint, latency the second.

## Open questions

- Does qwentts.cpp's voice quality and speed hold on Vulkan and Metal? Test before phase 3.
- Is CPU-only real-time possible on a fast desktop CPU, for people without a usable GPU?
- How much VRAM does one process actually save versus three?
