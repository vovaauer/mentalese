# Mentalese

Speak your language; people on Discord (or any call app) hear theirs, in your own voice. You get
live subtitles of what they say. Runs entirely on your computer: nothing you say leaves it.

```
your mic → Silero VAD → Qwen3-ASR-0.6B → Index-Translate-2B → Qwen3-TTS-0.6B-Base → virtual mic → Discord
others   → Silero VAD → Qwen3-ASR-0.6B → Index-Translate-2B → subtitles
                        └────────── llama.cpp ──────────┘     qwentts.cpp
```

About 0.2 s from the end of your sentence to the first translated sound. Graphics memory peaks
at 4.6 GB (4.3 GB on Vulkan): speech recognition 0.9 GB, translator 1.5 GB, voice 1.4 GB, plus
working memory while it speaks (measured on an RTX 5070 Ti). Output languages: English, Russian, Chinese, Japanese, Korean, German,
French, Spanish, Portuguese, Italian. It understands about 30 more.

## Install

Download the installer for your system from the
[latest release](https://github.com/vovaauer/mentalese/releases/latest):

| System | File | Needs |
|---|---|---|
| Windows 10 (2004+) / 11 | `Mentalese_…_x64-setup.exe` | [VB-CABLE](https://vb-audio.com/Cable/) (free) for the virtual mic |
| macOS 14.2+ on Apple Silicon | `Mentalese_…_aarch64.dmg` | [BlackHole 2ch](https://existential.audio/blackhole/) (free) for the virtual mic |
| Linux | `.AppImage`, `.deb` or `.rpm` | PipeWire or PulseAudio (the virtual mic is created for you) |

Any recent GPU with about 5 GB free works: NVIDIA, AMD or Intel through Vulkan, or Apple's.
The `-nvidia` downloads add CUDA: about 30 ms faster per sentence on NVIDIA cards, but a much
bigger download.

The first start downloads the speech models (3.4 GB, once) and asks you to read a short text
out loud for 15 seconds: that's the voice it speaks with. Then press **Start**.

In Discord: User Settings → Voice & Video → **Input Device**: the virtual mic (`Mentalese-Mic`,
`CABLE Output` or `BlackHole 2ch`). Turn off Noise Suppression, Echo Cancellation and Automatic
Gain Control there too. Leave your output on your headphones.

**Subtitles** of what others say: in Settings, pick the call app under "From an app's sound"
while the call is on. Everyone in the call is mixed into one stream, so it's best one-on-one. For
each person separately, use [Vesktop](https://github.com/Vencord/Vesktop) with the plugin in
`vencord/` (`vencord/build.sh`, then `vencord/install.sh`; Linux).

**Practice mode** (Settings) plays your translated voice in your headphones only, without a call.
Typing a line and pressing Enter speaks it in your voice, translated like a spoken sentence.

## Command line

The same translator without the window (`target/release/mentalese` in a source build):

```bash
mentalese record                  # once: record your voice sample
mentalese say --to ru "hello"     # hear your voice in Russian in your headphones
mentalese run --to ru             # go live; type a line + Enter to say it, /zh /en... to switch, /quit
mentalese devices                 # microphones and speakers
```

`mentalese` alone lists every option. Other voices: `mentalese record --voice mom`, then
`--voice mom`.

## Build from source

```bash
tools/setup.sh    # engines (newest llama.cpp + qwentts.cpp) for this GPU, then the app and CLI
target/release/mentalese-app
```

Needs git, cmake, a C++ compiler, [Rust](https://rustup.rs), and the CUDA toolkit for NVIDIA
(otherwise it builds for Vulkan; `BACKENDS="vulkan cpu" tools/setup.sh` to choose). Re-running it
updates the engines to their newest versions and records them in `tools/engines.lock`.

- `crates/core`: everything but the window: audio per OS, voice detection (a pure-Rust port of
  Silero VAD that reads its official model), the engines, the live pipeline.
- `crates/app`: the window (Tauri). `crates/cli`: the command line.
- `engines/llm`, `engines/voice`: small C wrappers that link llama.cpp and qwentts.cpp
  statically into one library each, so the two (differently built) copies of ggml can share a
  process. One build per GPU backend; the app picks the first that finds a GPU.
- `tools/fetch-engines.sh` patches qwentts.cpp's talker cache from 4096 to 1024 positions (670 MB
  less graphics memory; long text is split so a request never needs more) and stops if that patch
  or the library's interface stops matching.

Checks: `cargo test`, `cargo run --release -p mentalese-core --example smoke` (speed and graphics
memory of each model), and `--example e2e` (Linux: plays test clips through two throwaway audio
devices in practice mode and prints each translated line with its timing; nothing reaches the
virtual mic).

**Releases**: bump the version in `Cargo.toml` and `crates/app/tauri.conf.json`, push a tag
`vX.Y.Z`, and GitHub Actions builds every installer into a draft release. Signed auto-updates
need the `TAURI_SIGNING_PRIVATE_KEY` and `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` repository secrets.
The installers aren't code-signed yet, so Windows and macOS warn before the first start (see
[ROADMAP.md](ROADMAP.md)).

## Privacy

Speech, voices and translations never leave your computer. Mentalese goes online only to
download the models (from Hugging Face and GitHub) and to check GitHub for updates.

## License

Copyright (C) 2026 Vova Auer. Licensed under the GNU General Public License v3.0 or later: see
[LICENSE](LICENSE). The models aren't part of this repository; the app downloads each one from
its maker, under its maker's license (Qwen3-ASR, Qwen3-TTS and Index-Translate: Apache-2.0;
Silero VAD: MIT).
