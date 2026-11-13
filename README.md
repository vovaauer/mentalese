# mentalese

Speak one language; people on Discord hear another, in your own voice. Fully local, runs on your GPU.

```
mic → Silero VAD (ONNX) → Qwen3-ASR-0.6B → Index-Translate-2B → Qwen3-TTS-0.6B-Base → Mentalese-Mic → Discord
                          └─────────── llama.cpp ───────────┘   qwentts.cpp
```

About 0.2 s from the end-of-sentence pause (400 ms) to the first translated sound. Uses about
5 GB of VRAM at peak: speech recognition 1.0 GB, translator 1.5 GB, voice 2.5 GB. No PyTorch.
`--asr-gpu` makes recognition ~40 ms faster for 0.75 GB more.

## Use

```bash
uv run mentalese record                 # once: record your voice sample
uv run mentalese say --to ru "hello"    # hear your clone in your headphones
uv run mentalese run --to ru            # go live; type a line + Enter to say it in your voice,
                                        # /zh /en /ru... to switch, /mute, /quit
```

Other voices (e.g. family): `uv run mentalese record --voice mom`, then add `--voice mom` to
`run` or `say`. `uv run mentalese voices` lists them. To just hear yourself in another language
in your own headphones, no Discord: `uv run mentalese run --practice --to zh --voice mom`.

In Discord → Voice & Video: set **Input Device** to **Mentalese-Mic** (leave Output on your headphones; never pick
"Mentalese-Internal", it loops other people back into your mic), and turn off Noise Suppression,
Echo Cancellation and Automatic Gain Control.

Output languages: en ru zh ja ko de fr es pt it (what Qwen3-TTS can speak). Input: any of ~30 languages, auto-detected.

## Setup

```bash
tools/setup.sh    # builds vendor/llama.cpp and vendor/qwentts.cpp, downloads models/, runs uv sync
```

Re-running it updates both engines to their latest versions. It builds for RTX 50 (sm_120);
set `CUDA_ARCH=89` (or your GPU's) otherwise. It patches qwentts.cpp's talker cache from 4096 to
1024 positions (670 MB less VRAM) and stops if that patch or the library's ABI no longer matches.

`bench.py` measures each stage and the total VRAM. `tools/e2e.py` plays test clips through
Mentalese-Mic, so don't run it while you're in a call.

## Roadmap

mentalese runs on Linux today. [ROADMAP.md](ROADMAP.md) is the plan for a GUI and for
Windows, macOS and any GPU.

## License

Copyright (C) 2026 Vova Auer. Licensed under the GNU General Public License v3.0 or later: see
[LICENSE](LICENSE). The models aren't part of this repository; `tools/setup.sh` downloads each one
from its maker, under its maker's license.
