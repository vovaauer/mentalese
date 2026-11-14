# Roadmap: mentalese for everyone

Goal: anyone can download mentalese, click through a short setup, and talk in another language
in their own voice: on Windows, macOS or Linux, on any recent GPU, with the normal Discord app.

## Done

1. **Rust core, one process.** llama.cpp and qwentts.cpp run in-process through small C wrappers
   (`engines/`), each with its own private copy of ggml. Silero VAD is a pure-Rust port reading the
   official model. No Python. Same transcripts and translations as the Python version, 0.1–0.2 s
   to first sound, 4.6 GB of graphics memory at peak (was 5.0 GB in three processes).
2. **The app** (Tauri): first-run model download, voice recording, language and voice pickers,
   live lines and subtitles, typing to speak, settings, voice preview, practice mode.
3. **Windows**: microphone and speakers through WASAPI, the virtual mic through VB-CABLE (guided),
   subtitles from any app's sound through per-process loopback capture.
4. **macOS**: Metal engines, the virtual mic through BlackHole (guided), subtitles through Core
   Audio process taps.
5. **Releases**: GitHub Actions builds installers for all three (Vulkan for every GPU, plus CUDA
   builds for NVIDIA), into a draft release, with signed auto-updates.

## Not verified yet

- **Windows and macOS have only been compiled, never run.** The audio code, the capture code and
  the engines need a first real run on each: device names (VB-CABLE, BlackHole), process loopback
  formats, the macOS capture permission, Discord's process layout.
- **The release workflow hasn't run yet**, the CUDA steps in particular (toolkit install on the
  runners). The Linux Vulkan installers were built and checked locally.
- **AMD and Intel GPUs**: the Vulkan engines were tested on NVIDIA only.

## Next

- **Code signing**, so Windows and macOS don't warn on first start. Windows: free for open source
  through SignPath Foundation (apply with the repository). macOS: an Apple Developer account
  ($99/year), then notarization in the release workflow.
- **Our own virtual mic** instead of VB-CABLE/BlackHole: on macOS, bundling a BlackHole-style
  device (GPL-3.0, like mentalese); on Windows, a driver signed by Microsoft (needs an EV
  certificate).
- **Weaker computers**: pick settings by free graphics memory (speech recognition on the CPU on
  small GPUs), and test whether a fast CPU alone can keep up.

## Principles

- **Your own voice only.** Voices are recorded live in the app while reading a prompt, not
  imported from a file.
- **Stock Discord first.** The Vencord plugin (per-person subtitles in Vesktop) stays an optional
  extra, because client mods break Discord's terms.
- **Light enough for mid-range GPUs.** Graphics memory stays the first constraint, latency the
  second.
