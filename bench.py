"""Smoke test + latency benchmark for each stage: ASR -> MT -> TTS.

uv run bench.py [reference.wav]   (defaults to a Qwen sample voice)
Writes out/bench_<lang>.wav so you can listen to the clone.
"""

import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from mentalese.asr import ASR
from mentalese.mt import Translator
from mentalese.tts import TTS

SAMPLE = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
OUT = Path("out")
OUT.mkdir(exist_ok=True)


def timed(label, fn, *a, **kw):
    torch.cuda.synchronize()
    t = time.perf_counter()
    r = fn(*a, **kw)
    torch.cuda.synchronize()
    print(f"  {label:<28} {1000 * (time.perf_counter() - t):7.0f} ms")
    return r


def load(path):
    if path.startswith("http"):
        import io, requests
        data, sr = sf.read(io.BytesIO(requests.get(path, timeout=30).content), dtype="float32")
    else:
        data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


ref, ref_sr = load(sys.argv[1] if len(sys.argv) > 1 else SAMPLE)
ref16 = librosa.resample(ref, orig_sr=ref_sr, target_sr=16000)

print("loading models...")
t0 = time.perf_counter()
asr, mt, tts = ASR(), Translator(), TTS()
print(f"  loaded in {time.perf_counter() - t0:.1f}s, VRAM {torch.cuda.memory_allocated() / 1e9:.1f} GB (torch only)")

print("ASR on reference (warmup, then timed):")
asr.transcribe(ref16)
src, text = timed("transcribe", asr.transcribe, ref16)
print(f"  -> [{src}] {text!r}")

print("silence / noise (should be empty or tiny):")
for name, sig in [("silence", np.zeros(16000 * 2, np.float32)),
                  ("noise", (np.random.randn(16000 * 2) * 0.01).astype(np.float32))]:
    print(f"  {name}: {asr.transcribe(sig)}")

ref_path = OUT / "ref.wav"
sf.write(ref_path, ref, ref_sr)
tts.set_voice(str(ref_path), text)
mt.translate("hello", "en", "ru")
timed("tts warmup (one-time)", tts.warmup)


def first_audio(text, lang):
    t = time.perf_counter()
    chunks, ttfa, sr = [], None, 24000
    for c, sr in tts.stream(text, lang):
        ttfa = ttfa or time.perf_counter() - t
        chunks.append(c)
    total = time.perf_counter() - t
    wav = np.concatenate(chunks)
    print(f"  {'tts first sound':<28} {1000 * ttfa:7.0f} ms   (full {1000 * total:.0f} ms for {len(wav) / sr:.1f}s audio)")
    return wav, sr

sentence = "Hey, are you guys ready? Let's go mine some diamonds before the server restarts."
for lang in ["ru", "zh", "en"]:
    print(f"en -> {lang}:")
    out = sentence if lang == "en" else timed("translate", mt.translate, sentence, "en", lang)
    print(f"  -> {out}")
    wav, sr = first_audio(out, lang)
    sf.write(OUT / f"bench_{lang}.wav", wav, sr)

print("ru -> en:", mt.translate("Привет, ты где? Я тебя жду у спавна уже десять минут.", "ru", "en"))
print("zh -> en:", mt.translate("我们今晚一起打游戏吧，我有新的地图。", "zh", "en"))
print(f"peak VRAM (torch): {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
