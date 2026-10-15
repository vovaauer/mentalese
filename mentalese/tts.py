"""Speech in a cloned voice with Qwen3-TTS-0.6B-Base, via faster-qwen3-tts (CUDA graphs + streaming)."""

from collections.abc import Iterator

import numpy as np

from .langs import LANGS

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


class TTS:
    def __init__(self, chunk_size: int = 4):
        from faster_qwen3_tts import FasterQwen3TTS

        self.model = FasterQwen3TTS.from_pretrained(MODEL_ID)
        self.chunk_size = chunk_size  # codec steps per yielded chunk; 1 step = 80 ms of audio
        self.ref = None

    def set_voice(self, ref_audio: str, ref_text: str):
        """Reference recording for cloning; the model caches the voice after the first call."""
        self.ref = dict(ref_audio=ref_audio, ref_text=ref_text)

    def warmup(self):
        self.model.warmup(prefill_len=100)
        for _ in self.stream("Hello there.", "en"):
            pass

    def stream(self, text: str, lang: str) -> Iterator[tuple[np.ndarray, int]]:
        for chunk, sr, _ in self.model.generate_voice_clone_streaming(
            text=text, language=LANGS[lang][0], chunk_size=self.chunk_size, **self.ref,
        ):
            yield np.asarray(chunk, dtype=np.float32).reshape(-1), sr

    def speak(self, text: str, lang: str) -> tuple[np.ndarray, int]:
        chunks, sr = [], 24000
        for chunk, sr in self.stream(text, lang):
            chunks.append(chunk)
        return np.concatenate(chunks), sr
