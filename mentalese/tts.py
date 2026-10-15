"""Speech in a cloned voice with Qwen3-TTS-0.6B-Base, via faster-qwen3-tts (CUDA graphs + streaming)."""

from collections.abc import Iterator

import numpy as np

from .langs import LANGS

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


class TTS:
    def __init__(self, chunk_size: int = 4):
        from faster_qwen3_tts import FasterQwen3TTS

        # static KV cache: reference (~15 s = 190 frames) + text + reply; 1024 leaves ~55 s of speech
        self.model = FasterQwen3TTS.from_pretrained(MODEL_ID, max_seq_len=1024)
        self.chunk_size = chunk_size  # codec steps per yielded chunk; 1 step = 80 ms of audio
        self.ref = None

    def set_voice(self, ref_audio: str, ref_text: str):
        """Reference recording for cloning; the model caches the voice after the first call."""
        self.ref = dict(ref_audio=ref_audio, ref_text=ref_text)

    def warmup(self):
        self.model.warmup(prefill_len=100)
        for _ in self.stream("Hello there.", "en"):
            pass

    @staticmethod
    def max_frames(text: str) -> int:
        """Upper bound on codec frames (12/s) for a text. Measured ~0.7 frames per Latin/Cyrillic
        char and ~2.1 per CJK char; allow ~2.3x that. Sampling occasionally misses the stop
        token, and without a cap that babbles for up to 48 s."""
        cjk = sum(1 for c in text if "\u3000" <= c <= "\u9fff" or "\uac00" <= c <= "\ud7af")
        return int(1.6 * (len(text) - cjk) + 4.8 * cjk) + 24

    def stream(self, text: str, lang: str) -> Iterator[tuple[np.ndarray, int]]:
        for chunk, sr, _ in self.model.generate_voice_clone_streaming(
            text=text, language=LANGS[lang][0], chunk_size=self.chunk_size,
            max_new_tokens=self.max_frames(text), **self.ref,
        ):
            yield np.asarray(chunk, dtype=np.float32).reshape(-1), sr

    def speak(self, text: str, lang: str) -> tuple[np.ndarray, int]:
        chunks, sr = [], 24000
        for chunk, sr in self.stream(text, lang):
            chunks.append(chunk)
        return np.concatenate(chunks), sr
