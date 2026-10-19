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
        self.voice_lang = None
        self.keep_accent = False

    def set_voice(self, ref_audio: str, ref_text: str, lang: str | None = None, keep_accent: bool = False):
        """Reference recording for cloning; the model caches the voice after the first call.
        `lang` is the sample's language: other languages are spoken from the voice fingerprint
        only, unless keep_accent."""
        self.ref = dict(ref_audio=ref_audio, ref_text=ref_text)
        self.voice_lang, self.keep_accent = lang, keep_accent

    def warmup(self):
        self.model.warmup(prefill_len=100)
        # one same-language and one other-language sentence, so both clone modes are ready
        for text, lang in [("Hello there.", "en"), ("Привет.", "ru")]:
            for _ in self.stream(text, lang):
                pass

    @staticmethod
    def max_frames(text: str) -> int:
        """Upper bound on codec frames (12/s) for a text. Measured ~0.7 frames per Latin/Cyrillic
        char and ~2.1 per CJK char; allow ~2.3x that. Sampling occasionally misses the stop
        token, and without a cap that babbles for up to 48 s."""
        cjk = sum(1 for c in text if "\u3000" <= c <= "\u9fff" or "\uac00" <= c <= "\ud7af")
        return int(1.6 * (len(text) - cjk) + 4.8 * cjk) + 24

    def uses_fingerprint(self, lang: str) -> bool:
        return not self.keep_accent and self.voice_lang is not None and lang != self.voice_lang

    def stream(self, text: str, lang: str) -> Iterator[tuple[np.ndarray, int]]:
        # Default (ICL) mode continues the reference recording, so it also carries its
        # pronunciation: an English sample speaking Russian came out with a heavy accent (ASR
        # error 26% vs 0.7%). The x-vector alone keeps the voice without the accent; it's a bit
        # less similar (0.65 vs 0.77) but sounds native. Same-language output keeps ICL (0.96).
        xvec = self.uses_fingerprint(lang)
        for chunk, sr, _ in self.model.generate_voice_clone_streaming(
            text=text, language=LANGS[lang][0], chunk_size=self.chunk_size,
            max_new_tokens=self.max_frames(text), xvec_only=xvec, **self.ref,
        ):
            yield np.asarray(chunk, dtype=np.float32).reshape(-1), sr

    def speak(self, text: str, lang: str) -> tuple[np.ndarray, int]:
        chunks, sr = [], 24000
        for chunk, sr in self.stream(text, lang):
            chunks.append(chunk)
        return np.concatenate(chunks), sr
