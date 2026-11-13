"""Speech in a cloned voice with Qwen3-TTS-0.6B-Base, run by qwentts.cpp (C++/ggml) through its C API.

Replaces faster-qwen3-tts (PyTorch): the voice took 2.8 GB (3.4 GB while speaking) there.
"""

import ctypes as C
import queue
import re
import threading
from collections.abc import Iterator

import numpy as np
import soundfile as sf
import soxr

from .langs import LANGS
from .server import MODELS, ROOT

LIB = ROOT / "vendor/qwentts.cpp/build/libqwen.so"
TALKER = MODELS / "qwen-talker-0.6b-base-Q8_0.gguf"  # LM + code predictor + speaker encoder
CODEC = MODELS / "qwen-tokenizer-12hz-Q8_0.gguf"     # 12 Hz audio tokenizer; codebooks stay F32
SR = 24000
ABI = 6  # QT_ABI_VERSION of the qwen.h these structs mirror
MAX_PART_FRAMES = 480  # see TTS.parts


class InitParams(C.Structure):
    _fields_ = [("abi_version", C.c_int), ("talker_path", C.c_char_p), ("codec_path", C.c_char_p),
                ("use_fa", C.c_bool), ("clamp_fp16", C.c_bool), ("max_batch", C.c_int),
                ("codec_chunk_sec", C.c_float), ("device", C.c_char_p)]


class VoiceRef(C.Structure):
    _fields_ = [("ref_spk_emb", C.POINTER(C.c_float)), ("ref_spk_dim", C.c_int),
                ("ref_codes", C.POINTER(C.c_int32)), ("ref_T", C.c_int), ("num_codebooks", C.c_int)]


class Audio(C.Structure):
    _fields_ = [("samples", C.POINTER(C.c_float)), ("n_samples", C.c_int),
                ("sample_rate", C.c_int), ("channels", C.c_int)]


CANCEL_CB = C.CFUNCTYPE(C.c_bool, C.c_void_p)
CHUNK_CB = C.CFUNCTYPE(C.c_bool, C.POINTER(C.c_float), C.c_int, C.c_void_p)
LOG_CB = C.CFUNCTYPE(None, C.c_int, C.c_char_p, C.c_void_p)


class TTSParams(C.Structure):
    _fields_ = [("abi_version", C.c_int),
                ("text", C.c_char_p), ("lang", C.c_char_p), ("instruct", C.c_char_p), ("speaker", C.c_char_p),
                ("ref_audio_24k", C.POINTER(C.c_float)), ("ref_n_samples", C.c_int), ("ref_text", C.c_char_p),
                ("seed", C.c_int64), ("max_new_tokens", C.c_int), ("temperature", C.c_float), ("top_k", C.c_int),
                ("top_p", C.c_float), ("repetition_penalty", C.c_float), ("subtalker_temperature", C.c_float),
                ("subtalker_top_k", C.c_int), ("subtalker_top_p", C.c_float), ("dump_dir", C.c_char_p),
                ("cancel", CANCEL_CB), ("cancel_user_data", C.c_void_p),
                ("on_chunk", CHUNK_CB), ("on_chunk_user_data", C.c_void_p),
                ("ref_spk_emb", C.POINTER(C.c_float)), ("ref_spk_dim", C.c_int),
                ("ref_codes", C.POINTER(C.c_int32)), ("ref_T", C.c_int)]


def _load():
    lib = C.CDLL(str(LIB))
    lib.qt_version.restype = C.c_char_p
    lib.qt_last_error.restype = C.c_char_p
    lib.qt_init.restype = C.c_void_p
    lib.qt_init.argtypes = [C.POINTER(InitParams)]
    lib.qt_free.argtypes = [C.c_void_p]
    lib.qt_init_default_params.argtypes = [C.POINTER(InitParams)]
    lib.qt_tts_default_params.argtypes = [C.POINTER(TTSParams)]
    lib.qt_extract_voice_ref.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.c_int, C.POINTER(VoiceRef)]
    lib.qt_voice_ref_free.argtypes = [C.POINTER(VoiceRef)]
    lib.qt_synthesize.argtypes = [C.c_void_p, C.POINTER(TTSParams), C.POINTER(Audio)]
    lib.qt_audio_free.argtypes = [C.POINTER(Audio)]
    lib.qt_log_set.argtypes = [LOG_CB, C.c_void_p]
    return lib


@LOG_CB
def _log(level, msg, _):
    if level >= 2:  # WARN and ERROR; INFO would print on every sentence
        print(f"[voice] {msg.decode(errors='replace')}", flush=True)


class TTS:
    def __init__(self):
        self.lib = _load()
        self.lib.qt_log_set(_log, None)
        p = InitParams()
        self.lib.qt_init_default_params(C.byref(p))
        if p.abi_version != ABI:  # the structs above would no longer match the library's
            raise RuntimeError(f"qwentts.cpp ABI is {p.abi_version}, mentalese/tts.py mirrors {ABI}: "
                               "update the structs from vendor/qwentts.cpp/src/qwen.h")
        p.talker_path, p.codec_path = str(TALKER).encode(), str(CODEC).encode()
        self.ctx = self.lib.qt_init(C.byref(p))
        if not self.ctx:
            raise RuntimeError(f"qwentts: {self.lib.qt_last_error().decode()}")
        self.ref = VoiceRef()
        self.ref_text = b""
        self.voice_lang = None
        self.keep_accent = False

    def set_voice(self, ref_audio: str, ref_text: str, lang: str | None = None, keep_accent: bool = False):
        """Reference recording for cloning, encoded once. `lang` is the sample's language: other
        languages are spoken from the voice fingerprint only, unless keep_accent."""
        wav, sr = sf.read(ref_audio, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            wav = soxr.resample(wav, sr, SR)
        wav = np.ascontiguousarray(wav, dtype=np.float32)
        self.lib.qt_voice_ref_free(C.byref(self.ref))
        if self.lib.qt_extract_voice_ref(self.ctx, wav.ctypes.data_as(C.POINTER(C.c_float)), len(wav),
                                         C.byref(self.ref)):
            raise RuntimeError(f"qwentts: {self.lib.qt_last_error().decode()}")
        self.ref_text = ref_text.encode()
        self.voice_lang, self.keep_accent = lang, keep_accent

    def warmup(self):
        # one same-language and one other-language sentence, so both clone modes are ready
        for text, lang in [("Hello there.", "en"), ("Привет.", "ru")]:
            for _ in self.stream(text, lang):
                pass

    @staticmethod
    def max_frames(text: str) -> int:
        """Upper bound on codec frames (12.5/s) for a text. Measured ~0.7 frames per Latin/Cyrillic
        char and ~2.1 per CJK char; allow ~2.3x that. Sampling occasionally misses the stop
        token, and without a cap that babbles for up to 48 s."""
        cjk = sum(1 for c in text if "　" <= c <= "鿿" or "가" <= c <= "힯")
        return int(1.6 * (len(text) - cjk) + 4.8 * cjk) + 24

    def uses_fingerprint(self, lang: str) -> bool:
        return not self.keep_accent and self.voice_lang is not None and lang != self.voice_lang

    @classmethod
    def parts(cls, text: str) -> list[str]:
        """Long text (a typed paragraph) split at sentence ends, then commas/spaces, into pieces of
        at most MAX_PART_FRAMES. vendor/qwentts.cpp is patched to a 1024-position talker cache
        (upstream: 4096, 670 MB more VRAM); the longest prompt (15 s voice sample + its
        transcript + text) is ~400 positions, so each piece's frames must stay under ~600."""
        if cls.max_frames(text) <= MAX_PART_FRAMES:
            return [text]
        out, cur = [], ""
        for piece in re.findall(r".+?(?:[.!?。！？…]+\s*|[,;，、；]\s*|\s+|$)", text, re.S):
            while cls.max_frames(piece) > MAX_PART_FRAMES:  # one unbroken run (CJK): cut it
                if cur:
                    out.append(cur)
                    cur = ""
                n = len(piece)
                while cls.max_frames(piece[:n]) > MAX_PART_FRAMES:
                    n -= 1
                out.append(piece[:n])
                piece = piece[n:]
            if cur and cls.max_frames(cur + piece) > MAX_PART_FRAMES:
                out.append(cur)
                cur = ""
            cur += piece
        return [p.strip() for p in [*out, cur] if p.strip()]

    def stream(self, text: str, lang: str) -> Iterator[tuple[np.ndarray, int]]:
        """Yields (float32 mono chunk, 24000). The first chunk is one 80 ms frame."""
        for part in self.parts(text):
            yield from self._stream(part, lang)

    def _stream(self, text: str, lang: str) -> Iterator[tuple[np.ndarray, int]]:
        p = TTSParams()
        self.lib.qt_tts_default_params(C.byref(p))
        p.text, p.lang = text.encode(), LANGS[lang][0].lower().encode()
        p.max_new_tokens = self.max_frames(text)
        p.ref_spk_emb, p.ref_spk_dim = self.ref.ref_spk_emb, self.ref.ref_spk_dim
        # Default (ICL) mode continues the reference recording, so it also carries its
        # pronunciation: an English sample speaking Russian came out with a heavy accent (ASR
        # error 26% vs 0.7%). The x-vector alone keeps the voice without the accent; it's a bit
        # less similar but sounds native. Same-language output keeps ICL.
        if not self.uses_fingerprint(lang):
            p.ref_codes, p.ref_T, p.ref_text = self.ref.ref_codes, self.ref.ref_T, self.ref_text

        chunks: queue.SimpleQueue = queue.SimpleQueue()

        def on_chunk(samples, n, _):  # runs on the library's worker thread
            chunks.put(np.ctypeslib.as_array(samples, (n,)).copy())
            return True

        p.on_chunk = CHUNK_CB(on_chunk)  # the struct holds the only reference while synthesizing
        out = Audio()

        def run():
            rc = self.lib.qt_synthesize(self.ctx, C.byref(p), C.byref(out))
            # the error message is per thread: read it here
            chunks.put(self.lib.qt_last_error().decode(errors="replace") if rc else "")

        threading.Thread(target=run, daemon=True).start()
        while not isinstance(item := chunks.get(), str):
            yield item, SR
        self.lib.qt_audio_free(C.byref(out))
        if item:
            raise RuntimeError(f"qwentts: {item}")
