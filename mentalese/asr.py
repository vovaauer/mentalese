"""Speech to text with Qwen3-ASR-0.6B (Q8_0 GGUF), served by a local llama.cpp server.
Detects the spoken language on its own."""

import base64
import io
import re

import numpy as np
import soundfile as sf

from .langs import code_for
from .server import MODELS, LlamaServer

MODEL = MODELS / "Qwen3-ASR-0.6B-Q8_0.gguf"
MMPROJ = MODELS / "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"  # same transcripts as bf16 on every test clip
OUTPUT = re.compile(r"language\s*(\w*)\s*<asr_text>(.*)", re.S)


class ASR:
    """Starts its server in the background; call wait() before the first transcribe().

    gpu_encoder=False runs the audio encoder on the CPU: 0.76 GB less VRAM, ~40 ms slower on a
    typical 4-8 s sentence (94-132 vs 59-73 ms), still faster than the PyTorch version (~180 ms)."""

    def __init__(self, gpu_encoder: bool = False):
        # 15 s of speech is ~220 prompt tokens. --no-warmup: the warmup sizes the encoder's GPU
        # buffers for its maximum input and keeps them (~400 MB); our own warmup is 1 s.
        self.server = LlamaServer("-m", str(MODEL), "--mmproj", str(MMPROJ), "-ngl", "99",
                                  "-c", "1024", "-b", "256", "-ub", "256", "-fa", "on", "--no-warmup",
                                  *([] if gpu_encoder else ["--no-mmproj-offload"]))

    def wait(self):
        self.server.wait()
        return self

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> tuple[str | None, str]:
        """Returns (language code or None, text). Audio must be 16 kHz mono float32."""
        assert sr == 16000
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
        wav = base64.b64encode(buf.getvalue()).decode()
        # cap output by audio length (fast Russian is ~12 tokens/s; 20 leaves headroom): on tiny clips
        # the model can loop on one token until the cap
        out = self.server.chat([{"type": "input_audio", "input_audio": {"data": wav, "format": "wav"}}],
                               max_tokens=int(len(audio) / sr * 20) + 16, temperature=0)
        m = OUTPUT.match(out.strip())
        if not m:
            return None, ""
        return code_for(m.group(1)) if m.group(1) != "None" else None, m.group(2).strip()

    def close(self):
        self.server.close()
