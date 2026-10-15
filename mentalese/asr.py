"""Speech to text with Qwen3-ASR-0.6B (native transformers port). Detects the spoken language on its own."""

import numpy as np
import torch

from .langs import code_for

MODEL_ID = "Qwen/Qwen3-ASR-0.6B-hf"


class ASR:
    def __init__(self, hotwords: str | None = None):
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0")
        # names / slang to bias recognition towards, e.g. "Vocabulary: spawn, DonutSMP"
        self.hotwords = hotwords

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> tuple[str | None, str]:
        """Returns (language code or None, text). Audio must be 16 kHz mono float32."""
        assert sr == 16000
        inputs = self.processor.apply_transcription_request(
            audio=audio, prompt=self.hotwords).to(self.model.device, self.model.dtype)
        out = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        parsed = self.processor.decode(out[:, inputs["input_ids"].shape[1]:], return_format="parsed")[0]
        return code_for(parsed["language"]), parsed["transcription"].strip()
