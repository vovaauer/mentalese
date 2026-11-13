"""Silero VAD (ONNX) on the CPU with onnxruntime: the silero-vad package's OnnxWrapper and
VADIterator in numpy, so the app doesn't need PyTorch."""

import numpy as np
import onnxruntime

from .server import MODELS

MODEL = MODELS / "silero_vad.onnx"
SR = 16000
WINDOW = 512   # samples per call at 16 kHz (32 ms)
CONTEXT = 64   # trailing samples of the previous window the model sees again


_session = None


def _shared_session() -> onnxruntime.InferenceSession:
    """One model for every speaker: the recurrent state lives in each VAD, and run() is thread-safe."""
    global _session
    if _session is None:
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1
        _session = onnxruntime.InferenceSession(str(MODEL), sess_options=opts, providers=["CPUExecutionProvider"])
    return _session


class VAD:
    """Feed 512-sample frames; returns "start" when speech begins, "end" after `min_silence_ms`
    of silence, else None. Same thresholds as silero's VADIterator."""

    def __init__(self, threshold: float = 0.5, min_silence_ms: int = 100):
        self.session = _shared_session()
        self.threshold = threshold
        self.min_silence = SR * min_silence_ms // 1000
        self._sr = np.array(SR, dtype=np.int64)
        self.reset_states()

    def reset_states(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.context = np.zeros((1, CONTEXT), np.float32)
        self.triggered, self.temp_end, self.current = False, 0, 0

    def prob(self, frame: np.ndarray) -> float:
        x = np.concatenate([self.context, frame.reshape(1, WINDOW).astype(np.float32, copy=False)], axis=1)
        out, self.state = self.session.run(None, {"input": x, "state": self.state, "sr": self._sr})
        self.context = x[:, -CONTEXT:]
        return float(out[0, 0])

    def __call__(self, frame: np.ndarray) -> str | None:
        self.current += WINDOW
        p = self.prob(frame)
        if p >= self.threshold:
            self.temp_end = 0
            if not self.triggered:
                self.triggered = True
                return "start"
        elif p < self.threshold - 0.15 and self.triggered:
            self.temp_end = self.temp_end or self.current
            if self.current - self.temp_end >= self.min_silence:
                self.temp_end, self.triggered = 0, False
                return "end"
        return None
