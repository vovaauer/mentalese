"""Reference data for crates/core/tests/vad.rs: Silero VAD run by onnxruntime (the official way)
over test clips, so the Rust port can be checked against it frame by frame.

  uv run --no-project --with onnxruntime --with numpy --with soundfile --with soxr tools/vad_reference.py [clips/]

Reads <clips>/bench_{en,ru,zh}.wav (default out/), writes build/vad-ref/: 16 kHz audio, per-frame
probabilities and start/end events, each as the test expects.
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime
import soundfile as sf
import soxr

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "build/vad-ref"
SR, WINDOW, CONTEXT = 16000, 512, 64


class VAD:
    """Silero's OnnxWrapper + VADIterator, as mentalese used them before the Rust port."""

    session = None

    def __init__(self, threshold=0.5, min_silence_ms=100):
        if VAD.session is None:
            opts = onnxruntime.SessionOptions()
            opts.inter_op_num_threads = opts.intra_op_num_threads = 1
            VAD.session = onnxruntime.InferenceSession(str(REPO / "models/silero_vad.onnx"), opts,
                                                       providers=["CPUExecutionProvider"])
        self.threshold, self.min_silence = threshold, SR * min_silence_ms // 1000
        self.state = np.zeros((2, 1, 128), np.float32)
        self.context = np.zeros((1, CONTEXT), np.float32)
        self.triggered, self.temp_end, self.current = False, 0, 0

    def prob(self, frame):
        x = np.concatenate([self.context, frame.reshape(1, WINDOW)], axis=1)
        out, self.state = self.session.run(None, {"input": x, "state": self.state, "sr": np.array(SR, np.int64)})
        self.context = x[:, -CONTEXT:]
        return float(out[0, 0])

    def __call__(self, frame):
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


def main():
    clips_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "out"
    OUT.mkdir(parents=True, exist_ok=True)
    clips = {}
    for lang in ("en", "ru", "zh"):
        a, sr = sf.read(clips_dir / f"bench_{lang}.wav", dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        clips[lang] = soxr.resample(a, sr, SR).astype(np.float32)
    clips["silence"] = np.zeros(2 * SR, np.float32)
    clips["noise"] = (np.random.default_rng(0).standard_normal(2 * SR) * 1e-3).astype(np.float32)

    for name, audio in clips.items():
        audio.tofile(OUT / f"{name}.f32")
        n = len(audio) // WINDOW
        v = VAD()
        probs = np.array([v.prob(audio[i * WINDOW:(i + 1) * WINDOW]) for i in range(n)], np.float32)
        probs.tofile(OUT / f"{name}.probs.f32")
        lines = []
        for ms in (400, 100):
            v = VAD(threshold=0.5, min_silence_ms=ms)
            for i in range(n):
                if e := v(audio[i * WINDOW:(i + 1) * WINDOW]):
                    lines.append(f"{ms} {i} {e}")
        (OUT / f"{name}.events").write_text("".join(line + "\n" for line in lines))
        print(f"{name}: {n} frames, {len(lines)} events")


if __name__ == "__main__":
    main()
