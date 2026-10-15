"""PipeWire plumbing: a virtual "Mentalese Mic" for Discord, plus raw PCM in/out via parec/pacat."""

import queue
import subprocess
import threading

import numpy as np

SINK = "mentalese_sink"  # we play translated speech into this
MIC = "mentalese_mic"    # Discord records from this (a remap of the sink's monitor)


def _pactl(*args) -> str:
    return subprocess.run(["pactl", *args], capture_output=True, text=True, check=True).stdout


def ensure_virtual_mic():
    """Create the virtual mic if it isn't there yet. Idempotent; lives until logout."""
    names = _pactl("list", "short", "sinks") + _pactl("list", "short", "sources")
    if SINK not in names:
        _pactl("load-module", "module-null-sink", f"sink_name={SINK}", "rate=48000", "channels=1",
               "sink_properties=device.description=Mentalese-Internal-DONT-SELECT")
    if f"\t{MIC}\t" not in names:
        _pactl("load-module", "module-remap-source", f"master={SINK}.monitor", f"source_name={MIC}",
               "source_properties=device.description=Mentalese-Mic")


def remove_virtual_mic():
    for line in _pactl("list", "short", "modules").splitlines():
        if SINK in line or MIC in line:
            _pactl("unload-module", line.split("\t")[0])


class MicReader:
    """Reads mono float32 frames from a PipeWire source (default: system default mic)."""

    def __init__(self, sr=16000, frame=512, device=None):
        self.frame = frame
        cmd = ["parec", f"--rate={sr}", "--channels=1", "--format=float32le", "--latency-msec=20",
               "--client-name=mentalese"]
        if device:
            cmd.append(f"--device={device}")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    def read(self) -> np.ndarray:
        data = self.proc.stdout.read(self.frame * 4)
        if len(data) < self.frame * 4:
            raise EOFError("mic stream ended")
        return np.frombuffer(data, dtype=np.float32)

    def close(self):
        self.proc.terminate()


class Player:
    """Streams mono float32 audio to a PipeWire sink at a fixed sample rate.

    play() never blocks: pacat only drains in real time, so writing directly would stall the
    caller for the whole length of the audio. A writer thread feeds it instead."""

    def __init__(self, sr: int, device: str | None = None):
        cmd = ["pacat", "--playback", f"--rate={sr}", "--channels=1", "--format=float32le",
               "--latency-msec=40", "--client-name=mentalese"]
        if device:
            cmd.append(f"--device={device}")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.queue: queue.Queue[bytes | None] = queue.Queue()
        self.writer = threading.Thread(target=self._write, daemon=True)
        self.writer.start()

    def _write(self):
        while (data := self.queue.get()) is not None:
            self.proc.stdin.write(data)
            self.proc.stdin.flush()
        self.proc.stdin.close()

    def play(self, audio: np.ndarray):
        self.queue.put(audio.astype(np.float32).tobytes())

    def close(self):
        """Lets queued audio finish playing, then exits."""
        self.queue.put(None)
        self.writer.join(timeout=120)
        self.proc.wait(timeout=60)
