"""Subtitles from any app's audio (e.g. Telegram), recorded from PipeWire without rerouting it.

Unlike the Vesktop plugin this gets one mixed stream per app, so it's ideal for 1:1 calls
but people talking over each other in group calls get mixed together.
"""

import json
import subprocess
import threading
import time

import numpy as np


class AppCapture:
    """Calls on_frame(stream, frame) with 16 kHz float32 frames of every playback stream whose
    app name, binary or node name contains `app`. Picks up streams as they appear (a call starting)."""

    def __init__(self, app: str, on_frame, frame: int = 512, stream_base: int = 100_000):
        self.app, self.on_frame, self.frame, self.base = app.lower(), on_frame, frame, stream_base
        self.active: set[int] = set()
        threading.Thread(target=self._watch, daemon=True).start()

    def _matching(self) -> list[int]:
        out = subprocess.run(["pactl", "-f", "json", "list", "sink-inputs"], capture_output=True, text=True).stdout
        found = []
        for s in json.loads(out or "[]"):
            p = s.get("properties", {})
            names = " ".join(str(p.get(k, "")) for k in
                             ("application.name", "application.process.binary", "node.name")).lower()
            if self.app in names and "mentalese" not in names:
                found.append(s["index"])
        return found

    def _watch(self):
        while True:
            for idx in self._matching():
                if idx not in self.active:
                    self.active.add(idx)
                    threading.Thread(target=self._read, args=(idx,), daemon=True).start()
            time.sleep(2)

    def _read(self, idx: int):
        proc = subprocess.Popen(
            ["parec", f"--monitor-stream={idx}", "--rate=16000", "--channels=1", "--format=float32le",
             "--latency-msec=20", "--client-name=mentalese-listen"], stdout=subprocess.PIPE)
        size = self.frame * 4
        try:
            while len(data := proc.stdout.read(size)) == size:
                self.on_frame(self.base + idx, np.frombuffer(data, dtype=np.float32))
        finally:  # the stream ended (call over); a new one gets picked up by _watch
            proc.terminate()
            self.active.discard(idx)
