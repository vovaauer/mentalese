"""Receives each Discord call participant's audio, separately, from the Vencord plugin (vencord/mentalese)."""

import socket
import threading

import numpy as np

PORT = 47321
MSG_AUDIO, MSG_OPEN, MSG_CLOSE = 1, 2, 3


class Incoming:
    """Listens on UDP and calls on_frame(stream, frame) with 16 kHz float32 frames of `frame` samples."""

    def __init__(self, on_open, on_frame, on_close, frame=512):
        self.on_open, self.on_frame, self.on_close, self.frame = on_open, on_frame, on_close, frame
        self.buffers: dict[int, np.ndarray] = {}
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)
        self.sock.bind(("127.0.0.1", PORT))
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            data = self.sock.recv(65536)
            if len(data) < 5:
                continue
            kind, num = data[0], int.from_bytes(data[1:5], "little")
            if kind == MSG_OPEN:
                self.buffers[num] = np.zeros(0, np.float32)
                self.on_open(num, data[5:].decode(errors="replace"))
            elif kind == MSG_CLOSE:
                if self.buffers.pop(num, None) is not None:
                    self.on_close(num)
            elif kind == MSG_AUDIO:
                if num not in self.buffers:  # mentalese started mid-call: adopt the stream
                    self.buffers[num] = np.zeros(0, np.float32)
                    self.on_open(num, "")
                pcm = np.frombuffer(data[5:], dtype="<i2").astype(np.float32) / 32768
                buf = np.concatenate([self.buffers[num], pcm])
                n = len(buf) // self.frame * self.frame
                for i in range(0, n, self.frame):
                    self.on_frame(num, buf[i:i + self.frame])
                self.buffers[num] = buf[n:]
