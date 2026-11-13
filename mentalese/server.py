"""A local llama.cpp server (llama-server) that lives and dies with mentalese."""

import atexit
import ctypes
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
SERVER_BIN = ROOT / "vendor/llama.cpp/build/bin/llama-server"
MODELS = ROOT / "models"


def _die_with_parent():
    """Linux: kill the server when mentalese exits, even on a crash or a closed terminal.
    SIGKILL, not SIGTERM: llama-server sometimes ignores SIGTERM and lingers, holding VRAM."""
    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG


def kill_leftovers():
    """Kill servers orphaned by an earlier mentalese run: ours (matched by binary path) whose
    parent is no longer a mentalese process. A second mentalese running at the same time must
    not kill the first one's servers."""
    def cmdline(pid) -> list[str]:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().decode(errors="replace").split("\0")

    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            if cmdline(pid)[0] != str(SERVER_BIN):
                continue
            with open(f"/proc/{pid}/stat") as f:
                ppid = f.read().rsplit(")", 1)[1].split()[1]
            try:
                parent_alive = any("mentalese" in a for a in cmdline(ppid))
            except OSError:
                parent_alive = False
            if not parent_alive:
                os.kill(int(pid), signal.SIGKILL)
        except (OSError, IndexError):
            continue


_cleaned = False


class LlamaServer:
    """Starts llama-server on a free localhost port with `args`, without waiting for it: create
    several, then wait() on each, so they load in parallel."""

    def __init__(self, *args: str):
        global _cleaned
        if not _cleaned:
            kill_leftovers()
            _cleaned = True
        with socket.socket() as sock:  # grab a free port
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.session = requests.Session()  # keep-alive: no new TCP connection per sentence
        self.proc = subprocess.Popen(
            [str(SERVER_BIN), *args, "--port", str(port), "--host", "127.0.0.1", "-np", "1", "--no-webui"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=_die_with_parent,
        )
        atexit.register(self.close)

    def wait(self, timeout: float = 60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self.session.get(f"{self.url}/health", timeout=1).status_code == 200:
                    return self
            except requests.RequestException:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited during startup: {' '.join(self.proc.args[1:3])}")
            time.sleep(0.1)
        raise RuntimeError("llama-server did not become ready")

    def chat(self, content, max_tokens: int, **sampling) -> str:
        r = self.session.post(f"{self.url}/v1/chat/completions", json={
            "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}, **sampling,
        }, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
