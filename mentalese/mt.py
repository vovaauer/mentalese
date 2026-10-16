"""Translation with Tencent Hy-MT2-7B, served by a local llama.cpp server."""

import atexit
import ctypes
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import requests

from .langs import LANGS

ROOT = Path(__file__).resolve().parent.parent
SERVER_BIN = ROOT / "vendor/llama.cpp/build/bin/llama-server"
# 7B over 1.8B: the 1.8B ignores tone (formal "вы" 15/18 even when asked for casual) and often
# answers filler sounds by reciting the style instruction. ~+100 ms per sentence.
MODEL = ROOT / "models/Hy-MT2-7B-Q4_K_M.gguf"

# Hy-MT2 model card: one template for anything involving Chinese, one for the rest,
# each in a plain and a "style" variant (the style is any free-text description of tone).
ZH_TEMPLATE = "将以下文本翻译为{target}，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
EN_TEMPLATE = (
    "Translate the following text into {target}. "
    "Note that you should only output the translated result without any additional explanation:\n\n{text}"
)
ZH_STYLE_TEMPLATE = "请将以下文本翻译为{target}。\n注意翻译的风格要严格符合【{style}】\n\n{text}"
EN_STYLE_TEMPLATE = (
    "Please translate the following text into {target}. "
    "Note that the translation style must strictly conform to [{style}]:\n\n{text}"
)

# "add nothing" matters: without it, short lines get embellished ("bye" -> "ну пока, чуваки! 😎",
# a bare name -> "Yaron? Oh, hey! That's him!"); 13/24 short inputs before, 0/24 after, still informal.
DEFAULT_STYLE = "casual and informal, like talking to a close friend; translate only what is said, add nothing"


def _die_with_parent():
    """Linux: kill the server when mentalese exits, even on a crash or a closed terminal.
    SIGKILL, not SIGTERM: llama-server sometimes ignores SIGTERM and lingers, holding ~5 GB of VRAM."""
    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG


def _kill_leftovers():
    """Kill translator servers orphaned by an earlier mentalese run (only ours, matched by binary path)."""
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv0 = f.read().split(b"\0")[0].decode()
        except OSError:
            continue
        if argv0 == str(SERVER_BIN):
            os.kill(int(pid), signal.SIGKILL)


class Translator:
    def __init__(self, style: str | None = DEFAULT_STYLE):
        self.style = style  # None or "" = plain translation
        _kill_leftovers()
        with socket.socket() as sock:  # grab a free port
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.proc = subprocess.Popen(
            [str(SERVER_BIN), "-m", str(MODEL), "--port", str(port), "--host", "127.0.0.1",
             "-ngl", "99", "-np", "1", "--no-webui",
             # a spoken sentence is ~50 tokens: small context and batches keep VRAM down
             "-c", "1024", "-b", "512", "-ub", "256", "-fa", "on"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=_die_with_parent,
        )
        atexit.register(self.close)
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                if requests.get(f"{self.url}/health", timeout=1).status_code == 200:
                    return
            except requests.RequestException:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server exited during startup")
            time.sleep(0.2)
        raise RuntimeError("llama-server did not become ready")

    def translate(self, text: str, source: str | None, target: str) -> str:
        en_name, zh_name = LANGS[target]
        zh = "zh" in (source, target)
        if self.style:
            template = ZH_STYLE_TEMPLATE if zh else EN_STYLE_TEMPLATE
        else:
            template = ZH_TEMPLATE if zh else EN_TEMPLATE
        prompt = template.format(target=zh_name if zh else en_name, style=self.style, text=text)
        out = self._complete(prompt, text)
        if self.style and self._leaked(out, text):
            # the model recited the style instruction instead of translating: redo it plain
            plain = ZH_TEMPLATE if zh else EN_TEMPLATE
            out = self._complete(plain.format(target=zh_name if zh else en_name, text=text), text)
        return out

    def _leaked(self, out: str, text: str) -> bool:
        if any(b in out and b not in text for b in "[]【】"):
            return True
        style_words = {w for w in self.style.lower().split() if len(w) > 3}
        return len(style_words & set(out.lower().replace(",", " ").split())) >= len(style_words) / 2

    def _complete(self, prompt: str, text: str) -> str:
        r = requests.post(f"{self.url}/v1/chat/completions", json={
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7, "top_p": 0.6, "top_k": 20, "repeat_penalty": 1.05,
            "max_tokens": 512,
        }, timeout=30)
        r.raise_for_status()
        out = r.json()["choices"][0]["message"]["content"].strip()
        # With a style, the model can echo the instruction first ("Note: the style
        # must...\n\n<translation>"). Speech has no line breaks, so the last paragraph is the answer.
        if "\n" in out and "\n" not in text:
            out = [line for line in out.split("\n") if line.strip()][-1].strip()
        return out

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
