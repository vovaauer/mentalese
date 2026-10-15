"""Translation with Tencent Hy-MT2-1.8B, served by a local llama.cpp server."""

import atexit
import socket
import subprocess
import time
from pathlib import Path

import requests

from .langs import LANGS

ROOT = Path(__file__).resolve().parent.parent
SERVER_BIN = ROOT / "vendor/llama.cpp/build/bin/llama-server"
MODEL = ROOT / "models/Hy-MT2-1.8B-Q8_0.gguf"

# Hy-MT2 model card: one template for anything involving Chinese, one for the rest.
ZH_TEMPLATE = "将以下文本翻译为{target}，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
EN_TEMPLATE = (
    "Translate the following text into {target}.\n"
    "Note that you should only output the translated result without any additional explanation:\n\n{text}"
)


class Translator:
    def __init__(self):
        with socket.socket() as sock:  # grab a free port
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.proc = subprocess.Popen(
            [str(SERVER_BIN), "-m", str(MODEL), "--port", str(port), "--host", "127.0.0.1",
             "-ngl", "99", "-c", "2048", "-np", "1", "--no-webui"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
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
        if "zh" in (source, target):
            prompt = ZH_TEMPLATE.format(target=zh_name, text=text)
        else:
            prompt = EN_TEMPLATE.format(target=en_name, text=text)
        r = requests.post(f"{self.url}/v1/chat/completions", json={
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7, "top_p": 0.6, "top_k": 20, "repeat_penalty": 1.05,
            "max_tokens": 512,
        }, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
