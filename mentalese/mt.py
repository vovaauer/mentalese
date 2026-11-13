"""Translation with Bilibili's Index-Translate-2B, served by a local llama.cpp server."""

from .langs import LANGS
from .server import MODELS, LlamaServer

# 2B over Hy-MT2-7B (the previous translator): same tone on casual chat (0/26 formal "вы" for
# both), 2.4x faster (median 38 vs 93 ms a line) and 1.5 GB of VRAM instead of 4.8. It misses
# the odd idiom ("no way you pulled that off" came out as its opposite). Q6_K didn't fix that.
MODEL = MODELS / "Index-Translate-2B.Q4_K_M.gguf"

# The model card's two Chinese templates: plain, and "instTrans" with constraints, where the
# style goes in as a 【注意】 (soft) constraint. The source language name is left out when unknown.
PLAIN_TEMPLATE = "请将以下{source}文本翻译为{target}，直接输出翻译结果，不要进行任何解释。\n\n{text}"
STYLE_TEMPLATE = (
    "请将以下{source}对话翻译成{target}，并且严格遵循所有约束要求。\n\n"
    "【源文】\n{text}\n\n【约束要求】\n1. 【注意】{style}\n\n只输出译文，不要有任何额外说明。"
)

# Without a style the model goes formal in Russian ("вы", 3/26 lines); with it, 0/26.
# "add nothing" keeps short lines from being embellished ("bye" -> "ну пока, чуваки! 😎").
DEFAULT_STYLE = "casual and informal, like talking to a close friend; translate only what is said, add nothing"


class Translator:
    """Starts its server in the background; call wait() before the first translate()."""

    def __init__(self, style: str | None = DEFAULT_STYLE):
        self.style = style  # None or "" = plain translation
        self.server = LlamaServer("-m", str(MODEL), "-ngl", "99",
                                  # a spoken sentence is ~50 tokens: small context and batches keep VRAM down
                                  "-c", "1024", "-b", "512", "-ub", "256", "-fa", "on")

    def wait(self):
        self.server.wait()
        return self

    def translate(self, text: str, source: str | None, target: str) -> str:
        names = dict(source=LANGS[source][1] if source in LANGS else "", target=LANGS[target][1], text=text)
        if self.style:
            out = self._complete(STYLE_TEMPLATE.format(style=self.style, **names), text)
            if not self._leaked(out, text):
                return out
            # the model recited the instruction instead of translating: redo it plain
        return self._complete(PLAIN_TEMPLATE.format(**names), text)

    def _leaked(self, out: str, text: str) -> bool:
        if any(b in out and b not in text for b in "[]【】"):
            return True
        style_words = {w for w in self.style.lower().split() if len(w) > 3}
        return len(style_words & set(out.lower().replace(",", " ").split())) >= len(style_words) / 2

    def _complete(self, prompt: str, text: str) -> str:
        # greedy, as the model card evaluates it
        out = self.server.chat(prompt, max_tokens=512, temperature=0).split("</think>")[-1].strip()
        # speech has no line breaks: if the model added a preamble, the last paragraph is the answer
        if "\n" in out and "\n" not in text:
            out = [line for line in out.split("\n") if line.strip()][-1].strip()
        return out

    def close(self):
        self.server.close()
