"""The 10 languages Qwen3-TTS can speak. ASR and translation cover more, but output is capped here."""

# code -> (English name, Chinese name). Qwen3-TTS takes the English name; Hy-MT2's
# Chinese prompt template wants the Chinese name.
LANGS = {
    "en": ("English", "英语"),
    "ru": ("Russian", "俄语"),
    "zh": ("Chinese", "中文"),
    "ja": ("Japanese", "日语"),
    "ko": ("Korean", "韩语"),
    "de": ("German", "德语"),
    "fr": ("French", "法语"),
    "es": ("Spanish", "西班牙语"),
    "pt": ("Portuguese", "葡萄牙语"),
    "it": ("Italian", "意大利语"),
}

_BY_NAME = {name.lower(): code for code, (name, _) in LANGS.items()}


def code_for(name: str | None) -> str | None:
    """Map an ASR language name like 'English' or 'Chinese' to a code, if we know it."""
    if not name:
        return None
    name = name.strip().lower()
    if name in LANGS:
        return name
    return _BY_NAME.get(name)
