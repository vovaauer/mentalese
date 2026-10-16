"""Language of typed text, so text already in the target language is spoken as-is."""

from lingua import Language, LanguageDetectorBuilder

from .langs import LANGS

_detector = None


def detect(text: str, fallback: str) -> str:
    """Language code of `text`; `fallback` (your own language) when unsure. Short chat slang
    like "lol" or "gg" comes back as Italian with low confidence, and is almost always yours."""
    global _detector
    if _detector is None:
        _detector = LanguageDetectorBuilder.from_languages(
            *(getattr(Language, name.upper()) for name, _ in LANGS.values())
        ).with_preloaded_language_models().build()
    lang = _detector.detect_language_of(text)
    if lang is None or _detector.compute_language_confidence(text, lang) < 0.5:
        return fallback
    return lang.iso_code_639_1.name.lower()
