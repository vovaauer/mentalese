//! Language of typed text, so text already in the target language is spoken as-is.

use std::sync::LazyLock;

use lingua::{Language, LanguageDetector, LanguageDetectorBuilder};

static DETECTOR: LazyLock<LanguageDetector> = LazyLock::new(|| {
    use Language::*;
    LanguageDetectorBuilder::from_languages(&[
        English, Russian, Chinese, Japanese, Korean, German, French, Spanish, Portuguese, Italian,
    ])
    .with_preloaded_language_models()
    .build()
});

/// Loads the language models now rather than on the first typed line.
pub fn preload() {
    LazyLock::force(&DETECTOR);
}

/// Language code of `text`; `fallback` (your own language) when unsure. Short chat slang like
/// "lol" or "gg" comes back as Italian with low confidence, and is almost always yours.
pub fn detect(text: &str, fallback: &'static str) -> &'static str {
    // one or two Latin-alphabet words (names, "gg", "nice") are a coin flip ("yaron" -> Spanish 0.59)
    if text.split_whitespace().count() <= 2 && text.is_ascii() {
        return fallback;
    }
    let Some(lang) = DETECTOR.detect_language_of(text) else { return fallback };
    if DETECTOR.compute_language_confidence(text, lang) < 0.5 {
        return fallback;
    }
    use Language::*;
    match lang {
        English => "en",
        Russian => "ru",
        Chinese => "zh",
        Japanese => "ja",
        Korean => "ko",
        German => "de",
        French => "fr",
        Spanish => "es",
        Portuguese => "pt",
        Italian => "it",
    }
}
