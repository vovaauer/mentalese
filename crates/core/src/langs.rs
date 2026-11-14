//! The 10 languages Qwen3-TTS can speak. Speech recognition and translation cover more, but
//! spoken output is limited to these.

pub struct Lang {
    pub code: &'static str,
    /// What Qwen3-TTS and Qwen3-ASR call it.
    pub name: &'static str,
    /// What Index-Translate's Chinese prompt templates call it.
    pub zh: &'static str,
}

pub const LANGS: [Lang; 10] = [
    Lang { code: "en", name: "English", zh: "英语" },
    Lang { code: "ru", name: "Russian", zh: "俄语" },
    Lang { code: "zh", name: "Chinese", zh: "中文" },
    Lang { code: "ja", name: "Japanese", zh: "日语" },
    Lang { code: "ko", name: "Korean", zh: "韩语" },
    Lang { code: "de", name: "German", zh: "德语" },
    Lang { code: "fr", name: "French", zh: "法语" },
    Lang { code: "es", name: "Spanish", zh: "西班牙语" },
    Lang { code: "pt", name: "Portuguese", zh: "葡萄牙语" },
    Lang { code: "it", name: "Italian", zh: "意大利语" },
];

pub fn lang(code: &str) -> Option<&'static Lang> {
    LANGS.iter().find(|l| l.code == code)
}

/// Maps a code or an ASR language name like "English" to a code we can speak.
pub fn code_for(name: &str) -> Option<&'static str> {
    let name = name.trim();
    LANGS.iter().find(|l| l.code.eq_ignore_ascii_case(name) || l.name.eq_ignore_ascii_case(name)).map(|l| l.code)
}
