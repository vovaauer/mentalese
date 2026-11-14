//! Speech in a cloned voice with Qwen3-TTS-0.6B-Base (qwentts.cpp).

use std::ffi::CString;

use crate::engines::{Engines, Voice};
use crate::{langs, models};

pub const SR: u32 = 24_000;

/// Longest piece spoken in one request, in codec frames (12.5/s). The voice engine is patched to
/// a 1024-position talker cache (upstream: 4096, 670 MB more VRAM); the longest prompt (15 s
/// sample + its transcript + text) is ~400 positions, so a piece must stay under ~600 frames.
const MAX_PART_FRAMES: usize = 480;

pub struct Tts {
    voice: Voice,
    ref_text: CString,
    /// Language of the voice sample.
    pub voice_lang: Option<&'static str>,
    /// Speak other languages like the sample, accent and all (closer likeness, less native).
    pub keep_accent: bool,
}

impl Tts {
    pub fn load(engines: &Engines) -> Result<Tts, String> {
        let voice = engines.load_voice(&models::TALKER.path(), &models::CODEC.path())?;
        Ok(Tts { voice, ref_text: CString::default(), voice_lang: None, keep_accent: false })
    }

    /// The recording to clone (24 kHz mono), its transcript and language. Encoded once.
    pub fn set_voice(&mut self, wav: &[f32], text: &str, lang: Option<&'static str>) -> Result<(), String> {
        self.voice.set_ref(wav)?;
        self.ref_text = CString::new(text.replace('\0', "")).unwrap_or_default();
        self.voice_lang = lang;
        Ok(())
    }

    /// The default (in-context) cloning continues the sample recording, pronunciation included: an
    /// English sample speaking Russian came out with a heavy accent (ASR error 26% vs 0.7%). Other
    /// languages use the speaker fingerprint alone: a bit less similar but native-sounding.
    pub fn uses_fingerprint(&self, lang: &str) -> bool {
        !self.keep_accent && self.voice_lang.is_some_and(|v| v != lang)
    }

    pub fn warmup(&mut self) -> Result<(), String> {
        // one same-language and one other-language sentence, so both clone modes are ready
        for (text, lang) in [("Hello there.", "en"), ("Привет.", "ru")] {
            self.speak(text, lang, |_| true)?;
        }
        Ok(())
    }

    /// Speaks `text` in `lang` (a code), calling `on_chunk` with 24 kHz audio as it's made; the
    /// first chunk is one 80 ms frame. `on_chunk` returning false stops it.
    pub fn speak(&mut self, text: &str, lang: &str, mut on_chunk: impl FnMut(&[f32]) -> bool + Send) -> Result<(), String> {
        let name = langs::lang(lang).ok_or_else(|| format!("can't speak {lang}"))?.name.to_lowercase();
        let ref_text = (!self.uses_fingerprint(lang)).then_some(self.ref_text.as_c_str());
        for part in parts(text) {
            if !self.voice.say(&part, &name, max_frames(&part), ref_text, &mut on_chunk)? {
                break;
            }
        }
        Ok(())
    }
}

fn is_cjk(c: char) -> bool {
    ('\u{3000}'..='\u{9fff}').contains(&c) || ('\u{ac00}'..='\u{d7af}').contains(&c)
}

/// Upper bound on codec frames (12.5/s) for a text. Measured ~0.7 frames per Latin/Cyrillic char
/// and ~2.1 per CJK char; allow ~2.3x that. Sampling occasionally misses the stop token, and
/// without a cap that babbles for up to 48 s.
pub fn max_frames(text: &str) -> usize {
    let (mut cjk, mut other) = (0usize, 0usize);
    for c in text.chars() {
        if is_cjk(c) { cjk += 1 } else { other += 1 }
    }
    (1.6 * other as f32 + 4.8 * cjk as f32) as usize + 24
}

/// Long text (a typed paragraph) split at sentence ends, then commas and spaces, into pieces of at
/// most MAX_PART_FRAMES, in order.
pub fn parts(text: &str) -> Vec<String> {
    if max_frames(text) <= MAX_PART_FRAMES {
        return vec![text.to_string()];
    }
    let (mut out, mut cur) = (Vec::new(), String::new());
    for mut piece in pieces(text) {
        while max_frames(&piece) > MAX_PART_FRAMES {
            // one unbroken run (CJK): cut it
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            let chars: Vec<char> = piece.chars().collect();
            let mut n = chars.len();
            while max_frames(&chars[..n].iter().collect::<String>()) > MAX_PART_FRAMES {
                n -= 1;
            }
            out.push(chars[..n].iter().collect());
            piece = chars[n..].iter().collect();
        }
        if !cur.is_empty() && max_frames(&(cur.clone() + &piece)) > MAX_PART_FRAMES {
            out.push(std::mem::take(&mut cur));
        }
        cur.push_str(&piece);
    }
    out.push(cur);
    out.into_iter().map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect()
}

/// Splits after sentence ends, commas or whitespace, keeping the separators.
fn pieces(text: &str) -> Vec<String> {
    let ends = |c: char| ".!?。！？…".contains(c);
    let commas = |c: char| ",;，、；".contains(c);
    let chars: Vec<char> = text.chars().collect();
    let (mut out, mut i) = (Vec::new(), 0);
    while i < chars.len() {
        let start = i;
        i += 1;
        while i < chars.len() {
            let c = chars[i];
            if ends(c) {
                while i < chars.len() && ends(chars[i]) { i += 1 }
            } else if commas(c) {
                i += 1;
            } else if !c.is_whitespace() {
                i += 1;
                continue;
            }
            while i < chars.len() && chars[i].is_whitespace() { i += 1 }
            break;
        }
        out.push(chars[start..i].iter().collect());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_is_one_part() {
        assert_eq!(parts("Hello there."), vec!["Hello there."]);
    }

    #[test]
    fn long_text_keeps_everything_in_order() {
        let text = "This is a sentence that goes on for a while, with commas, and more words. ".repeat(12)
            + &"汉字".repeat(200);
        let ps = parts(&text);
        assert!(ps.len() > 1);
        assert!(ps.iter().all(|p| max_frames(p) <= MAX_PART_FRAMES));
        let squash = |s: &str| s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
        assert_eq!(squash(&ps.concat()), squash(&text));
    }
}
