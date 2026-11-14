//! Voices to speak with: a ~15 s recording plus its transcript, voices/<name>.wav and .txt.

use std::path::PathBuf;

use crate::{asr::Asr, langid, paths, resample, tts, wav};

/// What to read out loud when recording a voice.
pub const READ_ALOUD: &str = "Hey, it's me. I'm recording this so my voice can speak other languages. \
Honestly, I have no idea how it's going to sound, but let's find out. \
One, two, three, four, five. The quick brown fox jumps over the lazy dog.";

pub const RECORD_SECS: f32 = 15.0;

pub struct Voice {
    /// 24 kHz mono
    pub wav: Vec<f32>,
    pub text: String,
    pub lang: Option<&'static str>,
}

fn files(name: &str) -> (PathBuf, PathBuf) {
    let dir = paths::voices();
    (dir.join(format!("{name}.wav")), dir.join(format!("{name}.txt")))
}

/// A name that's safe as a file name.
pub fn valid_name(name: &str) -> bool {
    !name.is_empty() && name.len() <= 40 && name.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_')
}

pub fn list() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(paths::voices())
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.file_name().to_str()?.strip_suffix(".wav").map(String::from))
        .collect();
    names.sort();
    names
}

pub fn exists(name: &str) -> bool {
    files(name).0.exists()
}

pub fn load(name: &str) -> Result<Voice, String> {
    let (wav_path, txt_path) = files(name);
    if !wav_path.exists() {
        return Err(format!("there's no voice called '{name}' yet: record one first"));
    }
    let (wav, sr) = wav::read(&wav_path)?;
    let text = std::fs::read_to_string(txt_path).unwrap_or_default().trim().to_string();
    let lang = Some(langid::detect(&text, "en"));
    Ok(Voice { wav: resample::resample(&wav, sr, tts::SR), text, lang })
}

/// Checks a fresh recording (24 kHz), levels it, transcribes it and saves it as `name`.
/// Returns the transcript.
pub fn save(name: &str, mut wav: Vec<f32>, asr: &mut Asr) -> Result<String, String> {
    if !valid_name(name) {
        return Err("use letters, digits, - or _ for the voice name".into());
    }
    let peak = wav.iter().fold(0f32, |m, s| m.max(s.abs()));
    if peak < 0.02 {
        return Err("that was nearly silent: check which microphone is selected and try again".into());
    }
    for s in &mut wav {
        *s *= 0.9 / peak;
    }
    let (_, text) = asr.transcribe(&resample::resample(&wav, tts::SR, 16_000))?;
    if text.split_whitespace().count() < 5 && text.chars().count() < 10 {
        return Err("couldn't make out enough speech: read the text out loud, close to the mic".into());
    }
    std::fs::create_dir_all(paths::voices()).map_err(|e| e.to_string())?;
    let (wav_path, txt_path) = files(name);
    wav::write(&wav_path, &wav, tts::SR)?;
    std::fs::write(txt_path, &text).map_err(|e| e.to_string())?;
    Ok(text)
}

pub fn delete(name: &str) -> Result<(), String> {
    let (wav_path, txt_path) = files(name);
    std::fs::remove_file(wav_path).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(txt_path);
    Ok(())
}
