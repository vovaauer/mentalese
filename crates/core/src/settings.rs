//! Everything you can choose, saved between runs (settings.json in the data folder).

use serde::{Deserialize, Serialize};

use crate::{mt, paths};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Language others hear you in.
    pub to: String,
    /// Your language: what others' speech is subtitled in, and what typed text is assumed to be.
    pub me: String,
    /// Which recorded voice to speak with.
    pub voice: String,
    /// Speak other languages like the sample, accent and all.
    pub keep_accent: bool,
    /// Tone of translations, in plain words; empty for neutral.
    pub style: String,
    /// How long a pause ends your sentence (lower is faster but may cut you off).
    pub silence_ms: u32,
    /// The same for other people; longer, so subtitles get whole sentences.
    pub their_silence_ms: u32,
    /// Speech recognition fully on the GPU: ~40 ms faster, 0.75 GB more VRAM.
    pub asr_gpu: bool,
    /// Microphone device id; None for the system default.
    pub mic: Option<String>,
    /// Headphones device id for hearing yourself; None for the system default.
    pub headphones: Option<String>,
    /// Also play your translated voice in your headphones.
    pub monitor: bool,
    /// Hear yourself translated in your headphones only: no virtual mic, no subtitles.
    pub practice: bool,
    /// Subtitle each call participant separately via the Vesktop plugin.
    pub subtitles_plugin: bool,
    /// Subtitle an app's audio (e.g. "Discord", "Telegram"); best for 1:1 calls.
    pub subtitles_from: Option<String>,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            to: "ru".into(),
            me: "en".into(),
            voice: "me".into(),
            keep_accent: false,
            style: mt::DEFAULT_STYLE.into(),
            silence_ms: 400,
            their_silence_ms: 700,
            asr_gpu: false,
            mic: None,
            headphones: None,
            monitor: false,
            practice: false,
            subtitles_plugin: true,
            subtitles_from: None,
        }
    }
}

impl Settings {
    pub fn load() -> Settings {
        std::fs::read_to_string(paths::settings()).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
    }

    pub fn save(&self) -> Result<(), String> {
        let path = paths::settings();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(self).unwrap()).map_err(|e| e.to_string())?;
        std::fs::rename(tmp, path).map_err(|e| e.to_string())
    }
}
