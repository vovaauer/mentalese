//! Sound in and out, the same interface on every OS:
//! - the mic, as 16 kHz mono frames of 512 samples (what the voice detector wants),
//! - players for 24 kHz mono speech: your headphones, or the virtual mic other apps hear,
//! - the virtual mic itself (created on Linux; VB-CABLE on Windows, BlackHole on macOS),
//! - other apps' audio, for subtitles.

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
use linux as sys;

#[cfg(any(windows, target_os = "macos"))]
mod cpal_io;
#[cfg(any(windows, target_os = "macos"))]
use cpal_io as sys;

#[cfg(windows)]
mod windows_capture;
#[cfg(target_os = "macos")]
mod macos_capture;

pub const MIC_SR: u32 = 16_000;
pub const FRAME: usize = 512;
pub const SPEECH_SR: u32 = 24_000;

#[derive(Clone, Debug, serde::Serialize)]
pub struct Device {
    /// What to pass back to open a device.
    pub id: String,
    /// What to show.
    pub name: String,
}

/// Microphones (and other inputs), not including mentalese's own.
pub fn inputs() -> Vec<Device> {
    sys::inputs()
}

/// Speakers and headphones, not including the virtual mic's.
pub fn outputs() -> Vec<Device> {
    sys::outputs()
}

pub trait Mic: Send {
    /// Blocks until the next frame of FRAME samples at MIC_SR.
    fn read(&mut self, frame: &mut [f32; FRAME]) -> Result<(), String>;
}

/// `device`: an id from inputs(); None for the system default.
pub fn open_mic(device: Option<&str>) -> Result<Box<dyn Mic>, String> {
    sys::open_mic(device)
}

/// Records `secs` seconds from `device` (None: the default mic) at `rate`, at full quality (a
/// voice sample wants 24 kHz, not the 16 kHz the recognizer uses). `stop` ends it early.
pub fn record(device: Option<&str>, secs: f32, rate: u32, stop: &std::sync::atomic::AtomicBool) -> Result<Vec<f32>, String> {
    sys::record(device, secs, rate, stop)
}

#[derive(Clone, Debug)]
pub enum Output {
    /// The system's default speakers/headphones.
    Default,
    /// An id from outputs().
    Device(String),
    /// The virtual mic other apps (Discord...) record from.
    VirtualMic,
}

/// What each OS's player implements.
pub(crate) trait PlayerImpl: Send + Sync {
    fn play(&self, chunk: &[f32]);
    fn pending(&self) -> f32;
    fn clear(&self);
}

/// Plays 24 kHz mono speech without blocking the caller.
pub struct Player(Box<dyn PlayerImpl>);

impl Player {
    pub fn open(out: &Output) -> Result<Player, String> {
        sys::open_player(out).map(Player)
    }

    /// Queues audio; returns at once.
    pub fn play(&self, chunk: &[f32]) {
        self.0.play(chunk)
    }

    /// Seconds of audio queued but not yet heard.
    pub fn pending(&self) -> f32 {
        self.0.pending()
    }

    /// Drops everything queued (e.g. on stop).
    pub fn clear(&self) {
        self.0.clear()
    }
}

#[derive(Clone, Debug, serde::Serialize)]
#[serde(tag = "state")]
pub enum VirtualMic {
    /// Ready; `name` is what to pick as the microphone in Discord.
    Ready { name: String },
    /// Needs a one-time install; `how` says what to do, `url` where.
    Missing { how: String, url: String },
}

/// Makes sure the virtual mic exists (Linux creates it; elsewhere it's checked for).
pub fn virtual_mic() -> Result<VirtualMic, String> {
    sys::virtual_mic()
}

/// Removes the virtual mic, where mentalese created it (Linux).
pub fn remove_virtual_mic() -> Result<(), String> {
    sys::remove_virtual_mic()
}

/// Apps playing audio right now (for "subtitles from" pickers).
pub fn apps_playing() -> Vec<String> {
    sys::apps_playing()
}

/// Sends `on_frame(stream, frame)` 16 kHz frames of FRAME samples of every audio stream `app`
/// plays (name contains `app`, case-insensitive), picking up new streams as they start. One
/// stream per app instance: people talking at once in a group call get mixed together.
/// Stops when the returned handle is dropped.
pub fn capture_app(app: &str, on_frame: OnFrame) -> Result<Capture, String> {
    sys::capture_app(app, on_frame).map(Capture)
}

/// Receives (stream, frame) from capture_app.
pub type OnFrame = Box<dyn FnMut(u32, &[f32]) + Send>;

pub struct Capture(#[allow(dead_code)] Box<dyn Send>);
