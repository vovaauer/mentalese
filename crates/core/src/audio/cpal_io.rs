//! Windows and macOS: sound in and out through cpal (WASAPI / Core Audio). The virtual mic is a
//! driver the user installs once (VB-CABLE / BlackHole); hearing other apps is per-OS, in
//! windows_capture.rs and macos_capture.rs.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, FromSample, Sample, SampleFormat, SizedSample, StreamConfig, SupportedBufferSize};

#[cfg(target_os = "macos")]
use super::macos_capture as capture;
#[cfg(windows)]
use super::windows_capture as capture;
use super::{Device, FRAME, MIC_SR, Mic, Output, PlayerImpl, SPEECH_SR, VirtualMic};
use crate::resample::Resampler;

/// App-capture stream numbers start here so they never collide with the Vesktop plugin's.
pub(super) const CAPTURE_BASE: u32 = 100_000;

/// The virtual cable: speech is played into `play`, other apps record from `mic`.
struct Cable {
    play: &'static str,
    mic: &'static str,
    /// devices kept out of the mic and speaker lists
    hide: &'static [&'static str],
    how: &'static str,
    url: &'static str,
}

#[cfg(windows)]
const CABLE: Cable = Cable {
    play: "CABLE Input",
    mic: "CABLE Output",
    hide: &["CABLE Input", "CABLE Output", "VB-Audio Virtual Cable"],
    how: "Install VB-CABLE, a free virtual audio cable: unzip it, run VBCABLE_Setup_x64 as administrator, restart your computer, then try again.",
    url: "https://vb-audio.com/Cable/",
};

#[cfg(target_os = "macos")]
const CABLE: Cable = Cable {
    play: "BlackHole",
    mic: "BlackHole",
    hide: &["BlackHole"],
    how: "Install BlackHole 2ch, a free virtual audio driver, then try again.",
    url: "https://existential.audio/blackhole/",
};

fn name(d: &cpal::Device) -> Option<String> {
    d.description().ok().map(|d| d.name().to_owned())
}

/// The cable, and mentalese's own devices (macOS app capture's private aggregate device).
fn hidden(name: &str) -> bool {
    name.starts_with("mentalese") || CABLE.hide.iter().any(|h| name.contains(h))
}

fn devices(input: bool) -> impl Iterator<Item = cpal::Device> {
    let host = cpal::default_host();
    let devices = if input { host.input_devices() } else { host.output_devices() };
    devices.into_iter().flatten()
}

/// Ids are cpal's ("wasapi:{endpoint id}", "coreaudio:{device UID}"): stable across restarts.
fn listed(input: bool) -> Vec<Device> {
    devices(input)
        .filter_map(|d| {
            let name = name(&d).filter(|n| !hidden(n))?;
            let id = d.id().map_or_else(|_| name.clone(), |id| id.to_string());
            Some(Device { id, name })
        })
        .collect()
}

pub fn inputs() -> Vec<Device> {
    listed(true)
}

pub fn outputs() -> Vec<Device> {
    listed(false)
}

/// A device by id (or, failing that, by name).
fn find(id: &str, input: bool) -> Result<cpal::Device, String> {
    let host = cpal::default_host();
    if let Some(d) = id.parse().ok().and_then(|id| host.device_by_id(&id)) {
        return Ok(d);
    }
    devices(input).find(|d| name(d).as_deref() == Some(id)).ok_or_else(|| format!("no audio device {id:?}"))
}

/// The cable's input or output end, if it's installed.
fn cable(input: bool) -> Option<(cpal::Device, String)> {
    let part = if input { CABLE.mic } else { CABLE.play };
    devices(input).find_map(|d| name(&d).filter(|n| n.contains(part)).map(|n| (d, n)))
}

/// The device's own format, so nothing is reconfigured for other apps. Buffers: WASAPI's default
/// is already its shared-mode minimum (asking for less risks underruns); Core Audio gets ~10 ms
/// where the device allows it.
fn config(sc: &cpal::SupportedStreamConfig) -> StreamConfig {
    let buffer_size = match *sc.buffer_size() {
        SupportedBufferSize::Range { min, max } if cfg!(target_os = "macos") => {
            BufferSize::Fixed((sc.sample_rate() / 100).max(min).min(max))
        }
        _ => BufferSize::Default,
    };
    StreamConfig { channels: sc.channels(), sample_rate: sc.sample_rate(), buffer_size }
}

/// Whether a stream error means the stream is gone, rather than a glitch it recovers from.
/// (WASAPI reports a default-device change as StreamInvalidated: the stream doesn't follow it.)
fn fatal(e: &cpal::Error) -> bool {
    !matches!(e.kind(), cpal::ErrorKind::Xrun | cpal::ErrorKind::DeviceChanged | cpal::ErrorKind::RealtimeDenied)
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Mono samples from an audio callback to the thread that reads them.
pub(super) struct Feed {
    state: Mutex<FeedState>,
    ready: Condvar,
}

struct FeedState {
    samples: VecDeque<f32>,
    cap: usize,
    /// why the source stopped
    end: Option<String>,
}

impl Feed {
    pub(super) fn new(cap: usize) -> Arc<Feed> {
        let state = FeedState { samples: VecDeque::with_capacity(cap), cap, end: None };
        Arc::new(Feed { state: Mutex::new(state), ready: Condvar::new() })
    }

    /// From the audio callback: a copy under the lock, never an allocation.
    pub(super) fn push(&self, mono: impl Iterator<Item = f32>) {
        let mut s = lock(&self.state);
        for x in mono {
            if s.samples.len() == s.cap {
                s.samples.pop_front(); // the reader stalled: keep the newest
            }
            s.samples.push_back(x);
        }
        drop(s);
        self.ready.notify_one();
    }

    pub(super) fn end(&self, why: String) {
        lock(&self.state).end.get_or_insert(why);
        self.ready.notify_one();
    }

    /// Blocks until there's audio and moves all of it into `out`; Err once the source ended.
    pub(super) fn take(&self, out: &mut Vec<f32>) -> Result<(), String> {
        let mut s = lock(&self.state);
        while s.samples.is_empty() {
            if let Some(why) = &s.end {
                return Err(why.clone());
            }
            s = self.ready.wait(s).unwrap_or_else(PoisonError::into_inner);
        }
        out.clear();
        out.extend(s.samples.drain(..));
        Ok(())
    }
}

/// An input stream mixed to mono in the callback and resampled on the reading side.
struct Input {
    _stream: cpal::Stream,
    feed: Arc<Feed>,
    rs: Resampler,
}

fn open_input(device: Option<&str>, rate: u32) -> Result<Input, String> {
    let dev = match device {
        Some(id) => find(id, true)?,
        None => cpal::default_host().default_input_device().ok_or("no microphone found")?,
    };
    let sc = dev.default_input_config().map_err(|e| format!("can't use the microphone: {e}"))?;
    let feed = Feed::new(sc.sample_rate() as usize * 2);
    let cfg = config(&sc);
    let stream = match sc.sample_format() {
        SampleFormat::F32 => input_stream::<f32>(&dev, cfg, feed.clone()),
        SampleFormat::I16 => input_stream::<i16>(&dev, cfg, feed.clone()),
        SampleFormat::I32 => input_stream::<i32>(&dev, cfg, feed.clone()),
        f => return Err(format!("the microphone's sample format ({f:?}) isn't supported")),
    }
    .and_then(|s| s.play().map(|()| s))
    .map_err(|e| format!("can't open the microphone: {e}"))?;
    Ok(Input { _stream: stream, feed, rs: Resampler::new(sc.sample_rate(), rate) })
}

fn input_stream<T>(dev: &cpal::Device, cfg: StreamConfig, feed: Arc<Feed>) -> Result<cpal::Stream, cpal::Error>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    // only the first two channels: an audio interface's unused inputs would dilute the mic
    let (ch, used) = (cfg.channels as usize, (cfg.channels as usize).min(2));
    let ended = feed.clone();
    dev.build_input_stream::<T, _, _>(
        cfg,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            feed.push(data.chunks_exact(ch).map(|f| f[..used].iter().map(|&s| f32::from_sample(s)).sum::<f32>() / used as f32))
        },
        move |e| {
            if fatal(&e) {
                ended.end(format!("the microphone stopped: {e}"))
            }
        },
        None,
    )
}

struct CpalMic {
    device: Option<String>,
    rate: u32,
    input: Input,
    raw: Vec<f32>,
    out: Vec<f32>,
    reopened: Option<Instant>,
}

impl CpalMic {
    fn open(device: Option<&str>, rate: u32) -> Result<CpalMic, String> {
        let input = open_input(device, rate)?;
        Ok(CpalMic { device: device.map(String::from), rate, input, raw: Vec::new(), out: Vec::new(), reopened: None })
    }

    fn read_into(&mut self, frame: &mut [f32]) -> Result<(), String> {
        while self.out.len() < frame.len() {
            if let Err(e) = self.input.feed.take(&mut self.raw) {
                // unplugged, or (Windows) the default mic changed: reopen, but not over and over
                if self.reopened.is_some_and(|t| t.elapsed() < Duration::from_secs(2)) {
                    return Err(e);
                }
                self.reopened = Some(Instant::now());
                self.input = open_input(self.device.as_deref(), self.rate).map_err(|_| e)?;
                continue;
            }
            self.input.rs.process(&self.raw, &mut self.out);
        }
        frame.copy_from_slice(&self.out[..frame.len()]);
        self.out.drain(..frame.len());
        Ok(())
    }
}

impl Mic for CpalMic {
    fn read(&mut self, frame: &mut [f32; FRAME]) -> Result<(), String> {
        self.read_into(frame)
    }
}

pub fn open_mic(device: Option<&str>) -> Result<Box<dyn Mic>, String> {
    Ok(Box::new(CpalMic::open(device, MIC_SR)?))
}

pub fn record(device: Option<&str>, secs: f32, rate: u32, stop: &AtomicBool) -> Result<Vec<f32>, String> {
    let mut mic = CpalMic::open(device, rate)?;
    let mut frame = [0f32; FRAME];
    let mut wav = Vec::new();
    while wav.len() < (secs * rate as f32) as usize && !stop.load(Ordering::Relaxed) {
        mic.read_into(&mut frame)?;
        wav.extend_from_slice(&frame);
    }
    Ok(wav)
}

/// Speech waiting to be played, at SPEECH_SR.
struct Queue {
    pending: Mutex<Pending>,
    /// resampled by the callback but not yet played, counted at SPEECH_SR
    resampled: AtomicUsize,
    /// the stream died (unplugged, or the Windows default device changed): reopen on next play()
    broken: AtomicBool,
}

struct Pending {
    samples: VecDeque<f32>,
    /// clear() was called: the callback drops what it has resampled
    reset: bool,
}

/// Fed through the resampler after clear(), to flush the old speech out of the filter's history
/// (enough for any device rate down to 8 kHz).
static SILENCE: [f32; 128] = [0.0; 128];

fn open_output(out: &Output, q: &Arc<Queue>) -> Result<cpal::Stream, String> {
    let dev = match out {
        Output::Default => cpal::default_host().default_output_device().ok_or("no speakers or headphones found")?,
        Output::Device(id) => find(id, false)?,
        Output::VirtualMic => cable(false).map(|(d, _)| d).ok_or("the virtual mic isn't installed")?,
    };
    let sc = dev.default_output_config().map_err(|e| format!("can't play audio: {e}"))?;
    let cfg = config(&sc);
    match sc.sample_format() {
        SampleFormat::F32 => output_stream::<f32>(&dev, cfg, q.clone()),
        SampleFormat::I16 => output_stream::<i16>(&dev, cfg, q.clone()),
        SampleFormat::I32 => output_stream::<i32>(&dev, cfg, q.clone()),
        f => return Err(format!("the output's sample format ({f:?}) isn't supported")),
    }
    .and_then(|s| s.play().map(|()| s))
    .map_err(|e| format!("can't play audio: {e}"))
}

fn output_stream<T>(dev: &cpal::Device, cfg: StreamConfig, q: Arc<Queue>) -> Result<cpal::Stream, cpal::Error>
where
    T: SizedSample + FromSample<f32>,
{
    let (ch, rate) = (cfg.channels as usize, cfg.sample_rate as usize);
    let mut rs = Resampler::new(SPEECH_SR, cfg.sample_rate);
    // allocated here so the callback never does
    let mut ready: Vec<f32> = Vec::with_capacity(rate / 5);
    let mut take: Vec<f32> = Vec::with_capacity(SPEECH_SR as usize / 5);
    let broken = q.clone();
    dev.build_output_stream::<T, _, _>(
        cfg,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            let frames = data.len() / ch;
            loop {
                // the lock is held only to copy out what this callback needs
                let reset = {
                    let mut p = lock(&q.pending);
                    let want = (frames.saturating_sub(ready.len()) * SPEECH_SR as usize).div_ceil(rate);
                    let n = want.min(p.samples.len());
                    take.clear();
                    take.extend(p.samples.drain(..n));
                    std::mem::take(&mut p.reset)
                };
                if reset {
                    rs.process(&SILENCE, &mut ready);
                    ready.clear();
                }
                if take.is_empty() {
                    break;
                }
                rs.process(&take, &mut ready);
                if ready.len() >= frames {
                    break;
                }
            }
            data.fill(T::EQUILIBRIUM);
            let n = frames.min(ready.len());
            for (f, &s) in data.chunks_exact_mut(ch).zip(&ready[..n]) {
                // front left and right; any other channels stay silent
                let s = T::from_sample(s);
                f[0] = s;
                if ch > 1 {
                    f[1] = s;
                }
            }
            ready.drain(..n);
            q.resampled.store(ready.len() * SPEECH_SR as usize / rate, Ordering::Relaxed);
        },
        move |e| {
            if fatal(&e) {
                broken.broken.store(true, Ordering::Relaxed)
            }
        },
        None,
    )
}

struct CpalPlayer {
    out: Output,
    q: Arc<Queue>,
    /// the stream, and when it was last (re)opened
    stream: Mutex<(cpal::Stream, Instant)>,
}

pub(super) fn open_player(out: &Output) -> Result<Box<dyn PlayerImpl>, String> {
    let q = Arc::new(Queue {
        pending: Mutex::new(Pending { samples: VecDeque::new(), reset: false }),
        resampled: AtomicUsize::new(0),
        broken: AtomicBool::new(false),
    });
    let stream = open_output(out, &q)?;
    Ok(Box::new(CpalPlayer { out: out.clone(), q, stream: Mutex::new((stream, Instant::now())) }))
}

impl CpalPlayer {
    /// Blocks the caller for as long as opening a stream takes, but only after the old one died,
    /// and at most once a second.
    fn reopen(&self) {
        let mut s = lock(&self.stream);
        if s.1.elapsed() < Duration::from_secs(1) {
            return;
        }
        s.1 = Instant::now();
        self.q.broken.store(false, Ordering::Relaxed);
        match open_output(&self.out, &self.q) {
            Ok(stream) => s.0 = stream,
            Err(_) => self.q.broken.store(true, Ordering::Relaxed),
        }
    }
}

impl PlayerImpl for CpalPlayer {
    fn play(&self, chunk: &[f32]) {
        lock(&self.q.pending).samples.extend(chunk);
        if self.q.broken.load(Ordering::Relaxed) {
            self.reopen();
        }
    }

    fn pending(&self) -> f32 {
        let queued = lock(&self.q.pending).samples.len();
        (queued + self.q.resampled.load(Ordering::Relaxed)) as f32 / SPEECH_SR as f32
    }

    fn clear(&self) {
        let mut p = lock(&self.q.pending);
        p.samples.clear();
        p.reset = true;
        self.q.resampled.store(0, Ordering::Relaxed);
    }
}

pub fn virtual_mic() -> Result<VirtualMic, String> {
    Ok(match (cable(true), cable(false)) {
        (Some((_, name)), Some(_)) => VirtualMic::Ready { name },
        _ => VirtualMic::Missing { how: CABLE.how.into(), url: CABLE.url.into() },
    })
}

/// Nothing to remove: the user installed the cable, mentalese didn't create it.
pub fn remove_virtual_mic() -> Result<(), String> {
    Ok(())
}

pub fn apps_playing() -> Vec<String> {
    capture::apps_playing()
}

pub fn capture_app(app: &str, on_frame: super::OnFrame) -> Result<Box<dyn Send>, String> {
    capture::capture_app(app, on_frame)
}

pub(super) type OnFrame = Arc<Mutex<Box<dyn FnMut(u32, &[f32]) + Send>>>;

/// Resamples one captured stream's mono audio to MIC_SR and hands it on in FRAME-sample frames.
pub(super) struct Framer {
    rs: Resampler,
    buf: Vec<f32>,
    stream: u32,
    on_frame: OnFrame,
}

impl Framer {
    pub(super) fn new(rate: u32, stream: u32, on_frame: OnFrame) -> Framer {
        Framer { rs: Resampler::new(rate, MIC_SR), buf: Vec::new(), stream, on_frame }
    }

    pub(super) fn push(&mut self, mono: &[f32]) {
        self.rs.process(mono, &mut self.buf);
        let whole = self.buf.len() / FRAME * FRAME;
        if whole == 0 {
            return;
        }
        if let Ok(mut f) = self.on_frame.lock() {
            for frame in self.buf[..whole].as_chunks::<FRAME>().0 {
                f(self.stream, frame);
            }
        }
        self.buf.drain(..whole);
    }
}

/// Runs `poll` now and every 2 s on a thread of its own until dropped. Whatever `poll` owns is
/// dropped (and so stopped) on that thread before the drop returns.
pub(super) struct Watcher {
    stop: Option<mpsc::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl Watcher {
    pub(super) fn spawn(mut poll: impl FnMut() + Send + 'static) -> Result<Watcher, String> {
        let (stop, stopped) = mpsc::channel::<()>();
        let thread = std::thread::Builder::new()
            .name("app-capture".into())
            .spawn(move || {
                loop {
                    poll();
                    if !matches!(stopped.recv_timeout(Duration::from_secs(2)), Err(RecvTimeoutError::Timeout)) {
                        break;
                    }
                }
            })
            .map_err(|e| e.to_string())?;
        Ok(Watcher { stop: Some(stop), thread: Some(thread) })
    }
}

impl Drop for Watcher {
    fn drop(&mut self) {
        self.stop.take(); // disconnects: the watcher wakes and ends
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}
