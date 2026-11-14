//! The live translator. Your mic -> voice detector -> speech recognition -> translation -> your
//! cloned voice -> the virtual mic (and/or your headphones). Other people's audio -> recognition ->
//! translation -> subtitles. Lines you type are spoken like sentences you said.
//!
//! One sentence at a time, in order: each model runs one request at a time anyway.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::asr::Asr;
use crate::audio::{self, FRAME, MIC_SR, Output, Player, VirtualMic};
use crate::engines::Engines;
use crate::incoming::{self, Msg};
use crate::mt::Translator;
use crate::segmenter::Segmenter;
use crate::settings::Settings;
use crate::tts::{self, Tts};
use crate::{langid, langs, models, vad, voices};

#[derive(Clone, Debug, Serialize)]
pub struct Timing {
    /// From the end of your sentence (or Enter) to the first translated sound.
    pub total_ms: u32,
    /// Waiting for earlier sentences to finish.
    pub wait_ms: u32,
    /// None for typed lines.
    pub asr_ms: Option<u32>,
    pub mt_ms: u32,
    pub voice_ms: u32,
    /// Spoken from the voice fingerprint (native accent) rather than continuing the sample.
    pub native: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    Status { text: String },
    Download { what: String, done: u64, total: u64 },
    Ready { virtual_mic: Option<VirtualMic>, backend: String, gpu: Option<String> },
    Said { heard: String, said: String, from: Option<String>, to: String, typed: bool, timing: Timing },
    Subtitle { stream: u32, text: String, original: String, from: Option<String> },
    Notice { text: String },
    Error { text: String },
    Stopped,
}

enum Item {
    Out { t_end: Instant, audio: Vec<f32> },
    In { stream: u32, audio: Vec<f32> },
    Typed { t_end: Instant, text: String },
    Stop,
}

struct Shared {
    to: Mutex<&'static str>,
    muted: AtomicBool,
    keep_accent: AtomicBool,
    stop: Arc<AtomicBool>,
    /// When the translation playing in your headphones ends (the mic may hear it: don't re-translate).
    playing_until: Mutex<Instant>,
}

type OnEvent = Arc<dyn Fn(Event) + Send + Sync>;

/// A running translator. Dropping it stops it.
pub struct Session {
    tx: Sender<Item>,
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

impl Session {
    /// Starts loading in the background and returns at once; progress and results arrive as events.
    pub fn start(settings: Settings, on_event: impl Fn(Event) + Send + Sync + 'static) -> Session {
        let on_event: OnEvent = Arc::new(on_event);
        let (tx, rx) = channel();
        let shared = Arc::new(Shared {
            to: Mutex::new(langs::code_for(&settings.to).unwrap_or("en")),
            muted: AtomicBool::new(false),
            keep_accent: AtomicBool::new(settings.keep_accent),
            stop: Arc::new(AtomicBool::new(false)),
            playing_until: Mutex::new(Instant::now()),
        });
        let (s, t) = (shared.clone(), tx.clone());
        let worker = std::thread::Builder::new()
            .name("translator".into())
            .spawn(move || {
                if let Err(e) = run(settings, &s, t, rx, &on_event) {
                    on_event(Event::Error { text: e });
                }
                s.stop.store(true, Ordering::Relaxed);
                on_event(Event::Stopped);
            })
            .expect("can't start a thread");
        Session { tx, shared, worker: Some(worker) }
    }

    /// The language others hear you in, from now on.
    pub fn set_target(&self, code: &str) {
        if let Some(c) = langs::code_for(code) {
            *self.shared.to.lock().unwrap() = c;
        }
    }

    /// Stops listening to your mic (typing still works).
    pub fn set_muted(&self, muted: bool) {
        self.shared.muted.store(muted, Ordering::Relaxed);
    }

    pub fn set_keep_accent(&self, keep: bool) {
        self.shared.keep_accent.store(keep, Ordering::Relaxed);
    }

    /// Speaks a typed line in your voice, translated like a spoken one.
    pub fn say(&self, text: &str) {
        let text = text.trim();
        if !text.is_empty() {
            let _ = self.tx.send(Item::Typed { t_end: Instant::now(), text: text.into() });
        }
    }

    pub fn is_running(&self) -> bool {
        !self.shared.stop.load(Ordering::Relaxed)
    }

    /// Stops and waits until everything is unloaded (the GPU memory is free again).
    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        let _ = self.tx.send(Item::Stop);
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn ms(d: Duration) -> u32 {
    d.as_millis() as u32
}

/// The voice can't say emoji.
fn speakable(text: &str) -> String {
    let s: String = text
        .chars()
        .filter(|&c| {
            !matches!(c as u32, 0x1F000..=0x1FAFF | 0x2600..=0x27BF | 0xFE0F | 0x200D)
        })
        .collect();
    let s = s.trim();
    if s.is_empty() { text.to_string() } else { s.to_string() }
}

/// Downloads whatever model files are missing.
pub fn ensure_models(on_event: &dyn Fn(Event), stop: &AtomicBool) -> Result<(), String> {
    for m in models::missing() {
        if stop.load(Ordering::Relaxed) {
            return Err("stopped".into());
        }
        let what = m.what.to_string();
        models::download(m, |done, total| on_event(Event::Download { what: what.clone(), done, total }))?;
    }
    Ok(())
}

fn run(settings: Settings, shared: &Arc<Shared>, tx: Sender<Item>, rx: Receiver<Item>, on_event: &OnEvent) -> Result<(), String> {
    let status = |t: &str| on_event(Event::Status { text: t.into() });
    let me = langs::code_for(&settings.me).unwrap_or("en");
    let practice = settings.practice;
    let stop = shared.stop.clone();

    status("Checking the models");
    ensure_models(&|e| on_event(e), &stop)?;
    let voice = voices::load(&settings.voice)?;

    status("Starting the speech engines");
    let engines = Engines::load()?;
    let virtual_mic = if practice { None } else { Some(audio::virtual_mic()?) };
    let vad_model = Arc::new(vad::Model::load(&models::VAD.path())?);

    status("Loading speech recognition");
    let mut asr = Asr::load(&engines, settings.asr_gpu)?;
    status("Loading the translator");
    let mut mt = Translator::load(&engines, &settings.style)?;
    status("Loading your voice");
    let mut tts = Tts::load(&engines)?;
    let voice_text = if voice.text.is_empty() {
        asr.transcribe(&crate::resample::resample(&voice.wav, tts::SR, MIC_SR))?.1
    } else {
        voice.text.clone()
    };
    tts.set_voice(&voice.wav, &voice_text, voice.lang)?;
    tts.keep_accent = shared.keep_accent.load(Ordering::Relaxed);

    status("Warming up");
    // so the first real sentence isn't slow
    asr.transcribe(&vec![0.0; MIC_SR as usize])?;
    mt.translate("hello", Some("en"), "ru")?;
    tts.warmup()?;
    langid::preload();

    let player = match &virtual_mic {
        Some(VirtualMic::Ready { .. }) => Some(Player::open(&Output::VirtualMic)?),
        _ => None,
    };
    let headphones = settings.headphones.clone().map_or(Output::Default, Output::Device);
    let monitor = if settings.monitor || practice { Some(Player::open(&headphones)?) } else { None };
    let echo_guard = monitor.is_some();

    // your mic
    let mut mic = audio::open_mic(settings.mic.as_deref())?;
    {
        let (tx, shared, model, on_event) = (tx.clone(), shared.clone(), vad_model.clone(), on_event.clone());
        let silence = settings.silence_ms;
        std::thread::Builder::new()
            .name("mic".into())
            .spawn(move || {
                let mut seg = Segmenter::new(model, silence);
                let mut frame = [0f32; FRAME];
                while !shared.stop.load(Ordering::Relaxed) {
                    if let Err(e) = mic.read(&mut frame) {
                        on_event(Event::Error { text: e });
                        break;
                    }
                    let Some(utt) = seg.feed(&frame) else { continue };
                    // with your translation in speakers, the mic hears it: don't translate it again
                    let echo = echo_guard
                        && Instant::now() < *shared.playing_until.lock().unwrap() + Duration::from_millis(300);
                    if !shared.muted.load(Ordering::Relaxed) && !echo {
                        let _ = tx.send(Item::Out { t_end: Instant::now(), audio: utt });
                    }
                }
            })
            .map_err(|e| e.to_string())?;
    }

    // other people: each gets their own detector, so people talking over each other stay apart
    let mut _capture = None;
    if !practice {
        let segmenter = |silence: u32, model: Arc<vad::Model>, tx: Sender<Item>| {
            let mut segs: HashMap<u32, Segmenter> = HashMap::new();
            move |stream: u32, frame: &[f32; FRAME]| {
                let seg = segs.entry(stream).or_insert_with(|| Segmenter::new(model.clone(), silence));
                if let Some(audio) = seg.feed(frame) {
                    let _ = tx.send(Item::In { stream, audio });
                }
            }
        };
        if settings.subtitles_plugin {
            let mut feed = segmenter(settings.their_silence_ms, vad_model.clone(), tx.clone());
            let result = incoming::listen(stop.clone(), move |msg| match msg {
                Msg::Frame(n, f) => feed(n, f),
                Msg::Open(_) | Msg::Close(_) => {}
            });
            if let Err(e) = result {
                on_event(Event::Notice { text: e });
            }
        }
        if let Some(app) = settings.subtitles_from.as_deref().filter(|a| !a.is_empty()) {
            let mut feed = segmenter(settings.their_silence_ms, vad_model.clone(), tx.clone());
            let mut frame = [0f32; FRAME];
            let cap = audio::capture_app(
                app,
                Box::new(move |n, f| {
                    frame.copy_from_slice(&f[..FRAME]);
                    feed(n, &frame)
                }),
            );
            match cap {
                Ok(c) => _capture = Some(c),
                Err(e) => on_event(Event::Notice { text: format!("can't subtitle {app}: {e}") }),
            }
        }
    }
    drop(tx);

    let gpu = engines.gpus.first().map(|g| g.description.clone());
    on_event(Event::Ready { virtual_mic, backend: engines.backend.clone(), gpu });

    // one sentence; an error here is reported and the session carries on
    let mut handle = |item: Item| -> Result<(), String> {
        tts.keep_accent = shared.keep_accent.load(Ordering::Relaxed);
        let t_start = Instant::now();
        let (t_end, src, heard, typed) = match item {
            Item::Stop => return Ok(()),
            Item::In { stream, audio } => {
                let (src, text) = asr.transcribe(&audio)?;
                if text.is_empty() || src == Some(me) {
                    return Ok(()); // you understand it already
                }
                let sub = mt.translate(&text, src, me)?;
                on_event(Event::Subtitle { stream, text: sub, original: text, from: src.map(String::from) });
                return Ok(());
            }
            Item::Typed { t_end, text } => (t_end, Some(langid::detect(&text, me)), text, true),
            Item::Out { t_end, audio } => {
                if audio.len() < (MIC_SR as f32 * 0.3) as usize {
                    return Ok(());
                }
                let (src, text) = asr.transcribe(&audio)?;
                if text.is_empty() {
                    return Ok(());
                }
                (t_end, src, text, false)
            }
        };
        let t_asr = Instant::now();
        let to = *shared.to.lock().unwrap();
        let said = if src == Some(to) { heard.clone() } else { mt.translate(&heard, src, to)? };
        let t_mt = Instant::now();
        let mut first: Option<Instant> = None;
        let playing_until = &shared.playing_until;
        tts.speak(&speakable(&said), to, |chunk| {
            first.get_or_insert_with(Instant::now);
            if let Some(p) = &player {
                p.play(chunk);
            }
            if let Some(m) = &monitor {
                m.play(chunk);
                let mut until = playing_until.lock().unwrap();
                *until = (*until).max(Instant::now()) + Duration::from_secs_f32(chunk.len() as f32 / tts::SR as f32);
            }
            true
        })?;
        let t_voice = first.unwrap_or_else(Instant::now);
        on_event(Event::Said {
            heard,
            said,
            from: src.map(String::from),
            to: to.into(),
            typed,
            timing: Timing {
                total_ms: ms(t_voice.saturating_duration_since(t_end)),
                wait_ms: ms(t_start.saturating_duration_since(t_end)),
                asr_ms: (!typed).then(|| ms(t_asr - t_start)),
                mt_ms: ms(t_mt - t_asr),
                voice_ms: ms(t_voice.saturating_duration_since(t_mt)),
                native: tts.uses_fingerprint(to),
            },
        });
        Ok(())
    };
    while let Ok(item) = rx.recv() {
        if matches!(item, Item::Stop) || shared.stop.load(Ordering::Relaxed) {
            break;
        }
        if let Err(e) = handle(item) {
            on_event(Event::Error { text: e });
        }
    }
    drop(handle);
    // let what's queued finish playing
    for p in player.iter().chain(monitor.iter()) {
        let deadline = Instant::now() + Duration::from_secs(30);
        while p.pending() > 0.0 && Instant::now() < deadline && !shared.stop.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    Ok(())
}
