//! Mentalese's window: the same translator as the command line, with buttons.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use mentalese_core::audio::{self, Output, Player, VirtualMic};
use mentalese_core::engines::{self, Engines};
use mentalese_core::pipeline::Session;
use mentalese_core::settings::Settings;
use mentalese_core::tts::{self, Tts};
use mentalese_core::{asr::Asr, langs, models, paths, voices};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};

#[derive(Default)]
struct App {
    session: Mutex<Option<Session>>,
    /// Recording, previewing or downloading: they use the mic or the GPU, one at a time.
    busy: Arc<AtomicBool>,
    stop_recording: Arc<AtomicBool>,
}

/// Holds `busy` for as long as it lives.
struct Busy(Arc<AtomicBool>);

impl Busy {
    fn take(flag: &Arc<AtomicBool>) -> Result<Busy, String> {
        flag.compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .map(|_| Busy(flag.clone()))
            .map_err(|_| "one thing at a time: wait for the current one to finish".to_string())
    }
}

impl Drop for Busy {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

impl App {
    fn running(&self) -> bool {
        self.session.lock().unwrap().as_ref().is_some_and(Session::is_running)
    }

    fn idle(&self) -> Result<(), String> {
        if self.running() { Err("stop translating first".into()) } else { Ok(()) }
    }
}

#[derive(Serialize)]
struct Overview {
    settings: Settings,
    voices: Vec<String>,
    inputs: Vec<audio::Device>,
    outputs: Vec<audio::Device>,
    apps: Vec<String>,
    missing_mb: u64,
    data_dir: String,
    langs: Vec<(&'static str, &'static str)>,
    running: bool,
    read_aloud: &'static str,
    record_secs: f32,
    os: &'static str,
}

#[tauri::command]
fn overview(app: State<App>) -> Overview {
    Overview {
        settings: Settings::load(),
        voices: voices::list(),
        inputs: audio::inputs(),
        outputs: audio::outputs(),
        apps: audio::apps_playing(),
        missing_mb: models::missing_bytes() >> 20,
        data_dir: paths::home().display().to_string(),
        langs: langs::LANGS.iter().map(|l| (l.code, l.name)).collect(),
        running: app.running(),
        read_aloud: voices::READ_ALOUD,
        record_secs: voices::RECORD_SECS,
        os: std::env::consts::OS,
    }
}

#[tauri::command]
fn save_settings(settings: Settings, app: State<App>) -> Result<(), String> {
    settings.save()?;
    if let Some(s) = app.session.lock().unwrap().as_ref() {
        s.set_target(&settings.to);
        s.set_keep_accent(settings.keep_accent);
    }
    Ok(())
}

#[derive(Clone, Serialize)]
struct Progress {
    what: &'static str,
    done: u64,
    total: u64,
}

#[tauri::command]
async fn download(handle: AppHandle, app: State<'_, App>) -> Result<(), String> {
    let _busy = Busy::take(&app.busy)?;
    tauri::async_runtime::spawn_blocking(move || {
        for m in models::missing() {
            models::download(m, |done, total| {
                let _ = handle.emit("download", Progress { what: m.what, done, total });
            })?;
        }
        Ok(())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
fn start(handle: AppHandle, app: State<App>) -> Result<(), String> {
    if app.busy.load(Ordering::Acquire) {
        return Err("wait for the recording or download to finish".into());
    }
    let mut slot = app.session.lock().unwrap();
    if slot.as_ref().is_some_and(Session::is_running) {
        return Ok(());
    }
    let settings = Settings::load();
    if !voices::exists(&settings.voice) {
        return Err("record your voice first".into());
    }
    *slot = None; // the previous, stopped session
    *slot = Some(Session::start(settings, move |e| {
        let _ = handle.emit("mx", &e);
    }));
    Ok(())
}

#[tauri::command]
async fn stop(app: State<'_, App>) -> Result<(), String> {
    let session = app.session.lock().unwrap().take();
    if let Some(s) = session {
        tauri::async_runtime::spawn_blocking(move || s.stop()).await.map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
fn set_muted(muted: bool, app: State<App>) {
    if let Some(s) = app.session.lock().unwrap().as_ref() {
        s.set_muted(muted);
    }
}

#[tauri::command]
fn say(text: String, app: State<App>) -> Result<(), String> {
    match app.session.lock().unwrap().as_ref().filter(|s| s.is_running()) {
        Some(s) => {
            s.say(&text);
            Ok(())
        }
        None => Err("press Start first".into()),
    }
}

#[derive(Clone, Serialize)]
struct Recording {
    secs: f32,
    phase: &'static str,
}

#[tauri::command]
async fn record_voice(name: String, handle: AppHandle, app: State<'_, App>) -> Result<String, String> {
    app.idle()?;
    if !voices::valid_name(&name) {
        return Err("use letters, digits, - or _ for the name".into());
    }
    if [&models::ASR, &models::ASR_MMPROJ].iter().any(|m| !m.present()) {
        return Err("download the models first".into());
    }
    let _busy = Busy::take(&app.busy)?;
    let stop = app.stop_recording.clone();
    stop.store(false, Ordering::Relaxed);
    let mic = Settings::load().mic;
    tauri::async_runtime::spawn_blocking(move || {
        let done = Arc::new(AtomicBool::new(false));
        let (h, d, t0) = (handle.clone(), done.clone(), Instant::now());
        std::thread::spawn(move || {
            while !d.load(Ordering::Relaxed) {
                let _ = h.emit("recording", Recording { secs: t0.elapsed().as_secs_f32(), phase: "recording" });
                std::thread::sleep(Duration::from_millis(100));
            }
        });
        let wav = audio::record(mic.as_deref(), voices::RECORD_SECS, tts::SR, &stop);
        done.store(true, Ordering::Relaxed);
        let wav = wav?;
        if stop.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        let _ = handle.emit("recording", Recording { secs: voices::RECORD_SECS, phase: "transcribing" });
        let engines = Engines::load()?;
        let mut asr = Asr::load(&engines, false)?;
        voices::save(&name, wav, &mut asr)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
fn cancel_recording(app: State<App>) {
    app.stop_recording.store(true, Ordering::Relaxed);
}

/// A line in each language to hear what a voice sounds like.
fn sample(lang: &str) -> &'static str {
    match lang {
        "ru" => "Привет! Вот как я звучу по-русски.",
        "zh" => "你好！这就是我说中文的声音。",
        "ja" => "こんにちは！日本語だとこんな声です。",
        "ko" => "안녕하세요! 한국어로는 이렇게 들려요.",
        "de" => "Hallo! So klinge ich auf Deutsch.",
        "fr" => "Salut ! Voilà ma voix en français.",
        "es" => "¡Hola! Así sueno en español.",
        "pt" => "Olá! É assim que eu soo em português.",
        "it" => "Ciao! Ecco come suono in italiano.",
        _ => "Hi! This is what I sound like in English.",
    }
}

#[tauri::command]
async fn preview_voice(name: String, lang: String, app: State<'_, App>) -> Result<(), String> {
    app.idle()?;
    if models::missing().iter().any(|m| m.file == models::TALKER.file || m.file == models::CODEC.file) {
        return Err("download the models first".into());
    }
    let _busy = Busy::take(&app.busy)?;
    tauri::async_runtime::spawn_blocking(move || {
        let settings = Settings::load();
        let lang = langs::code_for(&lang).unwrap_or("en");
        let voice = voices::load(&name)?;
        let engines = Engines::load()?;
        let mut tts = Tts::load(&engines)?;
        tts.set_voice(&voice.wav, &voice.text, voice.lang)?;
        tts.keep_accent = settings.keep_accent;
        let player = Player::open(&settings.headphones.map_or(Output::Default, Output::Device))?;
        tts.speak(sample(lang), lang, |c| {
            player.play(c);
            true
        })?;
        while player.pending() > 0.0 {
            std::thread::sleep(Duration::from_millis(50));
        }
        std::thread::sleep(Duration::from_millis(150));
        Ok(())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
fn delete_voice(name: String, app: State<App>) -> Result<(), String> {
    app.idle()?;
    voices::delete(&name)
}

#[tauri::command]
fn virtual_mic() -> Result<VirtualMic, String> {
    audio::virtual_mic()
}

#[tauri::command]
fn apps_playing() -> Vec<String> {
    audio::apps_playing()
}

/// Opens one of the few pages the app links to in the browser.
#[tauri::command]
fn open_url(url: String) -> Result<(), String> {
    const ALLOWED: [&str; 3] =
        ["https://vb-audio.com/", "https://existential.audio/", "https://github.com/vovaauer/mentalese"];
    if !ALLOWED.iter().any(|a| url.starts_with(a)) {
        return Err("not a link mentalese opens".into());
    }
    let mut cmd = if cfg!(windows) {
        let mut c = std::process::Command::new("rundll32");
        c.arg("url.dll,FileProtocolHandler");
        c
    } else if cfg!(target_os = "macos") {
        std::process::Command::new("open")
    } else {
        std::process::Command::new("xdg-open")
    };
    cmd.arg(&url).spawn().map(|_| ()).map_err(|e| e.to_string())
}

#[derive(Serialize)]
struct UpdateInfo {
    version: String,
    notes: Option<String>,
}

/// A newer release, if there is one (signed with the key in tauri.conf.json).
#[tauri::command]
async fn check_update(handle: AppHandle) -> Result<Option<UpdateInfo>, String> {
    use tauri_plugin_updater::UpdaterExt;
    let update = handle.updater().map_err(|e| e.to_string())?.check().await.map_err(|e| e.to_string())?;
    Ok(update.map(|u| UpdateInfo { version: u.version, notes: u.body }))
}

/// Downloads and installs the newer release, then restarts into it.
#[tauri::command]
async fn install_update(handle: AppHandle, app: State<'_, App>) -> Result<(), String> {
    use tauri_plugin_updater::UpdaterExt;
    let Some(update) =
        handle.updater().map_err(|e| e.to_string())?.check().await.map_err(|e| e.to_string())?
    else {
        return Ok(());
    };
    let session = app.session.lock().unwrap().take();
    if let Some(s) = session {
        tauri::async_runtime::spawn_blocking(move || s.stop()).await.map_err(|e| e.to_string())?;
    }
    update.download_and_install(|_, _| {}, || {}).await.map_err(|e| e.to_string())?;
    handle.restart();
}

fn main() {
    #[cfg(target_os = "linux")]
    if std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
        // WebKitGTK's DMA-BUF renderer shows a blank window on many NVIDIA setups; nothing else
        // has started yet, so changing the environment is safe here
        unsafe { std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1") };
    }
    tauri::Builder::default()
        .manage(App::default())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(|app| {
            if let Ok(dir) = app.path().resource_dir() {
                engines::set_root(dir.join("engines"));
            }
            // mentalese-app --check: which engines and GPU it would use, for troubleshooting
            if std::env::args().any(|a| a == "--check") {
                match Engines::load() {
                    Ok(e) => println!("engines: {} ({}); GPUs: {:?}; data: {}", e.backend, e.path.display(), e.gpus, paths::home().display()),
                    Err(e) => println!("{e}"),
                }
                std::process::exit(0);
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            overview,
            save_settings,
            download,
            start,
            stop,
            set_muted,
            say,
            record_voice,
            cancel_recording,
            preview_voice,
            delete_voice,
            virtual_mic,
            apps_playing,
            open_url,
            check_update,
            install_update
        ])
        .build(tauri::generate_context!())
        .expect("can't start the window")
        .run(|handle, event| {
            if let tauri::RunEvent::Exit = event {
                // unload the models before the process goes
                let session = handle.state::<App>().session.lock().unwrap().take();
                drop(session);
            }
        });
}
