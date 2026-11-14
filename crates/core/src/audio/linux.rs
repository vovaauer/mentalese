//! Linux: PipeWire (or PulseAudio) through its Pulse tools: pactl, parec, pacat. Every desktop
//! distribution has them, and they handle device routing exactly like the system does.

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use super::{Device, FRAME, MIC_SR, Mic, Output, PlayerImpl, SPEECH_SR, VirtualMic};

const SINK: &str = "mentalese_sink"; // speech is played into this
const MIC: &str = "mentalese_mic"; // apps record from this (a remap of the sink's monitor)
/// App-capture stream numbers start here so they never collide with the Vesktop plugin's.
const CAPTURE_BASE: u32 = 100_000;

fn pactl(args: &[&str]) -> Result<String, String> {
    let out = Command::new("pactl")
        .args(args)
        .output()
        .map_err(|e| format!("can't run pactl (install pipewire-pulse or pulseaudio-utils): {e}"))?;
    if !out.status.success() {
        return Err(format!("pactl {}: {}", args.join(" "), String::from_utf8_lossy(&out.stderr).trim()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn list(kind: &str) -> Vec<serde_json::Value> {
    pactl(&["-f", "json", "list", kind]).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
}

fn devices(kind: &str, keep: impl Fn(&str) -> bool) -> Vec<Device> {
    list(kind)
        .iter()
        .filter_map(|d| {
            let id = d["name"].as_str()?;
            keep(id).then(|| Device { id: id.into(), name: d["description"].as_str().unwrap_or(id).into() })
        })
        .collect()
}

pub fn inputs() -> Vec<Device> {
    devices("sources", |id| !id.ends_with(".monitor") && !id.starts_with("mentalese"))
}

pub fn outputs() -> Vec<Device> {
    devices("sinks", |id| !id.starts_with("mentalese"))
}

fn parec(args: &[String]) -> Result<(Child, ChildStdout), String> {
    let mut child = Command::new("parec")
        .args(["--channels=1", "--format=float32le", "--latency-msec=20"])
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("can't run parec (install pipewire-pulse or pulseaudio-utils): {e}"))?;
    let out = child.stdout.take().unwrap();
    Ok((child, out))
}

fn read_frame(out: &mut ChildStdout, bytes: &mut [u8], frame: &mut [f32]) -> std::io::Result<()> {
    out.read_exact(bytes)?;
    for (s, b) in frame.iter_mut().zip(bytes.as_chunks::<4>().0) {
        *s = f32::from_le_bytes(*b);
    }
    Ok(())
}

struct Parec {
    child: Child,
    out: ChildStdout,
    bytes: Vec<u8>,
}

impl Mic for Parec {
    fn read(&mut self, frame: &mut [f32; FRAME]) -> Result<(), String> {
        read_frame(&mut self.out, &mut self.bytes, frame).map_err(|_| "the microphone stream ended".to_string())
    }
}

impl Drop for Parec {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub fn open_mic(device: Option<&str>) -> Result<Box<dyn Mic>, String> {
    let mut args = vec![format!("--rate={MIC_SR}"), "--client-name=mentalese".into()];
    args.extend(device.map(|d| format!("--device={d}")));
    let (child, out) = parec(&args)?;
    Ok(Box::new(Parec { child, out, bytes: vec![0; FRAME * 4] }))
}

pub fn record(device: Option<&str>, secs: f32, rate: u32, stop: &AtomicBool) -> Result<Vec<f32>, String> {
    let mut args = vec![format!("--rate={rate}"), "--client-name=mentalese".into()];
    args.extend(device.map(|d| format!("--device={d}")));
    let (child, out) = parec(&args)?;
    let mut p = Parec { child, out, bytes: vec![0; FRAME * 4] };
    let mut frame = [0f32; FRAME];
    let mut wav = Vec::new();
    while wav.len() < (secs * rate as f32) as usize && !stop.load(Ordering::Relaxed) {
        p.read(&mut frame)?;
        wav.extend_from_slice(&frame);
    }
    Ok(wav)
}

struct Queue {
    chunks: Mutex<VecDeque<Vec<f32>>>,
    ready: Condvar,
    /// samples queued or being written
    queued: AtomicU64,
    closed: AtomicBool,
}

/// pacat drains its stdin in real time, so writing to it directly would block the caller for as
/// long as the audio lasts; a writer thread feeds it instead.
struct Pacat {
    q: Arc<Queue>,
}

fn pacat(device: Option<&str>) -> Result<Child, String> {
    Command::new("pacat")
        .args(["--playback", &format!("--rate={SPEECH_SR}"), "--channels=1", "--format=float32le"])
        .args(["--latency-msec=40", "--client-name=mentalese"])
        .args(device.map(|d| format!("--device={d}")))
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("can't run pacat (install pipewire-pulse or pulseaudio-utils): {e}"))
}

pub(super) fn open_player(out: &Output) -> Result<Box<dyn PlayerImpl>, String> {
    let device = match out {
        Output::Default => None,
        Output::Device(d) => Some(d.clone()),
        Output::VirtualMic => Some(SINK.to_string()),
    };
    let mut child = pacat(device.as_deref())?;
    let q = Arc::new(Queue {
        chunks: Mutex::new(VecDeque::new()),
        ready: Condvar::new(),
        queued: AtomicU64::new(0),
        closed: AtomicBool::new(false),
    });
    let wq = q.clone();
    std::thread::Builder::new()
        .name("player".into())
        .spawn(move || {
            let mut bytes = Vec::new();
            loop {
                let chunk = {
                    let mut c = wq.chunks.lock().unwrap();
                    loop {
                        if let Some(chunk) = c.pop_front() {
                            break chunk;
                        }
                        if wq.closed.load(Ordering::Relaxed) {
                            let _ = child.kill();
                            return;
                        }
                        c = wq.ready.wait(c).unwrap();
                    }
                };
                bytes.clear();
                bytes.extend(chunk.iter().flat_map(|s| s.to_le_bytes()));
                let ok = child.stdin.as_mut().is_some_and(|i| i.write_all(&bytes).and_then(|_| i.flush()).is_ok());
                wq.queued.fetch_sub(chunk.len() as u64, Ordering::Relaxed);
                if !ok {
                    // the device went away (or pacat died): start over on the next chunk
                    let _ = child.kill();
                    std::thread::sleep(Duration::from_millis(500));
                    match pacat(device.as_deref()) {
                        Ok(c) => child = c,
                        Err(_) => return,
                    }
                }
            }
        })
        .map_err(|e| e.to_string())?;
    Ok(Box::new(Pacat { q }))
}

impl PlayerImpl for Pacat {
    fn play(&self, chunk: &[f32]) {
        // in pieces of ~40 ms, so clear() takes effect quickly
        let mut c = self.q.chunks.lock().unwrap();
        for piece in chunk.chunks(960) {
            c.push_back(piece.to_vec());
        }
        self.q.queued.fetch_add(chunk.len() as u64, Ordering::Relaxed);
        self.q.ready.notify_one();
    }

    fn pending(&self) -> f32 {
        self.q.queued.load(Ordering::Relaxed) as f32 / SPEECH_SR as f32
    }

    fn clear(&self) {
        let mut c = self.q.chunks.lock().unwrap();
        let dropped: usize = c.drain(..).map(|p| p.len()).sum();
        self.q.queued.fetch_sub(dropped as u64, Ordering::Relaxed);
    }
}

impl Drop for Pacat {
    fn drop(&mut self) {
        self.q.closed.store(true, Ordering::Relaxed);
        self.q.ready.notify_one();
    }
}

pub fn virtual_mic() -> Result<VirtualMic, String> {
    let names = pactl(&["list", "short", "sinks"])? + &pactl(&["list", "short", "sources"])?;
    if !names.contains(SINK) {
        pactl(&[
            "load-module",
            "module-null-sink",
            &format!("sink_name={SINK}"),
            "rate=48000",
            "channels=1",
            "sink_properties=device.description=Mentalese-Internal-DONT-SELECT",
        ])?;
    }
    if !names.contains(&format!("\t{MIC}\t")) {
        pactl(&[
            "load-module",
            "module-remap-source",
            &format!("master={SINK}.monitor"),
            &format!("source_name={MIC}"),
            "source_properties=device.description=Mentalese-Mic",
        ])?;
    }
    Ok(VirtualMic::Ready { name: "Mentalese-Mic".into() })
}

pub fn remove_virtual_mic() -> Result<(), String> {
    for line in pactl(&["list", "short", "modules"])?.lines() {
        if (line.contains(SINK) || line.contains(MIC))
            && let Some(id) = line.split('\t').next()
        {
            pactl(&["unload-module", id])?;
        }
    }
    Ok(())
}

fn app_names(stream: &serde_json::Value) -> String {
    let p = &stream["properties"];
    ["application.name", "application.process.binary", "node.name"]
        .iter()
        .filter_map(|k| p[*k].as_str())
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

pub fn apps_playing() -> Vec<String> {
    let mut apps: Vec<String> = list("sink-inputs")
        .iter()
        .filter(|s| !app_names(s).contains("mentalese"))
        .filter_map(|s| s["properties"]["application.name"].as_str().map(String::from))
        .collect();
    apps.sort();
    apps.dedup();
    apps
}

type SharedOnFrame = Arc<Mutex<super::OnFrame>>;

struct AppCapture {
    stop: Arc<AtomicBool>,
    children: Arc<Mutex<Vec<(u64, Child)>>>,
}

impl Drop for AppCapture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for (_, c) in self.children.lock().unwrap().iter_mut() {
            let _ = c.kill();
        }
    }
}

pub fn capture_app(app: &str, on_frame: super::OnFrame) -> Result<Box<dyn Send>, String> {
    pactl(&["info"])?; // fail now, not silently in the watcher
    let app = app.to_lowercase();
    let on_frame: SharedOnFrame = Arc::new(Mutex::new(on_frame));
    let stop = Arc::new(AtomicBool::new(false));
    let children: Arc<Mutex<Vec<(u64, Child)>>> = Arc::default();
    let (wstop, wchildren) = (stop.clone(), children.clone());
    std::thread::Builder::new()
        .name("app-capture".into())
        .spawn(move || {
            while !wstop.load(Ordering::Relaxed) {
                for s in list("sink-inputs") {
                    let names = app_names(&s);
                    let Some(idx) = s["index"].as_u64() else { continue };
                    if !names.contains(&app) || names.contains("mentalese") {
                        continue;
                    }
                    let mut kids = wchildren.lock().unwrap();
                    kids.retain_mut(|(_, c)| matches!(c.try_wait(), Ok(None)));
                    if kids.iter().any(|(i, _)| *i == idx) {
                        continue;
                    }
                    let args = [format!("--monitor-stream={idx}"), format!("--rate={MIC_SR}"),
                                "--client-name=mentalese-listen".to_string()];
                    let Ok((child, mut out)) = parec(&args) else { continue };
                    kids.push((idx, child));
                    let f = on_frame.clone();
                    std::thread::spawn(move || {
                        let (mut bytes, mut frame) = (vec![0u8; FRAME * 4], [0f32; FRAME]);
                        // ends when the stream does (the call is over) or capture stops
                        while read_frame(&mut out, &mut bytes, &mut frame).is_ok() {
                            (f.lock().unwrap())(CAPTURE_BASE + idx as u32, &frame);
                        }
                    });
                }
                std::thread::sleep(Duration::from_secs(2));
            }
        })
        .map_err(|e| e.to_string())?;
    Ok(Box::new(AppCapture { stop, children }))
}
