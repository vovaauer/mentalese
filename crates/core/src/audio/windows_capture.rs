//! Windows: other apps' audio through WASAPI process loopback (Windows 10 2004+ / 11), which
//! hears everything one process tree plays, on whatever device it plays to.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;

use windows::Win32::Foundation::{CloseHandle, HANDLE, S_OK, WAIT_OBJECT_0};
use windows::Win32::Media::Audio::{
    AUDCLNT_BUFFERFLAGS_SILENT, AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM, AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
    AUDCLNT_STREAMFLAGS_LOOPBACK, AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY, AUDIOCLIENT_ACTIVATION_PARAMS,
    AUDIOCLIENT_ACTIVATION_PARAMS_0, AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK, AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS,
    ActivateAudioInterfaceAsync, AudioSessionStateActive, DEVICE_STATE_ACTIVE, IActivateAudioInterfaceAsyncOperation,
    IActivateAudioInterfaceCompletionHandler, IActivateAudioInterfaceCompletionHandler_Impl, IAudioCaptureClient,
    IAudioClient, IAudioSessionControl2, IAudioSessionManager2, IMMDeviceEnumerator, MMDeviceEnumerator,
    PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE, VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK, WAVEFORMATEX, eRender,
};
use windows::Win32::System::Com::StructuredStorage::PROPVARIANT;
use windows::Win32::System::Com::{BLOB, CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoUninitialize};
use windows::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, PROCESSENTRY32W, Process32FirstW, Process32NextW, TH32CS_SNAPPROCESS,
};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::Win32::System::Variant::VT_BLOB;
use windows_core::{HRESULT, Interface, Ref, implement};

use super::cpal_io::{CAPTURE_BASE, Framer, OnFrame, Watcher};

/// COM on this thread, multithreaded: the activation callback comes from another thread.
struct Com;

impl Com {
    fn init() -> Result<Com, String> {
        unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) }.ok().map(|()| Com).map_err(|e| e.to_string())
    }
}

impl Drop for Com {
    fn drop(&mut self) {
        unsafe { CoUninitialize() }
    }
}

struct Event(HANDLE);

impl Drop for Event {
    fn drop(&mut self) {
        let _ = unsafe { CloseHandle(self.0) };
    }
}

struct Proc {
    pid: u32,
    parent: u32,
    /// the executable's name without ".exe"
    name: String,
}

fn processes() -> Vec<Proc> {
    let mut procs = Vec::new();
    unsafe {
        let Ok(snap) = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) else { return procs };
        let mut e = PROCESSENTRY32W { dwSize: size_of::<PROCESSENTRY32W>() as u32, ..Default::default() };
        let mut more = Process32FirstW(snap, &mut e).is_ok();
        while more {
            let len = e.szExeFile.iter().position(|&c| c == 0).unwrap_or(e.szExeFile.len());
            let mut name = String::from_utf16_lossy(&e.szExeFile[..len]);
            if name.to_ascii_lowercase().ends_with(".exe") {
                name.truncate(name.len() - 4);
            }
            procs.push(Proc { pid: e.th32ProcessID, parent: e.th32ParentProcessID, name });
            more = Process32NextW(snap, &mut e).is_ok();
        }
        let _ = CloseHandle(snap);
    }
    procs
}

/// The processes named like `app` whose parent isn't: one per app instance. Discord and other
/// Chromium apps play from a child process, which process loopback includes with its root.
fn roots(app: &str) -> Vec<u32> {
    let procs = processes();
    let me = std::process::id();
    let hit = |p: &Proc| p.pid != me && p.name.to_lowercase().contains(app);
    let by_pid: HashMap<u32, &Proc> = procs.iter().map(|p| (p.pid, p)).collect();
    procs.iter().filter(|p| hit(p) && !by_pid.get(&p.parent).is_some_and(|q| hit(q))).map(|p| p.pid).collect()
}

pub(super) fn apps_playing() -> Vec<String> {
    // on a thread of its own, so this thread's COM is ours to set up
    let found = std::thread::spawn(|| -> Option<Vec<String>> {
        let _com = Com::init().ok()?;
        let names: HashMap<u32, String> = processes().into_iter().map(|p| (p.pid, p.name)).collect();
        let me = std::process::id();
        let mut apps = Vec::new();
        unsafe {
            let enumerator: IMMDeviceEnumerator = CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL).ok()?;
            let outputs = enumerator.EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE).ok()?;
            for i in 0..outputs.GetCount().unwrap_or(0) {
                let sessions = outputs
                    .Item(i)
                    .and_then(|d| d.Activate::<IAudioSessionManager2>(CLSCTX_ALL, None))
                    .and_then(|m| m.GetSessionEnumerator());
                let Ok(sessions) = sessions else { continue };
                for j in 0..sessions.GetCount().unwrap_or(0) {
                    let Ok(s) = sessions.GetSession(j).and_then(|s| s.cast::<IAudioSessionControl2>()) else { continue };
                    if s.IsSystemSoundsSession() == S_OK || !s.GetState().is_ok_and(|st| st == AudioSessionStateActive) {
                        continue;
                    }
                    apps.extend(s.GetProcessId().ok().filter(|&pid| pid != me).and_then(|pid| names.get(&pid)).cloned());
                }
            }
        }
        Some(apps)
    })
    .join();
    let mut apps = found.ok().flatten().unwrap_or_default();
    apps.retain(|a| !a.to_lowercase().contains("mentalese"));
    apps.sort();
    apps.dedup();
    apps
}

#[implement(IActivateAudioInterfaceCompletionHandler)]
struct Activated(mpsc::Sender<()>);

impl IActivateAudioInterfaceCompletionHandler_Impl for Activated_Impl {
    fn ActivateCompleted(&self, _: Ref<IActivateAudioInterfaceAsyncOperation>) -> windows_core::Result<()> {
        let _ = self.0.send(());
        Ok(())
    }
}

/// An audio client hearing process `pid` and its descendants. Needs COM on this thread.
fn activate(pid: u32) -> Result<IAudioClient, String> {
    let params = AUDIOCLIENT_ACTIVATION_PARAMS {
        ActivationType: AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK,
        Anonymous: AUDIOCLIENT_ACTIVATION_PARAMS_0 {
            ProcessLoopbackParams: AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS {
                TargetProcessId: pid,
                ProcessLoopbackMode: PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE,
            },
        },
    };
    // PROPVARIANT here has no Drop (no PropVariantClear), so pointing it at `params` is fine
    let mut pv = PROPVARIANT::default();
    unsafe {
        let v = &mut *pv.Anonymous.Anonymous;
        v.vt = VT_BLOB;
        v.Anonymous.blob = BLOB { cbSize: size_of_val(&params) as u32, pBlobData: (&raw const params).cast_mut().cast() };
    }
    let (done, completed) = mpsc::channel();
    let handler: IActivateAudioInterfaceCompletionHandler = Activated(done).into();
    let op = unsafe {
        ActivateAudioInterfaceAsync(VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK, &IAudioClient::IID, Some(&raw const pv), &handler)
    }
    .map_err(|e| e.to_string())?;
    // Windows now holds the only handler reference: it either calls it or releases it, and either
    // ends this wait (`params` must outlive the activation).
    drop(handler);
    let _ = completed.recv();
    let (mut hr, mut client) = (HRESULT(0), None);
    unsafe { op.GetActivateResult(&mut hr, &mut client) }.map_err(|e| e.to_string())?;
    hr.ok().map_err(|e| e.to_string())?;
    client.ok_or("no audio client")?.cast().map_err(|e| e.to_string())
}

const WAVE_FORMAT_PCM: u16 = 1;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;

/// Process loopback has no mix format of its own: we pick one and Windows converts. Float 48 kHz
/// stereo first; Microsoft's ApplicationLoopback sample's 16-bit 44.1 kHz as a fallback (untested
/// which of these every Windows build accepts).
const FORMATS: [(u16, u32, u16, u32); 2] = [
    (WAVE_FORMAT_IEEE_FLOAT, 48_000, 32, AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY),
    (WAVE_FORMAT_PCM, 44_100, 16, 0),
];

/// An initialized loopback client for `pid`, with its rate and bits per sample (always stereo).
fn open(pid: u32) -> Result<(IAudioClient, u32, u16), String> {
    let mut last = String::new();
    for (tag, rate, bits, flags) in FORMATS {
        let client = activate(pid)?;
        let block = 2 * bits / 8;
        let format = WAVEFORMATEX {
            wFormatTag: tag,
            nChannels: 2,
            nSamplesPerSec: rate,
            nAvgBytesPerSec: rate * block as u32,
            nBlockAlign: block,
            wBitsPerSample: bits,
            cbSize: 0,
        };
        let flags = AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_EVENTCALLBACK | flags;
        // 20 ms buffer
        match unsafe { client.Initialize(AUDCLNT_SHAREMODE_SHARED, flags, 200_000, 0, &format, None) } {
            Ok(()) => return Ok((client, rate, bits)),
            Err(e) => last = e.to_string(),
        }
    }
    Err(last)
}

/// Captures `pid`'s tree until `stop`. Runs on its own thread, which owns every COM object here.
fn capture(pid: u32, stop: &AtomicBool, framer: impl FnOnce(u32) -> Framer) -> Result<(), String> {
    let e = |e: windows_core::Error| e.to_string();
    let _com = Com::init()?;
    // declared before the client so it's closed after
    let event = Event(unsafe { CreateEventW(None, false, false, None) }.map_err(e)?);
    let (client, rate, bits) = open(pid)?;
    let mut framer = framer(rate);
    let mut mono = Vec::new();
    // silence fed since the app last sent audio; starts as if a second was already fed
    let mut idle = rate as usize;
    unsafe {
        client.SetEventHandle(event.0).map_err(e)?;
        let cap: IAudioCaptureClient = client.GetService().map_err(e)?;
        client.Start().map_err(e)?;
        while !stop.load(Ordering::Relaxed) {
            if WaitForSingleObject(event.0, 100) != WAIT_OBJECT_0 {
                // Process loopback sends nothing while the app is silent (untested assumption,
                // as plain loopback does): feed up to a second of silence so a sentence in
                // progress ends.
                if idle < rate as usize {
                    mono.clear();
                    mono.resize(rate as usize / 10, 0.0);
                    idle += mono.len();
                    framer.push(&mono);
                }
                continue;
            }
            while cap.GetNextPacketSize().map_err(e)? > 0 {
                let (mut data, mut frames, mut flags) = (std::ptr::null_mut(), 0u32, 0u32);
                cap.GetBuffer(&mut data, &mut frames, &mut flags, None, None).map_err(e)?;
                let n = frames as usize;
                mono.clear();
                if flags & AUDCLNT_BUFFERFLAGS_SILENT.0 as u32 != 0 || data.is_null() {
                    mono.resize(n, 0.0);
                } else if bits == 32 {
                    let s = std::slice::from_raw_parts(data.cast::<[f32; 2]>(), n);
                    mono.extend(s.iter().map(|[l, r]| (l + r) / 2.0));
                } else {
                    let s = std::slice::from_raw_parts(data.cast::<[i16; 2]>(), n);
                    mono.extend(s.iter().map(|&[l, r]| (l as f32 + r as f32) / 65_536.0));
                }
                cap.ReleaseBuffer(frames).map_err(e)?;
                idle = 0;
                framer.push(&mono);
            }
        }
        let _ = client.Stop();
    }
    Ok(())
}

/// One app instance's capture thread; dropping it stops it.
struct Worker {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    fn spawn(pid: u32, stream: u32, on_frame: OnFrame) -> std::io::Result<Worker> {
        let stop = Arc::new(AtomicBool::new(false));
        let s = stop.clone();
        let thread = std::thread::Builder::new()
            .name("app-capture-pid".into())
            .spawn(move || {
                let _ = capture(pid, &s, |rate| Framer::new(rate, stream, on_frame));
            })?;
        Ok(Worker { stop, thread: Some(thread) })
    }

    fn finished(&self) -> bool {
        self.thread.as_ref().is_none_or(|t| t.is_finished())
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

pub(super) fn capture_app(app: &str, on_frame: super::OnFrame) -> Result<Box<dyn Send>, String> {
    // fail now, not silently in the watcher
    std::thread::spawn(|| Com::init().and_then(|_com| activate(std::process::id()).map(drop)))
        .join()
        .map_err(|_| "can't check for app capture".to_string())?
        .map_err(|e| format!("hearing other apps needs Windows 10 version 2004 or later ({e})"))?;
    let app = app.to_lowercase();
    let on_frame: OnFrame = Arc::new(Mutex::new(on_frame));
    let mut live: HashMap<u32, Worker> = HashMap::new();
    let mut next = CAPTURE_BASE;
    let watcher = Watcher::spawn(move || {
        let roots = roots(&app);
        // quit or restarted (new pids), or failed (retried below)
        live.retain(|pid, w| roots.contains(pid) && !w.finished());
        for pid in roots {
            if let Entry::Vacant(slot) = live.entry(pid)
                && let Ok(w) = Worker::spawn(pid, next, on_frame.clone())
            {
                slot.insert(w);
                next += 1;
            }
        }
    })?;
    Ok(Box::new(watcher))
}
