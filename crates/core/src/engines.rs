//! The native engines, loaded at runtime: libmentalese-llm (llama.cpp: speech recognition and
//! translation) and libmentalese-voice (qwentts.cpp). There is one build of each per GPU backend,
//! so one app binary runs on NVIDIA, AMD, Intel or Apple GPUs, or on the CPU.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use libloading::Library;

const LLM_ABI: c_int = 1;
const VOICE_ABI: c_int = 1;

type LogCb = extern "C" fn(c_int, *const c_char);
type ChunkCb = extern "C" fn(*const f32, c_int, *mut c_void) -> c_int;

extern "C" fn on_log(level: c_int, msg: *const c_char) {
    if msg.is_null() {
        return;
    }
    let msg = unsafe { CStr::from_ptr(msg) }.to_string_lossy();
    let kind = if level >= 4 { "error" } else { "warning" };
    // warnings are mostly model-file trivia ("unused tensor"); show them only when asked
    if level >= 4 || std::env::var_os("MENTALESE_DEBUG").is_some() {
        eprintln!("[engine {kind}] {}", msg.trim_end());
    }
}

struct LlmApi {
    _lib: Library,
    devices: unsafe extern "C" fn() -> *const c_char,
    marker: unsafe extern "C" fn() -> *const c_char,
    load: unsafe extern "C" fn(*const c_char, *const c_char, c_int, c_int, c_int, c_int, c_int) -> *mut c_void,
    generate: unsafe extern "C" fn(*mut c_void, *const c_char, *const f32, c_int, c_int) -> *const c_char,
    free: unsafe extern "C" fn(*mut c_void),
    last_error: unsafe extern "C" fn() -> *const c_char,
}

struct VoiceApi {
    _lib: Library,
    load: unsafe extern "C" fn(*const c_char, *const c_char, *const c_char) -> *mut c_void,
    set_ref: unsafe extern "C" fn(*mut c_void, *const f32, c_int) -> c_int,
    say: unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char, c_int, *const c_char, ChunkCb, *mut c_void)
        -> c_int,
    free: unsafe extern "C" fn(*mut c_void),
    last_error: unsafe extern "C" fn() -> *const c_char,
}

fn open(path: &Path) -> Result<Library, String> {
    #[cfg(windows)]
    let lib = {
        use libloading::os::windows::{LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR};
        // CUDA's DLLs ship next to ours; let Windows find them there
        unsafe {
            libloading::os::windows::Library::load_with_flags(
                path,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
            )
        }
        .map(Library::from)
    };
    #[cfg(not(windows))]
    let lib = unsafe { Library::new(path) };
    lib.map_err(|e| format!("{}: {e}", path.display()))
}

macro_rules! sym {
    ($lib:expr, $name:literal) => {
        *unsafe { $lib.get(concat!($name, "\0").as_bytes()) }.map_err(|e| format!("{}: {e}", $name))?
    };
}

impl LlmApi {
    fn open(path: &Path) -> Result<LlmApi, String> {
        let lib = open(path)?;
        let abi: unsafe extern "C" fn() -> c_int = sym!(lib, "mx_llm_abi");
        if unsafe { abi() } != LLM_ABI {
            return Err(format!("{} is a different version than this app", path.display()));
        }
        let log: unsafe extern "C" fn(Option<LogCb>) = sym!(lib, "mx_llm_log");
        unsafe { log(Some(on_log)) };
        Ok(LlmApi {
            devices: sym!(lib, "mx_llm_devices"),
            marker: sym!(lib, "mx_llm_marker"),
            load: sym!(lib, "mx_llm_load"),
            generate: sym!(lib, "mx_llm_generate"),
            free: sym!(lib, "mx_llm_free"),
            last_error: sym!(lib, "mx_llm_last_error"),
            _lib: lib,
        })
    }

    fn error(&self) -> String {
        unsafe { CStr::from_ptr((self.last_error)()) }.to_string_lossy().into_owned()
    }
}

impl VoiceApi {
    fn open(path: &Path) -> Result<VoiceApi, String> {
        let lib = open(path)?;
        let abi: unsafe extern "C" fn() -> c_int = sym!(lib, "mx_voice_abi");
        if unsafe { abi() } != VOICE_ABI {
            return Err(format!("{} is a different version than this app", path.display()));
        }
        let log: unsafe extern "C" fn(Option<LogCb>) = sym!(lib, "mx_voice_log");
        unsafe { log(Some(on_log)) };
        Ok(VoiceApi {
            load: sym!(lib, "mx_voice_load"),
            set_ref: sym!(lib, "mx_voice_set_ref"),
            say: sym!(lib, "mx_voice_say"),
            free: sym!(lib, "mx_voice_free"),
            last_error: sym!(lib, "mx_voice_last_error"),
            _lib: lib,
        })
    }

    fn error(&self) -> String {
        unsafe { CStr::from_ptr((self.last_error)()) }.to_string_lossy().into_owned()
    }
}

/// A GPU the engines can use.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Gpu {
    pub name: String,
    pub description: String,
    pub free_mb: u64,
    pub total_mb: u64,
}

/// Both engines, from one backend build.
#[derive(Clone)]
pub struct Engines {
    /// "cuda", "vulkan", "metal" or "cpu"
    pub backend: String,
    pub gpus: Vec<Gpu>,
    /// The speech-recognition/translation library in use.
    pub path: PathBuf,
    llm: Arc<LlmApi>,
    voice: Arc<VoiceApi>,
}

const BACKENDS: &[&str] = if cfg!(target_os = "macos") { &["metal", "cpu"] } else { &["cuda", "vulkan", "cpu"] };

fn lib_name(stem: &str) -> String {
    if cfg!(windows) {
        format!("{stem}.dll")
    } else if cfg!(target_os = "macos") {
        format!("lib{stem}.dylib")
    } else {
        format!("lib{stem}.so")
    }
}

static ROOT: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// Where an installed app keeps its engines (one folder per backend inside), searched first.
pub fn set_root(dir: PathBuf) {
    let _ = ROOT.set(dir);
}

/// Where each backend's libraries may be: (backend, llm library, voice library).
fn candidates() -> Vec<(String, PathBuf, PathBuf)> {
    let (llm, voice) = (lib_name("mentalese-llm"), lib_name("mentalese-voice"));
    let mut out = Vec::new();
    if let Some(dir) = std::env::var_os("MENTALESE_ENGINES") {
        let dir = PathBuf::from(dir);
        let backend = dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        out.push((backend, dir.join(&llm), dir.join(&voice)));
    }
    let mut roots: Vec<PathBuf> = ROOT.get().cloned().into_iter().collect();
    if let Some(exe_dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)) {
        roots.push(exe_dir.join("engines"));
        // macOS app bundles keep libraries in Contents/Resources
        roots.push(exe_dir.join("../Resources/engines"));
        // Linux packages: /usr/bin/app -> /usr/lib/<app>/engines
        roots.push(exe_dir.join("../lib/mentalese/engines"));
    }
    for backend in BACKENDS {
        for root in &roots {
            let dir = root.join(backend);
            out.push((backend.to_string(), dir.join(&llm), dir.join(&voice)));
        }
        // development: engines built by tools/build-engines into <repo>/build
        let build = crate::paths::repo().join("build");
        out.push((backend.to_string(), build.join(format!("llm-{backend}")).join(&llm),
                  build.join(format!("voice-{backend}")).join(&voice)));
    }
    // MENTALESE_BACKEND=vulkan (or cuda, metal, cpu) uses only that one
    if let Some(only) = std::env::var("MENTALESE_BACKEND").ok().filter(|b| !b.is_empty()) {
        out.retain(|(b, _, _)| b.eq_ignore_ascii_case(&only));
    }
    out
}

impl Engines {
    /// The first backend whose libraries load and, unless it's the CPU build, see a GPU.
    pub fn load() -> Result<Engines, String> {
        let mut tried = Vec::new();
        for (backend, llm_path, voice_path) in candidates() {
            if !llm_path.exists() || !voice_path.exists() {
                continue;
            }
            let llm = match LlmApi::open(&llm_path) {
                Ok(l) => l,
                Err(e) => {
                    tried.push(e);
                    continue;
                }
            };
            let gpus = parse_devices(unsafe { CStr::from_ptr((llm.devices)()) }.to_str().unwrap_or(""));
            if gpus.is_empty() && backend != "cpu" {
                tried.push(format!("{backend}: no GPU found"));
                continue;
            }
            match VoiceApi::open(&voice_path) {
                Ok(voice) => {
                    return Ok(Engines { backend, gpus, path: llm_path, llm: Arc::new(llm), voice: Arc::new(voice) });
                }
                Err(e) => tried.push(e),
            }
        }
        Err(if tried.is_empty() {
            "the speech engines aren't installed (no mentalese-llm / mentalese-voice libraries found)".into()
        } else {
            format!("couldn't start the speech engines: {}", tried.join("; "))
        })
    }

    pub fn load_llm(&self, model: &Path, mmproj: Option<&Path>, opts: LlmOptions) -> Result<Llm, String> {
        let model_c = cpath(model)?;
        let mmproj_c = mmproj.map(cpath).transpose()?;
        let h = unsafe {
            (self.llm.load)(
                model_c.as_ptr(),
                mmproj_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
                opts.mmproj_gpu as c_int,
                opts.n_ctx,
                opts.n_batch,
                opts.n_ubatch,
                threads(),
            )
        };
        if h.is_null() {
            return Err(self.llm.error());
        }
        let marker = unsafe { CStr::from_ptr((self.llm.marker)()) }.to_string_lossy().into_owned();
        Ok(Llm { api: self.llm.clone(), h, marker })
    }

    pub fn load_voice(&self, talker: &Path, codec: &Path) -> Result<Voice, String> {
        let (t, c) = (cpath(talker)?, cpath(codec)?);
        let h = unsafe { (self.voice.load)(t.as_ptr(), c.as_ptr(), std::ptr::null()) };
        if h.is_null() {
            return Err(self.voice.error());
        }
        Ok(Voice { api: self.voice.clone(), h })
    }
}

fn parse_devices(s: &str) -> Vec<Gpu> {
    s.lines()
        .filter_map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            let mb = |i: usize| f.get(i).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0) >> 20;
            (f.len() >= 4).then(|| Gpu { name: f[0].into(), description: f[1].into(), free_mb: mb(2), total_mb: mb(3) })
        })
        .collect()
}

/// CPU threads for the parts that run on the CPU (the speech recognizer's audio encoder): one per
/// physical core, like llama.cpp's own tools. mtmd's default of 4 makes recognition ~2x slower.
fn threads() -> c_int {
    let logical = std::thread::available_parallelism().map_or(4, |n| n.get());
    (logical / 2).clamp(1, 16) as c_int
}

fn cpath(p: &Path) -> Result<CString, String> {
    CString::new(p.to_string_lossy().as_bytes()).map_err(|_| format!("bad path {}", p.display()))
}

#[derive(Clone, Copy)]
pub struct LlmOptions {
    pub mmproj_gpu: bool,
    pub n_ctx: c_int,
    pub n_batch: c_int,
    pub n_ubatch: c_int,
}

/// A loaded language model. One request at a time.
pub struct Llm {
    api: Arc<LlmApi>,
    h: *mut c_void,
    /// Where the audio goes in an audio prompt.
    pub marker: String,
}

// The handle is only ever used by one thread at a time (&mut self).
unsafe impl Send for Llm {}

impl Llm {
    /// Greedy generation. `audio`: 16 kHz mono, placed where the prompt has `self.marker`.
    pub fn generate(&mut self, prompt: &str, audio: Option<&[f32]>, max_tokens: usize) -> Result<String, String> {
        let p = CString::new(prompt).map_err(|_| "prompt contains a NUL byte".to_string())?;
        let (ptr, n) = audio.map_or((std::ptr::null(), 0), |a| (a.as_ptr(), a.len() as c_int));
        let out = unsafe { (self.api.generate)(self.h, p.as_ptr(), ptr, n, max_tokens as c_int) };
        if out.is_null() {
            return Err(self.api.error());
        }
        Ok(unsafe { CStr::from_ptr(out) }.to_string_lossy().into_owned())
    }
}

impl Drop for Llm {
    fn drop(&mut self) {
        unsafe { (self.api.free)(self.h) };
    }
}

/// A loaded voice model.
pub struct Voice {
    api: Arc<VoiceApi>,
    h: *mut c_void,
}

unsafe impl Send for Voice {}

extern "C" fn chunk_trampoline<F: FnMut(&[f32]) -> bool>(samples: *const f32, n: c_int, user: *mut c_void) -> c_int {
    let f = unsafe { &mut *(user as *mut F) };
    let chunk = if samples.is_null() || n <= 0 { &[][..] } else { unsafe { std::slice::from_raw_parts(samples, n as usize) } };
    f(chunk) as c_int
}

impl Voice {
    /// The recording to clone: 24 kHz mono.
    pub fn set_ref(&mut self, wav: &[f32]) -> Result<(), String> {
        match unsafe { (self.api.set_ref)(self.h, wav.as_ptr(), wav.len() as c_int) } {
            0 => Ok(()),
            _ => Err(self.api.error()),
        }
    }

    /// Speaks `text`, calling `on_chunk` with 24 kHz audio as it's made (on the engine's own
    /// thread); `on_chunk` returning false stops it. With `ref_text` the voice continues the
    /// sample recording; without, it uses the speaker fingerprint only. Returns false if stopped.
    pub fn say<F: FnMut(&[f32]) -> bool + Send>(
        &mut self,
        text: &str,
        lang: &str,
        max_frames: usize,
        ref_text: Option<&CStr>,
        mut on_chunk: F,
    ) -> Result<bool, String> {
        let text = CString::new(text).map_err(|_| "text contains a NUL byte".to_string())?;
        let lang = CString::new(lang).unwrap_or_default();
        let rc = unsafe {
            (self.api.say)(
                self.h,
                text.as_ptr(),
                lang.as_ptr(),
                max_frames as c_int,
                ref_text.map_or(std::ptr::null(), CStr::as_ptr),
                chunk_trampoline::<F>,
                &mut on_chunk as *mut F as *mut c_void,
            )
        };
        match rc {
            0 => Ok(true),
            1 => Ok(false),
            _ => Err(self.api.error()),
        }
    }
}

impl Drop for Voice {
    fn drop(&mut self) {
        unsafe { (self.api.free)(self.h) };
    }
}
