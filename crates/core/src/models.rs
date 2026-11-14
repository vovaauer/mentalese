//! The model files, where they come from, and downloading them (checked against their SHA-256).

use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::paths;

pub struct Model {
    pub file: &'static str,
    pub url: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
    /// What it's for, shown while downloading.
    pub what: &'static str,
}

macro_rules! hf {
    ($repo:literal, $file:literal) => {
        concat!("https://huggingface.co/", $repo, "/resolve/main/", $file)
    };
}

pub const ASR: Model = Model {
    file: "Qwen3-ASR-0.6B-Q8_0.gguf",
    url: hf!("ggml-org/Qwen3-ASR-0.6B-GGUF", "Qwen3-ASR-0.6B-Q8_0.gguf"),
    bytes: 804_749_248,
    sha256: "bca259818b50ca7c4c05e9bdb35a5dc04fa039653a6d6f3f0f331f96f6aa1971",
    what: "speech recognition",
};
pub const ASR_MMPROJ: Model = Model {
    file: "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf",
    url: hf!("ggml-org/Qwen3-ASR-0.6B-GGUF", "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"),
    bytes: 214_392_480,
    sha256: "41a342b5e4c514e968cb756de6cd1b7be39eff43c44c57a2ef5fc6522e36603d",
    what: "speech recognition (audio encoder)",
};
pub const MT: Model = Model {
    file: "Index-Translate-2B.Q4_K_M.gguf",
    url: hf!("IndexTeam/Index-Translate-2B-GGUF", "Index-Translate-2B.Q4_K_M.gguf"),
    bytes: 1_312_164_352,
    sha256: "044b313d29342bd3b2c77cbb64023ca0d209bd9b3247763f9d162767ef2d746a",
    what: "translator",
};
pub const TALKER: Model = Model {
    file: "qwen-talker-0.6b-base-Q8_0.gguf",
    url: hf!("Serveurperso/Qwen3-TTS-GGUF", "qwen-talker-0.6b-base-Q8_0.gguf"),
    bytes: 992_615_488,
    sha256: "d54dbaf10591421fa764ed630d764efa717ae40cd959bd48c66d4eb1af226426",
    what: "voice",
};
pub const CODEC: Model = Model {
    file: "qwen-tokenizer-12hz-Q8_0.gguf",
    url: hf!("Serveurperso/Qwen3-TTS-GGUF", "qwen-tokenizer-12hz-Q8_0.gguf"),
    bytes: 291_150_624,
    sha256: "1883beeed99348fc35e23dd225e9082f93f6f8c109330a33d935baa8acdbfd94",
    what: "voice (audio codec)",
};
pub const VAD: Model = Model {
    file: "silero_vad.onnx",
    url: "https://github.com/snakers4/silero-vad/raw/v6.2.3/src/silero_vad/data/silero_vad.onnx",
    bytes: 2_327_524,
    sha256: "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
    what: "voice detection",
};

pub const ALL: [&Model; 6] = [&VAD, &ASR, &ASR_MMPROJ, &MT, &TALKER, &CODEC];

impl Model {
    pub fn path(&self) -> PathBuf {
        paths::models().join(self.file)
    }

    /// Present with the right size. (The hash is checked once, right after downloading.)
    pub fn present(&self) -> bool {
        std::fs::metadata(self.path()).is_ok_and(|m| m.len() == self.bytes)
    }
}

pub fn missing() -> Vec<&'static Model> {
    ALL.into_iter().filter(|m| !m.present()).collect()
}

/// Total bytes still to download.
pub fn missing_bytes() -> u64 {
    missing().iter().map(|m| m.bytes).sum()
}

/// Downloads `m` with curl (on every supported OS), resuming a partial download, and reports
/// progress as (bytes so far, total). Verifies the SHA-256 before putting the file in place.
pub fn download(m: &Model, mut progress: impl FnMut(u64, u64)) -> Result<(), String> {
    let dir = paths::models();
    std::fs::create_dir_all(&dir).map_err(|e| format!("can't create {}: {e}", dir.display()))?;
    let part = dir.join(format!("{}.part", m.file));
    let mut child = Command::new("curl")
        .args(["-fsSL", "--retry", "5", "--retry-delay", "2", "-C", "-", "-A", "mentalese", "-o"])
        .arg(&part)
        .arg(m.url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("can't run curl: {e}"))?;
    let status = loop {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            break status;
        }
        progress(std::fs::metadata(&part).map_or(0, |md| md.len()), m.bytes);
        std::thread::sleep(Duration::from_millis(250));
    };
    if !status.success() {
        let mut err = String::new();
        child.stderr.take().map(|mut s| s.read_to_string(&mut err));
        return Err(format!("downloading the {} failed: {}", m.what, err.trim()));
    }
    progress(m.bytes, m.bytes);
    if sha256(&part)? != m.sha256 {
        let _ = std::fs::remove_file(&part);
        return Err(format!("the {} download was corrupted; try again", m.what));
    }
    std::fs::rename(&part, m.path()).map_err(|e| e.to_string())
}

fn sha256(path: &std::path::Path) -> Result<String, String> {
    let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}
