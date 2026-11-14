//! Where mentalese keeps its models, voices and settings.

use std::path::PathBuf;
use std::sync::LazyLock;

/// The source checkout this binary was built from.
pub fn repo() -> PathBuf {
    let p = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."));
    p.canonicalize().unwrap_or(p)
}

static HOME: LazyLock<PathBuf> = LazyLock::new(|| {
    if let Some(h) = std::env::var_os("MENTALESE_HOME") {
        return h.into();
    }
    // running from the source checkout: use its models/ and voices/
    let repo = repo();
    if repo.join("models").is_dir() && repo.join("Cargo.toml").is_file() {
        return repo;
    }
    data_dir().join("mentalese")
});

/// MENTALESE_HOME, else the source checkout when running from one, else the per-user data
/// directory (~/.local/share/mentalese, ~/Library/Application Support/mentalese, %APPDATA%\mentalese).
pub fn home() -> PathBuf {
    HOME.clone()
}

pub fn models() -> PathBuf {
    home().join("models")
}

pub fn voices() -> PathBuf {
    home().join("voices")
}

pub fn settings() -> PathBuf {
    home().join("settings.json")
}

fn data_dir() -> PathBuf {
    let env = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    if cfg!(windows) {
        env("APPDATA").unwrap_or_else(|| PathBuf::from("."))
    } else if cfg!(target_os = "macos") {
        env("HOME").unwrap_or_default().join("Library/Application Support")
    } else {
        env("XDG_DATA_HOME").unwrap_or_else(|| env("HOME").unwrap_or_default().join(".local/share"))
    }
}
