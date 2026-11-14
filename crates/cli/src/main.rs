//! mentalese on the command line. The app (mentalese-app) does the same with buttons.

use std::io::{BufRead, Write};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use mentalese_core::audio::{self, Output, Player, VirtualMic};
use mentalese_core::engines::Engines;
use mentalese_core::pipeline::{Event, Session};
use mentalese_core::settings::Settings;
use mentalese_core::tts::{self, Tts};
use mentalese_core::{asr::Asr, langid, langs, models, mt::Translator, paths, voices};

const USAGE: &str = "mentalese: speak one language, others hear another, in your own voice.

  mentalese record [--voice NAME]       record a voice sample (yours: \"me\")
  mentalese run [options]               start translating; type to speak, /zh etc. to switch, /quit
  mentalese say --to ru \"text\"          hear your voice say something, translated, in your headphones
  mentalese voices                      list recorded voices
  mentalese devices                     list microphones and speakers
  mentalese download                    download the models now (the first run does it anyway)
  mentalese remove-mic                  remove the virtual mic (Linux)

run options (defaults come from the app's saved settings):
  --to LANG             language others hear: en ru zh ja ko de fr es pt it
  --me LANG             your language: subtitles and typed text
  --voice NAME          which recorded voice to speak with
  --keep-accent         other languages keep your sample's accent (closer to you, less native)
  --practice            hear yourself translated in your headphones; no virtual mic, no subtitles
  --monitor             also hear your translation in your headphones
  --style TEXT          tone of translations in plain words; --style \"\" for neutral
  --silence-ms N        pause that ends your sentence (default 400; lower is faster, may cut you off)
  --their-silence-ms N  the same for others (default 700)
  --asr-gpu             speech recognition fully on the GPU: ~40 ms faster, 0.75 GB more VRAM
  --no-subtitles        don't listen to the Vesktop plugin
  --subtitles-from APP  also subtitle an app's audio, e.g. Telegram (best for 1:1 calls)
  --mic DEVICE          microphone (an id from `mentalese devices`)
  --headphones DEVICE   where you hear yourself";

const COLORS: [&str; 6] = ["\x1b[96m", "\x1b[93m", "\x1b[95m", "\x1b[92m", "\x1b[94m", "\x1b[91m"];
const DIM: &str = "\x1b[2m";
const RED: &str = "\x1b[91m";
const RESET: &str = "\x1b[0m";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("record") => record(&args[1..]),
        Some("run") => run(&args[1..]),
        Some("say") => say(&args[1..]),
        Some("voices") => {
            let v = voices::list();
            println!("{}", if v.is_empty() { "no voices yet: mentalese record".into() } else { v.join("\n") });
            Ok(())
        }
        Some("devices") => {
            println!("microphones:");
            audio::inputs().iter().for_each(|d| println!("  {}  ({})", d.name, d.id));
            println!("speakers/headphones:");
            audio::outputs().iter().for_each(|d| println!("  {}  ({})", d.name, d.id));
            Ok(())
        }
        Some("download") => download(),
        Some("remove-mic") => audio::remove_virtual_mic(),
        _ => {
            println!("{USAGE}");
            Ok(())
        }
    };
    if let Err(e) = result {
        eprintln!("{RED}{e}{RESET}");
        std::process::exit(1);
    }
}

/// Simple flag parsing: --flag and --flag VALUE; the rest are positional.
struct Args {
    flags: Vec<(String, Option<String>)>,
    rest: Vec<String>,
}

impl Args {
    fn parse(args: &[String], with_value: &[&str]) -> Result<Args, String> {
        let (mut flags, mut rest) = (Vec::new(), Vec::new());
        let mut it = args.iter();
        while let Some(a) = it.next() {
            if let Some(name) = a.strip_prefix("--") {
                let (name, inline) = match name.split_once('=') {
                    Some((n, v)) => (n, Some(v.to_string())),
                    None => (name, None),
                };
                let value = if with_value.contains(&name) {
                    Some(inline.or_else(|| it.next().cloned()).ok_or(format!("--{name} needs a value"))?)
                } else {
                    None
                };
                flags.push((name.to_string(), value));
            } else {
                rest.push(a.clone());
            }
        }
        Ok(Args { flags, rest })
    }

    fn has(&self, name: &str) -> bool {
        self.flags.iter().any(|(n, _)| n == name)
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.flags.iter().rev().find(|(n, _)| n == name).and_then(|(_, v)| v.as_deref())
    }

    fn lang(&self, name: &str, default: &str) -> Result<String, String> {
        let v = self.get(name).unwrap_or(default);
        langs::code_for(v).map(String::from).ok_or(format!("--{name}: can't speak '{v}'; use one of {}", codes()))
    }

    fn number(&self, name: &str, default: u32) -> Result<u32, String> {
        self.get(name).map_or(Ok(default), |v| v.parse().map_err(|_| format!("--{name} wants a number")))
    }
}

fn codes() -> String {
    langs::LANGS.iter().map(|l| l.code).collect::<Vec<_>>().join(" ")
}

fn progress(what: &str, done: u64, total: u64) {
    print!("\rdownloading the {what}: {}% ({} / {} MB)   ", done * 100 / total.max(1), done >> 20, total >> 20);
    let _ = std::io::stdout().flush();
    if done >= total {
        println!();
    }
}

fn download() -> Result<(), String> {
    let missing = models::missing();
    if missing.is_empty() {
        println!("all models are in {}", paths::models().display());
    }
    for m in missing {
        models::download(m, |d, t| progress(m.what, d, t))?;
    }
    Ok(())
}

fn record(args: &[String]) -> Result<(), String> {
    let a = Args::parse(args, &["voice", "mic"])?;
    let name = a.get("voice").unwrap_or("me");
    if !voices::valid_name(name) {
        return Err("use letters, digits, - or _ for the voice name".into());
    }
    println!("Recording voice '{name}'. Read this out loud at a normal pace and volume,");
    println!("or just talk naturally for {} seconds in any language:\n", voices::RECORD_SECS);
    println!("  {}\n", voices::READ_ALOUD);
    print!("Press Enter to start recording (it stops by itself)...");
    let _ = std::io::stdout().flush();
    std::io::stdin().lock().read_line(&mut String::new()).map_err(|e| e.to_string())?;
    println!("recording...");
    let wav = audio::record(a.get("mic"), voices::RECORD_SECS, tts::SR, &AtomicBool::new(false))?;
    println!("transcribing your sample...");
    download()?;
    let engines = Engines::load()?;
    let mut asr = Asr::load(&engines, false)?;
    let text = voices::save(name, wav, &mut asr)?;
    println!("saved voice '{name}': {text}");
    Ok(())
}

fn say(args: &[String]) -> Result<(), String> {
    let a = Args::parse(args, &["to", "voice", "style", "me", "headphones"])?;
    let saved = Settings::load();
    let to = a.lang("to", &saved.to)?;
    let me = langs::code_for(&a.lang("me", &saved.me)?).unwrap_or("en");
    let text = a.rest.join(" ");
    if text.trim().is_empty() {
        return Err("say what? mentalese say --to ru \"hello\"".into());
    }
    download()?;
    let voice = voices::load(a.get("voice").unwrap_or(&saved.voice))?;
    let engines = Engines::load()?;
    let mut mt = Translator::load(&engines, a.get("style").unwrap_or(&saved.style))?;
    let mut tts = Tts::load(&engines)?;
    tts.set_voice(&voice.wav, &voice.text, voice.lang)?;
    tts.keep_accent = a.has("keep-accent") || saved.keep_accent;
    let src = langid::detect(&text, me);
    let out = if src == to { text.clone() } else { mt.translate(&text, Some(src), &to)? };
    println!("[{to}] {out}");
    let headphones = a.get("headphones").map(String::from).or(saved.headphones).map_or(Output::Default, Output::Device);
    let player = Player::open(&headphones)?;
    tts.speak(&out, &to, |c| {
        player.play(c);
        true
    })?;
    while player.pending() > 0.0 {
        std::thread::sleep(Duration::from_millis(50));
    }
    std::thread::sleep(Duration::from_millis(150)); // the device's own buffer
    Ok(())
}

fn run(args: &[String]) -> Result<(), String> {
    let a = Args::parse(
        args,
        &["to", "me", "voice", "style", "silence-ms", "their-silence-ms", "subtitles-from", "mic", "input", "headphones"],
    )?;
    let mut s = Settings::load();
    s.to = a.lang("to", &s.to)?;
    s.me = a.lang("me", &s.me)?;
    if let Some(v) = a.get("voice") {
        s.voice = v.into();
    }
    if let Some(v) = a.get("style") {
        s.style = v.into();
    }
    s.silence_ms = a.number("silence-ms", s.silence_ms)?;
    s.their_silence_ms = a.number("their-silence-ms", s.their_silence_ms)?;
    s.keep_accent |= a.has("keep-accent");
    s.asr_gpu |= a.has("asr-gpu");
    s.monitor |= a.has("monitor");
    s.practice = a.has("practice");
    if a.has("no-subtitles") {
        s.subtitles_plugin = false;
    }
    if let Some(app) = a.get("subtitles-from") {
        s.subtitles_from = Some(app.into());
    }
    // --input: read another PipeWire source (testing)
    if let Some(m) = a.get("mic").or(a.get("input")) {
        s.mic = Some(m.into());
    }
    if let Some(h) = a.get("headphones") {
        s.headphones = Some(h.into());
    }
    if !voices::exists(&s.voice) {
        let hint = if s.voice == "me" { String::new() } else { format!(" --voice {}", s.voice) };
        return Err(format!("No voice called '{}' yet. Record it with:  mentalese record{hint}", s.voice));
    }

    let practice = s.practice;
    let (voice, me) = (s.voice.clone(), s.me.clone());
    let to = Arc::new(Mutex::new(s.to.clone()));
    let ready = Arc::new(AtomicBool::new(false));
    let (ev_to, ev_ready) = (to.clone(), ready.clone());
    let commands = format!("/{}  /accent  /mute  /quit", codes().replace(' ', " /"));
    let cmds = commands.clone();
    println!("loading...");
    let session = Session::start(s.clone(), move |e| match e {
        Event::Status { text } => println!("{DIM}{text}...{RESET}"),
        Event::Download { what, done, total } => progress(&what, done, total),
        Event::Ready { virtual_mic, backend, gpu } => {
            ev_ready.store(true, std::sync::atomic::Ordering::Relaxed);
            let name = |c: &str| langs::lang(c).map_or("?", |l| l.name);
            println!("{DIM}engines: {backend}{}{RESET}", gpu.map(|g| format!(" on {g}")).unwrap_or_default());
            if practice {
                println!("\nReady (practice). Talk, and hear yourself in your headphones as '{voice}'.");
            } else {
                match virtual_mic {
                    Some(VirtualMic::Ready { name }) => println!(
                        "\nReady. In Discord pick input device '{name}' and turn OFF noise suppression."
                    ),
                    Some(VirtualMic::Missing { how, url }) => {
                        println!("\n{RED}No virtual mic: {how} {url}{RESET}\nUntil then, use --practice.")
                    }
                    None => {}
                }
            }
            let others = if practice { String::new() } else { format!(" Others → subtitles in {}.", name(&me)) };
            print!(
                "You speak → {}.{others}\nType anything + Enter to say it in your voice. Commands: {cmds}\n> ",
                name(&ev_to.lock().unwrap())
            );
            let _ = std::io::stdout().flush();
        }
        Event::Said { heard, said, typed, timing: t, .. } => {
            let wait = if t.wait_ms > 50 { format!("wait {} + ", t.wait_ms) } else { String::new() };
            let asr = t.asr_ms.map_or("typed".into(), |ms| format!("asr {ms}"));
            let mode = if t.native { "native" } else { "like sample" };
            let who = if typed { "typed" } else { "you" };
            print!(
                "\r{DIM}{who}: {heard}\n  → {said}   [{} ms: {wait}{asr} + mt {} + voice {} ({mode})]{RESET}\n> ",
                t.total_ms, t.mt_ms, t.voice_ms
            );
            let _ = std::io::stdout().flush();
        }
        Event::Subtitle { stream, text, original, .. } => {
            print!("\r{}▌ {text}{RESET}  {DIM}({original}){RESET}\n> ", COLORS[stream as usize % COLORS.len()]);
            let _ = std::io::stdout().flush();
        }
        Event::Notice { text } => println!("\r{DIM}{text}{RESET}"),
        Event::Error { text } => println!("\r{RED}{text}{RESET}"),
        Event::Stopped => {}
    });

    let (mut muted, mut keep_accent) = (false, s.keep_accent);
    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        let line = line.trim();
        if !session.is_running() {
            break;
        }
        let cmd = line.to_lowercase();
        if line.is_empty() {
        } else if !cmd.starts_with('/') {
            session.say(line);
            continue; // the result line prints the next prompt
        } else if let Some(c) = langs::code_for(&cmd[1..]).filter(|_| cmd.len() == 3) {
            session.set_target(c);
            *to.lock().unwrap() = c.into();
            println!("now translating into {}", langs::lang(c).unwrap().name);
        } else if cmd == "/accent" {
            keep_accent = !keep_accent;
            session.set_keep_accent(keep_accent);
            println!("{}", if keep_accent { "voice: like your sample, accent and all" } else { "voice: native pronunciation (a bit less like you)" });
        } else if cmd == "/m" || cmd == "/mute" {
            muted = !muted;
            session.set_muted(muted);
            println!("{}", if muted { "mic muted (typing still works)" } else { "mic unmuted" });
        } else if ["/q", "/quit", "/exit"].contains(&cmd.as_str()) {
            break;
        } else {
            println!("commands: {commands}  — anything else gets spoken");
        }
        if ready.load(std::sync::atomic::Ordering::Relaxed) {
            print!("> ");
            let _ = std::io::stdout().flush();
        }
    }
    session.stop();
    Ok(())
}
