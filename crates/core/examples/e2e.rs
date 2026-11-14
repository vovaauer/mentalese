//! End-to-end test on Linux without a real mic, Discord or speakers: two throwaway PipeWire
//! sinks stand in for the mic and the headphones, in practice mode, so nothing reaches the
//! virtual mic (apps recording it, like Discord, hear nothing). Plays test clips in, records
//! what comes out, prints each line with its timing.
//!   cargo run --release -p mentalese-core --example e2e [clips dir, default out/]

use std::path::PathBuf;
use std::process::Command;
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use mentalese_core::audio::{self, Output, Player};
use mentalese_core::pipeline::{Event, Session};
use mentalese_core::settings::Settings;
use mentalese_core::{paths, resample, wav};

const IN: &str = "mentalese_e2e_in";
const OUT: &str = "mentalese_e2e_out";

fn pactl(args: &[&str]) -> String {
    let o = Command::new("pactl").args(args).output().expect("pactl");
    String::from_utf8_lossy(&o.stdout).trim().to_string()
}

fn main() {
    let clips = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| paths::repo().join("out"));
    let modules: Vec<String> = [IN, OUT]
        .iter()
        .map(|s| pactl(&["load-module", "module-null-sink", &format!("sink_name={s}"), "rate=48000"]))
        .collect();
    let result = std::panic::catch_unwind(|| run(&clips));
    for m in modules {
        pactl(&["unload-module", &m]);
    }
    if result.is_err() {
        std::process::exit(1);
    }
}

fn run(clips: &std::path::Path) {
    let settings = Settings {
        practice: true,
        mic: Some(format!("{IN}.monitor")),
        headphones: Some(OUT.into()),
        ..Settings::load()
    };
    let (ready_tx, ready_rx) = channel();
    let lines = Arc::new(Mutex::new(Vec::new()));
    let l = lines.clone();
    let session = Session::start(settings, move |e| match e {
        Event::Ready { backend, gpu, .. } => {
            println!("ready: {backend} {}", gpu.unwrap_or_default());
            let _ = ready_tx.send(());
        }
        Event::Said { heard, said, typed, timing: t, .. } => {
            let how = if typed { "typed".into() } else { format!("asr {}", t.asr_ms.unwrap_or(0)) };
            let line = format!("{heard}\n  -> {said}   [{} ms: {how} + mt {} + voice {}]", t.total_ms, t.mt_ms, t.voice_ms);
            println!("{line}");
            l.lock().unwrap().push(line);
        }
        Event::Error { text } => println!("ERROR: {text}"),
        Event::Notice { text } => println!("notice: {text}"),
        _ => {}
    });
    ready_rx.recv_timeout(Duration::from_secs(600)).expect("never got ready");

    // record what comes out of the "headphones"
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let s2 = stop.clone();
    let recorder = std::thread::spawn(move || audio::record(Some(&format!("{OUT}.monitor")), 120.0, 24_000, &s2));

    let mic = Player::open(&Output::Device(IN.into())).expect("can't play into the test mic");
    for clip in ["bench_en", "bench_ru", "bench_zh", "bench_en"] {
        let (a, sr) = wav::read(&clips.join(format!("{clip}.wav"))).expect("missing clip (run on a checkout with out/)");
        mic.play(&resample::resample(&a, sr, 24_000));
        std::thread::sleep(Duration::from_secs_f32(a.len() as f32 / sr as f32 + 3.0));
    }
    session.say("Thanks, man, I owe you one.");
    std::thread::sleep(Duration::from_secs(4));
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let out = recorder.join().unwrap().unwrap_or_default();
    session.stop();

    // 0.1 s blocks with speech in them
    let speech = out.chunks(2400).filter(|b| b.iter().any(|s| s.abs() > 0.02)).count() as f32 / 10.0;
    println!("\n{} lines; the headphones got {speech:.1}s of speech", lines.lock().unwrap().len());
}
