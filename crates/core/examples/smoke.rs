//! Loads all three models in one process and runs the bench sentences through them.
//!   cargo run --release -p mentalese-core --example smoke
use std::time::Instant;

use mentalese_core::{asr::Asr, engines::Engines, mt::{Translator, DEFAULT_STYLE}, paths, resample, tts::Tts, wav};

fn main() {
    let t = Instant::now();
    let e = Engines::load().unwrap();
    println!("backend {} gpus {:?}", e.backend, e.gpus);
    let base = vram();
    let mut asr = Asr::load(&e, false).unwrap();
    let after_asr = vram();
    let mut mt = Translator::load(&e, DEFAULT_STYLE).unwrap();
    let after_mt = vram();
    let mut tts = Tts::load(&e).unwrap();
    let after_tts = vram();
    println!("loaded in {:.1}s; VRAM MiB: engines {base}, +asr {}, +translator {}, +voice {}",
             t.elapsed().as_secs_f32(), after_asr - base, after_mt - after_asr, after_tts - after_mt);
    let peak = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    let p2 = peak.clone();
    std::thread::spawn(move || loop {
        p2.fetch_max(vram(), std::sync::atomic::Ordering::Relaxed);
        std::thread::sleep(std::time::Duration::from_millis(50));
    });

    let out = paths::repo().join("out");
    for clip in ["bench_en", "bench_ru", "bench_zh"] {
        let (a, sr) = wav::read(&out.join(format!("{clip}.wav"))).unwrap();
        let a16 = resample::resample(&a, sr, 16_000);
        asr.transcribe(&a16).unwrap();
        let t = Instant::now();
        let r = asr.transcribe(&a16).unwrap();
        println!("asr {clip}: {:?} {} ms", r, t.elapsed().as_millis());
    }
    let sentence = "Hey, are you guys ready? Let's go mine some diamonds before the server restarts.";
    mt.translate("hello", Some("en"), "ru").unwrap();
    for lang in ["ru", "zh", "ja"] {
        let t = Instant::now();
        let r = mt.translate(sentence, Some("en"), lang).unwrap();
        println!("mt en->{lang}: {r} ({} ms)", t.elapsed().as_millis());
    }
    for (text, src) in [("Привет, ты где? Я тебя жду у спавна уже десять минут.", "ru"), ("我们今晚一起打游戏吧，我有新的地图。", "zh")] {
        println!("mt {src}->en: {}", mt.translate(text, Some(src), "en").unwrap());
    }

    let (ref_wav, ref_sr) = wav::read(&out.join("ref.wav")).unwrap();
    let ref16 = resample::resample(&ref_wav, ref_sr, 16_000);
    let (lang, text) = asr.transcribe(&ref16).unwrap();
    tts.set_voice(&resample::resample(&ref_wav, ref_sr, 24_000), &text, lang).unwrap();
    tts.warmup().unwrap();
    for (text, lang) in [("Эй, вы готовы? Пошли копать алмазы, пока сервер не перезапустился.", "ru"), (sentence, "en")] {
        let t = Instant::now();
        let mut first = None;
        let mut audio = Vec::new();
        tts.speak(text, lang, |c| {
            first.get_or_insert(t.elapsed());
            audio.extend_from_slice(c);
            true
        }).unwrap();
        println!("tts {lang}: first sound {} ms, {:.1}s of audio in {} ms", first.unwrap().as_millis(),
                 audio.len() as f32 / 24_000.0, t.elapsed().as_millis());
        wav::write(&out.join(format!("rust_{lang}.wav")), &audio, 24_000).unwrap();
        let back = asr.transcribe(&resample::resample(&audio, 24_000, 16_000)).unwrap();
        println!("  heard back: {:?}", back);
    }
    std::thread::sleep(std::time::Duration::from_millis(100));
    println!("VRAM peak while working: {} MiB", peak.load(std::sync::atomic::Ordering::Relaxed));
}

/// This process's GPU memory in MiB (NVIDIA only; 0 elsewhere).
fn vram() -> u32 {
    let pid = std::process::id().to_string();
    let Ok(o) = std::process::Command::new("nvidia-smi")
        .args(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
        .output() else { return 0 };
    String::from_utf8_lossy(&o.stdout).lines()
        .filter_map(|l| l.split_once(", ")).filter(|(p, _)| *p == pid)
        .filter_map(|(_, m)| m.trim().parse::<u32>().ok()).sum()
}
