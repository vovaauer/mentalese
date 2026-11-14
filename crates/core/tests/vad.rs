//! Compares the Rust Silero VAD with onnxruntime. The reference files (16 kHz f32 audio,
//! onnxruntime's per-frame probabilities and start/end events) come from tools/vad_reference.py,
//! in build/vad-ref; the tests skip when they or the model are missing. Override the reference
//! directory with MENTALESE_VAD_REF.

use mentalese_core::vad::{Event, Model, Vad, WINDOW};
use std::{path::PathBuf, sync::Arc, thread, time::Instant};

const CLIPS: [&str; 5] = ["en", "ru", "zh", "silence", "noise"];

fn model() -> Option<Arc<Model>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/silero_vad.onnx");
    if !path.exists() {
        eprintln!("skipped: {} missing", path.display());
        return None;
    }
    Some(Arc::new(Model::load(&path).unwrap()))
}

fn refs() -> Option<PathBuf> {
    let dir = std::env::var_os("MENTALESE_VAD_REF")
        .map_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../build/vad-ref"), PathBuf::from);
    if !dir.join("en.probs.f32").exists() {
        eprintln!("skipped: no reference data in {}", dir.display());
        return None;
    }
    Some(dir)
}

fn f32s(path: PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes.as_chunks().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

#[test]
fn model_is_shareable() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Model>();
    send_sync::<Vad>();
}

#[test]
fn probabilities_match_onnxruntime() {
    let (Some(model), Some(dir)) = (model(), refs()) else { return };
    // One shared model, a stream per thread, as in a call.
    let handles = CLIPS.map(|name| {
        let (model, dir) = (model.clone(), dir.clone());
        thread::spawn(move || {
            let audio = f32s(dir.join(format!("{name}.f32")));
            let want = f32s(dir.join(format!("{name}.probs.f32")));
            let mut vad = Vad::new(model, 0.5, 100);
            let mut max = 0f32;
            for pass in 0..2 {
                for (i, frame) in audio.as_chunks::<WINDOW>().0.iter().enumerate() {
                    let diff = (vad.prob(frame) - want[i]).abs();
                    assert!(diff < 1e-4, "{name} pass {pass} frame {i}: off by {diff}");
                    max = max.max(diff);
                }
                vad.reset();
            }
            (name, want.len(), max)
        })
    });
    for h in handles {
        let (name, frames, max) = h.join().unwrap();
        eprintln!("{name:>8}: {frames:>3} frames, max |rust - onnxruntime| = {max:.2e}");
    }
}

#[test]
fn events_match_python() {
    let (Some(model), Some(dir)) = (model(), refs()) else { return };
    for name in CLIPS {
        let audio = f32s(dir.join(format!("{name}.f32")));
        let want = std::fs::read_to_string(dir.join(format!("{name}.events"))).unwrap();
        let mut got = String::new();
        for ms in [400, 100] {
            let mut vad = Vad::new(model.clone(), 0.5, ms);
            for (i, frame) in audio.as_chunks::<WINDOW>().0.iter().enumerate() {
                match vad.feed(frame) {
                    Some(Event::Start) => got += &format!("{ms} {i} start\n"),
                    Some(Event::End) => got += &format!("{ms} {i} end\n"),
                    None => {}
                }
            }
        }
        assert_eq!(got, want, "{name}");
        eprintln!("{name:>8}: {} events match", got.lines().count());
    }
}

#[test]
fn speed() {
    let (Some(model), Some(dir)) = (model(), refs()) else { return };
    let audio = f32s(dir.join("en.f32"));
    let mut vad = Vad::new(model, 0.5, 100);
    let frames = audio.len() / WINDOW;
    let mut best = f64::MAX;
    for _ in 0..50 {
        vad.reset();
        let t = Instant::now();
        for frame in audio.as_chunks::<WINDOW>().0 {
            std::hint::black_box(vad.prob(std::hint::black_box(frame)));
        }
        best = best.min(t.elapsed().as_secs_f64() / frames as f64);
    }
    eprintln!("rust: {:.2} us/frame (best of 50 passes over {frames} frames)", best * 1e6);
}
