//! Cuts a stream of frames into utterances with the voice detector.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::audio::{FRAME, MIC_SR};
use crate::vad::{Event, Model, Vad};

/// ~320 ms kept from before speech starts, so the first syllable isn't clipped.
const PREROLL: usize = 10;
/// Longest utterance: 15 s, then it's cut and the detector starts over.
const MAX_SAMPLES: usize = MIC_SR as usize * 15;

pub struct Segmenter {
    vad: Vad,
    preroll: VecDeque<[f32; FRAME]>,
    buf: Option<Vec<f32>>,
}

impl Segmenter {
    /// `silence_ms`: how long a pause ends an utterance.
    pub fn new(model: Arc<Model>, silence_ms: u32) -> Segmenter {
        Segmenter { vad: Vad::new(model, 0.5, silence_ms), preroll: VecDeque::with_capacity(PREROLL), buf: None }
    }

    /// Returns a whole utterance when one ends.
    pub fn feed(&mut self, frame: &[f32; FRAME]) -> Option<Vec<f32>> {
        let ev = self.vad.feed(frame);
        let Some(buf) = &mut self.buf else {
            if self.preroll.len() == PREROLL {
                self.preroll.pop_front();
            }
            self.preroll.push_back(*frame);
            if matches!(ev, Some(Event::Start)) {
                self.buf = Some(self.preroll.drain(..).flatten().collect());
            }
            return None;
        };
        buf.extend_from_slice(frame);
        let ended = matches!(ev, Some(Event::End));
        if ended || buf.len() > MAX_SAMPLES {
            if !ended {
                self.vad.reset();
            }
            self.preroll.clear();
            return self.buf.take();
        }
        None
    }
}
