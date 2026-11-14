//! Speech to text with Qwen3-ASR-0.6B. Detects the spoken language on its own.

use crate::engines::{Engines, Llm, LlmOptions};
use crate::{langs, models};

pub struct Asr {
    llm: Llm,
    prompt: String,
}

impl Asr {
    /// gpu_encoder=false runs the audio encoder on the CPU: 0.75 GB less VRAM, ~40 ms slower on a
    /// typical 4-8 s sentence.
    pub fn load(engines: &Engines, gpu_encoder: bool) -> Result<Asr, String> {
        // 15 s of speech is ~220 prompt tokens
        let opts = LlmOptions { mmproj_gpu: gpu_encoder, n_ctx: 1024, n_batch: 256, n_ubatch: 256 };
        let llm = engines.load_llm(&models::ASR.path(), Some(&models::ASR_MMPROJ.path()), opts)?;
        let prompt = format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", llm.marker);
        Ok(Asr { llm, prompt })
    }

    /// (language code if it's one we speak, text). Audio: 16 kHz mono.
    pub fn transcribe(&mut self, audio: &[f32]) -> Result<(Option<&'static str>, String), String> {
        // cap output by audio length (fast Russian is ~12 tokens/s): on tiny clips the model can
        // loop on one token until the cap
        let max_tokens = audio.len() * 20 / 16_000 + 16;
        let out = self.llm.generate(&self.prompt, Some(audio), max_tokens)?;
        Ok(parse(&out))
    }
}

/// "language English<asr_text>hello there" -> (Some("en"), "hello there")
fn parse(out: &str) -> (Option<&'static str>, String) {
    let Some(rest) = out.trim().strip_prefix("language") else { return (None, String::new()) };
    let Some((lang, text)) = rest.split_once("<asr_text>") else { return (None, String::new()) };
    let lang = lang.trim();
    let code = if lang == "None" { None } else { langs::code_for(lang) };
    (code, text.trim().to_string())
}

#[cfg(test)]
mod tests {
    #[test]
    fn parse() {
        assert_eq!(super::parse("language English<asr_text>Hi there."), (Some("en"), "Hi there.".into()));
        assert_eq!(super::parse("language None<asr_text>"), (None, "".into()));
        assert_eq!(super::parse("language Arabic<asr_text> مرحبا "), (None, "مرحبا".into()));
        assert_eq!(super::parse("garbage"), (None, "".into()));
    }
}
