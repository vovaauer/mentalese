//! Translation with Bilibili's Index-Translate-2B.
//!
//! Chosen over Hy-MT2-7B (the previous translator): same tone on casual chat (0/26 formal "вы"
//! for both), 2.4x faster and 1.5 GB of VRAM instead of 4.8. It misses the odd idiom ("no way
//! you pulled that off" came out as its opposite); a larger quant didn't fix that.

use crate::engines::{Engines, Llm, LlmOptions};
use crate::{langs, models};

/// Without a style the model goes formal in Russian ("вы", 3/26 lines); with it, 0/26. "add
/// nothing" keeps short lines from being embellished ("bye" -> "ну пока, чуваки! 😎").
pub const DEFAULT_STYLE: &str =
    "casual and informal, like talking to a close friend; translate only what is said, add nothing";

pub struct Translator {
    llm: Llm,
    /// Empty: plain translation.
    pub style: String,
}

impl Translator {
    pub fn load(engines: &Engines, style: &str) -> Result<Translator, String> {
        // a spoken sentence is ~50 tokens: a small context and batches keep VRAM down
        let opts = LlmOptions { mmproj_gpu: false, n_ctx: 1024, n_batch: 512, n_ubatch: 256 };
        let llm = engines.load_llm(&models::MT.path(), None, opts)?;
        Ok(Translator { llm, style: style.into() })
    }

    pub fn translate(&mut self, text: &str, source: Option<&str>, target: &str) -> Result<String, String> {
        let src = source.and_then(langs::lang).map_or("", |l| l.zh);
        let tgt = langs::lang(target).ok_or_else(|| format!("unknown language {target}"))?.zh;
        // the model card's two Chinese templates: plain, and "instTrans" with the style as a
        // 【注意】 (soft) constraint. The source language is left out when unknown.
        if !self.style.is_empty() {
            let prompt = format!(
                "请将以下{src}对话翻译成{tgt}，并且严格遵循所有约束要求。\n\n【源文】\n{text}\n\n【约束要求】\n1. 【注意】{}\n\n只输出译文，不要有任何额外说明。",
                self.style
            );
            let out = self.complete(&prompt, text)?;
            if !self.leaked(&out, text) {
                return Ok(out);
            }
            // it recited the instruction instead of translating: redo it plain
        }
        self.complete(&format!("请将以下{src}文本翻译为{tgt}，直接输出翻译结果，不要进行任何解释。\n\n{text}"), text)
    }

    fn leaked(&self, out: &str, text: &str) -> bool {
        if "[]【】".chars().any(|b| out.contains(b) && !text.contains(b)) {
            return true;
        }
        let style = self.style.to_lowercase();
        let style_words: std::collections::HashSet<&str> =
            style.split_whitespace().filter(|w| w.chars().count() > 3).collect();
        let out = out.to_lowercase().replace(',', " ");
        let out_words: std::collections::HashSet<&str> = out.split_whitespace().collect();
        style_words.intersection(&out_words).count() as f32 >= style_words.len() as f32 / 2.0
    }

    fn complete(&mut self, content: &str, text: &str) -> Result<String, String> {
        // greedy, as the model card evaluates it; thinking off
        let prompt = format!("<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        let out = self.llm.generate(&prompt, None, 512)?;
        let out = out.rsplit("</think>").next().unwrap_or("").trim();
        // speech has no line breaks: if the model added a preamble, the last paragraph is the answer
        if out.contains('\n') && !text.contains('\n') {
            return Ok(out.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("").trim().to_string());
        }
        Ok(out.to_string())
    }
}
